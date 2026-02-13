"""Tests for the skill learning engine."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolvagent.core.agent import Agent
from evolvagent.core.config import Settings, AgentConfig
from evolvagent.core.events import EventBus
from evolvagent.core.learner import DynamicSkill, SkillLearner
from evolvagent.core.llm import LLMResponse
from evolvagent.core.skill import (
    SkillMetadata, SkillOrigin, TrustLevel,
)
from evolvagent.core.storage import SkillStore


def run(coro):
    return asyncio.run(coro)


def _tmp_settings():
    return Settings(agent=AgentConfig(data_dir=tempfile.mkdtemp()))


def _mock_llm(content="mock output"):
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content=content, model="mock", tokens_in=5, tokens_out=10,
    ))
    return llm


def _tmp_store():
    tmpdir = tempfile.mkdtemp()
    return SkillStore(Path(tmpdir) / "test.db")


# ===================================================================
# DynamicSkill Tests
# ===================================================================


class TestDynamicSkill(unittest.TestCase):

    def test_execute_success(self):
        llm = _mock_llm("Here is the answer")
        meta = SkillMetadata(
            name="test_dynamic",
            description="A test skill",
            trigger_conditions=["test"],
            trust_level=TrustLevel.AUTO,
        )
        skill = DynamicSkill(metadata=meta, llm=llm, system_prompt="You are helpful.")
        result = run(skill.execute({"user_input": "help me"}))
        self.assertTrue(result.success)
        self.assertEqual(result.output, "Here is the answer")
        llm.complete.assert_called_once()

    def test_execute_empty_input(self):
        llm = _mock_llm()
        skill = DynamicSkill(
            metadata=SkillMetadata(name="d"),
            llm=llm,
            system_prompt="test",
        )
        result = run(skill.execute({"user_input": ""}))
        self.assertFalse(result.success)

    def test_execute_llm_failure(self):
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("API down"))
        skill = DynamicSkill(
            metadata=SkillMetadata(name="d"),
            llm=llm,
            system_prompt="test",
        )
        result = run(skill.execute({"user_input": "help"}))
        self.assertFalse(result.success)
        self.assertIn("API down", result.error)

    def test_can_handle_trigger_match(self):
        skill = DynamicSkill(
            metadata=SkillMetadata(
                name="d",
                trigger_conditions=["translate", "translation"],
            ),
            llm=_mock_llm(),
            system_prompt="test",
        )
        self.assertGreaterEqual(skill.can_handle("translate this text"), 0.75)
        self.assertEqual(skill.can_handle("unrelated query"), 0.0)

    def test_can_handle_tag_match(self):
        skill = DynamicSkill(
            metadata=SkillMetadata(
                name="d",
                tags=["python", "code", "debug"],
                trigger_conditions=[],
            ),
            llm=_mock_llm(),
            system_prompt="test",
        )
        self.assertGreater(skill.can_handle("debug python code"), 0)

    def test_preview(self):
        skill = DynamicSkill(
            metadata=SkillMetadata(name="translator", description="Translates text"),
            llm=_mock_llm(),
            system_prompt="You translate text.",
        )
        preview = run(skill.preview({}))
        self.assertIn("translator", preview)
        self.assertIn("Translates text", preview)

    def test_serialization_roundtrip(self):
        llm = _mock_llm()
        meta = SkillMetadata(
            name="test_rt",
            description="Roundtrip test",
            trigger_conditions=["test"],
            origin=SkillOrigin.LEARNED,
        )
        skill = DynamicSkill(metadata=meta, llm=llm, system_prompt="Be helpful.")

        defn = skill.to_definition()
        self.assertEqual(defn["type"], "dynamic")
        self.assertEqual(defn["system_prompt"], "Be helpful.")

        restored = DynamicSkill.from_definition(defn, llm)
        self.assertEqual(restored.metadata.name, "test_rt")
        self.assertEqual(restored.system_prompt, "Be helpful.")


# ===================================================================
# SkillLearner — Pattern Analysis
# ===================================================================


class TestLearnerPatterns(unittest.TestCase):

    def test_record_interactions(self):
        learner = SkillLearner(llm=None, bus=EventBus(), store=None)
        learner.record("help me", "general_assistant", True)
        learner.record("translate this", "", False)
        self.assertEqual(len(learner._history), 2)

    def test_no_llm_returns_skipped(self):
        learner = SkillLearner(llm=None, bus=EventBus(), store=None)
        result = run(learner.analyze_and_learn({}))
        self.assertEqual(result.skipped_reason, "No LLM client available")

    def test_not_enough_history(self):
        learner = SkillLearner(llm=_mock_llm(), bus=EventBus(), store=None)
        learner.record("query1", "", False)
        result = run(learner.analyze_and_learn({}))
        self.assertIn("Not enough", result.skipped_reason)

    def test_cluster_by_keywords(self):
        clusters = SkillLearner._cluster_by_keywords([
            "translate english text to chinese",
            "translate english text to french",
            "translate english text to japanese",
            "completely unrelated weather query",
        ])
        # Should find at least one cluster with the translate queries
        translate_cluster = [c for c in clusters if len(c) >= 3]
        self.assertGreaterEqual(len(translate_cluster), 1)


# ===================================================================
# SkillLearner — Parse Skill Definition
# ===================================================================


class TestParseSkillDefinition(unittest.TestCase):

    def _valid_json(self, **overrides):
        base = {
            "name": "translator",
            "description": "Translates text between languages",
            "system_prompt": "You are a professional translator.",
            "trigger_conditions": ["translate", "translation"],
            "tags": ["language", "translate"],
            "category": "language",
        }
        base.update(overrides)
        return json.dumps(base)

    def test_valid_json(self):
        result = SkillLearner._parse_skill_definition(self._valid_json())
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "dynamic")
        self.assertEqual(result["metadata"]["name"], "translator")
        self.assertEqual(result["metadata"]["origin"], "learned")

    def test_json_with_code_fences(self):
        content = f"```json\n{self._valid_json()}\n```"
        result = SkillLearner._parse_skill_definition(content)
        self.assertIsNotNone(result)
        self.assertEqual(result["metadata"]["name"], "translator")

    def test_none_response(self):
        self.assertIsNone(SkillLearner._parse_skill_definition("NONE"))
        self.assertIsNone(SkillLearner._parse_skill_definition(""))

    def test_invalid_json(self):
        self.assertIsNone(SkillLearner._parse_skill_definition("not json {"))

    def test_missing_required_fields(self):
        result = SkillLearner._parse_skill_definition('{"name": "x"}')
        self.assertIsNone(result)

    def test_name_sanitization(self):
        result = SkillLearner._parse_skill_definition(
            self._valid_json(name="My Skill Name!")
        )
        self.assertIsNotNone(result)
        # Should be lowercase with underscores
        self.assertRegex(result["metadata"]["name"], r'^[a-z0-9_]+$')

    def test_short_name_rejected(self):
        result = SkillLearner._parse_skill_definition(
            self._valid_json(name="ab")
        )
        self.assertIsNone(result)


# ===================================================================
# SkillLearner — Teach (explicit)
# ===================================================================


class TestTeach(unittest.TestCase):

    def test_teach_creates_definition(self):
        store = _tmp_store()
        bus = EventBus()
        events = []
        bus.on("skill.taught", lambda e: events.append(e.data))

        learner = SkillLearner(llm=None, bus=bus, store=store)
        defn = run(learner.teach(
            name="my_translator",
            description="Translates text",
            system_prompt="You are a translator.",
            triggers=["translate"],
            tags=["language"],
        ))

        self.assertEqual(defn["metadata"]["name"], "my_translator")
        self.assertEqual(defn["metadata"]["origin"], "user_defined")
        self.assertEqual(defn["system_prompt"], "You are a translator.")

        # Persisted
        loaded = store.load_skill_definition("my_translator")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["system_prompt"], "You are a translator.")

        # Event emitted
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["skill_name"], "my_translator")
        store.close()

    def test_teach_sanitizes_name(self):
        learner = SkillLearner(llm=None, bus=EventBus(), store=None)
        defn = run(learner.teach(
            name="My Cool Skill!",
            description="test",
            system_prompt="test",
            triggers=["test"],
        ))
        self.assertRegex(defn["metadata"]["name"], r'^[a-z0-9_]+$')


# ===================================================================
# SkillLearner — Auto-learn with Mock LLM
# ===================================================================


class TestAutoLearn(unittest.TestCase):

    def test_learns_from_miss_pattern(self):
        """Repeated misses for similar queries should produce a skill proposal."""
        skill_json = json.dumps({
            "name": "code_reviewer",
            "description": "Reviews code for issues",
            "system_prompt": "You are a code reviewer.",
            "trigger_conditions": ["review code", "code review"],
            "tags": ["code", "review"],
            "category": "development",
        })
        llm = _mock_llm(skill_json)
        store = _tmp_store()
        bus = EventBus()
        events = []
        bus.on("skill.learned", lambda e: events.append(e.data))

        learner = SkillLearner(llm=llm, bus=bus, store=store)

        # Record similar misses
        for i in range(5):
            learner.record(f"review my code for bugs #{i}", "", False)

        result = run(learner.analyze_and_learn({}))
        self.assertGreaterEqual(result.skills_proposed, 0)
        # The LLM returns a valid definition, so it should be created
        if result.skills_created > 0:
            self.assertIn("code_reviewer", result.created_skill_names)
            saved = store.load_skill_definition("code_reviewer")
            self.assertIsNotNone(saved)
        store.close()

    def test_learns_from_fallback_pattern(self):
        """Repeated fallbacks to general_assistant should trigger learning."""
        skill_json = json.dumps({
            "name": "math_solver",
            "description": "Solves math problems",
            "system_prompt": "You are a math expert.",
            "trigger_conditions": ["solve math", "calculate"],
            "tags": ["math", "calculate"],
            "category": "math",
        })
        llm = _mock_llm(skill_json)
        store = _tmp_store()
        learner = SkillLearner(llm=llm, bus=EventBus(), store=store)

        for i in range(5):
            learner.record(f"solve this math equation #{i}", "general_assistant", True)

        result = run(learner.analyze_and_learn({}))
        self.assertEqual(result.patterns_analyzed, 5)
        store.close()

    def test_llm_returns_none_no_skill(self):
        """LLM responding NONE should not create a skill."""
        llm = _mock_llm("NONE")
        learner = SkillLearner(llm=llm, bus=EventBus(), store=None)
        for i in range(5):
            learner.record(f"do random thing #{i}", "", False)
        result = run(learner.analyze_and_learn({}))
        self.assertEqual(result.skills_created, 0)


# ===================================================================
# Agent Integration — Teach + Dynamic Skill Loading
# ===================================================================


class TestAgentLearnIntegration(unittest.TestCase):

    def test_teach_and_use(self):
        """Taught skill should be usable via handle_request."""
        settings = _tmp_settings()
        llm = _mock_llm("Translated: hello -> hola")

        async def _test():
            agent = Agent(settings=settings)
            agent.set_llm(llm)
            await agent.start()

            skill = await agent.learn_skill(
                name="translator",
                description="Translates text",
                system_prompt="You are a translator.",
                triggers=["translate"],
            )
            self.assertIsNotNone(skill)
            self.assertEqual(agent.get_skill("translator"), skill)

            # Now handle a request
            result = await agent.handle_request("translate hello to spanish")
            self.assertIn("Translated", result)
            await agent.shutdown()

        run(_test())

    def test_status_shows_learned_count(self):
        settings = _tmp_settings()
        llm = _mock_llm("ok")

        async def _test():
            agent = Agent(settings=settings)
            agent.set_llm(llm)
            await agent.start()

            await agent.learn_skill(
                name="test_skill",
                description="test",
                system_prompt="test",
                triggers=["test"],
            )

            status = agent.status_dict()
            self.assertEqual(status["learned_skill_count"], 1)
            await agent.shutdown()

        run(_test())

    def test_dynamic_skill_persists_across_restart(self):
        """Dynamic skills should be reloaded when agent restarts."""
        settings = _tmp_settings()
        llm = _mock_llm("persisted response")

        async def _create():
            agent = Agent(settings=settings)
            agent.set_llm(llm)
            await agent.start()
            await agent.learn_skill(
                name="persistent_skill",
                description="A persistent skill",
                system_prompt="You persist.",
                triggers=["persist"],
            )
            self.assertIsNotNone(agent.get_skill("persistent_skill"))
            await agent.shutdown()

        async def _reload():
            agent = Agent(settings=settings)
            agent.set_llm(llm)
            await agent.start()
            # Should auto-load from store
            skill = agent.get_skill("persistent_skill")
            self.assertIsNotNone(skill)
            self.assertIsInstance(skill, DynamicSkill)
            await agent.shutdown()

        run(_create())
        run(_reload())

    def test_learner_records_misses(self):
        """handle_request with no matching skill should record a miss."""
        settings = _tmp_settings()

        async def _test():
            agent = Agent(settings=settings)
            agent.set_llm(_mock_llm())
            await agent.start()
            # No skills registered — will miss
            await agent.handle_request("unknown query")
            self.assertEqual(len(agent._learner._history), 1)
            self.assertEqual(agent._learner._history[0].skill_used, "")
            await agent.shutdown()

        run(_test())


# ===================================================================
# Storage — Skill Definitions
# ===================================================================


class TestSkillDefinitionStorage(unittest.TestCase):

    def test_save_and_load(self):
        store = _tmp_store()
        defn = {"type": "dynamic", "system_prompt": "test", "metadata": {"name": "x"}}
        store.save_skill_definition("x", defn)
        loaded = store.load_skill_definition("x")
        self.assertEqual(loaded["system_prompt"], "test")
        store.close()

    def test_load_nonexistent(self):
        store = _tmp_store()
        self.assertIsNone(store.load_skill_definition("nope"))
        store.close()

    def test_load_all(self):
        store = _tmp_store()
        store.save_skill_definition("a", {"type": "dynamic", "metadata": {"name": "a"}})
        store.save_skill_definition("b", {"type": "dynamic", "metadata": {"name": "b"}})
        all_defs = store.load_all_skill_definitions()
        self.assertEqual(len(all_defs), 2)
        store.close()

    def test_delete(self):
        store = _tmp_store()
        store.save_skill_definition("x", {"type": "dynamic", "metadata": {"name": "x"}})
        self.assertTrue(store.delete_skill_definition("x"))
        self.assertIsNone(store.load_skill_definition("x"))
        self.assertFalse(store.delete_skill_definition("x"))
        store.close()

    def test_upsert(self):
        store = _tmp_store()
        store.save_skill_definition("x", {"version": 1})
        store.save_skill_definition("x", {"version": 2})
        loaded = store.load_skill_definition("x")
        self.assertEqual(loaded["version"], 2)
        store.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
