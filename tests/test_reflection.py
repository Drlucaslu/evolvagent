"""Tests for the reflection engine."""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolvagent.core.events import EventBus
from evolvagent.core.llm import LLMResponse
from evolvagent.core.reflection import ReflectionEngine
from evolvagent.core.skill import (
    BaseSkill, FailureCategory, FailureLesson, SkillMetadata,
    SkillOrigin, SkillResult, SkillStatus,
)
from evolvagent.core.storage import SkillStore


def run(coro):
    return asyncio.run(coro)


def _mock_llm(content="- Principle one\n- Principle two"):
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content=content, model="mock", tokens_in=5, tokens_out=10,
    ))
    return llm


def _make_skill(name="test_skill", **kw) -> BaseSkill:
    class DummySkill(BaseSkill):
        async def execute(self, context):
            return SkillResult(success=True, output="ok")

    defaults = {
        "name": name,
        "description": "A test skill",
        "status": SkillStatus.ACTIVE,
        "origin": SkillOrigin.LEARNED,
    }
    defaults.update(kw)
    return DummySkill(SkillMetadata(**defaults))


def _tmp_store():
    tmpdir = tempfile.mkdtemp()
    return SkillStore(Path(tmpdir) / "test.db")


# ===================================================================
# Eligibility Tests
# ===================================================================


class TestReflectionEligibility(unittest.TestCase):

    def _engine(self):
        return ReflectionEngine(llm=_mock_llm(), bus=EventBus(), store=None)

    def test_eligible_with_new_failure_lessons(self):
        """Skill with valuable failure after last_reflected_at is eligible."""
        skill = _make_skill(last_reflected_at=time.time() - 100)
        skill.metadata.failure_lessons = [
            FailureLesson(
                category=FailureCategory.NEAR_SUCCESS,
                description="Almost worked",
                root_cause="off by one",
                task_context="test",
                timestamp=time.time(),  # After last_reflected_at
            )
        ]
        eligible = self._engine()._find_eligible_skills({"s": skill})
        self.assertEqual(len(eligible), 1)

    def test_eligible_enough_executions_never_reflected(self):
        """Skill with >= 5 executions and never reflected is eligible."""
        skill = _make_skill(total_executions=5, last_reflected_at=0)
        eligible = self._engine()._find_eligible_skills({"s": skill})
        self.assertEqual(len(eligible), 1)

    def test_not_eligible_too_few_executions(self):
        """Skill with < 5 executions and no failures is not eligible."""
        skill = _make_skill(total_executions=3, last_reflected_at=0)
        eligible = self._engine()._find_eligible_skills({"s": skill})
        self.assertEqual(len(eligible), 0)

    def test_not_eligible_just_reflected(self):
        """Skill that was reflected recently is not eligible."""
        skill = _make_skill(
            total_executions=10,
            last_reflected_at=time.time() - 60,  # 1 min ago
            last_used_at=time.time(),
        )
        eligible = self._engine()._find_eligible_skills({"s": skill})
        self.assertEqual(len(eligible), 0)

    def test_not_eligible_archived(self):
        """ARCHIVED skill is not eligible."""
        skill = _make_skill(
            total_executions=10,
            last_reflected_at=0,
            status=SkillStatus.ARCHIVED,
        )
        eligible = self._engine()._find_eligible_skills({"s": skill})
        self.assertEqual(len(eligible), 0)

    def test_eligible_reflected_long_ago_with_new_use(self):
        """Skill reflected > 7 days ago with new usage is eligible."""
        skill = _make_skill(
            total_executions=10,
            last_reflected_at=time.time() - 8 * 86400,  # 8 days ago
            last_used_at=time.time() - 3600,  # used 1 hour ago (after last reflection)
        )
        eligible = self._engine()._find_eligible_skills({"s": skill})
        self.assertEqual(len(eligible), 1)


# ===================================================================
# No LLM Tests
# ===================================================================


class TestReflectionNoLLM(unittest.TestCase):

    def test_no_llm_returns_skipped(self):
        engine = ReflectionEngine(llm=None, bus=EventBus(), store=None)
        skill = _make_skill(total_executions=10, last_reflected_at=0)
        result = run(engine.reflect({"s": skill}))
        self.assertEqual(result.skipped_reason, "No LLM client available")
        self.assertEqual(result.skills_analyzed, 0)


# ===================================================================
# Mock LLM Tests
# ===================================================================


class TestReflectionWithMockLLM(unittest.TestCase):

    def test_extract_principles(self):
        """Normal reflection extracts principles and updates metadata."""
        llm = _mock_llm("- Always validate inputs\n- Handle edge cases early")
        store = _tmp_store()
        bus = EventBus()
        engine = ReflectionEngine(llm=llm, bus=bus, store=store)

        skill = _make_skill(total_executions=10, last_reflected_at=0)
        store.save(skill.metadata)

        result = run(engine.reflect({"s": skill}))
        self.assertEqual(result.skills_analyzed, 1)
        self.assertEqual(result.skills_updated, 1)
        self.assertEqual(result.principles_extracted, 2)
        self.assertIn("Always validate inputs", skill.metadata.distilled_principles)
        self.assertGreater(skill.metadata.last_reflected_at, 0)
        store.close()

    def test_llm_returns_none_no_principles(self):
        """LLM returning NONE means 0 principles extracted."""
        llm = _mock_llm("NONE")
        engine = ReflectionEngine(llm=llm, bus=EventBus(), store=None)

        skill = _make_skill(total_executions=10, last_reflected_at=0)
        result = run(engine.reflect({"s": skill}))
        self.assertEqual(result.skills_analyzed, 1)
        self.assertEqual(result.skills_updated, 0)
        self.assertEqual(result.principles_extracted, 0)

    def test_deduplication(self):
        """Existing principles are not added again."""
        llm = _mock_llm("- Already known\n- Brand new insight")
        engine = ReflectionEngine(llm=llm, bus=EventBus(), store=None)

        skill = _make_skill(total_executions=10, last_reflected_at=0)
        skill.metadata.distilled_principles = ["Already known"]

        result = run(engine.reflect({"s": skill}))
        self.assertEqual(result.principles_extracted, 1)
        # Should have 2 total: the existing one + the new one
        self.assertEqual(len(skill.metadata.distilled_principles), 2)

    def test_single_skill_failure_doesnt_block_others(self):
        """If one skill's LLM call fails, other skills still get reflected."""
        call_count = [0]

        async def side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("LLM exploded")
            return LLMResponse(content="- New principle", model="mock")

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=side_effect)
        engine = ReflectionEngine(llm=llm, bus=EventBus(), store=None)

        skill1 = _make_skill(name="fail_skill", total_executions=10, last_reflected_at=0)
        skill2 = _make_skill(name="ok_skill", total_executions=10, last_reflected_at=0)

        result = run(engine.reflect({"s1": skill1, "s2": skill2}))
        self.assertEqual(result.skills_analyzed, 2)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("fail_skill", result.errors[0])
        # skill2 should still have been updated
        self.assertEqual(result.skills_updated, 1)

    def test_emits_skill_reflected_event(self):
        """Reflection emits a skill.reflected event."""
        llm = _mock_llm("- Useful principle")
        bus = EventBus()
        events = []
        bus.on("skill.reflected", lambda e: events.append(e.data))

        engine = ReflectionEngine(llm=llm, bus=bus, store=None)
        skill = _make_skill(total_executions=10, last_reflected_at=0)

        run(engine.reflect({"s": skill}))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["skill_name"], "test_skill")
        self.assertEqual(events[0]["new_principles"], ["Useful principle"])

    def test_updates_last_reflected_at(self):
        """last_reflected_at should be updated after reflection."""
        llm = _mock_llm("- A principle")
        engine = ReflectionEngine(llm=llm, bus=EventBus(), store=None)

        skill = _make_skill(total_executions=10, last_reflected_at=0)
        before = time.time()
        run(engine.reflect({"s": skill}))
        self.assertGreaterEqual(skill.metadata.last_reflected_at, before)

    def test_persists_to_store(self):
        """Reflected skills should be persisted via SkillStore.save()."""
        llm = _mock_llm("- Persisted principle")
        store = _tmp_store()
        engine = ReflectionEngine(llm=llm, bus=EventBus(), store=store)

        skill = _make_skill(total_executions=10, last_reflected_at=0)
        store.save(skill.metadata)

        run(engine.reflect({"s": skill}))

        loaded = store.load("test_skill")
        self.assertIsNotNone(loaded)
        self.assertIn("Persisted principle", loaded.distilled_principles)
        self.assertGreater(loaded.last_reflected_at, 0)
        store.close()


# ===================================================================
# Parse Principles Tests
# ===================================================================


class TestParsePrinciples(unittest.TestCase):

    def test_dash_prefix(self):
        result = ReflectionEngine._parse_principles(
            "- Always validate inputs\n- Handle errors gracefully"
        )
        self.assertEqual(result, ["Always validate inputs", "Handle errors gracefully"])

    def test_star_prefix(self):
        result = ReflectionEngine._parse_principles(
            "* Always validate inputs\n* Handle errors gracefully"
        )
        self.assertEqual(result, ["Always validate inputs", "Handle errors gracefully"])

    def test_numbered_prefix(self):
        result = ReflectionEngine._parse_principles(
            "1. Always validate inputs\n2. Handle errors gracefully\n3. Log important events"
        )
        self.assertEqual(result, [
            "Always validate inputs", "Handle errors gracefully", "Log important events",
        ])

    def test_none_output(self):
        self.assertEqual(ReflectionEngine._parse_principles("NONE"), [])
        self.assertEqual(ReflectionEngine._parse_principles("none"), [])

    def test_empty(self):
        self.assertEqual(ReflectionEngine._parse_principles(""), [])
        self.assertEqual(ReflectionEngine._parse_principles("   "), [])

    def test_short_fragments_filtered(self):
        result = ReflectionEngine._parse_principles("- OK\n- ab\n- A real principle")
        # "OK" (2 chars) and "ab" (2 chars) should be filtered
        self.assertEqual(result, ["A real principle"])

    def test_mixed_formats(self):
        text = "- Bullet point one\n* Star point two\n3. Number point three"
        result = ReflectionEngine._parse_principles(text)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
