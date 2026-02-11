"""
EvolvAgent Phase 0 test suite — pure stdlib unittest.

Run: python -m unittest tests/test_all.py -v
"""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

# Ensure package is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolvagent.core.events import EventBus, Event
from evolvagent.core.skill import (
    BaseSkill, FailureCategory, FailureLesson, SkillMetadata,
    SkillOrigin, SkillResult, SkillStatus, TrustLevel,
)
from evolvagent.core.agent import Agent, AgentState, InvalidTransition
from evolvagent.core.config import Settings, AgentConfig, load_settings, reset_settings


def run(coro):
    """Helper to run async in sync test."""
    return asyncio.run(coro)


def _tmp_settings():
    """Create Settings with a temp data dir to avoid sharing state."""
    tmpdir = tempfile.mkdtemp()
    return Settings(agent=AgentConfig(data_dir=tmpdir))


# ===================================================================
# Event Bus Tests
# ===================================================================

class TestEventBus(unittest.TestCase):

    def setUp(self):
        self.bus = EventBus()

    def test_emit_and_receive(self):
        received = []
        self.bus.on("test.topic", lambda e: received.append(e))
        self.bus.emit("test.topic", {"key": "value"})
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["key"], "value")

    def test_multiple_handlers(self):
        count = {"a": 0, "b": 0}
        self.bus.on("t", lambda e: count.update(a=count["a"] + 1))
        self.bus.on("t", lambda e: count.update(b=count["b"] + 1))
        self.bus.emit("t")
        self.assertEqual(count["a"], 1)
        self.assertEqual(count["b"], 1)

    def test_wildcard(self):
        received = []
        self.bus.on("*", lambda e: received.append(e.topic))
        self.bus.emit("a")
        self.bus.emit("b")
        self.assertEqual(received, ["a", "b"])

    def test_unsubscribe(self):
        count = [0]
        handler = lambda e: count.__setitem__(0, count[0] + 1)
        self.bus.on("t", handler)
        self.bus.emit("t")
        self.assertEqual(count[0], 1)
        self.bus.off("t", handler)
        self.bus.emit("t")
        self.assertEqual(count[0], 1)

    def test_handler_error_doesnt_break_others(self):
        results = []
        def bad(e): raise ValueError("boom")
        self.bus.on("t", bad)
        self.bus.on("t", lambda e: results.append("ok"))
        self.bus.emit("t")
        self.assertEqual(results, ["ok"])

    def test_history(self):
        self.bus.emit("a", {"x": 1})
        self.bus.emit("b", {"x": 2})
        self.assertEqual(len(self.bus.history), 2)

    def test_async_emit(self):
        received = []
        async def handler(e): received.append(e.topic)
        self.bus.on("t", handler)
        run(self.bus.emit_async("t"))
        self.assertEqual(received, ["t"])

    def test_mixed_handlers(self):
        sync_r, async_r = [], []
        self.bus.on("m", lambda e: sync_r.append("s"))
        async def ah(e): async_r.append("a")
        self.bus.on("m", ah)
        run(self.bus.emit_async("m"))
        self.assertEqual(sync_r, ["s"])
        self.assertEqual(async_r, ["a"])

    def test_clear(self):
        self.bus.on("t", lambda e: None)
        self.bus.emit("t")
        self.bus.clear()
        self.assertEqual(len(self.bus.history), 0)


# ===================================================================
# Skill Tests
# ===================================================================

class TestTrustLevel(unittest.TestCase):

    def test_progression(self):
        self.assertEqual(TrustLevel.OBSERVE.next_level(), TrustLevel.SUGGEST)
        self.assertEqual(TrustLevel.SUGGEST.next_level(), TrustLevel.AUTO)
        self.assertEqual(TrustLevel.AUTO.next_level(), TrustLevel.AUTO)

    def test_auto_execute(self):
        self.assertFalse(TrustLevel.OBSERVE.can_auto_execute())
        self.assertTrue(TrustLevel.AUTO.can_auto_execute())


class TestSkillMetadata(unittest.TestCase):

    def make(self, **kw) -> SkillMetadata:
        defaults = {"name": "test_skill", "description": "A test skill"}
        defaults.update(kw)
        return SkillMetadata(**defaults)

    def test_initial_values(self):
        s = self.make()
        self.assertAlmostEqual(s.utility_score, 0.5)
        self.assertEqual(s.total_executions, 0)
        self.assertEqual(s.trust_level, TrustLevel.OBSERVE)

    def test_record_success_increases_utility(self):
        s = self.make()
        s.record_execution(success=True, learning_rate=0.1)
        self.assertGreater(s.utility_score, 0.5)
        self.assertEqual(s.success_count, 1)

    def test_record_failure_decreases_utility(self):
        s = self.make()
        s.record_execution(success=False, learning_rate=0.1)
        self.assertLess(s.utility_score, 0.5)

    def test_utility_converges_up(self):
        s = self.make(utility_score=0.5)
        for _ in range(20):
            s.record_execution(success=True, learning_rate=0.1)
        self.assertGreater(s.utility_score, 0.9)

    def test_utility_converges_down(self):
        s = self.make(utility_score=0.5)
        for _ in range(20):
            s.record_execution(success=False, learning_rate=0.1)
        self.assertLess(s.utility_score, 0.15)

    def test_utility_bounded(self):
        s = self.make(utility_score=0.99)
        s.record_execution(success=True, reward=1.0, learning_rate=0.5)
        self.assertLessEqual(s.utility_score, 1.0)
        s2 = self.make(utility_score=0.01)
        s2.record_execution(success=False, reward=0.0, learning_rate=0.5)
        self.assertGreaterEqual(s2.utility_score, 0.0)

    def test_decay(self):
        s = self.make(utility_score=0.8)
        s.last_used_at = time.time() - 5 * 86400  # 5 days ago
        s.apply_decay(decay_factor=0.95)
        expected = 0.8 * (0.95 ** 5)
        self.assertAlmostEqual(s.utility_score, expected, places=2)

    def test_no_decay_if_recent(self):
        s = self.make(utility_score=0.8)
        s.last_used_at = time.time()
        s.apply_decay(decay_factor=0.95)
        self.assertAlmostEqual(s.utility_score, 0.8)

    def test_is_stale(self):
        s = self.make(utility_score=0.1)
        s.last_used_at = time.time() - 60 * 86400
        self.assertTrue(s.is_stale)

    def test_not_stale_high_utility(self):
        s = self.make(utility_score=0.8)
        s.last_used_at = time.time() - 60 * 86400
        self.assertFalse(s.is_stale)

    def test_success_rate(self):
        s = self.make()
        s.success_count, s.failure_count, s.total_executions = 7, 3, 10
        self.assertAlmostEqual(s.success_rate, 0.7)

    def test_success_rate_zero(self):
        self.assertEqual(self.make().success_rate, 0.0)

    def test_failure_lessons_pruning(self):
        s = self.make()
        for i in range(15):
            s.add_failure_lesson(FailureLesson(
                category=FailureCategory.UNINFORMATIVE,
                description=f"fail {i}", root_cause="x", task_context="t",
            ))
        self.assertLessEqual(len(s.failure_lessons), 3)
        # Valuable ones are always kept
        for i in range(5):
            s.add_failure_lesson(FailureLesson(
                category=FailureCategory.NEAR_SUCCESS,
                description=f"near {i}", root_cause="y", task_context="t",
            ))
        valuable = [f for f in s.failure_lessons if f.category == FailureCategory.NEAR_SUCCESS]
        self.assertEqual(len(valuable), 5)

    def test_promote_trust(self):
        s = self.make()
        self.assertTrue(s.promote_trust())
        self.assertEqual(s.trust_level, TrustLevel.SUGGEST)
        self.assertTrue(s.promote_trust())
        self.assertEqual(s.trust_level, TrustLevel.AUTO)
        self.assertFalse(s.promote_trust())  # Ceiling

    def test_serialization_roundtrip(self):
        s = self.make(tags=["test"], origin=SkillOrigin.LEARNED)
        s.record_execution(success=True, learning_rate=0.1)
        s.add_failure_lesson(FailureLesson(
            category=FailureCategory.REUSABLE,
            description="Don't do X", root_cause="X breaks Y", task_context="t",
        ))
        data = s.to_dict()
        restored = SkillMetadata.from_dict(data)
        self.assertEqual(restored.name, s.name)
        self.assertAlmostEqual(restored.utility_score, s.utility_score)
        self.assertEqual(restored.origin, SkillOrigin.LEARNED)
        self.assertEqual(len(restored.failure_lessons), 1)

    def test_content_hash_deterministic(self):
        s1 = self.make(name="a", description="d")
        s2 = self.make(name="a", description="d")
        s3 = self.make(name="b", description="d")
        self.assertEqual(s1.content_hash(), s2.content_hash())
        self.assertNotEqual(s1.content_hash(), s3.content_hash())


class TestSkillResult(unittest.TestCase):

    def test_effective_reward_success(self):
        self.assertEqual(SkillResult(success=True).effective_reward, 1.0)

    def test_effective_reward_failure(self):
        self.assertEqual(SkillResult(success=False).effective_reward, 0.0)

    def test_effective_reward_explicit(self):
        self.assertEqual(SkillResult(success=True, reward=0.7).effective_reward, 0.7)


# ===================================================================
# Agent Tests
# ===================================================================

class EchoSkill(BaseSkill):
    async def execute(self, context):
        return SkillResult(success=True, output=f"Echo: {context.get('user_input', '')}")


class FailSkill(BaseSkill):
    async def execute(self, context):
        return SkillResult(success=False, error="Intentional failure")


class TestAgentLifecycle(unittest.TestCase):

    def test_start_and_idle(self):
        agent = Agent(settings=_tmp_settings())
        run(agent.start())
        self.assertEqual(agent.state, AgentState.IDLE)
        self.assertGreater(agent.uptime, 0)

    def test_shutdown(self):
        agent = Agent(settings=_tmp_settings())
        run(agent.start())
        run(agent.shutdown())
        self.assertEqual(agent.state, AgentState.STOPPED)

    def test_invalid_transition(self):
        agent = Agent(settings=_tmp_settings())
        with self.assertRaises(InvalidTransition):
            agent._transition(AgentState.ACTIVE)

    def test_state_events(self):
        agent = Agent(settings=_tmp_settings())
        transitions = []
        agent.bus.on("agent.state_changed", lambda e: transitions.append(
            (e.data["old_state"], e.data["new_state"])
        ))
        run(agent.start())
        self.assertIn(("initializing", "idle"), transitions)


class TestSkillRegistration(unittest.TestCase):

    def test_register_and_get(self):
        agent = Agent(settings=_tmp_settings())
        run(agent.start())
        skill = EchoSkill(SkillMetadata(name="echo", trigger_conditions=["echo"]))
        agent.register_skill(skill)
        self.assertEqual(agent.skill_count, 1)
        self.assertIs(agent.get_skill("echo"), skill)

    def test_unregister(self):
        agent = Agent(settings=_tmp_settings())
        run(agent.start())
        agent.register_skill(EchoSkill(SkillMetadata(name="echo")))
        agent.unregister_skill("echo")
        self.assertEqual(agent.skill_count, 0)

    def test_find_skill_for_intent(self):
        agent = Agent(settings=_tmp_settings())
        run(agent.start())
        skill = EchoSkill(SkillMetadata(name="echo", trigger_conditions=["echo", "repeat"]))
        agent.register_skill(skill)
        results = agent.find_skill_for_intent("echo this")
        self.assertEqual(len(results), 1)
        self.assertIs(results[0][0], skill)
        self.assertEqual(agent.find_skill_for_intent("unrelated"), [])


class TestRequestHandling(unittest.TestCase):

    def test_success(self):
        agent = Agent(settings=_tmp_settings())
        run(agent.start())
        agent.register_skill(EchoSkill(SkillMetadata(
            name="echo", trigger_conditions=["echo"], trust_level=TrustLevel.AUTO)))
        result = run(agent.handle_request("echo hello"))
        self.assertIn("Echo: echo hello", result)
        self.assertEqual(agent.state, AgentState.IDLE)
        self.assertEqual(agent.stats.successful_tasks, 1)

    def test_no_skill(self):
        agent = Agent(settings=_tmp_settings())
        run(agent.start())
        result = run(agent.handle_request("unknown"))
        self.assertIn("don't have a Skill", result)

    def test_failure(self):
        agent = Agent(settings=_tmp_settings())
        run(agent.start())
        agent.register_skill(FailSkill(SkillMetadata(
            name="fail", trigger_conditions=["fail"], trust_level=TrustLevel.AUTO)))
        result = run(agent.handle_request("fail now"))
        self.assertIn("failed", result.lower())

    def test_utility_updated(self):
        agent = Agent(settings=_tmp_settings())
        run(agent.start())
        skill = EchoSkill(SkillMetadata(
            name="echo", trigger_conditions=["echo"],
            utility_score=0.5, trust_level=TrustLevel.AUTO))
        agent.register_skill(skill)
        run(agent.handle_request("echo test"))
        self.assertGreater(skill.metadata.utility_score, 0.5)


class TestReflection(unittest.TestCase):

    def test_enter_and_exit(self):
        agent = Agent(settings=_tmp_settings())
        run(agent.start())
        events = []
        agent.bus.on("agent.reflection_started", lambda e: events.append("start"))
        agent.bus.on("agent.reflection_completed", lambda e: events.append("end"))
        run(agent.enter_reflection())
        self.assertEqual(events, ["start", "end"])
        self.assertEqual(agent.state, AgentState.IDLE)


# ===================================================================
# Config Tests
# ===================================================================

class TestConfig(unittest.TestCase):

    def test_defaults(self):
        s = Settings()
        self.assertEqual(s.agent.name, "agent-001")
        self.assertEqual(s.llm.model, "gpt-4o-mini")
        self.assertAlmostEqual(s.evolution.decay_factor, 0.95)

    def test_load_from_toml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[agent]\nname = "my-agent"\n\n[llm]\nmodel = "claude-sonnet-4-20250514"\n')
            f.flush()
            s = load_settings(config_path=Path(f.name))
        self.assertEqual(s.agent.name, "my-agent")
        self.assertEqual(s.llm.model, "claude-sonnet-4-20250514")
        self.assertEqual(s.trust.default_level, "observe")  # default

    def test_resolved_data_dir(self):
        s = Settings()
        self.assertTrue(s.agent.resolved_data_dir.is_absolute())

    def test_missing_config(self):
        s = load_settings(config_path=Path("/tmp/nonexistent.toml"))
        self.assertEqual(s.agent.name, "agent-001")

    def test_singleton_reset(self):
        reset_settings()
        reset_settings()  # Should not raise


class TestStatusDict(unittest.TestCase):

    def test_status_contents(self):
        agent = Agent(settings=_tmp_settings())
        run(agent.start())
        agent.register_skill(EchoSkill(SkillMetadata(name="echo")))
        status = agent.status_dict()
        self.assertEqual(status["state"], "idle")
        self.assertEqual(status["skill_count"], 1)
        self.assertIn("echo", status["active_skills"])


# ===================================================================
# Run
# ===================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
