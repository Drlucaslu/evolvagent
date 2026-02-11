"""Tests for trust level enforcement and built-in skills."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolvagent.core.agent import Agent
from evolvagent.core.config import Settings, AgentConfig
from evolvagent.core.llm import LLMResponse
from evolvagent.core.skill import BaseSkill, SkillMetadata, SkillResult, TrustLevel
from evolvagent.skills.shell import ShellCommandSkill
from evolvagent.skills.summarize import SummarizeSkill


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


# ===================================================================
# Trust Level Tests
# ===================================================================

class DummySkill(BaseSkill):
    async def execute(self, context):
        return SkillResult(success=True, output="executed")

    async def preview(self, context):
        return "Would do something"


class TestTrustObserve(unittest.TestCase):

    def test_observe_returns_preview_not_executed(self):
        agent = Agent(settings=_tmp_settings())
        skill = DummySkill(SkillMetadata(
            name="obs", trigger_conditions=["test"], trust_level=TrustLevel.OBSERVE))

        async def _test():
            await agent.start()
            agent.register_skill(skill)
            result = await agent.handle_request("test this")
            await agent.shutdown()
            return result

        result = run(_test())
        self.assertIn("[OBSERVE]", result)
        self.assertIn("Would do something", result)
        # Should NOT have executed
        self.assertEqual(agent.stats.successful_tasks, 0)
        self.assertEqual(skill.metadata.total_executions, 0)


class TestTrustSuggest(unittest.TestCase):

    def test_suggest_without_callback_returns_preview(self):
        agent = Agent(settings=_tmp_settings())
        skill = DummySkill(SkillMetadata(
            name="sug", trigger_conditions=["test"], trust_level=TrustLevel.SUGGEST))

        async def _test():
            await agent.start()
            agent.register_skill(skill)
            result = await agent.handle_request("test this")
            await agent.shutdown()
            return result

        result = run(_test())
        self.assertIn("[SUGGEST]", result)
        self.assertEqual(skill.metadata.total_executions, 0)

    def test_suggest_approved_executes(self):
        agent = Agent(settings=_tmp_settings())
        skill = DummySkill(SkillMetadata(
            name="sug", trigger_conditions=["test"], trust_level=TrustLevel.SUGGEST))

        async def approve(name, preview):
            return True

        async def _test():
            await agent.start()
            agent.register_skill(skill)
            result = await agent.handle_request("test this", confirm_callback=approve)
            await agent.shutdown()
            return result

        result = run(_test())
        self.assertEqual(result, "executed")
        self.assertEqual(agent.stats.successful_tasks, 1)

    def test_suggest_rejected_cancels(self):
        agent = Agent(settings=_tmp_settings())
        skill = DummySkill(SkillMetadata(
            name="sug", trigger_conditions=["test"], trust_level=TrustLevel.SUGGEST))

        async def reject(name, preview):
            return False

        async def _test():
            await agent.start()
            agent.register_skill(skill)
            result = await agent.handle_request("test this", confirm_callback=reject)
            await agent.shutdown()
            return result

        result = run(_test())
        self.assertIn("cancelled", result.lower())
        self.assertEqual(skill.metadata.total_executions, 0)


class TestTrustAuto(unittest.TestCase):

    def test_auto_executes_directly(self):
        agent = Agent(settings=_tmp_settings())
        skill = DummySkill(SkillMetadata(
            name="auto", trigger_conditions=["test"], trust_level=TrustLevel.AUTO))

        async def _test():
            await agent.start()
            agent.register_skill(skill)
            result = await agent.handle_request("test this")
            await agent.shutdown()
            return result

        result = run(_test())
        self.assertEqual(result, "executed")
        self.assertEqual(agent.stats.successful_tasks, 1)


# ===================================================================
# Trust Auto-Promotion Tests
# ===================================================================

class TestTrustAutoPromotion(unittest.TestCase):

    def test_promote_after_threshold(self):
        """Skill should auto-promote after promote_threshold successes."""
        settings = _tmp_settings()
        settings.trust.promote_threshold = 3
        agent = Agent(settings=settings)
        skill = DummySkill(SkillMetadata(
            name="promo", trigger_conditions=["test"],
            trust_level=TrustLevel.OBSERVE,
            # Pre-set success_count to threshold - 1
            success_count=2, total_executions=2,
        ))

        async def approve(name, preview):
            return True

        async def _test():
            await agent.start()
            # Set to SUGGEST so it can actually execute (with approval)
            skill.metadata.trust_level = TrustLevel.SUGGEST
            agent.register_skill(skill)
            # This execution (success #3) should trigger promotion
            await agent.handle_request("test this", confirm_callback=approve)
            await agent.shutdown()

        run(_test())
        # success_count is now 3, which >= threshold 3
        self.assertEqual(skill.metadata.trust_level, TrustLevel.AUTO)

    def test_no_promote_below_threshold(self):
        settings = _tmp_settings()
        settings.trust.promote_threshold = 10
        agent = Agent(settings=settings)
        skill = DummySkill(SkillMetadata(
            name="nopromo", trigger_conditions=["test"],
            trust_level=TrustLevel.SUGGEST,
            success_count=0, total_executions=0,
        ))

        async def approve(name, preview):
            return True

        async def _test():
            await agent.start()
            agent.register_skill(skill)
            await agent.handle_request("test this", confirm_callback=approve)
            await agent.shutdown()

        run(_test())
        # Only 1 success, threshold is 10 — should not promote
        self.assertEqual(skill.metadata.trust_level, TrustLevel.SUGGEST)

    def test_promote_emits_event(self):
        settings = _tmp_settings()
        settings.trust.promote_threshold = 1
        agent = Agent(settings=settings)
        skill = DummySkill(SkillMetadata(
            name="ev", trigger_conditions=["test"],
            trust_level=TrustLevel.SUGGEST,
            success_count=0, total_executions=0,
        ))
        events = []

        async def approve(name, preview):
            return True

        async def _test():
            await agent.start()
            agent.bus.on("skill.trust_promoted", lambda e: events.append(e.data))
            agent.register_skill(skill)
            await agent.handle_request("test this", confirm_callback=approve)
            await agent.shutdown()

        run(_test())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["skill_name"], "ev")
        self.assertEqual(events[0]["new_trust"], "auto")


# ===================================================================
# Conversation History Tests
# ===================================================================

class TestConversationHistory(unittest.TestCase):

    def test_history_passed_to_skill(self):
        """History in context should reach the skill's execute method."""
        received_ctx = {}

        class SpySkill(BaseSkill):
            async def execute(self, context):
                received_ctx.update(context)
                return SkillResult(success=True, output="ok")

        agent = Agent(settings=_tmp_settings())
        skill = SpySkill(SkillMetadata(
            name="spy", trigger_conditions=["test"], trust_level=TrustLevel.AUTO))

        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        async def _test():
            await agent.start()
            agent.register_skill(skill)
            await agent.handle_request("test this", context={"history": history})
            await agent.shutdown()

        run(_test())
        self.assertEqual(received_ctx["history"], history)


# ===================================================================
# ShellCommandSkill Tests
# ===================================================================

class TestShellCommandSkill(unittest.TestCase):

    def test_can_handle_strong_match(self):
        skill = ShellCommandSkill(_mock_llm())
        self.assertGreaterEqual(skill.can_handle("run ls -la"), 0.9)
        self.assertGreaterEqual(skill.can_handle("execute a command"), 0.9)

    def test_can_handle_weak_match(self):
        skill = ShellCommandSkill(_mock_llm())
        self.assertGreater(skill.can_handle("list files here"), 0)

    def test_can_handle_no_match(self):
        skill = ShellCommandSkill(_mock_llm())
        self.assertEqual(skill.can_handle("what is python?"), 0.0)

    def test_preview_shows_command(self):
        skill = ShellCommandSkill(_mock_llm("ls -la"))
        result = run(skill.preview({"user_input": "list files"}))
        self.assertIn("ls -la", result)
        self.assertIn("$", result)

    def test_execute_success(self):
        skill = ShellCommandSkill(_mock_llm("echo hello"))
        result = run(skill.execute({"user_input": "say hello"}))
        self.assertTrue(result.success)
        self.assertIn("hello", result.output)

    def test_execute_failure(self):
        skill = ShellCommandSkill(_mock_llm("false"))  # 'false' command returns exit 1
        result = run(skill.execute({"user_input": "fail"}))
        self.assertFalse(result.success)

    def test_metadata(self):
        skill = ShellCommandSkill(_mock_llm())
        self.assertEqual(skill.metadata.name, "shell_command")
        self.assertEqual(skill.metadata.trust_level, TrustLevel.SUGGEST)

    def test_strips_code_fences(self):
        skill = ShellCommandSkill(_mock_llm("```bash\nls -la\n```"))
        result = run(skill.preview({"user_input": "list files"}))
        self.assertIn("ls -la", result)
        self.assertNotIn("```", result)


# ===================================================================
# SummarizeSkill Tests
# ===================================================================

class TestSummarizeSkill(unittest.TestCase):

    def test_can_handle(self):
        skill = SummarizeSkill(_mock_llm())
        self.assertGreater(skill.can_handle("summarize this text"), 0.5)
        self.assertGreater(skill.can_handle("give me a tldr"), 0.5)
        self.assertEqual(skill.can_handle("what is python?"), 0.0)

    def test_execute(self):
        skill = SummarizeSkill(_mock_llm("This is a summary."))
        result = run(skill.execute({"user_input": "summarize: long text here"}))
        self.assertTrue(result.success)
        self.assertIn("summary", result.output)

    def test_metadata(self):
        skill = SummarizeSkill(_mock_llm())
        self.assertEqual(skill.metadata.name, "summarize")
        self.assertEqual(skill.metadata.trust_level, TrustLevel.AUTO)

    def test_empty_input(self):
        skill = SummarizeSkill(_mock_llm())
        result = run(skill.execute({"user_input": ""}))
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main(verbosity=2)
