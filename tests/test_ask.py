"""Tests for GeneralAssistantSkill and ask command integration."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolvagent.core.agent import Agent
from evolvagent.core.config import Settings, AgentConfig
from evolvagent.core.llm import LLMResponse
from evolvagent.core.skill import SkillMetadata, SkillResult, BaseSkill, TrustLevel
from evolvagent.skills.general import GeneralAssistantSkill


def _tmp_settings():
    return Settings(agent=AgentConfig(data_dir=tempfile.mkdtemp()))


def run(coro):
    return asyncio.run(coro)


def _make_mock_llm(content="Mock response", cost=0.001):
    """Create a mock LLMClient."""
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content=content,
        model="mock-model",
        tokens_in=10,
        tokens_out=20,
        cost_usd=cost,
    ))
    return llm


class TestGeneralAssistantSkill(unittest.TestCase):

    def test_can_handle_returns_low_confidence(self):
        skill = GeneralAssistantSkill(_make_mock_llm())
        self.assertAlmostEqual(skill.can_handle("anything"), 0.3)

    def test_execute_success(self):
        llm = _make_mock_llm("Python is a programming language.")
        skill = GeneralAssistantSkill(llm)
        result = run(skill.execute({"user_input": "What is Python?"}))
        self.assertTrue(result.success)
        self.assertIn("Python", result.output)
        self.assertGreater(result.execution_time_ms, 0)
        llm.complete.assert_called_once()

    def test_execute_empty_input(self):
        skill = GeneralAssistantSkill(_make_mock_llm())
        result = run(skill.execute({"user_input": ""}))
        self.assertFalse(result.success)
        self.assertIn("Empty", result.error)

    def test_execute_llm_error(self):
        llm = _make_mock_llm()
        llm.complete = AsyncMock(side_effect=RuntimeError("API down"))
        skill = GeneralAssistantSkill(llm)
        result = run(skill.execute({"user_input": "hello"}))
        self.assertFalse(result.success)
        self.assertIn("API down", result.error)

    def test_metadata_is_builtin(self):
        skill = GeneralAssistantSkill(_make_mock_llm())
        self.assertEqual(skill.metadata.name, "general_assistant")
        self.assertEqual(skill.metadata.origin.value, "builtin")


class TestAskIntegration(unittest.TestCase):
    """Integration test: Agent + GeneralAssistantSkill end-to-end."""

    def test_ask_uses_general_assistant_as_fallback(self):
        agent = Agent(settings=_tmp_settings())
        llm = _make_mock_llm("42 is the answer.")
        skill = GeneralAssistantSkill(llm)

        async def _test():
            await agent.start()
            agent.register_skill(skill)
            result = await agent.handle_request("What is the meaning of life?")
            await agent.shutdown()
            return result

        result = run(_test())
        self.assertIn("42", result)
        self.assertEqual(agent.stats.successful_tasks, 1)

    def test_specialized_skill_takes_priority(self):
        """When a specialized Skill matches, it should beat GeneralAssistant."""

        class MathSkill(BaseSkill):
            def can_handle(self, intent, context=None):
                return 0.9 if "calculate" in intent.lower() else 0.0
            async def execute(self, context):
                return SkillResult(success=True, output="Result: 4")

        agent = Agent(settings=_tmp_settings())
        general = GeneralAssistantSkill(_make_mock_llm("LLM says 4"))
        math_skill = MathSkill(SkillMetadata(
            name="math", trigger_conditions=["calculate"], trust_level=TrustLevel.AUTO))

        async def _test():
            await agent.start()
            agent.register_skill(general)
            agent.register_skill(math_skill)
            result = await agent.handle_request("calculate 2+2")
            await agent.shutdown()
            return result

        result = run(_test())
        self.assertEqual(result, "Result: 4")  # MathSkill, not GeneralAssistant

    def test_utility_updates_after_ask(self):
        agent = Agent(settings=_tmp_settings())
        llm = _make_mock_llm("Hello!")
        skill = GeneralAssistantSkill(llm)
        initial_utility = skill.metadata.utility_score

        async def _test():
            await agent.start()
            agent.register_skill(skill)
            await agent.handle_request("say hello")
            await agent.shutdown()

        run(_test())
        self.assertGreater(skill.metadata.utility_score, initial_utility)
        self.assertEqual(skill.metadata.total_executions, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
