"""
Built-in Skills for EvolvAgent.

GeneralAssistantSkill: LLM-backed fallback for general question answering.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.llm import LLMClient
from ..core.skill import BaseSkill, SkillMetadata, SkillOrigin, SkillResult, TrustLevel


class GeneralAssistantSkill(BaseSkill):
    """
    General-purpose question answering Skill.

    Wraps LLMClient to answer any user query. Acts as a low-priority fallback
    so that specialized Skills take precedence when available.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(metadata=SkillMetadata(
            name="general_assistant",
            category="general",
            description="General-purpose AI assistant for answering questions and helping with tasks",
            tags=["chat", "question", "help", "general"],
            trust_level=TrustLevel.AUTO,
            origin=SkillOrigin.BUILTIN,
            interaction_mode="active",
        ))
        self.llm = llm_client

    def can_handle(self, intent: str, context: dict[str, Any] | None = None) -> float:
        """Always returns low confidence — acts as fallback."""
        return 0.3

    async def execute(self, context: dict[str, Any]) -> SkillResult:
        """Send user input to LLM and return the response."""
        user_input = context.get("user_input", "")
        if not user_input.strip():
            return SkillResult(success=False, error="Empty input")

        history = context.get("history")

        start = time.time()
        try:
            response = await self.llm.complete(
                prompt=user_input,
                system=(
                    "You are a helpful, concise assistant. "
                    "Answer the user's question directly and clearly."
                ),
                history=history,
            )
            elapsed_ms = (time.time() - start) * 1000
            return SkillResult(
                success=True,
                output=response.content,
                data={"model": response.model, "cost_usd": response.cost_usd},
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return SkillResult(
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms,
            )
