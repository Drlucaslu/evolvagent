"""
SummarizeSkill: Summarize text or files using LLM.

Trust level AUTO — no side effects, purely read-only text transformation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..core.llm import LLMClient
from ..core.skill import BaseSkill, SkillMetadata, SkillOrigin, SkillResult, TrustLevel

_SYSTEM_PROMPT = """\
You are a concise summarizer. Summarize the given text clearly and accurately.
- Use bullet points for multi-topic content
- Keep the summary under 200 words unless asked otherwise
- Preserve key facts, numbers, and names
- Write in the same language as the input text
"""


class SummarizeSkill(BaseSkill):
    """Summarize text or file contents using LLM."""

    def __init__(self, llm_client: LLMClient):
        super().__init__(metadata=SkillMetadata(
            name="summarize",
            category="text",
            description="Summarize text, articles, or file contents",
            tags=["summarize", "summary", "tldr", "digest"],
            trigger_conditions=["summarize", "summary", "tldr", "sum up", "brief"],
            trust_level=TrustLevel.AUTO,
            origin=SkillOrigin.BUILTIN,
            interaction_mode="active",
        ))
        self.llm = llm_client

    def can_handle(self, intent: str, context: dict[str, Any] | None = None) -> float:
        intent_lower = intent.lower()
        for trigger in self.metadata.trigger_conditions:
            if trigger in intent_lower:
                return 0.85
        return 0.0

    async def execute(self, context: dict[str, Any]) -> SkillResult:
        user_input = context.get("user_input", "")
        if not user_input.strip():
            return SkillResult(success=False, error="Empty input")

        # Check if user references a file path
        text_to_summarize = user_input
        for word in user_input.split():
            p = Path(word).expanduser()
            if p.is_file():
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                    if len(content) > 50000:
                        content = content[:50000] + "\n... (truncated)"
                    text_to_summarize = f"Summarize this file ({p.name}):\n\n{content}"
                except Exception as e:
                    return SkillResult(success=False, error=f"Failed to read file: {e}")
                break

        start = time.time()
        try:
            response = await self.llm.complete(
                prompt=text_to_summarize,
                system=_SYSTEM_PROMPT,
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
