"""
Reflection engine for EvolvAgent.

LLM-driven offline reflection that analyzes skill execution history,
extracts distilled principles, and updates skill metadata. This is the
core of the "self-evolving" capability.

Usage:
    engine = ReflectionEngine(llm=llm_client, bus=event_bus, store=skill_store)
    result = await engine.reflect(skills)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .events import EventBus
    from .llm import LLMClient
    from .skill import BaseSkill, SkillMetadata
    from .storage import SkillStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ReflectionResult:
    """Summary of a reflection cycle."""

    skills_analyzed: int = 0
    skills_updated: int = 0
    principles_extracted: int = 0
    errors: list[str] = field(default_factory=list)
    skipped_reason: str = ""  # non-empty = entire reflection skipped


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

# Minimum seconds between reflections on the same skill (7 days)
_RE_REFLECT_INTERVAL = 7 * 86400
_MIN_EXECUTIONS_FOR_FIRST_REFLECT = 5
_MAX_PRINCIPLES = 20


class ReflectionEngine:
    """
    Analyzes skill execution history and uses an LLM to extract
    reusable principles (distilled knowledge).
    """

    def __init__(
        self,
        llm: LLMClient | None,
        bus: EventBus,
        store: SkillStore | None,
    ):
        self._llm = llm
        self._bus = bus
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reflect(self, skills: dict[str, BaseSkill]) -> ReflectionResult:
        """Run reflection on all eligible skills. Returns a summary."""
        if self._llm is None:
            return ReflectionResult(skipped_reason="No LLM client available")

        eligible = self._find_eligible_skills(skills)
        if not eligible:
            return ReflectionResult(skipped_reason="No skills eligible for reflection")

        result = ReflectionResult(skills_analyzed=len(eligible))

        for skill in eligible:
            meta = skill.metadata
            try:
                new_principles = await self._reflect_on_skill(meta)
                if new_principles:
                    # Deduplicate against existing
                    existing_lower = {p.lower() for p in meta.distilled_principles}
                    unique = [p for p in new_principles if p.lower() not in existing_lower]

                    if unique:
                        meta.distilled_principles.extend(unique)
                        # Cap at max
                        if len(meta.distilled_principles) > _MAX_PRINCIPLES:
                            meta.distilled_principles = meta.distilled_principles[
                                -_MAX_PRINCIPLES:
                            ]
                        result.principles_extracted += len(unique)
                        result.skills_updated += 1

                meta.last_reflected_at = time.time()

                # Persist
                if self._store:
                    self._store.save(meta)

                # Emit event
                await self._bus.emit_async(
                    "skill.reflected",
                    {
                        "skill_name": meta.name,
                        "new_principles": unique if new_principles else [],
                        "total_principles": len(meta.distilled_principles),
                    },
                    source="reflection",
                )

            except Exception as e:
                logger.warning("Reflection failed for skill '%s': %s", meta.name, e)
                result.errors.append(f"{meta.name}: {e}")

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_eligible_skills(self, skills: dict[str, BaseSkill]) -> list[BaseSkill]:
        """Return skills that qualify for reflection."""
        from .skill import SkillStatus

        eligible = []
        now = time.time()

        for skill in skills.values():
            meta = skill.metadata
            if meta.status != SkillStatus.ACTIVE:
                continue

            # Condition 1: has valuable failure lessons added after last reflection
            has_new_failures = any(
                f.is_valuable() and f.timestamp > meta.last_reflected_at
                for f in meta.failure_lessons
            )
            if has_new_failures:
                eligible.append(skill)
                continue

            # Condition 2: enough executions, never reflected
            if meta.total_executions >= _MIN_EXECUTIONS_FOR_FIRST_REFLECT and meta.last_reflected_at == 0:
                eligible.append(skill)
                continue

            # Condition 3: last reflection > 7 days ago AND new executions since
            if (
                meta.last_reflected_at > 0
                and (now - meta.last_reflected_at) > _RE_REFLECT_INTERVAL
                and meta.last_used_at > meta.last_reflected_at
            ):
                eligible.append(skill)
                continue

        return eligible

    async def _reflect_on_skill(self, meta: SkillMetadata) -> list[str]:
        """Use LLM to extract principles from a single skill's history."""
        prompt = self._build_reflection_prompt(meta)
        system = (
            "You are a reflection engine for an AI agent. "
            "Analyze the skill's execution history and extract reusable principles. "
            "Output each principle as a bullet starting with '- '. "
            "Each principle should be a concise, actionable insight (10-80 characters). "
            "If there is nothing useful to extract, output exactly: NONE"
        )

        response = await self._llm.complete(
            prompt=prompt,
            system=system,
            temperature=0.2,
            max_tokens=512,
        )

        return self._parse_principles(response.content)

    def _build_reflection_prompt(self, meta: SkillMetadata) -> str:
        """Build the user prompt for reflection LLM call."""
        parts = [
            f"Skill: {meta.name}",
            f"Description: {meta.description}",
            f"Executions: {meta.total_executions} (success: {meta.success_count}, "
            f"fail: {meta.failure_count})",
            f"Success rate: {meta.success_rate:.0%}",
            f"Utility score: {meta.utility_score:.2f}",
        ]

        # Existing principles (to avoid repetition)
        if meta.distilled_principles:
            parts.append("\nExisting principles (do NOT repeat these):")
            for p in meta.distilled_principles:
                parts.append(f"  - {p}")

        # Recent valuable failure lessons (last 5)
        valuable = [f for f in meta.failure_lessons if f.is_valuable()]
        recent = sorted(valuable, key=lambda f: f.timestamp, reverse=True)[:5]
        if recent:
            parts.append("\nRecent valuable failure lessons:")
            for f in recent:
                parts.append(
                    f"  [{f.category.value}] {f.description} "
                    f"(root cause: {f.root_cause})"
                )

        parts.append(
            "\nExtract reusable principles from this skill's history. "
            "Focus on patterns, pitfalls, and best practices."
        )

        return "\n".join(parts)

    @staticmethod
    def _parse_principles(content: str) -> list[str]:
        """Parse LLM output into a list of principle strings."""
        content = content.strip()
        if not content or content.upper() == "NONE":
            return []

        principles = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip bullet prefixes: "- ", "* ", "1. ", "2) " etc.
            cleaned = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", line).strip()
            # Filter out short fragments (<= 5 chars)
            if len(cleaned) > 5:
                principles.append(cleaned)

        return principles
