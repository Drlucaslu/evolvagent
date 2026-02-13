"""
Skill learning engine for EvolvAgent.

Enables the agent to create new skills from interaction patterns:
- DynamicSkill: a runtime-created skill backed by an LLM prompt template
- SkillLearner: observes interaction patterns and proposes new skills

This is the core of the "self-evolving" capability — the agent learns
new abilities beyond its built-in skill set.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .skill import (
    BaseSkill,
    SkillMetadata,
    SkillResult,
)

if TYPE_CHECKING:
    from .events import EventBus
    from .llm import LLMClient
    from .storage import SkillStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Interaction pattern tracking
# ---------------------------------------------------------------------------

_MAX_PATTERN_HISTORY = 200
_MIN_SIMILAR_FOR_PROPOSAL = 3
_SIMILARITY_KEYWORDS_THRESHOLD = 2


@dataclass
class InteractionRecord:
    """A recorded user interaction for pattern analysis."""
    query: str
    skill_used: str  # "" = no skill found
    success: bool
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# DynamicSkill — LLM-template skill created at runtime
# ---------------------------------------------------------------------------


class DynamicSkill(BaseSkill):
    """
    A skill created at runtime from a learned LLM prompt template.

    When executed, it calls the LLM with the skill's system_prompt
    combined with the user's input.
    """

    def __init__(
        self,
        metadata: SkillMetadata,
        llm: LLMClient,
        system_prompt: str,
    ):
        super().__init__(metadata)
        self._llm = llm
        self.system_prompt = system_prompt

    async def execute(self, context: dict[str, Any]) -> SkillResult:
        user_input = context.get("user_input", "")
        if not user_input:
            return SkillResult(success=False, error="No input provided")

        start = time.time()
        try:
            response = await self._llm.complete(
                prompt=user_input,
                system=self.system_prompt,
                history=context.get("history"),
            )
            elapsed = (time.time() - start) * 1000
            return SkillResult(
                success=True,
                output=response.content,
                execution_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return SkillResult(
                success=False,
                error=str(e),
                execution_time_ms=elapsed,
            )

    async def preview(self, context: dict[str, Any]) -> str:
        return (
            f"[{self.metadata.name}] {self.metadata.description}\n"
            f"Will answer using specialized prompt ({len(self.system_prompt)} chars)"
        )

    def can_handle(self, intent: str, context: dict[str, Any] | None = None) -> float:
        """Match against trigger conditions + keyword overlap."""
        intent_lower = intent.lower()
        # Check trigger conditions first
        for trigger in self.metadata.trigger_conditions:
            if trigger.lower() in intent_lower:
                return 0.75
        # Fallback: check tag overlap
        intent_words = set(intent_lower.split())
        tag_matches = sum(1 for t in self.metadata.tags if t.lower() in intent_words)
        if tag_matches >= 2:
            return 0.6
        if tag_matches == 1:
            return 0.3
        return 0.0

    def to_definition(self) -> dict[str, Any]:
        """Serialize the skill definition for storage."""
        return {
            "type": "dynamic",
            "system_prompt": self.system_prompt,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_definition(cls, defn: dict[str, Any], llm: LLMClient) -> DynamicSkill:
        """Recreate a DynamicSkill from a stored definition."""
        meta = SkillMetadata.from_dict(defn["metadata"])
        return cls(
            metadata=meta,
            llm=llm,
            system_prompt=defn["system_prompt"],
        )


# ---------------------------------------------------------------------------
# Learning result
# ---------------------------------------------------------------------------


@dataclass
class LearnResult:
    """Summary of a learning cycle."""
    patterns_analyzed: int = 0
    skills_proposed: int = 0
    skills_created: int = 0
    errors: list[str] = field(default_factory=list)
    skipped_reason: str = ""
    created_skill_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SkillLearner
# ---------------------------------------------------------------------------


class SkillLearner:
    """
    Observes interaction patterns and creates new DynamicSkills.

    Learning triggers:
    1. Repeated requests that no skill can handle (misses)
    2. Repeated fallback to GeneralAssistant for similar queries
    3. Explicit user teaching (/teach command)
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
        self._history: list[InteractionRecord] = []

    def record(self, query: str, skill_used: str, success: bool) -> None:
        """Record an interaction for pattern analysis."""
        self._history.append(InteractionRecord(
            query=query, skill_used=skill_used, success=success,
        ))
        if len(self._history) > _MAX_PATTERN_HISTORY:
            self._history = self._history[-_MAX_PATTERN_HISTORY:]

    # ------------------------------------------------------------------
    # Auto-learning from patterns
    # ------------------------------------------------------------------

    async def analyze_and_learn(
        self,
        existing_skills: dict[str, BaseSkill],
    ) -> LearnResult:
        """Analyze interaction history and propose new skills."""
        if self._llm is None:
            return LearnResult(skipped_reason="No LLM client available")

        if len(self._history) < _MIN_SIMILAR_FOR_PROPOSAL:
            return LearnResult(skipped_reason="Not enough interaction history")

        # Find clusters of unhandled or poorly-handled patterns
        clusters = self._find_learnable_clusters(existing_skills)
        if not clusters:
            return LearnResult(
                patterns_analyzed=len(self._history),
                skipped_reason="No learnable patterns found",
            )

        result = LearnResult(patterns_analyzed=len(self._history))

        for cluster in clusters:
            try:
                defn = await self._propose_skill(cluster, existing_skills)
                if defn:
                    result.skills_proposed += 1
                    # Check if we already have a skill with this name
                    name = defn["metadata"]["name"]
                    if name not in existing_skills:
                        result.skills_created += 1
                        result.created_skill_names.append(name)
                        # Store definition for later hydration
                        if self._store:
                            self._store.save_skill_definition(name, defn)
                        await self._bus.emit_async(
                            "skill.learned",
                            {
                                "skill_name": name,
                                "trigger_queries": cluster,
                                "description": defn["metadata"]["description"],
                            },
                            source="learner",
                        )
            except Exception as e:
                logger.warning("Failed to learn skill from cluster: %s", e)
                result.errors.append(str(e))

        return result

    def _find_learnable_clusters(
        self,
        existing_skills: dict[str, BaseSkill],
    ) -> list[list[str]]:
        """Find clusters of similar unhandled/poorly-handled queries."""
        # Bucket 1: queries with no skill match
        misses = [r.query for r in self._history if not r.skill_used]

        # Bucket 2: queries that fell back to general_assistant
        fallbacks = [
            r.query for r in self._history
            if r.skill_used == "general_assistant"
        ]

        clusters = []

        # Find groups of similar queries within each bucket
        for bucket in [misses, fallbacks]:
            if len(bucket) < _MIN_SIMILAR_FOR_PROPOSAL:
                continue
            groups = self._cluster_by_keywords(bucket)
            for group in groups:
                if len(group) >= _MIN_SIMILAR_FOR_PROPOSAL:
                    # Deduplicate and take representative samples
                    clusters.append(group[:5])

        return clusters

    @staticmethod
    def _cluster_by_keywords(queries: list[str]) -> list[list[str]]:
        """Simple keyword-overlap clustering."""
        if not queries:
            return []

        # Extract keyword sets for each query
        query_words = []
        for q in queries:
            words = set(re.findall(r'\b[a-z]{3,}\b', q.lower()))
            # Remove very common words
            words -= {
                "the", "and", "for", "that", "this", "with", "from",
                "what", "how", "can", "you", "please", "help", "want",
                "need", "about", "some", "have", "are", "was", "were",
            }
            query_words.append((q, words))

        used = set()
        clusters: list[list[str]] = []

        for i, (q1, w1) in enumerate(query_words):
            if i in used or not w1:
                continue
            cluster = [q1]
            used.add(i)
            for j, (q2, w2) in enumerate(query_words):
                if j in used or not w2:
                    continue
                overlap = len(w1 & w2)
                if overlap >= _SIMILARITY_KEYWORDS_THRESHOLD:
                    cluster.append(q2)
                    used.add(j)
            clusters.append(cluster)

        return clusters

    async def _propose_skill(
        self,
        sample_queries: list[str],
        existing_skills: dict[str, BaseSkill],
    ) -> dict[str, Any] | None:
        """Use LLM to propose a skill definition from sample queries."""
        existing_names = list(existing_skills.keys())

        system = (
            "You are a skill designer for an AI agent. "
            "Given sample user queries that the agent couldn't handle well, "
            "design a new specialized skill.\n\n"
            "Respond with ONLY valid JSON (no markdown fences):\n"
            "{\n"
            '  "name": "skill_name_snake_case",\n'
            '  "description": "What this skill does (1 sentence)",\n'
            '  "category": "category_name",\n'
            '  "tags": ["keyword1", "keyword2", ...],\n'
            '  "trigger_conditions": ["phrase that triggers this skill", ...],\n'
            '  "system_prompt": "You are a specialist in X. Your task is to Y..."\n'
            "}\n\n"
            "The system_prompt should be detailed enough to guide the LLM "
            "when this skill is executed. Include format instructions if needed.\n"
            "If these queries are too diverse or trivial to warrant a new skill, "
            "respond with: NONE"
        )

        query_list = "\n".join(f"- {q}" for q in sample_queries)
        prompt = (
            f"Existing skills (avoid duplicates): {', '.join(existing_names)}\n\n"
            f"Sample user queries that weren't handled well:\n{query_list}\n\n"
            f"Design a specialized skill for this pattern."
        )

        response = await self._llm.complete(
            prompt=prompt,
            system=system,
            temperature=0.3,
            max_tokens=1024,
        )

        return self._parse_skill_definition(response.content)

    @staticmethod
    def _parse_skill_definition(content: str) -> dict[str, Any] | None:
        """Parse LLM response into a skill definition dict."""
        content = content.strip()
        if not content or content.upper() == "NONE":
            return None

        # Strip markdown code fences if present
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse skill definition JSON")
            return None

        # Validate required fields
        required = {"name", "description", "system_prompt", "trigger_conditions"}
        if not required.issubset(data.keys()):
            logger.warning("Skill definition missing required fields: %s",
                           required - data.keys())
            return None

        # Sanitize name
        name = re.sub(r'[^a-z0-9_]', '_', data["name"].lower().strip())
        if not name or len(name) < 3:
            return None

        return {
            "type": "dynamic",
            "system_prompt": data["system_prompt"],
            "metadata": {
                "skill_id": name[:12],
                "name": name,
                "category": data.get("category", "learned"),
                "description": data["description"],
                "tags": data.get("tags", []),
                "trigger_conditions": data["trigger_conditions"],
                "trust_level": "observe",
                "origin": "learned",
                "status": "active",
                "utility_score": 0.5,
                "total_executions": 0,
                "success_count": 0,
                "failure_count": 0,
                "created_at": time.time(),
                "last_used_at": time.time(),
                "last_reflected_at": 0.0,
                "failure_lessons": [],
                "distilled_principles": [],
                "network_reputation": 0.0,
                "source_agent": "",
                "interaction_mode": "passive",
                "version": "0.1.0",
            },
        }

    # ------------------------------------------------------------------
    # Explicit teaching
    # ------------------------------------------------------------------

    async def teach(
        self,
        name: str,
        description: str,
        system_prompt: str,
        triggers: list[str],
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create a skill definition from explicit user input.

        No LLM call needed — the user provides all the pieces.
        Returns the definition dict.
        """
        clean_name = re.sub(r'[^a-z0-9_]', '_', name.lower().strip())

        defn = {
            "type": "dynamic",
            "system_prompt": system_prompt,
            "metadata": {
                "skill_id": clean_name[:12],
                "name": clean_name,
                "category": "user_defined",
                "description": description,
                "tags": tags or [],
                "trigger_conditions": triggers,
                "trust_level": "auto",
                "origin": "user_defined",
                "status": "active",
                "utility_score": 0.5,
                "total_executions": 0,
                "success_count": 0,
                "failure_count": 0,
                "created_at": time.time(),
                "last_used_at": time.time(),
                "last_reflected_at": 0.0,
                "failure_lessons": [],
                "distilled_principles": [],
                "network_reputation": 0.0,
                "source_agent": "",
                "interaction_mode": "passive",
                "version": "0.1.0",
            },
        }

        if self._store:
            self._store.save_skill_definition(clean_name, defn)

        await self._bus.emit_async(
            "skill.taught",
            {"skill_name": clean_name, "description": description},
            source="learner",
        )

        return defn
