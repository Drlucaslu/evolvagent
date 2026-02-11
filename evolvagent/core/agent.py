"""
Agent: the central orchestrator of EvolvAgent.

Manages the Agent lifecycle (init -> idle -> active -> reflecting -> shutdown),
coordinates all subsystems through the event bus, and enforces the
"human-first" resource policy.

State machine:
    INITIALIZING -> IDLE <-> ACTIVE
                     |        |
                  REFLECTING   |
                     |         |
                  IDLE <-------+
                     |
                  SHUTTING_DOWN
"""

from __future__ import annotations

import logging
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from .config import Settings, get_settings
from .events import EventBus
from .learner import DynamicSkill, LearnResult, SkillLearner
from .llm import LLMClient
from .reflection import ReflectionEngine, ReflectionResult
from .scheduler import AgentScheduler
from .skill import BaseSkill, SkillMetadata, SkillStatus, TrustLevel
from .storage import SkillStore

# Callback type for SUGGEST mode: (skill_name, preview_text) -> approved?
ConfirmCallback = Callable[[str, str], Coroutine[Any, Any, bool]]

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Agent lifecycle states."""
    INITIALIZING = "initializing"
    IDLE = "idle"              # Waiting for user input or idle tasks
    ACTIVE = "active"          # Processing a user request
    REFLECTING = "reflecting"  # Offline distillation phase
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


# Valid state transitions
_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.INITIALIZING: {AgentState.IDLE},
    AgentState.IDLE: {AgentState.ACTIVE, AgentState.REFLECTING, AgentState.SHUTTING_DOWN},
    AgentState.ACTIVE: {AgentState.IDLE, AgentState.SHUTTING_DOWN},
    AgentState.REFLECTING: {AgentState.IDLE, AgentState.SHUTTING_DOWN},
    AgentState.SHUTTING_DOWN: {AgentState.STOPPED},
    AgentState.STOPPED: set(),
}


class Agent:
    """
    The EvolvAgent core.

    Coordinates task engine, knowledge base, reflection module, learner,
    and scheduler through the event bus.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.state = AgentState.INITIALIZING
        self.bus = EventBus()

        # Metadata
        self.name = self.settings.agent.name
        self.agent_id = f"{self.name}-{uuid.uuid4().hex[:8]}"
        self.started_at: float = 0
        self.last_active_at: float = 0
        self.workspace: str = ""

        # Skill registry (in-memory, synced with SQLite store)
        self._skills: dict[str, BaseSkill] = {}
        self._store: SkillStore | None = None

        # Scheduler + reflection + learning
        self._scheduler: AgentScheduler | None = None
        self._llm: LLMClient | None = None
        self._learner: SkillLearner | None = None
        self._last_reflection_result: ReflectionResult | None = None
        self._last_learn_result: LearnResult | None = None

        # Statistics
        self.stats = AgentStats()

        logger.info("Agent '%s' (%s) created", self.name, self.agent_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize all subsystems and transition to IDLE."""
        logger.info("Starting agent '%s'...", self.name)

        # Ensure data directory exists
        data_dir = self.settings.agent.resolved_data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize persistent storage
        db_path = data_dir / self.settings.knowledge.db_name
        self._store = SkillStore(db_path)

        # Load persisted agent stats
        saved_stats = self._store.load_stats()
        if saved_stats:
            self.stats.total_requests = saved_stats.get("total_requests", 0)
            self.stats.successful_tasks = saved_stats.get("successful_tasks", 0)
            self.stats.failed_tasks = saved_stats.get("failed_tasks", 0)
            self.stats.no_skill_found = saved_stats.get("no_skill_found", 0)

        # Initialize learner
        self._learner = SkillLearner(
            llm=self._llm, bus=self.bus, store=self._store,
        )

        # Emit initialization event
        await self.bus.emit_async("agent.starting", {
            "agent_id": self.agent_id,
            "data_dir": str(data_dir),
            "settings": self.settings,
        }, source="agent")

        self.started_at = time.time()
        self._transition(AgentState.IDLE)

        # Load persisted dynamic skills
        self._load_dynamic_skills()

        # Start background scheduler
        self._scheduler = AgentScheduler(self)
        self._scheduler.start()

        await self.bus.emit_async("agent.started", {
            "agent_id": self.agent_id,
        }, source="agent")

        logger.info("Agent '%s' started successfully.", self.name)

    async def shutdown(self) -> None:
        """Gracefully shut down all subsystems."""
        logger.info("Shutting down agent '%s'...", self.name)

        # Stop scheduler before state transition
        if self._scheduler:
            await self._scheduler.stop()
            self._scheduler = None

        self._transition(AgentState.SHUTTING_DOWN)

        # Persist agent stats before shutdown
        if self._store:
            self._store.save_stats({
                "total_requests": self.stats.total_requests,
                "successful_tasks": self.stats.successful_tasks,
                "failed_tasks": self.stats.failed_tasks,
                "no_skill_found": self.stats.no_skill_found,
            })
            self._store.close()
            self._store = None

        await self.bus.emit_async("agent.shutting_down", {
            "agent_id": self.agent_id,
            "uptime_seconds": time.time() - self.started_at,
        }, source="agent")

        self._transition(AgentState.STOPPED)
        logger.info("Agent '%s' stopped.", self.name)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _transition(self, new_state: AgentState) -> None:
        """Transition to a new state, validating the transition is legal."""
        allowed = _TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidTransition(
                f"Cannot transition from {self.state.value} to {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        old_state = self.state
        self.state = new_state
        self.bus.emit("agent.state_changed", {
            "old_state": old_state.value,
            "new_state": new_state.value,
        }, source="agent")
        logger.debug("State: %s -> %s", old_state.value, new_state.value)

    @property
    def is_idle(self) -> bool:
        return self.state == AgentState.IDLE

    @property
    def is_active(self) -> bool:
        return self.state == AgentState.ACTIVE

    def set_llm(self, llm: LLMClient) -> None:
        """Set the LLM client for reflection and learning. Call before start()."""
        self._llm = llm

    @property
    def uptime(self) -> float:
        if self.started_at == 0:
            return 0
        return time.time() - self.started_at

    # ------------------------------------------------------------------
    # Skill management
    # ------------------------------------------------------------------

    def register_skill(self, skill: BaseSkill, persist: bool = True) -> None:
        """Register a Skill with the Agent."""
        name = skill.metadata.name
        self._skills[name] = skill
        if persist and self._store:
            self._store.save(skill.metadata)
        self.bus.emit("skill.registered", {
            "skill_name": name,
            "skill_id": skill.metadata.skill_id,
        }, source="agent")
        logger.info("Skill registered: %s", name)

    def unregister_skill(self, name: str) -> None:
        """Unregister a Skill."""
        if name in self._skills:
            del self._skills[name]
            if self._store:
                self._store.delete(name)
                self._store.delete_skill_definition(name)
            self.bus.emit("skill.unregistered", {"skill_name": name}, source="agent")

    def get_skill(self, name: str) -> BaseSkill | None:
        """Get a registered Skill by name."""
        return self._skills.get(name)

    @property
    def active_skills(self) -> list[BaseSkill]:
        """All skills with ACTIVE status."""
        return [
            s for s in self._skills.values()
            if s.metadata.status == SkillStatus.ACTIVE
        ]

    @property
    def skill_count(self) -> int:
        return len(self._skills)

    def find_skill_for_intent(self, intent: str) -> list[tuple[BaseSkill, float]]:
        """
        Find Skills that can handle an intent.

        Returns list of (skill, confidence) sorted by confidence * utility.
        """
        candidates = []
        for skill in self.active_skills:
            confidence = skill.can_handle(intent)
            if confidence > 0:
                score = confidence * 0.6 + skill.metadata.utility_score * 0.4
                candidates.append((skill, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _load_dynamic_skills(self) -> None:
        """Reload persisted DynamicSkills from the store."""
        if not self._store or not self._llm:
            return
        definitions = self._store.load_all_skill_definitions()
        for defn in definitions:
            name = defn.get("metadata", {}).get("name", "")
            if name and name not in self._skills:
                try:
                    skill = DynamicSkill.from_definition(defn, self._llm)
                    # Restore persisted metadata if available
                    saved = self._store.load(name)
                    if saved:
                        skill.metadata = saved
                    self._skills[name] = skill
                    logger.info("Loaded dynamic skill: %s", name)
                except Exception as e:
                    logger.warning("Failed to load dynamic skill '%s': %s", name, e)

    # ------------------------------------------------------------------
    # Task handling
    # ------------------------------------------------------------------

    async def handle_request(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
        confirm_callback: ConfirmCallback | None = None,
    ) -> str:
        """
        Handle a user request, enforcing trust levels.

        Trust behavior:
          OBSERVE  -- preview only, no execution
          SUGGEST  -- preview + ask user confirmation, execute if approved
          AUTO     -- execute directly
        """
        self._transition(AgentState.ACTIVE)
        self.last_active_at = time.time()
        self.stats.total_requests += 1

        await self.bus.emit_async("agent.request_received", {
            "input": user_input,
            "context": context or {},
        }, source="agent")

        try:
            # Find best Skill
            candidates = self.find_skill_for_intent(user_input)
            if not candidates:
                self.stats.no_skill_found += 1
                # Record miss for learner
                if self._learner:
                    self._learner.record(user_input, "", False)
                return "I don't have a Skill that can handle this request yet."

            skill, score = candidates[0]
            await self.bus.emit_async("skill.selected", {
                "skill_name": skill.metadata.name,
                "confidence": score,
                "intent": user_input,
            }, source="agent")

            exec_context = {"user_input": user_input, **(context or {})}
            trust = skill.metadata.trust_level

            # --- Trust level enforcement ---
            if trust == TrustLevel.OBSERVE:
                preview = await skill.preview(exec_context)
                return f"[OBSERVE] {preview}"

            if trust == TrustLevel.SUGGEST:
                preview = await skill.preview(exec_context)
                if confirm_callback:
                    approved = await confirm_callback(skill.metadata.name, preview)
                    if not approved:
                        return "Action cancelled by user."
                    # Fall through to execute
                else:
                    return f"[SUGGEST] {preview}"

            # --- AUTO or confirmed SUGGEST: execute ---
            result = await skill.execute(exec_context)

            # Record execution and update utility
            skill.metadata.record_execution(
                success=result.success,
                reward=result.effective_reward,
                learning_rate=self.settings.evolution.learning_rate,
            )

            # Auto-promote trust level after sustained success
            if result.success:
                threshold = self.settings.trust.promote_threshold
                if (skill.metadata.success_count >= threshold
                        and skill.metadata.trust_level != TrustLevel.AUTO):
                    old_trust = skill.metadata.trust_level
                    skill.metadata.promote_trust()
                    logger.info(
                        "Skill '%s' trust promoted: %s -> %s (after %d successes)",
                        skill.metadata.name, old_trust.value,
                        skill.metadata.trust_level.value, skill.metadata.success_count,
                    )
                    await self.bus.emit_async("skill.trust_promoted", {
                        "skill_name": skill.metadata.name,
                        "old_trust": old_trust.value,
                        "new_trust": skill.metadata.trust_level.value,
                    }, source="agent")

            # Persist updated metadata
            if self._store:
                self._store.save(skill.metadata)

            await self.bus.emit_async("skill.executed", {
                "skill_name": skill.metadata.name,
                "success": result.success,
                "utility_score": skill.metadata.utility_score,
                "execution_time_ms": result.execution_time_ms,
            }, source="agent")

            if result.success:
                self.stats.successful_tasks += 1
            else:
                self.stats.failed_tasks += 1

            # Record for learner
            if self._learner:
                self._learner.record(
                    user_input, skill.metadata.name, result.success,
                )

            # Log activity
            if self._store:
                self._store.log_activity(
                    workspace=self.workspace or "",
                    action="request",
                    query=user_input[:500],
                    skill_used=skill.metadata.name,
                    success=result.success,
                )

            return result.output if result.success else f"Task failed: {result.error}"

        finally:
            self._transition(AgentState.IDLE)

    # ------------------------------------------------------------------
    # Reflection + Learning
    # ------------------------------------------------------------------

    async def enter_reflection(self) -> ReflectionResult | None:
        """Enter offline reflection mode -- LLM-driven principle extraction + learning."""
        if self.state != AgentState.IDLE:
            logger.warning("Cannot reflect: not in IDLE state (current: %s)", self.state.value)
            return None

        self._transition(AgentState.REFLECTING)

        await self.bus.emit_async("agent.reflection_started", {
            "agent_id": self.agent_id,
        }, source="agent")

        try:
            # Phase 1: Principle extraction
            engine = ReflectionEngine(
                llm=self._llm,
                bus=self.bus,
                store=self._store,
            )
            result = await engine.reflect(self._skills)
            self._last_reflection_result = result

            # Phase 2: Skill learning
            if self._learner:
                learn_result = await self._learner.analyze_and_learn(self._skills)
                self._last_learn_result = learn_result
                # Hydrate any newly created skills
                for name in learn_result.created_skill_names:
                    self._hydrate_learned_skill(name)

            logger.info(
                "Reflection complete: analyzed=%d updated=%d principles=%d",
                result.skills_analyzed,
                result.skills_updated,
                result.principles_extracted,
            )
            return result
        except Exception as e:
            logger.exception("Reflection failed: %s", e)
            return ReflectionResult(errors=[str(e)])
        finally:
            self._transition(AgentState.IDLE)

            await self.bus.emit_async("agent.reflection_completed", {
                "agent_id": self.agent_id,
            }, source="agent")

    async def learn_skill(
        self,
        name: str,
        description: str,
        system_prompt: str,
        triggers: list[str],
        tags: list[str] | None = None,
    ) -> DynamicSkill | None:
        """Teach the agent a new skill explicitly. Returns the created skill."""
        if not self._learner:
            return None

        defn = await self._learner.teach(
            name=name,
            description=description,
            system_prompt=system_prompt,
            triggers=triggers,
            tags=tags,
        )

        return self._hydrate_learned_skill(defn["metadata"]["name"], defn)

    def _hydrate_learned_skill(
        self,
        name: str,
        defn: dict | None = None,
    ) -> DynamicSkill | None:
        """Load a skill definition from store and register it."""
        if not self._llm:
            return None
        if not defn and self._store:
            defn = self._store.load_skill_definition(name)
        if not defn:
            return None
        if name in self._skills:
            return None

        try:
            skill = DynamicSkill.from_definition(defn, self._llm)
            self.register_skill(skill)
            logger.info("Learned skill hydrated and registered: %s", name)
            return skill
        except Exception as e:
            logger.warning("Failed to hydrate learned skill '%s': %s", name, e)
            return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status_dict(self) -> dict[str, Any]:
        """Return a status summary for display."""
        last_ref = None
        if self._last_reflection_result:
            r = self._last_reflection_result
            last_ref = {
                "skills_analyzed": r.skills_analyzed,
                "skills_updated": r.skills_updated,
                "principles_extracted": r.principles_extracted,
                "errors": r.errors,
                "skipped_reason": r.skipped_reason,
            }

        last_learn = None
        if self._last_learn_result:
            lr = self._last_learn_result
            last_learn = {
                "patterns_analyzed": lr.patterns_analyzed,
                "skills_created": lr.skills_created,
                "created_skill_names": lr.created_skill_names,
                "skipped_reason": lr.skipped_reason,
            }

        # Count learned vs builtin
        learned_count = sum(
            1 for s in self._skills.values()
            if isinstance(s, DynamicSkill)
        )

        return {
            "name": self.name,
            "agent_id": self.agent_id,
            "state": self.state.value,
            "uptime_seconds": self.uptime,
            "skill_count": self.skill_count,
            "learned_skill_count": learned_count,
            "active_skills": [s.metadata.name for s in self.active_skills],
            "stats": {
                "total_requests": self.stats.total_requests,
                "successful_tasks": self.stats.successful_tasks,
                "failed_tasks": self.stats.failed_tasks,
                "no_skill_found": self.stats.no_skill_found,
            },
            "scheduler_running": self._scheduler is not None and self._scheduler.is_running,
            "last_reflection": last_ref,
            "last_learning": last_learn,
        }


class AgentStats:
    """Runtime statistics."""
    def __init__(self):
        self.total_requests: int = 0
        self.successful_tasks: int = 0
        self.failed_tasks: int = 0
        self.no_skill_found: int = 0
        self.total_llm_cost_usd: float = 0.0
        self.total_llm_calls: int = 0


class InvalidTransition(Exception):
    """Raised when an invalid state transition is attempted."""
    pass
