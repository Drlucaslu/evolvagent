"""
Background scheduler for EvolvAgent.

Runs as an asyncio.Task within the Agent's event loop, periodically checking
for idle opportunities to perform maintenance: reflection, Ebbinghaus decay,
and skill archival.

Usage:
    scheduler = AgentScheduler(agent)
    scheduler.start()       # creates background task
    await scheduler.stop()  # graceful shutdown
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resource monitoring
# ---------------------------------------------------------------------------


@dataclass
class ResourceSnapshot:
    """Point-in-time system resource reading."""

    cpu_percent: float
    memory_percent: float
    timestamp: float

    @staticmethod
    def capture() -> ResourceSnapshot:
        """Capture current CPU and memory usage. Blocking (~0.5s for CPU)."""
        import psutil

        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory().percent
        return ResourceSnapshot(
            cpu_percent=cpu,
            memory_percent=mem,
            timestamp=time.time(),
        )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class AgentScheduler:
    """
    Background maintenance scheduler.

    Periodically checks if the agent is idle and system resources are
    available, then runs reflection, decay, and archival.
    """

    def __init__(self, agent: Agent):
        self._agent = agent
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._cycle_count: int = 0

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the background scheduler task."""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        self._stop_event.clear()
        self._cycle_count = 0
        self._task = asyncio.get_event_loop().create_task(
            self._run_loop(), name="agent-scheduler"
        )
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Signal the scheduler to stop and wait for it to finish."""
        if not self.is_running:
            return
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning("Scheduler did not stop in time, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None
        logger.info("Scheduler stopped after %d cycles", self._cycle_count)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main scheduler loop — runs until stop_event is set."""
        cfg = self._agent.settings.scheduler
        interval = cfg.idle_check_interval

        while not self._stop_event.is_set():
            # Wait for the next interval, but break early if stopped
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=interval
                )
                # If we get here, stop was requested
                break
            except asyncio.TimeoutError:
                # Normal timeout — time to check
                pass

            try:
                await self._check_and_act()
            except Exception:
                logger.exception("Scheduler maintenance cycle failed")

    async def _check_and_act(self) -> None:
        """Single maintenance check — run if conditions are met."""
        from .agent import AgentState

        agent = self._agent
        cfg = agent.settings.scheduler

        # 1. Agent must be idle
        if agent.state != AgentState.IDLE:
            return

        # 2. Check system resources (in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        try:
            snapshot = await loop.run_in_executor(None, ResourceSnapshot.capture)
        except Exception as e:
            logger.warning("Failed to capture resource snapshot: %s", e)
            return

        if snapshot.cpu_percent >= cfg.cpu_threshold_percent:
            logger.debug(
                "Skipping maintenance: CPU %.1f%% >= %d%%",
                snapshot.cpu_percent,
                cfg.cpu_threshold_percent,
            )
            return

        if snapshot.memory_percent >= cfg.memory_threshold_percent:
            logger.debug(
                "Skipping maintenance: Memory %.1f%% >= %d%%",
                snapshot.memory_percent,
                cfg.memory_threshold_percent,
            )
            return

        # 3. Agent must have been idle long enough
        idle_seconds = time.time() - agent.last_active_at
        if idle_seconds < cfg.min_idle_for_reflection:
            return

        # All conditions met — run maintenance
        self._cycle_count += 1
        await self._agent.bus.emit_async(
            "scheduler.maintenance_started",
            {
                "cycle": self._cycle_count,
                "cpu_percent": snapshot.cpu_percent,
                "memory_percent": snapshot.memory_percent,
                "idle_seconds": idle_seconds,
            },
            source="scheduler",
        )

        logger.info(
            "Maintenance cycle %d: CPU=%.1f%% Mem=%.1f%% idle=%.0fs",
            self._cycle_count,
            snapshot.cpu_percent,
            snapshot.memory_percent,
            idle_seconds,
        )

        # Phase 1: Reflection
        await agent.enter_reflection()

        # Phase 2: Decay
        decayed_count = self._apply_decay()

        # Phase 3: Archive
        archived_skills = self._archive_stale_skills()

        # Persist all changes
        if agent._store:
            for skill in agent._skills.values():
                agent._store.save(skill.metadata)

        await self._agent.bus.emit_async(
            "scheduler.maintenance_completed",
            {
                "cycle": self._cycle_count,
                "decayed_count": decayed_count,
                "archived_count": len(archived_skills),
                "archived_skills": archived_skills,
            },
            source="scheduler",
        )

        logger.info(
            "Maintenance cycle %d complete: decayed=%d archived=%d",
            self._cycle_count,
            decayed_count,
            len(archived_skills),
        )

    # ------------------------------------------------------------------
    # Maintenance actions
    # ------------------------------------------------------------------

    def _apply_decay(self) -> int:
        """Apply Ebbinghaus decay to all active skills. Returns count of decayed."""
        from .skill import SkillStatus

        decay_factor = self._agent.settings.evolution.decay_factor
        decayed = 0

        for skill in self._agent._skills.values():
            meta = skill.metadata
            if meta.status != SkillStatus.ACTIVE:
                continue
            old_utility = meta.utility_score
            meta.apply_decay(decay_factor)
            if meta.utility_score < old_utility:
                decayed += 1

        return decayed

    def _archive_stale_skills(self) -> list[str]:
        """Archive skills that are stale and not builtin. Returns archived names."""
        from .skill import SkillOrigin, SkillStatus

        evo = self._agent.settings.evolution
        archived = []

        for skill in list(self._agent._skills.values()):
            meta = skill.metadata
            if meta.status != SkillStatus.ACTIVE:
                continue
            if meta.origin == SkillOrigin.BUILTIN:
                continue
            if (
                meta.days_since_last_use > evo.archive_after_days
                and meta.utility_score < evo.archive_threshold
            ):
                meta.status = SkillStatus.ARCHIVED
                archived.append(meta.name)
                self._agent.bus.emit(
                    "skill.archived",
                    {
                        "skill_name": meta.name,
                        "utility_score": meta.utility_score,
                        "days_since_last_use": meta.days_since_last_use,
                    },
                    source="scheduler",
                )
                logger.info(
                    "Skill '%s' archived: utility=%.2f days_idle=%.0f",
                    meta.name,
                    meta.utility_score,
                    meta.days_since_last_use,
                )

        return archived
