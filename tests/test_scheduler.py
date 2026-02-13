"""Tests for the background scheduler."""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolvagent.core.agent import Agent
from evolvagent.core.config import Settings, AgentConfig, SchedulerConfig, EvolutionConfig
from evolvagent.core.scheduler import ResourceSnapshot
from evolvagent.core.skill import (
    BaseSkill, SkillMetadata, SkillOrigin, SkillResult, SkillStatus,
)


def run(coro):
    return asyncio.run(coro)


def _tmp_settings(**overrides):
    tmpdir = tempfile.mkdtemp()
    kw = {"agent": AgentConfig(data_dir=tmpdir)}
    kw.update(overrides)
    return Settings(**kw)


class DummySkill(BaseSkill):
    async def execute(self, context):
        return SkillResult(success=True, output="ok")


# ===================================================================
# ResourceSnapshot Tests
# ===================================================================


class TestResourceSnapshot(unittest.TestCase):

    def test_capture_returns_valid_values(self):
        snap = ResourceSnapshot.capture()
        self.assertGreaterEqual(snap.cpu_percent, 0.0)
        self.assertLessEqual(snap.cpu_percent, 100.0)
        self.assertGreaterEqual(snap.memory_percent, 0.0)
        self.assertLessEqual(snap.memory_percent, 100.0)
        self.assertGreater(snap.timestamp, 0)


# ===================================================================
# Scheduler Lifecycle Tests
# ===================================================================


class TestSchedulerLifecycle(unittest.TestCase):

    def test_scheduler_starts_with_agent(self):
        """Agent.start() should create and start a scheduler."""
        agent = Agent(settings=_tmp_settings())

        async def _test():
            await agent.start()
            self.assertIsNotNone(agent._scheduler)
            self.assertTrue(agent._scheduler.is_running)
            await agent.shutdown()

        run(_test())

    def test_scheduler_stops_on_shutdown(self):
        """Agent.shutdown() should stop the scheduler."""
        agent = Agent(settings=_tmp_settings())

        async def _test():
            await agent.start()
            scheduler = agent._scheduler
            await agent.shutdown()
            self.assertFalse(scheduler.is_running)
            self.assertIsNone(agent._scheduler)

        run(_test())

    def test_scheduler_stop_idempotent(self):
        """Calling stop() multiple times should not raise."""
        agent = Agent(settings=_tmp_settings())

        async def _test():
            await agent.start()
            scheduler = agent._scheduler
            await scheduler.stop()
            await scheduler.stop()  # Should not raise
            await agent.shutdown()

        run(_test())

    def test_scheduler_completes_cycle(self):
        """Scheduler should complete at least one cycle with short interval."""
        settings = _tmp_settings(
            scheduler=SchedulerConfig(
                idle_check_interval=1,
                min_idle_for_reflection=0,
                cpu_threshold_percent=100,
                memory_threshold_percent=100,
            )
        )
        agent = Agent(settings=settings)
        events = []

        async def _test():
            await agent.start()
            agent.last_active_at = time.time() - 10  # make idle check pass
            agent.bus.on("scheduler.maintenance_completed",
                         lambda e: events.append(e.data))
            # Wait for at least one cycle
            await asyncio.sleep(3)
            await agent.shutdown()

        run(_test())
        self.assertGreaterEqual(len(events), 1)


# ===================================================================
# Decay Tests
# ===================================================================


class TestDecay(unittest.TestCase):

    def test_stale_skill_decayed(self):
        """Skill unused for several days should have utility decayed."""
        settings = _tmp_settings(
            evolution=EvolutionConfig(decay_factor=0.95)
        )
        agent = Agent(settings=settings)

        async def _test():
            await agent.start()
            skill = DummySkill(SkillMetadata(
                name="old_skill",
                utility_score=0.8,
                last_used_at=time.time() - 5 * 86400,  # 5 days ago
            ))
            agent.register_skill(skill)

            scheduler = agent._scheduler
            decayed = scheduler._apply_decay()
            self.assertEqual(decayed, 1)
            # utility should be < 0.8
            self.assertLess(skill.metadata.utility_score, 0.8)
            await agent.shutdown()

        run(_test())

    def test_recently_used_skill_not_decayed(self):
        """Skill used recently should not be decayed."""
        settings = _tmp_settings()
        agent = Agent(settings=settings)

        async def _test():
            await agent.start()
            skill = DummySkill(SkillMetadata(
                name="fresh_skill",
                utility_score=0.8,
                last_used_at=time.time(),
            ))
            agent.register_skill(skill)

            scheduler = agent._scheduler
            decayed = scheduler._apply_decay()
            self.assertEqual(decayed, 0)
            self.assertAlmostEqual(skill.metadata.utility_score, 0.8)
            await agent.shutdown()

        run(_test())

    def test_decay_persists(self):
        """After decay, skills should be persistable."""
        settings = _tmp_settings()
        agent = Agent(settings=settings)

        async def _test():
            await agent.start()
            skill = DummySkill(SkillMetadata(
                name="persist_decay",
                utility_score=0.8,
                last_used_at=time.time() - 10 * 86400,
            ))
            agent.register_skill(skill)

            agent._scheduler._apply_decay()
            # Save to store
            if agent._store:
                agent._store.save(skill.metadata)
                loaded = agent._store.load("persist_decay")
                self.assertIsNotNone(loaded)
                self.assertLess(loaded.utility_score, 0.8)
            await agent.shutdown()

        run(_test())


# ===================================================================
# Archival Tests
# ===================================================================


class TestArchival(unittest.TestCase):

    def test_stale_low_utility_archived(self):
        """Stale low-utility non-builtin skill should be archived."""
        settings = _tmp_settings(
            evolution=EvolutionConfig(
                archive_after_days=30,
                archive_threshold=0.2,
            )
        )
        agent = Agent(settings=settings)

        async def _test():
            await agent.start()
            skill = DummySkill(SkillMetadata(
                name="stale_skill",
                utility_score=0.1,
                last_used_at=time.time() - 60 * 86400,  # 60 days ago
                origin=SkillOrigin.LEARNED,
            ))
            agent.register_skill(skill)

            archived = agent._scheduler._archive_stale_skills()
            self.assertEqual(archived, ["stale_skill"])
            self.assertEqual(skill.metadata.status, SkillStatus.ARCHIVED)
            await agent.shutdown()

        run(_test())

    def test_builtin_never_archived(self):
        """BUILTIN skills should never be archived."""
        settings = _tmp_settings(
            evolution=EvolutionConfig(
                archive_after_days=1,
                archive_threshold=1.0,  # aggressive threshold
            )
        )
        agent = Agent(settings=settings)

        async def _test():
            await agent.start()
            skill = DummySkill(SkillMetadata(
                name="builtin_skill",
                utility_score=0.01,
                last_used_at=time.time() - 365 * 86400,
                origin=SkillOrigin.BUILTIN,
            ))
            agent.register_skill(skill)

            archived = agent._scheduler._archive_stale_skills()
            self.assertEqual(archived, [])
            self.assertEqual(skill.metadata.status, SkillStatus.ACTIVE)
            await agent.shutdown()

        run(_test())

    def test_high_utility_not_archived(self):
        """High-utility skill should not be archived even if old."""
        settings = _tmp_settings(
            evolution=EvolutionConfig(
                archive_after_days=30,
                archive_threshold=0.2,
            )
        )
        agent = Agent(settings=settings)

        async def _test():
            await agent.start()
            skill = DummySkill(SkillMetadata(
                name="good_skill",
                utility_score=0.9,
                last_used_at=time.time() - 60 * 86400,
                origin=SkillOrigin.LEARNED,
            ))
            agent.register_skill(skill)

            archived = agent._scheduler._archive_stale_skills()
            self.assertEqual(archived, [])
            self.assertEqual(skill.metadata.status, SkillStatus.ACTIVE)
            await agent.shutdown()

        run(_test())

    def test_archived_emits_event(self):
        """Archival should emit a skill.archived event."""
        settings = _tmp_settings(
            evolution=EvolutionConfig(
                archive_after_days=30,
                archive_threshold=0.2,
            )
        )
        agent = Agent(settings=settings)
        events = []

        async def _test():
            await agent.start()
            agent.bus.on("skill.archived", lambda e: events.append(e.data))
            skill = DummySkill(SkillMetadata(
                name="event_skill",
                utility_score=0.1,
                last_used_at=time.time() - 60 * 86400,
                origin=SkillOrigin.LEARNED,
            ))
            agent.register_skill(skill)

            agent._scheduler._archive_stale_skills()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["skill_name"], "event_skill")
            await agent.shutdown()

        run(_test())

    def test_archived_not_in_active_skills(self):
        """Archived skill should not appear in active_skills."""
        settings = _tmp_settings(
            evolution=EvolutionConfig(
                archive_after_days=30,
                archive_threshold=0.2,
            )
        )
        agent = Agent(settings=settings)

        async def _test():
            await agent.start()
            skill = DummySkill(SkillMetadata(
                name="to_archive",
                utility_score=0.1,
                last_used_at=time.time() - 60 * 86400,
                origin=SkillOrigin.LEARNED,
            ))
            agent.register_skill(skill)
            self.assertIn(skill, agent.active_skills)

            agent._scheduler._archive_stale_skills()
            self.assertNotIn(skill, agent.active_skills)
            await agent.shutdown()

        run(_test())


if __name__ == "__main__":
    unittest.main(verbosity=2)
