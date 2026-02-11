"""Tests for SkillStore (SQLite persistence layer)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evolvagent.core.storage import SkillStore
from evolvagent.core.skill import SkillMetadata, SkillOrigin, SkillStatus, TrustLevel


class TestSkillStore(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self._tmpdir) / "test.db"
        self.store = SkillStore(self.db_path)

    def tearDown(self):
        self.store.close()

    def _make_meta(self, name="test_skill", **kw) -> SkillMetadata:
        defaults = {"name": name, "description": "A test skill"}
        defaults.update(kw)
        return SkillMetadata(**defaults)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def test_save_and_load(self):
        meta = self._make_meta("echo")
        self.store.save(meta)
        loaded = self.store.load("echo")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "echo")
        self.assertEqual(loaded.skill_id, meta.skill_id)

    def test_load_nonexistent(self):
        self.assertIsNone(self.store.load("nope"))

    def test_save_updates_existing(self):
        meta = self._make_meta("echo")
        self.store.save(meta)
        meta.utility_score = 0.9
        meta.success_count = 5
        self.store.save(meta)
        loaded = self.store.load("echo")
        self.assertAlmostEqual(loaded.utility_score, 0.9)
        self.assertEqual(loaded.success_count, 5)

    def test_load_all(self):
        self.store.save(self._make_meta("a"))
        self.store.save(self._make_meta("b"))
        self.store.save(self._make_meta("c", status=SkillStatus.ARCHIVED))
        all_skills = self.store.load_all()
        self.assertEqual(len(all_skills), 3)

    def test_load_all_filtered(self):
        self.store.save(self._make_meta("a"))
        self.store.save(self._make_meta("b", status=SkillStatus.ARCHIVED))
        active = self.store.load_all(status="active")
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].name, "a")

    def test_delete(self):
        self.store.save(self._make_meta("echo"))
        self.assertTrue(self.store.delete("echo"))
        self.assertIsNone(self.store.load("echo"))

    def test_delete_nonexistent(self):
        self.assertFalse(self.store.delete("nope"))

    def test_count(self):
        self.store.save(self._make_meta("a"))
        self.store.save(self._make_meta("b"))
        self.assertEqual(self.store.count(), 2)
        self.assertEqual(self.store.count(status="active"), 2)

    # ------------------------------------------------------------------
    # Data integrity
    # ------------------------------------------------------------------

    def test_roundtrip_preserves_metadata(self):
        meta = self._make_meta(
            "roundtrip",
            tags=["t1", "t2"],
            trigger_conditions=["hello"],
            trust_level=TrustLevel.SUGGEST,
            origin=SkillOrigin.LEARNED,
            utility_score=0.75,
            success_count=10,
            failure_count=2,
            total_executions=12,
        )
        self.store.save(meta)
        loaded = self.store.load("roundtrip")
        self.assertEqual(loaded.tags, ["t1", "t2"])
        self.assertEqual(loaded.trigger_conditions, ["hello"])
        self.assertEqual(loaded.trust_level, TrustLevel.SUGGEST)
        self.assertEqual(loaded.origin, SkillOrigin.LEARNED)
        self.assertAlmostEqual(loaded.utility_score, 0.75)
        self.assertEqual(loaded.success_count, 10)
        self.assertEqual(loaded.total_executions, 12)

    def test_db_file_created(self):
        self.assertTrue(self.db_path.exists())

    # ------------------------------------------------------------------
    # Agent stats
    # ------------------------------------------------------------------

    def test_save_and_load_stats(self):
        self.store.save_stats({"total_requests": 42, "successful_tasks": 30})
        stats = self.store.load_stats()
        self.assertEqual(stats["total_requests"], 42)
        self.assertEqual(stats["successful_tasks"], 30)

    def test_stats_update(self):
        self.store.save_stats({"total_requests": 1})
        self.store.save_stats({"total_requests": 5})
        stats = self.store.load_stats()
        self.assertEqual(stats["total_requests"], 5)

    def test_stats_empty(self):
        stats = self.store.load_stats()
        self.assertEqual(stats, {})

    # ------------------------------------------------------------------
    # Reopening DB
    # ------------------------------------------------------------------

    def test_persistence_across_connections(self):
        self.store.save(self._make_meta("persistent"))
        self.store.save_stats({"count": 99})
        self.store.close()

        store2 = SkillStore(self.db_path)
        loaded = store2.load("persistent")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "persistent")
        stats = store2.load_stats()
        self.assertEqual(stats["count"], 99)
        store2.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
