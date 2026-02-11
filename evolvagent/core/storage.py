"""
SQLite-based persistence for EvolvAgent.

Stores SkillMetadata as JSON blobs with indexed columns for efficient queries.
Pure standard library — uses sqlite3 and json only.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from .skill import SkillMetadata, SkillStatus

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    skill_id   TEXT PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    category   TEXT DEFAULT 'general',
    status     TEXT DEFAULT 'active',
    utility    REAL DEFAULT 0.5,
    data       TEXT NOT NULL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS agent_stats (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activity_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  REAL NOT NULL,
    workspace  TEXT DEFAULT '',
    action     TEXT NOT NULL,
    query      TEXT DEFAULT '',
    skill_used TEXT DEFAULT '',
    success    INTEGER DEFAULT 1
);
"""


class SkillStore:
    """
    SQLite-backed storage for SkillMetadata.

    Uses JSON blob storage (via SkillMetadata.to_dict/from_dict) with
    indexed columns for name, status, and utility for fast lookups.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        logger.info("SkillStore opened: %s", db_path)

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Skill CRUD
    # ------------------------------------------------------------------

    def save(self, metadata: SkillMetadata) -> None:
        """Insert or update a Skill."""
        data_json = json.dumps(metadata.to_dict(), ensure_ascii=False)
        self._conn.execute(
            """
            INSERT INTO skills (skill_id, name, category, status, utility, data, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                skill_id=excluded.skill_id,
                category=excluded.category,
                status=excluded.status,
                utility=excluded.utility,
                data=excluded.data,
                updated_at=excluded.updated_at
            """,
            (
                metadata.skill_id,
                metadata.name,
                metadata.category,
                metadata.status.value,
                metadata.utility_score,
                data_json,
                time.time(),
            ),
        )
        self._conn.commit()

    def load(self, name: str) -> SkillMetadata | None:
        """Load a Skill by name. Returns None if not found."""
        row = self._conn.execute(
            "SELECT data FROM skills WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return SkillMetadata.from_dict(json.loads(row["data"]))

    def load_all(self, status: str | None = None) -> list[SkillMetadata]:
        """Load all Skills, optionally filtered by status."""
        if status:
            rows = self._conn.execute(
                "SELECT data FROM skills WHERE status = ? ORDER BY utility DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data FROM skills ORDER BY utility DESC"
            ).fetchall()
        return [SkillMetadata.from_dict(json.loads(r["data"])) for r in rows]

    def delete(self, name: str) -> bool:
        """Delete a Skill by name. Returns True if deleted."""
        cursor = self._conn.execute("DELETE FROM skills WHERE name = ?", (name,))
        self._conn.commit()
        return cursor.rowcount > 0

    def count(self, status: str | None = None) -> int:
        """Count Skills, optionally filtered by status."""
        if status:
            row = self._conn.execute(
                "SELECT COUNT(*) as n FROM skills WHERE status = ?", (status,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) as n FROM skills").fetchone()
        return row["n"]

    # ------------------------------------------------------------------
    # Agent stats
    # ------------------------------------------------------------------

    def save_stats(self, stats: dict[str, Any]) -> None:
        """Save agent statistics as key-value pairs."""
        for key, value in stats.items():
            self._conn.execute(
                "INSERT INTO agent_stats (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )
        self._conn.commit()

    def load_stats(self) -> dict[str, Any]:
        """Load all agent statistics."""
        rows = self._conn.execute("SELECT key, value FROM agent_stats").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    # ------------------------------------------------------------------
    # Activity log
    # ------------------------------------------------------------------

    def log_activity(
        self,
        workspace: str,
        action: str,
        query: str = "",
        skill_used: str = "",
        success: bool = True,
    ) -> None:
        """Log an agent interaction."""
        self._conn.execute(
            "INSERT INTO activity_log (timestamp, workspace, action, query, skill_used, success) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), workspace, action, query[:500], skill_used, int(success)),
        )
        self._conn.commit()

    def recent_activity(
        self, workspace: str = "", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Load recent activity entries, optionally filtered by workspace."""
        if workspace:
            rows = self._conn.execute(
                "SELECT * FROM activity_log WHERE workspace = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (workspace, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
