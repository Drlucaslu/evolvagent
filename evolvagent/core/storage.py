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

from .skill import SkillMetadata

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 2

# Version 1: core tables (skills, skill_definitions, agent_stats, activity_log)
# Version 2: network_peers + skill_reputation (Phase 4)

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS skills (
    skill_id   TEXT PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    category   TEXT DEFAULT 'general',
    status     TEXT DEFAULT 'active',
    utility    REAL DEFAULT 0.5,
    data       TEXT NOT NULL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS skill_definitions (
    name       TEXT PRIMARY KEY,
    definition TEXT NOT NULL,
    created_at REAL
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

_SCHEMA_V2_MIGRATION = """
CREATE TABLE IF NOT EXISTS network_peers (
    address     TEXT PRIMARY KEY,
    agent_id    TEXT DEFAULT '',
    agent_name  TEXT DEFAULT '',
    last_seen   REAL DEFAULT 0,
    trust_score REAL DEFAULT 0.5
);

CREATE TABLE IF NOT EXISTS skill_reputation (
    skill_name   TEXT NOT NULL,
    content_hash TEXT DEFAULT '',
    source_agent TEXT DEFAULT '',
    local_rating REAL DEFAULT 0.0,
    peer_ratings TEXT DEFAULT '[]',
    imported_at  REAL DEFAULT 0,
    PRIMARY KEY (skill_name, source_agent)
);
"""

# Full schema = V1 + V2 (for new databases)
_SCHEMA = _SCHEMA_V1 + _SCHEMA_V2_MIGRATION


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
        # Ensure schema_version table exists
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        self._conn.commit()

        row = self._conn.execute("SELECT version FROM schema_version").fetchone()
        current_ver = row["version"] if row else 0

        if current_ver == 0:
            # Fresh database — create everything
            self._conn.executescript(_SCHEMA)
            self._conn.execute("INSERT INTO schema_version (version) VALUES (?)",
                               (CURRENT_SCHEMA_VERSION,))
            self._conn.commit()
            logger.debug("Created schema at version %d", CURRENT_SCHEMA_VERSION)
        elif current_ver < CURRENT_SCHEMA_VERSION:
            self._migrate(current_ver, CURRENT_SCHEMA_VERSION)
        # else: already up-to-date

    def _migrate(self, from_ver: int, to_ver: int) -> None:
        """Incrementally migrate schema from from_ver to to_ver."""
        logger.info("Migrating schema from v%d to v%d", from_ver, to_ver)

        if from_ver < 1:
            self._conn.executescript(_SCHEMA_V1)

        if from_ver < 2:
            self._conn.executescript(_SCHEMA_V2_MIGRATION)

        self._conn.execute("UPDATE schema_version SET version = ?", (to_ver,))
        self._conn.commit()
        logger.info("Schema migration complete: v%d", to_ver)

    def schema_version(self) -> int:
        """Return the current schema version."""
        row = self._conn.execute("SELECT version FROM schema_version").fetchone()
        return row["version"] if row else 0

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

    def cleanup_activity_log(self, retention_days: int = 90) -> int:
        """Delete activity log entries older than retention_days. Returns count deleted."""
        cutoff = time.time() - (retention_days * 86400)
        cursor = self._conn.execute(
            "DELETE FROM activity_log WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.info("Cleaned up %d activity log entries older than %d days",
                        deleted, retention_days)
        return deleted

    def vacuum(self) -> None:
        """Optimize and compact the database."""
        self._conn.execute("PRAGMA optimize")
        self._conn.execute("VACUUM")
        logger.debug("Database vacuum complete")

    # ------------------------------------------------------------------
    # Skill definitions (for DynamicSkill persistence)
    # ------------------------------------------------------------------

    def save_skill_definition(self, name: str, definition: dict[str, Any]) -> None:
        """Save a DynamicSkill definition."""
        self._conn.execute(
            "INSERT INTO skill_definitions (name, definition, created_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET definition=excluded.definition",
            (name, json.dumps(definition, ensure_ascii=False), time.time()),
        )
        self._conn.commit()

    def load_skill_definition(self, name: str) -> dict[str, Any] | None:
        """Load a DynamicSkill definition by name."""
        row = self._conn.execute(
            "SELECT definition FROM skill_definitions WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["definition"])

    def load_all_skill_definitions(self) -> list[dict[str, Any]]:
        """Load all DynamicSkill definitions."""
        rows = self._conn.execute(
            "SELECT definition FROM skill_definitions ORDER BY created_at"
        ).fetchall()
        return [json.loads(r["definition"]) for r in rows]

    def delete_skill_definition(self, name: str) -> bool:
        """Delete a DynamicSkill definition."""
        cursor = self._conn.execute(
            "DELETE FROM skill_definitions WHERE name = ?", (name,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Network peers
    # ------------------------------------------------------------------

    def save_peer(
        self,
        address: str,
        agent_id: str = "",
        agent_name: str = "",
        last_seen: float = 0.0,
        trust_score: float = 0.5,
    ) -> None:
        """Save or update a network peer."""
        self._conn.execute(
            "INSERT INTO network_peers (address, agent_id, agent_name, last_seen, trust_score) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(address) DO UPDATE SET "
            "agent_id=excluded.agent_id, agent_name=excluded.agent_name, "
            "last_seen=excluded.last_seen, trust_score=excluded.trust_score",
            (address, agent_id, agent_name, last_seen, trust_score),
        )
        self._conn.commit()

    def load_peers(self) -> list[dict[str, Any]]:
        """Load all saved network peers."""
        rows = self._conn.execute(
            "SELECT address, agent_id, agent_name, last_seen, trust_score "
            "FROM network_peers ORDER BY last_seen DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_peer(self, address: str) -> bool:
        """Delete a network peer."""
        cursor = self._conn.execute(
            "DELETE FROM network_peers WHERE address = ?", (address,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Skill reputation
    # ------------------------------------------------------------------

    def save_skill_reputation(
        self,
        skill_name: str,
        content_hash: str = "",
        source_agent: str = "",
        local_rating: float = 0.0,
        imported_at: float = 0.0,
    ) -> None:
        """Save or update a skill reputation record."""
        self._conn.execute(
            "INSERT INTO skill_reputation "
            "(skill_name, content_hash, source_agent, local_rating, imported_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(skill_name, source_agent) DO UPDATE SET "
            "content_hash=excluded.content_hash, local_rating=excluded.local_rating",
            (skill_name, content_hash, source_agent, local_rating, imported_at or time.time()),
        )
        self._conn.commit()

    def load_skill_reputation(self, skill_name: str) -> list[dict[str, Any]]:
        """Load reputation records for a skill."""
        rows = self._conn.execute(
            "SELECT skill_name, content_hash, source_agent, local_rating, "
            "peer_ratings, imported_at FROM skill_reputation WHERE skill_name = ?",
            (skill_name,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["peer_ratings"] = json.loads(d["peer_ratings"])
            result.append(d)
        return result

    def add_peer_rating(
        self,
        skill_name: str,
        source_agent: str,
        peer_agent: str,
        rating: float,
    ) -> None:
        """Add a peer's rating to a skill reputation record."""
        row = self._conn.execute(
            "SELECT peer_ratings FROM skill_reputation "
            "WHERE skill_name = ? AND source_agent = ?",
            (skill_name, source_agent),
        ).fetchone()
        if row is None:
            return
        ratings = json.loads(row["peer_ratings"])
        # Update or append
        for r in ratings:
            if r.get("peer") == peer_agent:
                r["rating"] = rating
                r["timestamp"] = time.time()
                break
        else:
            ratings.append({
                "peer": peer_agent,
                "rating": rating,
                "timestamp": time.time(),
            })
        self._conn.execute(
            "UPDATE skill_reputation SET peer_ratings = ? "
            "WHERE skill_name = ? AND source_agent = ?",
            (json.dumps(ratings), skill_name, source_agent),
        )
        self._conn.commit()
