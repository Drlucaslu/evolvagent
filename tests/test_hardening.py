"""
Tests for Production Hardening features.

Covers: logging setup, config validation, protocol safety, rate limiting,
schema versioning, activity log cleanup, and graceful shutdown timeouts.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from evolvagent.core.config import (
    LoggingConfig,
    NetworkConfig,
    Settings,
    load_settings,
    reset_settings,
    validate_settings,
)
from evolvagent.core.log import reset_logging, setup_logging
from evolvagent.core.protocol import (
    ALL_MSG_TYPES,
    MAX_MESSAGE_SIZE,
    MSG_PING,
    NetworkMessage,
    validate_message,
)
from evolvagent.core.storage import CURRENT_SCHEMA_VERSION, SkillStore


# ---------------------------------------------------------------------------
# TestLogging — setup_logging, level control, rotation config
# ---------------------------------------------------------------------------


class TestLogging:
    """Test unified logging setup."""

    def setup_method(self):
        reset_logging()

    def teardown_method(self):
        reset_logging()

    def test_setup_logging_console_only(self):
        """setup_logging with no log_dir adds a StreamHandler."""
        setup_logging(level="WARNING")
        root = logging.getLogger()
        # Find our StreamHandler (not pytest's LogCaptureHandler)
        stream_handlers = [
            h for h in root.handlers
            if type(h) is logging.StreamHandler
        ]
        assert len(stream_handlers) >= 1
        assert stream_handlers[0].level == logging.WARNING

    def test_setup_logging_with_file(self, tmp_path):
        """setup_logging with log_dir creates rotating file handler."""
        from logging.handlers import RotatingFileHandler

        setup_logging(level="INFO", log_dir=str(tmp_path), log_file="test.log")
        root = logging.getLogger()
        # File handler should exist and be at DEBUG level
        file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.DEBUG
        # Log file should be created on first write
        test_logger = logging.getLogger("test.hardening")
        test_logger.debug("test message")
        assert (tmp_path / "test.log").exists()

    def test_setup_logging_idempotent(self):
        """Calling setup_logging twice does not add duplicate handlers."""
        setup_logging(level="INFO")
        count1 = len(logging.getLogger().handlers)
        setup_logging(level="DEBUG")  # second call — should be no-op
        count2 = len(logging.getLogger().handlers)
        assert count1 == count2

    def test_debug_level_captures_debug(self, tmp_path):
        """DEBUG level file handler captures debug messages."""
        setup_logging(level="DEBUG", log_dir=str(tmp_path))
        test_logger = logging.getLogger("test.debug")
        test_logger.debug("debug message captured")
        log_content = (tmp_path / "evolvagent.log").read_text()
        assert "debug message captured" in log_content

    def test_rotation_config(self, tmp_path):
        """File handler respects max_bytes and backup_count."""
        from logging.handlers import RotatingFileHandler
        setup_logging(
            level="INFO",
            log_dir=str(tmp_path),
            max_bytes=1000,
            backup_count=2,
        )
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        assert file_handlers[0].maxBytes == 1000
        assert file_handlers[0].backupCount == 2


# ---------------------------------------------------------------------------
# TestConfigValidation — valid/invalid settings detection
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Test validate_settings() for various fields."""

    def test_default_settings_pass(self):
        """Default settings should pass validation."""
        errors = validate_settings(Settings())
        assert errors == []

    def test_invalid_port_zero(self):
        """Port 0 should fail validation."""
        s = Settings(network=NetworkConfig(listen_port=0))
        errors = validate_settings(s)
        assert any("listen_port" in e for e in errors)

    def test_invalid_port_too_high(self):
        """Port > 65535 should fail."""
        s = Settings(network=NetworkConfig(listen_port=70000))
        errors = validate_settings(s)
        assert any("listen_port" in e for e in errors)

    def test_valid_port(self):
        """Valid port should pass."""
        s = Settings(network=NetworkConfig(listen_port=8080))
        errors = validate_settings(s)
        assert not any("listen_port" in e for e in errors)

    def test_invalid_temperature_negative(self):
        """Negative temperature should fail."""
        from evolvagent.core.config import LLMConfig
        s = Settings(llm=LLMConfig(temperature=-0.5))
        errors = validate_settings(s)
        assert any("temperature" in e for e in errors)

    def test_invalid_temperature_too_high(self):
        """Temperature > 2.0 should fail."""
        from evolvagent.core.config import LLMConfig
        s = Settings(llm=LLMConfig(temperature=3.0))
        errors = validate_settings(s)
        assert any("temperature" in e for e in errors)

    def test_invalid_thresholds(self):
        """Scheduler thresholds <= 0 should fail."""
        from evolvagent.core.config import SchedulerConfig
        s = Settings(scheduler=SchedulerConfig(cpu_threshold_percent=0))
        errors = validate_settings(s)
        assert any("cpu_threshold" in e for e in errors)

    def test_invalid_cost_limit(self):
        """Negative cost limit should fail."""
        from evolvagent.core.config import LLMConfig
        s = Settings(llm=LLMConfig(daily_cost_limit_usd=-1.0))
        errors = validate_settings(s)
        assert any("cost_limit" in e for e in errors)

    def test_invalid_log_level(self):
        """Invalid log level string should fail."""
        s = Settings(logging=LoggingConfig(level="TRACE"))
        errors = validate_settings(s)
        assert any("logging.level" in e for e in errors)

    def test_valid_log_level_case_insensitive(self):
        """'debug' (lowercase) should be accepted."""
        s = Settings(logging=LoggingConfig(level="debug"))
        errors = validate_settings(s)
        assert not any("logging.level" in e for e in errors)

    def test_multiple_errors(self):
        """Multiple invalid fields produce multiple errors."""
        from evolvagent.core.config import LLMConfig
        s = Settings(
            network=NetworkConfig(listen_port=0),
            llm=LLMConfig(temperature=5.0),
        )
        errors = validate_settings(s)
        assert len(errors) >= 2


# ---------------------------------------------------------------------------
# TestProtocolSafety — malformed JSON, oversized messages, unknown types
# ---------------------------------------------------------------------------


class TestProtocolSafety:
    """Test protocol-level safety checks."""

    def test_from_json_malformed_json(self):
        """Malformed JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            NetworkMessage.from_json("not json at all")

    def test_from_json_missing_type(self):
        """JSON without 'type' field raises ValueError."""
        with pytest.raises(ValueError, match="missing required 'type'"):
            NetworkMessage.from_json('{"sender_id": "abc"}')

    def test_from_json_not_object(self):
        """JSON array (not object) raises ValueError."""
        with pytest.raises(ValueError, match="must be a JSON object"):
            NetworkMessage.from_json('[1, 2, 3]')

    def test_from_json_valid(self):
        """Valid JSON parses successfully."""
        msg = NetworkMessage.from_json('{"type": "ping", "sender_id": "a1"}')
        assert msg.type == "ping"
        assert msg.sender_id == "a1"

    def test_max_message_size_constant(self):
        """MAX_MESSAGE_SIZE is 1MB."""
        assert MAX_MESSAGE_SIZE == 1_048_576

    def test_validate_message_valid(self):
        """Known message type passes validation."""
        msg = NetworkMessage(type=MSG_PING)
        assert validate_message(msg) is None

    def test_validate_message_unknown_type(self):
        """Unknown message type fails validation."""
        msg = NetworkMessage(type="unknown_type")
        err = validate_message(msg)
        assert err is not None
        assert "Unknown message type" in err

    def test_validate_all_known_types(self):
        """All types in ALL_MSG_TYPES pass validation."""
        for msg_type in ALL_MSG_TYPES:
            msg = NetworkMessage(type=msg_type)
            assert validate_message(msg) is None


# ---------------------------------------------------------------------------
# TestRateLimiter — rate limiting in NetworkServer
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """Test the _check_rate_limit method on NetworkServer."""

    def _make_server(self, rate_limit: int = 100):
        """Create a minimal NetworkServer with mocked agent."""
        from evolvagent.core.network import NetworkServer

        agent = MagicMock()
        agent.agent_id = "test-agent"
        agent.settings.network = NetworkConfig(rate_limit_per_min=rate_limit)
        agent.bus = MagicMock()

        server = NetworkServer(agent)
        return server

    def test_under_limit_passes(self):
        """Messages under rate limit are allowed."""
        server = self._make_server(rate_limit=10)
        for _ in range(10):
            assert server._check_rate_limit("peer1") is True

    def test_over_limit_blocked(self):
        """Messages exceeding rate limit are rejected."""
        server = self._make_server(rate_limit=5)
        for _ in range(5):
            server._check_rate_limit("peer2")
        # 6th message should be blocked
        assert server._check_rate_limit("peer2") is False

    def test_different_peers_independent(self):
        """Rate limits are tracked per-peer."""
        server = self._make_server(rate_limit=3)
        for _ in range(3):
            server._check_rate_limit("peer-a")
        # peer-a at limit
        assert server._check_rate_limit("peer-a") is False
        # peer-b should still work
        assert server._check_rate_limit("peer-b") is True

    def test_window_expiry(self):
        """Old timestamps are pruned from the sliding window."""
        server = self._make_server(rate_limit=2)
        # Fill limit
        server._check_rate_limit("peer3")
        server._check_rate_limit("peer3")
        assert server._check_rate_limit("peer3") is False

        # Simulate time passing — manually expire old timestamps
        server._rate_limiter["peer3"] = [time.time() - 120]  # 2 min ago
        assert server._check_rate_limit("peer3") is True


# ---------------------------------------------------------------------------
# TestSchemaVersion — version tracking and migration
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    """Test database schema versioning and migration."""

    def test_new_database_gets_current_version(self, tmp_path):
        """A fresh database should be at CURRENT_SCHEMA_VERSION."""
        store = SkillStore(tmp_path / "test.db")
        assert store.schema_version() == CURRENT_SCHEMA_VERSION
        store.close()

    def test_current_schema_version_is_2(self):
        """Current schema version should be 2."""
        assert CURRENT_SCHEMA_VERSION == 2

    def test_v1_database_migrates_to_v2(self, tmp_path):
        """A v1 database (missing network tables) auto-migrates to v2."""
        db_path = tmp_path / "v1.db"
        # Manually create a v1 database
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        # Create v1 tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL,
                category TEXT DEFAULT 'general', status TEXT DEFAULT 'active',
                utility REAL DEFAULT 0.5, data TEXT NOT NULL, updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS skill_definitions (
                name TEXT PRIMARY KEY, definition TEXT NOT NULL, created_at REAL
            );
            CREATE TABLE IF NOT EXISTS agent_stats (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
                workspace TEXT DEFAULT '', action TEXT NOT NULL,
                query TEXT DEFAULT '', skill_used TEXT DEFAULT '', success INTEGER DEFAULT 1
            );
        """)
        conn.commit()
        conn.close()

        # Open with SkillStore — should auto-migrate
        store = SkillStore(db_path)
        assert store.schema_version() == 2

        # Verify network_peers table exists
        row = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='network_peers'"
        ).fetchone()
        assert row is not None

        # Verify skill_reputation table exists
        row = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='skill_reputation'"
        ).fetchone()
        assert row is not None

        store.close()

    def test_pre_existing_no_version_table(self, tmp_path):
        """A database with no schema_version table gets full schema."""
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()

        store = SkillStore(db_path)
        assert store.schema_version() == CURRENT_SCHEMA_VERSION
        store.close()


# ---------------------------------------------------------------------------
# TestActivityCleanup — retention-based deletion and vacuum
# ---------------------------------------------------------------------------


class TestActivityCleanup:
    """Test activity log cleanup and vacuum operations."""

    def _make_store_with_logs(self, tmp_path, entries):
        """Create a store and insert activity log entries with given timestamps."""
        store = SkillStore(tmp_path / "cleanup.db")
        for ts, action in entries:
            store._conn.execute(
                "INSERT INTO activity_log (timestamp, workspace, action) VALUES (?, '', ?)",
                (ts, action),
            )
        store._conn.commit()
        return store

    def test_cleanup_deletes_old_entries(self, tmp_path):
        """Entries older than retention_days are deleted."""
        now = time.time()
        old = now - (100 * 86400)  # 100 days ago
        entries = [
            (old, "old_action"),
            (now, "recent_action"),
        ]
        store = self._make_store_with_logs(tmp_path, entries)

        deleted = store.cleanup_activity_log(retention_days=90)
        assert deleted == 1

        remaining = store._conn.execute(
            "SELECT action FROM activity_log"
        ).fetchall()
        assert len(remaining) == 1
        assert remaining[0][0] == "recent_action"
        store.close()

    def test_cleanup_no_old_entries(self, tmp_path):
        """No entries deleted when all are within retention."""
        now = time.time()
        entries = [(now, "action1"), (now - 3600, "action2")]
        store = self._make_store_with_logs(tmp_path, entries)

        deleted = store.cleanup_activity_log(retention_days=90)
        assert deleted == 0
        store.close()

    def test_cleanup_custom_retention(self, tmp_path):
        """Custom retention period works correctly."""
        now = time.time()
        entries = [
            (now - (10 * 86400), "10_days_old"),
            (now - (3 * 86400), "3_days_old"),
            (now, "fresh"),
        ]
        store = self._make_store_with_logs(tmp_path, entries)

        deleted = store.cleanup_activity_log(retention_days=5)
        assert deleted == 1  # only the 10-day-old entry

        remaining = store._conn.execute("SELECT COUNT(*) FROM activity_log").fetchone()
        assert remaining[0] == 2
        store.close()

    def test_vacuum_does_not_error(self, tmp_path):
        """vacuum() runs without raising exceptions."""
        store = SkillStore(tmp_path / "vacuum.db")
        store.log_activity("ws", "test")
        store.vacuum()  # should not raise
        store.close()


# ---------------------------------------------------------------------------
# TestGracefulShutdown — ws.close with timeout
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """Test graceful shutdown does not hang on ws.close()."""

    @pytest.mark.asyncio
    async def test_stop_with_hanging_connection(self):
        """stop() completes even when ws.close() hangs."""
        from evolvagent.core.network import NetworkServer

        agent = MagicMock()
        agent.agent_id = "shutdown-test"
        agent.settings.network = NetworkConfig()
        agent.bus = MagicMock()
        agent.bus.emit_async = AsyncMock()
        agent._store = None

        server = NetworkServer(agent)
        server._running = True
        server._server = None  # no real server

        # Create a mock peer with a hanging close
        hanging_ws = AsyncMock()
        hanging_close = asyncio.Future()
        # Never resolve — simulates a hang
        hanging_ws.close = MagicMock(return_value=hanging_close)

        mock_peer = MagicMock()
        mock_peer.connection = hanging_ws
        mock_peer.address = "hang:1234"
        mock_peer.agent_id = ""

        server._peer_manager._peers = {"hang:1234": mock_peer}

        # stop() should complete within a reasonable time (timeout=3s per peer)
        await asyncio.wait_for(server.stop(), timeout=10)
        assert server._running is False

    @pytest.mark.asyncio
    async def test_stop_with_normal_connection(self):
        """stop() works normally when ws.close() resolves quickly."""
        from evolvagent.core.network import NetworkServer

        agent = MagicMock()
        agent.agent_id = "normal-test"
        agent.settings.network = NetworkConfig()
        agent.bus = MagicMock()
        agent.bus.emit_async = AsyncMock()
        agent._store = None

        server = NetworkServer(agent)
        server._running = True
        server._server = None

        normal_ws = AsyncMock()
        mock_peer = MagicMock()
        mock_peer.connection = normal_ws
        mock_peer.address = "ok:5678"
        mock_peer.agent_id = ""

        server._peer_manager._peers = {"ok:5678": mock_peer}

        await server.stop()
        assert server._running is False
        normal_ws.close.assert_called_once()


# ---------------------------------------------------------------------------
# TestLoggingConfig — config dataclass
# ---------------------------------------------------------------------------


class TestLoggingConfig:
    """Test LoggingConfig integration with Settings."""

    def test_default_values(self):
        """LoggingConfig has correct defaults."""
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert cfg.log_file == "evolvagent.log"
        assert cfg.max_bytes == 5_000_000
        assert cfg.backup_count == 3

    def test_settings_includes_logging(self):
        """Settings includes logging field."""
        s = Settings()
        assert hasattr(s, "logging")
        assert isinstance(s.logging, LoggingConfig)

    def test_load_from_toml(self, tmp_path):
        """LoggingConfig loads from TOML file."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[logging]\nlevel = "DEBUG"\nlog_file = "custom.log"\n'
        )
        reset_settings()
        settings = load_settings(config_file)
        assert settings.logging.level == "DEBUG"
        assert settings.logging.log_file == "custom.log"


# ---------------------------------------------------------------------------
# TestNetworkConfigExtensions — TLS and rate limit config
# ---------------------------------------------------------------------------


class TestNetworkConfigExtensions:
    """Test new NetworkConfig fields."""

    def test_default_tls_disabled(self):
        """TLS is disabled by default."""
        cfg = NetworkConfig()
        assert cfg.enable_tls is False
        assert cfg.tls_cert == ""
        assert cfg.tls_key == ""

    def test_default_rate_limit(self):
        """Rate limit defaults to 100/min."""
        cfg = NetworkConfig()
        assert cfg.rate_limit_per_min == 100

    def test_custom_values(self):
        """Custom TLS and rate limit values work."""
        cfg = NetworkConfig(
            enable_tls=True,
            tls_cert="/path/cert.pem",
            tls_key="/path/key.pem",
            rate_limit_per_min=50,
        )
        assert cfg.enable_tls is True
        assert cfg.tls_cert == "/path/cert.pem"
        assert cfg.rate_limit_per_min == 50
