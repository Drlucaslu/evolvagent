"""
Tests for evolvagent.messaging — config, formatter, report, bridge, telegram adapter.

Uses a MockAdapter (in-memory NotifierAdapter) so no Telegram API is needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from evolvagent.core.config import (
    MessagingConfig,
    Settings,
    TelegramConfig,
    _make_section,
)
from evolvagent.core.events import Event, EventBus
from evolvagent.messaging.base import MessagingBridge, NotifierAdapter
from evolvagent.messaging.formatter import (
    _format_duration,
    _truncate,
    format_event,
    format_skills_list,
    format_status_report,
)
from evolvagent.messaging.report import ReportScheduler


# ---------------------------------------------------------------------------
# MockAdapter — in-memory NotifierAdapter for testing
# ---------------------------------------------------------------------------

class MockAdapter(NotifierAdapter):
    """In-memory adapter for testing, no external dependencies."""

    name = "mock"

    def __init__(self):
        self.is_running = False
        self.sent_messages: list[tuple[str, str]] = []
        self.sent_buttons: list[tuple[str, str, list]] = []
        self.active_chat_ids: set[int] = {123}
        self._inbound_handler = None
        self._callback_result: str | None = None

    async def start(self) -> None:
        self.is_running = True

    async def stop(self) -> None:
        self.is_running = False

    async def send_message(self, chat_id: str, text: str) -> bool:
        self.sent_messages.append((chat_id, text))
        return True

    async def send_message_with_buttons(
        self, chat_id: str, text: str, buttons: list[tuple[str, str]]
    ) -> None:
        self.sent_buttons.append((chat_id, text, buttons))

    async def wait_for_callback(self, chat_id: str, timeout: float) -> str | None:
        return self._callback_result


# ---------------------------------------------------------------------------
# Mock Agent for testing
# ---------------------------------------------------------------------------

def _make_mock_agent(messaging_config: MessagingConfig | None = None):
    """Create a minimal mock agent for bridge testing."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.agent_id = "test-agent-abc123"
    agent.state = MagicMock()
    agent.state.value = "idle"
    agent.bus = EventBus()
    agent.settings = Settings()
    if messaging_config:
        agent.settings.messaging = messaging_config
    agent.uptime = 3661.0
    agent.skill_count = 5
    agent._skills = {}
    agent.active_skills = []
    agent.stats = MagicMock()
    agent.stats.total_requests = 42
    agent.stats.successful_tasks = 38
    agent.stats.failed_tasks = 3
    agent.stats.no_skill_found = 1

    agent.status_dict.return_value = {
        "name": "test-agent",
        "agent_id": "test-agent-abc123",
        "state": "idle",
        "uptime_seconds": 3661.0,
        "skill_count": 5,
        "learned_skill_count": 2,
        "active_skills": ["general", "git_info"],
        "stats": {
            "total_requests": 42,
            "successful_tasks": 38,
            "failed_tasks": 3,
            "no_skill_found": 1,
        },
        "scheduler_running": True,
        "last_reflection": None,
        "last_learning": None,
        "network": None,
    }

    agent.handle_request = AsyncMock(return_value="Result from agent")
    agent.enter_reflection = AsyncMock(return_value=MagicMock(
        skipped_reason=None,
        skills_analyzed=3,
        skills_updated=1,
        principles_extracted=2,
    ))

    return agent


# ===========================================================================
# TestMessagingConfig
# ===========================================================================

class TestMessagingConfig:
    """Config loading, defaults, nested TOML parsing."""

    def test_default_messaging_config(self):
        settings = Settings()
        assert settings.messaging.enabled is False
        assert settings.messaging.telegram.enabled is False
        assert settings.messaging.telegram.bot_token == ""
        assert settings.messaging.report_interval_seconds == 3600
        assert "skill.executed" in settings.messaging.forward_events

    def test_telegram_config_defaults(self):
        tc = TelegramConfig()
        assert tc.enabled is False
        assert tc.bot_token == ""
        assert tc.allowed_chat_ids == []

    def test_nested_make_section(self):
        raw = {
            "enabled": True,
            "telegram": {
                "enabled": True,
                "bot_token": "123:ABC",
                "allowed_chat_ids": [111, 222],
            },
            "report_interval_seconds": 1800,
        }
        result = _make_section(MessagingConfig, raw)
        assert result.enabled is True
        assert result.telegram.enabled is True
        assert result.telegram.bot_token == "123:ABC"
        assert result.telegram.allowed_chat_ids == [111, 222]
        assert result.report_interval_seconds == 1800


# ===========================================================================
# TestFormatter
# ===========================================================================

class TestFormatter:
    """Event formatting, status report, skills list, truncation."""

    def test_format_skill_executed_success(self):
        event = Event(
            topic="skill.executed",
            data={"skill_name": "code_review", "success": True, "execution_time_ms": 320},
        )
        text = format_event(event)
        assert "code_review" in text
        assert "OK" in text
        assert "320" in text

    def test_format_skill_executed_failure(self):
        event = Event(
            topic="skill.executed",
            data={"skill_name": "deploy", "success": False, "execution_time_ms": 100},
        )
        text = format_event(event)
        assert "FAIL" in text

    def test_format_state_changed(self):
        event = Event(
            topic="agent.state_changed",
            data={"old_state": "idle", "new_state": "reflecting"},
        )
        text = format_event(event)
        assert "idle" in text
        assert "reflecting" in text

    def test_format_skill_learned(self):
        event = Event(
            topic="skill.learned",
            data={"skill_name": "data_analyzer"},
        )
        text = format_event(event)
        assert "data_analyzer" in text

    def test_format_maintenance_completed(self):
        event = Event(
            topic="scheduler.maintenance_completed",
            data={"decayed_count": 2, "archived_count": 1},
        )
        text = format_event(event)
        assert "decayed=2" in text
        assert "archived=1" in text

    def test_format_skill_taught(self):
        event = Event(
            topic="skill.taught",
            data={"skill_name": "my_custom_skill"},
        )
        text = format_event(event)
        assert "my_custom_skill" in text
        assert "taught" in text.lower()

    def test_format_reflection_started(self):
        event = Event(topic="agent.reflection_started", data={})
        text = format_event(event)
        assert "Reflection started" in text

    def test_format_reflection_completed(self):
        event = Event(topic="agent.reflection_completed", data={})
        text = format_event(event)
        assert "Reflection completed" in text

    def test_format_generic_event(self):
        event = Event(topic="custom.event", data={"foo": "bar"})
        text = format_event(event)
        assert "custom.event" in text

    def test_format_status_report(self):
        status = {
            "name": "my-agent",
            "state": "idle",
            "uptime_seconds": 7200,
            "skill_count": 10,
            "learned_skill_count": 3,
            "stats": {
                "total_requests": 50,
                "successful_tasks": 45,
                "failed_tasks": 4,
                "no_skill_found": 1,
            },
            "scheduler_running": True,
            "network": None,
        }
        text = format_status_report(status)
        assert "my-agent" in text
        assert "idle" in text
        assert "10" in text
        assert "50" in text

    def test_format_skills_list_empty(self):
        text = format_skills_list([])
        assert "No active skills" in text

    def test_format_skills_list_with_skills(self):
        from evolvagent.core.skill import SkillMetadata, SkillStatus, TrustLevel

        meta = SkillMetadata(
            skill_id="test1",
            name="test_skill",
            description="A test skill",
            trust_level=TrustLevel.SUGGEST,
        )
        meta.utility_score = 0.75
        meta.total_executions = 10
        meta.success_count = 8
        meta.status = SkillStatus.ACTIVE

        skill = MagicMock()
        skill.metadata = meta

        text = format_skills_list([skill])
        assert "test_skill" in text
        assert "suggest" in text
        assert "0.75" in text
        assert "10" in text

    def test_truncate(self):
        assert _truncate("hello", 10) == "hello"
        assert _truncate("hello world, this is long", 10) == "hello w..."

    def test_format_duration(self):
        assert _format_duration(30) == "30s"
        assert _format_duration(120) == "2m"
        assert _format_duration(7200) == "2.0h"
        assert _format_duration(172800) == "2.0d"


# ===========================================================================
# TestReportScheduler
# ===========================================================================

class TestReportScheduler:
    """Report scheduler start/stop and interval."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        agent = _make_mock_agent()
        bridge = MagicMock()
        bridge.broadcast = AsyncMock()

        scheduler = ReportScheduler(bridge=bridge, agent=agent, interval_seconds=3600)
        assert not scheduler.is_running

        scheduler.start()
        assert scheduler.is_running

        await scheduler.stop()
        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_disabled_when_interval_zero(self):
        agent = _make_mock_agent()
        bridge = MagicMock()

        scheduler = ReportScheduler(bridge=bridge, agent=agent, interval_seconds=0)
        scheduler.start()
        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_sends_report(self):
        agent = _make_mock_agent()
        bridge = MagicMock()
        bridge.broadcast = AsyncMock()

        scheduler = ReportScheduler(bridge=bridge, agent=agent, interval_seconds=1)
        scheduler.start()
        # Wait enough for one report cycle
        await asyncio.sleep(1.5)
        await scheduler.stop()

        assert bridge.broadcast.called
        text = bridge.broadcast.call_args[0][0]
        assert "test-agent" in text

    @pytest.mark.asyncio
    async def test_send_report_direct(self):
        agent = _make_mock_agent()
        bridge = MagicMock()
        bridge.broadcast = AsyncMock()

        scheduler = ReportScheduler(bridge=bridge, agent=agent)
        await scheduler._send_report()

        assert bridge.broadcast.called
        text = bridge.broadcast.call_args[0][0]
        assert "test-agent" in text
        assert "idle" in text


# ===========================================================================
# TestMessagingBridge
# ===========================================================================

class TestMessagingBridge:
    """Event forwarding, command routing, confirm flow."""

    @pytest.mark.asyncio
    async def test_event_forwarding(self):
        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)

        bridge = MessagingBridge(agent, config)
        adapter = MockAdapter()
        await adapter.start()
        bridge._adapters.append(adapter)

        # Subscribe to event
        for topic in config.forward_events:
            agent.bus.on(topic, bridge._on_event)

        # Emit an event
        await agent.bus.emit_async("skill.executed", {
            "skill_name": "test_skill",
            "success": True,
            "execution_time_ms": 100,
        })

        assert len(adapter.sent_messages) == 1
        assert "test_skill" in adapter.sent_messages[0][1]

        await bridge.stop()

    @pytest.mark.asyncio
    async def test_command_status(self):
        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        response = await bridge._handle_inbound("123", "/status")
        assert "test-agent" in response
        assert "idle" in response

    @pytest.mark.asyncio
    async def test_command_skills(self):
        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        response = await bridge._handle_inbound("123", "/skills")
        assert "No active skills" in response

    @pytest.mark.asyncio
    async def test_command_help(self):
        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        response = await bridge._handle_inbound("123", "/help")
        assert "/status" in response
        assert "/ask" in response

    @pytest.mark.asyncio
    async def test_command_ask(self):
        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        response = await bridge._handle_inbound("123", "/ask what is Python?")
        assert response == "Result from agent"
        agent.handle_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_plain_text_as_ask(self):
        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        response = await bridge._handle_inbound("123", "hello there")
        assert response == "Result from agent"

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        response = await bridge._handle_inbound("123", "/foobar")
        assert "Unknown command" in response

    @pytest.mark.asyncio
    async def test_start_command(self):
        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        response = await bridge._handle_inbound("123", "/start")
        assert "connected" in response.lower() or "EvolvAgent" in response

    @pytest.mark.asyncio
    async def test_confirm_callback(self):
        config = MessagingConfig(enabled=True, confirm_timeout_seconds=5)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        adapter = MockAdapter()
        adapter._callback_result = "approve"
        await adapter.start()
        bridge._adapters.append(adapter)

        confirm = bridge._make_confirm_callback("123")
        result = await confirm("test_skill", "Preview text")
        assert result is True

    @pytest.mark.asyncio
    async def test_confirm_callback_reject(self):
        config = MessagingConfig(enabled=True, confirm_timeout_seconds=5)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        adapter = MockAdapter()
        adapter._callback_result = "reject"
        await adapter.start()
        bridge._adapters.append(adapter)

        confirm = bridge._make_confirm_callback("123")
        result = await confirm("test_skill", "Preview text")
        assert result is False

    @pytest.mark.asyncio
    async def test_command_reflect(self):
        from evolvagent.core.agent import AgentState

        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)
        agent.state = AgentState.IDLE
        bridge = MessagingBridge(agent, config)

        response = await bridge._handle_inbound("123", "/reflect")
        assert "Reflection complete" in response
        agent.enter_reflection.assert_called_once()

    @pytest.mark.asyncio
    async def test_command_reflect_not_idle(self):
        from evolvagent.core.agent import AgentState

        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)
        agent.state = AgentState.ACTIVE
        bridge = MessagingBridge(agent, config)

        response = await bridge._handle_inbound("123", "/reflect")
        assert "Cannot reflect" in response

    @pytest.mark.asyncio
    async def test_enable_commands_disabled(self):
        config = MessagingConfig(enabled=True, enable_commands=False)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        response = await bridge._handle_inbound("123", "/status")
        assert "disabled" in response.lower()

    def test_bot_token_env_fallback(self):
        import os

        config = MessagingConfig(
            enabled=True,
            telegram=TelegramConfig(enabled=True, bot_token=""),
        )

        # Simulate env var set
        old = os.environ.get("EVOLVAGENT_TG_BOT_TOKEN")
        try:
            os.environ["EVOLVAGENT_TG_BOT_TOKEN"] = "env:token:123"
            # Same fallback logic used in MessagingBridge.start()
            token = (
                config.telegram.bot_token
                or os.environ.get("EVOLVAGENT_TG_BOT_TOKEN", "")
            )
            assert token == "env:token:123"
        finally:
            if old is None:
                os.environ.pop("EVOLVAGENT_TG_BOT_TOKEN", None)
            else:
                os.environ["EVOLVAGENT_TG_BOT_TOKEN"] = old

    @pytest.mark.asyncio
    async def test_command_ask_empty_arg(self):
        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        response = await bridge._handle_inbound("123", "/ask")
        assert "Usage" in response
        agent.handle_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_command_with_bot_username_suffix(self):
        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        # /status@MyBot should work like /status
        response = await bridge._handle_inbound("123", "/status@MyBot")
        assert "test-agent" in response
        assert "idle" in response

    @pytest.mark.asyncio
    async def test_broadcast_empty_active_chats(self):
        config = MessagingConfig(enabled=True)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        adapter = MockAdapter()
        adapter.active_chat_ids = set()  # No active chats
        await adapter.start()
        bridge._adapters.append(adapter)

        await bridge.broadcast("test message")
        assert len(adapter.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_confirm_callback_timeout(self):
        config = MessagingConfig(enabled=True, confirm_timeout_seconds=0.1)
        agent = _make_mock_agent(config)
        bridge = MessagingBridge(agent, config)

        adapter = MockAdapter()

        async def slow_wait(chat_id: str, timeout: float) -> str | None:
            await asyncio.sleep(timeout + 0.1)
            return None

        adapter.wait_for_callback = slow_wait
        await adapter.start()
        bridge._adapters.append(adapter)

        confirm = bridge._make_confirm_callback("123")
        result = await confirm("test_skill", "Preview text")
        assert result is False

    def test_bot_token_config_takes_precedence(self):
        import os

        config = MessagingConfig(
            enabled=True,
            telegram=TelegramConfig(enabled=True, bot_token="config:token"),
        )

        old = os.environ.get("EVOLVAGENT_TG_BOT_TOKEN")
        try:
            os.environ["EVOLVAGENT_TG_BOT_TOKEN"] = "env:token:123"
            token = (
                config.telegram.bot_token
                or os.environ.get("EVOLVAGENT_TG_BOT_TOKEN", "")
            )
            assert token == "config:token"
        finally:
            if old is None:
                os.environ.pop("EVOLVAGENT_TG_BOT_TOKEN", None)
            else:
                os.environ["EVOLVAGENT_TG_BOT_TOKEN"] = old


# ===========================================================================
# TestTelegramAdapter
# ===========================================================================

class TestTelegramAdapter:
    """Authorization, message splitting, InlineKeyboard construction."""

    def test_authorization_empty_allows_all(self):
        from evolvagent.messaging.telegram import TelegramAdapter

        adapter = TelegramAdapter(bot_token="fake:token", allowed_chat_ids=[])
        assert adapter._is_authorized(12345) is True
        assert adapter._is_authorized(99999) is True

    def test_authorization_restricted(self):
        from evolvagent.messaging.telegram import TelegramAdapter

        adapter = TelegramAdapter(
            bot_token="fake:token",
            allowed_chat_ids=[111, 222],
        )
        assert adapter._is_authorized(111) is True
        assert adapter._is_authorized(222) is True
        assert adapter._is_authorized(333) is False

    def test_message_splitting(self):
        from evolvagent.messaging.telegram import _split_message

        # Short message
        chunks = _split_message("hello")
        assert chunks == ["hello"]

        # Long message
        long_text = "x" * 5000
        chunks = _split_message(long_text)
        assert len(chunks) == 2
        assert len(chunks[0]) <= 4096
        total = sum(len(c) for c in chunks)
        assert total == 5000

    def test_message_splitting_at_newlines(self):
        from evolvagent.messaging.telegram import _split_message

        # Build text with newlines
        lines = ["Line " + str(i) for i in range(1000)]
        text = "\n".join(lines)
        chunks = _split_message(text)

        # Each chunk should be <= 4096
        for chunk in chunks:
            assert len(chunk) <= 4096

        # Reassembled text should contain all lines
        reassembled = "\n".join(chunks)
        for i in range(1000):
            assert f"Line {i}" in reassembled

    def test_requires_bot_token(self):
        from evolvagent.messaging.telegram import TelegramAdapter

        with pytest.raises(ValueError, match="bot_token"):
            TelegramAdapter(bot_token="")


# ===========================================================================
# TestNotifierAdapterABC
# ===========================================================================

class TestNotifierAdapterABC:
    """NotifierAdapter is an ABC and cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            NotifierAdapter()
