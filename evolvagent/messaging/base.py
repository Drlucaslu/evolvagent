"""
Messaging bridge: connects NotifierAdapters to the Agent's EventBus.

Architecture:
    EventBus events  -->  MessagingBridge  -->  NotifierAdapter(s)  -->  User
    User messages    <--  MessagingBridge  <--  NotifierAdapter(s)  <--  User
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from .formatter import format_event, format_skills_list, format_status_report

if TYPE_CHECKING:
    from evolvagent.core.agent import Agent
    from evolvagent.core.config import MessagingConfig
    from evolvagent.core.events import Event

logger = logging.getLogger(__name__)

# Type for inbound message handler: (chat_id, text) -> response text
InboundHandler = Callable[[str, str], Coroutine[Any, Any, str]]


class NotifierAdapter(ABC):
    """
    Abstract base class for messaging platform adapters.

    Subclass this to add support for Telegram, WhatsApp, Slack, etc.
    """

    name: str = ""
    is_running: bool = False

    @abstractmethod
    async def start(self) -> None:
        """Start the adapter (connect to platform, begin polling)."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the adapter gracefully."""

    @abstractmethod
    async def send_message(self, chat_id: str, text: str) -> bool:
        """Send a text message. Returns True on success."""

    @abstractmethod
    async def send_message_with_buttons(
        self, chat_id: str, text: str, buttons: list[tuple[str, str]]
    ) -> None:
        """
        Send a message with inline buttons.

        Args:
            buttons: list of (label, callback_data) tuples
        """

    @abstractmethod
    async def wait_for_callback(self, chat_id: str, timeout: float) -> str | None:
        """
        Wait for a button callback from the user.

        Returns the callback_data string, or None on timeout.
        """

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        """Set the handler for inbound user messages."""
        self._inbound_handler = handler


class MessagingBridge:
    """
    Central hub connecting the Agent's EventBus to messaging adapters.

    Responsibilities:
    - Subscribe to configured EventBus topics and forward formatted messages
    - Route inbound user messages to agent commands
    - Manage ReportScheduler lifecycle
    """

    def __init__(self, agent: Agent, config: MessagingConfig) -> None:
        self._agent = agent
        self._config = config
        self._adapters: list[NotifierAdapter] = []
        self._report_scheduler: Any = None  # ReportScheduler, set in start()
        self._request_lock = asyncio.Lock()  # Serialize /ask requests

    @property
    def is_running(self) -> bool:
        return any(a.is_running for a in self._adapters)

    @property
    def adapters(self) -> list[NotifierAdapter]:
        return list(self._adapters)

    async def start(self) -> None:
        """Initialize adapters and subscribe to events."""
        # Telegram adapter
        if self._config.telegram.enabled:
            try:
                from .telegram import TelegramAdapter

                # Env var fallback for bot token
                bot_token = (
                    self._config.telegram.bot_token
                    or os.environ.get("EVOLVAGENT_TG_BOT_TOKEN", "")
                )
                if not bot_token:
                    logger.warning(
                        "Telegram enabled but no bot_token found. "
                        "Set telegram.bot_token in config or EVOLVAGENT_TG_BOT_TOKEN env var."
                    )
                else:
                    adapter = TelegramAdapter(
                        bot_token=bot_token,
                        allowed_chat_ids=self._config.telegram.allowed_chat_ids,
                    )
                    adapter.set_inbound_handler(self._handle_inbound)
                    await adapter.start()
                    self._adapters.append(adapter)
                    logger.info("Telegram adapter started")
            except ImportError:
                logger.warning(
                    "python-telegram-bot not installed. "
                    "Run: pip install evolvagent[messaging]"
                )
            except Exception as e:
                logger.warning("Failed to start Telegram adapter: %s", e)

        if not self._adapters:
            logger.warning("No messaging adapters started")
            return

        # Subscribe to EventBus topics
        for topic in self._config.forward_events:
            self._agent.bus.on(topic, self._on_event)

        # Start report scheduler
        if self._config.report_interval_seconds > 0:
            from .report import ReportScheduler

            self._report_scheduler = ReportScheduler(
                bridge=self,
                agent=self._agent,
                interval_seconds=self._config.report_interval_seconds,
            )
            self._report_scheduler.start()

        logger.info(
            "MessagingBridge started with %d adapter(s), forwarding %d event topic(s)",
            len(self._adapters),
            len(self._config.forward_events),
        )

    async def stop(self) -> None:
        """Stop all adapters and unsubscribe from events."""
        # Stop report scheduler
        if self._report_scheduler:
            await self._report_scheduler.stop()
            self._report_scheduler = None

        # Unsubscribe from EventBus
        for topic in self._config.forward_events:
            self._agent.bus.off(topic, self._on_event)

        # Stop adapters
        for adapter in self._adapters:
            try:
                await adapter.stop()
            except Exception as e:
                logger.warning("Error stopping adapter %s: %s", adapter.name, e)

        self._adapters.clear()
        logger.info("MessagingBridge stopped")

    async def broadcast(self, text: str) -> None:
        """Send a message to all adapters (all authorized chats)."""
        for adapter in self._adapters:
            if not adapter.is_running:
                continue
            # Snapshot to avoid RuntimeError if set is modified during iteration
            chat_ids = set(getattr(adapter, "active_chat_ids", set()))
            for chat_id in chat_ids:
                try:
                    await adapter.send_message(str(chat_id), text)
                except Exception as e:
                    logger.warning(
                        "Failed to broadcast via %s to %s: %s",
                        adapter.name, chat_id, e,
                    )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _on_event(self, event: Event) -> None:
        """Handle an EventBus event by formatting and broadcasting."""
        text = format_event(event)
        await self.broadcast(text)

    # ------------------------------------------------------------------
    # Inbound command routing
    # ------------------------------------------------------------------

    async def _handle_inbound(self, chat_id: str, text: str) -> str:
        """Route an inbound user message to the appropriate agent action."""
        if not self._config.enable_commands:
            return "Commands are disabled."

        text = text.strip()

        # Parse slash commands
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            # Strip bot username suffix (e.g. /status@MyBot)
            if "@" in cmd:
                cmd = cmd.split("@")[0]
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/start":
                return (
                    "EvolvAgent connected.\n"
                    "Use /status, /skills, /reflect, /ask <query>, or /help."
                )

            if cmd == "/help":
                return (
                    "/status - Agent status report\n"
                    "/skills - List active skills\n"
                    "/reflect - Trigger reflection\n"
                    "/ask <query> - Ask the agent\n"
                    "/help - Show this help"
                )

            if cmd == "/status":
                status = self._agent.status_dict()
                return format_status_report(status)

            if cmd == "/skills":
                skills = self._agent.active_skills
                return format_skills_list(skills)

            if cmd == "/reflect":
                from evolvagent.core.agent import AgentState

                if self._agent.state != AgentState.IDLE:
                    return f"Cannot reflect: agent is {self._agent.state.value}"
                result = await self._agent.enter_reflection()
                if result is None:
                    return "Reflection failed to start."
                if result.skipped_reason:
                    return f"Reflection skipped: {result.skipped_reason}"
                return (
                    f"Reflection complete: analyzed={result.skills_analyzed} "
                    f"updated={result.skills_updated} "
                    f"principles={result.principles_extracted}"
                )

            if cmd == "/ask":
                if not arg:
                    return "Usage: /ask <your question>"
                return await self._handle_ask(chat_id, arg)

            return f"Unknown command: {cmd}. Type /help for available commands."

        # Plain text → treat as /ask
        return await self._handle_ask(chat_id, text)

    async def _handle_ask(self, chat_id: str, query: str) -> str:
        """Handle an /ask command with SUGGEST confirmation support.

        Uses a lock to serialize requests — the Agent state machine only
        allows one IDLE→ACTIVE transition at a time.
        """
        async with self._request_lock:
            confirm_callback = self._make_confirm_callback(chat_id)
            try:
                result = await self._agent.handle_request(
                    query, confirm_callback=confirm_callback,
                )
                return result
            except Exception as e:
                logger.exception("Error handling ask from %s: %s", chat_id, e)
                return f"Error: {e}"

    def _make_confirm_callback(self, chat_id: str):
        """Create a SUGGEST confirmation callback that uses Telegram buttons."""
        timeout = self._config.confirm_timeout_seconds

        async def confirm(skill_name: str, preview: str) -> bool:
            # Find the adapter for this chat
            for adapter in self._adapters:
                if not adapter.is_running:
                    continue
                buttons = [
                    ("Approve", "approve"),
                    ("Reject", "reject"),
                ]
                text = f"Skill: {skill_name}\n\n{preview}\n\nApprove execution?"
                await adapter.send_message_with_buttons(chat_id, text, buttons)
                result = await adapter.wait_for_callback(chat_id, timeout)
                return result == "approve"
            return False

        return confirm
