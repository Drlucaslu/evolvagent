"""
Event bus for inter-module communication.

Simple publish/subscribe pattern that decouples modules. All communication
between Agent subsystems flows through events rather than direct calls.
This makes it easy to add new modules without modifying existing ones.

Usage:
    bus = EventBus()
    bus.on("skill.executed", my_handler)
    bus.emit("skill.executed", {"skill_id": "file_organizer", "success": True})
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Union

logger = logging.getLogger(__name__)

# Type alias for event handlers
SyncHandler = Callable[["Event"], None]
AsyncHandler = Callable[["Event"], Coroutine[Any, Any, None]]
Handler = Union[SyncHandler, AsyncHandler]


@dataclass
class Event:
    """An event flowing through the bus."""

    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Which module emitted this event

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class EventBus:
    """
    Lightweight publish/subscribe event bus.

    Supports both sync and async handlers. Handlers are called in
    registration order. Errors in one handler don't prevent others
    from running.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self._history: list[Event] = []
        self._max_history: int = 1000

    def on(self, topic: str, handler: Handler) -> None:
        """Subscribe a handler to a topic."""
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)
        logger.debug("Handler registered for topic: %s", topic)

    def off(self, topic: str, handler: Handler) -> None:
        """Unsubscribe a handler from a topic."""
        if topic in self._handlers:
            self._handlers[topic] = [h for h in self._handlers[topic] if h != handler]

    def emit(self, topic: str, data: dict[str, Any] | None = None, source: str = "") -> Event:
        """
        Emit an event synchronously.

        All sync handlers are called immediately. Async handlers are
        scheduled on the running event loop if one exists.
        """
        event = Event(topic=topic, data=data or {}, source=source)
        self._record(event)
        self._dispatch(event)
        return event

    async def emit_async(
        self, topic: str, data: dict[str, Any] | None = None, source: str = ""
    ) -> Event:
        """Emit an event and await all async handlers."""
        event = Event(topic=topic, data=data or {}, source=source)
        self._record(event)
        await self._dispatch_async(event)
        return event

    def _dispatch(self, event: Event) -> None:
        """Call all handlers for an event topic."""
        handlers = self._handlers.get(event.topic, [])
        # Also call wildcard handlers
        handlers = handlers + self._handlers.get("*", [])

        for handler in handlers:
            try:
                result = handler(event)
                # If handler is async, try to schedule it
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        # No running loop - skip async handler in sync context
                        result.close()
                        logger.warning(
                            "Async handler %s skipped in sync context for topic %s",
                            handler.__name__,
                            event.topic,
                        )
            except Exception:
                logger.exception(
                    "Error in handler %s for topic %s", handler.__name__, event.topic
                )

    async def _dispatch_async(self, event: Event) -> None:
        """Call all handlers, awaiting async ones."""
        handlers = self._handlers.get(event.topic, [])
        handlers = handlers + self._handlers.get("*", [])

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Error in handler %s for topic %s", handler.__name__, event.topic
                )

    def _record(self, event: Event) -> None:
        """Record event in history ring buffer."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    @property
    def history(self) -> list[Event]:
        """Recent event history (read-only copy)."""
        return list(self._history)

    def clear(self) -> None:
        """Remove all handlers and history."""
        self._handlers.clear()
        self._history.clear()
