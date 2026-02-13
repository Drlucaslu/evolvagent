"""
Periodic report scheduler for EvolvAgent messaging.

Follows the same asyncio.Task + stop_event pattern as
evolvagent.core.scheduler.AgentScheduler.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .formatter import format_status_report

if TYPE_CHECKING:
    from evolvagent.core.agent import Agent

    from .base import MessagingBridge

logger = logging.getLogger(__name__)


class ReportScheduler:
    """
    Periodically sends status reports via the MessagingBridge.

    Default interval: 3600s (1 hour). Set to 0 to disable.
    """

    def __init__(
        self,
        bridge: MessagingBridge,
        agent: Agent,
        interval_seconds: int = 3600,
    ) -> None:
        self._bridge = bridge
        self._agent = agent
        self._interval = interval_seconds
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the periodic report task."""
        if self.is_running:
            return
        if self._interval <= 0:
            logger.info("Report scheduler disabled (interval=0)")
            return

        self._stop_event.clear()
        self._task = asyncio.get_event_loop().create_task(
            self._run_loop(), name="report-scheduler"
        )
        logger.info("Report scheduler started (interval=%ds)", self._interval)

    async def stop(self) -> None:
        """Stop the report scheduler."""
        if not self.is_running:
            return
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning("Report scheduler did not stop in time, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None
        logger.info("Report scheduler stopped")

    async def _run_loop(self) -> None:
        """Main loop — send report at each interval."""
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._interval
                )
                # Stop was requested
                break
            except asyncio.TimeoutError:
                # Normal timeout — time to report
                pass

            try:
                await self._send_report()
            except Exception:
                logger.exception("Failed to send periodic report")

    async def _send_report(self) -> None:
        """Generate and broadcast a status report."""
        status = self._agent.status_dict()
        text = format_status_report(status)
        await self._bridge.broadcast(text)
        logger.debug("Periodic status report sent")
