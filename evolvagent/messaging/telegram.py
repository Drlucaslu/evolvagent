"""
Telegram adapter for EvolvAgent messaging.

Uses python-telegram-bot v21+ (pure async). Install:
    pip install python-telegram-bot>=21.0
    # or: pip install evolvagent[messaging]
"""

from __future__ import annotations

import asyncio
import logging

from .base import InboundHandler, NotifierAdapter

logger = logging.getLogger(__name__)

# Telegram message length limit
MAX_MESSAGE_LENGTH = 4096


class TelegramAdapter(NotifierAdapter):
    """NotifierAdapter implementation for Telegram."""

    name = "telegram"

    def __init__(
        self,
        bot_token: str,
        allowed_chat_ids: list[int] | None = None,
    ) -> None:
        if not bot_token:
            raise ValueError("Telegram bot_token is required")

        self._bot_token = bot_token
        self._allowed_chat_ids: set[int] = set(allowed_chat_ids or [])
        self._app = None  # telegram.ext.Application
        self._inbound_handler: InboundHandler | None = None
        self.is_running = False

        # Track active chats for broadcasting
        self.active_chat_ids: set[int] = set(self._allowed_chat_ids)

        # Callback futures: chat_id -> Future
        self._callback_futures: dict[str, asyncio.Future] = {}

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        self._inbound_handler = handler

    async def start(self) -> None:
        """Start the Telegram bot with polling."""
        from telegram import Update
        from telegram.ext import (
            Application,
            CallbackQueryHandler,
            MessageHandler,
            filters,
        )

        self._app = Application.builder().token(self._bot_token).build()

        # Register handlers
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        self._app.add_handler(
            MessageHandler(filters.COMMAND, self._on_message)
        )
        self._app.add_handler(
            CallbackQueryHandler(self._on_callback_query)
        )

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(allowed_updates=Update.ALL_TYPES)

        self.is_running = True
        logger.info("Telegram adapter started (polling)")

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._app:
            try:
                if self._app.updater and self._app.updater.running:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                logger.warning("Error during Telegram shutdown: %s", e)
            self._app = None

        # Cancel pending callback futures
        for future in self._callback_futures.values():
            if not future.done():
                future.cancel()
        self._callback_futures.clear()

        self.is_running = False
        logger.info("Telegram adapter stopped")

    def _is_authorized(self, chat_id: int) -> bool:
        """Check if a chat_id is authorized. Empty set = allow all."""
        if not self._allowed_chat_ids:
            return True
        return chat_id in self._allowed_chat_ids

    async def send_message(self, chat_id: str, text: str) -> bool:
        """Send a text message, splitting if over 4096 chars."""
        if not self._app or not self.is_running:
            return False

        try:
            for chunk in _split_message(text):
                await self._app.bot.send_message(
                    chat_id=int(chat_id),
                    text=chunk,
                    parse_mode=None,  # Plain text to avoid Markdown parse errors
                )
            return True
        except Exception as e:
            logger.warning("Failed to send Telegram message to %s: %s", chat_id, e)
            return False

    async def send_message_with_buttons(
        self, chat_id: str, text: str, buttons: list[tuple[str, str]]
    ) -> None:
        """Send a message with InlineKeyboard buttons."""
        if not self._app or not self.is_running:
            return

        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = [
            [InlineKeyboardButton(label, callback_data=data)]
            for label, data in buttons
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        try:
            await self._app.bot.send_message(
                chat_id=int(chat_id),
                text=text,
                reply_markup=reply_markup,
            )
        except Exception as e:
            logger.warning("Failed to send Telegram buttons to %s: %s", chat_id, e)

    async def wait_for_callback(self, chat_id: str, timeout: float) -> str | None:
        """Wait for a callback query from the user, with timeout."""
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._callback_futures[chat_id] = future

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return None
        finally:
            self._callback_futures.pop(chat_id, None)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    async def _on_message(self, update, context) -> None:
        """Handle incoming text/command messages."""
        if not update.message or not update.message.text:
            return

        chat_id = update.message.chat_id

        if not self._is_authorized(chat_id):
            logger.warning("Unauthorized message from chat_id=%d", chat_id)
            await update.message.reply_text("Unauthorized.")
            return

        # Track active chats
        self.active_chat_ids.add(chat_id)

        if self._inbound_handler:
            response = await self._inbound_handler(str(chat_id), update.message.text)
            for chunk in _split_message(response):
                await update.message.reply_text(chunk)

    async def _on_callback_query(self, update, context) -> None:
        """Handle inline keyboard button clicks."""
        query = update.callback_query
        if not query:
            return

        chat_id = str(query.message.chat_id)
        data = query.data

        # Answer the callback to remove the loading state
        await query.answer()

        # Resolve the waiting future if any
        future = self._callback_futures.get(chat_id)
        if future and not future.done():
            future.set_result(data)
        else:
            # No one waiting — send a text response
            label = "Approved" if data == "approve" else "Rejected"
            await query.edit_message_text(
                text=f"{query.message.text}\n\n{label}."
            )


def _split_message(text: str) -> list[str]:
    """Split a message into chunks of MAX_MESSAGE_LENGTH."""
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]

    chunks = []
    while text:
        if len(text) <= MAX_MESSAGE_LENGTH:
            chunks.append(text)
            break
        # Try to split at a newline
        split_at = text.rfind("\n", 0, MAX_MESSAGE_LENGTH)
        if split_at == -1:
            split_at = MAX_MESSAGE_LENGTH
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    return chunks
