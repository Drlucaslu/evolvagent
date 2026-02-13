"""
Messaging integration for EvolvAgent.

Provides a bridge between the Agent's EventBus and external messaging
platforms (Telegram, etc.) for real-time notifications, periodic reports,
and remote command execution.

Install: pip install evolvagent[messaging]
"""

from .base import MessagingBridge, NotifierAdapter

__all__ = ["MessagingBridge", "NotifierAdapter"]
