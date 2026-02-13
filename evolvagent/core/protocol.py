"""
Network message protocol for EvolvAgent P2P communication.

Defines the message format, skill summaries for catalog exchange,
and content hashing for integrity verification. Pure data layer — no I/O.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .skill import SkillMetadata

# ---------------------------------------------------------------------------
# Message type constants
# ---------------------------------------------------------------------------

MSG_HELLO = "hello"
MSG_HELLO_ACK = "hello_ack"
MSG_PING = "ping"
MSG_PONG = "pong"
MSG_SKILL_CATALOG_REQUEST = "skill_catalog_request"
MSG_SKILL_CATALOG_RESPONSE = "skill_catalog_response"
MSG_SKILL_FETCH_REQUEST = "skill_fetch_request"
MSG_SKILL_FETCH_RESPONSE = "skill_fetch_response"
MSG_SKILL_FEEDBACK = "skill_feedback"
MSG_PEER_LIST_REQUEST = "peer_list_request"
MSG_PEER_LIST_RESPONSE = "peer_list_response"
MSG_ERROR = "error"

ALL_MSG_TYPES = {
    MSG_HELLO, MSG_HELLO_ACK, MSG_PING, MSG_PONG,
    MSG_SKILL_CATALOG_REQUEST, MSG_SKILL_CATALOG_RESPONSE,
    MSG_SKILL_FETCH_REQUEST, MSG_SKILL_FETCH_RESPONSE,
    MSG_SKILL_FEEDBACK,
    MSG_PEER_LIST_REQUEST, MSG_PEER_LIST_RESPONSE,
    MSG_ERROR,
}

MAX_MESSAGE_SIZE = 1_048_576  # 1 MB


# ---------------------------------------------------------------------------
# NetworkMessage — the universal wire format
# ---------------------------------------------------------------------------

@dataclass
class NetworkMessage:
    """A message exchanged between agents over WebSocket."""
    type: str
    sender_id: str = ""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string for WebSocket transmission."""
        return json.dumps({
            "type": self.type,
            "sender_id": self.sender_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> NetworkMessage:
        """Deserialize from JSON string.

        Raises ValueError on malformed data.
        """
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Invalid JSON message: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Message must be a JSON object")
        if "type" not in data:
            raise ValueError("Message missing required 'type' field")
        return cls(
            type=data["type"],
            sender_id=data.get("sender_id", ""),
            request_id=data.get("request_id", ""),
            timestamp=data.get("timestamp", 0.0),
            payload=data.get("payload", {}),
        )

    def reply(self, msg_type: str, payload: dict[str, Any] | None = None) -> NetworkMessage:
        """Create a reply message preserving request_id."""
        return NetworkMessage(
            type=msg_type,
            sender_id=self.sender_id,  # will be overwritten by sender
            request_id=self.request_id,
            payload=payload or {},
        )


def validate_message(msg: NetworkMessage) -> str | None:
    """Validate a parsed message. Returns error string or None if valid."""
    if msg.type not in ALL_MSG_TYPES:
        return f"Unknown message type: {msg.type}"
    return None


# ---------------------------------------------------------------------------
# SkillSummary — lightweight catalog entry for network browsing
# ---------------------------------------------------------------------------

@dataclass
class SkillSummary:
    """A lightweight skill entry for catalog exchange over the network."""
    name: str = ""
    description: str = ""
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    version: str = "0.1.0"
    agent_id: str = ""
    agent_name: str = ""
    utility_score: float = 0.5
    network_reputation: float = 0.0
    content_hash: str = ""

    @classmethod
    def from_metadata(cls, meta: SkillMetadata, agent_id: str, agent_name: str = "") -> SkillSummary:
        """Create a summary from full SkillMetadata."""
        return cls(
            name=meta.name,
            description=meta.description,
            category=meta.category,
            tags=list(meta.tags),
            version=meta.version,
            agent_id=agent_id,
            agent_name=agent_name,
            utility_score=meta.utility_score,
            network_reputation=meta.network_reputation,
            content_hash=meta.content_hash(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "version": self.version,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "utility_score": self.utility_score,
            "network_reputation": self.network_reputation,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillSummary:
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            version=data.get("version", "0.1.0"),
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            utility_score=data.get("utility_score", 0.5),
            network_reputation=data.get("network_reputation", 0.0),
            content_hash=data.get("content_hash", ""),
        )


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

def compute_skill_hash(definition: dict[str, Any]) -> str:
    """
    Compute a content hash for a skill definition.

    Includes the system_prompt and key metadata to detect changes.
    """
    parts = [
        definition.get("system_prompt", ""),
        definition.get("metadata", {}).get("name", ""),
        definition.get("metadata", {}).get("description", ""),
    ]
    content = "|".join(parts)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
