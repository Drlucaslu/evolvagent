"""
Peer node management for EvolvAgent P2P network.

Tracks known peers, their connection state, and cached skill catalogs.
Pure state tracking — no I/O or WebSocket logic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .events import EventBus

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Information about a known peer node."""
    host: str = ""
    port: int = 0
    agent_id: str = ""
    agent_name: str = ""
    last_seen: float = 0.0
    connected: bool = False
    connection: Any = None  # websocket connection object
    failed_attempts: int = 0
    skills_cached: list[dict[str, Any]] = field(default_factory=list)

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def is_stale(self) -> bool:
        """Peer not seen for > 5 minutes."""
        return self.last_seen > 0 and (time.time() - self.last_seen) > 300


class PeerManager:
    """
    Manages known and connected peers.

    Handles peer discovery via seed nodes and gossip protocol.
    Emits events for peer lifecycle changes.
    """

    def __init__(self, bus: EventBus, agent_id: str, max_peers: int = 20):
        self._bus = bus
        self._agent_id = agent_id
        self._max_peers = max_peers
        self._peers: dict[str, PeerInfo] = {}  # keyed by "host:port"

    def add_seed(self, host: str, port: int) -> PeerInfo:
        """Add a seed peer to the known list."""
        addr = f"{host}:{port}"
        if addr not in self._peers:
            peer = PeerInfo(host=host, port=port)
            self._peers[addr] = peer
            self._bus.emit("network.peer_discovered", {
                "address": addr,
                "source": "seed",
            }, source="peer_manager")
            logger.info("Seed peer added: %s", addr)
            return peer
        return self._peers[addr]

    def add_peers_from_gossip(self, peers: list[dict[str, Any]]) -> int:
        """Add peers from a gossip peer_list_response. Returns count of new peers."""
        added = 0
        for p in peers:
            host = p.get("host", "")
            port = p.get("port", 0)
            agent_id = p.get("agent_id", "")
            if not host or not port:
                continue
            # Don't add ourselves
            if agent_id == self._agent_id:
                continue
            addr = f"{host}:{port}"
            if addr not in self._peers and len(self._peers) < self._max_peers:
                self._peers[addr] = PeerInfo(
                    host=host, port=port,
                    agent_id=agent_id,
                    agent_name=p.get("agent_name", ""),
                )
                added += 1
                self._bus.emit("network.peer_discovered", {
                    "address": addr,
                    "agent_id": agent_id,
                    "source": "gossip",
                }, source="peer_manager")
        if added:
            logger.info("Discovered %d new peers via gossip", added)
        return added

    def mark_connected(
        self,
        addr: str,
        agent_id: str,
        agent_name: str,
        ws: Any,
    ) -> PeerInfo:
        """Mark a peer as connected after successful handshake."""
        if addr not in self._peers:
            host, port_str = addr.rsplit(":", 1)
            self._peers[addr] = PeerInfo(host=host, port=int(port_str))

        peer = self._peers[addr]
        peer.agent_id = agent_id
        peer.agent_name = agent_name
        peer.connected = True
        peer.connection = ws
        peer.last_seen = time.time()
        peer.failed_attempts = 0

        self._bus.emit("network.peer_connected", {
            "address": addr,
            "agent_id": agent_id,
            "agent_name": agent_name,
        }, source="peer_manager")
        logger.info("Peer connected: %s (%s)", agent_name or agent_id, addr)
        return peer

    def mark_disconnected(self, addr: str) -> None:
        """Mark a peer as disconnected."""
        if addr in self._peers:
            peer = self._peers[addr]
            was_connected = peer.connected
            peer.connected = False
            peer.connection = None
            if was_connected:
                self._bus.emit("network.peer_disconnected", {
                    "address": addr,
                    "agent_id": peer.agent_id,
                }, source="peer_manager")
                logger.info("Peer disconnected: %s", addr)

    def mark_failed(self, addr: str) -> None:
        """Record a failed connection attempt."""
        if addr in self._peers:
            self._peers[addr].failed_attempts += 1

    @property
    def connected_peers(self) -> list[PeerInfo]:
        """All currently connected peers."""
        return [p for p in self._peers.values() if p.connected]

    @property
    def known_peers(self) -> list[PeerInfo]:
        """All known peers (connected or not)."""
        return list(self._peers.values())

    def get_peer(self, agent_id: str) -> PeerInfo | None:
        """Find a peer by agent_id."""
        for peer in self._peers.values():
            if peer.agent_id == agent_id:
                return peer
        return None

    def get_peer_by_addr(self, addr: str) -> PeerInfo | None:
        """Find a peer by address."""
        return self._peers.get(addr)

    def peers_for_gossip(self) -> list[dict[str, Any]]:
        """Return peer list for gossip exchange (only connected or recently seen)."""
        result = []
        for peer in self._peers.values():
            if peer.agent_id and (peer.connected or not peer.is_stale):
                result.append({
                    "host": peer.host,
                    "port": peer.port,
                    "agent_id": peer.agent_id,
                    "agent_name": peer.agent_name,
                })
        return result

    def remove_stale(self) -> int:
        """Remove peers that have been stale and never connected. Returns count removed."""
        stale = [
            addr for addr, p in self._peers.items()
            if p.is_stale and not p.connected and p.failed_attempts >= 3
        ]
        for addr in stale:
            del self._peers[addr]
        return len(stale)
