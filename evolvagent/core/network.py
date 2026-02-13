"""
P2P network server for EvolvAgent.

Each agent runs both a WebSocket server (accepting inbound connections)
and a WebSocket client (connecting to known peers). Peers exchange skill
catalogs, fetch full skill definitions, and share reputation feedback.

Uses the `websockets` library for async WebSocket communication.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from .learner import DynamicSkill
from .peer import PeerManager
from .protocol import (
    MAX_MESSAGE_SIZE,
    MSG_ERROR,
    MSG_HELLO,
    MSG_HELLO_ACK,
    MSG_PEER_LIST_REQUEST,
    MSG_PEER_LIST_RESPONSE,
    MSG_PING,
    MSG_PONG,
    MSG_SKILL_CATALOG_REQUEST,
    MSG_SKILL_CATALOG_RESPONSE,
    MSG_SKILL_FEEDBACK,
    MSG_SKILL_FETCH_REQUEST,
    MSG_SKILL_FETCH_RESPONSE,
    NetworkMessage,
    SkillSummary,
    compute_skill_hash,
    validate_message,
)
from .skill import SkillOrigin

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


class NetworkServer:
    """
    P2P network node for an EvolvAgent instance.

    Runs a WebSocket server for inbound connections and manages
    outbound connections to known peers.
    """

    def __init__(self, agent: Agent):
        self._agent = agent
        self._settings = agent.settings.network
        self._peer_manager = PeerManager(
            bus=agent.bus,
            agent_id=agent.agent_id,
            max_peers=self._settings.max_peers,
        )
        self._server: Any = None  # websockets server
        self._running = False
        self._gossip_task: asyncio.Task | None = None
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._rate_limiter: dict[str, list[float]] = {}  # addr -> [timestamps]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        ssl_context: Any = None,
    ) -> None:
        """Start the WebSocket server and connect to seed peers."""
        if self._running:
            return

        try:
            import websockets
        except ImportError:
            raise RuntimeError(
                "websockets package required for P2P networking. "
                "Install with: pip install websockets"
            )

        listen_port = port or self._settings.listen_port

        serve_kwargs: dict[str, Any] = {}
        if ssl_context:
            serve_kwargs["ssl"] = ssl_context

        self._server = await websockets.serve(
            self._handle_inbound,
            host,
            listen_port,
            **serve_kwargs,
        )
        self._running = True

        # Add seed peers
        for seed in self._settings.seed_peers:
            if ":" in seed:
                h, p = seed.rsplit(":", 1)
                try:
                    self._peer_manager.add_seed(h, int(p))
                except ValueError:
                    logger.warning("Invalid seed peer: %s", seed)

        # Start gossip loop
        self._gossip_task = asyncio.create_task(self._gossip_loop())

        # Connect to seed peers in background
        asyncio.create_task(self._connect_to_seeds())

        await self._agent.bus.emit_async("network.started", {
            "host": host,
            "port": listen_port,
            "agent_id": self._agent.agent_id,
        }, source="network")

        logger.info("Network server started on %s:%d", host, listen_port)

    async def stop(self) -> None:
        """Stop the server and disconnect all peers."""
        if not self._running:
            return

        self._running = False

        # Cancel gossip
        if self._gossip_task:
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError:
                pass
            self._gossip_task = None

        # Cancel pending requests
        for fut in self._pending_requests.values():
            if not fut.done():
                fut.cancel()
        self._pending_requests.clear()

        # Close all peer connections with timeout
        for peer in self._peer_manager.connected_peers:
            if peer.connection:
                try:
                    await asyncio.wait_for(peer.connection.close(), timeout=3)
                except (asyncio.TimeoutError, Exception):
                    pass
            self._peer_manager.mark_disconnected(peer.address)

        # Close server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Persist known peers
        if self._agent._store:
            for peer in self._peer_manager.known_peers:
                if peer.agent_id:
                    self._agent._store.save_peer(
                        address=peer.address,
                        agent_id=peer.agent_id,
                        agent_name=peer.agent_name,
                        last_seen=peer.last_seen,
                    )

        await self._agent.bus.emit_async("network.stopped", {
            "agent_id": self._agent.agent_id,
        }, source="network")

        logger.info("Network server stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def peer_manager(self) -> PeerManager:
        return self._peer_manager

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect_to_peer(
        self, host: str, port: int, ssl_context: Any = None,
    ) -> bool:
        """Connect to a peer as a client."""
        try:
            import websockets
        except ImportError:
            return False

        addr = f"{host}:{port}"
        existing = self._peer_manager.get_peer_by_addr(addr)
        if existing and existing.connected:
            return True

        try:
            scheme = "wss" if ssl_context else "ws"
            connect_kwargs: dict[str, Any] = {}
            if ssl_context:
                connect_kwargs["ssl"] = ssl_context

            ws = await asyncio.wait_for(
                websockets.connect(f"{scheme}://{host}:{port}", **connect_kwargs),
                timeout=self._settings.connection_timeout,
            )

            # Send hello
            hello = NetworkMessage(
                type=MSG_HELLO,
                sender_id=self._agent.agent_id,
                payload={
                    "agent_name": self._agent.name,
                    "version": "0.4.0",
                },
            )
            await self._send(ws, hello)

            # Wait for hello_ack
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            ack = NetworkMessage.from_json(raw)

            if ack.type != MSG_HELLO_ACK:
                try:
                    await asyncio.wait_for(ws.close(), timeout=3)
                except (asyncio.TimeoutError, Exception):
                    pass
                return False

            peer_agent_id = ack.sender_id
            peer_name = ack.payload.get("agent_name", "")

            self._peer_manager.mark_connected(addr, peer_agent_id, peer_name, ws)

            # Start listening for messages from this peer
            asyncio.create_task(self._listen_peer(ws, addr))

            logger.info("Connected to peer: %s (%s)", peer_name or peer_agent_id, addr)
            return True

        except Exception as e:
            self._peer_manager.mark_failed(addr)
            logger.debug("Failed to connect to %s: %s", addr, e)
            return False

    async def _connect_to_seeds(self) -> None:
        """Connect to all seed peers."""
        for peer in self._peer_manager.known_peers:
            if not peer.connected:
                await self.connect_to_peer(peer.host, peer.port)

    async def _handle_inbound(self, ws: Any) -> None:
        """Handle a new inbound WebSocket connection."""
        addr = ""
        try:
            # Wait for hello message
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = NetworkMessage.from_json(raw)

            if msg.type != MSG_HELLO:
                err = NetworkMessage(
                    type=MSG_ERROR,
                    sender_id=self._agent.agent_id,
                    payload={"error": "Expected hello message"},
                )
                await self._send(ws, err)
                return

            peer_agent_id = msg.sender_id
            peer_name = msg.payload.get("agent_name", "")

            # Determine peer address from the WebSocket
            remote = ws.remote_address
            if remote:
                addr = f"{remote[0]}:{remote[1]}"
            else:
                addr = f"unknown:{id(ws)}"

            # Send hello_ack
            ack = NetworkMessage(
                type=MSG_HELLO_ACK,
                sender_id=self._agent.agent_id,
                payload={
                    "agent_name": self._agent.name,
                    "version": "0.4.0",
                },
            )
            await self._send(ws, ack)

            self._peer_manager.mark_connected(addr, peer_agent_id, peer_name, ws)

            # Listen for messages
            async for raw_msg in ws:
                if not self._running:
                    break
                # Size check
                if len(raw_msg) > MAX_MESSAGE_SIZE:
                    logger.warning("Oversized message from %s (%d bytes), disconnecting",
                                   addr, len(raw_msg))
                    break
                # Rate limit
                if not self._check_rate_limit(addr):
                    logger.warning("Rate limit exceeded for %s, disconnecting", addr)
                    break
                try:
                    message = NetworkMessage.from_json(raw_msg)
                    err = validate_message(message)
                    if err:
                        logger.warning("Invalid message from %s: %s", addr, err)
                        continue
                    await self._handle_message(message, ws, addr)
                except ValueError as e:
                    logger.warning("Malformed message from %s: %s", addr, e)
                except Exception as e:
                    logger.warning("Error handling message from %s: %s", addr, e)

        except Exception as e:
            logger.debug("Inbound connection error: %s", e)
        finally:
            if addr:
                self._peer_manager.mark_disconnected(addr)

    async def _listen_peer(self, ws: Any, addr: str) -> None:
        """Listen for messages from an outbound peer connection."""
        try:
            async for raw_msg in ws:
                if not self._running:
                    break
                if len(raw_msg) > MAX_MESSAGE_SIZE:
                    logger.warning("Oversized message from %s (%d bytes), disconnecting",
                                   addr, len(raw_msg))
                    break
                if not self._check_rate_limit(addr):
                    logger.warning("Rate limit exceeded for %s, disconnecting", addr)
                    break
                try:
                    message = NetworkMessage.from_json(raw_msg)
                    err = validate_message(message)
                    if err:
                        logger.warning("Invalid message from %s: %s", addr, err)
                        continue
                    # Check if this is a response to a pending request
                    req_id = message.request_id
                    if req_id in self._pending_requests:
                        fut = self._pending_requests.pop(req_id)
                        if not fut.done():
                            fut.set_result(message)
                    else:
                        await self._handle_message(message, ws, addr)
                except ValueError as e:
                    logger.warning("Malformed message from %s: %s", addr, e)
                except Exception as e:
                    logger.warning("Error handling message from %s: %s", addr, e)
        except Exception as e:
            logger.debug("Peer connection lost: %s (%s)", addr, e)
        finally:
            self._peer_manager.mark_disconnected(addr)

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _handle_message(
        self,
        msg: NetworkMessage,
        ws: Any,
        addr: str,
    ) -> None:
        """Route incoming message to appropriate handler."""
        handlers = {
            MSG_HELLO: self._handle_hello,
            MSG_PING: self._handle_ping,
            MSG_SKILL_CATALOG_REQUEST: self._handle_skill_catalog_request,
            MSG_SKILL_FETCH_REQUEST: self._handle_skill_fetch_request,
            MSG_SKILL_FEEDBACK: self._handle_skill_feedback,
            MSG_PEER_LIST_REQUEST: self._handle_peer_list_request,
        }

        handler = handlers.get(msg.type)
        if handler:
            await handler(msg, ws, addr)
        else:
            logger.debug("Unhandled message type: %s from %s", msg.type, addr)

    async def _handle_hello(
        self, msg: NetworkMessage, ws: Any, addr: str
    ) -> None:
        """Handle duplicate hello (already connected)."""
        ack = NetworkMessage(
            type=MSG_HELLO_ACK,
            sender_id=self._agent.agent_id,
            request_id=msg.request_id,
            payload={"agent_name": self._agent.name, "version": "0.4.0"},
        )
        await self._send(ws, ack)

    async def _handle_ping(
        self, msg: NetworkMessage, ws: Any, addr: str
    ) -> None:
        """Respond to heartbeat."""
        pong = NetworkMessage(
            type=MSG_PONG,
            sender_id=self._agent.agent_id,
            request_id=msg.request_id,
        )
        await self._send(ws, pong)
        # Update last_seen
        peer = self._peer_manager.get_peer_by_addr(addr)
        if peer:
            peer.last_seen = time.time()

    async def _handle_skill_catalog_request(
        self, msg: NetworkMessage, ws: Any, addr: str
    ) -> None:
        """Respond with our skill catalog."""
        query = msg.payload.get("query", "").lower()
        summaries = self._build_catalog(query)

        response = NetworkMessage(
            type=MSG_SKILL_CATALOG_RESPONSE,
            sender_id=self._agent.agent_id,
            request_id=msg.request_id,
            payload={"skills": [s.to_dict() for s in summaries]},
        )
        await self._send(ws, response)

        await self._agent.bus.emit_async("network.skill_shared", {
            "peer_address": addr,
            "skills_count": len(summaries),
        }, source="network")

    async def _handle_skill_fetch_request(
        self, msg: NetworkMessage, ws: Any, addr: str
    ) -> None:
        """Respond with full skill definition."""
        skill_name = msg.payload.get("skill_name", "")
        skill = self._agent.get_skill(skill_name)

        if not skill or not isinstance(skill, DynamicSkill):
            response = NetworkMessage(
                type=MSG_ERROR,
                sender_id=self._agent.agent_id,
                request_id=msg.request_id,
                payload={"error": f"Skill '{skill_name}' not found or not shareable"},
            )
            await self._send(ws, response)
            return

        # Check sharing policy
        if skill.metadata.origin == SkillOrigin.BUILTIN:
            if not self._settings.share_builtin_skills:
                response = NetworkMessage(
                    type=MSG_ERROR,
                    sender_id=self._agent.agent_id,
                    request_id=msg.request_id,
                    payload={"error": f"Skill '{skill_name}' sharing not allowed"},
                )
                await self._send(ws, response)
                return

        defn = skill.to_definition()
        defn["content_hash"] = compute_skill_hash(defn)

        response = NetworkMessage(
            type=MSG_SKILL_FETCH_RESPONSE,
            sender_id=self._agent.agent_id,
            request_id=msg.request_id,
            payload={"skill_name": skill_name, "definition": defn},
        )
        await self._send(ws, response)

    async def _handle_skill_feedback(
        self, msg: NetworkMessage, ws: Any, addr: str
    ) -> None:
        """Handle reputation feedback for a skill."""
        skill_name = msg.payload.get("skill_name", "")
        rating = msg.payload.get("rating", 0.0)
        peer_id = msg.sender_id

        if self._agent._store:
            self._agent._store.add_peer_rating(
                skill_name=skill_name,
                source_agent=self._agent.agent_id,
                peer_agent=peer_id,
                rating=rating,
            )

        logger.info("Received feedback for '%s' from %s: %.1f", skill_name, peer_id, rating)

    async def _handle_peer_list_request(
        self, msg: NetworkMessage, ws: Any, addr: str
    ) -> None:
        """Respond with known peer list for gossip."""
        peers = self._peer_manager.peers_for_gossip()
        response = NetworkMessage(
            type=MSG_PEER_LIST_RESPONSE,
            sender_id=self._agent.agent_id,
            request_id=msg.request_id,
            payload={"peers": peers},
        )
        await self._send(ws, response)

    # ------------------------------------------------------------------
    # High-level API (called by CLI / Agent)
    # ------------------------------------------------------------------

    async def browse_skills(
        self, peer_id: str = "", query: str = ""
    ) -> list[SkillSummary]:
        """Browse skills from connected peers."""
        results = []

        peers = self._peer_manager.connected_peers
        if peer_id:
            peer = self._peer_manager.get_peer(peer_id)
            peers = [peer] if peer and peer.connected else []

        for peer in peers:
            if not peer.connection:
                continue
            try:
                msg = NetworkMessage(
                    type=MSG_SKILL_CATALOG_REQUEST,
                    sender_id=self._agent.agent_id,
                    payload={"query": query},
                )
                response = await self._send_and_wait(peer.connection, msg, timeout=10)
                if response and response.type == MSG_SKILL_CATALOG_RESPONSE:
                    skills = response.payload.get("skills", [])
                    for s in skills:
                        results.append(SkillSummary.from_dict(s))
                    # Cache in peer info
                    peer.skills_cached = skills
            except Exception as e:
                logger.warning("Failed to browse skills from %s: %s", peer.address, e)

        return results

    async def fetch_skill(self, peer_id: str, skill_name: str) -> dict[str, Any] | None:
        """Fetch a complete skill definition from a peer."""
        peer = self._peer_manager.get_peer(peer_id)
        if not peer or not peer.connection:
            return None

        try:
            msg = NetworkMessage(
                type=MSG_SKILL_FETCH_REQUEST,
                sender_id=self._agent.agent_id,
                payload={"skill_name": skill_name},
            )
            response = await self._send_and_wait(peer.connection, msg, timeout=10)
            if response and response.type == MSG_SKILL_FETCH_RESPONSE:
                return response.payload.get("definition")
        except Exception as e:
            logger.warning("Failed to fetch skill '%s' from %s: %s", skill_name, peer_id, e)

        return None

    async def send_feedback(
        self, peer_id: str, skill_name: str, rating: float
    ) -> None:
        """Send reputation feedback to a peer about one of their skills."""
        peer = self._peer_manager.get_peer(peer_id)
        if not peer or not peer.connection:
            return

        msg = NetworkMessage(
            type=MSG_SKILL_FEEDBACK,
            sender_id=self._agent.agent_id,
            payload={"skill_name": skill_name, "rating": rating},
        )
        await self._send(peer.connection, msg)

    # ------------------------------------------------------------------
    # Gossip
    # ------------------------------------------------------------------

    async def _gossip_loop(self) -> None:
        """Periodically exchange peer lists with connected peers."""
        while self._running:
            try:
                await asyncio.sleep(self._settings.gossip_interval)
                if not self._running:
                    break
                await self._gossip_round()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Gossip round failed: %s", e)

    async def _gossip_round(self) -> None:
        """Single gossip round: ask each connected peer for their peer list."""
        for peer in self._peer_manager.connected_peers:
            if not peer.connection:
                continue
            try:
                msg = NetworkMessage(
                    type=MSG_PEER_LIST_REQUEST,
                    sender_id=self._agent.agent_id,
                )
                response = await self._send_and_wait(peer.connection, msg, timeout=5)
                if response and response.type == MSG_PEER_LIST_RESPONSE:
                    new_peers = response.payload.get("peers", [])
                    self._peer_manager.add_peers_from_gossip(new_peers)
            except Exception as e:
                logger.debug("Gossip failed with %s: %s", peer.address, e)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self, addr: str) -> bool:
        """Check if a peer is within rate limits. Returns False if over limit."""
        now = time.time()
        window = 60.0  # 1 minute sliding window
        limit = self._settings.rate_limit_per_min

        timestamps = self._rate_limiter.get(addr, [])
        # Prune old entries
        cutoff = now - window
        timestamps = [t for t in timestamps if t > cutoff]
        timestamps.append(now)
        self._rate_limiter[addr] = timestamps

        return len(timestamps) <= limit

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_catalog(self, query: str = "") -> list[SkillSummary]:
        """Build skill catalog from local skills."""
        summaries = []
        for skill in self._agent.active_skills:
            # Check sharing policy
            if skill.metadata.origin == SkillOrigin.BUILTIN:
                if not self._settings.share_builtin_skills:
                    continue
            elif skill.metadata.origin == SkillOrigin.LEARNED:
                if not self._settings.share_learned_skills:
                    continue

            # Only share DynamicSkills (they have transferable definitions)
            if not isinstance(skill, DynamicSkill):
                continue

            summary = SkillSummary.from_metadata(
                skill.metadata,
                agent_id=self._agent.agent_id,
                agent_name=self._agent.name,
            )

            # Filter by query if provided
            if query:
                searchable = f"{summary.name} {summary.description} {' '.join(summary.tags)}"
                if query not in searchable.lower():
                    continue

            summaries.append(summary)
        return summaries

    async def _send(self, ws: Any, msg: NetworkMessage) -> None:
        """Send a message over a WebSocket connection."""
        msg.sender_id = self._agent.agent_id
        await ws.send(msg.to_json())

    async def _send_and_wait(
        self, ws: Any, msg: NetworkMessage, timeout: float = 10
    ) -> NetworkMessage | None:
        """Send a message and wait for a response with matching request_id."""
        msg.sender_id = self._agent.agent_id

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[NetworkMessage] = loop.create_future()
        self._pending_requests[msg.request_id] = fut

        try:
            await ws.send(msg.to_json())
            return await asyncio.wait_for(fut, timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._pending_requests.pop(msg.request_id, None)
            return None
        except Exception:
            self._pending_requests.pop(msg.request_id, None)
            return None
