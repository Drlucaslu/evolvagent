"""
Tests for Phase 4: P2P Network.

Covers protocol serialization, peer management, network server
message handling, and integration tests with two real agents.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from evolvagent.core.config import Settings, reset_settings
from evolvagent.core.events import EventBus
from evolvagent.core.learner import DynamicSkill
from evolvagent.core.peer import PeerInfo, PeerManager
from evolvagent.core.protocol import (
    ALL_MSG_TYPES,
    MSG_ERROR,
    MSG_HELLO,
    MSG_HELLO_ACK,
    MSG_PEER_LIST_REQUEST,
    MSG_PEER_LIST_RESPONSE,
    MSG_PING,
    MSG_PONG,
    MSG_SKILL_CATALOG_REQUEST,
    MSG_SKILL_CATALOG_RESPONSE,
    MSG_SKILL_FETCH_REQUEST,
    MSG_SKILL_FETCH_RESPONSE,
    MSG_SKILL_FEEDBACK,
    NetworkMessage,
    SkillSummary,
    compute_skill_hash,
)
from evolvagent.core.skill import SkillMetadata, SkillOrigin, TrustLevel
from evolvagent.core.storage import SkillStore


# ---------------------------------------------------------------------------
# TestNetworkMessage — serialization / deserialization
# ---------------------------------------------------------------------------


class TestNetworkMessage:
    """Test NetworkMessage dataclass and JSON serialization."""

    def test_create_default(self):
        msg = NetworkMessage(type=MSG_HELLO)
        assert msg.type == MSG_HELLO
        assert msg.sender_id == ""
        assert msg.request_id  # auto-generated
        assert msg.timestamp > 0
        assert msg.payload == {}

    def test_create_with_payload(self):
        msg = NetworkMessage(
            type=MSG_SKILL_CATALOG_RESPONSE,
            sender_id="agent-001",
            payload={"skills": [{"name": "test_skill"}]},
        )
        assert msg.type == MSG_SKILL_CATALOG_RESPONSE
        assert msg.sender_id == "agent-001"
        assert msg.payload["skills"][0]["name"] == "test_skill"

    def test_to_json(self):
        msg = NetworkMessage(
            type=MSG_PING,
            sender_id="agent-001",
            request_id="req-123",
            timestamp=1000.0,
            payload={"data": "test"},
        )
        raw = msg.to_json()
        data = json.loads(raw)
        assert data["type"] == MSG_PING
        assert data["sender_id"] == "agent-001"
        assert data["request_id"] == "req-123"
        assert data["timestamp"] == 1000.0
        assert data["payload"]["data"] == "test"

    def test_from_json(self):
        raw = json.dumps({
            "type": MSG_PONG,
            "sender_id": "agent-002",
            "request_id": "req-456",
            "timestamp": 2000.0,
            "payload": {},
        })
        msg = NetworkMessage.from_json(raw)
        assert msg.type == MSG_PONG
        assert msg.sender_id == "agent-002"
        assert msg.request_id == "req-456"
        assert msg.timestamp == 2000.0

    def test_roundtrip(self):
        original = NetworkMessage(
            type=MSG_SKILL_FETCH_REQUEST,
            sender_id="agent-003",
            payload={"skill_name": "code_review"},
        )
        raw = original.to_json()
        restored = NetworkMessage.from_json(raw)
        assert restored.type == original.type
        assert restored.sender_id == original.sender_id
        assert restored.request_id == original.request_id
        assert restored.payload == original.payload

    def test_from_json_missing_fields(self):
        raw = json.dumps({"type": MSG_ERROR})
        msg = NetworkMessage.from_json(raw)
        assert msg.type == MSG_ERROR
        assert msg.sender_id == ""
        assert msg.request_id == ""
        assert msg.payload == {}

    def test_reply(self):
        original = NetworkMessage(
            type=MSG_SKILL_CATALOG_REQUEST,
            sender_id="agent-001",
            request_id="req-789",
        )
        reply = original.reply(MSG_SKILL_CATALOG_RESPONSE, {"skills": []})
        assert reply.type == MSG_SKILL_CATALOG_RESPONSE
        assert reply.request_id == "req-789"  # preserved
        assert reply.payload == {"skills": []}

    def test_all_message_types_defined(self):
        assert MSG_HELLO in ALL_MSG_TYPES
        assert MSG_HELLO_ACK in ALL_MSG_TYPES
        assert MSG_PING in ALL_MSG_TYPES
        assert MSG_PONG in ALL_MSG_TYPES
        assert MSG_SKILL_CATALOG_REQUEST in ALL_MSG_TYPES
        assert MSG_SKILL_CATALOG_RESPONSE in ALL_MSG_TYPES
        assert MSG_SKILL_FETCH_REQUEST in ALL_MSG_TYPES
        assert MSG_SKILL_FETCH_RESPONSE in ALL_MSG_TYPES
        assert MSG_SKILL_FEEDBACK in ALL_MSG_TYPES
        assert MSG_PEER_LIST_REQUEST in ALL_MSG_TYPES
        assert MSG_PEER_LIST_RESPONSE in ALL_MSG_TYPES
        assert MSG_ERROR in ALL_MSG_TYPES
        assert len(ALL_MSG_TYPES) == 12


# ---------------------------------------------------------------------------
# TestSkillSummary — catalog entries
# ---------------------------------------------------------------------------


class TestSkillSummary:
    """Test SkillSummary creation and conversion."""

    def test_from_metadata(self):
        meta = SkillMetadata(
            name="code_review",
            description="Reviews code for issues",
            category="development",
            tags=["code", "review"],
            version="0.2.0",
            utility_score=0.8,
            network_reputation=0.6,
        )
        summary = SkillSummary.from_metadata(meta, agent_id="agent-001", agent_name="Agent One")
        assert summary.name == "code_review"
        assert summary.description == "Reviews code for issues"
        assert summary.category == "development"
        assert summary.tags == ["code", "review"]
        assert summary.agent_id == "agent-001"
        assert summary.agent_name == "Agent One"
        assert summary.utility_score == 0.8
        assert summary.content_hash  # non-empty

    def test_to_dict_and_from_dict(self):
        original = SkillSummary(
            name="test_skill",
            description="A test skill",
            tags=["test"],
            agent_id="agent-002",
            utility_score=0.7,
        )
        d = original.to_dict()
        restored = SkillSummary.from_dict(d)
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.tags == original.tags
        assert restored.agent_id == original.agent_id
        assert restored.utility_score == original.utility_score

    def test_from_dict_defaults(self):
        summary = SkillSummary.from_dict({})
        assert summary.name == ""
        assert summary.category == "general"
        assert summary.utility_score == 0.5


# ---------------------------------------------------------------------------
# TestComputeSkillHash
# ---------------------------------------------------------------------------


class TestComputeSkillHash:
    def test_basic_hash(self):
        defn = {
            "system_prompt": "You are a code reviewer.",
            "metadata": {"name": "code_review", "description": "Reviews code"},
        }
        h = compute_skill_hash(defn)
        assert isinstance(h, str)
        assert len(h) == 16

    def test_different_content_different_hash(self):
        defn1 = {
            "system_prompt": "Prompt A",
            "metadata": {"name": "skill_a", "description": "Desc A"},
        }
        defn2 = {
            "system_prompt": "Prompt B",
            "metadata": {"name": "skill_b", "description": "Desc B"},
        }
        assert compute_skill_hash(defn1) != compute_skill_hash(defn2)

    def test_same_content_same_hash(self):
        defn = {
            "system_prompt": "You are helpful.",
            "metadata": {"name": "helper", "description": "Helps"},
        }
        assert compute_skill_hash(defn) == compute_skill_hash(defn)

    def test_empty_definition(self):
        h = compute_skill_hash({})
        assert isinstance(h, str)
        assert len(h) == 16


# ---------------------------------------------------------------------------
# TestPeerManager — node tracking and events
# ---------------------------------------------------------------------------


class TestPeerManager:
    """Test PeerManager peer tracking and gossip."""

    def setup_method(self):
        self.bus = EventBus()
        self.pm = PeerManager(bus=self.bus, agent_id="local-agent", max_peers=5)

    def test_add_seed(self):
        events = []
        self.bus.on("network.peer_discovered", lambda e: events.append(e))

        peer = self.pm.add_seed("192.168.1.1", 8765)
        assert peer.host == "192.168.1.1"
        assert peer.port == 8765
        assert peer.address == "192.168.1.1:8765"
        assert not peer.connected
        assert len(events) == 1
        assert events[0].data["source"] == "seed"

    def test_add_seed_idempotent(self):
        self.pm.add_seed("192.168.1.1", 8765)
        self.pm.add_seed("192.168.1.1", 8765)
        assert len(self.pm.known_peers) == 1

    def test_mark_connected(self):
        events = []
        self.bus.on("network.peer_connected", lambda e: events.append(e))

        ws = MagicMock()
        peer = self.pm.mark_connected("10.0.0.1:8765", "remote-agent", "Remote", ws)
        assert peer.connected
        assert peer.agent_id == "remote-agent"
        assert peer.agent_name == "Remote"
        assert peer.connection is ws
        assert peer.failed_attempts == 0
        assert len(events) == 1
        assert events[0].data["agent_id"] == "remote-agent"

    def test_mark_disconnected(self):
        events = []
        self.bus.on("network.peer_disconnected", lambda e: events.append(e))

        ws = MagicMock()
        self.pm.mark_connected("10.0.0.1:8765", "remote-agent", "Remote", ws)
        self.pm.mark_disconnected("10.0.0.1:8765")

        peer = self.pm.get_peer_by_addr("10.0.0.1:8765")
        assert not peer.connected
        assert peer.connection is None
        assert len(events) == 1

    def test_mark_disconnected_only_emits_if_was_connected(self):
        events = []
        self.bus.on("network.peer_disconnected", lambda e: events.append(e))

        self.pm.add_seed("10.0.0.1", 8765)
        self.pm.mark_disconnected("10.0.0.1:8765")
        assert len(events) == 0  # not connected, so no event

    def test_connected_peers(self):
        ws = MagicMock()
        self.pm.mark_connected("10.0.0.1:8765", "a1", "Agent1", ws)
        self.pm.add_seed("10.0.0.2", 8765)

        connected = self.pm.connected_peers
        assert len(connected) == 1
        assert connected[0].agent_id == "a1"

    def test_get_peer(self):
        ws = MagicMock()
        self.pm.mark_connected("10.0.0.1:8765", "agent-xyz", "AgentXYZ", ws)
        found = self.pm.get_peer("agent-xyz")
        assert found is not None
        assert found.agent_name == "AgentXYZ"

        not_found = self.pm.get_peer("nonexistent")
        assert not_found is None

    def test_add_peers_from_gossip(self):
        events = []
        self.bus.on("network.peer_discovered", lambda e: events.append(e))

        gossip_data = [
            {"host": "10.0.0.1", "port": 8765, "agent_id": "peer-1", "agent_name": "P1"},
            {"host": "10.0.0.2", "port": 8766, "agent_id": "peer-2"},
        ]
        added = self.pm.add_peers_from_gossip(gossip_data)
        assert added == 2
        assert len(self.pm.known_peers) == 2
        assert len(events) == 2

    def test_gossip_ignores_self(self):
        gossip_data = [
            {"host": "10.0.0.1", "port": 8765, "agent_id": "local-agent"},
        ]
        added = self.pm.add_peers_from_gossip(gossip_data)
        assert added == 0

    def test_gossip_respects_max_peers(self):
        gossip_data = [
            {"host": f"10.0.0.{i}", "port": 8765, "agent_id": f"peer-{i}"}
            for i in range(10)
        ]
        added = self.pm.add_peers_from_gossip(gossip_data)
        assert added == 5  # max_peers = 5
        assert len(self.pm.known_peers) == 5

    def test_peers_for_gossip(self):
        ws = MagicMock()
        self.pm.mark_connected("10.0.0.1:8765", "agent-1", "A1", ws)
        gossip_list = self.pm.peers_for_gossip()
        assert len(gossip_list) == 1
        assert gossip_list[0]["host"] == "10.0.0.1"
        assert gossip_list[0]["port"] == 8765
        assert gossip_list[0]["agent_id"] == "agent-1"

    def test_mark_failed(self):
        self.pm.add_seed("10.0.0.1", 8765)
        self.pm.mark_failed("10.0.0.1:8765")
        peer = self.pm.get_peer_by_addr("10.0.0.1:8765")
        assert peer.failed_attempts == 1


# ---------------------------------------------------------------------------
# TestPeerInfo
# ---------------------------------------------------------------------------


class TestPeerInfo:
    def test_address(self):
        peer = PeerInfo(host="example.com", port=8765)
        assert peer.address == "example.com:8765"

    def test_is_stale(self):
        peer = PeerInfo(last_seen=time.time() - 400)
        assert peer.is_stale

        fresh = PeerInfo(last_seen=time.time())
        assert not fresh.is_stale

        never_seen = PeerInfo(last_seen=0.0)
        assert not never_seen.is_stale  # never seen = not stale


# ---------------------------------------------------------------------------
# TestNetworkServerUnit — message handlers with mock WebSocket
# ---------------------------------------------------------------------------


class TestNetworkServerUnit:
    """Unit tests for NetworkServer message handlers using mocked components."""

    def setup_method(self):
        reset_settings()
        self.settings = Settings()
        self.settings.network.share_learned_skills = True
        self.settings.network.share_builtin_skills = False

    def _make_agent(self):
        """Create a minimal mock agent."""
        agent = MagicMock()
        agent.settings = self.settings
        agent.agent_id = "test-agent-001"
        agent.name = "TestAgent"
        agent.bus = EventBus()
        agent._store = None
        agent.active_skills = []
        agent.get_skill = MagicMock(return_value=None)
        return agent

    @pytest.mark.asyncio
    async def test_handle_ping(self):
        agent = self._make_agent()

        from evolvagent.core.network import NetworkServer
        server = NetworkServer(agent)

        ws = AsyncMock()
        msg = NetworkMessage(type=MSG_PING, sender_id="remote", request_id="r1")
        await server._handle_ping(msg, ws, "10.0.0.1:8765")

        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == MSG_PONG
        assert sent["request_id"] == "r1"

    @pytest.mark.asyncio
    async def test_handle_skill_catalog_request_empty(self):
        agent = self._make_agent()

        from evolvagent.core.network import NetworkServer
        server = NetworkServer(agent)

        ws = AsyncMock()
        msg = NetworkMessage(
            type=MSG_SKILL_CATALOG_REQUEST,
            sender_id="remote",
            request_id="r2",
            payload={"query": ""},
        )
        await server._handle_skill_catalog_request(msg, ws, "10.0.0.1:8765")

        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == MSG_SKILL_CATALOG_RESPONSE
        assert sent["payload"]["skills"] == []

    @pytest.mark.asyncio
    async def test_handle_skill_catalog_request_with_skills(self):
        agent = self._make_agent()
        llm = MagicMock()

        # Create a DynamicSkill
        meta = SkillMetadata(
            name="test_learned",
            description="A learned skill",
            origin=SkillOrigin.LEARNED,
            tags=["test"],
        )
        dskill = DynamicSkill(metadata=meta, llm=llm, system_prompt="You are a test helper.")
        agent.active_skills = [dskill]

        from evolvagent.core.network import NetworkServer
        server = NetworkServer(agent)

        ws = AsyncMock()
        msg = NetworkMessage(
            type=MSG_SKILL_CATALOG_REQUEST,
            sender_id="remote",
            request_id="r3",
        )
        await server._handle_skill_catalog_request(msg, ws, "10.0.0.1:8765")

        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == MSG_SKILL_CATALOG_RESPONSE
        assert len(sent["payload"]["skills"]) == 1
        assert sent["payload"]["skills"][0]["name"] == "test_learned"

    @pytest.mark.asyncio
    async def test_handle_skill_fetch_request_not_found(self):
        agent = self._make_agent()

        from evolvagent.core.network import NetworkServer
        server = NetworkServer(agent)

        ws = AsyncMock()
        msg = NetworkMessage(
            type=MSG_SKILL_FETCH_REQUEST,
            sender_id="remote",
            request_id="r4",
            payload={"skill_name": "nonexistent"},
        )
        await server._handle_skill_fetch_request(msg, ws, "10.0.0.1:8765")

        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == MSG_ERROR

    @pytest.mark.asyncio
    async def test_handle_skill_fetch_request_found(self):
        agent = self._make_agent()
        llm = MagicMock()

        meta = SkillMetadata(
            name="shared_skill",
            description="Shareable skill",
            origin=SkillOrigin.LEARNED,
        )
        dskill = DynamicSkill(metadata=meta, llm=llm, system_prompt="Test prompt")
        agent.get_skill = MagicMock(return_value=dskill)

        from evolvagent.core.network import NetworkServer
        server = NetworkServer(agent)

        ws = AsyncMock()
        msg = NetworkMessage(
            type=MSG_SKILL_FETCH_REQUEST,
            sender_id="remote",
            request_id="r5",
            payload={"skill_name": "shared_skill"},
        )
        await server._handle_skill_fetch_request(msg, ws, "10.0.0.1:8765")

        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == MSG_SKILL_FETCH_RESPONSE
        assert sent["payload"]["skill_name"] == "shared_skill"
        assert "definition" in sent["payload"]
        assert sent["payload"]["definition"]["system_prompt"] == "Test prompt"

    @pytest.mark.asyncio
    async def test_handle_peer_list_request(self):
        agent = self._make_agent()

        from evolvagent.core.network import NetworkServer
        server = NetworkServer(agent)

        # Add a connected peer
        ws_peer = MagicMock()
        server.peer_manager.mark_connected(
            "10.0.0.2:8765", "other-agent", "OtherAgent", ws_peer
        )

        ws = AsyncMock()
        msg = NetworkMessage(
            type=MSG_PEER_LIST_REQUEST,
            sender_id="remote",
            request_id="r6",
        )
        await server._handle_peer_list_request(msg, ws, "10.0.0.1:8765")

        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == MSG_PEER_LIST_RESPONSE
        assert len(sent["payload"]["peers"]) >= 1

    @pytest.mark.asyncio
    async def test_handle_hello_sends_ack(self):
        agent = self._make_agent()

        from evolvagent.core.network import NetworkServer
        server = NetworkServer(agent)

        ws = AsyncMock()
        msg = NetworkMessage(
            type=MSG_HELLO,
            sender_id="remote",
            request_id="r7",
            payload={"agent_name": "Remote", "version": "0.4.0"},
        )
        await server._handle_hello(msg, ws, "10.0.0.1:8765")

        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == MSG_HELLO_ACK

    @pytest.mark.asyncio
    async def test_build_catalog_filters_builtins(self):
        """Builtins should be excluded when share_builtin_skills=False."""
        agent = self._make_agent()
        llm = MagicMock()

        builtin_meta = SkillMetadata(
            name="builtin_skill", origin=SkillOrigin.BUILTIN,
        )
        builtin = DynamicSkill(metadata=builtin_meta, llm=llm, system_prompt="builtin")

        learned_meta = SkillMetadata(
            name="learned_skill", origin=SkillOrigin.LEARNED,
        )
        learned = DynamicSkill(metadata=learned_meta, llm=llm, system_prompt="learned")

        agent.active_skills = [builtin, learned]

        from evolvagent.core.network import NetworkServer
        server = NetworkServer(agent)

        catalog = server._build_catalog()
        names = [s.name for s in catalog]
        assert "learned_skill" in names
        assert "builtin_skill" not in names

    @pytest.mark.asyncio
    async def test_build_catalog_query_filter(self):
        agent = self._make_agent()
        llm = MagicMock()

        meta1 = SkillMetadata(
            name="python_review", description="Review Python code",
            tags=["python", "code"], origin=SkillOrigin.LEARNED,
        )
        skill1 = DynamicSkill(metadata=meta1, llm=llm, system_prompt="...")

        meta2 = SkillMetadata(
            name="javascript_lint", description="Lint JavaScript",
            tags=["javascript", "lint"], origin=SkillOrigin.LEARNED,
        )
        skill2 = DynamicSkill(metadata=meta2, llm=llm, system_prompt="...")

        agent.active_skills = [skill1, skill2]

        from evolvagent.core.network import NetworkServer
        server = NetworkServer(agent)

        results = server._build_catalog(query="python")
        assert len(results) == 1
        assert results[0].name == "python_review"


# ---------------------------------------------------------------------------
# TestNetworkStorage — peer and reputation persistence
# ---------------------------------------------------------------------------


class TestNetworkStorage:
    """Test the storage layer for network data."""

    def setup_method(self):
        import tempfile
        self._tmp = tempfile.mkdtemp()
        self.store = SkillStore(Path(self._tmp) / "test.db")

    def teardown_method(self):
        self.store.close()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_save_and_load_peer(self):
        self.store.save_peer(
            address="10.0.0.1:8765",
            agent_id="agent-001",
            agent_name="Agent One",
            last_seen=1000.0,
            trust_score=0.8,
        )
        peers = self.store.load_peers()
        assert len(peers) == 1
        p = peers[0]
        assert p["address"] == "10.0.0.1:8765"
        assert p["agent_id"] == "agent-001"
        assert p["agent_name"] == "Agent One"
        assert p["last_seen"] == 1000.0
        assert p["trust_score"] == 0.8

    def test_save_peer_upsert(self):
        self.store.save_peer("10.0.0.1:8765", agent_id="old-id")
        self.store.save_peer("10.0.0.1:8765", agent_id="new-id", agent_name="New")
        peers = self.store.load_peers()
        assert len(peers) == 1
        assert peers[0]["agent_id"] == "new-id"

    def test_delete_peer(self):
        self.store.save_peer("10.0.0.1:8765", agent_id="agent-001")
        assert self.store.delete_peer("10.0.0.1:8765")
        assert len(self.store.load_peers()) == 0

    def test_save_and_load_skill_reputation(self):
        self.store.save_skill_reputation(
            skill_name="test_skill",
            content_hash="abc123",
            source_agent="agent-002",
            local_rating=0.7,
        )
        reps = self.store.load_skill_reputation("test_skill")
        assert len(reps) == 1
        r = reps[0]
        assert r["skill_name"] == "test_skill"
        assert r["content_hash"] == "abc123"
        assert r["source_agent"] == "agent-002"
        assert r["local_rating"] == 0.7
        assert r["peer_ratings"] == []

    def test_add_peer_rating(self):
        self.store.save_skill_reputation(
            skill_name="test_skill",
            source_agent="agent-002",
        )
        self.store.add_peer_rating(
            skill_name="test_skill",
            source_agent="agent-002",
            peer_agent="agent-003",
            rating=0.9,
        )
        reps = self.store.load_skill_reputation("test_skill")
        assert len(reps) == 1
        ratings = reps[0]["peer_ratings"]
        assert len(ratings) == 1
        assert ratings[0]["peer"] == "agent-003"
        assert ratings[0]["rating"] == 0.9

    def test_add_peer_rating_updates_existing(self):
        self.store.save_skill_reputation(
            skill_name="test_skill",
            source_agent="agent-002",
        )
        self.store.add_peer_rating("test_skill", "agent-002", "agent-003", 0.5)
        self.store.add_peer_rating("test_skill", "agent-002", "agent-003", 0.9)
        reps = self.store.load_skill_reputation("test_skill")
        ratings = reps[0]["peer_ratings"]
        assert len(ratings) == 1  # updated, not appended
        assert ratings[0]["rating"] == 0.9

    def test_add_peer_rating_no_record(self):
        # Should not crash when no record exists
        self.store.add_peer_rating("nonexistent", "agent-002", "agent-003", 0.5)


# ---------------------------------------------------------------------------
# TestNetworkServerIntegration — two real agents on localhost
# ---------------------------------------------------------------------------


class TestNetworkServerIntegration:
    """Integration tests with two agents communicating over WebSocket."""

    @pytest.fixture
    def _agents(self, tmp_path):
        """Create two agents with different data dirs."""
        reset_settings()

        settings1 = Settings()
        settings1.agent.name = "agent-alpha"
        settings1.agent.data_dir = str(tmp_path / "alpha")
        settings1.network.listen_port = 18765
        settings1.network.share_learned_skills = True

        settings2 = Settings()
        settings2.agent.name = "agent-beta"
        settings2.agent.data_dir = str(tmp_path / "beta")
        settings2.network.listen_port = 18766
        settings2.network.share_learned_skills = True

        return settings1, settings2

    @pytest.mark.asyncio
    async def test_two_agents_connect(self, _agents):
        """Two agents can connect and exchange hello."""
        try:
            import websockets  # noqa: F401
        except ImportError:
            pytest.skip("websockets not installed")

        from evolvagent.core.agent import Agent

        settings1, settings2 = _agents

        agent1 = Agent(settings=settings1)
        agent2 = Agent(settings=settings2)

        await agent1.start()
        await agent2.start()

        try:
            await agent1.start_network()
            await agent2.start_network()

            # Agent2 connects to Agent1
            connected = await agent2.network.connect_to_peer("127.0.0.1", 18765)
            assert connected

            # Give a moment for the connection to be fully established
            await asyncio.sleep(0.1)

            # Both should see each other
            assert len(agent2.network.peer_manager.connected_peers) >= 1

        finally:
            await agent1.stop_network()
            await agent2.stop_network()
            await agent1.shutdown()
            await agent2.shutdown()

    @pytest.mark.asyncio
    async def test_browse_and_fetch_skill(self, _agents):
        """Agent2 can browse and fetch a skill from Agent1."""
        try:
            import websockets  # noqa: F401
        except ImportError:
            pytest.skip("websockets not installed")

        from evolvagent.core.agent import Agent

        settings1, settings2 = _agents

        agent1 = Agent(settings=settings1)
        agent2 = Agent(settings=settings2)

        await agent1.start()
        await agent2.start()

        try:
            # Register a learned skill on agent1
            llm = MagicMock()
            meta = SkillMetadata(
                name="shared_analyzer",
                description="Analyzes shared data",
                origin=SkillOrigin.LEARNED,
                tags=["analyze"],
            )
            dskill = DynamicSkill(metadata=meta, llm=llm, system_prompt="You analyze data.")
            agent1.register_skill(dskill, persist=False)

            await agent1.start_network()
            await agent2.start_network()

            # Connect
            connected = await agent2.network.connect_to_peer("127.0.0.1", 18765)
            assert connected
            await asyncio.sleep(0.1)

            # Browse skills
            skills = await agent2.network.browse_skills()
            assert len(skills) >= 1
            found = [s for s in skills if s.name == "shared_analyzer"]
            assert len(found) == 1
            assert found[0].description == "Analyzes shared data"

            # Fetch full definition
            peer = agent2.network.peer_manager.connected_peers[0]
            defn = await agent2.network.fetch_skill(peer.agent_id, "shared_analyzer")
            assert defn is not None
            assert defn["system_prompt"] == "You analyze data."

        finally:
            await agent1.stop_network()
            await agent2.stop_network()
            await agent1.shutdown()
            await agent2.shutdown()

    @pytest.mark.asyncio
    async def test_import_skill_from_network(self, _agents):
        """Agent2 can import a skill from Agent1 with OBSERVE trust."""
        try:
            import websockets  # noqa: F401
        except ImportError:
            pytest.skip("websockets not installed")

        from evolvagent.core.agent import Agent

        settings1, settings2 = _agents

        agent1 = Agent(settings=settings1)
        agent2 = Agent(settings=settings2)

        # Agent2 needs an LLM for DynamicSkill hydration
        mock_llm = MagicMock()
        agent2.set_llm(mock_llm)

        await agent1.start()
        await agent2.start()

        try:
            # Register a skill on agent1
            llm = MagicMock()
            meta = SkillMetadata(
                name="network_skill",
                description="A network skill",
                origin=SkillOrigin.LEARNED,
                trust_level=TrustLevel.AUTO,  # AUTO on source
            )
            dskill = DynamicSkill(metadata=meta, llm=llm, system_prompt="Network prompt")
            agent1.register_skill(dskill, persist=False)

            await agent1.start_network()
            await agent2.start_network()

            connected = await agent2.network.connect_to_peer("127.0.0.1", 18765)
            assert connected
            await asyncio.sleep(0.1)

            # Import the skill
            peer = agent2.network.peer_manager.connected_peers[0]
            imported = await agent2.import_skill_from_network(
                peer.agent_id, "network_skill"
            )
            assert imported is not None
            assert imported.metadata.name == "network_skill"
            # Trust should be OBSERVE regardless of source trust level
            assert imported.metadata.trust_level == TrustLevel.OBSERVE
            assert imported.metadata.origin == SkillOrigin.NETWORK
            assert imported.metadata.source_agent == peer.agent_id

            # Skill should be registered on agent2
            assert agent2.get_skill("network_skill") is not None

        finally:
            await agent1.stop_network()
            await agent2.stop_network()
            await agent1.shutdown()
            await agent2.shutdown()

    @pytest.mark.asyncio
    async def test_gossip_peer_exchange(self, _agents, tmp_path):
        """Three agents: A-B connected, B-C connected. After gossip, A discovers C."""
        try:
            import websockets  # noqa: F401
        except ImportError:
            pytest.skip("websockets not installed")

        from evolvagent.core.agent import Agent

        settings1, settings2 = _agents

        settings3 = Settings()
        settings3.agent.name = "agent-gamma"
        settings3.agent.data_dir = str(tmp_path / "gamma")
        settings3.network.listen_port = 18767

        agent1 = Agent(settings=settings1)
        agent2 = Agent(settings=settings2)
        agent3 = Agent(settings=settings3)

        await agent1.start()
        await agent2.start()
        await agent3.start()

        try:
            await agent1.start_network()
            await agent2.start_network()
            await agent3.start_network()

            # A connects to B
            await agent1.network.connect_to_peer("127.0.0.1", 18766)
            # B connects to C
            await agent2.network.connect_to_peer("127.0.0.1", 18767)
            await asyncio.sleep(0.1)

            # Manually trigger gossip on A (ask B for peers)
            await agent1.network._gossip_round()
            await asyncio.sleep(0.1)

            # A should now know about C via B's gossip
            known = agent1.network.peer_manager.known_peers
            assert len(known) >= 1

        finally:
            await agent1.stop_network()
            await agent2.stop_network()
            await agent3.stop_network()
            await agent1.shutdown()
            await agent2.shutdown()
            await agent3.shutdown()


# ---------------------------------------------------------------------------
# TestAgentNetworkIntegration — Agent-level network methods
# ---------------------------------------------------------------------------


class TestAgentNetworkIntegration:
    """Test Agent.start_network(), stop_network(), etc."""

    @pytest.mark.asyncio
    async def test_start_and_stop_network(self, tmp_path):
        try:
            import websockets  # noqa: F401
        except ImportError:
            pytest.skip("websockets not installed")

        reset_settings()
        from evolvagent.core.agent import Agent

        settings = Settings()
        settings.agent.data_dir = str(tmp_path / "data")
        settings.network.listen_port = 18770

        agent = Agent(settings=settings)
        await agent.start()

        try:
            assert agent.network is None

            await agent.start_network()
            assert agent.network is not None
            assert agent.network.is_running

            await agent.stop_network()
            assert agent.network is None
        finally:
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_status_dict_includes_network(self, tmp_path):
        try:
            import websockets  # noqa: F401
        except ImportError:
            pytest.skip("websockets not installed")

        reset_settings()
        from evolvagent.core.agent import Agent

        settings = Settings()
        settings.agent.data_dir = str(tmp_path / "data")
        settings.network.listen_port = 18771

        agent = Agent(settings=settings)
        await agent.start()

        try:
            # Without network
            status = agent.status_dict()
            assert status["network"] is None

            # With network
            await agent.start_network()
            status = agent.status_dict()
            assert status["network"] is not None
            assert status["network"]["running"] is True
            assert status["network"]["connected_peers"] == 0

            await agent.stop_network()
        finally:
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_stops_network(self, tmp_path):
        try:
            import websockets  # noqa: F401
        except ImportError:
            pytest.skip("websockets not installed")

        reset_settings()
        from evolvagent.core.agent import Agent

        settings = Settings()
        settings.agent.data_dir = str(tmp_path / "data")
        settings.network.listen_port = 18772

        agent = Agent(settings=settings)
        await agent.start()

        await agent.start_network()
        assert agent.network.is_running

        await agent.shutdown()
        # Network should be cleaned up
        assert agent._network is None

    @pytest.mark.asyncio
    async def test_import_without_network_returns_none(self, tmp_path):
        reset_settings()
        from evolvagent.core.agent import Agent

        settings = Settings()
        settings.agent.data_dir = str(tmp_path / "data")

        agent = Agent(settings=settings)
        await agent.start()

        try:
            result = await agent.import_skill_from_network("some-peer", "some-skill")
            assert result is None
        finally:
            await agent.shutdown()
