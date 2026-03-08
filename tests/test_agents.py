"""Unit tests for all 5 agents — bus and storage layers are fully mocked."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cognitiveweave.agents.curator import CuratorAgent
from cognitiveweave.agents.retriever import RetrieverAgent
from cognitiveweave.agents.reconciler import ReconcilerAgent, _FORK_KEY
from cognitiveweave.agents.epistemologist import EpistemologistAgent
from cognitiveweave.agents.monitor import MonitorAgent, _EVENT_TO_COUNTER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bus():
    b = AsyncMock()
    b.publish = AsyncMock()
    b.set_state = AsyncMock()
    b.get_state = AsyncMock(return_value=None)
    b.increment_counter = AsyncMock(return_value=1)
    b.subscribe = AsyncMock()
    b.lock = MagicMock()
    b.lock.return_value.__aenter__ = AsyncMock(return_value=None)
    b.lock.return_value.__aexit__ = AsyncMock(return_value=False)
    return b


@pytest.fixture()
def neo4j():
    n = MagicMock()
    n.decay_edge_weights = MagicMock(return_value=100)
    n.prune_weak_edges = MagicMock(return_value=5)
    n.upsert_node = MagicMock(return_value="node-winner")
    n.get_node = MagicMock(return_value=None)
    return n


# ---------------------------------------------------------------------------
# CuratorAgent
# ---------------------------------------------------------------------------

class TestCuratorAgent:
    @pytest.mark.asyncio
    async def test_decay_trigger_calls_neo4j(self, bus, neo4j):
        agent = CuratorAgent(bus, neo4j)
        await agent.handle_event({"type": "decay:trigger"})
        # decay_edge_weights now uses Ebbinghaus formula — no factor argument
        neo4j.decay_edge_weights.assert_called_once_with()
        neo4j.prune_weak_edges.assert_called_once_with(0.10)

    @pytest.mark.asyncio
    async def test_decay_publishes_complete_event(self, bus, neo4j):
        agent = CuratorAgent(bus, neo4j)
        await agent.handle_event({"type": "decay:trigger"})
        bus.publish.assert_called()
        event = bus.publish.call_args[0][1]
        assert event["type"] == "decay:complete"
        assert "decayed_edges" in event
        assert "pruned_edges" in event

    @pytest.mark.asyncio
    async def test_unknown_event_ignored(self, bus, neo4j):
        agent = CuratorAgent(bus, neo4j)
        await agent.handle_event({"type": "something:else"})
        neo4j.decay_edge_weights.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_prune_threshold(self, bus, neo4j):
        agent = CuratorAgent(bus, neo4j, prune_threshold=0.05)
        await agent.handle_event({"type": "decay:trigger"})
        neo4j.decay_edge_weights.assert_called_once_with()
        neo4j.prune_weak_edges.assert_called_once_with(0.05)


# ---------------------------------------------------------------------------
# RetrieverAgent
# ---------------------------------------------------------------------------

class TestRetrieverAgent:
    @pytest.fixture()
    def retriever(self):
        r = MagicMock()
        from cognitiveweave.retrieval.hybrid_retriever import RetrievalResult
        r.retrieve = MagicMock(return_value=[
            RetrievalResult(id="n1", rrf_score=0.9, temporal_score=0.85,
                            faiss_rank=1, graph_rank=None, recency_boost=0.94,
                            metadata={"content": "test"})
        ])
        return r

    @pytest.mark.asyncio
    async def test_retrieve_request_publishes_response(self, bus, retriever):
        agent = RetrieverAgent(bus, retriever)
        await agent.handle_event({
            "type": "retrieve:request",
            "query": "memory consolidation",
            "top_k": 5,
        })
        bus.publish.assert_called_once()
        channel, event = bus.publish.call_args[0]
        assert event["type"] == "retrieve:response"
        assert event["results"][0]["id"] == "n1"

    @pytest.mark.asyncio
    async def test_reply_to_custom_channel(self, bus, retriever):
        agent = RetrieverAgent(bus, retriever)
        await agent.handle_event({
            "type": "retrieve:request",
            "query": "test",
            "reply_to": "cw:reconciler",
        })
        channel, _ = bus.publish.call_args[0]
        assert channel == "cw:reconciler"

    @pytest.mark.asyncio
    async def test_non_retrieve_event_ignored(self, bus, retriever):
        agent = RetrieverAgent(bus, retriever)
        await agent.handle_event({"type": "other:event"})
        bus.publish.assert_not_called()


# ---------------------------------------------------------------------------
# ReconcilerAgent
# ---------------------------------------------------------------------------

@pytest.fixture()
def ollama():
    o = MagicMock()
    o.evaluate_conflict = MagicMock(
        return_value={"winner": "a", "confidence": 0.88, "reason": "more specific"}
    )
    o.score_belief = MagicMock(return_value=0.75)
    return o


class TestReconcilerAgent:
    @pytest.mark.asyncio
    async def test_conflict_resolved_emits_event(self, bus, neo4j, ollama):
        agent = ReconcilerAgent(bus, neo4j, ollama)
        await agent.handle_event({
            "type": "conflict:detected",
            "node_id": "node-x",
            "version_a": {"content": "Ebbinghaus curve is exponential", "confidence": 0.9},
            "version_b": {"content": "Ebbinghaus curve is logarithmic", "confidence": 0.3},
        })
        bus.publish.assert_called()
        channel, event = bus.publish.call_args[0]
        assert event["type"] == "conflict:resolved"
        assert event["node_id"] == "node-x"

    @pytest.mark.asyncio
    async def test_ollama_called_for_evaluation(self, bus, neo4j, ollama):
        agent = ReconcilerAgent(bus, neo4j, ollama)
        await agent.handle_event({
            "type": "conflict:detected",
            "node_id": "n1",
            "version_a": {"content": "A"},
            "version_b": {"content": "B"},
        })
        ollama.evaluate_conflict.assert_called_once()

    @pytest.mark.asyncio
    async def test_ollama_winner_respected(self, bus, neo4j, ollama):
        ollama.evaluate_conflict.return_value = {"winner": "b", "confidence": 0.9, "reason": ""}
        agent = ReconcilerAgent(bus, neo4j, ollama)
        await agent.handle_event({
            "type": "conflict:detected",
            "node_id": "n1",
            "version_a": {"content": "A"},
            "version_b": {"content": "B"},
        })
        channel, event = bus.publish.call_args[0]
        assert event["winner"] == "b"
        assert event["winning_content"] == "B"

    @pytest.mark.asyncio
    async def test_forks_stored_in_redis(self, bus, neo4j, ollama):
        agent = ReconcilerAgent(bus, neo4j, ollama)
        await agent.handle_event({
            "type": "conflict:detected",
            "node_id": "fork-node",
            "version_a": {"content": "va"},
            "version_b": {"content": "vb"},
        })
        set_state_calls = [c[0][0] for c in bus.set_state.call_args_list]
        assert any("fork-node" in k and ":a" in k for k in set_state_calls)
        assert any("fork-node" in k and ":b" in k for k in set_state_calls)

    def test_heuristic_score_explicit_confidence(self):
        score = ReconcilerAgent._heuristic_score({"content": "x" * 100, "confidence": 0.75})
        assert 0.75 <= score <= 1.0

    def test_heuristic_score_length_bonus(self):
        short = ReconcilerAgent._heuristic_score({"content": "short"})
        long_ = ReconcilerAgent._heuristic_score({"content": "x" * 600})
        assert long_ > short


# ---------------------------------------------------------------------------
# EpistemologistAgent
# ---------------------------------------------------------------------------

class TestEpistemologistAgent:
    @pytest.mark.asyncio
    async def test_low_confidence_emits_flagged(self, bus, neo4j, ollama):
        ollama.score_belief.return_value = 0.20  # below threshold
        agent = EpistemologistAgent(bus, neo4j, ollama)
        await agent.handle_event({
            "type": "node:created",
            "node_id": "weak-node",
            "content": "This might possibly be unclear",
            "source": "unknown",
        })
        events = [c[0][1]["type"] for c in bus.publish.call_args_list]
        assert "node:flagged" in events

    @pytest.mark.asyncio
    async def test_high_confidence_not_flagged(self, bus, neo4j, ollama):
        ollama.score_belief.return_value = 0.90  # above threshold
        agent = EpistemologistAgent(bus, neo4j, ollama)
        await agent.handle_event({
            "type": "node:created",
            "node_id": "strong-node",
            "content": "Ebbinghaus forgetting curve follows a power law decay",
            "source": "arxiv",
        })
        events = [c[0][1]["type"] for c in bus.publish.call_args_list]
        assert "node:flagged" not in events

    @pytest.mark.asyncio
    async def test_node_scored_always_emitted(self, bus, neo4j, ollama):
        agent = EpistemologistAgent(bus, neo4j, ollama)
        await agent.handle_event({
            "type": "node:created",
            "node_id": "any-node",
            "content": "Some content",
            "source": "verified",
        })
        events = [c[0][1]["type"] for c in bus.publish.call_args_list]
        assert "node:scored" in events

    @pytest.mark.asyncio
    async def test_explicit_confidence_skips_ollama(self, bus, neo4j, ollama):
        agent = EpistemologistAgent(bus, neo4j, ollama)
        await agent.handle_event({
            "type": "node:created",
            "node_id": "n1",
            "content": "content",
            "source": "verified",
            "confidence": 0.9,  # explicit — Ollama should NOT be called
        })
        ollama.score_belief.assert_not_called()

    def test_heuristic_score_arxiv(self):
        score = EpistemologistAgent._heuristic_score(
            "Definitive result on memory decay", "arxiv"
        )
        assert score > 0.7

    def test_heuristic_score_hedge_penalty(self):
        no_hedge = EpistemologistAgent._heuristic_score("fact", "verified")
        hedge = EpistemologistAgent._heuristic_score(
            "this might possibly be unclear", "verified"
        )
        assert no_hedge > hedge


# ---------------------------------------------------------------------------
# MonitorAgent
# ---------------------------------------------------------------------------

class TestMonitorAgent:
    @pytest.mark.asyncio
    async def test_metrics_collect_emits_report(self, bus):
        bus.get_state = AsyncMock(return_value=0)
        agent = MonitorAgent(bus)
        await agent.handle_event({"type": "metrics:collect"})
        events = [c[0][1]["type"] for c in bus.publish.call_args_list]
        assert "metrics:report" in events

    @pytest.mark.asyncio
    async def test_known_event_increments_counter(self, bus):
        agent = MonitorAgent(bus)
        await agent.handle_event({"type": "node:created"})
        bus.increment_counter.assert_called_once_with("cw_nodes_created_total")

    @pytest.mark.asyncio
    async def test_unknown_event_no_counter(self, bus):
        agent = MonitorAgent(bus)
        await agent.handle_event({"type": "some:unknown:event"})
        bus.increment_counter.assert_not_called()

    def test_render_prometheus_format(self, bus):
        agent = MonitorAgent(bus)
        output = agent.render_prometheus({"cw_nodes_created_total": 42})
        assert "cw_nodes_created_total 42" in output
        assert "# TYPE cw_nodes_created_total counter" in output

    def test_all_event_types_mapped(self):
        mapped_types = set(_EVENT_TO_COUNTER.keys())
        expected = {
            "node:created", "node:flagged", "conflict:detected",
            "conflict:resolved", "decay:complete", "retrieve:request",
        }
        assert expected.issubset(mapped_types)
