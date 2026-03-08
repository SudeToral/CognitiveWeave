"""Unit tests for HybridRetriever RRF + temporal scoring — storage layers are mocked."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult


def make_retriever(
    faiss_results: list[dict],
    graph_results: list[dict] | None = None,
    recency_halflife: float = 30.0,
) -> HybridRetriever:
    faiss_index = MagicMock()
    faiss_index.search.return_value = faiss_results

    neo4j_client = MagicMock()
    neo4j_client.structural_search.return_value = graph_results or []

    return HybridRetriever(faiss_index, neo4j_client, recency_halflife=recency_halflife)


class TestRRFFusion:
    def test_faiss_only_returns_results(self):
        r = make_retriever(
            faiss_results=[{"id": "n1", "score": 0.9}, {"id": "n2", "score": 0.7}]
        )
        results = r.retrieve("query", top_k=2)
        assert len(results) == 2
        assert "n1" in [x.id for x in results]

    def test_rrf_score_formula(self):
        """Node present in both lists should have higher rrf_score."""
        r = make_retriever(
            faiss_results=[{"id": "shared", "score": 0.5}, {"id": "faiss-only", "score": 0.4}],
            graph_results=[{"id": "shared", "score": 0.6}, {"id": "graph-only", "score": 0.5}],
        )
        results = r.retrieve("query", seed_node_ids=["seed"], top_k=3)
        scores = {x.id: x.rrf_score for x in results}
        assert scores["shared"] > scores["faiss-only"]
        assert scores["shared"] > scores["graph-only"]

    def test_rrf_rank_attributes(self):
        r = make_retriever(
            faiss_results=[{"id": "n1", "score": 0.9}],
            graph_results=[{"id": "n1", "score": 0.8}],
        )
        results = r.retrieve("query", seed_node_ids=["seed"], top_k=1)
        assert results[0].faiss_rank == 1
        assert results[0].graph_rank == 1

    def test_faiss_only_node_has_none_graph_rank(self):
        r = make_retriever(
            faiss_results=[{"id": "f-only", "score": 0.9}],
            graph_results=[{"id": "g-only", "score": 0.8}],
        )
        results = r.retrieve("query", seed_node_ids=["seed"], top_k=2)
        result_map = {x.id: x for x in results}
        assert result_map["f-only"].graph_rank is None
        assert result_map["g-only"].faiss_rank is None

    def test_top_k_limits_output(self):
        faiss = [{"id": f"n{i}", "score": 1.0 - i * 0.1} for i in range(10)]
        r = make_retriever(faiss_results=faiss)
        results = r.retrieve("query", top_k=3)
        assert len(results) == 3

    def test_empty_returns_empty(self):
        r = make_retriever(faiss_results=[], graph_results=[])
        assert r.retrieve("query", seed_node_ids=["seed"], top_k=5) == []

    def test_graph_search_skipped_when_no_seeds(self):
        neo4j_client = MagicMock()
        faiss_index = MagicMock()
        faiss_index.search.return_value = [{"id": "n1", "score": 0.9}]
        r = HybridRetriever(faiss_index, neo4j_client)
        r.retrieve("query")
        neo4j_client.structural_search.assert_not_called()

    def test_result_type_is_retrieval_result(self):
        r = make_retriever(faiss_results=[{"id": "n1", "score": 0.5}])
        assert all(isinstance(x, RetrievalResult) for x in r.retrieve("query"))

    def test_metadata_included_from_faiss(self):
        r = make_retriever(
            faiss_results=[{"id": "n1", "score": 0.9, "content": "hello world"}]
        )
        assert r.retrieve("query", top_k=1)[0].metadata.get("content") == "hello world"


class TestTemporalScoring:
    def test_recency_boost_field_populated(self):
        r = make_retriever(faiss_results=[{"id": "n1", "score": 0.9}])
        result = r.retrieve("query", top_k=1)[0]
        assert 0 < result.recency_boost <= 1.0

    def test_no_created_at_gets_boost_one(self):
        """Missing created_at → no penalty → recency_boost ≈ 1.0."""
        r = make_retriever(faiss_results=[{"id": "n1", "score": 0.9}])
        result = r.retrieve("query", top_k=1)[0]
        assert result.recency_boost == pytest.approx(1.0, abs=0.01)

    def test_recent_node_scores_higher_than_old(self):
        recent_ts = datetime.now(timezone.utc).isoformat()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        r = make_retriever(
            faiss_results=[
                {"id": "old", "score": 0.9, "created_at": old_ts},
                {"id": "new", "score": 0.9, "created_at": recent_ts},
            ]
        )
        results = r.retrieve("query", top_k=2)
        result_map = {x.id: x for x in results}
        assert result_map["new"].temporal_score > result_map["old"].temporal_score

    def test_temporal_score_equals_rrf_times_boost(self):
        r = make_retriever(faiss_results=[{"id": "n1", "score": 0.9}])
        result = r.retrieve("query", top_k=1)[0]
        assert result.temporal_score == pytest.approx(
            result.rrf_score * result.recency_boost, rel=1e-6
        )

    def test_sorted_by_temporal_score(self):
        recent_ts = datetime.now(timezone.utc).isoformat()
        old_ts = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        # old node has slightly higher RRF but should lose on temporal
        r = make_retriever(
            faiss_results=[
                {"id": "old", "score": 0.95, "created_at": old_ts},
                {"id": "new", "score": 0.90, "created_at": recent_ts},
            ],
            recency_halflife=30.0,
        )
        results = r.retrieve("query", top_k=2)
        temporal_scores = [x.temporal_score for x in results]
        assert temporal_scores == sorted(temporal_scores, reverse=True)

    def test_larger_halflife_penalises_old_nodes_less(self):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        short = make_retriever(
            faiss_results=[{"id": "n1", "score": 0.9, "created_at": old_ts}],
            recency_halflife=15,
        ).retrieve("query", top_k=1)[0].temporal_score
        long_ = make_retriever(
            faiss_results=[{"id": "n1", "score": 0.9, "created_at": old_ts}],
            recency_halflife=60,
        ).retrieve("query", top_k=1)[0].temporal_score
        assert long_ > short
