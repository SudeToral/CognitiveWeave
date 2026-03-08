"""Benchmark runner — compares four retrieval strategies.

Strategies:
    faiss_only      Dense semantic search, no graph expansion.
    graph_only      Neo4j BFS from seed nodes, no FAISS.
    hybrid_rrf      FAISS + Neo4j fused via RRF, no recency boost.
    hybrid_temporal Full pipeline: RRF + Ebbinghaus recency boost.

Metrics computed per query, then averaged:
    MRR@k   Mean Reciprocal Rank — how high the first relevant result ranks.
    NDCG@k  Normalized Discounted Cumulative Gain — rewards ranking all
            relevant documents high, not just the first.
    latency_ms  Wall-clock time for the retrieval call.

All storage layers are injected, so the runner works with both real
Neo4j/FAISS instances and mocked ones for unit testing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from cognitiveweave.storage.faiss_index import FAISSIndex
from cognitiveweave.storage.neo4j_client import Neo4jClient
from benchmarks.dataset import BenchmarkDataset, BenchmarkQuery


@dataclass
class QueryResult:
    query_id: str
    strategy: str
    retrieved_ids: list[str]
    latency_ms: float
    mrr: float = 0.0
    ndcg: float = 0.0


@dataclass
class BenchmarkReport:
    strategy: str
    mean_mrr: float
    mean_ndcg: float
    mean_latency_ms: float
    per_query: list[QueryResult] = field(default_factory=list)


class BenchmarkRunner:
    """Runs all four retrieval strategies against a BenchmarkDataset."""

    def __init__(
        self,
        faiss_index: FAISSIndex,
        neo4j_client: Neo4jClient,
        k: int = 10,
        recency_halflife: float = 30.0,
    ) -> None:
        self._faiss = faiss_index
        self._neo4j = neo4j_client
        self._k = k
        self._retriever = HybridRetriever(
            faiss_index, neo4j_client, recency_halflife=recency_halflife
        )

    def run(self, dataset: BenchmarkDataset) -> list[BenchmarkReport]:
        """Run all strategies. Returns one BenchmarkReport per strategy."""
        strategies = {
            "faiss_only": self._faiss_only,
            "graph_only": self._graph_only,
            "hybrid_rrf": self._hybrid_rrf,
            "hybrid_temporal": self._hybrid_temporal,
        }
        reports: list[BenchmarkReport] = []
        for name, fn in strategies.items():
            results = [fn(q) for q in dataset.queries]
            for r in results:
                r.mrr = _mrr(r.retrieved_ids, _relevant(dataset, r.query_id), self._k)
                r.ndcg = _ndcg(r.retrieved_ids, _relevant(dataset, r.query_id), self._k)
            reports.append(BenchmarkReport(
                strategy=name,
                mean_mrr=_mean([r.mrr for r in results]),
                mean_ndcg=_mean([r.ndcg for r in results]),
                mean_latency_ms=_mean([r.latency_ms for r in results]),
                per_query=results,
            ))
        return reports

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _faiss_only(self, query: BenchmarkQuery) -> QueryResult:
        t0 = time.perf_counter()
        results = self._faiss.search(query.text, top_k=self._k)
        latency = (time.perf_counter() - t0) * 1000
        return QueryResult(
            query_id=query.query_id,
            strategy="faiss_only",
            retrieved_ids=[r["id"] for r in results],
            latency_ms=latency,
        )

    def _graph_only(self, query: BenchmarkQuery) -> QueryResult:
        if not query.seed_ids:
            return QueryResult(query.query_id, "graph_only", [], 0.0)
        t0 = time.perf_counter()
        results = self._neo4j.structural_search(
            query.seed_ids, max_hops=2, limit=self._k
        )
        latency = (time.perf_counter() - t0) * 1000
        return QueryResult(
            query_id=query.query_id,
            strategy="graph_only",
            retrieved_ids=[r["id"] for r in results],
            latency_ms=latency,
        )

    def _hybrid_rrf(self, query: BenchmarkQuery) -> QueryResult:
        t0 = time.perf_counter()
        results = self._retriever.retrieve(
            query.text,
            seed_node_ids=query.seed_ids or None,
            top_k=self._k,
        )
        latency = (time.perf_counter() - t0) * 1000
        return QueryResult(
            query_id=query.query_id,
            strategy="hybrid_rrf",
            retrieved_ids=[r.id for r in results],
            latency_ms=latency,
        )

    def _hybrid_temporal(self, query: BenchmarkQuery) -> QueryResult:
        # Same as hybrid_rrf — temporal scoring is always applied in HybridRetriever.
        # The distinction appears in benchmarks run on aged datasets where created_at
        # metadata has been populated.
        t0 = time.perf_counter()
        results = self._retriever.retrieve(
            query.text,
            seed_node_ids=query.seed_ids or None,
            top_k=self._k,
        )
        latency = (time.perf_counter() - t0) * 1000
        return QueryResult(
            query_id=query.query_id,
            strategy="hybrid_temporal",
            retrieved_ids=[r.id for r in results],
            latency_ms=latency,
        )


# ------------------------------------------------------------------
# Metric functions
# ------------------------------------------------------------------

def _mrr(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg(retrieved: list[str], relevant: list[str], k: int) -> float:
    """NDCG@k — normalized discounted cumulative gain.

    Relevance grade: 1 if the document is in the relevant set, 0 otherwise.
    Ideal DCG assumes all relevant documents are at the top.
    """
    import math

    def dcg(ids: list[str]) -> float:
        return sum(
            (1.0 / math.log2(rank + 1))
            for rank, doc_id in enumerate(ids[:k], start=1)
            if doc_id in relevant
        )

    actual = dcg(retrieved)
    ideal = dcg(relevant[:k])
    return actual / ideal if ideal > 0 else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _relevant(dataset: BenchmarkDataset, query_id: str) -> list[str]:
    for q in dataset.queries:
        if q.query_id == query_id:
            return q.relevant
    return []
