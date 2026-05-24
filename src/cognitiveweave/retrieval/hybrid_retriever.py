from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognitiveweave.storage.faiss_index import FAISSIndex
from cognitiveweave.storage.neo4j_client import Neo4jClient
from cognitiveweave.storage.temporal import DEFAULT_HALFLIFE, recency_boost_from_iso


@dataclass
class RetrievalResult:
    id: str
    rrf_score: float        # pure RRF — rank signal only
    temporal_score: float   # rrf_score × recency_boost — final sort key
    faiss_rank: int | None
    graph_rank: int | None
    recency_boost: float    # kept for inspection / Monitor dashboards
    metadata: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """Reciprocal Rank Fusion (RRF) + temporal recency boost.

    Pipeline:
        1. FAISS dense search  → semantic similarity ranking
        2. Neo4j BFS           → structural neighbourhood ranking (optional)
        3. RRF fusion          → combined rank signal (k=60, Cormack 2009)
        4. Recency boost       → multiply by e^(-age/halflife)
        5. Sort by temporal_score

    Separating RRF from recency keeps the two signals independent:
    - rrf_score  tells you *relevance*.
    - temporal_score tells you *relevance × freshness*.
    Both are exposed so Monitor can track drift.
    """

    RRF_K = 60

    def __init__(
        self,
        faiss_index: FAISSIndex,
        neo4j_client: Neo4jClient,
        *,
        recency_halflife: float = DEFAULT_HALFLIFE,
    ) -> None:
        self._faiss = faiss_index
        self._neo4j = neo4j_client
        self._halflife = recency_halflife

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        seed_node_ids: list[str] | None = None,
        top_k: int = 10,
        faiss_top_k: int = 40,
        graph_max_hops: int = 2,
        graph_limit: int = 40,
    ) -> list[RetrievalResult]:
        """Hybrid retrieval with temporal scoring.

        Args:
            query:          Natural language query string.
            seed_node_ids:  Known node ids for BFS structural expansion.
                            If None, graph search is skipped.
            top_k:          Final result count.
            faiss_top_k:    FAISS candidate pool size.
            graph_max_hops: Neo4j BFS depth.
            graph_limit:    Neo4j candidate pool size.
        """
        faiss_results = self._faiss.search(query, top_k=faiss_top_k)
        graph_results: list[dict[str, Any]] = []
        if seed_node_ids:
            graph_results = self._neo4j.structural_search(
                seed_node_ids,
                max_hops=graph_max_hops,
                limit=graph_limit,
            )

        return self._fuse(faiss_results, graph_results, top_k=top_k)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fuse(
        self,
        faiss_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        # Step 1 — RRF rank maps (1-indexed)
        faiss_rank: dict[str, int] = {r["id"]: i + 1 for i, r in enumerate(faiss_results)}
        graph_rank: dict[str, int] = {r["id"]: i + 1 for i, r in enumerate(graph_results)}
        all_ids = set(faiss_rank) | set(graph_rank)

        rrf_scores: dict[str, float] = {}
        for node_id in all_ids:
            rrf = 0.0
            if node_id in faiss_rank:
                rrf += 1.0 / (self.RRF_K + faiss_rank[node_id])
            if node_id in graph_rank:
                rrf += 1.0 / (self.RRF_K + graph_rank[node_id])
            rrf_scores[node_id] = rrf

        # Step 2 — merge metadata (faiss wins on conflicts — has content)
        faiss_meta: dict[str, dict] = {r["id"]: r for r in faiss_results}
        graph_meta: dict[str, dict] = {r["id"]: r for r in graph_results}

        # Step 3 — apply recency boost and build results
        results: list[RetrievalResult] = []
        for node_id, rrf in rrf_scores.items():
            meta = {**graph_meta.get(node_id, {}), **faiss_meta.get(node_id, {})}
            meta.pop("id", None)
            meta.pop("score", None)

            boost = recency_boost_from_iso(meta.get("created_at"), self._halflife)
            temporal = rrf * boost

            results.append(
                RetrievalResult(
                    id=node_id,
                    rrf_score=rrf,
                    temporal_score=temporal,
                    faiss_rank=faiss_rank.get(node_id),
                    graph_rank=graph_rank.get(node_id),
                    recency_boost=boost,
                    metadata=meta,
                )
            )

        # Step 4 — sort by temporal_score, return top_k
        results.sort(key=lambda r: r.temporal_score, reverse=True)
        return results[:top_k]
