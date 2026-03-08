from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from cognitiveweave.agents.base import BaseAgent
from cognitiveweave.bus.redis_bus import RedisBus, CH_RETRIEVER
from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """Handles hybrid retrieval requests from the bus.

    Event contract:
        Request  → channel cw:retriever
            {type: "retrieve:request", query: str, seed_ids?: list[str],
             top_k?: int, reply_to?: str}

        Response → channel `reply_to` (default: cw:retriever)
            {type: "retrieve:response", results: list[{id, rrf_score, ...}]}

    `reply_to` lets other agents specify their own channel as the return
    address — e.g. ReconcilerAgent can fire a retrieval and get results
    directly on cw:reconciler.
    """

    CHANNEL = CH_RETRIEVER

    def __init__(self, bus: RedisBus, retriever: HybridRetriever) -> None:
        super().__init__(bus)
        self._retriever = retriever

    async def handle_event(self, event: dict[str, Any]) -> None:
        if event.get("type") != "retrieve:request":
            return

        query = event.get("query", "")
        seed_ids = event.get("seed_ids")
        top_k = int(event.get("top_k", 10))
        reply_to = event.get("reply_to", CH_RETRIEVER)

        # FAISS search runs in thread (sentence-transformers is CPU-bound)
        results = await self.run_in_thread(
            self._sync_retrieve, query, seed_ids, top_k
        )

        await self._bus.publish(
            reply_to,
            {"type": "retrieve:response", "query": query, "results": results},
        )

    def _sync_retrieve(
        self, query: str, seed_ids: list[str] | None, top_k: int
    ) -> list[dict[str, Any]]:
        results = self._retriever.retrieve(query, seed_node_ids=seed_ids, top_k=top_k)
        return [
            {
                "id": r.id,
                "rrf_score": r.rrf_score,
                "faiss_rank": r.faiss_rank,
                "graph_rank": r.graph_rank,
                **r.metadata,
            }
            for r in results
        ]
