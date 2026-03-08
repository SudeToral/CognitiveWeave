from __future__ import annotations

import asyncio
import logging
from typing import Any

from cognitiveweave.agents.base import BaseAgent
from cognitiveweave.bus.redis_bus import RedisBus, CH_RECONCILER
from cognitiveweave.llm.ollama_client import OllamaClient
from cognitiveweave.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

_FORK_KEY = "cw:fork:{node_id}:{version}"


class ReconcilerAgent(BaseAgent):
    """Resolves conflicting beliefs via fork → LLM eval → merge.

    Conflict event contract:
        {type: "conflict:detected",
         node_id: str,
         version_a: {content: str, confidence?: float, source?: str},
         version_b: {content: str, confidence?: float, source?: str}}

    Pipeline:
    1. Fork: store both versions in Redis under isolated namespaces.
    2. Call Ollama `evaluate_conflict` in a thread — LLM picks the winner.
    3. Acquire distributed lock on the node.
    4. Write winner to Neo4j with updated confidence.
    5. Emit `conflict:resolved`.

    Fallback: if Ollama is unavailable, `_heuristic_score` is used so
    the agent never stalls.
    """

    CHANNEL = CH_RECONCILER

    def __init__(
        self,
        bus: RedisBus,
        neo4j: Neo4jClient,
        ollama: OllamaClient,
    ) -> None:
        super().__init__(bus, max_workers=4)
        self._neo4j = neo4j
        self._ollama = ollama

    async def handle_event(self, event: dict[str, Any]) -> None:
        if event.get("type") != "conflict:detected":
            return
        await self._reconcile(event["node_id"], event["version_a"], event["version_b"])

    async def _reconcile(
        self,
        node_id: str,
        version_a: dict[str, Any],
        version_b: dict[str, Any],
    ) -> None:
        logger.info("ReconcilerAgent: reconciling node %s", node_id)

        # Fork — cache both in Redis
        await self._bus.set_state(_FORK_KEY.format(node_id=node_id, version="a"), version_a)
        await self._bus.set_state(_FORK_KEY.format(node_id=node_id, version="b"), version_b)

        # LLM evaluation in thread (blocking HTTP)
        verdict = await self.run_in_thread(
            self._ollama.evaluate_conflict, version_a, version_b
        )

        winner_key = verdict.get("winner", "a")
        winner = version_a if winner_key == "a" else version_b
        winner_score = float(verdict.get("confidence", 0.5))
        reason = verdict.get("reason", "")

        # Distributed lock before graph write
        try:
            async with self._bus.lock(f"node:{node_id}"):
                await self.run_in_thread(
                    self._neo4j.upsert_node,
                    node_id,
                    winner["content"],
                    {**winner, "confidence": winner_score, "reconciled_by": "ollama"},
                )
        except RuntimeError:
            logger.warning("ReconcilerAgent: lock contention for %s — skipping", node_id)
            return

        await self.emit(
            CH_RECONCILER,
            {
                "type": "conflict:resolved",
                "node_id": node_id,
                "winner": winner_key,
                "winner_score": winner_score,
                "winning_content": winner.get("content"),
                "reason": reason,
            },
        )
        logger.info("ReconcilerAgent: resolved %s — winner=%s score=%.3f",
                    node_id, winner_key, winner_score)

    @staticmethod
    def _heuristic_score(version: dict[str, Any]) -> float:
        """Fallback when Ollama is unavailable."""
        base = float(version.get("confidence", 0.5))
        length_bonus = min(len(version.get("content", "")) / 500, 0.1)
        return min(base + length_bonus, 1.0)
