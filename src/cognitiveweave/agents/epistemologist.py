from __future__ import annotations

import logging
from typing import Any

from cognitiveweave.agents.base import BaseAgent
from cognitiveweave.bus.redis_bus import CH_EPISTEMOLOGIST, CH_RECONCILER, RedisBus
from cognitiveweave.llm.ollama_client import OllamaClient
from cognitiveweave.storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.35
CONFLICT_CONFIDENCE_THRESHOLD = 0.50


class EpistemologistAgent(BaseAgent):
    """Monitors belief quality and reduces hallucination risk.

    `node:created` / `node:updated` → score confidence via Ollama.
    Below LOW_CONFIDENCE_THRESHOLD → flag in Neo4j + emit `node:flagged`.
    Large divergence with existing node → emit `conflict:detected`.

    Scoring priority:
    1. Explicit confidence field from upstream (most trusted, no LLM call).
    2. Ollama LLM evaluation (main path).
    3. Heuristic fallback if Ollama is unavailable.
    """

    CHANNEL = CH_EPISTEMOLOGIST

    def __init__(
        self,
        bus: RedisBus,
        neo4j: Neo4jClient,
        ollama: OllamaClient,
    ) -> None:
        super().__init__(bus)
        self._neo4j = neo4j
        self._ollama = ollama

    async def handle_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype in ("node:created", "node:updated"):
            await self._evaluate_node(event)

    async def _evaluate_node(self, event: dict[str, Any]) -> None:
        node_id = event.get("node_id")
        content = event.get("content", "")
        source = event.get("source", "unknown")
        explicit_confidence = event.get("confidence")

        score = await self.run_in_thread(
            self._compute_confidence, content, source, explicit_confidence
        )

        if score < LOW_CONFIDENCE_THRESHOLD:
            await self.run_in_thread(
                self._neo4j.upsert_node,
                node_id,
                content,
                {"uncertain": True, "confidence": score},
            )
            await self.emit(
                CH_EPISTEMOLOGIST,
                {"type": "node:flagged", "node_id": node_id,
                 "confidence": score, "reason": "below_threshold"},
            )
            logger.info("EpistemologistAgent: flagged %s (confidence=%.3f)", node_id, score)

        await self.emit(
            CH_EPISTEMOLOGIST,
            {"type": "node:scored", "node_id": node_id, "confidence": score},
        )

        existing = await self.run_in_thread(self._neo4j.get_node, node_id)
        if existing and existing.get("confidence") is not None:
            existing_conf = float(existing["confidence"])
            if abs(existing_conf - score) > CONFLICT_CONFIDENCE_THRESHOLD:
                await self.emit(
                    CH_RECONCILER,
                    {
                        "type": "conflict:detected",
                        "node_id": node_id,
                        "version_a": {
                            "content": existing.get("content", ""),
                            "confidence": existing_conf,
                            "source": existing.get("source", "graph"),
                        },
                        "version_b": {
                            "content": content,
                            "confidence": score,
                            "source": source,
                        },
                    },
                )

    def _compute_confidence(
        self,
        content: str,
        source: str,
        explicit: float | None,
    ) -> float:
        """Score confidence for a knowledge claim.

        Priority:
        1. Explicit value — skip LLM, return immediately.
        2. Ollama LLM call — most accurate.
        3. Heuristic fallback if Ollama fails.
        """
        if explicit is not None:
            return max(0.0, min(1.0, float(explicit)))

        # Try Ollama
        try:
            return self._ollama.score_belief(content, context=f"source: {source}")
        except Exception as e:
            logger.warning("EpistemologistAgent: Ollama unavailable (%s), using heuristic", e)
            return self._heuristic_score(content, source)

    @staticmethod
    def _heuristic_score(content: str, source: str) -> float:
        """Fallback when Ollama is unavailable."""
        source_weights = {
            "arxiv": 0.85, "verified": 0.80, "llm": 0.55, "unknown": 0.40,
        }
        base = source_weights.get(source.lower(), 0.45)
        hedges = ("may", "might", "possibly", "unclear", "uncertain", "probably")
        hedge_penalty = sum(0.05 for h in hedges if h in content.lower())
        length_penalty = 0.10 if len(content) < 30 else 0.0
        return max(0.0, base - hedge_penalty - length_penalty)
