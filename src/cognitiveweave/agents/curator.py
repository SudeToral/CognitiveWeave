from __future__ import annotations

import asyncio
import logging
from typing import Any

from cognitiveweave.agents.base import BaseAgent
from cognitiveweave.bus.redis_bus import CH_CURATOR, RedisBus
from cognitiveweave.storage.neo4j_client import Neo4jClient
from cognitiveweave.telemetry import decay_edges_updated, tracer

logger = logging.getLogger(__name__)


class CuratorAgent(BaseAgent):
    """Maintains graph health via periodic Ebbinghaus decay + pruning.

    Two modes:
    1. Event-driven: responds to `decay:trigger` on its channel.
    2. Scheduled:    `run_periodic(interval_s)` fires decay on a timer.

    decay_edge_weights() now uses the real Ebbinghaus formula (exp(-Δt/stability))
    computed inside Neo4j with Cypher's exp() — no decay_factor param needed.
    """

    CHANNEL = CH_CURATOR

    def __init__(
        self,
        bus: RedisBus,
        neo4j: Neo4jClient,
        *,
        prune_threshold: float = 0.10,
    ) -> None:
        super().__init__(bus)
        self._neo4j = neo4j
        self._prune_threshold = prune_threshold

    async def handle_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "decay:trigger":
            await self._run_decay_cycle()

    async def _run_decay_cycle(self) -> None:
        logger.info("CuratorAgent: starting decay cycle")

        with tracer.start_as_current_span("curator.decay_cycle") as span:
            decayed = await self.run_in_thread(self._neo4j.decay_edge_weights)
            pruned = await self.run_in_thread(
                self._neo4j.prune_weak_edges, self._prune_threshold
            )
            span.set_attribute("edges.decayed", decayed)
            span.set_attribute("edges.pruned", pruned)

        decay_edges_updated.add(decayed)
        stats = {"type": "decay:complete", "decayed_edges": decayed, "pruned_edges": pruned}
        await self._bus.set_state("curator:last_cycle", stats)
        await self.emit(CH_CURATOR, stats)
        logger.info("CuratorAgent: decay done — decayed=%d pruned=%d", decayed, pruned)

    async def run_periodic(self, interval_s: float = 3_600.0) -> None:
        """Run decay every `interval_s` seconds. Cancel to stop."""
        while True:
            await asyncio.sleep(interval_s)
            await self._run_decay_cycle()
