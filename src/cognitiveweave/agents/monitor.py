from __future__ import annotations

import logging
import time
from typing import Any

from cognitiveweave.agents.base import BaseAgent
from cognitiveweave.bus.redis_bus import CH_MONITOR, RedisBus

logger = logging.getLogger(__name__)

# Prometheus-compatible metric names
_METRIC_KEYS = [
    "cw_nodes_created_total",
    "cw_nodes_flagged_total",
    "cw_conflicts_detected_total",
    "cw_conflicts_resolved_total",
    "cw_edges_decayed_total",
    "cw_edges_pruned_total",
    "cw_retrieval_requests_total",
]


class MonitorAgent(BaseAgent):
    """Aggregates metrics from all agent events and writes Prometheus counters.

    Listens on its own channel for `metrics:collect` triggers.
    Passively increments counters by listening to cross-agent events
    (registered separately via `watch_channel`).

    Exposes `render_prometheus()` which returns the Prometheus text format
    string — the MCP server and a future HTTP endpoint can serve this.
    """

    CHANNEL = CH_MONITOR

    def __init__(self, bus: RedisBus) -> None:
        super().__init__(bus)
        self._start_time = time.time()

    async def handle_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type", "")

        # Direct commands
        if etype == "metrics:collect":
            report = await self._collect()
            await self.emit(CH_MONITOR, {"type": "metrics:report", **report})
            return

        # Passive counter increments from watched channels
        counter = _EVENT_TO_COUNTER.get(etype)
        if counter:
            await self._bus.increment_counter(counter)

    async def _collect(self) -> dict[str, Any]:
        metrics: dict[str, int] = {}
        for key in _METRIC_KEYS:
            val = await self._bus.get_state(key)
            metrics[key] = int(val) if val is not None else 0
        curator_state = await self._bus.get_state("curator:last_cycle")
        return {
            "metrics": metrics,
            "curator_last_cycle": curator_state,
            "uptime_s": time.time() - self._start_time,
        }

    def render_prometheus(self, metrics: dict[str, int]) -> str:
        """Render metrics dict to Prometheus text exposition format."""
        lines = []
        for key, value in metrics.items():
            lines.append(f"# TYPE {key} counter")
            lines.append(f"{key} {value}")
        return "\n".join(lines) + "\n"


# Map event types to Redis counter keys
_EVENT_TO_COUNTER: dict[str, str] = {
    "node:created": "cw_nodes_created_total",
    "node:flagged": "cw_nodes_flagged_total",
    "conflict:detected": "cw_conflicts_detected_total",
    "conflict:resolved": "cw_conflicts_resolved_total",
    "decay:complete": "cw_edges_decayed_total",
    "retrieve:request": "cw_retrieval_requests_total",
}
