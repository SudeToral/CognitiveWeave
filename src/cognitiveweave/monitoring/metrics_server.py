"""Lightweight HTTP server that exposes Prometheus metrics on /metrics.

Uses only stdlib — no prometheus_client dependency. The MonitorAgent
collects counters in Redis; this server reads them and formats the output.

Runs in a separate thread so it never blocks the asyncio event loop.

Usage:
    server = MetricsServer(bus, host="0.0.0.0", port=8000)
    server.start()   # non-blocking, runs in daemon thread
    ...
    server.stop()
"""
from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from cognitiveweave.bus.redis_bus import RedisBus
from cognitiveweave.agents.monitor import MonitorAgent, _METRIC_KEYS

logger = logging.getLogger(__name__)

_HELP_TEXTS: dict[str, str] = {
    "cw_nodes_created_total":    "Total knowledge nodes ingested",
    "cw_nodes_flagged_total":    "Nodes flagged as uncertain by EpistemologistAgent",
    "cw_conflicts_detected_total": "Belief conflicts detected",
    "cw_conflicts_resolved_total": "Belief conflicts resolved by ReconcilerAgent",
    "cw_edges_decayed_total":    "Decay cycles completed by CuratorAgent",
    "cw_edges_pruned_total":     "Edges pruned below weight threshold",
    "cw_retrieval_requests_total": "Total retrieval requests handled",
}


class MetricsServer:
    """Prometheus /metrics HTTP endpoint, stdlib only."""

    def __init__(self, bus: RedisBus, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._bus = bus
        self._host = host
        self._port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        bus = self._bus

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path != "/metrics":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = _render_sync(bus)
                encoded = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, fmt, *args):  # silence default access log
                pass

        self._server = HTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("MetricsServer started on %s:%d", self._host, self._port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        logger.info("MetricsServer stopped")


def _render_sync(bus: RedisBus) -> str:
    """Read counters from Redis synchronously and render Prometheus text.

    Called from a non-async thread — uses a new event loop via asyncio.run
    to execute the async Redis reads.
    """
    import asyncio

    async def _collect() -> dict[str, int]:
        metrics: dict[str, int] = {}
        for key in _METRIC_KEYS:
            val = await bus.get_state(key)
            metrics[key] = int(val) if val is not None else 0
        return metrics

    try:
        metrics = asyncio.run(_collect())
    except Exception as e:
        logger.error("MetricsServer: failed to collect metrics: %s", e)
        metrics = {k: 0 for k in _METRIC_KEYS}

    lines: list[str] = []
    for key, value in metrics.items():
        help_text = _HELP_TEXTS.get(key, key)
        lines.append(f"# HELP {key} {help_text}")
        lines.append(f"# TYPE {key} counter")
        lines.append(f"{key} {value}")
    return "\n".join(lines) + "\n"
