"""CognitiveWeave — OpenTelemetry instrumentation.

Single entry-point for all observability primitives used across the system.
Call `setup_telemetry()` once at startup (main.py / app.py).

Exports
-------
tracer          — for adding spans to agent/retrieval code
meter           — for recording custom metrics
setup_telemetry — initialises SDK + exporters from env/settings

Custom Metrics
--------------
cw.retrieval.duration_ms        — Histogram: hybrid retrieval latency
cw.retrieval.results_count      — Histogram: how many nodes returned per query
cw.agent.cycle.duration_ms      — Histogram: full agent cycle latency
cw.agent.belief_drift           — Histogram: cosine distance between successive beliefs
cw.cross_pollination.total      — Counter:   cross-agent read events
cw.graph.decay.edges_updated    — Counter:   edges updated per decay cycle
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)

_SERVICE_NAME = "cognitiveweave"

# ---------------------------------------------------------------------------
# Module-level singletons — import these anywhere in the codebase
# ---------------------------------------------------------------------------

tracer: trace.Tracer = trace.get_tracer(_SERVICE_NAME)
meter: metrics.Meter = metrics.get_meter(_SERVICE_NAME)

# --- Metrics (created lazily after setup_telemetry() is called) ------------

retrieval_duration = meter.create_histogram(
    "cw.retrieval.duration_ms",
    unit="ms",
    description="End-to-end hybrid retrieval latency",
)

retrieval_results_count = meter.create_histogram(
    "cw.retrieval.results_count",
    description="Number of nodes returned per retrieval query",
)

agent_cycle_duration = meter.create_histogram(
    "cw.agent.cycle.duration_ms",
    unit="ms",
    description="Full SocietyAgent cycle latency (retrieve + synthesise + write)",
)

belief_drift = meter.create_histogram(
    "cw.agent.belief_drift",
    description=(
        "Cosine distance between an agent's belief at cycle N and cycle N-1. "
        "0 = identical, 1 = orthogonal. Tracks epistemic change over time."
    ),
)

cross_pollination_counter = meter.create_counter(
    "cw.cross_pollination.total",
    description=(
        "Incremented each time an agent writes a node DERIVED_FROM another agent's node. "
        "Primary signal for knowledge diffusion across the population."
    ),
)

decay_edges_updated = meter.create_counter(
    "cw.graph.decay.edges_updated",
    description="Total edges updated by Ebbinghaus decay cycles",
)

system_entropy = meter.create_histogram(
    "cw.system.entropy",
    description=(
        "Mean pairwise cosine distance between agent belief embeddings per cycle. "
        "0=homogenized, 0.5=healthy diversity, 1=max divergence."
    ),
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_telemetry(
    otlp_endpoint: str | None = None,
    *,
    console_fallback: bool = False,
) -> None:
    """Initialise TracerProvider and MeterProvider.

    Args:
        otlp_endpoint:    gRPC endpoint for OTLP export, e.g. "localhost:4317".
                          Defaults to OTEL_EXPORTER_OTLP_ENDPOINT env var, then
                          "localhost:4317".
        console_fallback: If True and OTLP is unreachable, fall back to stdout.
                          Useful for local dev without Jaeger running.
    """
    endpoint: str = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317") or "localhost:4317"
    resource = Resource.create({"service.name": _SERVICE_NAME})

    # --- Traces ---
    span_exporter = _build_span_exporter(endpoint, console_fallback)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # --- Metrics ---
    metric_exporter = _build_metric_exporter(endpoint, console_fallback)
    reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15_000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    # Re-bind module-level singletons to the now-configured providers
    global tracer, meter  # noqa: PLW0603
    global retrieval_duration, retrieval_results_count
    global agent_cycle_duration, belief_drift
    global cross_pollination_counter, decay_edges_updated, system_entropy

    tracer = trace.get_tracer(_SERVICE_NAME)
    meter = metrics.get_meter(_SERVICE_NAME)

    retrieval_duration = meter.create_histogram("cw.retrieval.duration_ms", unit="ms")
    retrieval_results_count = meter.create_histogram("cw.retrieval.results_count")
    agent_cycle_duration = meter.create_histogram("cw.agent.cycle.duration_ms", unit="ms")
    belief_drift = meter.create_histogram("cw.agent.belief_drift")
    cross_pollination_counter = meter.create_counter("cw.cross_pollination.total")
    decay_edges_updated = meter.create_counter("cw.graph.decay.edges_updated")
    system_entropy = meter.create_histogram("cw.system.entropy")

    logger.info("OpenTelemetry configured → %s", endpoint)


def _build_span_exporter(
    endpoint: str, console_fallback: bool
) -> OTLPSpanExporter | ConsoleSpanExporter:
    try:
        exp = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        logger.debug("OTLPSpanExporter → %s", endpoint)
        return exp
    except Exception as exc:
        if console_fallback:
            logger.warning("OTLP span exporter failed (%s), falling back to console", exc)
            return ConsoleSpanExporter()
        raise


def _build_metric_exporter(
    endpoint: str, console_fallback: bool
) -> OTLPMetricExporter | ConsoleMetricExporter:
    try:
        exp = OTLPMetricExporter(endpoint=endpoint, insecure=True)
        logger.debug("OTLPMetricExporter → %s", endpoint)
        return exp
    except Exception as exc:
        if console_fallback:
            logger.warning("OTLP metric exporter failed (%s), falling back to console", exc)
            return ConsoleMetricExporter()
        raise


# ---------------------------------------------------------------------------
# Helpers used by instrumented code
# ---------------------------------------------------------------------------

def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine distance in [0, 1] between two embedding vectors.

    0 = identical direction, 1 = orthogonal, 2 = opposite.
    We clamp to [0, 1] since beliefs are L2-normalised.
    """
    import numpy as np  # local import — numpy always available

    va = np.array(a, dtype="float32")
    vb = np.array(b, dtype="float32")
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    similarity = float(np.dot(va, vb) / (na * nb))
    return float(max(0.0, min(1.0, 1.0 - similarity)))
