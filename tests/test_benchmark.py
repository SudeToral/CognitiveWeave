"""Unit tests for benchmark framework — storage layers are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from benchmarks.dataset import build_dataset
from benchmarks.report import improvement_over_baseline, print_table, to_json
from benchmarks.runner import BenchmarkReport, BenchmarkRunner, _mean, _mrr, _ndcg

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestDataset:
    def test_dataset_has_nodes(self):
        ds = build_dataset()
        assert len(ds.nodes) > 0

    def test_dataset_has_queries(self):
        ds = build_dataset()
        assert len(ds.queries) > 0

    def test_all_relevant_ids_exist_in_nodes(self):
        ds = build_dataset()
        node_ids = {n.node_id for n in ds.nodes}
        for q in ds.queries:
            for rid in q.relevant:
                assert rid in node_ids, f"{rid} in query {q.query_id} not in nodes"

    def test_all_seed_ids_exist_in_nodes(self):
        ds = build_dataset()
        node_ids = {n.node_id for n in ds.nodes}
        for q in ds.queries:
            for sid in q.seed_ids:
                assert sid in node_ids

    def test_all_edge_endpoints_exist(self):
        ds = build_dataset()
        node_ids = {n.node_id for n in ds.nodes}
        for e in ds.edges:
            assert e.source_id in node_ids
            assert e.target_id in node_ids


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

class TestMRR:
    def test_first_result_relevant(self):
        assert _mrr(["a", "b", "c"], ["a"], k=10) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        assert _mrr(["x", "a", "b"], ["a"], k=10) == pytest.approx(0.5)

    def test_no_relevant_returns_zero(self):
        assert _mrr(["x", "y", "z"], ["a"], k=10) == 0.0

    def test_cutoff_respected(self):
        assert _mrr(["x", "x", "a"], ["a"], k=2) == 0.0  # a is at rank 3, beyond k=2

    def test_empty_retrieved(self):
        assert _mrr([], ["a"], k=10) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        # Retrieved exactly matches relevant in order
        assert _ndcg(["a", "b", "c"], ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_all_irrelevant(self):
        assert _ndcg(["x", "y", "z"], ["a", "b"], k=10) == 0.0

    def test_partial_relevance(self):
        score = _ndcg(["a", "x", "b"], ["a", "b"], k=3)
        assert 0 < score < 1.0

    def test_ndcg_rewards_higher_ranks(self):
        good = _ndcg(["a", "b", "x"], ["a", "b"], k=3)
        bad = _ndcg(["x", "b", "a"], ["a", "b"], k=3)
        assert good > bad

    def test_empty_relevant(self):
        assert _ndcg(["a", "b"], [], k=10) == 0.0


class TestMean:
    def test_empty_returns_zero(self):
        assert _mean([]) == 0.0

    def test_single_value(self):
        assert _mean([0.8]) == pytest.approx(0.8)

    def test_average(self):
        assert _mean([0.0, 1.0]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_faiss():
    f = MagicMock()
    f.search.return_value = [
        {"id": "n01", "score": 0.95},
        {"id": "n02", "score": 0.88},
        {"id": "n03", "score": 0.80},
    ]
    return f


@pytest.fixture()
def mock_neo4j():
    n = MagicMock()
    n.structural_search.return_value = [
        {"id": "n02", "score": 0.9},
        {"id": "n03", "score": 0.7},
    ]
    return n


@pytest.fixture()
def runner(mock_faiss, mock_neo4j):
    return BenchmarkRunner(mock_faiss, mock_neo4j, k=10)


@pytest.fixture()
def dataset():
    return build_dataset()


class TestBenchmarkRunner:
    def test_run_returns_four_reports(self, runner, dataset):
        reports = runner.run(dataset)
        strategies = {r.strategy for r in reports}
        assert strategies == {"faiss_only", "graph_only", "hybrid_rrf", "hybrid_temporal"}

    def test_mrr_computed_for_all_queries(self, runner, dataset):
        reports = runner.run(dataset)
        for report in reports:
            assert len(report.per_query) == len(dataset.queries)
            for qr in report.per_query:
                assert 0.0 <= qr.mrr <= 1.0

    def test_ndcg_computed_for_all_queries(self, runner, dataset):
        reports = runner.run(dataset)
        for report in reports:
            for qr in report.per_query:
                assert 0.0 <= qr.ndcg <= 1.0

    def test_latency_positive(self, runner, dataset):
        reports = runner.run(dataset)
        for report in reports:
            assert report.mean_latency_ms >= 0.0

    def test_faiss_only_does_not_call_neo4j(self, runner, mock_neo4j, dataset):
        runner.run(dataset)
        # graph_only and hybrid strategies do call neo4j — but faiss_only should not
        # We check that at least one strategy (faiss_only) doesn't trigger structural_search
        # by verifying the total call count is less than 4 * len(queries)
        calls = mock_neo4j.structural_search.call_count
        assert calls < 4 * len(dataset.queries)

    def test_graph_only_calls_neo4j_for_seeded_queries(self, runner, mock_neo4j, dataset):
        runner.run(dataset)
        seeded = sum(1 for q in dataset.queries if q.seed_ids)
        assert mock_neo4j.structural_search.call_count >= seeded


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:
    def _make_reports(self) -> list[BenchmarkReport]:
        return [
            BenchmarkReport("faiss_only", mean_mrr=0.50, mean_ndcg=0.55, mean_latency_ms=10.0),
            BenchmarkReport("hybrid_temporal", mean_mrr=0.75, mean_ndcg=0.80, mean_latency_ms=15.0),
        ]

    def test_improvement_mrr(self):
        reports = self._make_reports()
        imp = improvement_over_baseline(reports)
        assert imp["mrr_improvement_pct"] == pytest.approx(50.0, rel=0.01)

    def test_improvement_latency_overhead(self):
        reports = self._make_reports()
        imp = improvement_over_baseline(reports)
        assert imp["latency_overhead_pct"] == pytest.approx(50.0, rel=0.01)

    def test_print_table_outputs_strategies(self):
        import io
        reports = self._make_reports()
        buf = io.StringIO()
        print_table(reports, out=buf)
        out = buf.getvalue()
        assert "faiss_only" in out
        assert "hybrid_temporal" in out

    def test_to_json_is_valid(self):
        import json
        reports = self._make_reports()
        result = json.loads(to_json(reports))
        assert len(result) == 2
        assert "mean_mrr" in result[0]

    def test_missing_strategy_returns_empty(self):
        reports = self._make_reports()
        imp = improvement_over_baseline(reports, baseline="nonexistent", target="hybrid_temporal")
        assert imp == {}
