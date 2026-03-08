"""Unit tests for FAISSIndex — model is mocked, pure numpy logic tested."""
from __future__ import annotations

import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from cognitiveweave.config.settings import FAISSSettings
from cognitiveweave.storage.faiss_index import FAISSIndex


@pytest.fixture()
def settings(tmp_path) -> FAISSSettings:
    return FAISSSettings(
        dimension=4,
        index_path=str(tmp_path / "faiss.index"),
        model_name="mock-model",
    )


@pytest.fixture()
def index_with_mock_model(settings) -> FAISSIndex:
    idx = FAISSIndex(settings)
    mock_model = MagicMock()
    # Deterministic unit vectors for 4-dim space
    mock_model.encode = MagicMock(
        side_effect=lambda texts, normalize_embeddings=True: np.eye(4, dtype="float32")[: len(texts)]
    )
    idx._model = mock_model
    idx.build_index()
    return idx


class TestBuildAndAdd:
    def test_empty_index_len_zero(self, index_with_mock_model):
        assert len(index_with_mock_model) == 0

    def test_add_increments_length(self, index_with_mock_model):
        idx = index_with_mock_model
        idx.add("node-1", "text one")
        assert len(idx) == 1

    def test_add_batch(self, index_with_mock_model):
        idx = index_with_mock_model
        items = [
            {"id": "n1", "text": "alpha"},
            {"id": "n2", "text": "beta"},
            {"id": "n3", "text": "gamma"},
        ]
        # Return 3 unit vectors
        idx._model.encode = MagicMock(
            return_value=np.eye(4, dtype="float32")[:3]
        )
        idx.add_batch(items)
        assert len(idx) == 3

    def test_id_map_populated(self, index_with_mock_model):
        idx = index_with_mock_model
        idx.add("node-X", "some text")
        assert "node-X" in idx._id_map


class TestSearch:
    def test_empty_index_returns_empty(self, index_with_mock_model):
        results = index_with_mock_model.search("anything")
        assert results == []

    def test_search_returns_correct_id(self, index_with_mock_model):
        idx = index_with_mock_model
        idx.add("node-A", "text A")  # vector = [1,0,0,0]
        # Query also encodes to [1,0,0,0] — perfect match score = 1.0
        results = idx.search("text A", top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "node-A"
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_top_k_limits_results(self, index_with_mock_model):
        idx = index_with_mock_model
        idx._model.encode = MagicMock(
            return_value=np.eye(4, dtype="float32")
        )
        idx.add_batch([{"id": f"n{i}", "text": f"t{i}"} for i in range(4)])
        idx._model.encode = MagicMock(
            return_value=np.array([[1, 0, 0, 0]], dtype="float32")
        )
        results = idx.search("query", top_k=2)
        assert len(results) == 2

    def test_metadata_included_in_result(self, index_with_mock_model):
        idx = index_with_mock_model
        idx.add("node-M", "text", metadata={"source": "paper-1"})
        results = idx.search("text")
        assert results[0]["source"] == "paper-1"


class TestRemove:
    def test_remove_decrements_length(self, index_with_mock_model):
        idx = index_with_mock_model
        idx.add("rm-node", "to remove")
        assert len(idx) == 1
        with patch.object(idx._index, "reconstruct", side_effect=lambda i, out: None):
            idx.remove("rm-node")
        assert len(idx) == 0

    def test_remove_nonexistent_is_noop(self, index_with_mock_model):
        idx = index_with_mock_model
        idx.remove("does-not-exist")  # should not raise
        assert len(idx) == 0


class TestPersistence:
    def test_save_and_load(self, index_with_mock_model, settings):
        idx = index_with_mock_model
        idx.add("persist-node", "text", metadata={"tag": "v1"})
        idx.save()

        idx2 = FAISSIndex(settings)
        idx2.load()
        assert len(idx2) == 1
        assert idx2._id_map[0] == "persist-node"
        assert idx2._meta_map["persist-node"]["tag"] == "v1"
