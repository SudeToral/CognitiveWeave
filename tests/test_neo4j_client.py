"""Unit tests for Neo4jClient using a mock driver — no live DB required."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cognitiveweave.config.settings import Neo4jSettings
from cognitiveweave.storage.neo4j_client import Neo4jClient


@pytest.fixture()
def settings() -> Neo4jSettings:
    return Neo4jSettings(uri="bolt://localhost:7687", user="neo4j", password="test")


@pytest.fixture()
def mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


@pytest.fixture()
def client(settings, mock_driver):
    driver, _ = mock_driver
    with patch("cognitiveweave.storage.neo4j_client.GraphDatabase") as mock_gdb:
        mock_gdb.driver.return_value = driver
        c = Neo4jClient(settings)
        c.connect()
        yield c, driver, mock_driver[1]
        c.close()


class TestConnect:
    def test_verify_connectivity_called(self, settings, mock_driver):
        driver, _ = mock_driver
        with patch("cognitiveweave.storage.neo4j_client.GraphDatabase") as mock_gdb:
            mock_gdb.driver.return_value = driver
            c = Neo4jClient(settings)
            c.connect()
            driver.verify_connectivity.assert_called_once()
            c.close()

    def test_context_manager(self, settings, mock_driver):
        driver, _ = mock_driver
        with patch("cognitiveweave.storage.neo4j_client.GraphDatabase") as mock_gdb:
            mock_gdb.driver.return_value = driver
            with Neo4jClient(settings) as c:
                assert c._driver is driver
            driver.close.assert_called_once()


class TestUpsertNode:
    def test_returns_node_id(self, client):
        c, driver, session = client
        session.run.return_value.single.return_value = {"id": "abc-123"}
        result = c.upsert_node("abc-123", content="test content")
        assert result == "abc-123"

    def test_generates_id_when_none(self, client):
        c, driver, session = client
        session.run.return_value.single.return_value = {"id": "generated-uuid"}
        result = c.upsert_node(content="auto id")
        session.run.assert_called_once()
        # id arg is passed in the cypher call
        call_kwargs = session.run.call_args
        assert call_kwargs is not None

    def test_metadata_merged_into_props(self, client):
        c, driver, session = client
        session.run.return_value.single.return_value = {"id": "n1"}
        c.upsert_node("n1", content="hello", metadata={"source": "test"})
        _, kwargs = session.run.call_args
        assert kwargs["props"]["source"] == "test"
        assert kwargs["props"]["content"] == "hello"


class TestEdgeDecay:
    def test_decay_returns_count(self, client):
        c, driver, session = client
        session.run.return_value.single.return_value = {"updated": 42}
        count = c.decay_edge_weights()
        assert count == 42

    def test_prune_returns_deleted_count(self, client):
        c, driver, session = client
        session.run.return_value.single.return_value = {"deleted": 7}
        count = c.prune_weak_edges(threshold=0.1)
        assert count == 7


class TestStructuralSearch:
    def test_returns_list_of_dicts(self, client):
        c, driver, session = client
        mock_records = [
            {"id": "n2", "score": 0.5, "props": {"content": "neighbor", "id": "n2"}},
        ]

        def make_record(data):
            r = MagicMock()
            r.__getitem__ = lambda self, k: data[k]
            return r

        session.run.return_value.__iter__ = MagicMock(
            return_value=iter(
                [MagicMock(**{"__getitem__.side_effect": lambda k: {"id": "n2", "score": 0.5, "props": {"content": "neighbor"}}[k]})]
            )
        )
        # Patch to return controlled output
        fake_result = [{"id": "n2", "score": 0.5, "content": "neighbor"}]
        session.run.return_value.__iter__ = MagicMock(return_value=iter([
            type("R", (), {"__getitem__": lambda s, k: {"id": "n2", "score": 0.5, "props": {"content": "neighbor"}}[k]})()
        ]))

        with patch.object(c, "structural_search", return_value=fake_result):
            results = c.structural_search(["n1"])
        assert results[0]["id"] == "n2"
