"""Unit tests for OllamaClient — httpx is fully mocked, no real Ollama needed."""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from cognitiveweave.config.settings import OllamaSettings
from cognitiveweave.llm.ollama_client import OllamaClient, _parse_json


def make_response(content: str, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"message": {"content": content}}
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture()
def settings():
    return OllamaSettings(base_url="http://localhost:11434", model="llama3.2", timeout=5.0)


@pytest.fixture()
def client(settings):
    with patch("cognitiveweave.llm.ollama_client.httpx.Client") as mock_cls:
        mock_http = MagicMock()
        mock_cls.return_value = mock_http
        c = OllamaClient(settings)
        yield c, mock_http


class TestScoreBelief:
    def test_returns_confidence_from_llm(self, client):
        c, http = client
        http.post.return_value = make_response('{"confidence": 0.82, "reason": "factual"}')
        score = c.score_belief("Ebbinghaus discovered the forgetting curve in 1885.")
        assert score == pytest.approx(0.82)

    def test_fallback_on_unparseable_response(self, client):
        c, http = client
        http.post.return_value = make_response("I cannot evaluate this.")
        score = c.score_belief("some claim")
        assert score == 0.5

    def test_fallback_on_http_error(self, client):
        import httpx
        c, http = client
        http.post.side_effect = httpx.ConnectError("refused")
        score = c.score_belief("some claim")
        assert score == 0.5

    def test_score_in_zero_one(self, client):
        c, http = client
        http.post.return_value = make_response('{"confidence": 0.95, "reason": "ok"}')
        score = c.score_belief("claim")
        assert 0.0 <= score <= 1.0

    def test_context_included_in_request(self, client):
        c, http = client
        http.post.return_value = make_response('{"confidence": 0.7, "reason": "ok"}')
        c.score_belief("claim", context="arxiv paper")
        payload = http.post.call_args[1]["json"]
        user_msg = payload["messages"][1]["content"]
        assert "arxiv paper" in user_msg


class TestEvaluateConflict:
    def test_returns_winner_a(self, client):
        c, http = client
        http.post.return_value = make_response(
            '{"winner": "a", "confidence": 0.88, "reason": "more specific"}'
        )
        result = c.evaluate_conflict(
            {"content": "Synaptic strengthening drives LTP"},
            {"content": "LTP is caused by calcium influx"},
        )
        assert result["winner"] == "a"
        assert result["confidence"] == pytest.approx(0.88)

    def test_returns_winner_b(self, client):
        c, http = client
        http.post.return_value = make_response(
            '{"winner": "b", "confidence": 0.75, "reason": "cites primary source"}'
        )
        result = c.evaluate_conflict({"content": "A"}, {"content": "B"})
        assert result["winner"] == "b"

    def test_fallback_on_invalid_winner(self, client):
        c, http = client
        http.post.return_value = make_response('{"winner": "c", "confidence": 0.5}')
        result = c.evaluate_conflict({"content": "A"}, {"content": "B"})
        assert result["winner"] == "a"
        assert result["reason"] == "fallback"

    def test_fallback_on_network_error(self, client):
        import httpx
        c, http = client
        http.post.side_effect = httpx.ConnectError("refused")
        result = c.evaluate_conflict({"content": "A"}, {"content": "B"})
        assert result["winner"] == "a"
        assert result["confidence"] == 0.5

    def test_reason_included_in_result(self, client):
        c, http = client
        http.post.return_value = make_response(
            '{"winner": "b", "confidence": 0.9, "reason": "peer-reviewed source"}'
        )
        result = c.evaluate_conflict({"content": "A"}, {"content": "B"})
        assert "peer-reviewed" in result["reason"]


class TestParseJson:
    def test_parses_clean_json(self):
        assert _parse_json('{"confidence": 0.8}') == {"confidence": 0.8}

    def test_extracts_from_prose(self):
        text = 'Sure! Here is the result: {"confidence": 0.7, "reason": "ok"} Hope that helps.'
        result = _parse_json(text)
        assert result["confidence"] == 0.7

    def test_extracts_from_markdown_fence(self):
        text = '```json\n{"winner": "a"}\n```'
        result = _parse_json(text)
        assert result["winner"] == "a"

    def test_returns_none_for_empty(self):
        assert _parse_json("") is None

    def test_returns_none_for_no_json(self):
        assert _parse_json("no json here at all") is None
