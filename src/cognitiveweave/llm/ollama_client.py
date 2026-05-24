"""Async Ollama client for local LLM inference.

Used by ReconcilerAgent and EpistemologistAgent. Both agents call Ollama
from within a ThreadPoolExecutor (via run_in_thread), so this client uses
a synchronous httpx.Client internally — one client per thread is safe.

If Ollama is unreachable or returns an error, methods return a fallback
value rather than raising, so agent loops are never killed by LLM issues.

API: Ollama /api/chat — JSON request/response over HTTP.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from cognitiveweave.config.settings import OllamaSettings

logger = logging.getLogger(__name__)

# System prompts are kept short to minimise token cost on local models.
_SCORE_SYSTEM = (
    "You are a knowledge quality evaluator. "
    "Respond ONLY with a JSON object: {\"confidence\": <float 0-1>, \"reason\": <string>}. "
    "No markdown, no extra text."
)

_RECONCILE_SYSTEM = (
    "You are a knowledge reconciler. Given two conflicting statements, "
    "decide which is more accurate and credible. "
    "Respond ONLY with JSON: "
    "{\"winner\": \"a\" or \"b\", \"confidence\": <float 0-1>, \"reason\": <string>}. "
    "No markdown, no extra text."
)


class OllamaClient:
    """Synchronous Ollama /api/chat wrapper.

    Designed to be called from a ThreadPoolExecutor so async agents
    can offload LLM calls without blocking the event loop.
    """

    def __init__(self, settings: OllamaSettings) -> None:
        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.base_url,
            timeout=settings.timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_belief(self, content: str, context: str = "") -> float:
        """Ask the LLM to score how credible/accurate a belief statement is.

        Returns a float in [0, 1]. Falls back to 0.5 on any error so the
        EpistemologistAgent can still make a decision.

        Args:
            content: The belief text to evaluate.
            context: Optional surrounding context (e.g. source document).
        """
        user_msg = f"Statement: {content}"
        if context:
            user_msg += f"\nContext: {context}"

        raw = self._chat(_SCORE_SYSTEM, user_msg)
        parsed = _parse_json(raw)
        if parsed is None:
            logger.warning("OllamaClient.score_belief: unparseable response — using 0.5")
            return 0.5
        return float(parsed.get("confidence", 0.5))

    def evaluate_conflict(
        self,
        version_a: dict[str, Any],
        version_b: dict[str, Any],
    ) -> dict[str, Any]:
        """Ask the LLM to pick the more credible of two conflicting beliefs.

        Returns:
            {"winner": "a"|"b", "confidence": float, "reason": str}
            Falls back to {"winner": "a", "confidence": 0.5, "reason": "fallback"}
            on any error.
        """
        user_msg = (
            f"Statement A: {version_a.get('content', '')}\n"
            f"Statement B: {version_b.get('content', '')}\n"
            "Which is more accurate and credible?"
        )
        raw = self._chat(_RECONCILE_SYSTEM, user_msg)
        parsed = _parse_json(raw)
        if parsed is None or parsed.get("winner") not in ("a", "b"):
            logger.warning("OllamaClient.evaluate_conflict: unparseable — defaulting to 'a'")
            return {"winner": "a", "confidence": 0.5, "reason": "fallback"}
        return {
            "winner": parsed["winner"],
            "confidence": float(parsed.get("confidence", 0.5)),
            "reason": parsed.get("reason", ""),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _chat(self, system: str, user: str) -> str:
        """Send a chat request to Ollama. Returns raw response string."""
        payload = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        try:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
            logger.error("OllamaClient._chat error: %s", e)
            return ""


def _parse_json(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM response.

    Some models wrap JSON in markdown fences or add prose around it —
    this handles that gracefully.
    """
    if not text:
        return None
    # Try raw first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown fence
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
