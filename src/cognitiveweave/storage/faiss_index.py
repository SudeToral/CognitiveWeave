from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from cognitiveweave.config.settings import FAISSSettings


class FAISSIndex:
    """Dense vector index backed by FAISS + sentence-transformers.

    Maintains a parallel id→metadata mapping so callers can round-trip
    back to graph node ids after a similarity search.
    """

    def __init__(self, settings: FAISSSettings) -> None:
        self._settings = settings
        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        # Ordered list of node ids matching FAISS internal integer indices.
        self._id_map: list[str] = []
        self._meta_map: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        self._model = SentenceTransformer(self._settings.model_name)

    def build_index(self) -> None:
        """Create an empty inner-product (cosine, after normalisation) index."""
        self._index = faiss.IndexFlatIP(self._settings.dimension)
        self._id_map = []
        self._meta_map = {}

    def save(self) -> None:
        path = Path(self._settings.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({"id_map": self._id_map, "meta_map": self._meta_map}, f)

    def load(self) -> None:
        path = Path(self._settings.index_path)
        self._index = faiss.read_index(str(path))
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path) as f:
            data = json.load(f)
        self._id_map = data["id_map"]
        self._meta_map = data["meta_map"]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return vecs.astype("float32")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings for the given texts (float32)."""
        return self._embed(texts)

    def add(self, node_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        vec = self._embed([text])
        self._index.add(vec)
        self._id_map.append(node_id)
        self._meta_map[node_id] = metadata or {}

    def add_batch(self, items: list[dict[str, Any]]) -> None:
        """items: list of {id, text, metadata?}"""
        texts = [it["text"] for it in items]
        vecs = self._embed(texts)
        self._index.add(vecs)
        for it in items:
            self._id_map.append(it["id"])
            self._meta_map[it["id"]] = it.get("metadata", {})

    def remove(self, node_id: str) -> None:
        """Remove a vector by node id. Rebuilds the index — use sparingly."""
        if node_id not in self._id_map:
            return
        idx = self._id_map.index(node_id)
        keep = [i for i in range(len(self._id_map)) if i != idx]
        if not keep:
            self.build_index()
            return
        # Reconstruct remaining vectors
        all_vecs = np.zeros((self._index.ntotal, self._settings.dimension), dtype="float32")
        faiss.extract_index_ivf  # noqa: just ensure faiss is loaded
        for i in range(self._index.ntotal):
            self._index.reconstruct(i, all_vecs[i])
        kept_vecs = all_vecs[keep]
        self._index = faiss.IndexFlatIP(self._settings.dimension)
        self._index.add(kept_vecs)
        del self._meta_map[node_id]
        self._id_map = [self._id_map[i] for i in keep]

    def prune_experiment_nodes(self) -> int:
        """Remove all experiment nodes (id starts with 'exp_') from the index.

        Called on experiment reset so stale vectors don't pollute future runs.
        Returns the number of entries removed.
        """
        exp_ids = [nid for nid in self._id_map if nid.startswith("exp_")]
        if not exp_ids:
            return 0
        keep_mask = [not nid.startswith("exp_") for nid in self._id_map]
        keep_indices = [i for i, k in enumerate(keep_mask) if k]
        if not keep_indices:
            self.build_index()
            return len(exp_ids)
        all_vecs = np.zeros((self._index.ntotal, self._settings.dimension), dtype="float32")
        for i in range(self._index.ntotal):
            self._index.reconstruct(i, all_vecs[i])
        kept_vecs = all_vecs[keep_indices]
        self._index = faiss.IndexFlatIP(self._settings.dimension)
        self._index.add(kept_vecs)
        for nid in exp_ids:
            self._meta_map.pop(nid, None)
        self._id_map = [self._id_map[i] for i in keep_indices]
        return len(exp_ids)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, *, top_k: int = 20) -> list[dict[str, Any]]:
        """Return top_k results with cosine similarity scores."""
        if self._index is None or self._index.ntotal == 0:
            return []
        q_vec = self._embed([query])
        scores, indices = self._index.search(q_vec, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            node_id = self._id_map[idx]
            results.append({
                "id": node_id,
                "score": float(score),
                **self._meta_map.get(node_id, {}),
            })
        return results

    def __len__(self) -> int:
        return self._index.ntotal if self._index else 0
