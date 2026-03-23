from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import faiss
import numpy as np


@dataclass(frozen=True)
class PassageMeta:
    passage_id: str
    doc_id: str
    chunk_id: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    passage_id: str
    doc_id: str
    chunk_id: int
    score: float
    text: str


class FaissPassageIndex:
    """FAISS-backed ANN index over passage embeddings with metadata."""

    def __init__(self, index_type: str = "IndexFlatIP") -> None:
        self.index_type = index_type
        self._index: Optional[faiss.Index] = None
        self._meta: List[PassageMeta] = []

    @property
    def is_built(self) -> bool:
        return self._index is not None

    def build(self, embeddings: np.ndarray, metas: List[PassageMeta]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D [n, dim]")
        if len(metas) != embeddings.shape[0]:
            raise ValueError("metas length must match embeddings rows")
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot build index over zero embeddings")

        dim = embeddings.shape[1]
        embeddings = embeddings.astype(np.float32)

        if self.index_type == "IndexFlatIP":
            self._index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unsupported faiss index_type: {self.index_type}")

        self._meta = list(metas)
        self._index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        if self._index is None:
            raise RuntimeError("Index not built yet")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[None, :]

        query_embedding = query_embedding.astype(np.float32)
        scores, indices = self._index.search(query_embedding, top_k)
        idxs = indices[0].tolist()
        scs = scores[0].tolist()

        results: List[SearchResult] = []
        for i, s in zip(idxs, scs):
            if i < 0 or i >= len(self._meta):
                continue
            m = self._meta[i]
            results.append(
                SearchResult(
                    passage_id=m.passage_id,
                    doc_id=m.doc_id,
                    chunk_id=m.chunk_id,
                    score=float(s),
                    text=m.text,
                )
            )
        return results

