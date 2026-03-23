from __future__ import annotations

from typing import List

import numpy as np

from src.retrieval.embedder import SentenceTransformerEmbedder
from src.retrieval.index import FaissPassageIndex, SearchResult


def retrieve_top_k(
    *,
    query: str,
    embedder: SentenceTransformerEmbedder,
    index: FaissPassageIndex,
    top_k: int,
) -> List[SearchResult]:
    """Embed the query and return FAISS top-k passage results."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    q_emb: np.ndarray = embedder.embed([query])  # shape [1, dim]
    results = index.search(q_emb[0], top_k=top_k)
    return results

