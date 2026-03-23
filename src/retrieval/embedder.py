from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EmbedderConfig:
    model_name: str
    device: Optional[str] = None
    normalize: bool = True


class SentenceTransformerEmbedder:
    """Thin wrapper around sentence-transformers with optional embedding normalization."""

    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        self.model = SentenceTransformer(config.model_name, device=config.device)

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of strings into a float32 matrix [n_texts, dim]."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        # sentence-transformers returns numpy arrays already, but we standardize dtype.
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize ourselves for explicitness
        ).astype(np.float32)

        if self.config.normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / norms
        return embs

