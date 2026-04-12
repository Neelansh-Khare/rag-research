from __future__ import annotations

import random
from typing import List

from src.data.loaders import CorpusDoc


class CorpusRedundancyShiftGenerator:
    """Generator for creating redundant variations of a corpus to test RAG stability."""

    def __init__(self, redundancy_factor: int = 1, seed: int = 42) -> None:
        self.redundancy_factor = redundancy_factor
        self.seed = seed

    def apply(self, corpus: List[CorpusDoc]) -> List[CorpusDoc]:
        """Apply redundancy shift to the corpus.
        
        This currently supports exact duplication of documents.
        Each document in the original corpus is repeated `redundancy_factor` times.
        To ensure unique IDs for FAISS internal metadata (if needed), 
        we can append a suffix to the doc_id, but the project seems to handle 
        duplicate doc_ids fine for retrieval metrics as long as we track them correctly.
        """
        if self.redundancy_factor <= 1:
            return list(corpus)

        expanded_corpus: List[CorpusDoc] = []
        for doc in corpus:
            for i in range(self.redundancy_factor):
                # We keep the same doc_id so that retrieval metrics (Recall@k) 
                # still work by matching the original doc_id.
                expanded_corpus.append(
                    CorpusDoc(
                        doc_id=doc.doc_id,
                        text=doc.text
                    )
                )
        
        # Shuffle to avoid having all duplicates clustered together, 
        # which might affect some retrieval algorithms (though not basic FAISS).
        rng = random.Random(self.seed)
        rng.shuffle(expanded_corpus)
        
        return expanded_corpus
