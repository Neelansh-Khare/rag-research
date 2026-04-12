from src.data.redundancy import CorpusRedundancyShiftGenerator
from src.data.loaders import CorpusDoc


def test_redundancy_shift_no_change():
    gen = CorpusRedundancyShiftGenerator(redundancy_factor=1)
    corpus = [
        CorpusDoc(doc_id="1", text="text 1"),
        CorpusDoc(doc_id="2", text="text 2"),
    ]
    shifted = gen.apply(corpus)
    assert len(shifted) == 2
    assert set(d.doc_id for d in shifted) == {"1", "2"}


def test_redundancy_shift_multiplication():
    gen = CorpusRedundancyShiftGenerator(redundancy_factor=3, seed=42)
    corpus = [
        CorpusDoc(doc_id="1", text="text 1"),
        CorpusDoc(doc_id="2", text="text 2"),
    ]
    shifted = gen.apply(corpus)
    assert len(shifted) == 6
    # Each doc should appear 3 times.
    doc_counts = {}
    for d in shifted:
        doc_counts[d.doc_id] = doc_counts.get(d.doc_id, 0) + 1
    assert doc_counts["1"] == 3
    assert doc_counts["2"] == 3


def test_redundancy_shift_shuffling():
    # Use a larger corpus to make non-shuffled very unlikely.
    gen = CorpusRedundancyShiftGenerator(redundancy_factor=2, seed=42)
    corpus = [CorpusDoc(doc_id=str(i), text=f"text {i}") for i in range(10)]
    shifted = gen.apply(corpus)
    
    # Check if it's NOT in the order [0, 0, 1, 1, 2, 2, ...]
    # Or [0, 1, 2, ..., 0, 1, 2, ...] depending on implementation.
    # My implementation appends all duplicates of a doc then shuffles.
    # So if it's NOT shuffled, it would be [0, 0, 1, 1, ...].
    
    ids = [d.doc_id for d in shifted]
    non_shuffled_ids = []
    for i in range(10):
        non_shuffled_ids.extend([str(i), str(i)])
    
    assert ids != non_shuffled_ids
