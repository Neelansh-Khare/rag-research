# rag-saturation-paper Weekly Update

## Week 1

Date: 2026-03-23

Progress:

- Repo scaffolding created for `rag-saturation-paper` (configs, src modules, docs, scripts).
- Implemented minimal runnable RAG baseline: JSONL/CSV loading, chunking, sentence-transformer embeddings, FAISS indexing, top-k retrieval, prompt construction, and generator interface (mock + OpenAI-compatible).
- Implemented week-1 evaluation from saved predictions: Exact Match, token-level F1, Recall@k, mean top-k similarity, top-1 minus top-k gap, and entropy over normalized top-k scores.

Highlights:

- Ran sanity mode on 20 examples and verified `predictions.jsonl` + `metrics.json` generation.
- Ran frozen baseline mode on 200 examples (seeded sampling with replacement to reach 200) and recorded metrics for reproducibility.

Risks / blockers:

- HF Hub downloads may be slow without an `HF_TOKEN`, but the pipeline is functional (will use cached artifacts once available).

Next week plan:

- Add scaled-corpus configs and run-sweep scaffolding (1k/10k/100k), plus basic rank-stability diagnostics integration.
- Start the first corpus-size comparison experiment and populate `docs/results_summary.md` with recorded metrics.

## Week 2

Date: 2026-03-30

Progress:

- Implemented `QueryPhrasingShiftGenerator` to create query variations for RAG stability testing.
- Added `PARAPHRASE_PROMPT` for LLM-based query variation generation.
- Created unit tests for the phrasing generator and verified with a small-scale validation script.
- Scaffolding for rank-stability diagnostics (`jaccard_at_k`, `entropy_tracking`) is ready for integration.

Highlights:

- Verified `QueryPhrasingShiftGenerator` functionality on `sample_qa.jsonl` with successful variation generation.
- Established a baseline for testing retrieval stability across paraphrased queries.

Risks / blockers:

- Cost and latency of LLM-based paraphrasing for large-scale sweeps (considering caching or offline generation).

Next week plan:

- Integrate phrasing shifts directly into the `run_pipeline.py` execution loop.
- Execute the full scale-sweep (1k, 10k, 100k) and analyze retrieval stability metrics.
- Update `docs/results_summary.md` with the first set of saturation results.

