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

## Week 3

Date: 2026-04-05

Progress:

- Completed integration of `QueryPhrasingShiftGenerator` into `run_pipeline.py`.
- Added `run_generation_with_stability` logic to compute rank stability (Jaccard@k) across query variations.
- Prepared scaled corpora (1k, 10k, 100k) from the base dataset.
- Executed full scale-sweep for all three sizes and successfully logged stability metrics to `metrics_summary.csv`.

Highlights:

- Infrastructure for tracking retrieval saturation (discriminability collapse + rank instability) is now fully operational.
- Verified that stability metrics can be correctly aggregated and saved alongside traditional QA metrics.

Risks / blockers:

- Current experiments use a small base corpus with replacement; results are stable but do not yet show "saturation."
- Scaling to much larger, unique corpora will be the next challenge for storage and indexing time.

Next week plan:

- Source or generate a significantly larger unique corpus (e.g., 1M+ docs) to observe actual saturation effects.
- Conduct a deeper gap analysis on top-k scores to detect discriminability collapse.

