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

