# rag-saturation-paper Plan

## Research Question

How does retrieval saturation manifest in retrieval-augmented generation (RAG) as corpus size increases?

Concretely: does retrieval become less discriminative (worse top-k separation and rank stability) and does downstream QA performance plateau or degrade?

## Hypotheses

H1. As corpus size `N` increases (holding retrieval depth `k` fixed), top-k similarity scores become less separable (reduced score gap), indicating retrieval discriminability collapse.

H2. As corpus size `N` increases, the composition of top-k rankings becomes less stable (higher rank instability / lower rank overlap).

H3. As retrieval becomes less discriminative and less stable, downstream QA metrics plateau or degrade after some effective corpus size.

H4. Rank instability signals (e.g., entropy over top-k scores, top-1 minus top-k score gap) correlate with the onset of downstream saturation.

## First-week Milestones (baseline MVP)

1. Implement a runnable minimal RAG baseline:
   - JSONL/CSV loaders for corpus + QA pairs
   - passage chunking
   - sentence-transformer embedding
   - FAISS indexing and top-k retrieval
   - pluggable generator interface (mock + OpenAI-compatible)
2. Implement week-1 evaluation metrics:
   - Exact Match and token-level F1
   - Retrieval Recall@k
   - mean top-k similarity score
   - top-1 minus top-k score gap
   - entropy over normalized top-k similarity scores
3. Add reproducible config + logging:
   - frozen baseline config
   - seed control
   - save full config + predictions + metrics
4. Run sanity/baseline experiments on 20 and 200 examples and record results.

## Completed This Week

1. Built repo scaffolding and a runnable baseline pipeline (chunking -> embedding -> FAISS -> top-k -> prompt -> generation).
2. Implemented evaluation metrics (EM/F1 + retrieval diagnostics) computed from saved `predictions.jsonl`.
3. Added frozen baseline configuration + seed control and run scripts for 20-sample sanity and 200-sample baseline (with deterministic sampling when dataset < subset_size).
4. Added rank-stability scaffolding (`src/eval/stability.py`) and corpus scaling config skeletons (`configs/scale_*.yaml` + `scripts/run_scale_sweep.sh`).

## Evidence for "Saturation"

Saturation is operationally defined as:

- Retrieval discriminability collapse: decreasing `top1 - topk_gap` and/or decreasing mean top-k similarity score across increasing corpus size.
- Rank instability: decreasing rank overlap signals across corpus sizes.
- Downstream saturation: QA EM/F1 plateauing or degrading despite increased corpus size.

## Experiment Design: Varying `N` and `k`

We vary corpus size `N` and retrieval depth `k` in a controlled grid:

- Fix `k` while sweeping `N` upward (H1, H2, H3, H4).
- Fix `N` while sweeping `k` across a small set (e.g., `k in {1, 5, 10}`) to test whether deeper retrieval postpones saturation or accelerates rank instability (H2/H4).

What counts as saturation evidence:

- Retrieval discriminability collapse: `top1_minus_topk_gap` decreases and/or `mean_top_k_similarity` drifts toward a flatter distribution as `N` increases.
- Rank instability: top-k entropy increases and rank-overlap/Jaccard@k decreases across corpus-size neighbors.
- Downstream saturation: QA EM/F1 stops improving and then plateaus or degrades despite increasing `N`.

