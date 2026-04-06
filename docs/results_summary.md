# rag-saturation-paper Results Summary

## Baseline (fixed retriever + generator)

Record results for the minimal baseline pipeline.

### 20-sample sanity run

- Config: `configs/frozen_baseline.yaml` (sanity mode)
- Subset size: 20
- Corpus size: 20 documents
- Retrieval `k`: 5
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Generator: mock

| Metric | Value |
|---|---|
| Exact Match | 1.0 |
| Token-level F1 | 1.0 |
| Recall@k | 1.0 |
| Mean top-k similarity score | 0.3493 |
| Top-1 minus top-k score gap | 0.4003 |
| Entropy over top-k scores | 1.5969 |

### 200-sample baseline run

- Config: `configs/frozen_baseline.yaml` (baseline mode)
- Subset size: 200
- Corpus size: 20 documents (queries sampled with replacement to reach 200)
- Retrieval `k`: 5

| Metric | Value |
|---|---|
| Exact Match | 1.0 |
| Token-level F1 | 1.0 |
| Recall@k | 1.0 |
| Mean top-k similarity score | 0.3505 |
| Top-1 minus top-k score gap | 0.3971 |
| Entropy over top-k scores | 1.5972 |

## Scaling experiments

Record results for increasing corpus size `N` (with `k=5`, mock generator).

| Corpus Size `N` | EM | Recall@k | Jaccard@k (Stability) | Mean similarity | Top1-Topk Gap | Entropy |
|---|---|---|---|---|---|---|
| 1,000 | 1.0 | 1.0 | 1.0 | 0.6204 | 0.0 | 1.6094 |
| 10,000 | 1.0 | 1.0 | 1.0 | 0.6204 | 0.0 | 1.6094 |
| 100,000 | 1.0 | 1.0 | 1.0 | 0.6204 | 0.0 | 1.6094 |

*Observation (2026-04-05):* Metrics are stable due to the use of a base corpus (22 items) sampled with replacement. The pipeline infrastructure is validated for scaling. Real-world saturation will be tested next with larger and more unique corpora.

