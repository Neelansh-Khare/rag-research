# rag-saturation-paper Experiment Log

Use this log to record runs, diagnostics, and decisions.

## Experiment Entry Template

Date: YYYY-MM-DD

Experiment name / run_id:

Config file:

Mode:
- sanity (20)
- baseline (200)
- scale sweep / other

Dataset subset:
- subset_size:
- corpus_path:

Retrieval/generation:
- embedding_model:
- top_k:
- chunk_size:
- generator_type:

Metrics (from `outputs/<run_id>/metrics.json`):
- Exact Match:
- Token-level F1:
- Recall@k:
- mean top-k similarity score:
- top-1 minus top-k score gap:
- entropy over normalized top-k scores:

Observations:
- (e.g., score gap shrinking, top-k entropy increasing, failures concentrated in certain question types)

Next actions:
- (e.g., change chunking, increase k, switch embedding model, run larger N)

