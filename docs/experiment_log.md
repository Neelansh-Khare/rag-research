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

## Experiment Entries

### 2026-03-29

**Summary:** Implement Query Phrasing Shift generator; Small-scale tests

**Details:**
- Implemented `QueryPhrasingShiftGenerator` in `src/generation/phrasing.py`.
- Supports both `MockGenerator` for testing and `OpenAICompatibleGenerator` for real use.
- Added a basic `PARAPHRASE_PROMPT` in `src/generation/prompting.py`.
- Verified with unit tests in `tests/test_phrasing.py`.
- Ran small-scale test script `scripts/test_shifts_small_scale.py` on `data/sample_qa.jsonl`.

**Results:**
- Generated query variations successfully saved to `outputs/small_scale_shifts.json`.
- Initial shift results show correctly formatted paraphrases (with mock shifts as placeholders).
- Repro baseline results: Standard baseline pipeline remains functional.

**Next actions:**
- Integrate phrasing shifts into `run_pipeline.py` or create a new stability evaluation script.
- Evaluate retrieval stability (Jaccard@k) across these shifts for different corpus scales.

### 2026-04-05

**Summary:** Full Query Shift experiments; Log retrieval vs generation degradation

**Details:**
- Integrated `QueryPhrasingShiftGenerator` into `src.pipeline.run_pipeline`.
- Added `run_generation_with_stability` to compute Jaccard@k across query shifts.
- Updated metrics logging to include `stability.jaccard_at_k` in `metrics.json` and `metrics_summary.csv`.
- Executed full scale sweep (1k, 10k, 100k) using `configs/scale_*.yaml` with `n_shifts: 3`.

**Results:**
- **1k corpus:** EM=1.0, Jaccard@k=1.0, Mean similarity=0.6204
- **10k corpus:** EM=1.0, Jaccard@k=1.0, Mean similarity=0.6204
- **100k corpus:** EM=1.0, Jaccard@k=1.0, Mean similarity=0.6204
- *Note:* Metrics are identical due to `MockGenerator` and small base corpus sampled with replacement. The infrastructure is now verified and ready for real scaling tests with larger, unique corpora.

**Next actions:**
- Conduct gap analysis on retrieval discriminability as corpus size increases further.
- Test with non-mock generator to observe real QA degradation.

