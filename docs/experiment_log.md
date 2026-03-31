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

