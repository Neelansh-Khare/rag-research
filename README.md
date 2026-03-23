# rag-saturation-paper

Research codebase for studying *retrieval saturation* in retrieval-augmented generation (RAG) systems.

Hypothesis: as corpus size increases, retrieval becomes less discriminative, top-k rankings become less stable, and downstream QA performance plateaus or degrades.

This repo provides:
- A minimal, runnable RAG baseline (FAISS + sentence-transformers + pluggable generators)
- Reusable evaluation utilities (EM/F1, retrieval recall, and ranking-score stability signals)
- Scaffolding for future corpus-scaling and rank-stability experiments

## Quick start

1. Install dependencies: `pip install -r requirements.txt`
2. Run the baseline sanity test:
   - `bash scripts/run_20_sample_sanity.sh`

Outputs and run configs are written under `outputs/`.

