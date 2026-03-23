# rag-saturation-paper Draft Outline

## Workshop-Style Paper Structure

1. **Motivation**
   - Retrieval-augmented generation (RAG) depends on discriminative retrieval.
   - But retrieval is performed over finite embeddings and approximate nearest neighbors; at large `N`, top-k can become crowded and unstable.
2. **Problem Statement**
   - Define retrieval saturation: retrieval becomes less discriminative and ranking stability degrades as corpus size grows.
   - Downstream effect: QA quality may plateau or degrade once retrieval saturation begins.
3. **Method**
   - Minimal RAG pipeline (chunking -> embeddings -> FAISS -> top-k -> prompt -> generator).
   - Controlled corpus scaling experiment (vary `N`, keep retrieval hyperparameters constant initially).

## Retrieval Discriminability Collapse

We operationalize discriminability collapse using retrieval score diagnostics as corpus size `N` increases (with retrieval depth `k` fixed): decreasing `top1_minus_topk_gap`, drifting `mean_top_k_similarity`, and increasing entropy over normalized top-k similarity scores. We interpret these as the retriever producing a less separable score distribution, i.e., the top-k set becomes “crowded” with near-ties.

## Rank Instability

We test rank instability by measuring overlap between top-k results across corpus-size neighbors. Primary signals include Jaccard@k between doc-id rankings and (for future paraphrased-query groups) an aggregate “Jaccard across paraphrases” stability statistic. Evidence of instability is a systematic drop in overlap as `N` increases.

## Downstream Saturation

We quantify downstream saturation by tracking Exact Match and token-level F1 over increasing `N`. The key empirical claim is that QA improvements plateau or degrade after retrieval discriminability/rank-instability signals cross a threshold. We also record where failures concentrate (e.g., questions with weak lexical grounding).

## Discussion and Practical Implications

We connect the saturation onset to system design knobs: when retriever capacity or reranking is necessary, how retrieval depth `k` changes the stability/quality trade-off, and how evaluation protocols should reflect saturation (rather than assuming monotonic gains with larger corpora).

## Limitations and Future Work
   - Corpus/domain mismatch, approximate search artifacts, generator sensitivity, embedding model effects.

## (Optional) Appendices

- Hyperparameters table, computational budget, qualitative examples.

