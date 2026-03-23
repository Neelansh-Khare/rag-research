from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]) -> float:
    """Recall@k based on doc-id overlap."""
    rel = list(relevant_doc_ids)
    if not rel:
        return 0.0
    retrieved_set = set(retrieved_doc_ids)
    overlap = sum(1 for d in rel if d in retrieved_set)
    return overlap / len(rel)


def mean_top_k_score(top_k_scores: Sequence[float]) -> float:
    if not top_k_scores:
        return 0.0
    return float(sum(top_k_scores) / len(top_k_scores))


def top1_minus_topk_gap(top_k_scores: Sequence[float]) -> float:
    if not top_k_scores:
        return 0.0
    if len(top_k_scores) == 1:
        return 0.0
    return float(top_k_scores[0] - top_k_scores[-1])


def entropy_over_scores(top_k_scores: Sequence[float]) -> float:
    """Entropy of softmax-normalized top-k scores."""
    if not top_k_scores:
        return 0.0
    # Softmax for numerical stability.
    mx = max(top_k_scores)
    exps = [math.exp(s - mx) for s in top_k_scores]
    denom = sum(exps) + 1e-12
    ps = [e / denom for e in exps]
    ent = -sum(p * math.log(p + 1e-12) for p in ps)
    return float(ent)


def compute_retrieval_metrics_for_example(
    *,
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    top_k_scores: Sequence[float],
) -> Dict[str, float]:
    return {
        "recall_at_k": recall_at_k(retrieved_doc_ids, relevant_doc_ids),
        "mean_top_k_similarity": mean_top_k_score(top_k_scores),
        "top1_minus_topk_gap": top1_minus_topk_gap(top_k_scores),
        "entropy_topk_scores": entropy_over_scores(top_k_scores),
    }


def aggregate_metrics(items: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    items_list = list(items)
    if not items_list:
        return {}

    keys = set()
    for d in items_list:
        keys |= set(d.keys())

    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(d.get(k, 0.0)) for d in items_list]
        out[k] = sum(vals) / len(vals)
    return out

