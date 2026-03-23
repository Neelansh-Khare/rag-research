from __future__ import annotations

import math
from typing import Dict, List, Sequence


def jaccard_at_k(ranking_a: Sequence[str], ranking_b: Sequence[str], k: int) -> float:
    """Jaccard overlap between top-k items of two rankings (doc-id level)."""
    if k <= 0:
        raise ValueError("k must be > 0")
    a = set(ranking_a[:k])
    b = set(ranking_b[:k])
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def jaccard_across_paraphrased_queries(
    *,
    rankings_by_query: Dict[str, Sequence[str]],
    paraphrase_groups: List[List[str]],
    k: int,
) -> Dict[str, float]:
    """Placeholder for paraphrase stability (Jaccard@k across paraphrase groups)."""
    if not paraphrase_groups:
        return {}

    vals: List[float] = []
    for group in paraphrase_groups:
        if len(group) < 2:
            continue
        base = rankings_by_query.get(group[0], [])
        for q in group[1:]:
            vals.append(jaccard_at_k(base, rankings_by_query.get(q, []), k))

    if not vals:
        return {}
    return {"jaccard_at_k_mean": float(sum(vals) / len(vals))}


def score_gap_tracking(scores_by_corpus_size: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """Score-gap tracking placeholder: top-1 minus top-k gap per corpus size."""
    out: Dict[str, float] = {}
    for size, scores in scores_by_corpus_size.items():
        if not scores:
            continue
        if len(scores) == 1:
            out[f"{size}_top1_minus_topk_gap"] = 0.0
        else:
            out[f"{size}_top1_minus_topk_gap"] = float(scores[0] - scores[-1])
    return out


def entropy_tracking(scores_by_corpus_size: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """Entropy tracking placeholder: entropy over softmax-normalized top-k scores."""
    out: Dict[str, float] = {}
    for size, scores in scores_by_corpus_size.items():
        if not scores:
            continue
        mx = max(scores)
        exps = [math.exp(s - mx) for s in scores]
        denom = sum(exps) + 1e-12
        ps = [e / denom for e in exps]
        ent = -sum(p * math.log(p + 1e-12) for p in ps)
        out[f"{size}_entropy_topk"] = float(ent)
    return out

