from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple


_punct_re = re.compile(r"[^\w\s]")


def normalize_text(s: str) -> str:
    """Normalize answers for EM/F1 (lowercase + punctuation stripping)."""
    s = s.strip().lower()
    s = _punct_re.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    if not s:
        return []
    return s.split(" ")


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def f1_token(pred: str, gold: str) -> float:
    """Token-level F1 between prediction and gold."""
    pred_toks = tokenize(pred)
    gold_toks = tokenize(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2.0 * precision * recall / (precision + recall)


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den

