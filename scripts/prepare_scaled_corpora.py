"""
Utilities for creating (placeholder) scaled corpora.

This is intentionally simple: it duplicates/samples documents from a base corpus
to reach a target number of documents while preserving `doc_id` and text.

The goal is scaffolding for corpus scaling experiments, not large-scale processing.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    text: str


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_scaled_corpus(base_corpus: List[CorpusDoc], target_docs: int, seed: int) -> List[CorpusDoc]:
    rng = random.Random(seed)
    if target_docs <= len(base_corpus):
        # Deterministic truncation for small targets.
        return list(base_corpus)[:target_docs]
    # Sample with replacement to reach target size.
    return [base_corpus[rng.randrange(0, len(base_corpus))] for _ in range(target_docs)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-corpus", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-doc-counts", required=True, help="Comma-separated integers, e.g. 1000,10000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--doc-id-field", default="doc_id")
    parser.add_argument("--text-field", default="document")
    args = parser.parse_args()

    base_path = Path(args.base_corpus)
    out_dir = Path(args.output_dir)
    base_rows = load_jsonl(base_path)

    base_docs: List[CorpusDoc] = []
    for r in base_rows:
        if args.doc_id_field not in r or args.text_field not in r:
            continue
        base_docs.append(CorpusDoc(doc_id=str(r[args.doc_id_field]), text=str(r[args.text_field])))

    if not base_docs:
        raise ValueError(f"No corpus documents found in {base_path}")

    target_counts = [int(x.strip()) for x in args.target_doc_counts.split(",") if x.strip()]
    for n in target_counts:
        scaled = build_scaled_corpus(base_docs, n, seed=args.seed)
        out_path = out_dir / f"corpus_{n}.jsonl"
        rows = [{"doc_id": d.doc_id, "document": d.text} for d in scaled]
        write_jsonl(out_path, rows)
        print(f"Wrote {len(rows)} docs to {out_path}")


if __name__ == "__main__":
    main()

