from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import pandas as pd


@dataclass(frozen=True)
class CorpusDoc:
    doc_id: str
    text: str


@dataclass(frozen=True)
class QAPair:
    query: str
    answer: str
    relevant_doc_ids: List[str]
    example_id: str


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_corpus(
    corpus_path: str | Path,
    text_field: str = "document",
    doc_id_field: str = "doc_id",
) -> List[CorpusDoc]:
    """Load corpus documents from JSONL or CSV.

    Expected schema:
    - JSONL: each row has at least `doc_id_field` and `text_field`
    - CSV: columns named `doc_id_field` and `text_field`
    """
    p = Path(corpus_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() == ".jsonl":
        rows = _read_jsonl(p)
        docs: List[CorpusDoc] = []
        for r in rows:
            if doc_id_field not in r or text_field not in r:
                continue
            docs.append(CorpusDoc(doc_id=str(r[doc_id_field]), text=str(r[text_field])))
        if not docs:
            raise ValueError(f"No corpus docs found in JSONL: {p}")
        return docs

    if p.suffix.lower() == ".csv":
        df = _read_csv(p)
        if doc_id_field not in df.columns or text_field not in df.columns:
            raise ValueError(f"CSV missing required columns: {doc_id_field}, {text_field}")
        return [CorpusDoc(doc_id=str(row[doc_id_field]), text=str(row[text_field])) for _, row in df.iterrows()]

    raise ValueError(f"Unsupported corpus format: {p.suffix}")


def _parse_relevant_doc_ids(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    # Allow single id stored as string
    return [str(value)]


def load_qa_pairs(
    qa_path: str | Path,
    query_field: str = "query",
    answer_field: str = "answer",
    relevant_doc_ids_field: str = "relevant_doc_ids",
    relevant_doc_id_field: str = "relevant_doc_id",
) -> List[QAPair]:
    """Load QA pairs from JSONL or CSV.

    Expected schema:
    - query_field: string
    - answer_field: string
    - relevant_doc_ids_field or relevant_doc_id_field: doc ids for Recall@k
    """
    p = Path(qa_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    records: List[Dict[str, Any]]
    if p.suffix.lower() == ".jsonl":
        records = _read_jsonl(p)
    elif p.suffix.lower() == ".csv":
        df = _read_csv(p)
        records = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported QA format: {p.suffix}")

    out: List[QAPair] = []
    for i, r in enumerate(records):
        if query_field not in r or answer_field not in r:
            continue
        rel = None
        if relevant_doc_ids_field in r:
            rel = _parse_relevant_doc_ids(r.get(relevant_doc_ids_field))
        elif relevant_doc_id_field in r:
            rel = _parse_relevant_doc_ids(r.get(relevant_doc_id_field))
        else:
            rel = []

        out.append(
            QAPair(
                query=str(r[query_field]),
                answer=str(r[answer_field]),
                relevant_doc_ids=rel,
                example_id=str(r.get("id", f"ex_{i:05d}")),
            )
        )
    if not out:
        raise ValueError(f"No QA pairs found in: {p}")
    return out


def chunk_text_fixed_size(
    text: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> List[str]:
    """Chunk text into fixed-size word passages with optional overlap."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be > 0")
    if chunk_overlap_words < 0:
        raise ValueError("chunk_overlap_words must be >= 0")
    if chunk_overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")

    words = text.split()
    if not words:
        return []

    step = chunk_size_words - chunk_overlap_words
    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks

