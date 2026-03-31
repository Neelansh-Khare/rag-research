from __future__ import annotations

from typing import List

from src.retrieval.index import SearchResult


def build_context(retrieved: List[SearchResult], max_passages: int | None = None) -> str:
    """Format retrieved passages into a single context block."""
    if max_passages is not None:
        retrieved = retrieved[:max_passages]

    parts: List[str] = []
    for i, r in enumerate(retrieved, start=1):
        parts.append(f"[{i}] (doc_id={r.doc_id}, chunk_id={r.chunk_id}) {r.text}")
    return "\n\n".join(parts)


def build_prompt(
    *,
    question: str,
    retrieved: List[SearchResult],
    prompt_template: str,
    max_passages: int | None = None,
) -> str:
    context = build_context(retrieved, max_passages=max_passages)
    return prompt_template.format(context=context, question=question)


PARAPHRASE_PROMPT = """You are an expert at paraphrasing questions while preserving their original meaning and intent.
Generate {n_shifts} different variations of the following question.
Each variation should be on a new line. Do not add any extra text or numbering.

Question: {question}

Variations:"""

