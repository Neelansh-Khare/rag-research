from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.loaders import chunk_text_fixed_size, load_corpus, load_qa_pairs
from src.generation.generator import generator_from_config
from src.generation.prompting import build_prompt
from src.retrieval.embedder import EmbedderConfig, SentenceTransformerEmbedder
from src.retrieval.index import FaissPassageIndex, PassageMeta
from src.retrieval.retrieve import retrieve_top_k
from src.utils.config import config_to_yaml, deep_merge, ensure_output_dir, load_yaml
from src.utils.logging import setup_logging

from src.eval.metrics import exact_match, f1_token
from src.eval.retrieval_metrics import (
    aggregate_metrics,
    compute_retrieval_metrics_for_example,
)
from src.eval.stability import jaccard_at_k
from src.generation.phrasing import QueryPhrasingShiftGenerator
from src.data.redundancy import CorpusRedundancyShiftGenerator


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_config_fingerprint(config: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> str:
    payload = {"config": config, "extra": extra or {}}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]


def read_predictions_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_from_predictions_jsonl(
    *,
    predictions_path: Path,
    top_k: int,
) -> Dict[str, Any]:
    """Compute QA + retrieval metrics from a saved predictions JSONL file."""
    preds = read_predictions_jsonl(predictions_path)

    ems: List[float] = []
    f1s: List[float] = []
    retrieval_items: List[Dict[str, float]] = []
    stability_items: List[Dict[str, float]] = []

    for ex in preds:
        pred_text = str(ex.get("prediction", ""))
        gold = str(ex.get("gold_answer", ""))

        ems.append(float(exact_match(pred_text, gold)))
        f1s.append(float(f1_token(pred_text, gold)))

        retrieved_doc_ids = ex.get("retrieved_doc_ids", [])[:top_k]
        relevant_doc_ids = ex.get("relevant_doc_ids", [])
        top_k_scores = ex.get("retrieved_scores", [])[:top_k]
        retrieval_items.append(
            compute_retrieval_metrics_for_example(
                retrieved_doc_ids=retrieved_doc_ids,
                relevant_doc_ids=relevant_doc_ids,
                top_k_scores=top_k_scores,
            )
        )

        if "stability" in ex:
            stability_items.append(ex["stability"])

    retrieval_agg = aggregate_metrics(retrieval_items)
    qa_agg = {
        "exact_match": float(sum(ems) / max(1, len(ems))),
        "token_f1": float(sum(f1s) / max(1, len(f1s))),
    }
    stability_agg = aggregate_metrics(stability_items)

    # Merge aggregated metric dicts.
    all_metrics: Dict[str, Any] = {
        "qa": qa_agg,
        "retrieval": retrieval_agg,
        "stability": stability_agg,
        "n_examples": len(preds),
    }
    return all_metrics


def save_metrics(
    *,
    metrics: Dict[str, Any],
    output_dir: Path,
    run_meta: Dict[str, Any],
) -> None:
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Flat summary for easy plotting.
    summary_row: Dict[str, Any] = {"run_id": run_meta.get("run_id"), **run_meta}
    qa = metrics.get("qa", {})
    retrieval = metrics.get("retrieval", {})
    stability = metrics.get("stability", {})
    summary_row.update({f"qa.{k}": v for k, v in qa.items()})
    summary_row.update({f"retrieval.{k}": v for k, v in retrieval.items()})
    summary_row.update({f"stability.{k}": v for k, v in stability.items()})
    summary_row["n_examples"] = metrics.get("n_examples", 0)

    # Per-run summary (for easy debugging).
    per_run_summary_path = output_dir / "metrics_summary.csv"
    pd.DataFrame([summary_row]).to_csv(per_run_summary_path, index=False)

    # Global summary (appended) under the output root.
    global_summary_path = output_dir.parent / "metrics_summary.csv"
    if global_summary_path.exists():
        prev = pd.read_csv(global_summary_path)
        # Ensure we handle potentially different columns.
        next_df = pd.concat([prev, pd.DataFrame([summary_row])], ignore_index=True, sort=False)
        next_df.to_csv(global_summary_path, index=False)
    else:
        pd.DataFrame([summary_row]).to_csv(global_summary_path, index=False)


def build_passage_index(
    *,
    corpus_docs: List[Dict[str, Any]],
    embedding_model: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
    index_type: str,
    device: Optional[str] = None,
    top_k: int = 5,
    embedder: Optional[SentenceTransformerEmbedder] = None,
) -> tuple[SentenceTransformerEmbedder, FaissPassageIndex]:
    del top_k  # top_k is retrieval-time only

    if embedder is None:
        embedder = SentenceTransformerEmbedder(
            EmbedderConfig(model_name=embedding_model, device=device, normalize=True)
        )

    # Chunk corpus into fixed-size passages.
    metas: List[PassageMeta] = []
    passages: List[str] = []
    for doc in tqdm(corpus_docs, desc="Chunking corpus"):
        doc_id = str(doc["doc_id"])
        text = str(doc["text"])
        chunks = chunk_text_fixed_size(
            text=text,
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
        )
        for i, chunk in enumerate(chunks):
            passages.append(chunk)
            metas.append(
                PassageMeta(
                    passage_id=f"{doc_id}_chunk_{i:04d}",
                    doc_id=doc_id,
                    chunk_id=i,
                    text=chunk,
                )
            )

    if not passages:
        raise ValueError("No passages produced from corpus (check chunking config).")

    # Embed all passages then build FAISS index.
    embeddings = embedder.embed(passages, batch_size=32)
    index = FaissPassageIndex(index_type=index_type)
    index.build(embeddings, metas=metas)
    return embedder, index


def run_generation(
    *,
    examples: List[Dict[str, Any]],
    embedder: SentenceTransformerEmbedder,
    index: FaissPassageIndex,
    generator: Any,
    prompt_template: str,
    top_k: int,
    max_passages_in_prompt: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run retrieval + generation and return per-example prediction records."""
    predictions: List[Dict[str, Any]] = []

    for ex in tqdm(examples, desc="Retrieval+generation"):
        query = str(ex["query"])

        retrieved = retrieve_top_k(
            query=query,
            embedder=embedder,
            index=index,
            top_k=top_k,
        )

        prompt = build_prompt(
            question=query,
            retrieved=retrieved,
            prompt_template=prompt_template,
            max_passages=max_passages_in_prompt,
        )
        pred_text = generator.generate(prompt=prompt)

        retrieved_doc_ids = [r.doc_id for r in retrieved]
        retrieved_scores = [r.score for r in retrieved]
        predictions.append(
            {
                "example_id": ex["example_id"],
                "query": query,
                "gold_answer": ex["answer"],
                "prediction": pred_text,
                "relevant_doc_ids": ex.get("relevant_doc_ids", []),
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_scores": retrieved_scores,
                "retrieved_passages": [
                    {
                        "passage_id": r.passage_id,
                        "doc_id": r.doc_id,
                        "chunk_id": r.chunk_id,
                        "score": r.score,
                        "text": r.text,
                    }
                    for r in retrieved
                ],
            }
        )
    return predictions


def run_generation_with_stability(
    *,
    examples: List[Dict[str, Any]],
    embedder: SentenceTransformerEmbedder,
    index: FaissPassageIndex,
    generator: Any,
    phrasing_gen: QueryPhrasingShiftGenerator,
    prompt_template: str,
    top_k: int,
    max_passages_in_prompt: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run retrieval + generation with query shifts for stability analysis."""
    predictions: List[Dict[str, Any]] = []

    for ex in tqdm(examples, desc="Retrieval+generation (stability)"):
        query = str(ex["query"])
        shifts = phrasing_gen.generate_shifts(query)

        example_results: Dict[str, Any] = {
            "example_id": ex["example_id"],
            "query": query,
            "gold_answer": ex["answer"],
            "relevant_doc_ids": ex.get("relevant_doc_ids", []),
            "shifts": [],
        }

        shift_rankings: List[List[str]] = []

        for i, s_query in enumerate(shifts):
            retrieved = retrieve_top_k(
                query=s_query,
                embedder=embedder,
                index=index,
                top_k=top_k,
            )

            prompt = build_prompt(
                question=s_query,
                retrieved=retrieved,
                prompt_template=prompt_template,
                max_passages=max_passages_in_prompt,
            )
            pred_text = generator.generate(prompt=prompt)

            retrieved_doc_ids = [r.doc_id for r in retrieved]
            retrieved_scores = [r.score for r in retrieved]
            shift_rankings.append(retrieved_doc_ids)

            shift_data = {
                "query": s_query,
                "prediction": pred_text,
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_scores": retrieved_scores,
            }
            example_results["shifts"].append(shift_data)

            if i == 0:
                # Store primary results at top level for compatibility.
                example_results.update(shift_data)

        # Compute Jaccard@k between original (shifts[0]) and all other shifts.
        jaccards = []
        if len(shift_rankings) > 1:
            base_ranking = shift_rankings[0]
            for other_ranking in shift_rankings[1:]:
                jaccards.append(jaccard_at_k(base_ranking, other_ranking, k=top_k))

        example_results["stability"] = {
            "jaccard_at_k": float(np.mean(jaccards)) if jaccards else 1.0,
            "n_shifts": len(shifts),
        }
        predictions.append(example_results)

    return predictions


def run_generation_with_corpus_stability(
    *,
    examples: List[Dict[str, Any]],
    embedder: SentenceTransformerEmbedder,
    corpus_docs: List[Any],
    chunk_cfg: Dict[str, Any],
    index_type: str,
    generator: Any,
    stability_cfg: Dict[str, Any],
    prompt_template: str,
    top_k: int,
    n_shifts: int,
    max_passages_in_prompt: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run retrieval + generation with corpus redundancy shifts for stability analysis."""
    predictions: List[Dict[str, Any]] = []

    redundancy_factor = int(stability_cfg.get("redundancy_factor", 2))
    
    # Build shifted indices.
    indices: List[FaissPassageIndex] = []
    for i in range(n_shifts):
        # Shift 0: 1x, Shift 1: R, Shift 2: R^2, ...
        factor = redundancy_factor ** i
        print(f"Building redundant index {i} (factor={factor}x)...")
        
        # Use existing CorpusDoc objects from loaders.py
        gen = CorpusRedundancyShiftGenerator(redundancy_factor=factor, seed=42 + i)
        shifted_docs = gen.apply(corpus_docs)
        
        _, shift_index = build_passage_index(
            corpus_docs=[{"doc_id": d.doc_id, "text": d.text} for d in shifted_docs],
            embedding_model="", # Not used if embedder is provided
            chunk_size_words=int(chunk_cfg.get("chunk_size_words", 80)),
            chunk_overlap_words=int(chunk_cfg.get("chunk_overlap_words", 10)),
            index_type=index_type,
            embedder=embedder,
        )
        indices.append(shift_index)

    for ex in tqdm(examples, desc="Retrieval+generation (corpus stability)"):
        query = str(ex["query"])
        
        example_results: Dict[str, Any] = {
            "example_id": ex["example_id"],
            "query": query,
            "gold_answer": ex["answer"],
            "relevant_doc_ids": ex.get("relevant_doc_ids", []),
            "shifts": [],
        }

        shift_rankings: List[List[str]] = []

        for i, index in enumerate(indices):
            retrieved = retrieve_top_k(
                query=query,
                embedder=embedder,
                index=index,
                top_k=top_k,
            )

            prompt = build_prompt(
                question=query,
                retrieved=retrieved,
                prompt_template=prompt_template,
                max_passages=max_passages_in_prompt,
            )
            pred_text = generator.generate(prompt=prompt)

            retrieved_doc_ids = [r.doc_id for r in retrieved]
            retrieved_scores = [r.score for r in retrieved]
            shift_rankings.append(retrieved_doc_ids)

            shift_data = {
                "query": query,
                "prediction": pred_text,
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieved_scores": retrieved_scores,
                "redundancy_factor": redundancy_factor ** i,
            }
            example_results["shifts"].append(shift_data)

            if i == 0:
                example_results.update(shift_data)

        # Compute Jaccard@k between original (shift 0) and all other shifts.
        jaccards = []
        if len(shift_rankings) > 1:
            base_ranking = shift_rankings[0]
            for other_ranking in shift_rankings[1:]:
                jaccards.append(jaccard_at_k(base_ranking, other_ranking, k=top_k))

        example_results["stability"] = {
            "jaccard_at_k": float(np.mean(jaccards)) if jaccards else 1.0,
            "n_shifts": len(indices),
        }
        predictions.append(example_results)

    return predictions


def select_subset(
    qa_pairs: List[Any],
    subset_size: int,
    seed: int,
) -> List[Any]:
    """Select exactly `subset_size` examples, sampling with replacement if needed."""
    if subset_size <= 0:
        raise ValueError("subset_size must be > 0")
    if subset_size <= len(qa_pairs):
        # Deterministic subset: keep first N rows.
        return qa_pairs[:subset_size]

    rng = random.Random(seed)
    out: List[Any] = list(qa_pairs)
    while len(out) < subset_size:
        out.append(rng.choice(qa_pairs))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["sanity", "baseline", "scale_sweep"], default="baseline")
    parser.add_argument("--output-root", default=None, help="Override output root (e.g. outputs/baseline)")
    args = parser.parse_args()

    base_config = load_yaml(args.config)

    # Mode -> fixed subset sizes for week-1 reproducibility.
    mode_overrides: Dict[str, Any] = {}
    if args.mode == "sanity":
        mode_overrides = {"run": {"subset_size": 20}}
    elif args.mode == "baseline":
        mode_overrides = {"run": {"subset_size": 200}}
    else:
        # For scaffolding runs we keep whatever is set in the config.
        mode_overrides = {}

    config = deep_merge(base_config, mode_overrides)

    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    output_root = args.output_root or config.get("run", {}).get("output_dir", "outputs")
    output_dir = Path(output_root)
    ensure_output_dir(output_dir)

    run_id = stable_config_fingerprint(
        config,
        extra={"mode": args.mode, "subset_size": config.get("run", {}).get("subset_size")},
    )
    run_dir = output_dir / run_id
    ensure_output_dir(run_dir)

    setup_logging(run_dir)
    import logging

    log = logging.getLogger("run_pipeline")
    log.info("Starting run_id=%s mode=%s", run_id, args.mode)

    # Save full config used for each run.
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        f.write(config_to_yaml(config))
    with (run_dir / "cli_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    subset_size = int(config.get("run", {}).get("subset_size", 200))

    # Load corpus + QA pairs.
    data_cfg = config.get("data", {})
    corpus_docs = load_corpus(
        corpus_path=data_cfg["corpus_path"],
        text_field=data_cfg.get("corpus_text_field", "document"),
        doc_id_field=data_cfg.get("corpus_doc_id_field", "doc_id"),
    )
    # `load_qa_pairs` is also used for relevant_doc_ids.
    qa_pairs = load_qa_pairs(
        qa_path=data_cfg["qa_path"],
        query_field=data_cfg.get("qa_query_field", "query"),
        answer_field=data_cfg.get("qa_answer_field", "answer"),
        relevant_doc_ids_field=data_cfg.get("qa_relevant_doc_ids_field", "relevant_doc_ids"),
    )

    # For week-1 MVP: subset QA examples by deterministic selection.
    qa_pairs = select_subset(qa_pairs, subset_size=subset_size, seed=seed)

    # Build FAISS index.
    retrieval_cfg = config.get("retrieval", {})
    chunk_cfg = retrieval_cfg.get("chunking", {})
    chunk_size_words = int(chunk_cfg.get("chunk_size_words", 80))
    chunk_overlap_words = int(chunk_cfg.get("chunk_overlap_words", 10))
    embedding_model = str(retrieval_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    top_k = int(retrieval_cfg.get("top_k", 5))
    index_type = str(retrieval_cfg.get("faiss", {}).get("index_type", "IndexFlatIP"))

    embedder, index = build_passage_index(
        corpus_docs=[{"doc_id": d.doc_id, "text": d.text} for d in corpus_docs],
        embedding_model=embedding_model,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
        index_type=index_type,
        device=None,
        top_k=top_k,
    )

    prompt_template = str(config.get("generation", {}).get("prompt_template", "{context}\n\n{question}"))

    generator_type = str(config.get("generation", {}).get("generator_type", "mock"))
    generator = generator_from_config(
        generator_type=generator_type,
        generation_config=config.get("generation", {}),
    )

    # Prepare examples in plain dicts for JSON serialization.
    examples = [
        {
            "example_id": q.example_id,
            "query": q.query,
            "answer": q.answer,
            "relevant_doc_ids": q.relevant_doc_ids,
        }
        for q in qa_pairs
    ]

    stability_cfg = config.get("stability", {})
    n_shifts = int(stability_cfg.get("n_shifts", 0))
    stability_type = str(stability_cfg.get("type", "phrasing"))

    if n_shifts > 0:
        log.info("Running with stability shifts (type=%s, n_shifts=%d)", stability_type, n_shifts)
        if stability_type == "corpus_redundancy":
            predictions = run_generation_with_corpus_stability(
                examples=examples,
                embedder=embedder,
                corpus_docs=corpus_docs,
                chunk_cfg=chunk_cfg,
                index_type=index_type,
                generator=generator,
                stability_cfg=stability_cfg,
                prompt_template=prompt_template,
                top_k=top_k,
                n_shifts=n_shifts,
            )
        else:
            phrasing_gen = QueryPhrasingShiftGenerator(
                generator=generator,
                n_shifts=n_shifts,
            )
            predictions = run_generation_with_stability(
                examples=examples,
                embedder=embedder,
                index=index,
                generator=generator,
                phrasing_gen=phrasing_gen,
                prompt_template=prompt_template,
                top_k=top_k,
            )
    else:
        predictions = run_generation(
            examples=examples,
            embedder=embedder,
            index=index,
            generator=generator,
            prompt_template=prompt_template,
            top_k=top_k,
        )

    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Evaluate strictly from the saved predictions file.
    metrics = evaluate_from_predictions_jsonl(
        predictions_path=predictions_path,
        top_k=top_k,
    )

    run_meta = {
        "run_name": config.get("run_name", ""),
        "mode": args.mode,
        "seed": seed,
        "subset_size": subset_size,
        "corpus_path": str(data_cfg.get("corpus_path", "")),
        "qa_path": str(data_cfg.get("qa_path", "")),
        "embedding_model": embedding_model,
        "top_k": top_k,
        "chunk_size_words": chunk_size_words,
        "chunk_overlap_words": chunk_overlap_words,
        "generator_type": generator_type,
        "index_type": index_type,
        "run_timestamp": int(time.time()),
        "run_id": run_id,
    }
    save_metrics(metrics=metrics, output_dir=run_dir, run_meta=run_meta)

    log.info("Finished. Metrics saved to %s", run_dir)


if __name__ == "__main__":
    main()

