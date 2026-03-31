from __future__ import annotations

import json
from pathlib import Path
from src.generation.generator import MockGenerator
from src.generation.phrasing import QueryPhrasingShiftGenerator
from src.data.loaders import load_qa_pairs


def main() -> None:
    # Small-scale test using first few examples from sample_qa.jsonl
    qa_path = Path("data/sample_qa.jsonl")
    if not qa_path.exists():
        print(f"File not found: {qa_path}")
        return

    qa_pairs = load_qa_pairs(qa_path)
    subset = qa_pairs[:3]

    generator = MockGenerator()
    phrasing_gen = QueryPhrasingShiftGenerator(generator=generator, n_shifts=3)

    print(f"Running phrasing shift on {len(subset)} examples...")
    
    results = []
    for ex in subset:
        shifts = phrasing_gen.generate_shifts(ex.query)
        print(f"\nOriginal: {ex.query}")
        for i, s in enumerate(shifts):
            print(f"  Shift {i}: {s}")
        results.append({
            "query": ex.query,
            "shifts": shifts
        })

    output_path = Path("outputs/small_scale_shifts.json")
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
