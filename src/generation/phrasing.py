from __future__ import annotations

from typing import List

from src.generation.generator import BaseGenerator, MockGenerator
from src.generation.prompting import PARAPHRASE_PROMPT


class QueryPhrasingShiftGenerator:
    """Generator for creating variations (shifts) of a query to test RAG stability."""

    def __init__(
        self,
        generator: BaseGenerator,
        n_shifts: int = 3,
        prompt_template: str = PARAPHRASE_PROMPT,
    ) -> None:
        self.generator = generator
        self.n_shifts = n_shifts
        self.prompt_template = prompt_template

    def generate_shifts(self, query: str) -> List[str]:
        """Generate a list of paraphrased variations of the input query."""
        if isinstance(self.generator, MockGenerator):
            # For testing: return original query and simple variations.
            return [query] + [f"{query} ({i+1})" for i in range(self.n_shifts - 1)]

        prompt = self.prompt_template.format(question=query, n_shifts=self.n_shifts)
        response = self.generator.generate(prompt=prompt)

        # Split by newlines and clean up.
        shifts = [s.strip() for s in response.split("\n") if s.strip()]
        
        # Ensure we have the right number of shifts if possible.
        if len(shifts) > self.n_shifts:
            shifts = shifts[:self.n_shifts]
        
        return shifts
