from __future__ import annotations

import pytest
from src.generation.generator import MockGenerator
from src.generation.phrasing import QueryPhrasingShiftGenerator


def test_mock_phrasing_generator() -> None:
    mock_gen = MockGenerator()
    phrasing_gen = QueryPhrasingShiftGenerator(generator=mock_gen, n_shifts=3)
    
    query = "What is RAG?"
    shifts = phrasing_gen.generate_shifts(query)
    
    assert len(shifts) == 3
    assert query in shifts
    assert "What is RAG? (1)" in shifts
    assert "What is RAG? (2)" in shifts


class ConstantGenerator:
    """A generator that always returns the same fixed text."""
    def generate(self, *, prompt: str) -> str:
        return "Shift 1\nShift 2\nShift 3"


def test_custom_generator_phrasing() -> None:
    gen = ConstantGenerator()
    phrasing_gen = QueryPhrasingShiftGenerator(generator=gen, n_shifts=3)
    
    query = "Dummy query"
    shifts = phrasing_gen.generate_shifts(query)
    
    assert len(shifts) == 3
    assert shifts == ["Shift 1", "Shift 2", "Shift 3"]


def test_extra_shifts_handled() -> None:
    gen = ConstantGenerator() # returns 3 lines
    phrasing_gen = QueryPhrasingShiftGenerator(generator=gen, n_shifts=2)
    
    shifts = phrasing_gen.generate_shifts("Dummy query")
    assert len(shifts) == 2
    assert shifts == ["Shift 1", "Shift 2"]
