"""Different mathematical and statistical formulas and utilities."""

from __future__ import annotations

import math
import numbers
import typing
from collections.abc import Callable, Iterable

import numpy as np


def catalan_number(n: int) -> int:
    """Computes the n-th catalan number. See https://en.wikipedia.org/wiki/Catalan_number."""
    return round(math.comb(2 * n, n) / (n + 1))


def jaccard(a: set | frozenset, b: set | frozenset) -> float:
    """Jaccard coefficient between a and b. Defined as |a ∩ b| / |a ∪ b|"""
    return len(a & b) / len(a | b)


T = typing.TypeVar("T")


def score_matrix(elems: Iterable[T], scoring: Callable[[T, T], numbers.Number]) -> np.ndarray:
    elems = list(elems)
    n = len(elems)

    matrix = np.ones((n, n))
    for i, elem_i in enumerate(elems):
        for j, elem_j in enumerate(elems):
            matrix[i, j] = scoring(elem_i, elem_j)

    return matrix


def trigrams(text: str) -> list[str]:
    """Generates the trigrams of a given text."""
    text = f"  {text.lower()}  "
    return [text[i : i + 3] for i in range(len(text) - 2)]
