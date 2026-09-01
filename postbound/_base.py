from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

T = TypeVar("T")
"""Typed expressions use this generic type variable."""

pbdir = Path.home() / ".postbound"


class SupportsLT[T](Protocol):
    def __lt__(self, other: T) -> bool: ...


class SupportsGT[T](Protocol):
    def __gt__(self, other: T) -> bool: ...


class SupportsSubtraction[T](Protocol):
    def __sub__(self, other: T) -> T: ...


type SupportsRichComparison[T] = SupportsLT[T] | SupportsGT[T]
