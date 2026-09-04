"""Provides additional type hints, type decorators, ..."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable

from .._base import T


def deprecated(func: Callable) -> Callable:
    """Indicates that the given function or class should no longer be used."""

    @functools.wraps(func)
    def deprecation_wrapper(*args, **kwargs) -> Callable:
        warnings.warn(
            f"Usage of {getattr(func, '__name__', func)} is deprecated",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return deprecation_wrapper


def module_local(func: Callable) -> Callable:
    """
    Marker decorator to show that a seemingly private method of a class is intended to be used by other objects from
    the same module.
    """
    return func


Lazy = None
"""A placeholder to indicate that a value is not yet computed, but will be computed lazily."""

LazyVal = T | Lazy
"""Type hint for a value that is computed lazily."""
