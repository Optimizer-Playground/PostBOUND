"""Utilities centered around numbers."""

from __future__ import annotations

import math
import numbers
import threading
from typing import Any


def represents_number(val: str) -> bool:
    """Checks, whether `val` can be cast into an integer/float value."""
    try:
        float(val)
    except (TypeError, ValueError):
        return False
    return True


class AtomicInt(numbers.Integral):
    """An atomic int allows for multi-threaded access to the integer value."""

    def __init__(self, value: int = 0):
        self._value = value
        self._lock = threading.Lock()

    def increment(self, by: int = 1) -> None:
        with self._lock:
            self._value += by

    def reset(self) -> None:
        with self._lock:
            self._value = 0

    def _get_value(self) -> int:
        with self._lock:
            return self._value

    def _set_value(self, value: int) -> None:
        with self._lock:
            self._value = value

    def _assert_integral(self, other: Any):
        if not isinstance(other, numbers.Integral):
            raise TypeError(f"Cannot add argument of type {type(other)} to object of type AtomicInt")

    def _unwrap_atomic(self, other: Any):
        return other._value if isinstance(other, AtomicInt) else other

    value = property(_get_value, _set_value)

    def __abs__(self) -> int:
        with self._lock:
            return abs(self._value)

    def __add__(self, other: Any) -> AtomicInt:
        self._assert_integral(other)
        other = self._unwrap_atomic(other)
        with self._lock:
            return AtomicInt(self._value + other)

    def __and__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value & other

    def __ceil__(self) -> int:
        with self._lock:
            return math.ceil(self._value)

    def __eq__(self, other: object) -> bool:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value == other

    def __floor__(self) -> int:
        with self._lock:
            return math.floor(self._value)

    def __floordiv__(self, other: Any) -> AtomicInt:
        other = self._unwrap_atomic(other)
        with self._lock:
            return AtomicInt(self._value // other)

    def __int__(self) -> int:
        with self._lock:
            return int(self._value)

    def __invert__(self) -> Any:
        with self._lock:
            return ~self._value

    def __le__(self, other: Any) -> bool:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value <= other

    def __lshift__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value << other

    def __lt__(self, other: Any) -> bool:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value < other

    def __mod__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value % other

    def __mul__(self, other: Any) -> AtomicInt:
        self._assert_integral(other)
        other = self._unwrap_atomic(other)
        with self._lock:
            return AtomicInt(self._value * other)

    def __neg__(self) -> AtomicInt:
        with self._lock:
            return AtomicInt(-self._value)

    def __or__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value | other

    def __pos__(self) -> Any:
        with self._lock:
            return +self.value

    def __pow__(self, exponent: Any, modulus: Any | None = ...) -> AtomicInt:
        with self._lock:
            res = self._value**exponent
            if res != int(res):
                raise ValueError(f"Power not supported for type AtomicInt with argument {exponent}")
            return AtomicInt(res)

    def __radd__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other + self._value

    def __rand__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other + self._value

    def __rfloordiv__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other // self._value

    def __rlshift__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other << self._value

    def __rmod__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other % self._value

    def __rmul__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other * self._value

    def __ror__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other | self._value

    def __round__(self, ndigits: int | None = None) -> int:
        with self._lock:
            return self._value

    def __rpow__(self, base: Any) -> Any:
        base = self._unwrap_atomic(base)
        with self._lock:
            return base**self._value

    def __rrshift__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other >> self._value

    def __rshift__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value >> other

    def __rtruediv__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other / self._value

    def __rxor__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other ^ self._value

    def __truediv__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value / other

    def __trunc__(self) -> int:
        with self._lock:
            return math.trunc(self._value)

    def __xor__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value ^ other

    def __hash__(self) -> int:
        with self._lock:
            return hash(self._value)

    def __repr__(self) -> str:
        with self._lock:
            return f"AtomicInt({self._value})"

    def __str__(self) -> str:
        with self._lock:
            return str(self._value)


class BoundedInt(numbers.Integral):
    """A bounded int cannot become larger and/or smaller than a specified interval.

    If the bounded integer does leave the allowed interval, it will be snapped back to the minimum/maximum allowed
    number, respectively.
    """

    @staticmethod
    def non_neg(value: int, *, allowed_max: int | None = None) -> BoundedInt:
        return BoundedInt(value, allowed_min=0, allowed_max=allowed_max)

    def __init__(
        self,
        value: int,
        *,
        allowed_min: int | None = None,
        allowed_max: int | None = None,
    ):
        if not isinstance(value, int):
            raise TypeError(f"Only integer values allowed, but {type(value)} given!")
        if allowed_min is not None and allowed_max is not None and allowed_min > allowed_max:
            raise ValueError("Allowed minimum may not be larger than allowed maximum!")

        self._value = value
        self._allowed_min = allowed_min
        self._allowed_max = allowed_max

        # don't forget the first update!
        self._snap_to_min_max()

    def _snap_to_min_max(self) -> None:
        if self._allowed_min is not None and self._value < self._allowed_min:
            self._value = self._allowed_min
        if self._allowed_max is not None and self._value > self._allowed_max:
            self._value = self._allowed_max

    def _unwrap_atomic(self, value: Any) -> int:
        return value._value if isinstance(value, BoundedInt) else value

    def _get_value(self) -> int:
        return self._value

    def _set_value(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"Only integer values allowed, but {type(value)} given!")
        self._value = value
        self._snap_to_min_max()

    value = property(_get_value, _set_value)

    def __abs__(self) -> int:
        return abs(self._value)

    def __add__(self, other: int | BoundedInt) -> BoundedInt:
        other_value = self._unwrap_atomic(other)
        return BoundedInt(
            self._value + other_value,
            allowed_min=self._allowed_min,
            allowed_max=self._allowed_max,
        )

    def __and__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self._value & other_value

    def __ceil__(self) -> int:
        return self._value

    def __eq__(self, other: object) -> bool:
        other_value = self._unwrap_atomic(other)
        return self._value == other_value

    def __floor__(self) -> int:
        return self._value

    def __floordiv__(self, other: Any) -> int:
        other_value = self._unwrap_atomic(other)
        return self._value // other_value

    def __int__(self) -> int:
        return self._value

    def __invert__(self) -> Any:
        return ~self._value

    def __le__(self, other: Any) -> bool:
        other_value = self._unwrap_atomic(other)
        return self._value <= other_value

    def __lshift__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self.value << other_value

    def __lt__(self, other: Any) -> bool:
        other_value = self._unwrap_atomic(other)
        return self._value < other_value

    def __mod__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self._value % other_value

    def __mul__(self, other: Any) -> BoundedInt:
        other_value = self._unwrap_atomic(other)
        return BoundedInt(
            self._value * other_value,
            allowed_min=self._allowed_min,
            allowed_max=self._allowed_max,
        )

    def __neg__(self) -> BoundedInt:
        return BoundedInt(-self._value, allowed_min=self._allowed_min, allowed_max=self._allowed_max)

    def __or__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self._value | other_value

    def __pos__(self) -> Any:
        return +self._value

    def __pow__(self, exponent: Any, modulus: Any | None = ...) -> BoundedInt:
        res = self._value**exponent
        if res != int(res):
            raise ValueError(f"Power not support for type BoundedInt with argument {exponent}")
        return BoundedInt(res, allowed_min=self._allowed_min, allowed_max=self._allowed_max)

    def __radd__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value + self._value

    def __rand__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value & self._value

    def __rfloordiv__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value // self._value

    def __rlshift__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value << self._value

    def __rmod__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value % self._value

    def __rmul__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value * self._value

    def __ror__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value | self._value

    def __round__(self, ndigits: int | None = None) -> int:
        return self._value

    def __rpow__(self, base: Any) -> Any:
        other_value = self._unwrap_atomic(base)
        return other_value**self._value

    def __rrshift__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value >> self._value

    def __rshift__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self._value >> other_value

    def __rtruediv__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value / self._value

    def __rxor__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value ^ self._value

    def __truediv__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self._value / other_value

    def __trunc__(self) -> int:
        return math.trunc(self._value)

    def __xor__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self._value ^ other_value

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"BoundedInt({self._value}; min={self._allowed_min}, max={self._allowed_max})"

    def __str__(self) -> str:
        return str(self._value)
