"""Contains utilities to access and modify dictionaries more conveniently."""

from __future__ import annotations

import collections
import itertools
import typing
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Optional, Protocol

import numpy as np

T = typing.TypeVar("T")
K = typing.TypeVar("K")
V = typing.TypeVar("V")


class SupportsLE(Protocol):
    def __le__(self, other) -> bool: ...


class SupportsLT(Protocol):
    def __lt__(self, other) -> bool: ...


class SupportsGE(Protocol):
    def __ge__(self, other) -> bool: ...


class SupportsGT(Protocol):
    def __gt__(self, other) -> bool: ...


# the Python standard library often implements the comparison functions using the `*args, **kwargs` signature instead
# of the more specific `other` argument. We replicate these protocols here to allow standard Python types to be passed
# to our functions as well.


class PyStdLE(Protocol):
    def __le__(self, *args, **kwargs) -> bool: ...


class PyStdLT(Protocol):
    def __lt__(self, *args, **kwargs) -> bool: ...


class PyStdGE(Protocol):
    def __ge__(self, *args, **kwargs) -> bool: ...


class PyStdGT(Protocol):
    def __gt__(self, *args, **kwargs) -> bool: ...


type SupportsRichCmp = SupportsLT | SupportsLE | SupportsGE | SupportsGT | PyStdLE | PyStdGT | PyStdGE


def stringify(d: dict[K, V]) -> str:
    """Generates a string-representation of a dictionary.

    In contrast to calling ``str()`` directly, this method generates proper string representations of both keys and values and
    does not use ``repr()`` for them. Nested objects are stilled formatted according to ``str()`` however.

    Parameters
    ----------
    d : dict[K, V]
        The dictionary to stringify

    Returns
    -------
    str
        The string representation
    """
    items_str = ", ".join(f"{k}: {v}" for k, v in d.items())
    return "{" + items_str + "}"


def key(dictionary: dict[K, V]) -> K:
    """Provides the key of a dictionary with just 1 item.

    `key({'a': 1}) = 'a'`
    """
    if not len(dictionary) == 1:
        nvals = len(dictionary)
        raise ValueError(f"Dictionary must contain exactly 1 entry, not {nvals}: {dictionary}")
    return next(iter(dictionary.keys()))


def value(dictionary: dict[K, V]) -> V:
    """Provides the value of a dictionary with just 1 item.

    `value({'a': 1}) = 1`
    """
    if not len(dictionary) == 1:
        raise ValueError("Dictionary must contain exactly 1 entry, not " + str(len(dictionary)))
    return next(iter(dictionary.values()))


def difference(a: dict[K, V], b: dict[K, V]) -> dict[K, V]:
    """Computes the set difference between two dictionaries based on their keys.

    Parameters
    ----------
    a : dict[K, V]
        The dict to remove entries from
    b : dict[K, V]
        The entries to remove

    Returns
    -------
    dict[K, V]
        A dictionary that contains all *key, value* pairs from *a* where the *key* is not in *b*.
    """
    return {k: v for k, v in a.items() if k not in b}


def intersection(a: dict[K, V], b: dict[K, V]) -> dict[K, V]:
    """Computes the set intersection between two dictionaries based on their keys.

    Parameters
    ----------
    a : dict[K, V]
        The first dictionary
    b : dict[K, V]
        The second dictionary

    Returns
    -------
    dict[K, V]
        A dictionary that contains all *key, value* pairs from *a* where the *key* is also in *b*.
    """
    return {k: v for k, v in a.items() if k in b}


def merge(a: dict[K, V], b: dict[K, V], *, updater: Optional[Callable[[K, V, V], V]] = None) -> dict[K, V]:
    """Creates a new dict containing all key/values pairs from both argument dictionaries.

    If keys overlap, entries from dictionary `b` will take priority, unless an `update` method is given.
    If `update` is given, and `a[k] = v` and `b[k] = v'` (i.e. both `a` and `b` share a key `k`) the merged dictionary
    will contain the result of `update(k, v, v')` as entry for `k`.

    Note that as of Python 3.9, such a method was added to dictionaries as well (via the `|=` syntax). Our current
    implementation is not optimized for larger dictionaries and will probably have a pretty bad performance on such
    input data.
    """
    if not updater:
        return dict([*a.items()] + [*b.items()])
    else:
        merged = dict(a)
        for k, v in b.items():
            if k in merged:
                merged[k] = updater(k, merged[k], v)
            else:
                merged[k] = v
        return merged


def update(dictionary: dict[K, V], updater: Callable[[K, V], T]) -> dict[K, T]:
    """Creates a new dict by calling update on each key/value pair on the old dict, retaining its keys."""
    return {k: updater(k, v) for k, v in dictionary.items()}


def explode(dictionary: dict[K, list[V]]) -> list[tuple[K, V]]:
    """Transforms dicts mapping keys to lists of values to a list of key/value pairs."""
    values: list[tuple[K, V]] = []
    for k, dict_values in dictionary.items():
        values.extend(zip(itertools.cycle([k]), dict_values))
    return values


def hash_dict(dictionary: dict[K, V]) -> int:
    """Calculates a hash value based on the current dict contents (keys and values)."""
    keys = list(dictionary.keys())
    values = []
    for val in dictionary.values():
        if isinstance(val, collections.abc.Hashable):
            values.append(hash(val))
        elif isinstance(val, list) or isinstance(val, set):
            values.append(hash(tuple(val)))
        elif isinstance(val, dict):
            values.append(hash_dict(val))
        else:
            warnings.warn(f"Unhashable type, skipping: {type(val)}", stacklevel=2)
    keys_hash = hash(tuple(keys))
    values_hash = hash(tuple(values))
    return hash((keys_hash, values_hash))


def generate_multi(entries: Iterable[tuple[K, V]]) -> dict[K, list[V]]:
    """Generates a multi-dict based on its entries.

    Each key can occur multiple times and values will be aggregated in a list.
    """
    collector = collections.defaultdict(list)
    for k, v in entries:
        collector[k].append(v)
    return dict(collector)


def reduce_multi(multi_dict: dict[K, list[V]], reduction: Callable[[K, list[V]], V]) -> dict[K, V]:
    """Ungroups a multi-dict by aggregating the values based on key and values."""
    return {k: reduction(k, vs) for k, vs in multi_dict.items()}


def invert_multi(mapping: dict[K, list[V]]) -> dict[V, list[K]]:
    """Inverts the `key -> values` mapping of a dict to become `value -> keys` instead.

    Supppose a multi-dict has the following contents: `{'a': [1, 2], 'b': [2, 3]}`.
    Calling `invert` transforms this mapping to `{1: ['a'], 2: ['a', 'b'], 3: ['b']}`.
    """
    level1 = {tuple(vs): k for k, vs in mapping.items()}
    level2: dict[V, list[K]] = {}
    for vs, k in level1.items():
        for v in vs:
            if v not in level2:
                level2[v] = [k]
            else:
                level2[v].append(k)
    return level2


def aggregate(dictionaries: Iterable[dict[K, V]]) -> dict[K, Sequence[V]]:
    aggregated_dict = collections.defaultdict(list)
    for d in dictionaries:
        for k, v in d.items():
            aggregated_dict[k].append(v)
    return dict(aggregated_dict)


def invert(mapping: dict[K, V]) -> dict[V, K]:
    """Inverts the `key -> value` mapping of a dict to become `value -> key` instead.

    In contrast to `invert_multi` this does not handle duplicate values (which leads to duplicate keys), nor does it
    process the original values of `mapping` in any way.

    Basically, this function is just a better-readable shortcut for `{v: k for k, v in d.items()}`.
    """
    return {v: k for k, v in mapping.items()}


def argmin[K, V: SupportsRichCmp](mapping: Mapping[K, V]) -> K:
    """
    For a dict mapping keys to numeric types, returns the key `k` with minimum value `v`, s.t. for all keys `k'` with
    values `v'` it holds that `v <= v'`.
    """
    # ty complains here because get() might return None. But we know for a fact that this cannot be the case since
    # get() will only be called for the existing keys. So this is safe to ignore.
    return min(mapping, key=mapping.get)  # type: ignore


def argmax[K, V: SupportsRichCmp](mapping: dict[K, V]) -> K:
    """
    For a dict mapping keys to numeric types, returns the key `k` with maximum value `v`, s.t. for all keys `k'` with
    values `v'` it holds that `v >= v'`.
    """
    # ty complains here because get() might return None. But we know for a fact that this cannot be the case since
    # get() will only be called for the existing keys. So this is safe to ignore.
    return max(mapping, key=mapping.get)  # type: ignore


def dict_to_numpy(data: dict[K, V]) -> np.array[V]:
    sorted_dict = {k: data[k] for k in sorted(data)}
    return np.asarray(list(sorted_dict.values()))


class HashableDict(collections.UserDict[K, V]):
    """A dictionary implementation that can be hashed.

    Warnings
    --------
    This type should be used with extreme caution in order to not violate any invariants due to unintended data modification.
    """

    def __hash__(self) -> int:
        return hash_dict(self.data)


class CustomHashDict[K, V](Mapping[K, V]):
    """Wrapper of a normal Python dictionary that uses a custom hash function instead of the default hash() method.

    All non-hashing related behavior is directly inherited from the default Python dictionary. Only the item access is changed
    to enforce the usage of the new hashing function.

    Notice that since the custom hash function always provides an integer value, collision detection is weaker than originally.
    This is because the actual dictionary never sees the original keys to run an equality comparison. Instead, the comparison
    is based on the integer values.

    Parameters
    ----------
    hash_func : Callable[[K], int]
        The hashing function to use. It receives the key as input and must produce a valid hash value as output.
    **kwargs : dict, optional
        Additional keyword arguments that should be passed to the hashing function upon each invocation.
    """

    def __init__(self, hash_func: Callable[[K], int], **kwargs) -> None:
        self.hash_function = hash_func
        self._hash_args = kwargs

        self._map: dict[int, V] = {}
        self._keys: set[K] = set()

    def __getitem__(self, key: K) -> V:
        idx = self.hash_function(key, **self._hash_args)
        return self._map[idx]

    def __setitem__(self, key: K, item: V) -> None:
        idx = self.hash_function(key, **self._hash_args)
        self._keys.add(key)
        self._map[idx] = item

    def __delitem__(self, key: K) -> None:
        idx = self.hash_function(key, **self._hash_args)
        del self._map[idx]
        self._keys.remove(key)

    def __contains__(self, key: object) -> bool:
        idx = self.hash_function(key, **self._hash_args)  # type: ignore
        return idx in self._map

    def __len__(self) -> int:
        return len(self._map)

    def __iter__(self) -> Iterator[K]:
        return iter(self._keys)


class DynamicDefaultDict(collections.UserDict[K, V]):
    """Wrapper of a normal Python `defaultdict` that permits dynamic default values.

    When using a standard Python `defaultdict`, the default value must be decided up-front. This value is used for all missing
    keys. In contrast, this dictionary implementation allows for the default value to be calculated based on the requested key.

    Parameters
    ----------
    factory : Callable[[K], V]
        A function that generates the value based on a missing key. It receives the key as input and must return the value.
    """

    def __init__(self, factory: Callable[[K], V]) -> None:
        super().__init__()
        self.factory = factory

    def __getitem__(self, key: K) -> V:
        if key not in self.data:
            self.data[key] = self.factory(key)
        return self.data[key]


class frozendict(collections.UserDict[K, V]):
    """Read-only variant of a normal Python dictionary.

    Once the dictionary has been created, its key/value pairs can no longer be modified. At the same time, this allows the
    dictionary to be hashable by default.

    Parameters
    ----------
    items : any, optional
        Supports the same argument types as the normal dictionary. If no items are supplied, an empty frozen dictionary is
        returned.
    """

    def __init__(self, items=None) -> None:
        self._frozen = False
        super().__init__(items)
        self.clear = None
        self.pop = None
        self.popitem = None
        self.update
        self._frozen = True

    def __setitem__(self, key: K, item: V) -> None:
        if self._frozen:
            raise TypeError("Cannot set frozendict entries after creation")
        return super().__setitem__(key, item)

    def __delitem__(self, key: K) -> None:
        if self._frozen:
            raise TypeError("Cannot remove frozendict entries after creation")
        return super().__delitem__(key)

    def __hash__(self) -> int:
        return hash_dict(self)
