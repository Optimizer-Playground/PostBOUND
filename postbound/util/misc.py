"""Contains various utilities that did not fit any other category."""

from __future__ import annotations

import collections
import re
from collections.abc import Generator, Iterable
from datetime import datetime
from typing import Any, Generic, TypeVar

from .jsonize import jsondict

T = TypeVar("T")


def current_timestamp() -> str:
    """Provides the current time (year-month-day hour:minute)"""
    return datetime.now().strftime("%y-%m-%d %H:%M")


_CamelCasePattern = re.compile(r"(?<!^)(?=[A-Z])")


def camel_case2snake_case(camel_case: str) -> str:
    # adapted from https://stackoverflow.com/a/1176023
    return _CamelCasePattern.sub("_", camel_case).lower()


def _wrap_version(v: Any) -> Version:
    """Transforms any object into a Version instance if it is not already."""
    return v if isinstance(v, Version) else Version(v)


class Version:
    """Version instances represent versioning information and ensure that comparison operations work as expected.

    For example, Version instances can be created for strings such as "14.6" or "1.3.1" and ensure that 14.6 > 1.3.1
    """

    @staticmethod
    def is_valid(ver: str | int | list[str] | list[int]) -> bool:
        """Checks whether the provided version representation is valid."""
        try:
            _ = _wrap_version(ver)
            return True
        except ValueError:
            return False

    @staticmethod
    def wrap(ver: Version | str | int | list[str] | list[int]) -> Version | None:
        """Tries to wrap the provided version representation into a Version instance. Returns None if not possible."""
        try:
            return _wrap_version(ver)
        except ValueError:
            return None

    def __init__(self, ver: str | int | list[str] | list[int]) -> None:
        try:
            if isinstance(ver, int):
                self._version = [ver]
            elif isinstance(ver, str):
                self._version = [int(v) for v in ver.split(".")]
            elif isinstance(ver, list) and ver:
                self._version = [int(v) for v in ver]
            else:
                raise ValueError(f"Unknown version string: '{ver}'")
        except ValueError as e:
            raise ValueError(f"Unknown version string: '{ver}'") from e

    @property
    def major(self) -> int:
        """Get the major version component."""
        return self._version[0]

    @property
    def minor(self) -> int | None:
        """Get the minor version component, if available."""
        return self._version[1] if len(self._version) > 1 else None

    @property
    def patch(self) -> int | None:
        """Get the patch version component, if available."""
        return self._version[2] if len(self._version) > 2 else None

    @property
    def components(self) -> Iterable[int]:
        """Get all version components as a list of integers."""
        return self._version.copy()

    def formatted(self, *, prefix: str = "", suffix: str = "", separator: str = "."):
        return prefix + separator.join(str(v) for v in self._version) + suffix

    def __json__(self) -> jsondict:
        return str(self)

    def __eq__(self, __o: object) -> bool:
        try:
            other = _wrap_version(__o)
            if not len(self) == len(other):
                return False
            return all(self._version[i] == other._version[i] for i in range(len(self)))
        except ValueError:
            return False

    def __ge__(self, other: object) -> bool:
        other = Version.wrap(other)
        if other is None:
            return NotImplemented
        return not self < other

    def __gt__(self, other: object) -> bool:
        other = Version.wrap(other)
        if other is None:
            return NotImplemented
        return not self <= other

    def __le__(self, other: object) -> bool:
        other = Version.wrap(other)
        if other is None:
            return NotImplemented

        for comp in zip(self._version, other._version, strict=False):
            own_version, other_version = comp
            if own_version < other_version:
                return True
            if other_version < own_version:
                return False
        return len(self) <= len(other)

    def __lt__(self, other: object) -> bool:
        other = Version.wrap(other)
        if other is None:
            return NotImplemented

        for comp in zip(self._version, other._version, strict=False):
            own_version, other_version = comp
            if own_version < other_version:
                return True
            if other_version < own_version:
                return False
        return len(self) < len(other)

    def __len__(self) -> int:
        return len(self._version)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "v" + ".".join(str(v) for v in self._version)


class DependencyGraph(Generic[T]):
    """A simple dependency graph abstraction. Entries are added via `add_task` and iteration yields the source nodes first."""

    def __init__(self) -> None:
        self._source_nodes: set[int] = set()
        self._dependencies: dict[int, list[int]] = collections.defaultdict(list)
        self._nodes: dict[int, T] = {}

    def add_task(self, node: T, *, depends_on: Iterable[T] | None = None) -> None:
        """Queues a new task/entry/whatever.

        Parameters
        ----------
        node : T
            The new task
        depends_on : Optional[Iterable[T]], optional
            Optional other tasks that have to be completed before this one. If those task have not been added yet, this will
            be done automatically.
        """
        node_id = hash(node)
        self._nodes[node_id] = node

        if not depends_on:
            self._source_nodes.add(node_id)
            return

        if node_id in self._source_nodes:
            self._source_nodes.remove(node_id)

        for dep in depends_on:
            dep_id = hash(dep)
            self._dependencies[dep_id].append(node_id)

            if dep_id not in self._nodes:
                self._source_nodes.add(dep_id)
                self._nodes[dep_id] = dep

    def __iter__(self) -> Generator[T, Any, None]:
        provided_nodes: set[int] = set()
        for node_id in self._source_nodes:
            yield self._nodes[node_id]

            provided_nodes.add(node_id)
            dependency_stack = list(self._dependencies[node_id])

            while dependency_stack:
                dep_id = dependency_stack.pop()
                if dep_id in provided_nodes:
                    continue

                yield self._nodes[dep_id]

                provided_nodes.add(dep_id)
                dependency_stack.extend(self._dependencies[dep_id])
        provided_nodes = set()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        nodes_str = ", ".join(str(node) for node in self._nodes.values())
        return f"DependencyGraph({nodes_str})"
