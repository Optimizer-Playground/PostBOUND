from __future__ import annotations

import dataclasses
import unittest
from collections.abc import Collection, Iterator

from postbound.util import collections as collection_utils


@dataclasses.dataclass(frozen=True)
class MyCustomType:
    arg: int


class MyCustomCollection(Collection):
    def __init__(self, contents: list) -> None:
        self.contents = contents

    def __len__(self) -> int:
        return len(self.contents)

    def __iter__(self) -> Iterator:
        return iter(self.contents)

    def __contains__(self, __x: object) -> bool:
        return __x in self.contents

    def __eq__(self, other: object):
        return isinstance(other, type(self)) and self.contents == other.contents


class CollectionsTests(unittest.TestCase):
    def test_enlist(self) -> None:
        arg = 42
        self.assertEqual(collection_utils.enlist(arg), [42], msg="Scalar value should be enlisted")

        arg = "hello world"
        self.assertEqual(collection_utils.enlist(arg), ["hello world"], msg="String value should be enlisted")

        arg = [42]
        self.assertEqual(collection_utils.enlist(arg), [42], msg="Should not enlist list arguments")

        arg = {42}
        self.assertEqual(collection_utils.enlist(arg), {42}, msg="Should not enlist set arguments")

        arg = (42, 24)
        self.assertEqual(collection_utils.enlist(arg), (42, 24), msg="Should not enlist tuples by default")
        self.assertEqual(
            collection_utils.enlist(arg, enlist_tuples=True), [(42, 24)], msg="Should enlist tuples if asked to"
        )

        arg = {42: "hello world"}
        self.assertEqual(collection_utils.enlist(arg), [{42: "hello world"}], msg="Should enlist dictionaries")

        arg = MyCustomType(42)
        self.assertEqual(collection_utils.enlist(arg), [MyCustomType(42)], msg="Should enlist custom types")

        arg = MyCustomCollection([42])
        self.assertEqual(
            collection_utils.enlist(arg), [MyCustomCollection([42])], msg="Should enlist arbitrary containers"
        )
