from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator
from typing import override


@dataclasses.dataclass(frozen=True)
class MyCustomType:
    arg: int


class MyCustomCollection(Collection):
    def __init__(self, contents: list) -> None:
        self.contents = contents

    @override
    def __len__(self) -> int:
        return len(self.contents)

    @override
    def __iter__(self) -> Iterator:
        return iter(self.contents)

    @override
    def __contains__(self, __x: object) -> bool:
        return __x in self.contents

    @override
    def __eq__(self, other: object):
        return isinstance(other, type(self)) and self.contents == other.contents
