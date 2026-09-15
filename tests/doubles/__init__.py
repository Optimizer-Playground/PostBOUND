"""Hand-written test doubles for PostBOUND's database interfaces.

These exist so that the large amount of code sitting *between* pure logic and the wire protocol can be tested
without a server. See `tests/conftest.py` for the tier model they belong to (tier 0).

Why hand-written rather than `unittest.mock`
--------------------------------------------
`Mock` and `MagicMock` auto-create any attribute that is accessed. A renamed or newly added abstract method
therefore keeps passing silently, which is precisely the broken-refactor failure mode this suite exists to
catch. The doubles here subclass the real ABCs instead, so an interface change fails loudly at instantiation
and `ty` checks them like any other code.

`unittest.mock` remains the right tool for genuinely incidental patching -- `subprocess`, `atexit`,
`urllib.request.urlretrieve` -- just not for the `Database` interfaces.

What *not* to use these for
---------------------------
Do not use a double to test connection handling, psycopg type adapters, query cancellation, prewarming or
transactions. A double shares whatever mistaken assumption about the server the code under test makes, so it
can only confirm the bug. Those belong in the live tier.

See Also
--------
ScriptedCursor : the highest-leverage double; unlocks the whole generic `DatabaseSchema`.
FakeDatabase : a minimal in-memory `Database` for everything that just needs a handle.
StaticSchema : a declarative `DatabaseSchema` built from a table/column spec.
"""

from ._doubles import (
    ExpectedQuery,
    FakeDatabase,
    FakeHintService,
    FakeOptimizer,
    FakeStatistics,
    ScriptedCursor,
    StaticSchema,
    UnexpectedQueryError,
    column_spec,
)

__all__ = [
    "ExpectedQuery",
    "FakeDatabase",
    "FakeHintService",
    "FakeOptimizer",
    "FakeStatistics",
    "ScriptedCursor",
    "StaticSchema",
    "UnexpectedQueryError",
    "column_spec",
]
