"""Contains the DuckDB implementation of the Database interface.

DuckDB itself provides no support for the kind of optimizer hints that PostBOUND relies on. Therefore, this backend is
built on top of `quacklab <https://github.com/rbergm/quacklab>`_, a DuckDB fork that adds the necessary hinting
extension points. The *quacklab* Python package is a drop-in replacement for the official *duckdb* package and can be
installed alongside it.

Compared to the Postgres backend, a number of limitations remain: DuckDB maintains only a minimal set of statistics
(essentially just row counts, everything else falls back to `PreciseStatistics`), it does not expose cost estimates,
and operator hints must be used with great care because DuckDB's physical operators are tightly coupled to the rules
that select them.
"""

from ._duckdb import (
    DuckDBHintService,
    DuckDBInterface,
    DuckDBOptimizer,
    DuckDBSchema,
    DuckDBStatistics,
    connect,
    parse_duckdb_plan,
)

__all__ = [
    "DuckDBHintService",
    "DuckDBInterface",
    "DuckDBOptimizer",
    "DuckDBSchema",
    "DuckDBStatistics",
    "connect",
    "parse_duckdb_plan",
]
