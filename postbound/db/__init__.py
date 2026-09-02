"""The `db` module provides tools to interact with various database instances.

Generally speaking, the interactions are bidirectional: on the one hand, common database concepts like retrieving statistical
information, introspecting the logical schema or obtaining physical query execution plans are enabled through abstract
interfaces and can be used by the optimizer modules. On the other hand, the `db` modules provides tools to enforce optimization
decisions made as part of an optimization pipeline or other modules when executing the query on the actual database system.

Recall that PostBOUND does not interact with the query optimizer directly and instead relies on system-specific hints or other
special properties of the target database system to influence the optimizer behaviour. Therefore, the optimization process
usually terminates with transforming the original input query to a logically equivalent query that at the same time contains
the necessary modifications for optimization.

The central entrypoint to all database interaction is the abstract `Database` class. This class is inherited by all supported
database systems (currently PostgreSQL and MySQL). Each `Database` instace provides some basic functionality on its own (such
as the ability to execute queries), but delegates most of the work to specific and tailored interfaces. For example, the
`DatabaseSchema` models all access to the logical schema of a database and the `OptimizerInterface` encapsulates the
functionality to retrieve cost estimates or phyiscal query execution plans. All of these interfaces are once again abstract and
implemented according to the specifics of the actual database system.

Take a look at the individual interfaces for further information about their functionality and intended usage.

This module provide direct access to the Postgres interface along with a shortcut method to retrieve the current database
(aptly called `current_database`). In the background, this method delegates to the `DatabasePool`.
If you want to use the MySQL interface, make sure to install PostBOUND with MySQL support enabled and import `mysql` from
the `db` package.
"""

from __future__ import annotations

from ._cache import ResultCache
from ._db import (
    Connection,
    Cursor,
    Database,
    DatabasePool,
    DatabaseSchema,
    DatabaseServerError,
    DatabaseStatistics,
    DatabaseUserError,
    ForeignKeyRef,
    HintService,
    HintWarning,
    Histogram,
    HistogramApproximation,
    MostCommonValues,
    OptimizerInterface,
    PrewarmingSupport,
    QueryCacheWarning,
    ResultRow,
    ResultSet,
    StopwatchSupport,
    TimeoutSupport,
    UnsupportedDatabaseFeatureError,
    current_database,
    simplify_result_set,
)
from ._stats import PreciseStatistics

enable_emulation_fallback: bool = True
"""
Controls, whether database systems that do not maintain a specific kind of statistic are allowed to compute a similar
value instead. For example, some DBMS such as DuckDB do not maintain histograms for column distributions. With the
fallback, DuckDB's statistics catalog would be allowed to compute the histogram on the fly.

Note that allowing the fallback might result in a significant advantage in terms of precision, because such an emulation
is usually based on exact calculation, rather than approximation which most systems that actually implement the
statistic must use.

Disabling the fallback forces DBMSes to raise an `UnsupportedDatabaseFeatureError` if a statistic is not maintained.
Also note that this is a different failure situation than if a DBMS is capable of maintaining a statistic, but the
statistic is not available for a specific object. For example, PostgreSQL only creates most common values lists for
columns that are considered "sufficiently skewed". In this case, the absence of a statistic is indicated by a *None*
value.
"""

__all__ = [
    "Cursor",
    "Connection",
    "Database",
    "DatabaseSchema",
    "DatabaseStatistics",
    "DatabasePool",
    "ForeignKeyRef",
    "HintWarning",
    "HintService",
    "Histogram",
    "HistogramApproximation",
    "MostCommonValues",
    "OptimizerInterface",
    "PrewarmingSupport",
    "PreciseStatistics",
    "QueryCacheWarning",
    "DatabaseServerError",
    "DatabaseUserError",
    "ResultCache",
    "ResultSet",
    "ResultRow",
    "StopwatchSupport",
    "TimeoutSupport",
    "UnsupportedDatabaseFeatureError",
    "current_database",
    "enable_emulation_fallback",
    "simplify_result_set",
]
