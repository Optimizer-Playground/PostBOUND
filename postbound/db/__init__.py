"""The *db* module provides tools to interact with physical database instances.

The central `Database` abstraction allows to execute queries and retrieve information from the specific target database
independent of the underlying physical database. Users can interact with the interface without considering the peculiarities of
differnt database systems.

In particular, the database abstraction provides the following core functionality:

- Execution of arbitrary SQL queries (`Database.execute_query`)
- Unified access to the database schema and statistics catalog (via `DatabaseSchema` and `StatisticsCatalog`)
- Functionality to retrieve information from a query optimizer, such as execution plans or cost estimates
  (in `OptimizerInterface`)
- Query hinting to enforce optimizer decisions during query execution on the native database engine (via `HintService`)

Consult the documentation of the indivdual classes for more details.

Connection Management
---------------------

PostBOUND provides a connection pool for database instances. By convention, calling the (system-specific) `connect` function
will first attempt to retrieve a connection for the same physical database from the pool. If no such connection exists, a new
connection is established.

Typically, the pool contains just a single database instance. In this case, the `DatabasePool` can be used to retrieve the
current database connection. There also exists a module-level `current_database` utility that provides the same functionality.

If pooling is not desired, the `connect` functions provide a ``private`` argument to suppress it (once again by convention).

Supported Systems
-----------------

PostBOUND ships database backends for PostgreSQL and DuckDB out of the box. Both are available in dedicated top-level modules.
In addition, there is rudimentary support for MySQL, but this backend is not actively maintained.

Users can implement their own database backends for specific systems by subclassing `Database` and the other relevant classes.

Query Execution
---------------

Each `Database` provides a central method to execute arbitrary SQL queries: `execute_query`. This method runs the given query
in a blocking manner and returns the result set. As a convenience, the result set is simplified by default. This means that
instead of returning sets/rows with just a single element, the element is returned directly. For example, the result of the
query ``SELECT count(*) FROM title`` will return the count directly as an integer, rather than as ``[(count,)]``.
If a result set is required in any case, it can be enforced by passing `raw=True` to the method.

In addition to this basic execution functionality, some database systems also provide support for timeouts, prewarming of the
shared buffer pool, or measuring the execution time of a query. These features are specified using different mixin protocols.
See `TimeoutSupport`, `PrewarmingSupport`, and `StopwatchSupport` for more information.

All queries are assumed to be read-only with respect to the actual data, i.e., they should not modify the underlying data
schema, but can change configuration of the database. Violating this assumption may lead to wrong results. For example, the
database schema might cache the available tables. If a new table is created, the cache will not be updated accordingly. As a
rule of thumb, whenever a query modifies the underlying data schema, the connection to the database should be re-established.

If complex queries with stable results need to be executed frequently, the `ResultCache` can be used to store the results of a
query and prevent re-execution of the query. The cache functions as a wrapper around a `Database` instance and can be used in
the same way as a regular database instance.

Statistics Catalog
------------------

Database statistics are highly system-specific and often not all statistics are available on a given database system. PostBOUND
tries it best to unify access and availability of different statistics via the `StatisticsCatalog` interface. This interfaces
provides commonly-used statistics such as distinct value counts or histograms. If a statistic is not available on a database
system, the catalog will try to emulate it by issuing an equivalent query to the database.

If this is not desired, the `enable_emulation_fallback` flag can be set to *False*. In this case, the catalog will raise an
error if the database system does not support a specific statistic.

One key property of statistic emulation is that it is always exact. In contrast, most systems only compute approximate
statistics. As a consequence, emulated statistics benefit systems that only maintain a very small subset of statistics, such as
DuckDB (because most statistics are emulated and therefore exact). To "level the field" between different database systems,
the `PreciseStatistics` service can be used. It computes all statistics in an exact manner, regardless of the underlying
database system. Further, this service can be combined with a `ResultCache` to avoid re-executing the same (expensive) queries
for statistics computation.
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
    DatabaseUserError,
    ForeignKeyRef,
    HintService,
    HintWarning,
    Histogram,
    HistogramApproximation,
    MostCommonValues,
    OptimizerInterface,
    PrewarmingSupport,
    ResultRow,
    ResultSet,
    StatisticsCatalog,
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
    "Connection",
    "Cursor",
    "Database",
    "DatabasePool",
    "DatabaseSchema",
    "DatabaseServerError",
    "DatabaseUserError",
    "ForeignKeyRef",
    "HintService",
    "HintWarning",
    "Histogram",
    "HistogramApproximation",
    "MostCommonValues",
    "OptimizerInterface",
    "PreciseStatistics",
    "PrewarmingSupport",
    "ResultCache",
    "ResultRow",
    "ResultSet",
    "StatisticsCatalog",
    "StopwatchSupport",
    "TimeoutSupport",
    "UnsupportedDatabaseFeatureError",
    "current_database",
    "enable_emulation_fallback",
    "simplify_result_set",
]
