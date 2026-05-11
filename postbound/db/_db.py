"""This module provides PostBOUNDs basic interaction with databases.

More specifically, this includes

- an interface to interact with databases (the `Database` interface)
- an interface to retrieve schema information (the `DatabaseSchema` interface)
- an interface to obtain different table-level and column-level statistics (the `DatabaseStatistics` interface)
- an interface to modify queries such that optimization decisions are respected during the actual query execution (the
  `HintService` interface)
- an interface to access information of the native optimizer of the database system (the `OptimizerInterface` class)
- a utility to easily obtain database connections (the `DatabasePool` singleton class).

Take a look at the central `Database` class for more details. All concrete database systems need to implement this
interface.
"""

from __future__ import annotations

import abc
import atexit
import bisect
import collections
import json
import os
import textwrap
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Literal, Optional, Protocol, Type, overload, runtime_checkable

import networkx as nx

from postbound.util import set_union

from .. import util
from .._base import SupportsRichComparison
from .._core import (
    BoundColumnReference,
    Cardinality,
    ColumnReference,
    Cost,
    TableReference,
    UnboundColumnError,
    VirtualTableError,
)
from .._hints import (
    HintType,
    JoinTree,
    PhysicalOperator,
    PhysicalOperatorAssignment,
    PlanParameterization,
)
from .._qep import QueryPlan
from ..qal import SqlQuery
from ..util import jsondict

ResultRow = tuple
"""Simple type alias to denote a single tuple from a result set."""

ResultSet = Sequence[ResultRow]
"""Simple type alias to denote the result relation of a query."""


class Cursor(Protocol):
    """Interface for database cursors that adhere to the Python Database API specification.

    This is not a complete representation and only focuses on the parts of the specification that are important for
    PostBOUND right now. In the future, additional methods might get added.

    This type is only intended to denote the expected return type of certain methods, the cursors themselves are
    supplied by the respective database integrations. There should be no need to implement one manually and all cursors
    should be compatible with this interface by default (since they are DB API 2.0 cursor objects).

    See PEP 249 for details (https://peps.python.org/pep-0249/)
    """

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def execute(
        self, operation: str, parameters: Optional[dict | Sequence] = None
    ) -> Optional[Cursor]:
        raise NotImplementedError

    @abc.abstractmethod
    def fetchone(self) -> Optional[ResultRow]:
        raise NotImplementedError

    @abc.abstractmethod
    def fetchall(self) -> Optional[ResultSet]:
        raise NotImplementedError


class Connection(Protocol):
    """Interface for database connections that adhere to the Python Database API specification.

    This is not a complete representation and only focuses on the parts of the specification that are important for
    PostBOUND right now. In the future, additional methods might get added.

    This type is only intended to denote the expected return type of certain methods, the connections themselves are
    supplied by the respective database integrations. There should be no need to implement one manually and all
    connections should be compatible with this interface by default (since they are DB API 2.0 connection objects).

    See PEP 249 for details (https://peps.python.org/pep-0249/)
    """

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def cursor(self) -> Cursor:
        raise NotImplementedError


@runtime_checkable
class PrewarmingSupport(Protocol):
    """Some databases might support adding specific tables to their shared buffer.

    If so, they should implement this protocol to allow other parts of the framework to exploit this feature.
    """

    @abc.abstractmethod
    def prewarm_tables(
        self,
        tables: Optional[TableReference | Iterable[TableReference]] = None,
        *more_tables: TableReference,
        exclude_table_pages: bool = False,
        include_primary_index: bool = True,
        include_secondary_indexes: bool = True,
    ) -> None:
        """Prepares the database buffer pool with tuples from specific tables.

        Parameters
        ----------
        tables : Optional[TableReference  |  Iterable[TableReference]], optional
            The tables that should be placed into the buffer pool
        *more_tables : TableReference
            More tables that should be placed into the buffer pool, enabling a more convenient usage of this method.
            See examples for details on the usage.
        exclude_table_pages : bool, optional
            Whether the table data (i.e. pages containing the actual tuples) should *not* be prewarmed. This is off by default,
            meaning that prewarming is applied to the data pages. This can be toggled on to only prewarm index pages (see
            `include_primary_index` and `include_secondary_index`).
        include_primary_index : bool, optional
            Whether the pages of the primary key index should also be prewarmed. Enabled by default.
        include_secondary_indexes : bool, optional
            Whether the pages for secondary indexes should also be prewarmed. Enabled by default.

        Notes
        -----
        If the database should prewarm more table pages than can be contained in the shared buffer, the actual contents of the
        pool are not specified. All prewarming tasks might happen sequentially, in which case the first prewarmed relations
        will typically be evicted and only the last relations (tables or indexes) are retained in the shared buffer. The
        precise order in which the prewarming tasks are executed is not specified and depends on the actual relations.

        Examples
        --------
        >>> database.prewarm_tables([table1, table2])
        >>> database.prewarm_tables(table1, table2)
        """
        ...


@runtime_checkable
class TimeoutSupport(Protocol):
    """Marks database systems that support executing queries with a timeout."""

    def execute_with_timeout(
        self, query: SqlQuery | str, *, timeout: float = 60.0
    ) -> Optional[ResultSet]:
        """Executes a query with a specific timeout.

        For query execution, we use the following rules in contrast to `Database.execute_query`:

        1. We never make use of the database interfaces' cache, even if it the query is contained in the cache
        2. We never attempt to simplify the result set, even if this would be possible (e.g., for single-row result sets).
           This is more of a pragmatic decision to be able to indicate a timeout with *None* and distinguishing it from a
           valid result set of a single *NULL* tuple. Otherwise, we would have to resort to raising *TimeoutError* or similar
           strategies, which complicates the control flow for the caller.

        Parameters
        ----------
        query : SqlQuery | str
            The query to execute. If this contains hints or other special features, those will be treated normally.
        timeout : float, optional
            The timeout in seconds. If the query takes longer (inlcuding all special treatment of the database interface),
            it will be cancelled. Defaults to 60 seconds.

        Returns
        -------
        Optional[ResultSet]
            The result set of the query. If the query was cancelled, this will be *None*.
        """
        ...


@runtime_checkable
class StopwatchSupport(Protocol):
    """Marks the database systems that support measurement of query execution times."""

    def time_query(
        self, query: SqlQuery | str, *, timeout: Optional[float] = None
    ) -> float:
        """Determines the execution time of a query.

        The execution time is measured from the moment the query is passed to the internal cursor (i.e. including sending the
        query to the database server), until the execution is finished. Therfore, it does not include the time required to
        transfer the result set back to the client.

        Parameters
        ----------
        query : SqlQuery | str
            The query to execute.
        timeout : Optional[float], optional
            Cancels the query execution if it takes longer than this number (in seconds). Notice that this parameter requires
            timeout support from the database system.

        Returns
        -------
        float
            The runtime of the query in seconds. The result set is ignored.

        Raises
        ------
        UnsupportedDatabaseFeatureError
            If the database system does not support timeouts. You can use the `TimeoutSupport` protocol to check this
            beforehand.
        """
        ...

    def last_query_runtime(self) -> float:
        """Get the runtime of the last executed query.

        The execution time is measured from the moment the query is passed to the internal cursor (i.e. including sending the
        query to the database server), until the execution is finished. Therfore, it does not include the time required to
        transfer the result set back to the client.

        Returns
        -------
        float
            The runtime of the last executed query in seconds. If no query has been executed before, *NaN* is returned.
        """
        ...


class QueryCacheWarning(UserWarning):
    """Warning to indicate that the query result cache was not found."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


def simplify_result_set(result_set: ResultSet | None) -> Any:
    """Default implementation of the result set simplification logic outlined in `Database.execute_query`.

    Parameters
    ----------
    result_set : ResultSet | None
        Result set to simplify: each entry in the list corresponds to one row in the result set and each component of the
        tuples corresponds to one column in the result set

    Returns
    -------
    Any
        The simplified result set: if the result set consists just of a single row, this row is unwrapped from the list. If the
        result set contains just a single column, this is unwrapped from the tuple. Both simplifications are also combined,
        such that a result set of a single row of a single column is turned into the single value.
        As a special case, an empty result set is turned into *None*.
    """
    # simplify the query result as much as possible: [(42, 24)] becomes (42, 24) and [(1,), (2,)] becomes [1, 2]
    # [(42, 24), (4.2, 2.4)] is left as-is
    if not result_set:
        return None

    result_structure = result_set[0]  # what do the result tuples look like?
    if len(result_structure) == 1:  # do we have just one column?
        result_set = [
            row[0] for row in result_set
        ]  # if it is just one column, unwrap it

    if len(result_set) == 1:  # if it is just one row, unwrap it
        return result_set[0]
    return result_set


class _DBCacheJsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return {"$datetime": o.isoformat()}
        elif isinstance(o, date):
            return {"$date": o.isoformat()}
        elif isinstance(o, time):
            return {"$time": o.isoformat()}
        elif isinstance(o, timedelta):
            return {"$timedelta": o.total_seconds()}
        return super().default(o)


class _DBCacheJsonDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        self._second_hook = kwargs.get("object_hook")
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: Any) -> Any:
        if self._second_hook:
            return self._second_hook(obj)

        if "$datetime" in obj:
            return datetime.fromisoformat(obj["$datetime"])
        elif "$date" in obj:
            return date.fromisoformat(obj["$date"])
        elif "$time" in obj:
            return time.fromisoformat(obj["$time"])
        elif "$timedelta" in obj:
            return timedelta(seconds=obj["$timedelta"])
        return obj


class Database(abc.ABC):
    """A `Database` is PostBOUND's logical abstraction of physical database management systems.

    It provides high-level access to internal functionality provided by such systems. More specifically, each
    `Database` instance supports the following functionality:

    - executing arbitrary SQL queries
    - retrieving schema information, most importantly about primary keys and foreign keys
    - accessing statistical information, such as most common values or the number of rows in a table
    - query formatting and generation of query hints to enforce optimizer decisions (join orders, operators, etc.)
    - introspection of the query optimizer to retrieve query execution plans, cost estimates, etc.

    Notice, that all this information is by design read-only and functionality to write queries is intentionally not
    implemented (although one could issue `INSERT`/`UPDATE`/`DELETE` queries via the query execution functionality).

    This restriction to read-only information enables the caching of query results to provide them without running a
    query over and over again. This is achieved by storing the results of past queries in a special JSON file, which is
    read upon creation of the `Database` instance. If this behavior is not desired, it can simply be turned off
    globally via the `cache_enabled` property, or on a per-method-call basis by setting the corresponding parameter. If
    no such parameter is available, the specific method does not make use of the caching mechanic.

    Each database management system will need to implement this basic interface to enable PostBOUND to access the
    necessary information.

    Parameters
    ----------
    system_name : str
        The name of the database system for which the connection is established. This is only really important to
        distinguish different instances of the interface in a convenient manner.
    cache_enabled : bool, optional
        Whether complex queries that are executed against the database system should be cached. This is especially useful to
        emulate certain statistics that are not maintained by the specific database system (see `DatabaseStatistics` for
        details). If this is *False*, the query cache will not be loaded as well. Defaults to *True*.

    Notes
    -----
    When the `__init__` method is called, the connection to the specific database system has to be established already,
    i.e. calling any of the public methods should provide a valid result. This is particularly important, because this
    method takes care of the cache initialization. This initialization in turn relies on identifying the correct
    cache file, which in turn depends on the system name, system version and database name of the connection.
    """

    def __init__(self, system_name: str, *, cache_enabled: bool = True) -> None:
        self.system_name = system_name

        self._cache_enabled = cache_enabled
        self._query_cache: dict[str, ResultSet] = {}
        if self._cache_enabled:
            self._inflate_query_cache()
        atexit.register(self.close)

    @abc.abstractmethod
    def schema(self) -> DatabaseSchema:
        """Provides access to the underlying schema information of the database.

        Returns
        -------
        DatabaseSchema
            An object implementing the schema interface for the actual database system. This should normally be
            completely stateless.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def statistics(self) -> DatabaseStatistics:
        """Provides access to the current statistics of the database.

        Implementing generalized statistics for a framework that supports multiple different physical database systems
        is much more complicated than it might seem at first. Therefore, different modes for the statistics
        provisioning exist. These modes can be changed by setting the properties of the interface. See the
        documentation of `DatabaseStatistics` for more details.

        Repeated calls to this method are guaranteed to provide the same object. Therefore, changes to the statistics
        interface configuration are guaranteed to be persisted accross multiple accesses to the statistics system.

        Returns
        -------
        DatabaseStatistics
            The statistics interface. Repeated calls to this method are guaranteed to provide the same object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def hinting(self) -> HintService:
        """Provides access to the hint generation facilities for the current database system.

        Returns
        -------
        HintService
            The hinting service. This should normally be completely stateless.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def optimizer(self) -> OptimizerInterface:
        """Provides access to optimizer-related functionality of the database system.

        Returns
        -------
        OptimizerInterface
            The optimizer interface. This should normally be completely stateless.

        Raises
        ------
        UnsupportedDatabaseFeatureError
            If the database system does not provide any sort of external access to the optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def execute_query(
        self,
        query: SqlQuery | str,
        *,
        cache_enabled: Optional[bool] = None,
        raw: bool = False,
    ) -> Any:
        """Executes the given query and returns the associated result set.

        Parameters
        ----------
        query : SqlQuery | str
            The query to execute. If it contains a `Hint` with `preparatory_statements`, these will be executed
            beforehand. Notice that such statements are never subject to caching.
        cache_enabled : Optional[bool], optional
            Controls the caching behavior for just this one query. The default value of *None* indicates that the
            "global" configuration of the database system should be used. Setting this parameter to a boolean value
            forces or deactivates caching for the specific query for the specific execution no matter what the "global"
            configuration is.
        raw : bool, optional
            Whether the result set should be returned as-is. By default, the result set is simplified. Raw mode skips this
            step.

        Returns
        -------
        Any
            Result set of the input query. This is a list of equal-length tuples in the most general case. Each
            component of the tuple corresponds to a specific column of the result set and each tuple corresponds to a
            row in the result set. However, many queries do not provide a 2-dimensional result set (e.g. *COUNT(\\*)*
            queries). In such cases, the nested structure of the result set makes it quite cumbersome to use.
            Therefore, this method tries to simplify the return value of the query for more convenient use (if `raw` mode is
            disabled). More specifically, if the query returns just a single row, this row is returned directly as a tuple.
            Furthermore, if the query returns just a single column, the values of that column are returned directly in
            a list. Both simplifications will also be combined, such that a result set of a single row of a single
            value will be returned as that single value directly. In all other cases, the result will be a list
            consisting of the different result tuples.

        Notes
        -----
        This method is mainly intended to execute read-only SQL queries. In fact, the only types of SQL queries that
        can be modelled by the query abstraction layer are precisely such read-only queries. However, if one really
        needs to execute mutating queries, they can be issued as plain text. Just remember that this behavior is
        heavily discouraged!

        The precise behavior of this method depends on whether caching is enabled or not. If it is, the query will
        only be executed against the live database system, if it is not in the cache. Otherwise, the result will simply
        be retrieved. Caching can be enabled/disabled for just this one query via the `cache_enabled` switch. If this
        is not specified, caching depends on the `cache_enabled` property.

        If caching should be used for this method, but is disabled at a database-level, the current cache will still
        be read and persisted. This ensures that all cached queries are properly saved and none of the previous cache
        content is lost.
        """
        raise NotImplementedError

    def explain(
        self,
        query: SqlQuery,
        *,
        plan: Optional[QueryPlan] = None,
        join_order: Optional[JoinTree] = None,
        physical_ops: Optional[PhysicalOperatorAssignment] = None,
        plan_params: Optional[PlanParameterization] = None,
    ) -> QueryPlan:
        """Shortcut method to obtain the query plan for a given query.

        Calling this method is equivalent to calling ``.optimizer().query_plan()`` on the query, but it also allows to
        directly pass hints for the plan generation. These are passed as-is to the hinting backend.
        """
        needs_hinting = any(
            hint is not None for hint in [plan, join_order, physical_ops, plan_params]
        )
        if needs_hinting:
            query = self.hinting().generate_hints(
                query,
                plan=plan,
                join_order=join_order,
                physical_operators=physical_ops,
                plan_parameters=plan_params,
            )
        return self.optimizer().query_plan(query)

    @abc.abstractmethod
    def database_name(self) -> str:
        """Provides the name of the (physical) database that the database interface is connected to.

        Returns
        -------
        str
            The database name, e.g. *imdb* or *tpc-h*
        """
        raise NotImplementedError

    def database_system_name(self) -> str:
        """Provides the name of the database management system that this interface is connected to.

        Returns
        -------
        str
            The database system name, e.g. *PostgreSQL*
        """
        return self.system_name

    @abc.abstractmethod
    def database_system_version(self) -> util.Version:
        """Returns the release version of the database management system that this interface is connected to.

        Returns
        -------
        util.Version
            The version
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> jsondict:
        """Provides a representation of the current database connection as well as its system settings.

        This description is intended to transparently document which customizations have been applied, thereby giving
        an idea of how the default query execution might have been affected. It can be JSON-serialized and will be
        included by most of the output of the utilities in the `runner` module of the `experiments` package.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset_connection(self) -> Any:
        """Obtains a new network connection for the database. Useful for debugging purposes or in case of crashes.

        Notice that resetting the connection can have unintended side-effects if other methods rely on the cursor
        object. After resetting, the former cursor object will probably no longer be valid. Therefore, this method
        should be used with caution.

        Returns
        -------
        Any
            Backend-specific information about the new connection. E.g., this might be a connection ID or nothing at all.

        See Also
        --------
        Database.cursor
        """
        raise NotImplementedError

    def reset_cache(self) -> None:
        """Removes all results from the query cache. Useful for debugging purposes."""
        self._query_cache = {}

    @abc.abstractmethod
    def cursor(self) -> Cursor:
        """Provides a cursor to execute queries and iterate over result sets manually.

        Returns
        -------
        Cursor
            A cursor compatible with the Python DB API specification 2.0 (PEP 249). The specific cursor type depends on
            the concrete database implementation however.

        References
        ----------

        .. Python DB API specification 2.0 (PEP 249): https://peps.python.org/pep-0249/
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Shuts down all currently open connections to the database."""
        raise NotImplementedError

    def provides(self, support: Type) -> bool:
        """Checks, whether the database interface supports a specific protocol."""
        return isinstance(self, support)

    def _get_cache_enabled(self) -> bool:
        """Getter for the `cache_enabled` property.

        Returns
        -------
        bool
            Whether caching is currently enabled
        """
        return self._cache_enabled

    def _set_cache_enabled(self, enabled: bool) -> None:
        """Setter for the `cache_enabled` property. Inflates the query cache if necessary.

        If the cache should be enabled now, but no cached data exists, the cache will be inflated from disk.

        Parameters
        ----------
        enabled : bool
            Whether caching should be enabled
        """
        if enabled and not self._query_cache:
            self._inflate_query_cache()
        self._cache_enabled = enabled

    cache_enabled = property(_get_cache_enabled, _set_cache_enabled)
    """Controls, whether the results of executed queries should be cached to prevent future re-execution.

    If caching should be enabled later on and no cached data exists, the cache will be inflated from disk.
    """

    def _inflate_query_cache(self) -> None:
        """Tries to read the query cache for this database.

        This reads a JSON file that contains all cached queries and their result sets. It should not be edited
        manually.
        """
        if self._query_cache:
            return
        query_cache_name = self._query_cache_name()
        if os.path.isfile(query_cache_name):
            with open(query_cache_name, "r") as cache_file:
                try:
                    self._query_cache = json.load(cache_file, cls=_DBCacheJsonDecoder)
                except json.JSONDecodeError as e:
                    warnings.warn(
                        f"Could not read query cache: {e}",
                        category=QueryCacheWarning,
                        stacklevel=4,
                    )
                    self._query_cache = {}
        else:
            warnings.warn(
                f"Could not read query cache: File {query_cache_name} does not exist",
                category=QueryCacheWarning,
                stacklevel=4,
            )
            self._query_cache = {}
        atexit.register(self._store_query_cache, query_cache_name)

    def _store_query_cache(self, query_cache_name: str) -> None:
        """Stores the query cache into a JSON file.

        Parameters
        ----------
        query_cache_name : str
            The path where to write the file to. If it exists, it will be overwritten.
        """
        with open(query_cache_name, "w") as cache_file:
            json.dump(self._query_cache, cache_file, cls=_DBCacheJsonEncoder)

    def _query_cache_name(self) -> str:
        """Provides a normalized file name for the query cache.

        Returns
        -------
        str
            The cache file name. It consists of the database system name, system version and the name of the database
        """
        identifier = "_".join(
            [
                self.database_system_name(),
                self.database_system_version().formatted(prefix="v", separator="_"),
                self.database_name(),
            ]
        )
        return f".query_cache_{identifier}.json"

    def __hash__(self) -> int:
        return hash(self._query_cache_name())

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._query_cache_name() == other._query_cache_name()
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.database_name()} @ {self.database_system_name()} ({self.database_system_version()})"


ForeignKeyRef = collections.namedtuple("ForeignKeyRef", ["fk_col", "referenced_col"])
"""
A foreign key references has a foreign key column `fk_col` (the first element) that requires a matching value in the
`referenced_col` (the second element) of the target table.
"""


@dataclass
class ColumnInfo:
    column: BoundColumnReference
    table: TableReference

    datatype: str
    nullable: bool
    primary_key: bool
    indexed: bool


@dataclass
class TableInfo(Mapping[BoundColumnReference, ColumnInfo]):
    table: TableReference
    columns: Sequence[ColumnInfo]

    primary_key: BoundColumnReference | None
    outgoing_foreign_keys: set[ForeignKeyRef]
    incoming_foreign_keys: set[ForeignKeyRef]

    def __getitem__(self, key: ColumnReference | str) -> ColumnInfo:
        if isinstance(key, str):
            key = ColumnReference(key, self.table)
        for column_info in self.columns:
            if column_info.column == key:
                return column_info
        raise KeyError(f"Column '{key}' not found in table '{self.table}'")

    def __len__(self) -> int:
        return len(self.columns)

    def __iter__(self) -> Iterator[BoundColumnReference]:
        return iter(column_info.column for column_info in self.columns)


class DatabaseSchema(abc.ABC, Mapping[TableReference, TableInfo]):
    """This interface provides access to different information about the logical structure of a database.

    In contrast to database statistics, schema information is much more standardized. PostBOUND therefore only takes on
    the role of a mediator to delegate requests to different parts of the schema to the approapriate - and sometimes
    system specific - metadata catalogs of the database systems. For each kind of schema information a dedicated query
    method exists. Take a look at these methods to understand the functionality provided by the database schema
    interface.

    The database schema can be iterated over to obtain all tables in the current database. Key-based access is supported as
    follows: ``schema[table]`` provides a `TableInfo` that contains a summary of the tables structure (e.g. its columns,
    primary key, foreign keys, etc.). Similarly, ``schema[column]`` provides a `ColumnInfo` with detailed information about
    the column (e.g. its datatype, whether it is nullable, etc.). Generally speaking, you should pass a proper
    `TableReference` or `ColumnReference`. It is also possible to pass just a string, but in this case the schema has to
    guess whether you are referring to a table or a column.

    Parameters
    ----------
    db : Database
        The database for which the schema information should be read. This is required to obtain cursors that request
        the desired data.
    prep_placeholder : str, optional
        The placeholder that is used for prepared statements. Some systems use `?` as a placeholder, while others use *%s*
        (the default). This needs to be specified to ensure that the information_schema queries are correctly formatted.

    Notes
    -----
    **Hint for implementors:** the database schema contains no abstract methods that need to be overridden. All methods come
    with a default implementation that uses the *information_schema* to retrieve the necessary information. However, if the
    target database system does not support specific features of the information_schema, the corresponding methods need to be
    overridden to provide the necessary functionality. The documentation of each method details which parts of the
    information_schema it needs.
    """

    def __init__(self, db: Database, *, prep_placeholder: str = "%s"):
        self._db = db
        self._prep_placeholder = prep_placeholder

    def tables(self, *, include_system_tables: bool = False) -> set[TableReference]:
        """Fetches all user-defined tables that are contained in the current database.

        Parameters
        ----------
        include_system_tables : bool, optional
            Whether system tables should also be included. By default, only user-defined tables are returned.

        Returns
        -------
        set[TableReference]
            All tables in the current schema, including materialized views, etc.

        Notes
        -----
        **Hint for implementors:** the default implementation of this method relies on the *information_schema.tables* view.
        """
        query_template = textwrap.dedent(f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_catalog = {self._prep_placeholder}
                AND table_schema = current_schema()
            """)
        cur = self._db.cursor()
        cur.execute(query_template, (self._db.database_name(),))
        result_set = cur.fetchall()
        assert result_set is not None
        return set(TableReference(row[0]) for row in result_set)

    def columns(self, table: TableReference | str) -> Sequence[BoundColumnReference]:
        """Fetches all columns of the given table.

        Parameters
        ----------
        table : TableReference | str
            A table in the current schema

        Returns
        -------
        Sequence[BoundColumnReference]
            All columns for the given table. Columns are ordered according to their position in the table.
            Will be empty if the table is not found or does not contain any columns.

        Raises
        ------
        postbound.qal.VirtualTableError
            If the given table is virtual (e.g. subquery or CTE)

        Notes
        -----
        **Hint for implementors:** the default implementation of this method relies on the *information_schema.columns* view.
        """

        # The documentation of lookup_column() reference an implementation detail of this method.
        # Make sure to keep the two in sync.

        table = table if isinstance(table, TableReference) else TableReference(table)
        if table.virtual:
            raise VirtualTableError(table)
        schema_placeholder = (
            self._prep_placeholder if table.schema else "current_schema()"
        )
        query_template = textwrap.dedent(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = {self._prep_placeholder}
                AND table_catalog = current_database()
                AND table_schema = {schema_placeholder}
            ORDER BY ordinal_position
            """)
        params = [table.full_name]
        if table.schema:
            params.append(table.schema)
        cur = self._db.cursor()
        cur.execute(query_template, params)
        result_set = cur.fetchall()
        assert result_set is not None
        return [BoundColumnReference(row[0], table) for row in result_set]

    def is_view(self, table: TableReference | str) -> bool:
        """Checks, whether a specific table is actually is a view.

        Parameters
        ----------
        table : TableReference | str
            The table to check. May not be a virtual table.

        Returns
        -------
        bool
            Whether the table is a view

        Raises
        ------
        ValueError
            If the table was not found in the current database

        Notes
        -----
        **Hint for implementors:** the default implementation of this method relies on the *information_schema.tables* view.
        """
        if isinstance(table, TableReference) and table.virtual:
            raise VirtualTableError(table)
        table = table if isinstance(table, str) else table.full_name
        db_name = self._db.database_name()

        query_template = textwrap.dedent(f"""
            SELECT table_type
            FROM information_schema.tables
            WHERE table_catalog = {self._prep_placeholder}
                AND table_name = {self._prep_placeholder}
                AND table_catalog = current_database()
            """)
        cur = self._db.cursor()
        cur.execute(query_template, (db_name, table))
        result_set = cur.fetchall()

        assert result_set is not None
        if not result_set:
            raise ValueError(f"Table '{table}' not found in database '{db_name}'")
        table_type = result_set[0][0]
        return table_type == "VIEW"

    def lookup_column(
        self,
        column: ColumnReference | str,
        candidate_tables: Iterable[TableReference],
        *,
        expect_match: bool = False,
    ) -> Optional[TableReference]:
        """Searches for a table that owns the given column.

        Parameters
        ----------
        column : ColumnReference | str
            The column that is being looked up
        candidate_tables : Iterable[TableReference]
            Tables that could possibly own the given column
        expect_match : bool, optional
            If enabled, an error is raised whenever no table is found. Otherwise *None* is returned. By default, this is
            disabled.

        Returns
        -------
        TableReference
            The first of the `candidate_tables` that has a column of similar name.

        Raises
        ------
        ValueError
            If `expect_match` is enabled and none of the candidate tables has a column of the given name.

        Notes
        -----
        **Hint for implementors:** the default implementation of this method (transitively) relies on the
        *information_schema.columns* view.
        """
        for candidate in candidate_tables:
            candidate_cols = self.columns(candidate)
            if column in candidate_cols:
                return candidate

        if expect_match:
            raise ValueError(
                f"Column '{column}' not found in any of the candidate tables: {candidate_tables}"
            )
        return None

    def is_primary_key(self, column: ColumnReference) -> bool:
        """Checks, whether a column is the primary key for its associated table.

        Parameters
        ----------
        column : ColumnReference
            The column to check

        Returns
        -------
        bool
            Whether the column is the primary key of its table. If it is part of a compound primary key, this is *False*.

        Raises
        ------
        postbound.qal.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)

        Notes
        -----
        **Hint for implementors:** the default implementation of this method relies on the
        *information_schema.table_constraints* and *information_schema.constraint_column_usage* views.
        """
        if (table := column.table) is None:
            raise UnboundColumnError(column)

        schema_placeholder = (
            self._prep_placeholder if table.schema else "current_schema()"
        )
        query_template = textwrap.dedent(f"""
            SELECT ccu.column_name
            FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
                    AND tc.table_catalog = ccu.table_catalog
                    AND tc.table_schema = ccu.table_schema
                    AND tc.table_name = ccu.table_name
                    AND tc.constraint_catalog = ccu.constraint_catalog
            WHERE tc.table_name = {self._prep_placeholder}
                AND ccu.column_name = {self._prep_placeholder}
                AND tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_catalog = current_database()
                AND tc.table_schema = {schema_placeholder};
            """)

        params = [table.full_name, column.name]
        if table.schema:
            params.append(table.schema)

        cur = self._db.cursor()
        cur.execute(query_template, params)
        result_set = cur.fetchone()

        return result_set is not None

    def primary_key_column(
        self, table: TableReference | str
    ) -> Optional[BoundColumnReference]:
        """Determines the primary key column of a specific table.

        Parameters
        ----------
        table : TableReference | str
            The table to check

        Returns
        -------
        Optional[BoundColumnReference]
            The primary key if it exists, or *None* otherwise.

        Notes
        -----
        **Hint for implementors:** the default implementation of this method relies on the
        *information_schema.table_constraints* and *information_schema.constraint_column_usage* views.
        """
        table = table if isinstance(table, TableReference) else TableReference(table)
        schema_placeholder = (
            self._prep_placeholder if table.schema else "current_schema()"
        )
        query_template = textwrap.dedent(f"""
            SELECT ccu.column_name
            FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
                    AND tc.table_catalog = ccu.table_catalog
                    AND tc.table_schema = ccu.table_schema
                    AND tc.table_name = ccu.table_name
                    AND tc.constraint_catalog = ccu.constraint_catalog
            WHERE tc.table_name = {self._prep_placeholder}
                AND tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_catalog = current_database()
                AND tc.table_schema = {schema_placeholder};
            """)

        params = [table.full_name]
        if table.schema:
            params.append(table.schema)

        cur = self._db.cursor()
        cur.execute(query_template, params)
        result_set = cur.fetchall()

        if not result_set:
            return None
        elif len(result_set) > 1:
            raise ValueError(
                f"Table {table} has multiple primary key columns: {result_set}"
            )
        col = result_set[0][0]
        return BoundColumnReference(col, table)

    def has_secondary_index(self, column: ColumnReference) -> bool:
        """Checks, whether a secondary index is available for a specific column.

        Parameters
        ----------
        column : ColumnReference
            The column to check

        Returns
        -------
        bool
            Whether a secondary index of any kind was created for the column. Compound indexes and primary key indexes
            fail this test.

        Raises
        ------
        postbound.qal.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)

        Notes
        -----
        **Hints for implementors:**
        The default implementation of this method assumes that each foreign key column and each column with a UNIQUE constraint
        has an associated index. If this should not be the case, a custom implementation needs to be supplied.
        Furthermore, the implementation relies on the *information_schema.table_constraints*,
        *information_schema.constraint_column_usage* and *information_schema.key_column_usage* views.
        """

        # The documentation of has_index() references an implementation detail of this method.
        # Make sure to keep the two in sync.

        if (table := column.table) is None:
            raise UnboundColumnError(column)

        schema_placeholder = (
            self._prep_placeholder if table.schema else "current_schema()"
        )

        # The query template is much more complicated here, due to the different semantics of the constraint_column_usage
        # view. For UNIQUE constraints, the column is the column that is constrained. However, for foreign keys, the column
        # is the column that is being referenced.
        query_template = textwrap.dedent(f"""
            SELECT ccu.column_name
            FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
                    AND tc.table_catalog = ccu.table_catalog
                    AND tc.table_schema = ccu.table_schema
                    AND tc.table_name = ccu.table_name
                    AND tc.constraint_catalog = ccu.constraint_catalog
            WHERE tc.table_name = {self._prep_placeholder}
                AND ccu.column_name = {self._prep_placeholder}
                AND tc.constraint_type = 'UNIQUE'
                AND tc.table_catalog = current_database()
                AND tc.table_schema = {schema_placeholder}
            UNION
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_catalog = kcu.table_catalog
                    AND tc.table_schema = kcu.table_schema
                    AND tc.table_name = kcu.table_name
                    AND tc.constraint_catalog = kcu.constraint_catalog
            WHERE tc.table_name = {self._prep_placeholder}
                AND kcu.column_name = {self._prep_placeholder}
                AND tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_catalog = current_database()
                AND tc.table_schema = {schema_placeholder};
            """)

        # Due to the UNION query, we need to repeat the placeholders. While the implementation is definitely not elegant,
        # this solution is arguably better than relying on named parameters which might or might not be supported by the
        # target database.
        params = [table.full_name, column.name]
        if table.schema:
            params.append(table.schema)
        params.extend([table.full_name, column.name])
        if table.schema:
            params.append(table.schema)

        cur = self._db.cursor()
        cur.execute(query_template, params)
        result_set = cur.fetchone()

        return result_set is not None

    def foreign_keys_on(self, column: ColumnReference) -> set[BoundColumnReference]:
        """Fetches all foreign key constraints that are specified on a specific column.

        The provided columns are the target columns that are referenced by the foreign key constraint. E.g., suppose there are
        tables A and B with columns x and y. We specify a foreign key constraint on column y to ensure that all values in y
        reference a value in x. Then, calling this method on column y will return column x. If there are multiple foreign key
        constraints on the same column, all of them will be returned.

        Parameters
        ----------
        column : ColumnReference
            The column to check. All foreign keys that are "pointing from" this column to another column are returned.

        Returns
        -------
        set[BoundColumnReference]
            The columns that are "pointed to" by foreign key constraints on the given column. If no such foreign keys exist,
            an empty set is returned.

        Raises
        ------
        postbound.qal.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)
        """
        if (table := column.table) is None:
            raise UnboundColumnError(column)

        schema_placeholder = (
            self._prep_placeholder if table.schema else "current_schema()"
        )
        query_template = f"""
            SELECT ccu.table_schema, ccu.table_name, ccu.column_name
            FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                ON tc.table_name = kcu.table_name
                    AND tc.table_catalog = kcu.table_catalog
                    AND tc.table_schema = kcu.table_schema
                    AND tc.constraint_catalog = kcu.constraint_catalog
                    AND tc.constraint_schema = kcu.constraint_schema
                    AND tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_catalog = ccu.constraint_catalog
                    AND tc.constraint_schema = ccu.constraint_schema
                    AND tc.constraint_name = ccu.constraint_name
            WHERE tc.table_catalog = current_database()
                AND tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = {schema_placeholder}
                AND tc.table_name = {self._prep_placeholder}
                AND kcu.column_Name = {self._prep_placeholder};
            """
        params = [table.full_name, column.name]
        if table.schema:
            params = [table.schema] + params

        cur = self._db.cursor()
        cur.execute(query_template, params)
        result_set = cur.fetchall()
        assert result_set is not None

        return {
            BoundColumnReference(row[1], TableReference(row[0], schema=table.schema))
            for row in result_set
        }

    def has_index(self, column: ColumnReference) -> bool:
        """Checks, whether there is any index structure available on a column

        Parameters
        ----------
        column : ColumnReference
            The column to check

        Returns
        -------
        bool
            Whether any kind of index (primary, or secondary) is available for the column. Only compound indexes will
            fail this test.

        Raises
        ------
        postbound.qal.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)

        Notes
        -----
        **Hints for implementors:** the default implementation of this method (transitively) relies on the
        **information_schema.table_constraints** and **information_schema.constraint_column_usage** views. It assumes that
        primary keys, foreign keys and unique constraints are all associated with an index structure. If this is not the case,
        a custom implementation needs to be supplied.
        """
        return self.is_primary_key(column) or self.has_secondary_index(column)

    def indexes_on(self, column: ColumnReference) -> set[str]:
        """Retrieves the names of all indexes of a specific column.

        Parameters
        ----------
        column : ColumnReference
            The column to check.

        Returns
        -------
        set[str]
            The indexes. If no indexes are available, the set will be empty.

        Raises
        ------
        postbound.qal.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)

        Notes
        -----
        **Hints for implementors:** the default implementation of this method assumes that primary keys, foreign keys and
        unique constraints are all associated with an index structure. It provides the names of the corresponding constraints.
        The implementation relies on the *information_schema.table_constraints*, *information_schema.constraint_column_usage*
        and *information_schema.key_column_usage* views.
        """
        if (table := column.table) is None:
            raise UnboundColumnError(column)

        schema_placeholder = (
            self._prep_placeholder if table.schema else "current_schema()"
        )

        # The query template is much more complicated here, due to the different semantics of the constraint_column_usage
        # view. For UNIQUE constraints, the column is the column that is constrained. However, for foreign keys, the column
        # is the column that is being referenced.
        query_template = textwrap.dedent(f"""
            SELECT tc.constraint_name
            FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
                    AND tc.table_catalog = ccu.table_catalog
                    AND tc.table_schema = ccu.table_schema
                    AND tc.table_name = ccu.table_name
                    AND tc.constraint_catalog = ccu.constraint_catalog
            WHERE tc.table_name = {self._prep_placeholder}
                AND ccu.column_name = {self._prep_placeholder}
                AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
                AND tc.table_catalog = current_database()
                AND tc.table_schema = {schema_placeholder}
            UNION
            SELECT tc.constraint_name
            FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_catalog = kcu.table_catalog
                    AND tc.table_schema = kcu.table_schema
                    AND tc.table_name = kcu.table_name
                    AND tc.constraint_catalog = kcu.constraint_catalog
            WHERE tc.table_name = {self._prep_placeholder}
                AND kcu.column_name = {self._prep_placeholder}
                AND tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_catalog = current_database()
                AND tc.table_schema = {schema_placeholder};
            """)

        # Due to the UNION query, we need to repeat the placeholders. While the implementation is definitely not elegant,
        # this solution is arguably better than relying on named parameters which might or might not be supported by the
        # target database.
        params = [table.full_name, column.name]
        if table.schema:
            params.append(table.schema)
        params.extend([table.full_name, column.name])
        if table.schema:
            params.append(table.schema)

        cur = self._db.cursor()
        cur.execute(query_template, params)
        result_set = cur.fetchall()
        assert result_set is not None

        return {row[0] for row in result_set}

    def datatype(self, column: ColumnReference) -> str:
        """Retrieves the (physical) data type of a column.

        The provided type can be a standardized SQL-type, but it can be a type specific to the concrete database
        system just as well. It is up to the user to figure this out and to react accordingly.

        Parameters
        ----------
        column : ColumnReference
            The colum to check

        Returns
        -------
        str
            The datatype. Will never be empty.

        Raises
        ------
        postbound.qal.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)

        Notes
        -----
        **Hint for implementors:** the default implementation of this method relies on the *information_schema.columns* view.
        """
        if (table := column.table) is None:
            raise UnboundColumnError(column)

        schema_placeholder = (
            self._prep_placeholder if table.schema else "current_schema()"
        )
        query_template = textwrap.dedent(f"""
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = {self._prep_placeholder}
                AND column_name = {self._prep_placeholder}
                AND table_catalog = current_database()
                AND table_schema = {schema_placeholder};
            """)

        params = [table.full_name, column.name]
        if table.schema:
            params.append(table.schema)

        cur = self._db.cursor()
        cur.execute(query_template, params)
        result_set = cur.fetchone()
        assert result_set

        return result_set[0]

    def is_nullable(self, column: ColumnReference) -> bool:
        """Checks, whether a specific column may contain NULL values.

        Parameters
        ----------
        column : ColumnReference
            The column to check

        Returns
        -------
        bool
            Whether the column may contain NULL values

        Raises
        ------
        postbound.qal.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)

        Notes
        -----
        **Hint for implementors:** the default implementation of this method relies on the *information_schema.columns* view.
        """
        if (table := column.table) is None:
            raise UnboundColumnError(column)

        schema_placeholder = (
            self._prep_placeholder if table.schema else "current_schema()"
        )
        query_template = textwrap.dedent(f"""
            SELECT is_nullable
            FROM information_schema.columns
            WHERE table_name = {self._prep_placeholder}
                AND column_name = {self._prep_placeholder}
                AND table_catalog = current_database()
                AND table_schema = {schema_placeholder};
            """)

        params = [table.full_name, column.name]
        if table.schema:
            params.append(table.schema)

        cur = self._db.cursor()
        cur.execute(query_template, params)
        result_set = cur.fetchone()
        assert result_set

        return result_set[0] == "YES"

    def as_graph(self) -> nx.DiGraph:
        """Constructs a compact representation of the database schema.

        The schema is expressed as a directed graph. Each table is represented as a node. Nodes contain the following
        attributes:
        - `columns`: a list of all columns in the table
        - `data_type`: a dictionary mapping each column to its data type
        - `primary_key`: the primary key of the table (if it exists, otherwise *None*)

        In addition, edges are used to model foreign key constraints. Each edge points from the table that contains the foreign
        key (column *y* in the example in `foreign_keys_on`) to the table that is referenced by the foreign key (*x* in the
        example in `foreign_keys_on`). Edges contain an attribute `foreign_keys` with a list of the foreign key
        relationships. Each such constraint is described by a `ForeignKeyRef`.
        """
        g = nx.DiGraph()
        all_columns: set[BoundColumnReference] = set()

        for table in self.tables():
            if self.is_view(table):
                continue

            cols = self.columns(table)
            dtypes = {col: self.datatype(col) for col in cols}
            pkey = self.primary_key_column(table)
            g.add_node(table, columns=cols, data_type=dtypes, primary_key=pkey)

            all_columns |= set(cols)

        for col in all_columns:
            foreign_keys = self.foreign_keys_on(col)
            for fk_target in foreign_keys:
                fk_constraint = ForeignKeyRef(col, fk_target)
                current_edge = g.edges.get([col.table, fk_target.table])

                if current_edge:
                    current_edge["foreign_keys"].append(fk_constraint)
                else:
                    g.add_edge(col.table, fk_target.table, foreign_keys=[fk_constraint])

        return g

    def join_equivalence_keys(
        self,
    ) -> dict[BoundColumnReference, set[BoundColumnReference]]:
        """Calculates the equivalence classes of joinable columns in the database schema.

        Two columns are considered joinable, if they are linked by a foreign key constraint.
        For example, consider a schema with three tables R, S and T with foreign keys R.a -> S.b and S.b -> T.c.
        Then, the columns R.a, S.b and T.c are all joinable and form an equivalence class.
        Likewise, the constraints R.a -> T.c and S.b -> T.c would establish the same equivalence class.
        On the other hand, the constraints R.a -> S.b and S.c -> T.d create two different equivalence classes.

        Returns
        -------
        dict[BoundColumnReference, set[BoundColumnReference]]
            A mapping from each column to its equivalence class, i.e. the set of all columns that are joinable with it
            (including itself).
        """
        columns = util.flatten(self.columns(table) for table in self.tables())
        g = nx.Graph()
        for col in columns:
            edges = [(col, fk_target) for fk_target in self.foreign_keys_on(col)]
            g.add_edges_from(edges)

        eq_keys: dict[BoundColumnReference, set[BoundColumnReference]] = {}
        for component in nx.connected_components(g):
            for key in component:
                eq_keys[key] = component

        return eq_keys

    def join_equivalence_classes(self) -> Iterable[set[BoundColumnReference]]:
        """Calculates the quivalence classes of joinable columns in the database schema.

        This method is similar to `join_equivalence_keys`, but returns the different equivalence classes instead of a
        mapping. See its documentation for more details.

        See Also
        --------
        join_equivalence_keys
        """
        columns = util.flatten(self.columns(table) for table in self.tables())
        g = nx.Graph()
        for col in columns:
            edges = [(col, fk_target) for fk_target in self.foreign_keys_on(col)]
            g.add_edges_from(edges)
        return list(nx.connected_components(g))

    @overload
    def __getitem__(self, key: TableReference) -> TableInfo: ...

    @overload
    def __getitem__(self, key: ColumnReference) -> ColumnInfo: ...

    def __getitem__(self, key):
        match key:
            case ColumnReference():
                return ColumnInfo(
                    column=key,
                    table=key.table,
                    datatype=self.datatype(key),
                    nullable=self.is_nullable(key),
                    indexed=self.has_index(key),
                    primary_key=self.is_primary_key(key),
                )
            case TableReference():
                schema_graph = self.as_graph()
                col_infos = [self[col] for col in self.columns(key)]
                outgoing_fks = set_union(
                    edge[2] for edge in schema_graph.out_edges(key, data="foreign_keys")
                )
                incoming_fks = set_union(
                    edge[2] for edge in schema_graph.in_edges(key, data="foreign_keys")
                )
                return TableInfo(
                    table=key,
                    columns=col_infos,
                    primary_key=self.primary_key_column(key),
                    outgoing_foreign_keys=outgoing_fks,
                    incoming_foreign_keys=incoming_fks,
                )
            case str():
                try:
                    ref = TableReference(key)
                    return self[ref]
                except KeyError:
                    pass

                try:
                    col, tab = key.split(".", 1)
                    ref = ColumnReference(col, TableReference(tab))
                    return self[ref]
                except KeyError:
                    raise KeyError(f"Key '{key}' not found in database schema")
            case _:
                raise KeyError(
                    f"Unsupported key type: {type(key).__name__}. "
                    "Only TableReference, ColumnReference, or str are supported."
                )

    def __iter__(self) -> Iterator[TableReference]:
        return iter(self.tables())

    def __len__(self) -> int:
        return len(self.tables())

    def __hash__(self) -> int:
        return hash(self._db)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._db == other._db

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Database schema of {self._db}"


class MostCommonValues[T]:
    """Most Common Values contain an ordered list of (column value, frequency) pairs.

    The list is sorted in descending order of the frequencies. Ties are broken according to
    the column values (once again in descending order).

    By default, Most Common Values acts as a drop-in replacement for a list of tuples, so all
    sequence-like methods like *__iter__*, *__getitem__*, etc. are implemented in terms of
    lists.

    In addition, Most Common Values provide a number of utility functions like `min_freq` or
    `frequency_of` to extract commonly-used information.

    Parameters
    ----------
    mcvs: Sequence[tuple[T, int]]
        The value, frequency pairs in the list.
    """

    def __init__(self, mcvs: Iterable[tuple[T, int]]) -> None:
        self.mcvs: Sequence[tuple[T, int]] = list(mcvs)
        self.mcvs.sort(key=lambda pair: (pair[1], pair[0]), reverse=True)
        self._mcv_map = dict(self.mcvs)
        if self.mcvs:
            self._min_freq = self.mcvs[-1][1]
            self._max_freq = self.mcvs[0][1]
        else:
            self._min_freq = 0
            self._max_freq = 0

    @property
    def k(self) -> int:
        """Get the number of values in the MCV list."""
        return len(self.mcvs)

    @property
    def min_freq(self) -> int:
        """Get the lowest frequency in the MCV list.

        This is equivalent to the frequeny of the final value in the list.
        """
        return self._min_freq

    @property
    def max_freq(self) -> int:
        """Get the highest frequency in the MCV list.

        This is equivalent to the frequency of the first value in the list.
        """
        return self._max_freq

    @property
    def values(self) -> Sequence[T]:
        """Get the values in the MCV list, ordered by their frequency (highest first)."""
        return list(self._mcv_map.keys())

    @property
    def frequencies(self) -> Sequence[int]:
        """Get the frequencies in the MCV list, starting with the highest frequency."""
        return list(self._mcv_map.values())

    @overload
    def frequency_of(self, value: T) -> Optional[int]: ...

    @overload
    def frequency_of(self, value: T, *, bound_missing: Literal[True]) -> int: ...

    @overload
    def frequency_of(
        self, value: T, *, bound_missing: Literal[False]
    ) -> Optional[int]: ...

    @overload
    def frequency_of(self, value: T, *, default: int) -> int: ...

    def frequency_of(
        self, value: T, *, default: int | None = None, bound_missing: bool = False
    ) -> Optional[int]:
        """Load the frequency of a specific value.

        Parameters
        ----------
        value : T
            The value to check
        default : int, optional
            The frequency to return if the given value is not in the MCV list. If not provided,
            the behavior depends on the `bound_missing` parameter.
        bound_missing : bool, optional
            If enabled, the frequency of missing values (i.e. values that are not in the MCV list)
            is assumed to be at least the frequency of the least common value in the MCV list.
            Otherwise, *None* is returned to indicate a missing value.
            If a `default` value is provided, this parameter is ignored. By default, this is disabled.
        """
        if default is not None:
            fallback = default
        elif bound_missing:
            fallback = self._min_freq
        else:
            fallback = None
        return self._mcv_map.get(value, fallback)

    def top(self, k: int) -> MostCommonValues:
        """Provide the most common values up-to (and including) the k-th most frequent."""
        return MostCommonValues(self.mcvs[:k])

    def below(self, k: int) -> MostCommonValues:
        """Provide the most common values after the k-th most frequent."""
        return MostCommonValues(self.mcvs[k:])

    def __bool__(self) -> bool:
        return bool(self.mcvs)

    def __len__(self) -> int:
        return len(self.mcvs)

    def __contains__(self, value: object) -> bool:
        return value in self._mcv_map

    def __iter__(self) -> Iterator[tuple[T, int]]:
        return iter(self.mcvs)

    def __getitem__(self, idx: int) -> tuple[T, int]:
        return self.mcvs[idx]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.mcvs)


type HistogramApproximation = Literal["approx-uni", "bound"]
"""The strategy to estimate the frequency of values that are not exactly on the bucket bounds of a histogram."""


def _infer_histogram_bounds[T](
    frequencies: Sequence[tuple[T, int]], *, n_bins: int, n_rows: int
) -> tuple[T, Sequence[T], Sequence[int]]:
    """Infer the bucket bounds and frequencies for a histogram from a list of (value, frequency) pairs."""
    if not frequencies:
        raise ValueError("Cannot infer histogram bounds from empty frequency list")

    bucket_size = n_rows // n_bins

    bounds: list[T] = []
    buckets: list[int] = []
    cumulative_freq = 0
    for value, freq in frequencies:
        cumulative_freq += freq
        if cumulative_freq < bucket_size:
            continue
        bounds.append(value)
        buckets.append(cumulative_freq)
        cumulative_freq = 0

    return frequencies[0][0], bounds, buckets


class Histogram[T: SupportsRichComparison[T]]:
    """Histograms are a common way to approximate the distribution of values in a column.

    Our implementation models a general histogram as a list of buckets. Each bucket is associated with a frequency, i.e.
    the number of elements that fall into the bucket. This might be an equi-depth or an equi-width histogram, but this is
    not a requirement.

    Currently, we only provide two high-level methods for cardinality estimation: `frequency_below` and `frequency_above`.
    The former estimates the number of elements that are less than or equal to a given value, while the latter estimates
    the number of elements that are greater than a given value. If the queried value does not align perfectly with the
    bucket bounds, the behavior depends on the selected `bucket_interpolation` strategy:

    - "bound" is a conservative strategy. It *bounds* the frequency of the queried value by the frequency of the entire
      bucket that contains it.
    - "approx-uni" is a more aggressive strategy. It assumes that the values within each bucket are uniformly distributed
      and uses a simple linear interpolation to estimate the frequency of the queried value.

    Histograms can be iterated over to access the individual (bucket bound, frequency) pairs. In addition, we provide
    *__getitem__* access to get the i-th bucket bound and its frequency.

    Parameters
    ----------
    bounds : Sequence[T]
        The upper bounds of the buckets. These must be sorted in ascending order. The lower bound of the first bucket is
        given by the additional `lower` parameter. The upper bound of the final bucket is implicitly given by the final
        element in this list.
    frequencies : Sequence[int]
        The frequencies of the buckets. The i-th element in this list corresponds to the i-th bucket.
    lower : T
        The lower bound of the first bucket. This is required to be less than or equal to the first element in `bounds`.
    bucket_interpolation : Literal["approx-uni", "bound"], optional
        The strategy to estimate the frequency of values that fall into a bucket, but are not exactly on the bucket bounds.
        The default is "bound".
    """

    def __init__(
        self,
        bounds: Iterable[T],
        frequencies: Iterable[int],
        *,
        lower: T,
        bucket_interpolation: HistogramApproximation = "bound",
    ) -> None:
        self._bounds = list(bounds)
        self._frequencies = list(frequencies)
        if not self.bounds:
            raise ValueError("Bounds cannot be empty")
        elif len(self._bounds) != len(self._frequencies):
            raise ValueError(
                "Bounds and frequencies must have the same length. "
                f"Got {len(self._bounds)} bounds and {len(self._frequencies)} frequencies."
            )

        self._n_rows = sum(self._frequencies)
        self._lower = lower
        self._freq_per_bucket = self._n_rows // len(self._bounds)
        self._bucket_interpolation = bucket_interpolation

    @property
    def bounds(self) -> Sequence[T]:
        """Get the upper bounds of all buckets."""
        return list(self._bounds)

    @property
    def n_buckets(self) -> int:
        """Get the number of buckets in the histogram."""
        return len(self._bounds)

    @property
    def freq_per_bucket(self) -> int:
        """Get the average number of elements per bucket.

        For equi-depth histograms, this number will be pretty accurate. For all other kinds of histograms, this number is
        really just an average.
        """
        return self._freq_per_bucket

    @property
    def n_rows(self) -> int:
        """Get the total number of rows that are represented by the histogram."""
        return self._n_rows

    @property
    def lower(self) -> T:
        """Get the lower bound of the first bucket (and thus of the entire histogram)."""
        return self._lower

    @property
    def upper(self) -> T:
        """Get the upper bound of the last bucket (and thus of the entire histogram)."""
        return self._bounds[-1]

    @property
    def bucket_interpolation(self) -> HistogramApproximation:
        """Get the strategy to estimate the frequency of values that are not exactly on the bucket bounds."""
        return self._bucket_interpolation

    @bucket_interpolation.setter
    def bucket_interpolation(self, strategy: HistogramApproximation) -> None:
        if strategy not in ("approx-uni", "bound"):
            raise ValueError(
                f"Unsupported bucket interpolation strategy: {strategy}. "
                "Supported strategies are 'approx-uni' and 'bound'."
            )
        self._bucket_interpolation = strategy

    def frequency_below(self, value: T) -> int:
        """Estimate the frequency of values that are less than or equal to a given value.

        If the value does not align exactly with the bucket bounds, the behavior depends on the selected
        `bucket_interpolation` strategy. See the class documentation for details.
        """
        upper_idx = bisect.bisect_right(self._bounds, value)
        if upper_idx == 0:
            return 0
        elif upper_idx == len(self._bounds):
            return self._n_rows

        if self._bucket_interpolation == "bound":
            return upper_idx * self._freq_per_bucket

        assert self._bucket_interpolation == "approx-uni"

        if not isinstance(value, (int, float, date, datetime)):
            raise ValueError(
                "Cannot estimate frequency within bucket. "
                f"Strategy '{self._bucket_interpolation}' is unknown or unsupported "
                f"for histograms of type {type(value).__name__}."
            )

        lower_idx = upper_idx - 1
        upper_bound = self._bounds[upper_idx]
        lower_bound = self._bounds[lower_idx]
        in_bucket_frac = (value - lower_bound) / (upper_bound - lower_bound)
        return (lower_idx + in_bucket_frac) * self._freq_per_bucket

    def frequency_above(self, value: T) -> int:
        """Estimate the frequency of values that are greater than a given value.

        Due to the way we model histograms, we cannot directly estimate the frequency of values greater or equal to a given
        value, but only the frequency of values that are strictly greater than it.

        If the value does not align exactly with the bucket bounds, the behavior depends on the selected
        `bucket_interpolation` strategy. See the class documentation for details.
        """
        lower_idx = bisect.bisect_left(self._bounds, value)
        if lower_idx == 0:
            return self._n_rows
        elif lower_idx == len(self._bounds):
            return 0

        if self._bucket_interpolation == "bound":
            return lower_idx * self._freq_per_bucket

        assert self._bucket_interpolation == "approx-uni"

        if not isinstance(value, (int, float, date, datetime)):
            raise ValueError(
                "Cannot estimate frequency within bucket. "
                f"Strategy '{self._bucket_interpolation}' is unknown or unsupported "
                f"for histograms of type {type(value).__name__}."
            )

        upper_idx = lower_idx + 1
        upper_bound = self._bounds[upper_idx]
        lower_bound = self._bounds[lower_idx]
        in_bucket_frac = (value - lower_bound) / (upper_bound - lower_bound)
        return (upper_idx + in_bucket_frac) * self._freq_per_bucket

    def __len__(self) -> int:
        return len(self._bounds)

    def __iter__(self) -> Iterator[tuple[T, int]]:
        return iter(zip(self._bounds, self._frequencies))

    def __getitem__(self, idx: int) -> tuple[T, int]:
        return self._bounds[idx], self._frequencies[idx]

    def __json__(self) -> jsondict:
        return {
            "bounds": self._bounds,
            "frequencies": self._frequencies,
            "lower": self._lower,
            "n_rows": self._n_rows,
            "bucket_interpolation": self._bucket_interpolation,
        }

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._bounds == other._bounds
            and self._frequencies == other._frequencies
            and self._lower == other._lower
            and self._n_rows == other._n_rows
            and self._bucket_interpolation == other._bucket_interpolation
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        buckets = ", ".join(
            f"{bound} ({freq})" for bound, freq in zip(self._bounds, self._frequencies)
        )
        return f"Histogram(buckets=[{buckets}], lower={self._lower})"


class DatabaseStatistics(abc.ABC):
    """The statistics interface provides unified access to table-level and column-level statistics.

    There are two main challenges when implementing a generalized statistics interface for different database systems.
    The first one is the non-deterministic creation and maintenance of statistics by most database systems. This means
    that creating two identical databases on the same database system on the same machine might still yield different
    statistical values. This is because database systems oftentimes create statistics from random samples of column
    values to speed up computation. However, such variability hurts our efforts to enable reproducible experiments
    since different performances metrics might not be due to differences in the optimization algorithms but due to bad
    luck when creating the statistics (whether it is a good sign if an algorithm is that fragile to deviations in
    statistics is another question). The second main challenge is that different database systems maintain different
    statistics. Even though many statistics are considered quite "basic" by the research community, not all systems
    developers deemed all statistics necessary for their optimizer. Once again, this can severly hinder the application
    of an optimization algorithm if it relies on a basic statistic that just happens to not be available on the desired
    target database system.

    To address both of these issues, the statistics interface operates in two different modes: in *native* mode it
    simply delegates all requests to statistical information to the corresponding catalogs of the database systems.
    Alternatively, the statistics interface can create the illusion of a normalized and standardized statistics
    catalogue. This so-called *emulated* mode does not rely on the statistics catalogs and issues equivalent SQL
    queries instead. For example, if a statistic on the number of distinct values of a column is requested, this
    emulated by running a *SELECT COUNT(DISTINCT column) FROM table* query.

    The current mode can be customized using the boolean `emulated` property. If the statistics interface operates in
    native mode (i.e. based on the actual statistics catalog) and the user requests a statistic that is not available
    in the selected database system, the behavior depends on another attribute: `enable_emulation_fallback`. If this
    boolean attribute is *True*, an emulated statistic will be calculated instead. Otherwise, an
    `UnsupportedDatabaseFeatureError` is raised.

    Since the live computation of emulated statistics can be costly, the statistics interface has its own
    `cache_enabled` attribute. It can be set to `None` to use the default caching behavior of the database system.
    However, if this attribute is set to `True` or `False` directly, caching will be used accordingly for all
    compute-intensive statistics operations (and only such operations). Once again, this only works because PostBOUND
    assumes the database to be immutable.

    Parameters
    ----------
    db : Database
        The database for which the schema information should be read. This is required to hook into the database cache
        and to obtain the cursors to actuall execute queries.
    emulated : bool, optional
        Whether the statistics interface should operate in emulation mode. To enable reproducibility, this is *True*
        by default
    enable_emulation_fallback : bool, optional
        Whether emulation should be used for unsupported statistics when running in native mode, by default True
    cache_enabled : Optional[bool], optional
        Whether emulated statistics queries should be subject to caching, by default True. Set to *None* to use the
        caching behavior of the `db`

    See Also
    --------
    postbound.postbound.OptimizationPipeline : The basic optimization process applied by PostBOUND
    """

    def __init__(
        self,
        db: Database,
        *,
        emulated: bool = True,
        enable_emulation_fallback: bool = True,
        cache_enabled: Optional[bool] = True,
    ) -> None:
        self.emulated = emulated
        self.enable_emulation_fallback = enable_emulation_fallback
        self.cache_enabled = cache_enabled
        self._db = db

    @overload
    def total_rows(self, table: TableReference) -> Optional[int]: ...

    @overload
    def total_rows(
        self,
        table: TableReference,
        *,
        emulated: Literal[True],
    ) -> int: ...

    @overload
    def total_rows(
        self, table: TableReference, *, emulated: Literal[False, None]
    ) -> Optional[int]: ...

    @overload
    def total_rows(
        self, table: TableReference, *, emulated: Optional[bool]
    ) -> Optional[bool]: ...

    @overload
    def total_rows(
        self, table: TableReference, *, cache_enabled: Optional[bool]
    ) -> Optional[int]: ...

    @overload
    def total_rows(
        self,
        table: TableReference,
        *,
        emulated: Literal[True],
        cache_enabled: Optional[bool],
    ) -> Optional[int]: ...

    @overload
    def total_rows(
        self,
        table: TableReference,
        *,
        emulated: Literal[False, None],
        cache_enabled: Optional[bool],
    ) -> Optional[int]: ...

    @overload
    def total_rows(
        self,
        table: TableReference,
        *,
        emulated: Optional[bool],
        cache_enabled: Optional[bool],
    ) -> Optional[int]: ...

    def total_rows(
        self,
        table: TableReference,
        *,
        emulated: Optional[bool] = None,
        cache_enabled: Optional[bool] = None,
    ) -> Optional[int]:
        """Provides (an estimate of) the total number of rows in a table.

        Parameters
        ----------
        table : TableReference
            The table to check
        emulated : Optional[bool], optional
            Whether to force emulation mode for this single call. Defaults to *None* which indicates that the
            emulation setting of the statistics interface should be used.
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to *None* which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        Optional[int]
            The total number of rows in the table. If no such statistic exists, but the database system in principle
            maintains the statistic, *None* is returned. For example, this situation can occur if the database system
            only maintains a row count if the table has at least a certain size and the table in question did not reach
            that size yet.

        Raises
        ------
        VirtualTableError
            If the given table is virtual (e.g. subquery or CTE)
        """
        if table.virtual:
            raise VirtualTableError(table)
        if emulated or (emulated is None and self.emulated):
            return self._calculate_total_rows(
                table, cache_enabled=self._determine_caching_behavior(cache_enabled)
            )
        else:
            return self._retrieve_total_rows_from_stats(table)

    def distinct_values(
        self,
        column: ColumnReference,
        *,
        emulated: Optional[bool] = None,
        cache_enabled: Optional[bool] = None,
    ) -> Optional[int]:
        """Legacy alias for `num_distinct`.

        .. deprecated:: v0.20.2
            distinct_values() is deprecated due to the confusing name. Use the more aptly-named num_distinct(),
            which provides exactly the same interface and functionality.
        """
        return self.num_distinct(column, emulated=emulated, cache_enabled=cache_enabled)

    @overload
    def num_distinct(self, column: ColumnReference) -> Optional[int]: ...

    @overload
    def num_distinct(
        self,
        column: ColumnReference,
        *,
        emulated: Literal[True],
    ) -> int: ...

    @overload
    def num_distinct(
        self,
        column: ColumnReference,
        *,
        emulated: Literal[False, None],
    ) -> Optional[int]: ...

    @overload
    def num_distinct(
        self,
        column: ColumnReference,
        *,
        emulated: Optional[bool],
    ) -> Optional[int]: ...

    @overload
    def num_distinct(
        self, column: ColumnReference, *, cache_enabled: Optional[bool]
    ) -> Optional[int]: ...

    @overload
    def num_distinct(
        self,
        column: ColumnReference,
        *,
        emulated: Literal[True],
        cache_enabled: Optional[bool],
    ) -> int: ...

    @overload
    def num_distinct(
        self,
        column: ColumnReference,
        *,
        emulated: Literal[False, None],
        cache_enabled: Optional[bool],
    ) -> Optional[int]: ...

    @overload
    def num_distinct(
        self,
        column: ColumnReference,
        *,
        emulated: Optional[bool],
        cache_enabled: Optional[bool],
    ) -> Optional[int]: ...

    def num_distinct(
        self,
        column: ColumnReference,
        *,
        emulated: Optional[bool] = None,
        cache_enabled: Optional[bool] = None,
    ) -> Optional[int]:
        """Provides (an estimate of) the total number of different column values of a specific column.

        Parameters
        ----------
        column : ColumnReference
            The column to check
        emulated : Optional[bool], optional
            Whether to force emulation mode for this single call. Defaults to *None* which indicates that the
            emulation setting of the statistics interface should be used.
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to *None* which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        Optional[int]
            The number of distinct values in the column. If no such statistic exists, but the database system in
            principle maintains the statistic, *None* is returned. For example, this situation can occur if the
            database system only maintains a distinct value count if the column values are distributed in a
            sufficiently diverse way.

        Raises
        ------
        postbound.qal.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)
        """
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        elif column.table.virtual:
            raise VirtualTableError(column.table)
        if emulated or (emulated is None and self.emulated):
            return self._calculate_distinct_values(
                column, cache_enabled=self._determine_caching_behavior(cache_enabled)
            )
        else:
            return self._retrieve_distinct_values_from_stats(column)

    @overload
    def min_max(self, column: ColumnReference) -> Optional[tuple[Any, Any]]: ...

    @overload
    def min_max(
        self, column: ColumnReference, *, emulated: Literal[True]
    ) -> tuple[Any, Any]: ...

    @overload
    def min_max(
        self, column: ColumnReference, *, emulated: Literal[False, None]
    ) -> Optional[tuple[Any, Any]]: ...

    @overload
    def min_max(
        self, column: ColumnReference, *, emulated: Optional[bool]
    ) -> Optional[tuple[Any, Any]]: ...

    @overload
    def min_max(
        self, column: ColumnReference, *, cache_enabled: Optional[bool]
    ) -> Optional[tuple[Any, Any]]: ...

    @overload
    def min_max(
        self,
        column: ColumnReference,
        *,
        emulated: Literal[True],
        cache_enabled: Optional[bool],
    ) -> tuple[Any, Any]: ...

    @overload
    def min_max(
        self,
        column: ColumnReference,
        *,
        emulated: Literal[False, None],
        cache_enabled: Optional[bool],
    ) -> Optional[tuple[Any, Any]]: ...

    @overload
    def min_max(
        self,
        column: ColumnReference,
        *,
        emulated: Optional[bool],
        cache_enabled: Optional[bool],
    ) -> Optional[tuple[Any, Any]]: ...

    def min_max(
        self,
        column: ColumnReference,
        *,
        emulated: Optional[bool] = None,
        cache_enabled: Optional[bool] = None,
    ) -> Optional[tuple[Any, Any]]:
        """Provides (an estimate of) the minimum and maximum values in a column.

        Parameters
        ----------
        column : ColumnReference
            The column to check
        emulated : Optional[bool], optional
            Whether to force emulation mode for this single call. Defaults to *None* which indicates that the
            emulation setting of the statistics interface should be used.
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to *None* which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        Optional[tuple[Any, Any]]
            A tuple of minimum and maximum value. If no such statistic exists, but the database system in principle
            maintains the statistic, *None* is returned. For example, this situation can occur if thec database
            system only maintains the min/max value if they are sufficiently far apart.

        Raises
        ------
        postbound.qal.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)
        """
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        elif column.table.virtual:
            raise VirtualTableError(column.table)
        if emulated or (emulated is None and self.emulated):
            return self._calculate_min_max_values(
                column, cache_enabled=self._determine_caching_behavior(cache_enabled)
            )
        else:
            return self._retrieve_min_max_values_from_stats(column)

    def most_common_values(
        self,
        column: ColumnReference,
        *,
        k: Optional[int] = 10,
        emulated: Optional[bool] = None,
        cache_enabled: Optional[bool] = None,
    ) -> MostCommonValues:
        """Provides (an estimate of) the total number of occurrences of the `k` most frequent values of a column.

        Parameters
        ----------
        column : ColumnReference
            The column to check
        k : Optional[int], optional
            The maximum number of most common values to return. Defaults to 10. If there are less values available, all
            of the available values will be returned. Setting this to *None* or a negative value will return the frequency of all distinct values.
        emulated : Optional[bool], optional
            Whether to force emulation mode for this single call. Defaults to *None* which indicates that the
            emulation setting of the statistics interface should be used.
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to *None* which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        MostCommonValues
            The most common values in pairs of (value, frequency), starting with the highest frequency. Notice that
            this sequence can be empty if no values are available. This can happen if the database system in principle
            maintains this statistic but does considers the value distribution to uniform to make the maintenance
            worthwhile. Likewise, if less common values exist than the requested `k` value, only the available values
            will be returned (and the sequence will be shorter than `k` in that case).

        Raises
        ------
        postbound.qal.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)
        """
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        elif column.table.virtual:
            raise VirtualTableError(column.table)

        if emulated or (emulated is None and self.emulated):
            mcv_list = self._calculate_most_common_values(
                column, k, cache_enabled=self._determine_caching_behavior(cache_enabled)
            )
        else:
            mcv_list = self._retrieve_most_common_values_from_stats(column, k)
        return MostCommonValues(mcv_list)

    @overload
    def histogram(self, column: ColumnReference) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self, column: ColumnReference, *, n_bins: Optional[int]
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self, column: ColumnReference, *, interpolation: HistogramApproximation
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self, column: ColumnReference, *, emulated: Literal[True]
    ) -> Histogram: ...

    @overload
    def histogram(
        self, column: ColumnReference, *, emulated: Literal[False, None]
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self, column: ColumnReference, *, emulated: Optional[bool]
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self, column: ColumnReference, *, cache_enabled: Optional[bool]
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        interpolation: HistogramApproximation,
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self, column: ColumnReference, *, n_bins: Optional[int], emulated: Literal[True]
    ) -> Histogram: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        emulated: Literal[False, None],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        emulated: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        cache_enabled: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        interpolation: HistogramApproximation,
        emulated: Literal[True],
    ) -> Histogram: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        interpolation: HistogramApproximation,
        emulated: Literal[False, None],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        interpolation: HistogramApproximation,
        emulated: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        interpolation: HistogramApproximation,
        cache_enabled: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        emulated: Literal[True],
        cache_enabled: Optional[bool],
    ) -> Histogram: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        emulated: Literal[False, None],
        cache_enabled: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        emulated: Optional[bool],
        cache_enabled: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        interpolation: HistogramApproximation,
        emulated: Literal[True],
    ) -> Histogram: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        interpolation: HistogramApproximation,
        emulated: Literal[False, None],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        interpolation: HistogramApproximation,
        emulated: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        emulated: Literal[True],
        cache_enabled: Optional[bool],
    ) -> Histogram: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        emulated: Literal[False, None],
        cache_enabled: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        emulated: Optional[bool],
        cache_enabled: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        interpolation: HistogramApproximation,
        cache_enabled: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        interpolation: HistogramApproximation,
        emulated: Literal[True],
        cache_enabled: Optional[bool],
    ) -> Histogram: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        interpolation: HistogramApproximation,
        emulated: Literal[False, None],
        cache_enabled: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        interpolation: HistogramApproximation,
        emulated: Optional[bool],
        cache_enabled: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        interpolation: HistogramApproximation,
        emulated: Literal[True],
        cache_enabled: Optional[bool],
    ) -> Histogram: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        interpolation: HistogramApproximation,
        emulated: Literal[False, None],
        cache_enabled: Optional[bool],
    ) -> Optional[Histogram]: ...

    @overload
    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int],
        interpolation: HistogramApproximation,
        emulated: Optional[bool],
        cache_enabled: Optional[bool],
    ) -> Optional[Histogram]: ...

    def histogram(
        self,
        column: ColumnReference,
        *,
        n_bins: Optional[int] = None,
        interpolation: HistogramApproximation = "approx-uni",
        emulated: Optional[bool] = None,
        cache_enabled: Optional[bool] = None,
    ) -> Optional[Histogram]:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        elif column.table.virtual:
            raise VirtualTableError(column.table)

        use_emulation = emulated or (emulated is None and self.emulated)
        if use_emulation and not n_bins:
            raise ValueError("n_bins must be set for emulated histograms.")
        if not use_emulation and n_bins:
            warnings.warn(
                "n_bins is ignored if the histogram is loaded from stats", stacklevel=2
            )

        if use_emulation:
            assert n_bins is not None
            return self._calculate_histogram(
                column, n_bins, interpolation=interpolation, cache_enabled=cache_enabled
            )
        else:
            return self._retrieve_histogram_from_stats(
                column, interpolation=interpolation
            )

    def _calculate_total_rows(
        self, table: TableReference, *, cache_enabled: Optional[bool] = None
    ) -> int:
        """Retrieves the total number of rows of a table by issuing a *COUNT(\\*)* query against the live database.

        The table is assumed to be non-virtual.

        Parameters
        ----------
        table : TableReference
            The table to check
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to *None* which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        int
            The total number of rows in the table.
        """
        query_template = "SELECT COUNT(*) FROM {tab}".format(tab=table.full_name)
        return self._db.execute_query(
            query_template,
            cache_enabled=self._determine_caching_behavior(cache_enabled),
        )

    def _calculate_distinct_values(
        self, column: BoundColumnReference, *, cache_enabled: Optional[bool] = None
    ) -> int:
        """Retrieves the number of distinct column values by issuing a *COUNT(\\*)* / *GROUP BY* query over that
        column against the live database.

        The column is assumed to be bound to a (non-virtual) table.

        Parameters
        ----------
        column : ColumnReference
            The column to check
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to *None* which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        int
            The number of distinct values in the column
        """
        query_template = "SELECT COUNT(DISTINCT {col}) FROM {tab}".format(
            col=column.name, tab=column.table.full_name
        )
        return self._db.execute_query(
            query_template,
            cache_enabled=self._determine_caching_behavior(cache_enabled),
        )

    def _calculate_min_max_values(
        self, column: BoundColumnReference, *, cache_enabled: Optional[bool] = None
    ) -> tuple[Any, Any]:
        """Retrieves the minimum/maximum values in a column by issuing an aggregation query for that column against the
        live database.

        The column is assumed to be bound to a (non-virtual) table.

        Parameters
        ----------
        column : ColumnReference
            The column to check
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to *None* which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        tuple[Any, Any]
            A tuple of *(min, max)*
        """
        query_template = "SELECT MIN({col}), MAX({col}) FROM {tab}".format(
            col=column.name, tab=column.table.full_name
        )
        return self._db.execute_query(
            query_template,
            cache_enabled=self._determine_caching_behavior(cache_enabled),
        )

    def _calculate_most_common_values(
        self,
        column: BoundColumnReference,
        k: int | None,
        *,
        cache_enabled: Optional[bool] = None,
    ) -> Sequence[tuple[Any, int]]:
        """Retrieves the `k` most frequent values of a column along with their frequencies by issuing a query over that
        column against the live database.

        The actual query combines a *COUNT(\\*)* aggregation, with a grouping over the column values, followed by a
        count-based ordering and limit.

        The column is assumed to be bound to a (non-virtual) table.

        Parameters
        ----------
        column : ColumnReference
            The column to check
        k : Optional[int]
            The number of most frequent values to retrieve. If less values are available (because there are not as much
            distinct values in the column), the frequencies of all values is returned. If *None*, all values are returned.
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to *None* which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        Sequence[tuple[Any, int]]
            The most common values in *(value, frequency)* pairs, ordered by largest frequency first. Can be smaller
            than the requested `k` value if the column contains less distinct values.
        """
        if k is None or k <= 0:
            query_template = """
                SELECT {col}, COUNT(*) AS n
                FROM {tab}
                GROUP BY {col}
                ORDER BY n DESC, {col}""".format(
                col=column.name, tab=column.table.full_name
            )
        else:
            query_template = textwrap.dedent(
                """
                SELECT {col}, COUNT(*) AS n
                FROM {tab}
                GROUP BY {col}
                ORDER BY n DESC, {col}
                LIMIT {k}""".format(col=column.name, tab=column.table.full_name, k=k)
            )

        return self._db.execute_query(
            query_template,
            cache_enabled=self._determine_caching_behavior(cache_enabled),
            raw=True,
        )

    def _calculate_histogram(
        self,
        column: BoundColumnReference,
        n_bins: int,
        *,
        interpolation: HistogramApproximation,
        cache_enabled: Optional[bool] = None,
    ) -> Optional[Histogram]:
        n_rows = self._calculate_total_rows(column.table)
        if n_rows == 0:
            return None

        query_template = """
            SELECT {col}, COUNT(*) AS n
            FROM {tab}
            GROUP BY {col}
            ORDER BY {col}
            """.format(col=column.name, tab=column.table.full_name)
        result_set = self._db.execute_query(
            query_template,
            cache_enabled=self._determine_caching_behavior(cache_enabled),
            raw=True,
        )
        assert result_set

        lo, bounds, buckets = _infer_histogram_bounds(
            result_set, n_bins=n_bins, n_rows=n_rows
        )
        return Histogram(
            bounds,
            buckets,
            lower=lo,
            bucket_interpolation=interpolation,
        )

    @abc.abstractmethod
    def _retrieve_total_rows_from_stats(self, table: TableReference) -> Optional[int]:
        """Queries the DBMS-internal metadata for the number of rows in a table.

        The table is assumed to be non-virtual.

        Parameters
        ----------
        table : TableReference
            The table to check

        Returns
        -------
        Optional[int]
            The total number of rows in the table. If no such statistic exists, but the database system in principle
            maintains the statistic, *None* is returned. For example, this situation can occur if the database system
            only maintains a row count if the table has at least a certain size and the table in question did not reach
            that size yet.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_distinct_values_from_stats(
        self, column: BoundColumnReference
    ) -> Optional[int]:
        """Queries the DBMS-internal metadata for the number of distinct values of the column.

        The column is assumed to be bound to a (non-virtual) table.

        Parameters
        ----------
        column : ColumnReference
            The column to check

        Returns
        -------
        Optional[int]
            The number of distinct values in the column. If no such statistic exists, but the database system in
            principle maintains the statistic, *None* is returned. For example, this situation can occur if the
            database system only maintains a distinct value count if the column values are distributed in a
            sufficiently diverse way.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_min_max_values_from_stats(
        self, column: BoundColumnReference
    ) -> Optional[tuple[Any, Any]]:
        """Queries the DBMS-internal metadata for the minimum / maximum value in a column.

        The column is assumed to be bound to a (non-virtual) table.

        Parameters
        ----------
        column : ColumnReference
            The column to check

        Returns
        -------
        Optional[tuple[Any, Any]]
            A tuple of minimum and maximum value. If no such statistic exists, but the database system in principle
            maintains the statistic, *None* is returned. For example, this situation can occur if thec database
            system only maintains the min/max value if they are sufficiently far apart.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_most_common_values_from_stats(
        self, column: BoundColumnReference, k: int | None
    ) -> Sequence[tuple[Any, int]]:
        """Queries the DBMS-internal metadata for the `k` most common values of the `column`.

        The column is assumed to be bound to a (non-virtual) table.

        Parameters
        ----------
        column : ColumnReference
            The column to check
        k : int, optional
            The maximum number of most common values to return. If there are less values available, all
            of the available values will be returned. If `None`, all available values will be returned.

        Returns
        -------
        Sequence[tuple[Any, int]]
            The most common values in pairs of (value, frequency), starting with the highest frequency. Notice that
            this sequence can be empty if no values are available. This can happen if the database system in principle
            maintains this statistic but does considers the value distribution to uniform to make the maintenance
            worthwhile. Likewise, if less common values exist than the requested `k` value, only the available values
            will be returned (and the sequence will be shorter than `k` in that case).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_histogram_from_stats(
        self, column: BoundColumnReference, *, interpolation: HistogramApproximation
    ) -> Optional[Histogram]:
        raise NotImplementedError

    def _determine_caching_behavior(
        self, local_cache_enabled: Optional[bool]
    ) -> Optional[bool]:
        """Utility to quickly figure out which caching behavior to use.

        This method is intended to be called by the top-level methods that provide statistics and enable a selective
        caching which overwrites the caching behavior of the statistics interface.

        Parameters
        ----------
        local_cache_enabled : Optional[bool]
            The caching setting selected by the callee / user.

        Returns
        -------
        Optional[bool]
            Whether caching should be enabled or the determined by the actual database interface.
        """
        return (
            self.cache_enabled if local_cache_enabled is None else local_cache_enabled
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Database statistics of {self._db}"


class HintWarning(UserWarning):
    """Custom warning category for hinting-related problems."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class HintService(abc.ABC):
    """Provides the necessary tools to generate system-specific query instances based on optimizer decisions.

    Hints are PostBOUNDs way to enforce that decisions made in the optimization pipeline are respected by the native
    query optimizer once the query is executed in an actual database system. The general documentation provides much
    more information about why this is necessary and how PostBOUND approaches query optimization and query generation.

    Each database system has to implement this interface to be usable as part of an optimization pipeline.

    See Also
    --------
    OptimizationPipeline.optimize_query : For a general introduction into the query optimization process
    """

    @abc.abstractmethod
    def generate_hints(
        self,
        query: SqlQuery,
        plan: Optional[QueryPlan] = None,
        *,
        join_order: Optional[JoinTree] = None,
        physical_operators: Optional[PhysicalOperatorAssignment] = None,
        plan_parameters: Optional[PlanParameterization] = None,
    ) -> SqlQuery:
        """Transforms the input query such that the given optimization decisions are respected during query execution.

        In the most common case this involves building a `Hint` clause that encodes the optimization decisions in a
        system-specific way. However, depending on the concrete database system, this might also involve a
        restructuring of certain parts of the query, e.g. the usage of specific join statements, the introduction of
        non-standard SQL statements, or a reordering of the *FROM* clause.

        Notice that all optimization information is optional. If individual parameters are set to *None*, nothing
        has been enforced by PostBOUND's optimization process and the native optimizer of the database system should
        "fill the gaps".

        Implementations of this method are required to adhere to operators for joins and scans as much as possible. However,
        there is no requirement to represent auxiliary nodes (e.g. sorts) if this is not possible or meaningful for the plan.
        As a rule of thumb, implementations should rate the integrity of the plan in the database higher than a perfect
        representation of the input data.

        Parameters
        ----------
        query : SqlQuery
            The query that should be transformed
        plan : Optional[QueryPlan], optional
            The query execution plan. If this is given, all other parameters should be *None*. This essentially
            enforces the given query plan.
        join_order : Optional[JoinTree], optional
            The sequence in which individual joins should be executed.
        physical_operators : Optional[PhysicalOperatorAssignment], optional
            The physical operators that should be used for the query execution. In addition to selecting specific
            operators for specific joins or scans, this can also include disabling certain operators for the entire
            query.
        plan_parameters : Optional[PlanParameterization], optional
            Additional parameters and metadata for the native optimizer of the database system. Probably the most
            important use-case of these parameters is the supply of cardinality estimates for different joins and
            scans. For example, these can be combined with a join order to influence the physical operators that the
            native optimizer chooses. Another scenario is to only supply such cardinality estimates and leave the
            `join_order` and `physical_operators` completely empty, which essentially simulates a different cardinality
            estimation algorithm for the query. Notice however, that in this scenario cardinality estimates for all
            possible intermediate results of the query have to be supplied. Otherwise, the native optimizer once
            again "fills the gaps" and uses its own estimates for the remaining intermediate results that it explores
            during plan enumeration. This would probably effectively break the estimation algorithm.

        Returns
        -------
        SqlQuery
            The transformed query. It contains all necessary information to enforce the optimization decisions as best
            as possible. Notice that whether the native optimizer of the database system is obliged to respect the
            optimization decisions depends on the specific system. For example, for MySQL hints are really just hints
            and the optimizer is only encouraged to use specific operators but not forced to do so.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def format_query(self, query: SqlQuery) -> str:
        """Transforms the query into a database-specific string, mostly to incorporate deviations from standard SQL.

        This method is necessary because the query abstraction layer is focused on modelling and unifying different
        parts of an SQL query. However, some database systems (cough .. MySQL .. cough) deviate from standard SQL
        syntax and express different parts of a query different. The most prominent example are older versions of
        MySQL that used double quotes for string values rather than the SQL standard single quotes. Therefore, the
        `format_query` method takes an abstract representation of an SQL query as input and turns it into a string
        representation that accounts for all such deviations.

        Parameters
        ----------
        query : SqlQuery
            The query that should be adapted for the database system

        Returns
        -------
        str
            An equivalent notation of the query that incorporates system-specific deviations from standard SQL.
            Notice that this query possibly can no longer be parsed by the query abstraction layer. It is a one-way
            process.

        See Also
        --------
        postbound.qal : the query abstraction layer provided by PostBOUND
        """
        raise NotImplementedError

    @abc.abstractmethod
    def supports_hint(self, hint: PhysicalOperator | HintType) -> bool:
        """Checks, whether the database system is capable of using the specified hint or operator

        Parameters
        ----------
        hint : PhysicalOperator | HintType
            The hint/feature to check

        Returns
        -------
        bool
            Indicates whether the feature is supported by the specific database system.
        """
        raise NotImplementedError


class OptimizerInterface(abc.ABC):
    """Provides high-level access to internal optimizer-related data for the database system.

    Each funtionality is available through a dedicated method. Notice that not all database systems necessarily
    support all of this functions.
    """

    @abc.abstractmethod
    def query_plan(self, query: SqlQuery | str) -> QueryPlan:
        """Obtains the query execution plan for a specific query.

        This respects all hints that potentially influence the optimization process.

        Parameters
        ----------
        query : SqlQuery | str
            The input query

        Returns
        -------
        QueryPlan
            The corresponding execution plan. This will never be an *ANALYZE* plan, but contain as much meaningful
            information as can be derived for the specific database system (e.g. regarding cardinality and cost
            estimates)
        """
        raise NotImplementedError

    def explain(self, query: SqlQuery | str) -> QueryPlan:
        """Alias for `query_plan`."""
        return self.query_plan(query)

    @abc.abstractmethod
    def analyze_plan(self, query: SqlQuery) -> QueryPlan:
        """Executes a specific query and provides the query execution plan supplemented with runtime information.

        This respects all hints that potentially influence the optimization process.

        Parameters
        ----------
        query : SqlQuery
            The input query

        Returns
        -------
        QueryPlan
            The corresponding execution plan. This plan will be an *ANALYZE* plan and contain all information that
            can be derived for the specific database system (e.g. cardinality estimates as well as true cardinality
            counts)
        """
        raise NotImplementedError

    def explain_analyze(self, query: SqlQuery) -> QueryPlan:
        """Alias for `analyze_plan`."""
        return self.analyze_plan(query)

    @abc.abstractmethod
    def parse_plan(self, plan: Any, *, query: Optional[SqlQuery] = None) -> QueryPlan:
        """Transforms the system-specific EXPLAIN output into a standardized `QueryPlan`.

        The optional `query` can be used to provide additional context for the plan. This can be used by
        some database systems for correct parsing (e.g. DuckDB), while others might not use it at all.
        Omitting the `query` parameter is acceptable.

        This method is mainly inteded to parse a previously generated plan, or when the plan could not simply
        be obtained via `query_plan` or `analyze_plan`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cardinality_estimate(self, query: SqlQuery | str) -> Cardinality:
        """Queries the DBMS query optimizer for its cardinality estimate, instead of executing the query.

        The cardinality estimate will correspond to the estimate for the final node. Therefore, running this method
        with aggregate queries is not particularly meaningful.

        Parameters
        ----------
        query : SqlQuery | str
            The input query

        Returns
        -------
        Cardinality
            The cardinality estimate of the native optimizer for the database system.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cost_estimate(self, query: SqlQuery | str) -> Cost:
        """Queries the DBMS query optimizer for the estimated cost of executing the query.

        The cost estimate will correspond to the estimate for the final node. Typically, this cost includes the cost
        of all sub-operators as well.

        Parameters
        ----------
        query : SqlQuery | str
            The input query

        Returns
        -------
        Cost
            The cost estimate of the native optimizer for the database system.
        """
        raise NotImplementedError


_DB_POOL: DatabasePool | None = None
"""Private variable that captures the current singleton instance of the `DatabasePool`."""


class DatabasePool:
    """The database pool allows different parts of the code base to easily obtain access to a database.

    This is achieved by maintaining one global pool of database connections which is shared by the entire system.
    New database instances can be registered and retrieved via unique keys. As long as there is just a single database
    instance, it can be accessed via the `current_database` method.

    The database pool implementation follows the singleton pattern. Use the static `get_instance` method to retrieve
    the database pool instance. All other functionality is provided based on that pool instance.

    References
    ----------

    .. Singleton pattern: https://en.wikipedia.org/wiki/Singleton_pattern
    """

    @staticmethod
    def get_instance() -> DatabasePool:
        """Provides access to the singleton database pool, creating a new pool instance if necessary.

        Returns
        -------
        DatabasePool
            The current pool instance
        """
        global _DB_POOL
        if _DB_POOL is None:
            _DB_POOL = DatabasePool()
        return _DB_POOL

    def __init__(self):
        self._pool: dict[str, Database] = {}

    def current_database(self) -> Database:
        """Provides the database that is currently stored in the pool, provided there is just one.

        Returns
        -------
        Database
            The only database in the pool

        Raises
        ------
        ValueError
            If there are multiple database instances registered in the pool
        """
        if self.n_databases() != 1:
            raise ValueError(
                f"Expected exactly one database in the pool, but found {self.n_databases()}."
            )
        return util.dicts.value(self._pool)

    def any_database(self) -> Database:
        """Provides any database that is currently stored in the pool, provided there is at least one."""
        if self.empty():
            raise ValueError("No database instance registered in the pool.")
        return next(iter(self._pool.values()))

    def register_database(self, key: str, db: Database) -> None:
        """Stores a new database in the pool.

        This method is typically called by the connect methods of the respective database system implementations.

        Parameters
        ----------
        key : str
            A unique identifier under which the database can be retrieved
        db : Database
            The database to store
        """
        self._pool[key] = db

    def retrieve_database(self, key: str) -> Database:
        """Provides the database that is registered under a specific key.

        Parameters
        ----------
        key : str
            The key that was previously used to register the database

        Returns
        -------
        Database
            The corresponding database

        Raises
        ------
        KeyError
            If no database was registered under the given key.
        """
        return self._pool[key]

    def empty(self) -> bool:
        """Checks, whether the database pool is currently emtpy (i.e. no database are registered).

        Returns
        -------
        bool
            *True* if the pool is empty.
        """
        return len(self._pool) == 0

    def n_databases(self) -> int:
        """Returns the number of databases currently registered in the pool."""
        return len(self._pool)

    def clear(self) -> None:
        """Removes all currently registered databases from the pool."""
        self._pool.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._pool

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"DatabasePool {self._pool}"


def current_database() -> Database:
    """Provides the current database from the `DatabasePool`.

    Returns
    -------
    Database
        The current database instance. If there is not exactly one database in the pool, a `ValueError` is raised.

    See Also
    --------
    DatabasePool.current_database
    """
    return DatabasePool.get_instance().current_database()


class UnsupportedDatabaseFeatureError(RuntimeError):
    """Indicates that some requested feature is not supported by the database.

    For example, PostgreSQL (at least up to version 15) does not capture minimum or maximum column values in its
    system statistics. Therefore, forcing the DBS to retrieve such information from its metadata could result in this
    error.

    Parameters
    ----------
    database : Database
        The database that was requested to provide the problematic feature
    feature : str
        A textual description for the requested feature
    """

    def __init__(self, database: Database, feature: str | PhysicalOperator) -> None:
        super().__init__(
            f"Database {database.system_name} does not support feature {feature}"
        )
        self.database = database
        self.feature = feature


class DatabaseServerError(RuntimeError):
    """Indicates an error caused by the database server occured while executing a database operation.

    The error was **not** due to a mistake in the user input (such as an SQL syntax error or access privilege
    violation), but an implementation issue instead (such as out of memory during query execution).

    Parameters
    ----------
    message : str, optional
        A textual description of the error, e.g. *out of memory*. Can be left empty by default.
    context : Optional[object], optional
        Additional context information for when the error occurred, e.g. the query that caused the error. Mainly
        intended for debugging purposes.
    """

    def __init__(self, message: str = "", context: Optional[object] = None) -> None:
        super().__init__(message)
        self.ctx = context


class DatabaseUserError(RuntimeError):
    """Indicates that a database operation failed due to an error on the user's end.

    The error could be due to an SQL syntax error, access privilege violation, etc.

    Parameters
    ----------
    message : str, optional
        A textual description of the error, e.g. *no such table*. Can be left empty by default.
    context : Optional[object], optional
        Additional context information for when the error occurred, e.g. the query that caused the error. Mainly
        intended for debugging purposes.
    """

    def __init__(self, message: str = "", context: Optional[object] = None) -> None:
        super().__init__(message)
        self.ctx = context
