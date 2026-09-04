from __future__ import annotations

import bisect
import configparser
import datetime
import functools
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import textwrap
import time
import tomllib
import warnings
from collections import UserString
from collections.abc import Iterable, Sized
from dataclasses import dataclass
from multiprocessing import connection as mp_conn
from pathlib import Path
from typing import Any, Literal, TextIO, overload

import psycopg
import psycopg.rows
import psycopg.types.datetime as psycopg_datetime

from .. import db, qal, transform, util
from .._core import (
    BoundColumnReference,
    Cardinality,
    ColumnReference,
    IntermediateOperator,
    JoinOperator,
    PhysicalOperator,
    ScanOperator,
    TableReference,
    UnboundColumnError,
    VirtualTableError,
)
from .._hints import (
    HintType,
    JoinTree,
    PhysicalOperatorAssignment,
    PlanParameterization,
    jointree_from_plan,
    operators_from_plan,
    parameters_from_plan,
)
from .._qep import QueryPlan
from ..db import (
    Database,
    DatabasePool,
    DatabaseSchema,
    DatabaseServerError,
    DatabaseStatistics,
    DatabaseUserError,
    HintService,
    HintWarning,
    Histogram,
    HistogramApproximation,
    MostCommonValues,
    OptimizerInterface,
    PreciseStatistics,
    QueryCacheWarning,
    ResultSet,
    UnsupportedDatabaseFeatureError,
    simplify_result_set,
)
from ..qal import (
    Hint,
    SqlQuery,
)
from ..util import StateError, Version, jsondict
from ._config import (
    PostgresConfiguration,
    PostgresSetting,
    RuntimeChangeablePostgresSettings,
    SignificantPostgresSettings,
)
from ._explain import PostgresExplainPlan


class PostgresConfigInterface:
    """A thin wrapper that provides read-only access to Postgres configuration settings using __getitem__ syntax."""

    def __init__(self, pg_instance: PostgresInterface) -> None:
        self._pg = pg_instance

    def __getitem__(self, key: str) -> Any:
        return self._pg.execute_query(f"SHOW {key};")


_PGVersionPattern = re.compile(r"^PostgreSQL (?P<pg_ver>[\d]+(\.[\d]+)?).*$")
"""Regular expression to extract the Postgres server version from the *VERSION()* function.

References
----------

.. Pattern debugging: https://regex101.com/r/UTQkfa/1
"""


#
# Psycopg struggles with infinity date/time values.
# These are valid Postgres values, e.g. SELECT date 'infinity'; is a completely valid PG query. However, they cannot be
# converted to a Python object because Python has no concept of infinity in this context. What is even worse, Postgres
# allows for dates that are larger than the largest date that can be represented in Python. As of Python 3.14 this is
# any date starting on 10000-01-01 or later. For Postgres this is once again completely valid. To mitigate these issues across
# the different date/time types, we introduce a number of custom loaders that handle these values appropriately.
# Likewise, we also introduce some dumpers to send the correct value back to the Postgres session.
#


class _PsycopgDateDumper(psycopg_datetime.DateDumper):
    def dump(self, obj):
        if obj == datetime.date.max:
            return b"infinity"
        elif obj == datetime.date.min:
            return b"-infinity"

        return super().dump(obj)


class _PsycopgDateLoader(psycopg_datetime.DateLoader):
    def load(self, data):
        if data == b"infinity":
            return datetime.date.max
        elif data == b"-infinity":
            return datetime.date.min

        try:
            return super().load(data)
        except psycopg.errors.DataError as e:
            if not e.args:
                raise e
            msg = e.args[0]
            if not isinstance(msg, str) or not msg.startswith("date too large"):
                raise e
            return datetime.date.max


class _PsycopgTimestampDumper(psycopg_datetime.DatetimeNoTzDumper):
    def dump(self, obj):
        if obj == datetime.datetime.max:
            return b"infinity"
        elif obj == datetime.datetime.min:
            return b"-infinity"

        return super().dump(obj)


class _PsycopgTimestampLoader(psycopg_datetime.TimestampLoader):
    def load(self, data):
        if data == b"infinity":
            return datetime.datetime.max
        elif data == b"-infinity":
            return datetime.datetime.min

        try:
            return super().load(data)
        except psycopg.errors.DataError as e:
            if not e.args:
                raise e
            msg = e.args[0]
            if not isinstance(msg, str) or not msg.startswith("timestamp too large"):
                raise e
            return datetime.datetime.max


class _PsycopgTimestampTzDumper(psycopg_datetime.DatetimeDumper):
    def dump(self, obj):
        if obj == datetime.datetime.max:
            return b"infinity"
        elif obj == datetime.datetime.min:
            return b"-infinity"

        return super().dump(obj)


class _PsycopgTimestampTzLoader(psycopg_datetime.TimestamptzLoader):
    def load(self, data):
        if data == b"infinity":
            return datetime.datetime.max.replace(tzinfo=datetime.UTC)
        elif data == b"-infinity":
            return datetime.datetime.min.replace(tzinfo=datetime.UTC)

        try:
            return super().load(data)
        except psycopg.errors.DataError as e:
            if not e.args:
                raise e
            msg = e.args[0]
            if not isinstance(msg, str) or not msg.startswith("timestamp too large"):
                raise e
            return datetime.datetime.max.replace(tzinfo=datetime.UTC)


class _PsycopgIntervalLoader(psycopg_datetime.IntervalLoader):
    def load(self, data):
        if data == b"infinity":
            return datetime.timedelta.max
        elif data == b"-infinity":
            return datetime.timedelta.min

        # Python and Postgres can represent the same range of intervals/timedeltas
        return super().load(data)


def _apply_preparatory_statements(query: SqlQuery, *, cur: psycopg.Cursor) -> SqlQuery:
    if not query.hints or not query.hints.preparatory_statements:
        return query

    cur.execute(query.hints.preparatory_statements)  # type: ignore
    return transform.drop_hints(query, preparatory_statements_only=True)


class PostgresInterface(Database):
    """Database implementation for PostgreSQL backends.

    The `config` attribute provides read-only access to the current GUC values of the server.

    Parameters
    ----------
    connect_string : str
        Connection string for `psycopg` to establish a connection to the Postgres server
    system_name : str, optional
        Description of the specific Postgres server, by default *Postgres*
    application_name : str, optional
        Identifier for the Postgres server. This will be the name that is shown in the server logs and process lists.
    client_encoding : str, optional
        The client encoding to use for the connection, by default *UTF8*
    cache_enabled : bool, optional
        Whether to enable caching of database queries, by default *False*
    debug : bool, optional
        Whether additional debug information should be printed during database interaction. Defaults to *False*.
    """

    def __init__(
        self,
        connect_string: str,
        system_name: str = "Postgres",
        *,
        application_name: str = "PostBOUND",
        client_encoding: str = "UTF8",
        debug: bool = False,
    ) -> None:
        self.connect_string = connect_string
        self.debug = debug
        self.config = PostgresConfigInterface(self)
        self._application_name = application_name or "PostBOUND"
        self._client_encoding = client_encoding
        self._init_connection()

        self._db_stats = PostgresStatisticsInterface(self)
        self._db_schema = PostgresSchemaInterface(self)
        self._hinting_backend = PostgresHintService(self)

        self._timeout_executor = _TimeoutQueryExecutor(self)
        self._last_query_runtime = math.nan

        super().__init__(system_name)

    def schema(self) -> PostgresSchemaInterface:
        return self._db_schema

    def statistics(self) -> PostgresStatisticsInterface:
        return self._db_stats

    def hinting(self) -> PostgresHintService:
        return self._hinting_backend

    def execute_query(
        self,
        query: SqlQuery | str,
        *,
        plan: QueryPlan | None = None,
        join_order: JoinTree | None = None,
        physical_operators: PhysicalOperatorAssignment | None = None,
        plan_parameters: PlanParameterization | None = None,
        raw: bool = False,
        timeout: float | None = None,
    ) -> Any:
        # NB: some of the execution logic is duplicated in TimeoutQueryExecutor.execute_query.
        # Make sure to keep both implementations in sync.
        query = self._apply_query_hints(
            query,
            plan,
            join_order=join_order,
            physical_operators=physical_operators,
            plan_parameters=plan_parameters,
        )

        if timeout is not None and timeout > 0:
            return self._timeout_executor.execute_query(query, timeout=timeout, raw=raw)

        if isinstance(query, UserString):
            query = str(query)
        elif isinstance(query, SqlQuery):
            # If there are any preparatory statements we must execute them now. If we would simply include them in the query,
            # psycopg would use the first statement to determine the result type of the entire query. If this were a SET
            # statement, no results could be fetched. _apply_preparatory_statements takes care of these and gives a cleaned-up
            # query to execute.
            query = _apply_preparatory_statements(query, cur=self._cursor)
            query = self._hinting_backend.format_query(query)

        try:
            start_time = time.perf_counter_ns()
            self._cursor.execute(query)  # type: ignore
            end_time = time.perf_counter_ns()
            self._last_query_runtime = (end_time - start_time) / 10**9  # convert to seconds

            if self._cursor.rownumber is None:
                # For statements that do not return a result (e.g. SET),
                # rownumber is None. We can use this as an indicator whether
                # fetching results is possible.
                return None

            query_result = self._cursor.fetchall()
        except (psycopg.InternalError, psycopg.OperationalError) as e:
            msg = "\n".join(
                [
                    f"At {util.timestamp()}",
                    "For query:",
                    str(query),
                    "Message:",
                    str(e),
                ]
            )
            raise DatabaseServerError(msg, e) from e
        except psycopg.Error as e:
            msg = "\n".join(
                [
                    f"At {util.timestamp()}",
                    "For query:",
                    str(query),
                    "Message:",
                    str(e),
                ]
            )
            raise DatabaseUserError(msg, e) from e

        return query_result if raw else simplify_result_set(query_result)

    def execute_with_timeout(self, query: SqlQuery | str, timeout: float = 60.0) -> ResultSet | None:
        try:
            result = self.execute_query(query, timeout=timeout, raw=True)
            return result
        except TimeoutError:
            return None

    def last_query_runtime(self) -> float:
        return self._last_query_runtime

    def time_query(self, query: SqlQuery, *, timeout: float | None = None) -> float:
        self.execute_query(query, raw=True, timeout=timeout)
        return self.last_query_runtime()

    def optimizer(self) -> PostgresOptimizer:
        return PostgresOptimizer(self)

    def database_name(self) -> str:
        self._cursor.execute("SELECT CURRENT_DATABASE();")
        db_name = self._cursor.fetchone()[0]  # type: ignore
        return db_name

    def dbms_version(self) -> Version:
        self._cursor.execute("SELECT VERSION();")
        version_string = self._cursor.fetchone()[0]  # type: ignore
        version_match = _PGVersionPattern.match(version_string)
        if not version_match:
            raise RuntimeError(f"Could not extract Postgres version from string '{version_string}'")
        pg_ver = version_match.group("pg_ver")
        return Version(pg_ver)

    def backend_pid(self) -> int:
        """Provides the backend process ID of the current connection.

        Returns
        -------
        int
            The process ID
        """
        return self._connection.info.backend_pid

    def data_dir(self) -> Path:
        """Get the data directory of the Postgres server.

        Returns
        -------
        Path
            The data directory path
        """
        self._cursor.execute("SHOW data_directory;")
        data_dir = self._cursor.fetchone()[0]  # type: ignore
        return Path(data_dir)

    def logfile(self) -> Path | None:
        """Get the log file of the (local) Postgres server."""
        proc_path = Path(f"/proc/{self.backend_pid()}/fd/1")
        if not proc_path.exists() or not proc_path.is_symlink():
            return None
        return proc_path.resolve()

    def describe(self) -> jsondict:
        base_info: dict[str, Any] = {
            "system_name": self.dbms_name(),
            "system_version": self.dbms_version(),
            "database": self.database_name(),
            "hinting_mode": self._hinting_backend.describe(),
            "statistics": self.statistics().describe(),
        }
        self._cursor.execute("SELECT name, setting FROM pg_settings")
        system_settings = self._cursor.fetchall()
        base_info["system_settings"] = {
            setting: value for setting, value in system_settings if setting in SignificantPostgresSettings
        }

        schema_info: list[jsondict] = []
        for table in self._db_schema.tables():
            if table.full_name.startswith("pg_"):
                continue  # skip system tables

            column_info: list[jsondict] = []

            for column in self._db_schema.columns(table):
                column_info.append(
                    {
                        "column": str(column),
                        "indexed": self.schema().has_index(column),
                        "foreign_keys": self._db_schema.foreign_keys_on(column),
                    }
                )

            pk_col = self._db_schema.primary_key_column(table)
            schema_info.append(
                {
                    "table": str(table),
                    "n_rows": self.statistics().total_rows(table),
                    "columns": column_info,
                    "primary_key": pk_col.name if pk_col else None,
                }
            )

        base_info["schema_info"] = schema_info
        return base_info

    def reset_connection(self) -> int:
        try:
            self._timeout_executor.close()
            self._connection.cancel()
            self._cursor.close()
            self._connection.close()
        except psycopg.Error:
            pass
        return self._init_connection()

    def cursor(self) -> psycopg.Cursor:  # type: ignore - see comment below
        # The psycopg client is not entirely DB API 2.0 compatible because it violates (for the most part) unimportant
        # details of the execute() method
        return self._cursor

    def connection(self) -> psycopg.Connection:
        """Provides the current database connection.

        Returns
        -------
        psycopg.Connection
            The connection
        """
        return self._connection

    def rollback_tx(self) -> None:
        """Perform a ROLLBACK on the current transaction (connection).

        This should be executed if any errors occurred while running a query.
        """
        self._connection.rollback()

    def obtain_new_local_connection(self) -> psycopg.Connection:
        """Provides a new database connection to be used exclusively be the client.

        The current connection maintained by the `PostgresInterface` is not affected by obtaining a new connection in any
        way.

        Returns
        -------
        psycopg.Connection
            The connection
        """
        return psycopg.connect(self.connect_string)

    def close(self) -> None:
        self._cursor.close()
        self._connection.close()
        self._timeout_executor.close()

    def prewarm_tables(
        self,
        tables: TableReference | Iterable[TableReference] | None = None,
        *more_tables: TableReference,
        exclude_table_pages: bool = False,
        include_primary_index: bool = True,
        include_secondary_indexes: bool = True,
    ) -> None:
        """Prepares the Postgres buffer pool with tuples from specific tables.

        Parameters
        ----------
        tables : Optional[TableReference | Iterable[TableReference]], optional
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
        pool are not specified. Since all prewarming tasks happen sequentially, the first prewarmed relations will typically
        be evicted and only the last relations (tables or indexes) are retained in the shared buffer. The precise order in
        which the prewarming tasks are executed is not specified and depends on the actual relations.

        Examples
        --------
        >>> pg.prewarm_tables([table1, table2])
        >>> pg.prewarm_tables(table1, table2)
        >>> pg.prewarm_tables(query.tables())
        """
        self._assert_active_extension("pg_prewarm")

        match tables:
            case None:
                tables = []
            case TableReference():
                tables = [tables]
            case _:
                tables = list(tables)
        tables += list(more_tables)

        if not tables:
            return

        selected_tables: set[str] = set(
            tab.full_name for tab in tables if not tab.virtual
        )  # eliminate duplicates if tables are selected multiple times

        table_indexes = (
            [self._fetch_index_relnames(tab) for tab in selected_tables]
            if include_primary_index or include_secondary_indexes
            else []
        )
        indexes_to_prewarm = {
            idx
            for idx, primary in util.flatten(table_indexes)
            if (primary and include_primary_index) or (not primary and include_secondary_indexes)
        }

        rels_to_prewarm = indexes_to_prewarm if exclude_table_pages else (selected_tables | indexes_to_prewarm)
        if not rels_to_prewarm:
            return

        prewarm_invocations = [f"pg_prewarm('{rel}')" for rel in rels_to_prewarm]
        prewarm_text = ", ".join(prewarm_invocations)
        prewarm_query = f"SELECT {prewarm_text}"

        self._cursor.execute(prewarm_query)  # type: ignore

    def cooldown_tables(
        self,
        tables: TableReference | Iterable[TableReference] | None = None,
        *more_tables: TableReference,
        exclude_table_pages: bool = False,
        include_primary_index: bool = True,
        include_secondary_indexes: bool = True,
    ) -> None:
        """Removes tuples from specific tables from  the Postgres buffer pool.

        This method can be used to simulate a cold start for the next incoming query. It requires the *pg_temperature*
        extension that is part of the pg_lab project.

        Parameters
        ----------
        tables : Optional[TableReference  |  Iterable[TableReference]], optional
            The tables that should be removed from the buffer pool
        *more_tables : TableReference
            More tables that should be removed into the buffer pool, enabling a more convenient usage of this method.
            See examples for details on the usage.
        exclude_table_pages : bool, optional
            Whether the table data (i.e. pages containing the actual tuples) should *not* be removed. This is off by default,
            meaning that the cooldown is applied to the data pages. This can be toggled on to only cooldown index pages (see
            `include_primary_index` and `include_secondary_index`).
        include_primary_index : bool, optional
            Whether the pages of the primary key index should also be cooled down. Enabled by default.
        include_secondary_indexes : bool, optional
            Whether the pages for secondary indexes should also be cooled down. Enabled by default.

        Examples
        --------
        >>> pg.cooldown_tables([table1, table2])
        >>> pg.cooldown_tables(table1, table2)
        >>> pg.cooldown_tables(query.tables())

        References
        ----------
        pg_lab : https://github.com/rbergm/pg_lab
        """
        self._assert_active_extension("pg_temperature")

        match tables:
            case None:
                tables = []
            case TableReference():
                tables = [tables]
            case _:
                tables = list(tables)
        tables += list(more_tables)

        if not tables:
            return

        selected_tables: set[str] = set(
            tab.full_name for tab in tables if not tab.virtual
        )  # eliminate duplicates if tables are selected multiple times

        table_indexes = (
            [self._fetch_index_relnames(tab) for tab in selected_tables]
            if include_primary_index or include_secondary_indexes
            else []
        )
        indexes_to_cooldown = {
            idx
            for idx, primary in util.flatten(table_indexes)
            if (primary and include_primary_index) or (not primary and include_secondary_indexes)
        }

        rels_to_cooldown = indexes_to_cooldown if exclude_table_pages else (selected_tables | indexes_to_cooldown)
        if not rels_to_cooldown:
            return

        cooldown_invocations = [f"pg_cooldown('{tab}')" for tab in rels_to_cooldown]
        cooldown_text = ", ".join(cooldown_invocations)
        cooldown_query = f"SELECT {cooldown_text}"

        self._cursor.execute(cooldown_query)  # type: ignore

    def current_configuration(self, *, runtime_changeable_only: bool = False) -> PostgresConfiguration:
        """Provides all current configuration settings in the current Postgres connection.

        Parameters
        ----------
        runtime_changeable_only : bool, optional
            Whether only such settings that can be changed at runtime should be provided. Defaults to *False*.

        Returns
        -------
        PostgresConfiguration
            The current configuration.
        """
        self._cursor.execute("SELECT name, setting FROM pg_settings")
        system_settings = self._cursor.fetchall()
        allowed_settings = RuntimeChangeablePostgresSettings if runtime_changeable_only else SignificantPostgresSettings
        configuration = {setting: value for setting, value in system_settings if setting in allowed_settings}
        return PostgresConfiguration.load(**configuration)

    def apply_configuration(self, configuration: PostgresConfiguration | PostgresSetting | str) -> None:
        """Changes specific configuration parameters of the Postgres server or current connection.

        Parameters
        ----------
        configuration : PostgresConfiguration | PostgresSetting | str
            The desired setting values. If a string is supplied directly, it already has to be a valid setting update such as
            *SET geqo = FALSE;*.
        """
        if (
            isinstance(configuration, PostgresSetting)
            and configuration.parameter not in RuntimeChangeablePostgresSettings
        ):
            warnings.warn(
                f"Cannot apply configuration setting '{configuration.parameter}' at runtime",
                stacklevel=2,
            )
            return
        elif isinstance(configuration, PostgresConfiguration):
            supported_settings: list[PostgresSetting] = []
            unsupported_settings: list[str] = []
            for setting in configuration.settings:
                if setting.parameter in RuntimeChangeablePostgresSettings:
                    supported_settings.append(setting)
                else:
                    unsupported_settings.append(setting.parameter)
            if unsupported_settings:
                warnings.warn(
                    f"Skipping configuration settings {unsupported_settings} because they cannot be changed at runtime",
                    stacklevel=2,
                )
            configuration = str(PostgresConfiguration(supported_settings))

        self._cursor.execute(configuration)  # type: ignore

    def has_extension(self, extension_name: str, *, is_shared_object: bool = True) -> bool:
        """Checks, whether the current Postgres database has a specific extension loaded and active.

        Extensions can be either created using the *CREATE EXTENSION* command, or by loading the shared object via *LOAD*.
        For the shared object-based check to work correctly, the Postgres server has to run in the same namespace as the
        PostBOUND client.

        Parameters
        ----------
        extension_name : str
            The name of the extension to be checked. In case of shared objects, this should be equivalent to the name of said
            object. In this case, the suffix is optional.
        is_shared_object : bool, optional
            Whether the extension is a shared object that is loaded into the Postgres server. By default this is set to *True*,
            which assumes that the extension is loaded as a shared object, rather than as a default extension.


        Returns
        -------
        bool
            Whether the extension is loaded and active in the current Postgres database.
        """
        match sys.platform:
            case "win32" | "cygwin":
                lib_suffix = ".dll"
            case "darwin":
                lib_suffix = ".dylib"
            case "linux":
                lib_suffix = ".so"
            case _:
                raise RuntimeError(f"Plaform '{sys.platform}' is not supported by extension check.")

        if is_shared_object or extension_name in ("pg_hint_plan", "pg_lab"):
            shared_object_name = (
                f"{extension_name}{lib_suffix}" if not extension_name.endswith(lib_suffix) else extension_name
            )
            loaded_shared_objects = util.system.open_files(self._connection.info.backend_pid)
            return any(so.endswith(shared_object_name) for so in loaded_shared_objects)
        else:
            self._cursor.execute("SELECT extname FROM pg_extension;")
            return any(ext[0] == extension_name for ext in self._cursor.fetchall())

    def _init_connection(self) -> int:
        """Sets all default connection parameters and creates the actual database cursor.

        Returns
        -------
        int
            The backend process ID of the new connection
        """
        self._connection: psycopg.Connection = psycopg.connect(
            self.connect_string,
            application_name=self._application_name,
            client_encoding=self._client_encoding,
            row_factory=psycopg.rows.tuple_row,
        )
        self._connection.autocommit = True  # pg_hint_plan hinting backend currently relies on autocommit!
        self._connection.prepare_threshold = None

        self._connection.adapters.register_dumper(datetime.date, _PsycopgDateDumper)
        self._connection.adapters.register_dumper(datetime.datetime, _PsycopgTimestampDumper)
        self._connection.adapters.register_dumper(datetime.datetime, _PsycopgTimestampTzDumper)
        self._connection.adapters.register_loader("date", _PsycopgDateLoader)
        self._connection.adapters.register_loader("timestamp", _PsycopgTimestampLoader)
        self._connection.adapters.register_loader("timestamptz", _PsycopgTimestampTzLoader)
        self._connection.adapters.register_loader("interval", _PsycopgIntervalLoader)

        self._cursor: psycopg.Cursor = self._connection.cursor()
        return self.backend_pid()

    def _apply_query_hints(
        self,
        query: str | SqlQuery,
        plan: QueryPlan | None,
        *,
        join_order: JoinTree | None,
        physical_operators: PhysicalOperatorAssignment | None,
        plan_parameters: PlanParameterization | None,
    ) -> str | SqlQuery:
        if isinstance(query, str):
            # XXX: should we rather parse the query here?
            return query

        has_hint = any(hint is not None for hint in (plan, join_order, physical_operators, plan_parameters))
        if not has_hint:
            return query

        return self.hinting().generate_hints(
            query,
            plan,
            join_order=join_order,
            physical_operators=physical_operators,
            plan_parameters=plan_parameters,
        )

    def _fetch_index_relnames(self, table: TableReference | str) -> Iterable[tuple[str, bool]]:
        """Loads all physical index relations for a physical table.

        Parameters
        ----------
        table : TableReference
            The table for which to load the indexes

        Returns
        -------
        Iterable[tuple[str, bool]]
            All indexes as pairs *(relation name, primary)*. Relation name corresponds to the table-like object that Postgres
            created internally to store the index (e.g. for a table called *title*, this is typically called *title_pkey* for
            the primary key index). The *primary* boolean indicates whether this is the primary key index of the table.
        """
        query_template = textwrap.dedent("""
                                         SELECT cls.relname, idx.indisprimary
                                         FROM pg_index idx
                                            JOIN pg_class cls ON idx.indexrelid = cls.oid
                                            JOIN pg_class owner_cls ON idx.indrelid = owner_cls.oid
                                         WHERE owner_cls.relname = %s;
                                         """)
        table = table.full_name if isinstance(table, TableReference) else table
        self._cursor.execute(query_template, (table,))  # type: ignore
        return list(self._cursor.fetchall())

    def _assert_active_extension(self, extension_name: str, *, is_shared_object: bool = False) -> None:
        """Raises an error if the current postgres database does not have the desired extension.

        Extensions can be created using the *CREATE EXTENSION* command, or by loading the shared object via *LOAD*. In either
        case, this method can check whether they are indeed active.

        Parameters
        ----------
        extension_name : str
            The name of the extension to be checked.
        is_shared_object : bool, optional
            Whether the extension is activated using *LOAD*. If this it the case, the shared objects owned by the database
            process rather than the internal extension catalogs will be checked. The extension name will be automatically
            suffixed with *.so* if necessary. As a special case, for checking the *pg_hint_plan* extension this parameter does
            not need to be true. This is due to the central importance of that extension for the entire Postgres hinting
            system and saves some typing in that case.

        Raises
        ------
        StateError
            If the given extension is not active
        """
        extension_is_active = self.has_extension(extension_name, is_shared_object=is_shared_object)
        if not extension_is_active:
            raise StateError(f"Extension '{extension_name}' is not active in database '{self.database_name()}'")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.connect_string == other.connect_string

    def __hash__(self) -> int:
        return hash(self.connect_string)


class PostgresSchemaInterface(DatabaseSchema):
    """Database schema implementation for Postgres systems.

    Parameters
    ----------
    postgres_db : PostgresInterface
        The database for which schema information should be retrieved
    """

    def __init__(self, postgres_db: PostgresInterface) -> None:
        super().__init__(postgres_db)
        self._table_cache = False
        self._user_tables: set[TableReference] = set()
        self._system_tables: set[TableReference] = set()

    def tables(
        self, *, catalog: str = "", schema: str = "", include_system_tables: bool = False
    ) -> set[TableReference]:
        if self._table_cache:
            return self._user_tables | self._system_tables if include_system_tables else self._user_tables

        all_tables = super().tables(catalog=catalog, schema=schema, include_system_tables=True)
        self._system_tables = {tab for tab in all_tables if tab.full_name.startswith("pg_")}
        self._user_tables = all_tables - self._system_tables

        return self._user_tables | self._system_tables if include_system_tables else self._user_tables

    def lookup_column(
        self,
        column: ColumnReference | str,
        candidate_tables: Iterable[TableReference],
        *,
        expect_match: bool = False,
    ) -> TableReference | None:
        if not isinstance(candidate_tables, Sized):
            candidate_tables = list(candidate_tables)
        candidate_tables = set(candidate_tables) if len(candidate_tables) > 5 else list(candidate_tables)
        column = column.name if isinstance(column, ColumnReference) else column
        lower_col = column.lower()

        for table in candidate_tables:
            table_columns = self._fetch_columns(table)
            if column in table_columns or lower_col in table_columns:
                return table

        if not expect_match:
            return None
        table_strings = [table.qualified_name() for table in candidate_tables]
        raise ValueError(f"Column '{column}' not found in candidate tables {table_strings}")

    def is_primary_key(self, column: ColumnReference) -> bool:
        if not column.table:
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)
        index_map = self._fetch_indexes(column.table)
        return index_map.get(column.name, False)

    def has_secondary_index(self, column: ColumnReference) -> bool:
        if not column.table:
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)
        index_map = self._fetch_indexes(column.table)

        # The index map contains an entry for each attribute that actually has an index. The value is True, if the
        # attribute (which is known to be indexed), is even the Primary Key
        # Our method should return False in two cases: 1) the attribute is not indexed at all; and 2) the attribute
        # actually is the Primary key. Therefore, by assuming it is the PK in case of absence, we get the correct
        # value.
        return not index_map.get(column.name, True)

    def indexes_on(self, column: ColumnReference) -> set[str]:
        if not column.table:
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)
        schema = column.table.schema or "public"
        query_template = textwrap.dedent("""
            SELECT cls.relname
            FROM pg_index idx
                JOIN pg_class cls ON idx.indexrelid = cls.oid
                JOIN pg_class rel ON idx.indrelid = rel.oid
                JOIN pg_attribute att ON att.attnum = ANY(idx.indkey) AND idx.indrelid = att.attrelid
                JOIN pg_namespace nsp ON cls.relnamespace = nsp.oid AND rel.relnamespace = nsp.oid
            WHERE rel.relname = %s
                AND att.attname = %s
                AND nsp.nspname = %s
        """)

        self._db.cursor().execute(query_template, (column.table.full_name, column.name, schema))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None

        return {row[0] for row in result_set}

    def indexed_column(self, index: str, *, schema: str = "public") -> BoundColumnReference | None:
        """Retrieves the column that is indexed by a specific index.

        Returns
        -------
        Optional[ColumnReference]
            The column or *None*, if the index does not exist (in the given schema). For multi-indexes, i.e. indexes over
            multiple columns, this returns the first column only.
        """
        query_template = textwrap.dedent("""
            SELECT att.attname, rel.relname
            FROM pg_index idx
                JOIN pg_class cls ON idx.indexrelid = cls.oid
                JOIN pg_class rel ON idx.indrelid = rel.oid
                JOIN pg_attribute att ON att.attnum = ANY(idx.indkey) AND idx.indrelid = att.attrelid
                JOIN pg_namespace nsp ON cls.relnamespace = nsp.oid AND rel.relnamespace = nsp.oid
            WHERE cls.relname = %s
                AND nsp.nspname = %s
        """)

        self._db.cursor().execute(query_template, (index, schema))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None

        if not result_set:
            return None
        if len(result_set) > 1:
            warnings.warn(
                f"Multi-index {index} detected. Only returning the first column",
                stacklevel=2,
            )
            result_set = result_set[:1]

        col, tab = result_set[0]
        return ColumnReference.create(col, table=tab)

    def foreign_keys_on(self, column: ColumnReference) -> set[BoundColumnReference]:
        if not column.table:
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)
        schema = column.table.schema or "public"
        query_template = textwrap.dedent("""
            SELECT target.table_name, target.column_name
            FROM information_schema.key_column_usage AS fk_sources
                JOIN information_schema.table_constraints AS all_constraints
                ON fk_sources.constraint_name = all_constraints.constraint_name
                    AND fk_sources.table_schema = all_constraints.table_schema
                JOIN information_schema.constraint_column_usage AS target
                ON fk_sources.constraint_name = target.constraint_name
                    AND fk_sources.table_schema = target.table_schema
            WHERE fk_sources.table_name = %s
                AND fk_sources.column_name = %s
                AND fk_sources.table_schema = %s
                AND all_constraints.constraint_type = 'FOREIGN KEY'
            """)

        self._db.cursor().execute(query_template, (column.table.full_name, column.name, schema))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None

        return {BoundColumnReference(row[1], TableReference(row[0])) for row in result_set}

    def datatype(self, column: ColumnReference, *, raw: bool = False) -> str:
        if not column.table:
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)
        schema = column.table.schema or "public"
        query_template = textwrap.dedent("""
            SELECT data_type FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s AND table_schema = %s""")

        self._db.cursor().execute(query_template, (column.table.full_name, column.name, schema))
        result_set = self._db.cursor().fetchone()
        assert result_set

        dtype: str = result_set[0]
        return dtype if raw else dtype.lower()

    def is_nullable(self, column: ColumnReference) -> bool:
        if not column.table:
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)
        schema = column.table.schema or "public"
        query_tempalte = textwrap.dedent("""
            SELECT is_nullable = 'YES' FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s AND table_schema = %s""")

        self._db.cursor().execute(query_tempalte, (column.table.full_name, column.name, schema))
        result_set = self._db.cursor().fetchone()
        assert result_set

        return result_set[0]

    def _fetch_columns(self, table: TableReference) -> list[str]:
        """Retrieves all physical columns for a given table from the PG metadata catalogs.

        Parameters
        ----------
        table : TableReference
            The table whose columns should be loaded

        Returns
        -------
        list[str]
            The names of all columns

        Raises
        ------
        VirtualTableError
            If the table is a virtual table (e.g. subquery or CTE)
        """
        if table.virtual:
            raise VirtualTableError(table)
        schema = table.schema or "public"
        query_template = (
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s AND table_schema = %s"
        )

        self._db.cursor().execute(query_template, (table.full_name, schema))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None

        return [col[0] for col in result_set]

    def _fetch_indexes(self, table: TableReference) -> dict[str, bool]:
        """Retrieves all index structures for a given table based on the PG metadata catalogs.

        Parameters
        ----------
        table : TableReference
            The table whose indexes should be loaded

        Returns
        -------
        dict
            Contains a key for each column that has an index. The column keys map to booleans that indicate whether
            the corresponding index is a primary key index. Columns without any index do not appear in the dictionary.

        Raises
        ------
        VirtualTableError
            If the table is a virtual table (e.g. subquery or CTE)
        """
        if table.virtual:
            raise VirtualTableError(table)
        # query adapted from https://wiki.postgresql.org/wiki/Retrieve_primary_key_columns
        table_name = table.full_name
        schema = table.schema or "public"
        index_query = textwrap.dedent("""
            SELECT attr.attname, idx.indisprimary
            FROM pg_index idx
                JOIN pg_attribute attr ON idx.indrelid = attr.attrelid AND attr.attnum = ANY(idx.indkey)
                JOIN pg_class cls ON idx.indrelid = cls.oid
                JOIN pg_namespace nsp ON cls.relnamespace = nsp.oid
            WHERE cls.relname = %s
                AND nsp.nspname = %s
        """)

        self._db.cursor().execute(index_query, (table_name, schema))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None

        index_map = dict(result_set)
        return index_map

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._db == other._db

    def __hash__(self):
        return hash(self._db)


# Postgres stores its array datatypes in a more general array-type structure (anyarray).
# However, to extract the individual entries from such an array, the need to be casted to a typed array structure.
# This dictionary contains the necessary casts for the actual column types.
# For example, suppose a column contains integer values. If this column is aggregated into an anyarray entry, the
# appropriate converter for this array is int[]. In other words DTypeArrayConverters["integer"] = "int[]"
_DTypeArrayConverters = {
    "integer": "int[]",
    "text": "text[]",
    "character varying": "text[]",
}


class PostgresStatisticsInterface(DatabaseStatistics):
    """Statistics implementation for Postgres systems.

    Parameters
    ----------
    postgres_db : PostgresInterface
        The database instance for which the statistics should be retrieved
    emulated : bool, optional
        Whether the statistics interface should operate in emulation mode. To enable reproducibility, this is *True*
        by default
    enable_emulation_fallback : bool, optional
        Whether emulation should be used for unsupported statistics when running in native mode, by default True
    cache_enabled : Optional[bool], optional
        Whether emulated statistics queries should be subject to caching, by default True. Set to *None* to use the
        caching behavior of the `db`
    """

    def __init__(self, postgres_db: PostgresInterface) -> None:
        super().__init__()
        self._db = postgres_db

    def n_pages(self, table: TableReference | str) -> int:
        table = table if isinstance(table, TableReference) else TableReference(table)
        schema = table.schema or "public"

        query_template = """
            SELECT relpages
            FROM pg_class
            WHERE oid = %s::regclass
                AND relnamespace = %s::regnamespace
        """

        self._db.cursor().execute(query_template, (table.full_name, schema))
        result_set = self._db.cursor().fetchone()
        if result_set is None:
            raise ValueError(f"Relation not found: {table}")
        return result_set[0]

    def n_buffered(self, table: TableReference | str) -> int:
        """Retrieves the number of buffered pages for the specified table.

        The table can either be a base table or the name of an index.

        Notes
        -----

        The current implementation of this method relies on the *pg_buffercache* extension and works by scanning the
        entire buffer cache. Therefore, it incurs a slight overhead and should not be called inside of hot loops.
        Instead, you can use `buffer_state` to retrieve the number of buffered pages for all relations in a single pass.

        See Also
        --------
        buffer_state
        """

        table = table if isinstance(table, TableReference) else TableReference(table)
        schema = table.schema or "public"

        query_template = """
            SELECT count(*) AS buffers
            FROM pg_buffercache buf
            JOIN pg_class cls ON buf.relfilenode = pg_relation_filenode(cls.oid)
            WHERE cls.relnamespace = %s::regnamespace
                AND cls.relname = %s
                AND buf.reldatabase IN (0, (SELECT oid FROM pg_database WHERE datname = current_database()))
        """

        self._db.cursor().execute(query_template, (schema, table.full_name))
        result_set = self._db.cursor().fetchone()
        if result_set is None:
            # No pages have been buffered
            return 0
        return result_set[0]

    def buffer_state(self, *, schema: str = "public") -> dict[str, int]:
        """Retrieves the current buffer state for all relations in the given schema (*public* by default).

        If a relation is not contained in the result, none of its pages are currently buffered.

        The result contains *all* PG relations, i.e. including indexes and other non-table relations. Typically,
        primary key indexes are named *<relation_name>_pkey*. The name of secondary indexes depends on the schema.

        See Also
        --------
        n_buffered
        """

        query_template = """
            SELECT cls.relname, count(*) AS buffers
            FROM pg_buffercache buf
            JOIN pg_class cls ON buf.relfilenode = pg_relation_filenode(cls.oid)
            WHERE cls.relnamespace = %s::regnamespace
                AND buf.reldatabase IN (0, (SELECT oid FROM pg_database WHERE datname = current_database()))
            GROUP BY cls.relname
        """

        self._db.cursor().execute(query_template, (schema,))
        result_set = self._db.cursor().fetchall()
        if result_set is None:
            # No pages have been buffered
            return {}

        return {row[0]: row[1] for row in result_set}

    def update_statistics(
        self,
        columns: ColumnReference | Iterable[ColumnReference] | None = None,
        *,
        tables: TableReference | Iterable[TableReference] | None = None,
        perfect_mcv: bool = False,
        perfect_n_distinct: bool = False,
        verbose: bool = False,
    ) -> None:
        """Instructs the Postgres server to update statistics for specific columns.

        Notice that is one of the methods of the database interface that explicitly mutates the state of the database system.

        Parameters
        ----------
        columns : Optional[ColumnReference  |  Iterable[ColumnReference]], optional
            The columns for which statistics should be updated. If no columns are given, columns are inferred based on the
            `tables` and all detected columns are used.
        tables : Optional[TableReference  |  Iterable[TableReference]], optional
            The table for which statistics should be updated. If `columns` are given, this parameter is completely ignored. If
            no columns and no tables are given, all tables in the current database are used.
        perfect_mcv : bool, optional
            Whether the database system should attempt to create perfect statistics. Perfect statistics means that for each of
            the columns MCV lists are created such that each distinct value is contained within the list. For large and diverse
            columns, this might lots of compute time as well as storage space. Notice, that the database system still has the
            ultimate decision on whether to generate MCV lists in the first place. Postgres also imposes a hard limit on the
            maximum allowed length of MCV lists and histogram widths.
        perfect_n_distinct : bool, optional
            Whether to set the number of distinct values to its true value.
        verbose : bool, optional
            Whether to print some progress information to standard error.
        """
        if not columns and not tables:
            tables = [tab for tab in self._db.schema().tables() if not self._db.schema().is_view(tab)]
        if not columns and tables:
            tables = util.enlist(tables)
            columns = util.set_union(self._db.schema().columns(tab) for tab in tables)

        assert columns is not None
        columns: Iterable[ColumnReference] = util.enlist(columns)
        columns_map: dict[TableReference, list[str]] = util.dicts.generate_multi(
            (col.table, col.name) for col in columns
        )
        distinct_values: dict[ColumnReference, int] = {}

        # in the first phase, we update the PG statistics settings to use as many values as possible during ANALYZE

        if perfect_mcv or perfect_n_distinct:
            precise_stats = PreciseStatistics(self._db)

            for column in columns:
                util.logging.print_if(
                    verbose,
                    util.timestamp(),
                    ":: Now preparing column",
                    column,
                    use_stderr=True,
                )
                raw = precise_stats.num_distinct(column)
                assert raw is not None
                n_distinct = round(raw)
                if perfect_n_distinct:
                    distinct_values[column] = n_distinct
                if not perfect_mcv:
                    continue

                stats_target_query = textwrap.dedent(f"""
                                                     ALTER TABLE {column.table.full_name}
                                                     ALTER COLUMN {column.name}
                                                     SET STATISTICS {n_distinct};
                                                     """)
                # This query might issue a warning if the requested stats target is larger than the allowed maximum value
                # However, Postgres simply uses the maximum value in this case. To permit different maximum values in different
                # Postgres versions, we accept the warning and do not use a hard-coded maximum value with snapping logic
                # ourselves.
                self._db.cursor().execute(stats_target_query)  # type: ignore - weird psycopg stuff

        columns_str = {table: ", ".join(col for col in columns) for table, columns in columns_map.items()}
        tables_and_columns = ", ".join(f"{table.full_name}({cols})" for table, cols in columns_str.items())

        util.logging.print_if(
            verbose,
            util.timestamp(),
            ":: Now analyzing columns",
            tables_and_columns,
            use_stderr=True,
        )
        query_template = f"ANALYZE {tables_and_columns}"
        self._db.cursor().execute(query_template)  # type: ignore - weird psycopg stuff

        for column, n_distinct in distinct_values.items():
            assert column.table is not None, "Unbound table"
            distinct_update_query = textwrap.dedent(f"""
                                                    ALTER TABLE {column.table.full_name}
                                                    ALTER COLUMN {column.name}
                                                    SET (n_distinct = {n_distinct});
                                                    """)
            self._db.cursor().execute(distinct_update_query)  # type: ignore - weird psycopg stuff

    def total_rows(self, table: TableReference) -> Cardinality | None:
        schema = table.schema or "public"
        count_query = "SELECT reltuples FROM pg_class WHERE oid = %s::regclass AND relnamespace = %s::regnamespace"
        self._db.cursor().execute(count_query, (table.full_name, schema))
        result_set = self._db.cursor().fetchone()
        if not result_set:
            return None
        count = result_set[0]
        return Cardinality.of(count)

    def num_distinct(self, column: ColumnReference) -> int | None:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        elif column.table.virtual:
            raise VirtualTableError(column.table)

        schema = column.table.schema or "public"
        dist_query = """
            SELECT n_distinct
            FROM pg_stats
            WHERE tablename = %s
                AND attname = %s
                AND schemaname = %s
        """
        self._db.cursor().execute(dist_query, (column.table.full_name, column.name, schema))
        result_set = self._db.cursor().fetchone()
        if not result_set:
            return None
        dist_values = result_set[0]

        null_frac = self.null_frac(column)
        assert null_frac is not None
        null_correction = 1 if null_frac > 0 else 0

        # interpreting the n_distinct column is difficult, since different value ranges indicate different things
        # (see https://www.postgresql.org/docs/current/view-pg-stats.html)
        # If the value is >= 0, it represents the actual (approximated) number of distinct non-zero values in the
        # column.
        # If the value is < 0, it represents 'the negative of the number of distinct values divided by the number of
        # rows'. Therefore, we have to correct the number of distinct values manually in this case.
        if dist_values >= 0:
            return int(dist_values) + null_correction

        # correct negative values
        n_rows = self.total_rows(column.table)
        assert n_rows is not None, "Could not retrieve total row count for table"

        return int(-1 * n_rows * dist_values) + null_correction

    def null_frac(self, column: ColumnReference) -> float | None:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        elif column.table.virtual:
            raise VirtualTableError(column.table)

        schema = column.table.schema or "public"
        null_query = """
            SELECT null_frac
            FROM pg_stats
            WHERE tablename = %s
                AND attname = %s
                AND schemaname = %s
        """
        self._db.cursor().execute(null_query, (column.table.full_name, column.name, schema))
        result_set = self._db.cursor().fetchone()
        if not result_set:
            return None
        return float(result_set[0])

    def min_max(self, column: ColumnReference) -> tuple[Any, Any]:
        # Postgres does not keep track of min/max values, so we need to determine them manually
        # XXX: maybe it would be possible to infer min/max values from the histogram. Do we want to do this instead?
        if not db.enable_emulation_fallback:
            raise UnsupportedDatabaseFeatureError(
                self._db, "min/max value statistics. Set db.enable_emulation_fallback to activate."
            )
        return PreciseStatistics(self._db).min_max(column)

    def most_common_values(self, column: ColumnReference) -> MostCommonValues | None:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        elif column.table.virtual:
            raise VirtualTableError(column.table)

        schema = column.table.schema or "public"
        # Postgres stores the Most common values in a column of type anyarray (since in this column, many MCVs from
        # many different tables and data types are present). However, this type is not very convenient to work on.
        # Therefore, we first need to convert the anyarray to an array of the actual attribute type.
        attribute_converter = self._array_cast(column)

        # now, load the most frequent values. Since the frequencies are expressed as a fraction of the total number of
        # rows, we need to multiply this number again to obtain the true number of occurrences
        mcv_query = f"""
            SELECT UNNEST(most_common_vals::text::{attribute_converter}),
                UNNEST(most_common_freqs)
            FROM pg_stats
            WHERE tablename = %s
                AND attname = %s
                AND schemaname = %s
        """

        # NB: we have to repeat a few parameters here. Unfortunately, it seems that psycopg
        # does not support casts for named parameters - %(tab)s::regclass does not work
        self._db.cursor().execute(
            mcv_query,  # type: ignore - weird psycopg stuff
            (
                column.table.full_name,
                column.name,
                schema,
            ),
        )
        result_set = self._db.cursor().fetchall()
        assert result_set is not None
        if not result_set:
            return None

        # Sadly our job is not done yet:
        #
        # Postgres handles NULLs separately and importantly does not allow them to become part of the MCVs.
        # But since the MCV frequencies are relative to the total number of (none-NULL) values, we need to scale
        # them up to obtain the true frequency.
        #
        # What is worse, is that NULL might itself be one of the MCVs.
        # Therefore, we also have to insert it back in.

        n_rows = self.total_rows(column.table)
        null_frac = self.null_frac(column)
        assert n_rows is not None and null_frac is not None
        scale_up = float((1 - null_frac) * n_rows)

        result_set: list[tuple[Any, int]] = [(val, int(freq * scale_up)) for val, freq in result_set]

        min_freq = result_set[-1][1]
        n_nulls = int(null_frac * n_rows)
        if n_nulls > min_freq:
            null_tuple = (None, n_nulls)
            bisect.insort_left(result_set, null_tuple, key=lambda x: x[1])

        return MostCommonValues(result_set)

    def histogram(
        self, column: ColumnReference, *, interpolation: HistogramApproximation = "approx-uni"
    ) -> Histogram | None:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        elif column.table.virtual:
            raise VirtualTableError(column.table)

        attribute_converter = self._array_cast(column)
        schema = column.table.schema or "public"

        query_template = f"""
            SELECT UNNEST(histogram_bounds::text::{attribute_converter})
            FROM pg_stats
            WHERE schemaname = %s AND tablename = %s AND attname = %s
            """

        self._db.cursor().execute(
            query_template,  # type: ignore - weird psycopg stuff
            (schema, column.table.full_name, column.name),
        )
        result_set = self._db.cursor().fetchall()
        if not result_set:
            return None
        bounds = [row[0] for row in result_set]

        # Sadly our job is not done yet:
        #
        # Postgres builds a histogram only for all values that are not part of the MCV list
        # At runtime, the histogram is merged with the MCVs to obtain the "true" distribution
        # We must do the same here
        #
        # Similarly, Postgres does not store the bucket frequency, but only the bucket boundaries.
        # This is because PG constructs an equi-depth histogram so all buckets (should) have (more or less) the same
        # frequency. As a consequence it would be redundant to explicitly store this information. We need to infer
        # the frequency based on the total number of rows and tuples.
        # But there is more, because we ALSO need to take care with NULL values because they appear in the total rows
        # but not in the histogram.
        #
        # Stuff is complicated, man.

        n_rows = self.total_rows(column.table)
        null_frac = self.null_frac(column)
        assert n_rows is not None and null_frac is not None

        bucket_freq = (1 - null_frac) * n_rows // len(bounds)

        mcvs = self.most_common_values(column)
        if mcvs is None:
            mcvs = MostCommonValues.empty()

        normalized_bounds = []
        normalized_frequencies: list[int] = []
        lo = None
        hist_iter = iter(bounds)
        mcv_iter = iter(mcvs)
        cur_hist = next(hist_iter, None)
        cur_mcv = next(mcv_iter, None)
        while cur_hist is not None or cur_mcv is not None:
            if cur_mcv is not None:
                cur_mcv_val, cur_mcv_freq = cur_mcv
            else:
                cur_mcv_val, cur_mcv_freq = None, None

            if lo is None:
                lo = min(
                    filter(None, [cur_hist, cur_mcv_val]),
                    default=None,
                )

            if cur_hist is None:
                assert cur_mcv_val is not None and cur_mcv_freq is not None
                normalized_bounds.append(cur_mcv_val)
                normalized_frequencies.append(cur_mcv_freq)
                cur_mcv = next(mcv_iter, None)

            elif cur_mcv_val is None:
                normalized_bounds.append(cur_hist)
                normalized_frequencies.append(bucket_freq)

                prev_hist = cur_hist
                while (cur_hist := next(hist_iter, None)) == prev_hist:
                    # for very frequent values, Postgres might put the same value into multiple (equi-depth) buckets
                    # in this case, we need to sum up the frequencies of all buckets with the same bound value
                    normalized_frequencies[-1] += bucket_freq

            elif cur_hist < cur_mcv_val:
                normalized_bounds.append(cur_hist)
                normalized_frequencies.append(bucket_freq)

                prev_hist = cur_hist
                while (cur_hist := next(hist_iter, None)) == prev_hist:
                    # see comment above for the case of duplicate histogram bounds
                    normalized_frequencies[-1] += bucket_freq

            else:
                assert cur_mcv_val < cur_hist and cur_mcv_freq is not None
                normalized_bounds.append(cur_mcv_val)
                normalized_frequencies.append(cur_mcv_freq)
                cur_mcv = next(mcv_iter, None)

        assert lo is not None
        return Histogram(
            normalized_bounds,
            normalized_frequencies,
            lower=lo,
            bucket_interpolation=interpolation,
        )

    def _array_cast(self, column: BoundColumnReference) -> str:
        # determine the attributes data type to figure out how it should be converted
        attribute_query = "SELECT data_type FROM information_schema.columns WHERE table_name = %s AND column_name = %s"

        self._db.cursor().execute(attribute_query, (column.table.full_name, column.name))
        result_set = self._db.cursor().fetchone()
        assert result_set

        attribute_dtype = result_set[0]
        converter = _DTypeArrayConverters.get(attribute_dtype)
        if not converter:
            raise ValueError("Cannot cast column array of type {attribute_dtype} - no converter found.")
        return converter


PostgresOptimizerSettings = {
    JoinOperator.NestedLoopJoin: "enable_nestloop",
    JoinOperator.HashJoin: "enable_hashjoin",
    JoinOperator.SortMergeJoin: "enable_mergejoin",
    ScanOperator.SequentialScan: "enable_seqscan",
    ScanOperator.IndexScan: "enable_indexscan",
    ScanOperator.IndexOnlyScan: "enable_indexonlyscan",
    ScanOperator.BitmapScan: "enable_bitmapscan",
    IntermediateOperator.Memoize: "enable_memoize",
    IntermediateOperator.Materialize: "enable_material",
    IntermediateOperator.Sort: "enable_sort",
}
"""All (session-global) optimizer settings that modify the allowed physical operators."""

PGHintPlanOptimizerHints: dict[PhysicalOperator, str] = {
    JoinOperator.NestedLoopJoin: "NestLoop",
    JoinOperator.HashJoin: "HashJoin",
    JoinOperator.SortMergeJoin: "MergeJoin",
    ScanOperator.SequentialScan: "SeqScan",
    ScanOperator.IndexScan: "IndexOnlyScan",
    ScanOperator.IndexOnlyScan: "IndexOnlyScan",
    ScanOperator.BitmapScan: "BitmapScan",
    IntermediateOperator.Memoize: "Memoize",
}
"""All physical operators that can be enforced by pg_hint_plan.

These settings operate on a per-relation basis and overwrite the session-global optimizer settings.

References
----------

.. pg_hint_plan hints: https://github.com/ossc-db/pg_hint_plan/blob/master/docs/hint_list.md
"""

PGLabOptimizerHints: dict[PhysicalOperator, str] = {
    JoinOperator.NestedLoopJoin: "NestLoop",
    JoinOperator.HashJoin: "HashJoin",
    JoinOperator.SortMergeJoin: "MergeJoin",
    ScanOperator.SequentialScan: "SeqScan",
    ScanOperator.IndexScan: "IdxScan",
    ScanOperator.IndexOnlyScan: "IdxScan",
    ScanOperator.BitmapScan: "BitmapScan",
    IntermediateOperator.Materialize: "Material",
    IntermediateOperator.Memoize: "Memo",
}
"""All physical operators that can be enforced by pg_lab.

These settings operate on a per-relation basis and overwrite the session-global optimizer settings.

References
----------

.. pg_lab extension: https://github.com/rbergm/pg_lab/blob/main/docs/hinting.md

"""


PostgresJoinHints = {
    JoinOperator.NestedLoopJoin,
    JoinOperator.HashJoin,
    JoinOperator.SortMergeJoin,
}
"""All join operators that are supported by Postgres."""

PostgresScanHints = {
    ScanOperator.SequentialScan,
    ScanOperator.IndexScan,
    ScanOperator.IndexOnlyScan,
    ScanOperator.BitmapScan,
}
"""All scan operators that are supported by Postgres."""

PostgresPlanHints = {
    HintType.Cardinality,
    HintType.Parallelization,
    HintType.LinearJoinOrder,
    HintType.BushyJoinOrder,
    HintType.JoinDirection,
    HintType.Operator,
}
"""All non-operator hints supported by Postgres, that can be used to enforce additional optimizer behaviour."""


PostgresHintingBackend = Literal["pg_hint_plan", "pg_lab", "none"]
"""The hinting backend being used.

If pg_lab is available, this is the preferred extension. Otherwise, pg_hint_plan is used as a fallback.
If the hint service is inactive, the backend is set to _none_.
"""


def _walk_join_order(node: JoinTree) -> str:
    if node.is_scan():
        return node.base_table.identifier()

    outer = _walk_join_order(node.outer_child)
    inner = _walk_join_order(node.inner_child)
    return f"({outer} {inner})"


def _generate_pghintplan_hints(
    query: SqlQuery,
    join_order: JoinTree | None,
    phys_ops: PhysicalOperatorAssignment | None,
    plan_params: PlanParameterization | None,
    *,
    pg_instance: PostgresInterface,
) -> Hint:
    hints: list[str] = []
    prep_statements: list[str] = []

    geqo_thresh: str = pg_instance.config["geqo_threshold"]
    if len(query.tables()) > int(geqo_thresh):
        warnings.warn(
            "Temporarily disabling GEQO. pg_hint_plan only works with the DP optimizer.",
            category=HintWarning,
            stacklevel=3,
        )
        hints.append("Set(geqo off)")

    if join_order and len(join_order) > 1:
        join_str = _walk_join_order(join_order)
        hints.append(f"Leading({join_str})")

    if phys_ops:
        for scan in phys_ops.scan_operators.values():
            op = PGHintPlanOptimizerHints[scan.operator]
            tab = scan.table.identifier()
            hints.append(f"{op}({tab})")
            if scan.parallel_workers > 1:
                hints.append(f"Parallel({tab} {scan.parallel_workers} hard)")

        for join in phys_ops.join_operators.values():
            op = PGHintPlanOptimizerHints[join.operator]
            intermediate = " ".join(tab.identifier() for tab in join.intermediate)
            hints.append(f"{op}({intermediate})")
            if join.parallel_workers > 1:
                warnings.warn(
                    "Cannot directly set parallel workers on a join with pg_hint_plan. "
                    "Setting on all base tables instead.",
                    category=HintWarning,
                    stacklevel=3,
                )
                for tab in join.intermediate:
                    hints.append(f"Parallel({tab.identifier()} {join.parallel_workers} hard)")

        for tabs, intermediate_op in phys_ops.intermediate_operators.items():
            op = PGHintPlanOptimizerHints.get(intermediate_op)
            if not op:
                warnings.warn(
                    f"Cannot enforce operator {intermediate_op} with pg_hint_plan. Ignoring this hint",
                    category=HintWarning,
                    stacklevel=3,
                )
                continue
            intermediate = " ".join(tab.identifier() for tab in tabs)
            hints.append(f"{op}({intermediate})")

        for op, val in phys_ops.global_settings.items():
            setting = PostgresOptimizerSettings[op]
            hints.append(f"Set({setting} {val})")

    if plan_params:
        for tabs, card in plan_params.cardinalities.items():
            if card.isnan():
                continue

            intermediate = " ".join(tab.identifier() for tab in tabs)
            if card.isinf():
                warnings.warn(
                    f"Ignoring infinite cardinality for intermediate {intermediate}",
                    category=HintWarning,
                    stacklevel=3,
                )
                continue

            hints.append(f"Rows({intermediate} #{card.value})")

        for tabs, workers in plan_params.parallel_workers.items():
            if workers == 1:
                continue

            intermediate = " ".join(tab.identifier() for tab in tabs)
            hints.append(f"Parallel({intermediate} {workers} hard)")

        for setting, val in plan_params.system_settings.items():
            # TODO: we could be smart here and differentiate betwen settings that only affect the optimizer and settings
            # that also affect the execution engine. The former can be set in pg_hint_plan via Set(...), while the latter
            # must be set via a preparatory SET statement. We should avoid this second case if at all possible since it
            # affects the entire session and not just the current query.
            # For now, we mitigate this issue in a different way: we emit SET LOCAL statements which only modify the
            # current transaction. Since the Postgres interface runs in autocommit mode, each query is executed within
            # its own transaction. Therefore, all changes are reverted immediately after the query has finished.
            prep_statements.append(f"SET LOCAL {setting} TO '{val}';")

        if plan_params.execution_mode is not None:
            warnings.warn(
                "pg_hint_plan does not support execution mode hints",
                category=HintWarning,
                stacklevel=3,
            )

    hints = [f" {line}" for line in hints]
    hints.insert(0, "/*+")
    hints.append(" */")

    return Hint("\n".join(prep_statements), "\n".join(hints))


def _generate_pglab_hints(
    join_order: JoinTree | None,
    phys_ops: PhysicalOperatorAssignment | None,
    plan_params: PlanParameterization | None,
) -> Hint:
    hints: list[str] = []
    prep_statements: list[str] = []

    has_worker_params = plan_params is not None and plan_params.parallel_workers

    if has_worker_params and phys_ops is None:
        warnings.warn(
            "pg_lab can only force parallel execution of nodes with known operators. Ignoring worker hints.",
            category=HintWarning,
            stacklevel=3,
        )
    elif has_worker_params:
        assert plan_params is not None and phys_ops is not None
        has_dangling_worker_hints = any(intermediate not in phys_ops for intermediate in plan_params.parallel_workers)
        if has_dangling_worker_hints:
            warnings.warn(
                "pg_lab can only force parallel execution of nodes with known operators. Ignoring additional hints.",
                category=HintWarning,
                stacklevel=3,
            )
        phys_ops = phys_ops.integrate_workers_from(plan_params)

    hints.append("Config(plan_mode=anchored)")

    if join_order and len(join_order) > 1:
        join_str = _walk_join_order(join_order)
        hints.append(f"JoinOrder({join_str})")

    if phys_ops:
        for scan in phys_ops.scan_operators.values():
            op = PGLabOptimizerHints[scan.operator]
            table = scan.table.identifier()

            if scan.parallel_workers > 1:
                # TODO: check for off-by-one errors!!!
                hint = f"{op}({table} (workers={scan.parallel_workers}))"
            else:
                hint = f"{op}({table})"
            hints.append(hint)

        for join in phys_ops.join_operators.values():
            op = PGLabOptimizerHints[join.operator]
            intermediate = " ".join(tab.identifier() for tab in join.intermediate)

            if join.parallel_workers > 1:
                hint = f"{op}({intermediate} (workers={join.parallel_workers}))"
            else:
                hint = f"{op}({intermediate})"
            hints.append(hint)

        for tabs, intermediate_op in phys_ops.intermediate_operators.items():
            op = PGLabOptimizerHints[intermediate_op]
            intermediate = " ".join(tab.identifier() for tab in tabs)
            hints.append(f"{op}({intermediate})")

        for op, enabled in phys_ops.global_settings.items():
            setting = PostgresOptimizerSettings[op]
            value = "on" if enabled else "off"
            hints.append(f"Set({setting} = '{value}')")

    if plan_params:
        for tabs, card in plan_params.cardinalities.items():
            if card.isnan():
                continue

            intermediate = " ".join(tab.identifier() for tab in tabs)
            if card.isinf():
                warnings.warn(
                    f"Ignoring infinite cardinality for intermediate {intermediate}",
                    category=HintWarning,
                    stacklevel=3,
                )
                continue

            hints.append(f"Card({intermediate} #{card})")

        for setting, val in plan_params.system_settings.items():
            hints.append(f"Set({setting} = '{val}')")

        if plan_params.execution_mode is not None:
            mode = "sequential" if plan_params.execution_mode == "sequential" else "parallel"
            hints.append(f"Config(exec_mode={mode})")

    hints = [f"  {line}" for line in hints]
    hints.insert(0, "/*=pg_lab=")
    hints.append(" */")

    return Hint("\n".join(prep_statements), "\n".join(hints))


def _extract_plan_join_order(plan: QueryPlan) -> str:
    if plan.is_scan():
        assert plan.base_table is not None
        return plan.base_table.identifier()
    elif plan.input_node:
        return _extract_plan_join_order(plan.input_node)

    assert plan.outer_child is not None and plan.inner_child is not None
    outer = _extract_plan_join_order(plan.outer_child)
    inner = _extract_plan_join_order(plan.inner_child)
    return f"({outer} {inner})"


def _generate_pglab_plan(
    node: QueryPlan,
    *,
    par_workers: int = 0,
    in_upperrel: bool = True,
    level: int = 0,
) -> list[str]:
    if level == 0:
        join_order = _extract_plan_join_order(node)
        hints: list[str] = [
            "Config(plan_mode=full)",
            f"JoinOrder({join_order})",
        ]
    else:
        hints: list[str] = []

    indentation = " " * level
    hintable_node = node.is_scan() or node.is_join()
    in_upperrel = in_upperrel and not hintable_node

    if par_workers and in_upperrel:
        hints.append(f"Result(workers={par_workers})")

        for child in node.children:
            hints.extend(
                _generate_pglab_plan(
                    child,
                    par_workers=0,
                    in_upperrel=True,
                    level=level + 1,
                )
            )

        return hints

    # We need to set par_workers _after_ we did the upperrel processing to distinguish between a Gather in the upperrel and
    # a gather on top of a join i.e. to distinguish between
    #
    # Finalize Aggregate
    #   -> Gather (workers=4)
    #       -> Partial Aggregate
    #          -> ...
    #
    # and
    #
    # Aggregate
    #   -> Gather (workers=4)
    #       -> Nested Loop
    #          -> ...
    #
    # If we do not delay the par_workers update, we would get the parallel workers from the Gather node, and (since we are
    # in the Gather node) detect that we do not have a join/scan node and thus remain in the upperrel. In turn, we would always
    # create a Result() hint.
    # By delaying the par_workers update, we compare the par_workers value as given by the parent _generate_pglab_plan()
    # invocation and perform the local upperrel check. For a gather node, we would "fall through" to the child node with the
    # actual par_workers. If the child is still an upperrel node, we generate the (now correct) Result(), if we encouter a
    # scan/join, we parallelize it as intended.
    par_workers = node.parallel_workers or par_workers

    operator = PGLabOptimizerHints.get(node.operator) if node.operator else None
    if operator is None:
        for child in node.children:
            hints.extend(
                _generate_pglab_plan(
                    child,
                    par_workers=par_workers,
                    in_upperrel=in_upperrel,
                    level=max(level, 1),
                )
            )
        return hints

    metadata_elems: list[str] = []
    if par_workers:
        metadata_elems.append(f"workers={par_workers}")
        par_workers = 0

    intermediate = " ".join(tab.identifier() for tab in node.tables())
    metadata = " (" + " ".join(metadata_elems) + ")" if metadata_elems else ""
    hints.append(f"{indentation}{operator}({intermediate}{metadata})")

    # HOTFIX: we need to inject the cardinality estimate, even if we hint the operator explicitly
    # This is because pg_lab/Postgres still have some freedom regarding the operator parameterization.
    #
    # Consider Index scans as an example: the Postgres optimizer is free to choose which index to scan.
    # Now, depending on the chosen index, an upper level merge join might require explicit sorting or not.
    # This becomes relevant in the following experiment:
    # For a specific query, inject (arbitrary non-native) cardinality estimates. Capture the execution plan.
    # Afterwards, hint the same plan, but without the cardinality estimates. Now, the optimizer needs to base all decisions on
    # its own native estimates.
    # In experiments, we have actually seen that this shift in cardinality estimates might lead to the optimizer preferring a
    # different index scan and thus require additional sorting for the merge join.
    # Something similar can also happen with memoize nodes: they currently require at least two (estimated) rows in the outer
    # relation to be considered during path generation. If we change the estimates on the outer relation, the optimizer might
    # never even consider a memoize node and thus we cannot enforce the memoization hint at all.
    # Lastly, the same reasoning also applies to switching between Index scan and Index-only scan, as well as to the choice
    # of the bitmap index scans.
    #
    # To circumvent all of these issues, we just inject the cardinality estimates for all hinted nodes.
    # Note that this is still an imperfect band-aid, because
    # a) the cardinalities themselves might be unreliable (e.g. when parallel workers are involved), and
    # b) this should actually be addressed at pg_lab level. For memoize this has already been done, but for index scans we
    #    would essentially need to be able to hint the exact index in addition to the operator
    #
    # As a final note, we always use the estimated cardinality to make sure that the plan contains the intended _estimates_
    # if the user explicitly wants true cardinalities, the QueryPlan provides a with_actual_card() method.
    card = node.estimated_cardinality
    if card.is_valid() and node.operator not in IntermediateOperator:
        hints.append(f"{indentation}Card({intermediate} #{card})")

    if node.is_scan():
        return hints

    for child in node.children:
        hints.extend(
            _generate_pglab_plan(
                child,
                par_workers=par_workers,
                in_upperrel=in_upperrel,
                level=level + 1,
            )
        )

    return hints


def _expand_pglab_hints(raw_hints: list[str]) -> Hint:
    hints = [f"  {line}" for line in raw_hints]
    hints.insert(0, "/*=pg_lab=")
    hints.append(" */")
    return Hint("", "\n".join(hints))


class PostgresHintService(HintService):
    """Postgres-specific implementation of the hinting capabilities.

    Most importantly, this service implements a mapping from the abstract optimization descisions (join order + operators) to
    their counterparts in the hinting backend and integrates Postgres' few deviations from standard SQL syntax (*CAST*
    expressions and *LIMIT* clauses).

    The hinting service supports two different kinds of backends: pg_lab or pg_hint_plan. The former is the preferred option
    since it provides cardinality hints for base joins and does not require management of the GeQO optimizer.

    Notice that by delegating the adaptation of Postgres' native optimizer to the pg_hint_plan extension, a couple of
    undesired side-effects have to be accepted:

    1. forcing a join order also involves forcing a specific join direction. Our implementation applies a couple of heuristics
       to mitigate a bad impact on performance
    2. the extension only instruments the dynamic programming-based optimizer. If the *geqo_threshold* is reached and the
       genetic optimizer takes over, no modifications are applied. Therefore, it is best to disable GeQO while working with
       Postgres. At the same time, this means that certain scenarios like custom cardinality estimation for the genetic
       optimizer cannot currently be tested

    Parameters
    ----------
    postgres_db : PostgresInterface
        A postgres database with an active hinting backend (pg_hint_plan or pg_lab)

    Raises
    ------
    ValueError
        If the supplied `postgres_db` does not have a supported hinting backend enabled.

    See Also
    --------
    _generate_pg_join_order_hint

    References
    ----------

    .. pg_hint_plan extension: https://github.com/ossc-db/pg_hint_plan
    .. Postgres query planning configuration: https://www.postgresql.org/docs/current/runtime-config-query.html
    """

    def __init__(self, postgres_db: PostgresInterface) -> None:
        self._postgres_db = postgres_db
        self._inactive = True
        self._backend = "none"
        self._infer_pg_backend()

    def _get_backend(self) -> PostgresHintingBackend:
        return self._backend

    def _set_backend(self, backend_name: PostgresHintingBackend) -> None:
        self._inactive = backend_name == "none"
        self._backend = backend_name

    backend = property(_get_backend, _set_backend, doc="The hinting backend in use.")

    def generate_hints(
        self,
        query: SqlQuery,
        plan: QueryPlan | None = None,
        *,
        join_order: JoinTree | None = None,
        physical_operators: PhysicalOperatorAssignment | None = None,
        plan_parameters: PlanParameterization | None = None,
    ) -> SqlQuery:
        self._assert_active_backend()

        has_param = any(param is not None for param in (join_order, physical_operators, plan_parameters))
        if plan is not None and has_param:
            raise ValueError("Can only hint an entire query plan, or individual parts, not both.")

        match self._backend:
            case "pg_hint_plan":
                if plan is not None:
                    join_order = jointree_from_plan(plan)
                    physical_operators = operators_from_plan(plan, include_workers=False)
                    plan_parameters = parameters_from_plan(
                        plan,
                        target_cardinality="actual",
                        fallback_estimated=True,
                    )

                hints = _generate_pghintplan_hints(
                    query,
                    join_order,
                    physical_operators,
                    plan_parameters,
                    pg_instance=self._postgres_db,
                )
            case "pg_lab" if plan is not None:
                raw_hints = _generate_pglab_plan(plan)
                hints = _expand_pglab_hints(raw_hints)
            case "pg_lab":
                hints = _generate_pglab_hints(
                    join_order,
                    physical_operators,
                    plan_parameters,
                )

        return transform.add_clause(query, hints)

    def format_query(self, query: SqlQuery) -> str:
        return qal.format_quick(query, flavor="postgres")

    def supports_hint(self, hint: PhysicalOperator | HintType) -> bool:
        self._assert_active_backend()
        return hint in PostgresJoinHints | PostgresScanHints | PostgresPlanHints

    def describe(self) -> dict[str, str]:
        """Provides a JSON-serializable description of the hint service.

        Returns
        -------
        dict[str, str]
            Information about the hinting backend
        """
        return {"backend": self._backend}

    def _infer_pg_backend(self) -> None:
        """Determines the hinting backend that is provided by the current Postgres instance."""

        # We first try the easy route: checking whether any of the settings related to the hinting backends are available and
        # activated. If this is the case, we are already done.
        # Otherwise, we need to become more creative and rely on more advanced heuristics.
        # Note that on recent installations of Postgres/pg_hint_plan or pg_lab, we can expect that the easy route does indeed
        # work. It is just on older versions that the settings were not available.

        cur = self._postgres_db.cursor()
        try:
            cur.execute("SHOW pg_hint_plan.enable_hint;")
            res = cur.fetchone()
            if res and res[0] == "on":
                util.logging.print_if(
                    self._postgres_db.debug,
                    "Using pg_hint_plan hinting backend",
                    file=sys.stderr,
                )
                self._inactive = False
                self._backend = "pg_hint_plan"
                return
        except psycopg.errors.UndefinedObject:
            pass

        try:
            cur.execute("SHOW enable_pglab;")
            res = cur.fetchone()
            if res and res[0] == "on":
                util.logging.print_if(
                    self._postgres_db.debug,
                    "Using pg_lab hinting backend",
                    file=sys.stderr,
                )
                self._inactive = False
                self._backend = "pg_lab"
                return
        except psycopg.errors.UndefinedObject:
            pass

        # At this point the easy route failed and we need to rely on more advanced heuristics.
        # Specifically, we try to check whether a shared library related to one of the backends is currently loaded
        # in the backend process. See the later comment for the reasoning.
        #
        # All code below should be considered legacy and we might in fact remove it entirely in future versions of PostBOUND.

        if os.name != "posix":
            warnings.warn(
                "It seems you are running PostBOUND on a non-POSIX system. "
                "Please beware that PostBOUND is currently not intended to run on different systems and "
                "there might be (many) dragons. "
                "Proceed at your own risk. "
                "We assume that the Postgres server has pg_hint_plan enabled. "
                "Please set the backend property to pg_lab manually if you are using pg_lab.",
                stacklevel=3,
            )
            self._backend = "pg_hint_plan"
            self._inactive = False
            return

        connection = self._postgres_db.connection()
        backend_pid = connection.info.backend_pid
        hostname = connection.info.host

        # Postgres does not provide a direct method to determine which extensions are currently active if they have only
        # been loaded as a shared library (as is the case for both pg_hint_plan and pg_lab). Therefore, we have to rely on
        # the assumption that the Postgres server is running on the same (virtual) machine as our PostBOUND process and can
        # rely on the operating system to determine open files of the backend process (which will include the shared libaries)

        if sys.platform == "darwin":
            pg_candidates = subprocess.run(
                ["lsof -p " + str(backend_pid) + " | awk '/postgres/{print $1}'"],
                capture_output=True,
                shell=True,
                text=True,
            )
        else:
            pg_candidates = subprocess.run(
                ["ps -aux | awk '/" + str(backend_pid) + "/{print $11}'"],
                capture_output=True,
                shell=True,
                text=True,
            )
        found_pg = any(candidate.lower().startswith("postgres") for candidate in pg_candidates.stdout.split())

        # There are some rare edge cases where our heuristics fail. We have to accept them for now, but should improve the
        # backend detection in the future. Most importantly, the heuristic will pass if we are connected to a remote server
        # on localhost (e.g. via SSH tunneling or WSL instances) and there is a different Postgres server running on the same
        # machine as the PostBOUND process. In this case, our heuristics assume that these are the same servers.
        # In the future, we might want to check the ports as well, but this probably requires superuser privileges
        # (for netstat).

        if hostname not in ["localhost", "127.0.0.1", "::1"] or not found_pg:
            warnings.warn(
                "It seems you are connecting to a remote Postgres instance. "
                "PostBOUND cannot infer the hinting backend for such connections. "
                "We assume that the this server has pg_hint_plan enabled. "
                "Please set the backend property to pg_lab manually if you are using pg_lab.",
                stacklevel=3,
            )
            self._backend = "pg_hint_plan"
            self._inactive = False
            return

        lib_ext = "dylib" if sys.platform == "darwin" else "so"
        active_extensions = util.system.open_files(backend_pid)
        if any(ext.endswith(f"pg_lab.{lib_ext}") for ext in active_extensions):
            util.logging.print_if(
                self._postgres_db.debug,
                "Using pg_lab hinting backend",
                file=sys.stderr,
            )
            self._inactive = False
            self._backend = "pg_lab"
        elif any(ext.endswith(f"pg_hint_plan.{lib_ext}") for ext in active_extensions):
            util.logging.print_if(
                self._postgres_db.debug,
                "Using pg_hint_plan hinting backend",
                file=sys.stderr,
            )
            self._inactive = False
            self._backend = "pg_hint_plan"
        else:
            warnings.warn(
                "No supported hinting backend found. "
                "Please ensure that either pg_hint_plan or pg_lab is available in your Postgres instance.",
                stacklevel=3,
            )
            self._inactive = True
            self._backend = "none"

    def _assert_active_backend(self) -> None:
        """Ensures that a proper hinting backend is available.

        Raises
        ------
        ValueError
            If no backend is available.
        """
        if self._inactive:
            connection_pid = self._postgres_db._connection.info.backend_pid
            raise ValueError(f"No supported hinting backend found for backend with PID {connection_pid}")

    def __repr__(self) -> str:
        return f"PostgresHintService(db={self._postgres_db} backend={self._backend})"

    def __str__(self) -> str:
        return repr(self)


class PostgresOptimizer(OptimizerInterface):
    """Optimizer introspection for Postgres.

    Parameters
    ----------
    postgres_instance : PostgresInterface
        The database whose optimizer should be introspected
    """

    def __init__(self, postgres_instance: PostgresInterface) -> None:
        self._pg_instance = postgres_instance

    def query_plan(self, query: SqlQuery | str) -> QueryPlan:
        if isinstance(query, SqlQuery):
            query = transform.as_explain(query)
            query = self._pg_instance._hinting_backend.format_query(query)
        else:
            query = self._explainify(query)
        raw_query_plan: list = self._pg_instance.execute_query(query)
        query_plan = PostgresExplainPlan(raw_query_plan[0])
        return query_plan.as_qep()

    @overload
    def analyze_plan(self, query: SqlQuery) -> QueryPlan: ...

    @overload
    def analyze_plan(self, query: SqlQuery, *, timeout: float) -> QueryPlan | None: ...

    @overload
    def analyze_plan(self, query: SqlQuery, *, timeout: Literal[None]) -> QueryPlan: ...

    def analyze_plan(self, query: SqlQuery, *, timeout: float | None = None) -> QueryPlan | None:
        query = transform.as_explain_analyze(query)

        try:
            raw_query_plan: dict = self._pg_instance.execute_query(query, timeout=timeout)[0]
        except TimeoutError:
            return None

        query_plan = PostgresExplainPlan(raw_query_plan)
        return query_plan.as_qep()

    def parse_plan(self, plan: Any, *, query: SqlQuery | None = None) -> QueryPlan:
        # We should be graceful and handle both simplified and unsimplified
        # versions of the execute_query() output. This only works because PostgresExplainPlan
        # is also cooperative and excepts a dictionary and a list-of-dictionary input as well
        # Therefore, we can aggressively unwrap a list, which either yields a dictionary (in
        # case of simplified result sets), or a tuple-of-list-of-dictionary (in case of a raw
        # result set).
        # If we should ever encouter an EXPLAIN query whose list contains multiple dictionaries
        # we have a problem.
        if isinstance(plan, list):
            plan = plan[0]
        if isinstance(plan, tuple):
            plan = plan[0]

        pg_plan = PostgresExplainPlan(plan)
        return pg_plan.as_qep()

    def cardinality_estimate(self, query: SqlQuery | str) -> Cardinality:
        if isinstance(query, SqlQuery):
            query = transform.as_explain(query)
            query = self._pg_instance._hinting_backend.format_query(query)
        else:
            query = self._explainify(query)
        query_plan = self._pg_instance.execute_query(query)
        estimate: int = query_plan[0]["Plan"]["Plan Rows"]
        return Cardinality(estimate)

    def cost_estimate(self, query: SqlQuery | str) -> float:
        if isinstance(query, SqlQuery):
            query = transform.as_explain(query)
            query = self._pg_instance._hinting_backend.format_query(query)
        else:
            query = self._explainify(query)
        query_plan = self._pg_instance.execute_query(query)
        estimate: float = query_plan[0]["Plan"]["Total Cost"]
        return estimate

    def configure_operator(self, operator: PhysicalOperator, *, enabled: bool) -> None:
        """Enables or disables a specific physical operator for the current Postgres connection.

        Parameters
        ----------
        operator : PhysicalOperator
            The operator to configure.
        enabled : bool
            Whether the operator should be allowed or not.

        References
        ----------
        https://www.postgresql.org/docs/current/runtime-config-query.html
        """
        setting_name = PostgresOptimizerSettings.get(operator)
        if not setting_name:
            raise ValueError(f"Cannot configure operator {operator} as it is not supported by Postgres")
        status = "on" if enabled else "off"
        self._pg_instance.cursor.execute(f"SET {setting_name} TO {status}")  # type: ignore

    def _explainify(self, query: str) -> str:
        if not query.upper().startswith("EXPLAIN (FORMAT JSON)"):
            query = f"EXPLAIN (FORMAT JSON) {query}"
        return query


@dataclass
class _BackendConnectedEvent:
    backend_pid: int


@dataclass
class _WorkerErrorEvent:
    error: Exception


@dataclass
class _QueryReadyEvent:
    pass


@dataclass
class _QueryFinishedEvent:
    pass


@dataclass
class _ResultEvent:
    status: Literal["success", "timeout", "failure"]
    result_set: ResultSet | None
    exec_time: float
    error: Exception | None = None

    @staticmethod
    def ok(result_set: ResultSet | None, exec_time: float) -> _ResultEvent:
        return _ResultEvent("success", result_set, exec_time)

    @staticmethod
    def timeout(duration: float) -> _ResultEvent:
        return _ResultEvent("timeout", None, duration)

    @staticmethod
    def failed(error: Exception) -> _ResultEvent:
        return _ResultEvent("failure", None, math.nan, error=error)


def _timeout_worker_ctl(
    status_pipe: mp_conn.Connection,
    *,
    timeout: float,
    executor: _TimeoutQueryExecutor,
) -> _ResultEvent:
    event = status_pipe.recv()
    match event:
        case _BackendConnectedEvent(pid):
            return _timeout_worker_prep(status_pipe, timeout=timeout, backend_pid=pid, executor=executor)
        case _WorkerErrorEvent(e):
            return _ResultEvent.failed(e)
        case _:
            raise StateError("Unexpected event", event)


def _timeout_worker_prep(
    status_pipe: mp_conn.Connection,
    *,
    timeout: float,
    backend_pid: int,
    executor: _TimeoutQueryExecutor,
) -> _ResultEvent:
    event = status_pipe.recv()
    match event:
        case _QueryReadyEvent():
            return _timeout_worker_run_query(
                status_pipe,
                timeout=timeout,
                backend_pid=backend_pid,
                executor=executor,
            )
        case _WorkerErrorEvent(e):
            _timeout_worker_abort(e, backend_pid, executor=executor)
            return _ResultEvent.failed(e)
        case _:
            raise StateError("Unexpected event", event)


def _timeout_worker_run_query(
    status_pipe: mp_conn.Connection,
    *,
    timeout: float,
    backend_pid: int,
    executor: _TimeoutQueryExecutor,
) -> _ResultEvent:
    query_finished = status_pipe.poll(timeout)
    if not query_finished:
        return _timeout_worker_timeout(timeout, backend_pid=backend_pid, executor=executor)

    event = status_pipe.recv()
    match event:
        case _QueryFinishedEvent():
            return _timeout_worker_await_result(status_pipe, backend_pid=backend_pid, executor=executor)
        case _WorkerErrorEvent(e):
            _timeout_worker_abort(e, backend_pid, executor=executor)
            return _ResultEvent.failed(e)
        case _:
            raise StateError("Unexpected event", event)


def _timeout_worker_await_result(
    status_pipe: mp_conn.Connection,
    *,
    backend_pid: int,
    executor: _TimeoutQueryExecutor,
) -> _ResultEvent:
    event = status_pipe.recv()
    match event:
        case _ResultEvent():
            return event
        case _WorkerErrorEvent(e):
            _timeout_worker_abort(e, backend_pid, executor=executor)
            return _ResultEvent.failed(e)
        case _:
            raise StateError("Unexpected event", event)


def _timeout_worker_timeout(timeout: float, *, backend_pid: int, executor: _TimeoutQueryExecutor) -> _ResultEvent:
    executor._abort_backend(backend_pid)
    return _ResultEvent.timeout(timeout)


def _timeout_worker_abort(error: Exception, backend_pid: int, *, executor: _TimeoutQueryExecutor) -> _ResultEvent:
    executor._abort_backend(backend_pid)
    return _ResultEvent.failed(error)


def _timeout_query_worker(
    query: SqlQuery | str,
    *,
    pg_config: dict,
    status_pipe: mp_conn.Connection,
    **kwargs,
) -> None:
    """Internal function to the `TimeoutQueryExecutor` to run individual queries.

    Query results are sent via the `result_send` pipe, not as a return value. In case of any errors, these are sent via the
    `err_send` pipe. Therefore, it is best to check the `err_send` pipe first, before reading from the `result_send` pipe.

    Parameters
    ----------
    query : SqlQuery | str
        Query to execute
    pg_config : dict
        Pickable representation of the current Postgres connection. This is used to re-establish the connection in the parallel
        worker.
    result_send : mp_conn.Connection
        Pipe connection to send the query result
    err_send : mp_conn.Connection
        Pipe connection to send any errors that occurred during the query execution
    backend_send : mp_conn.Connection
        Pipe connection to send the backend PID
    kwargs : Any
        Additional parameters to pass to the `PostgresInterface.execute_query` method.
    """
    pg_instance: PostgresInterface | None = None
    try:
        connect_string = pg_config["connect_string"]
        pg_instance = PostgresInterface(
            connect_string,
            application_name="PostBOUND Timeout Worker",
        )
        status_pipe.send(_BackendConnectedEvent(pg_instance.backend_pid()))
        pg_instance.apply_configuration(pg_config["config"])
        cursor = pg_instance.cursor()
        if isinstance(query, SqlQuery):
            query = _apply_preparatory_statements(query, cur=cursor)
            query = pg_instance._hinting_backend.format_query(query)
        elif isinstance(query, UserString):
            query = str(query)

        # NB: The query execution logic is a slightly modified version of the one in PostgresInterface.execute_query
        # Make sure to keep them in sync.
        # We duplicate the logic here rather than calling the method directly to make the timeout measurement as accurate
        # as possible. By just delegating to execute_query(), we would also include stuff like caching checks and result
        # simplification in the timeout measurement, which is not desired.

        try:
            status_pipe.send(_QueryReadyEvent())
            start_time = time.perf_counter_ns()
            cursor.execute(query)  # type: ignore
            end_time = time.perf_counter_ns()
            status_pipe.send(_QueryFinishedEvent())
        except (psycopg.InternalError, psycopg.OperationalError) as e:
            msg = "\n".join(
                [
                    f"At {util.timestamp()}",
                    "For query:",
                    str(query),
                    "Message:",
                    str(e),
                ]
            )
            raise DatabaseServerError(msg, e) from e
        except psycopg.Error as e:
            msg = "\n".join(
                [
                    f"At {util.timestamp()}",
                    "For query:",
                    str(query),
                    "Message:",
                    str(e),
                ]
            )
            raise DatabaseUserError(msg, e) from e

        runtime = (end_time - start_time) / 10**9
        pg_instance._last_query_runtime = runtime

        if cursor.rownumber is None:
            # For statements that do not return a result (e.g. SET),
            # rownumber is None. We can use this as an indicator whether
            # fetching results is possible.
            status_pipe.send(_ResultEvent.ok(None, runtime))

        raw_result_set = cursor.fetchall()
        if kwargs.get("raw", False):
            result = raw_result_set
        else:
            result = simplify_result_set(raw_result_set)

        status_pipe.send(_ResultEvent.ok(result, runtime))
    except Exception as e:
        status_pipe.send(_WorkerErrorEvent(e))
    finally:
        if pg_instance is not None:
            pg_instance.close()


class _TimeoutQueryExecutor:
    """The TimeoutQueryExecutor provides a mechanism to execute queries with a timeout attached.

    If the query takes longer than the designated timeout, its execution is cancelled. The query execution itself is delegated
    to the `PostgresInterface`, so all its rules still apply. At the same time, using the timeout executor service can
    invalidate some of the state that is exposed by the database interface (see *Warnings* below). Therefore, the relevant
    variables should be refreshed once the timeout executor was used.

    In addition to calling the `execute_query` method directly, the executor also implements *__call__* for more convenient
    access. Both methods accept the same parameters.

    Parameters
    ----------
    postgres_instance : Optional[PostgresInterface], optional
        Database to execute the queries. If omitted, this is inferred from the `DatabasePool`.

    Warnings
    --------
    When a query gets cancelled due to the timeout being reached, the current cursor as well as database connection might be
    refreshed. Any direct references to these instances should no longer be used.
    """

    def __init__(self, postgres_instance: PostgresInterface | None = None) -> None:
        self._pg_instance: PostgresInterface
        if postgres_instance is not None:
            self._pg_instance = postgres_instance
        else:
            fallback = DatabasePool.get_instance().current_database()
            if not isinstance(fallback, PostgresInterface):
                raise ValueError(
                    "Cannot create TimeoutQueryExecutor: No Postgres instance was supplied and the current database is not a "
                    "Postgres instance."
                )
            self._pg_instance = fallback

        self._timeout_watchdog = None

    def execute_query(self, query: SqlQuery | str, timeout: float, **kwargs) -> Any:
        """Runs a query on the database connection, cancelling if it takes longer than a specific timeout.

        Parameters
        ----------
        query : SqlQuery | str
            Query to execute
        timeout : float
            Maximum query execution time in seconds.
        **kwargs
            Additional parameters to pass to the `PostgresInterface.execute_query` method.

        Returns
        -------
        Any
            The query result if it terminated timely. Rules from `PostgresInterface.execute_query` apply.

        Raises
        ------
        TimeoutError
            If the query execution was not finished after `timeout` seconds.

        See Also
        --------
        PostgresInterface.execute_query
        PostgresInterface.reset_connection
        """
        cached_query: bool = kwargs.get("cache_enabled", False) and query in self._pg_instance._query_cache
        if cached_query:
            return self._pg_instance._query_cache[str(query)]

        self._init_watchdog()

        # We implement the timeout mechanism in a separate worker process. The main process keeps track of the progress of that
        # worker using a state pattern. Each major step in the process is represented by a separate event that is in turn
        # processed by a separate state handler (implement as a simple function). The protocol looks like this:
        #
        # 1. The worker process is started and establishes a connection to the database. Once connected, it sends a
        #    _BackendConnectedEvent to the state machine entry (_timeout_worker_ctl). This event contains the backend PID of
        #    the connection. In case of any errors later on, we use this PID to cancel the query (especially if there is a
        #    timeout). If an error occurs during connection establishment, a _WorkerErrorEvent is sent instead. This
        #    effectively cancels the entire query execution.
        # 2. Upon receiving the _BackendConnectedEvent, the state machine transitions to the query preparation state
        #    (_timeout_worker_prep). In parallel, the worker prepares the query for execution, which involves applying any
        #    necessary configuration settings, running preparatory statements, etc. Once the query is ready to be executed, a
        #    _QueryReadyEvent is sent and the state machine transitions to the query execution state. In case of any errors we
        #    shut down the backend.
        # 3. The worker starts executing the query and the state machine starts the timeout countdown
        #    (_timeout_worker_run_query). We only start the timeout now because connection establishment takes some time and we
        #    have seen that this can lead to spurious timeouts if the timeout is very short.
        # 4. As soon as the query terminates, the worker sends a _QueryFinishedEvent. In the meantime, the main process waits
        #    for this event with the designated timeout. If the timeout is reached before the event arrives, the query did not
        #    finish in time and we proceed to cancel it (_timeout_worker_timeout). If the query finishes in time, we transition
        #    to wait for the final result (_timeout_worker_await_result).
        # 5. The worker prepares the final result set (mostly simplification) and calculates the actual execution time. This
        #    is wrapped in a _ResultEvent and sent to the main process. Afterwards, the worker proceeds to close the
        #    backend connection.
        # 6. The main process receives the _ResultEvent and the timeout executor proceeds to shut down the worker process.
        # 7. The main process checks and handles the _ResultEvent as necessary.

        status_recv, status_send = mp.Pipe(False)

        query_worker = mp.Process(
            target=_timeout_query_worker,
            args=(query,),
            kwargs={
                "pg_config": self._pg_fingerprint(),
                "status_pipe": status_send,
                **kwargs,
            },
        )

        query_worker.start()
        result = _timeout_worker_ctl(status_recv, timeout=timeout, executor=self)
        timed_out = result.status == "timeout"
        if timed_out:
            query_worker.terminate()

        query_worker.join()
        query_worker.close()
        status_send.close()
        status_recv.close()

        match result.status:
            case "success" | "timeout":
                self._pg_instance._last_query_runtime = result.exec_time
                query_result = result.result_set
            case "failure":
                assert result.error is not None
                self._pg_instance._last_query_runtime = math.nan
                raise result.error
            case _:
                raise StateError("Unexpected result status", result.status)

        if not timed_out and kwargs.get("cache_enabled", False):
            warnings.warn(
                "Cannot cache query results that were obtained with a timeout.",
                category=QueryCacheWarning,
                stacklevel=2,
            )

        if timed_out:
            raise TimeoutError(query)
        else:
            return query_result

    def close(self) -> None:
        """Closes any internal resources held by the timeout executor.

        After calling this method, the timeout executor should no longer be used.
        """
        if self._timeout_watchdog is not None:
            self._timeout_watchdog.close()
            self._timeout_watchdog = None

    def _pg_fingerprint(self) -> dict:
        """Generate a pickable representation of the current Postgres connection."""
        return {
            "connect_string": self._pg_instance.connect_string,
            "config": self._pg_instance.current_configuration(runtime_changeable_only=True),
        }

    def _init_watchdog(self) -> None:
        if self._timeout_watchdog is not None:
            return

        assert self._pg_instance is not None
        self._timeout_watchdog = psycopg.connect(
            self._pg_instance.connect_string,
            application_name="PostBOUND Timeout Watchdog",
        )

    def _abort_backend(self, pid: int) -> None:
        assert self._timeout_watchdog is not None
        with self._timeout_watchdog.cursor() as cursor:
            cursor.execute(f"SELECT pg_cancel_backend({pid});")  # type: ignore
        self._timeout_watchdog.rollback()

    def __call__(self, query: SqlQuery | str, timeout: float, **kwargs) -> Any:
        return self.execute_query(query, timeout, **kwargs)


def _reconnect(key: str, *, pool: DatabasePool) -> PostgresInterface:
    """Fetches a connection from the database pool.

    If the connection is in a bad state (e.g. because the user called close() before), it is re-established.

    Parameters
    ----------
    key : str
        The name of the database connection in the pool.
    pool : DatabasePool
        The current pool.
    """
    current_instance = pool.retrieve_database(key)
    assert isinstance(current_instance, PostgresInterface)

    status = current_instance._connection.info.status
    if status != psycopg.pq.ConnStatus.OK:
        # Actually there are a lot of other ConnStatus values beyond OK and Bad
        # We could handle them explicitly here, or we might just define anything that is not OK as Bad.
        # The latter seems much simpler so let's just do this for now.
        current_instance.reset_connection()

    return current_instance


def _ini_config_reader(file: TextIO, *, path: Path) -> dict[str, str]:
    config = configparser.ConfigParser()
    config.read_file(file)
    if len(config.sections()) != 1:
        raise ValueError(f"Malformed INI file '{path}': INI config file must contain exactly one section.")
    section = config.sections()[0]
    return dict(config.items(section))


def _read_connection_from_file(config_file: Path) -> str:
    extension = config_file.suffix.lower()
    if extension == "":
        with open(config_file) as f:
            return f.readline().strip()

    match extension:
        case ".toml":
            reader = tomllib.load
            mode = "rb"
        case ".json":
            reader = json.load
            mode = "r"
        case ".ini":
            reader = functools.partial(_ini_config_reader, path=config_file)
            mode = "r"
        case ".yml" | ".yaml":
            import yaml

            reader = yaml.safe_load
            mode = "r"
        case _:
            raise ValueError(f"Unsupported config file format. Could read config file '{config_file}'.")

    with open(config_file, mode) as f:
        config_data = reader(f)  #  type: ignore - correct mode is guarded by the match above
    parts = (f"{param} = '{value}'" for param, value in config_data.items())
    return " ".join(parts)


def connect(
    *,
    application_name: str = "PostBOUND",
    connect_string: str = "",
    config_file: str | Path = "",
    encoding: str = "UTF8",
    refresh: bool = False,
    private: bool = False,
    debug: bool = False,
) -> PostgresInterface:
    """Convenience function to seamlessly connect to a Postgres instance.

    This function obtains a connection to a Postgres database by trying the following methods in order:

    1. if the connect-string is supplied directly via the `connect_string` parameter, this is used
    2. the connect string is read from the `config_file` if this parameter is supplied. This file has to be located in the
       current working directory, but absolute and relative paths are supported. If the file does not exist, an error is
       raised.
    3. the connect string is read from the default connection file *.psycopg_connection* in the current working directory
    4. the connection parameters are read from the standard Postgres environment variables (e.g. *PGDATABASE*, *PGHOST*, ...).
       This method is triggered via the presence of the *PGDATABASE* environment variable. Note that this method is generally
       discouraged due to its implicit and non-obvious nature. A warning is emitted if this method is used.

    If none of these methods worked, an error is raised.

    After a connection to the Postgres instance has been obtained, it is registered automatically on the current
    `DatabasePool` instance. This can be changed via the `private` parameter.

    Config file formats
    -------------------

    The configuration file can be supplied in different formats. Currently supported are:

    - Plain text files (no extension), such as *.psycopg_connection*: The entire file contents are read as a single
      psycopg-compatible connect string
    - INI files (*.ini*): The file must contain exactly one section. All key-value pairs in this section are treated as
      connection parameters.
    - TOML files (*.toml*): The file is parsed as a TOML document. All top-level key-value pairs are treated as connection
      parameters.
    - JSON files (*.json*): The file is parsed as a JSON document. All top
      level key-value pairs are treated as connection parameters.
    - YAML files (*.yml* or *.yaml*): The file is parsed as a YAML document. All top-level key-value pairs are treated as
      connection parameters. This requires the PyYAML package to be installed.

    Parameters
    ----------
    application_name : str, optional
        Identifier for the Postgres server. This will be the name that is shown in the server logs and process lists.
    connect_string : str, optional
        A Psycopg-compatible connect string for the database. Supplying this parameter overwrites any other connection
        information
    config_file : str | Path, optional
        A file containing a Psycopg-compatible connect string for the database. This is the default and preferred method of
        connecting to a Postgres database. Defaults to *.psycopg_connection*
        See the section on config_file formats for supported file types. The appropriate parser is selected based on the
        file extension.
    encoding : str, optional
        The client enconding of the connection. Defaults to *UTF8*.
    refresh : bool, optional
        If true, a new connection to the database will always be established, even if a connection to the same database is
        already pooled. The registration key will be suffixed to prevent collisions. By default, the current connection is
        re-used. If that is the case, no further information (e.g. config strings) is read and only the `name` is accessed.
    private : bool, optional
        If true, skips registration of the new instance on the `DatabasePool`. Registration is performed by default.

    Returns
    -------
    PostgresInterface
        The Postgres database object

    Raises
    ------
    ValueError
        If neither a config file nor a connect string was given, or if the connect file should be used but does not exist

    References
    ----------

    .. Psyopg v3: https://www.psycopg.org/psycopg3/ This is used internally by the Postgres interface to interact with the
       database
    .. Postgres environment variables: https://www.postgresql.org/docs/current/libpq-envars.html
    """
    if connect_string:
        connect_string = connect_string.strip()
    elif config_file:
        config_file = Path(config_file)
        if not config_file.is_file():
            wdir = os.getcwd()
            raise ValueError(
                f"Failed to obtain a database connection. Tried to read the config file '{config_file}' from "
                f"your current working directory, but the file was not found. Your working directory is {wdir}. "
                "Please either supply the connect string directly to the connect() method, or ensure that the "
                "config file exists."
            )
        connect_string = _read_connection_from_file(config_file)
    elif Path(".psycopg_connection").is_file():
        with open(".psycopg_connection") as f:
            connect_string = f.readline().strip()
    elif os.getenv("PGDATABASE"):
        warnings.warn(
            "Using environment variables to construct connection string.",
            stacklevel=2,
        )
        env_vars = {
            "PGDATABASE": "dbname",
            "PGHOST": "host",
            "PGPORT": "port",
            "PGUSER": "user",
            "PGPASSWORD": "password",
            "PGPASSFILE": "passfile",
        }
        components: list[str] = []
        for var, key in env_vars.items():
            val = os.getenv(var)
            if not val:
                continue
            components.append(f"{key} = '{val}'")
        connect_string = " ".join(components)
    else:
        raise ValueError(
            "Failed to obtain a database connection. Please either supply the connect string directly to the "
            "connect() method, or put a configuration file in your working directory. See the documentation of "
            "the connect() method for more details."
        )

    pool_key = f"postgres[{connect_string}]"
    db_pool = DatabasePool.get_instance()
    if pool_key in db_pool and not refresh:
        return _reconnect(pool_key, pool=db_pool)

    postgres_db = PostgresInterface(
        connect_string,
        application_name=application_name,
        client_encoding=encoding,
        debug=debug,
    )

    if not private:
        db_pool.register_database(pool_key, postgres_db)

    return postgres_db
