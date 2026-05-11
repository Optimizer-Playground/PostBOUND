"""Contains the Postgres implementation of the Database interface.

In many ways the Postgres implementation can be thought of as the reference or blueprint implementation of the database
interface. This is due to two main reasons: first up, Postgres' capabilities follow a traditional architecture and its
features cover most of the general aspects of query optimization (i.e. supported operators, join orders and statistics).
Secondly, and on a more pragmatic note Potsgres was the first database system that was supported by PostBOUND and therefore
a lot of the original Postgres interfaces eventually evolved into the more abstract database-independent interfaces.
"""

from __future__ import annotations

import collections
import concurrent
import concurrent.futures
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
import threading
import time
import tomllib
import warnings
from collections import UserString
from collections.abc import Callable, Iterable, Sequence, Sized
from dataclasses import dataclass
from multiprocessing import connection as mp_conn
from pathlib import Path
from typing import Any, Literal, Optional, TextIO, get_args, overload

import psycopg
import psycopg.rows
import psycopg.types.datetime as psycopg_datetime

from . import qal, transform, util
from ._core import (
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
from ._hints import (
    HintType,
    JoinTree,
    PhysicalOperatorAssignment,
    PlanParameterization,
    jointree_from_plan,
    operators_from_plan,
    parameters_from_plan,
)
from ._qep import QueryPlan, SortKey
from .db import (
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
    OptimizerInterface,
    QueryCacheWarning,
    ResultSet,
    UnsupportedDatabaseFeatureError,
    simplify_result_set,
)
from .qal import (
    Explain,
    Hint,
    Limit,
    SqlQuery,
)
from .util import StateError, Version, jsondict

_SignificantPostgresSettings = {
    # Resource consumption settings (see https://www.postgresql.org/docs/current/runtime-config-resource.html)
    # Memory
    "shared_buffers",
    "huge_pages",
    "huge_page_size",
    "temp_buffers",
    "max_prepared_transactions",
    "work_mem",
    "hash_mem_multiplier",
    "maintenance_work_mem",
    "autovacuum_work_mem",
    "vacuum_buffer_usage_limit",
    "logical_decoding_work_mem",
    "max_stack_depth",
    "shared_memory_type",
    "dynamic_shared_memory_type",
    "min_dynamic_shared_memory",
    # Disk
    "temp_file_limit",
    # Kernel Resource Usage
    "max_files_per_process",
    # Cost-based Vacuum Delay
    "vacuum_cost_delay",
    "vacuum_cost_page_hit",
    "vacuum_cost_page_miss",
    "vacuum_cost_page_dirty",
    "vacuum_cost_limit",
    # Background Writer
    "bgwriter_delay",
    "bgwriter_lru_maxpages",
    "bgwriter_lru_multiplier",
    "bgwriter_flush_after",
    # Asynchronous Behavior
    "backend_flush_after",
    "effective_io_concurrency",
    "maintenance_io_concurrency",
    "max_worker_processes",
    "max_parallel_workers_per_gather",
    "max_parallel_maintenance_workers",
    "max_parallel_workers",
    "parallel_leader_participation",
    "old_snapshot_threshold",
    # Query Planning Settings (see https://www.postgresql.org/docs/current/runtime-config-query.html)
    # Planner Method Configuration
    "enable_async_append",
    "enable_bitmapscan",
    "enable_gathermerge",
    "enable_hashagg",
    "enable_hashjoin",
    "enable_incremental_sort",
    "enable_indexscan",
    "enable_indexonlyscan",
    "enable_material",
    "enable_memoize",
    "enable_mergejoin",
    "enable_nestloop",
    "enable_parallel_append",
    "enable_parallel_hash",
    "enable_partition_pruning",
    "enable_partitionwise_join",
    "enable_partitionwise_aggregate",
    "enable_presorted_aggregate",
    "enable_seqscan",
    "enable_sort",
    "enable_tidscan",
    # Planner Cost Constants
    "seq_page_cost",
    "random_page_cost",
    "cpu_tuple_cost",
    "cpu_index_tuple_cost",
    "cpu_operator_cost",
    "parallel_setup_cost",
    "parallel_tuple_cost",
    "min_parallel_table_scan_size",
    "min_parallel_index_scan_size",
    "effective_cache_size",
    "jit_above_cost",
    "jit_inline_above_cost",
    "jit_optimize_above_cost",
    # Genetic Query Optimizer
    "geqo",
    "geqo_threshold",
    "geqo_effort",
    "geqo_pool_size",
    "geqo_generations",
    "geqo_selection_bias",
    "geqo_seed",
    # Other Planner Options
    "default_statistics_target",
    "constraint_exclusion",
    "cursor_tuple_fraction",
    "from_collapse_limit",
    "jit",
    "join_collapse_limit",
    "plan_cache_mode",
    "recursive_worktable_factor"
    # Automatic Vacuuming (https://www.postgresql.org/docs/current/runtime-config-autovacuum.html)
    "autovacuum",
    "autovacuum_max_workers",
    "autovacuum_naptime",
    "autovacuum_threshold",
    "autovacuum_insert_threshold",
    "autovacuum_analyze_threshold",
    "autovacuum_scale_factor",
    "autovacuum_analyze_scale_factor",
    "autovacuum_freeze_max_age",
    "autovacuum_multixact_freeze_max_age",
    "autovacuum_cost_delay",
    "autovacuum_cost_limit",
}
"""Postgres settings that are relevant to many PostBOUND workflows.

These settings can influence performance measurements of different benchmarks. Therefore, we want to make their values
transparent in order to assess the results.

As a rule of thumb we include settings from three major categories: resource consumption (e.g. size of shared buffers),
optimizer settings (e.g. enable operators) and auto vacuum. The final category is required because it determines how good the
statistics are once a new database dump has been loaded or a data shift has been simulated. For all of these categories we
include all settings, even if they are not important right now to the best of our knowledge. This is done to prevent tedious
debugging if setting is later found to be indeed important: if the category to which it belongs is present in our "significant
settings", it is guaranteed to be monitored.

Most notably settings regarding replication, logging and network settings are excluded, as well as settings regarding locking.
This is done because PostBOUNDs database abstraction assumes read-only workloads with a single query at a time. If data shifts
are simulated, these are supposed to be happen strictly before or after a read-only workload is executed and benchmarked.

All settings are up-to-date as of Postgres version 16.
"""

_RuntimeChangeablePostgresSettings = {
    setting for setting in _SignificantPostgresSettings
} - {
    "autovacuum_max_workers",
    "autovacuum_naptime",
    "autovacuum_threshold",
    "autovacuum_insert_threshold",
    "autovacuum_analyze_threshold",
    "autovacuum_scale_factor",
    "autovacuum_analyze_scale_factor",
    "autovacuum_freeze_max_age",
    "autovacuum_multixact_freeze_max_age",
    "autovacuum_cost_delay",
    "autovacuum_cost_limit",
    "autovacuum_work_mem",
    "bgwriter_delay",
    "bgwriter_lru_maxpages",
    "bgwriter_lru_multiplier",
    "bgwriter_flush_after",
    "dynamic_shared_memory_type",
    "huge_pages",
    "huge_page_size",
    "max_files_per_process",
    "max_prepared_transactions",
    "max_worker_processes",
    "min_dynamic_shared_memory",
    "old_snapshot_threshold",
    "shared_buffers",
    "shared_memory_type",
}
"""These are exactly those settings from `_SignificantPostgresSettings` that can be changed at runtime."""


class PostgresSetting(str):
    """Model for a single Postgres configuration such as *SET enable_nestloop = 'off';*.

    This setting can be used directly as a replacement where a string is expected, or its different components can be accessed
    via the `parameter` and `value` attribute.

    Parameters
    ----------
    parameter : str
        The name of the setting
    value : object
        The setting's current or desired value
    """

    def __init__(self, parameter: str, value: object) -> None:
        self._param = parameter
        self._val = value

    def __new__(cls, parameter: str, value: object):
        value = "on" if value is True else "off" if value is False else value
        return super().__new__(cls, f"SET {parameter} = '{value}';")

    __match_args__ = ("parameter", "value")

    @property
    def parameter(self) -> str:
        """Gets the name of the setting.

        Returns
        -------
        str
            The name
        """
        return self._param

    @property
    def value(self) -> object:
        """Gets the current or desired value of the setting.

        Returns
        -------
        object
            The raw, i.e. un-escaped value of the setting.
        """
        return self._val

    def update(self, value: object) -> PostgresSetting:
        """Creates a new setting with the same name but a different value.

        Parameters
        ----------
        value : object
            The new value

        Returns
        -------
        PostgresSetting
            The new setting
        """
        return PostgresSetting(self.parameter, value)

    def __getnewargs__(self):
        return (self.parameter, self.value)


class PostgresConfiguration(collections.UserString):
    """Model for a collection of different postgres settings that form a complete server configuration.

    Each configuration is build of indivdual `PostgresSetting` objects. The configuration can be used directly as a replacement
    when a string is expected, or its different settings can be accessed individually - either through the accessor methods, or
    by using a dict-like syntax: calling ``config[setting]`` with a string setting value will provide the matching
    `PostgresSetting`. Since the configuration also subclasses string, the precise behavior of `__getitem__` depends on the
    argument type: string arguments provide settings whereas integer arguments result in specific characters. All other string
    methods are implemented such that the normal string behavior is retained. All additional behavior is part of new methods.

    Parameters
    ----------
    settings : Iterable[PostgresSetting]
        The settings that form the configuration.

    Warnings
    --------
    Notice that while the configuration is a *UserString*, pyscopg currently does not support executing the configuration, i.e.
    executing ``cursor.execute(config)`` will not work. Instead, the configuration has to be manually converted into a string
    first by calling *str* as in ``cursor.execute(str(config))``. This also applies to the `execute_query()` method of the
    `PostgresInterface` class, since it uses psycopg under the hood.
    """

    @staticmethod
    def load(*args, **kwargs) -> PostgresConfiguration:
        """Generates a new configuration based on (setting name, value) pairs.

        Parameters
        ----------
        args
            Ready-to-use `PostgresSetting` objects
        kwargs
            Additional settings

        Returns
        -------
        PostgresConfiguration
            The configuration
        """
        return PostgresConfiguration(
            list(args) + [PostgresSetting(key, val) for key, val in kwargs.items()]
        )

    def __init__(self, settings: Iterable[PostgresSetting]) -> None:
        self._settings: dict[str, PostgresSetting] = {
            setting.parameter: setting for setting in settings
        }
        super().__init__(self._format())

    @property
    def settings(self) -> Sequence[PostgresSetting]:
        """Gets the settings that are part of the configuration.

        Returns
        -------
        Sequence[PostgresSetting]
            The settings in the order in which they were originally specified.
        """
        return list(self._settings.values())

    def parameters(self) -> Sequence[str]:
        """Provides all setting names that are specified in this configuration.

        Returns
        -------
        Sequence[str]
            The setting names in the order in which they were orignally specified.
        """
        return list(self._settings.keys())

    def add(
        self,
        setting: PostgresSetting | str | None = None,
        value: object = None,
        **kwargs,
    ) -> PostgresConfiguration:
        """Creates a new configuration with additional settings.

        The setting can be supplied either as a `PostgresSetting` object or as a key-value pair.
        The latter case allows both positional arguments, as well as as keyword arguments.

        Parameters
        ----------
        setting : PostgresSetting | str
            The setting to add. This can either be a readily created `PostgresSetting` object or a string that will be used as
            the setting name. In the latter case, the `value` has to be supplied as well.
        value : object
            The value of the setting. This is only used if `setting` is a string.
        kwargs
            If the setting is not specified as a string, nor as a `PostgresSetting` object, it has to be specified as keyword
            arguments. The keyword argument names are used as the setting names, the values are used as the setting values.

        Returns
        -------
        PostgresConfiguration
            The updated configuration. The original config is not modified.
        """
        if isinstance(setting, str):
            setting = PostgresSetting(setting, value)

        target_settings = dict(self._settings)
        if isinstance(setting, PostgresSetting):
            target_settings[setting.parameter] = setting
        else:
            settings = {key: PostgresSetting(key, val) for key, val in kwargs.items()}
            target_settings.update(settings)

        return PostgresConfiguration(target_settings.values())

    def remove(self, setting: PostgresSetting | str) -> PostgresConfiguration:
        """Creates a new configuration without a specific setting.

        Parameters
        ----------
        setting : PostgresSetting
            The setting to remove

        Returns
        -------
        PostgresConfiguration
            The updated configuration. The original config is not modified.
        """
        parameter = (
            setting.parameter if isinstance(setting, PostgresSetting) else setting
        )
        target_settings = dict(self._settings)
        target_settings.pop(parameter, None)
        return PostgresConfiguration(target_settings.values())

    def update(
        self, setting: PostgresSetting | str, value: object
    ) -> PostgresConfiguration:
        """Creates a new configuration with an updated setting.

        Parameters
        ----------
        setting : PostgresSetting | str
            The setting to update. This can either be the raw setting name, or a `PostgresSetting` object. In either case,
            the updated value has to be supplied via the `value` parameter. (When supplying a `PostgresSetting`, only its
            name is used.)
        value : object
            The updated value of the setting.

        Returns
        -------
        PostgresConfiguration
            The updated configuration. The original config is not modified.
        """
        match setting:
            case str():
                setting = PostgresSetting(setting, value)
            case PostgresSetting(name, _):
                setting = PostgresSetting(name, value)

        target_settings = dict(self._settings)
        target_settings[setting.parameter] = setting

        return PostgresConfiguration(target_settings.values())

    def as_dict(self) -> dict[str, object]:
        """Provides all settings as setting name -> setting value mappings.

        Returns
        -------
        dict[str, object]
            The settings. Changes to this dictionary will not be reflected in the configuration object.
        """
        return dict(self._settings)

    def _format(self) -> str:
        """Provides the string representation of the configuration.

        Returns
        -------
        str
            The string representation
        """
        return "\n".join([str(setting) for setting in self.settings])

    def __getitem__(self, index):
        if isinstance(index, str):
            return self._settings[index]
        return super().__getitem__(index)

    def __setitem__(self, key: str, value: str | PostgresSetting) -> None:
        if isinstance(value, str):
            value = PostgresSetting(key, value)
        self._settings[key] = value
        self.data = self._format()


class PostgresConfigInterface:
    """A thin wrapper that provides read-only access to Postgres configuration settings using __getitem__ syntax."""

    def __init__(self, pg_instance: PostgresInterface) -> None:
        self._pg = pg_instance

    def __getitem__(self, key: str) -> Any:
        return self._pg.execute_query(f"SHOW {key};", cache_enabled=False, raw=False)


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
            return datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)
        elif data == b"-infinity":
            return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

        try:
            return super().load(data)
        except psycopg.errors.DataError as e:
            if not e.args:
                raise e
            msg = e.args[0]
            if not isinstance(msg, str) or not msg.startswith("timestamp too large"):
                raise e
            return datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)


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
        cache_enabled: bool = False,
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

        self._timeout_executor = TimeoutQueryExecutor(self)
        self._last_query_runtime = math.nan

        super().__init__(system_name, cache_enabled=cache_enabled)

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
        plan: Optional[QueryPlan] = None,
        join_order: Optional[JoinTree] = None,
        physical_operators: Optional[PhysicalOperatorAssignment] = None,
        plan_parameters: Optional[PlanParameterization] = None,
        cache_enabled: Optional[bool] = None,
        raw: bool = False,
        timeout: Optional[float] = None,
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
            return self._timeout_executor.execute_query(
                query, timeout=timeout, cache_enabled=cache_enabled, raw=raw
            )

        cache_enabled = cache_enabled or (cache_enabled is None and self._cache_enabled)
        if isinstance(query, UserString):
            query = str(query)
        elif isinstance(query, SqlQuery):
            # If there are any preparatory statements we must execute them now. If we would simply include them in the query,
            # psycopg would use the first statement to determine the result type of the entire query. If this were a SET
            # statement, no results could be fetched. _apply_preparatory_statements takes care of these and gives a cleaned-up
            # query to execute.
            query = _apply_preparatory_statements(query, cur=self._cursor)
            query = self._hinting_backend.format_query(query)

        if cache_enabled and query in self._query_cache:
            query_result = self._query_cache[query]
            return query_result if raw else simplify_result_set(query_result)

        try:
            start_time = time.perf_counter_ns()
            self._cursor.execute(query)  # type: ignore
            end_time = time.perf_counter_ns()
            self._last_query_runtime = (
                end_time - start_time
            ) / 10**9  # convert to seconds

            query_result = self._cursor.fetchall()
            if cache_enabled:
                self._inflate_query_cache()
                self._query_cache[query] = query_result
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
            raise DatabaseServerError(msg, e)
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
            raise DatabaseUserError(msg, e)

        return query_result if raw else simplify_result_set(query_result)

    def execute_with_timeout(
        self, query: SqlQuery | str, timeout: float = 60.0
    ) -> Optional[ResultSet]:
        try:
            result = self.execute_query(
                query, timeout=timeout, cache_enabled=False, raw=True
            )
            return result
        except TimeoutError:
            return None

    def last_query_runtime(self) -> float:
        return self._last_query_runtime

    def time_query(self, query: SqlQuery, *, timeout: Optional[float] = None) -> float:
        self.execute_query(query, cache_enabled=False, raw=True, timeout=timeout)
        return self.last_query_runtime()

    def optimizer(self) -> PostgresOptimizer:
        return PostgresOptimizer(self)

    def database_name(self) -> str:
        self._cursor.execute("SELECT CURRENT_DATABASE();")
        db_name = self._cursor.fetchone()[0]  # type: ignore
        return db_name

    def database_system_version(self) -> Version:
        self._cursor.execute("SELECT VERSION();")
        version_string = self._cursor.fetchone()[0]  # type: ignore
        version_match = _PGVersionPattern.match(version_string)
        if not version_match:
            raise RuntimeError(
                f"Could not extract Postgres version from string '{version_string}'"
            )
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

    def logfile(self) -> Optional[Path]:
        """Get the log file of the (local) Postgres server."""
        proc_path = Path(f"/proc/{self.backend_pid()}/fd/1")
        if not proc_path.exists() or not proc_path.is_symlink():
            return None
        return proc_path.resolve()

    def describe(self) -> jsondict:
        base_info: dict[str, Any] = {
            "system_name": self.database_system_name(),
            "system_version": self.database_system_version(),
            "database": self.database_name(),
            "statistics_settings": {
                "emulated": self._db_stats.emulated,
                "cache_enabled": self._db_stats.cache_enabled,
            },
            "hinting_mode": self._hinting_backend.describe(),
            "query_cache": self.cache_enabled,
        }
        self._cursor.execute("SELECT name, setting FROM pg_settings")
        system_settings = self._cursor.fetchall()
        base_info["system_settings"] = {
            setting: value
            for setting, value in system_settings
            if setting in _SignificantPostgresSettings
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
                    "n_rows": self.statistics().total_rows(table, emulated=True),
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

    def cursor(self) -> psycopg.Cursor:
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
        tables: Optional[TableReference | Iterable[TableReference]] = None,
        *more_tables: TableReference,
        exclude_table_pages: bool = False,
        include_primary_index: bool = True,
        include_secondary_indexes: bool = True,
    ) -> None:
        """Prepares the Postgres buffer pool with tuples from specific tables.

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
        if tables is not None:
            tables = list(util.enlist(tables))
        else:
            tables = []
        tables += list(more_tables)

        if not tables:
            return
        tables = set(
            tab.full_name for tab in tables if not tab.virtual
        )  # eliminate duplicates if tables are selected multiple times

        table_indexes = (
            [self._fetch_index_relnames(tab) for tab in tables]
            if include_primary_index or include_secondary_indexes
            else []
        )
        indexes_to_prewarm = {
            idx
            for idx, primary in util.flatten(table_indexes)
            if (primary and include_primary_index)
            or (not primary and include_secondary_indexes)
        }
        tables = (
            indexes_to_prewarm if exclude_table_pages else tables | indexes_to_prewarm
        )
        if not tables:
            return

        prewarm_invocations = [f"pg_prewarm('{tab}')" for tab in tables]
        prewarm_text = ", ".join(prewarm_invocations)
        prewarm_query = f"SELECT {prewarm_text}"

        self._cursor.execute(prewarm_query)  # type: ignore

    def cooldown_tables(
        self,
        tables: Optional[TableReference | Iterable[TableReference]] = None,
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
        if tables is not None:
            tables = list(util.enlist(tables))
        else:
            tables = []
        tables += list(more_tables)

        if not tables:
            return
        tables = set(
            tab.full_name for tab in tables
        )  # eliminate duplicates if tables are selected multiple times

        table_indexes = (
            [self._fetch_index_relnames(tab) for tab in tables]
            if include_primary_index or include_secondary_indexes
            else []
        )
        indexes_to_cooldown = {
            idx
            for idx, primary in util.flatten(table_indexes)
            if (primary and include_primary_index)
            or (not primary and include_secondary_indexes)
        }
        tables = (
            indexes_to_cooldown if exclude_table_pages else tables | indexes_to_cooldown
        )
        if not tables:
            return

        cooldown_invocations = [f"pg_cooldown('{tab}')" for tab in tables]
        cooldown_text = ", ".join(cooldown_invocations)
        cooldown_query = f"SELECT {cooldown_text}"

        self._cursor.execute(cooldown_query)  # type: ignore

    def current_configuration(
        self, *, runtime_changeable_only: bool = False
    ) -> PostgresConfiguration:
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
        allowed_settings = (
            _RuntimeChangeablePostgresSettings
            if runtime_changeable_only
            else _SignificantPostgresSettings
        )
        configuration = {
            setting: value
            for setting, value in system_settings
            if setting in allowed_settings
        }
        return PostgresConfiguration.load(**configuration)

    def apply_configuration(
        self, configuration: PostgresConfiguration | PostgresSetting | str
    ) -> None:
        """Changes specific configuration parameters of the Postgres server or current connection.

        Parameters
        ----------
        configuration : PostgresConfiguration | PostgresSetting | str
            The desired setting values. If a string is supplied directly, it already has to be a valid setting update such as
            *SET geqo = FALSE;*.
        """
        if (
            isinstance(configuration, PostgresSetting)
            and configuration.parameter not in _RuntimeChangeablePostgresSettings
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
                if setting.parameter in _RuntimeChangeablePostgresSettings:
                    supported_settings.append(setting)
                else:
                    unsupported_settings.append(setting.parameter)
            if unsupported_settings:
                warnings.warn(
                    f"Skipping configuration settings {unsupported_settings} "
                    "because they cannot be changed at runtime",
                    stacklevel=2,
                )
            configuration = str(PostgresConfiguration(supported_settings))

        self._cursor.execute(configuration)  # type: ignore

    def has_extension(
        self, extension_name: str, *, is_shared_object: bool = True
    ) -> bool:
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
                raise RuntimeError(
                    f"Plaform '{sys.platform}' is not supported by extension check."
                )

        if is_shared_object or extension_name in ("pg_hint_plan", "pg_lab"):
            shared_object_name = (
                f"{extension_name}{lib_suffix}"
                if not extension_name.endswith(lib_suffix)
                else extension_name
            )
            loaded_shared_objects = util.system.open_files(
                self._connection.info.backend_pid
            )
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
        self._connection.autocommit = (
            True  # pg_hint_plan hinting backend currently relies on autocommit!
        )
        self._connection.prepare_threshold = None

        self._connection.adapters.register_dumper(datetime.date, _PsycopgDateDumper)
        self._connection.adapters.register_dumper(
            datetime.datetime, _PsycopgTimestampDumper
        )
        self._connection.adapters.register_dumper(
            datetime.datetime, _PsycopgTimestampTzDumper
        )
        self._connection.adapters.register_loader("date", _PsycopgDateLoader)
        self._connection.adapters.register_loader("timestamp", _PsycopgTimestampLoader)
        self._connection.adapters.register_loader(
            "timestamptz", _PsycopgTimestampTzLoader
        )
        self._connection.adapters.register_loader("interval", _PsycopgIntervalLoader)

        self._cursor: psycopg.Cursor = self._connection.cursor()
        return self.backend_pid()

    def _apply_query_hints(
        self,
        query: str | SqlQuery,
        plan: Optional[QueryPlan],
        *,
        join_order: Optional[JoinTree],
        physical_operators: Optional[PhysicalOperatorAssignment],
        plan_parameters: Optional[PlanParameterization],
    ) -> str | SqlQuery:
        if isinstance(query, str):
            # XXX: should we rather parse the query here?
            return query

        has_hint = any(
            hint is not None
            for hint in (plan, join_order, physical_operators, plan_parameters)
        )
        if not has_hint:
            return query

        return self.hinting().generate_hints(
            query,
            plan,
            join_order=join_order,
            physical_operators=physical_operators,
            plan_parameters=plan_parameters,
        )

    def _fetch_index_relnames(
        self, table: TableReference | str
    ) -> Iterable[tuple[str, bool]]:
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

    def _assert_active_extension(
        self, extension_name: str, *, is_shared_object: bool = False
    ) -> None:
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
        extension_is_active = self.has_extension(
            extension_name, is_shared_object=is_shared_object
        )
        if not extension_is_active:
            raise StateError(
                f"Extension '{extension_name}' is not active in database '{self.database_name()}'"
            )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.connect_string == other.connect_string
        )

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
        self._tables: set[TableReference] = set()

    def tables(
        self, *, include_system_tables: bool = False, schema: str = "public"
    ) -> set[TableReference]:
        if self._tables:
            return self._tables

        query_template = textwrap.dedent("""
                                         SELECT table_name
                                         FROM information_schema.tables
                                         WHERE table_catalog = %s AND table_schema = %s""")
        self._db.cursor().execute(query_template, (self._db.database_name(), schema))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None

        tables = {TableReference(row[0]) for row in result_set}
        if not include_system_tables:
            tables = {tab for tab in tables if not tab.full_name.startswith("pg_")}
        self._tables = tables
        return self._tables

    def lookup_column(
        self,
        column: ColumnReference | str,
        candidate_tables: Iterable[TableReference],
        *,
        expect_match: bool = False,
    ) -> Optional[TableReference]:
        if not isinstance(candidate_tables, Sized):
            candidate_tables = list(candidate_tables)
        candidate_tables = (
            set(candidate_tables)
            if len(candidate_tables) > 5
            else list(candidate_tables)
        )
        column = column.name if isinstance(column, ColumnReference) else column
        lower_col = column.lower()

        for table in candidate_tables:
            table_columns = self._fetch_columns(table)
            if column in table_columns or lower_col in table_columns:
                return table

        if not expect_match:
            return None
        table_strings = [table.qualified_name() for table in candidate_tables]
        raise ValueError(
            f"Column '{column}' not found in candidate tables {table_strings}"
        )

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

        self._db.cursor().execute(
            query_template, (column.table.full_name, column.name, schema)
        )
        result_set = self._db.cursor().fetchall()
        assert result_set is not None

        return {row[0] for row in result_set}

    def indexed_column(
        self, index: str, *, schema: str = "public"
    ) -> Optional[BoundColumnReference]:
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

        self._db.cursor().execute(
            query_template, (column.table.full_name, column.name, schema)
        )
        result_set = self._db.cursor().fetchall()
        assert result_set is not None

        return {
            BoundColumnReference(row[1], TableReference(row[0])) for row in result_set
        }

    def datatype(self, column: ColumnReference) -> str:
        if not column.table:
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)
        schema = column.table.schema or "public"
        query_template = textwrap.dedent("""
            SELECT data_type FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s AND table_schema = %s""")

        self._db.cursor().execute(
            query_template, (column.table.full_name, column.name, schema)
        )
        result_set = self._db.cursor().fetchone()
        assert result_set

        return result_set[0]

    def is_nullable(self, column: ColumnReference) -> bool:
        if not column.table:
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)
        schema = column.table.schema or "public"
        query_tempalte = textwrap.dedent("""
            SELECT is_nullable = 'YES' FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s AND table_schema = %s""")

        self._db.cursor().execute(
            query_tempalte, (column.table.full_name, column.name, schema)
        )
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
        query_template = "SELECT column_name FROM information_schema.columns WHERE table_name = %s AND table_schema = %s"

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

    def __init__(
        self,
        postgres_db: PostgresInterface,
        *,
        emulated: bool = True,
        enable_emulation_fallback: bool = True,
        cache_enabled: Optional[bool] = True,
    ) -> None:
        super().__init__(
            postgres_db,
            emulated=emulated,
            enable_emulation_fallback=enable_emulation_fallback,
            cache_enabled=cache_enabled,
        )

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
        columns: Optional[ColumnReference | Iterable[ColumnReference]] = None,
        *,
        tables: Optional[TableReference | Iterable[TableReference]] = None,
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
            tables = [
                tab
                for tab in self._db.schema().tables()
                if not self._db.schema().is_view(tab)
            ]
        if not columns and tables:
            tables = util.enlist(tables)
            columns = util.set_union(self._db.schema().columns(tab) for tab in tables)

        assert columns is not None
        columns: Iterable[ColumnReference] = util.enlist(columns)
        columns_map: dict[TableReference, list[str]] = util.dicts.generate_multi(
            (col.table, col.name) for col in columns
        )
        distinct_values: dict[ColumnReference, int] = {}

        if perfect_mcv or perfect_n_distinct:
            for column in columns:
                util.logging.print_if(
                    verbose,
                    util.timestamp(),
                    ":: Now preparing column",
                    column,
                    use_stderr=True,
                )
                raw = self.num_distinct(column, emulated=True, cache_enabled=True)
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
                self._db.cursor().execute(stats_target_query)

        columns_str = {
            table: ", ".join(col for col in columns)
            for table, columns in columns_map.items()
        }
        tables_and_columns = ", ".join(
            f"{table.full_name}({cols})" for table, cols in columns_str.items()
        )

        util.logging.print_if(
            verbose,
            util.timestamp(),
            ":: Now analyzing columns",
            tables_and_columns,
            use_stderr=True,
        )
        query_template = f"ANALYZE {tables_and_columns}"
        self._db.cursor().execute(query_template)

        for column, n_distinct in distinct_values.items():
            assert column.table is not None, "Unbound table"
            distinct_update_query = textwrap.dedent(f"""
                                                    ALTER TABLE {column.table.full_name}
                                                    ALTER COLUMN {column.name}
                                                    SET (n_distinct = {n_distinct});
                                                    """)
            self._db.cursor().execute(distinct_update_query)

    def _retrieve_total_rows_from_stats(self, table: TableReference) -> Optional[int]:
        schema = table.schema or "public"
        count_query = "SELECT reltuples FROM pg_class WHERE oid = %s::regclass AND relnamespace = %s::regnamespace"
        self._db.cursor().execute(count_query, (table.full_name, schema))
        result_set = self._db.cursor().fetchone()
        if not result_set:
            return None
        count = result_set[0]
        return int(count)

    def _retrieve_distinct_values_from_stats(
        self, column: BoundColumnReference
    ) -> Optional[int]:
        schema = column.table.schema or "public"
        dist_query = """
            SELECT n_distinct
            FROM pg_stats
            WHERE tablename = %s
                AND attname = %s
                AND schemaname = %s
        """
        self._db.cursor().execute(
            dist_query, (column.table.full_name, column.name, schema)
        )
        result_set = self._db.cursor().fetchone()
        if not result_set:
            return None
        dist_values = result_set[0]

        # interpreting the n_distinct column is difficult, since different value ranges indicate different things
        # (see https://www.postgresql.org/docs/current/view-pg-stats.html)
        # If the value is >= 0, it represents the actual (approximated) number of distinct non-zero values in the
        # column.
        # If the value is < 0, it represents 'the negative of the number of distinct values divided by the number of
        # rows'. Therefore, we have to correct the number of distinct values manually in this case.
        if dist_values >= 0:
            return int(dist_values)

        # correct negative values
        n_rows = self._retrieve_total_rows_from_stats(column.table)
        assert n_rows is not None, "Could not retrieve total row count for table"

        return int(-1 * n_rows * dist_values)

    def _retrieve_min_max_values_from_stats(
        self, column: BoundColumnReference
    ) -> Optional[tuple[Any, Any]]:
        # Postgres does not keep track of min/max values, so we need to determine them manually
        if not self.enable_emulation_fallback:
            raise UnsupportedDatabaseFeatureError(self._db, "min/max value statistics")
        return self._calculate_min_max_values(column, cache_enabled=True)

    def _retrieve_most_common_values_from_stats(
        self, column: BoundColumnReference, k: int | None
    ) -> Sequence[tuple[Any, int]]:
        table = column.table
        schema = table.schema or "public"
        # Postgres stores the Most common values in a column of type anyarray (since in this column, many MCVs from
        # many different tables and data types are present). However, this type is not very convenient to work on.
        # Therefore, we first need to convert the anyarray to an array of the actual attribute type.
        attribute_converter = self._array_cast(column)

        # now, load the most frequent values. Since the frequencies are expressed as a fraction of the total number of
        # rows, we need to multiply this number again to obtain the true number of occurrences
        mcv_query = """
            SELECT UNNEST(most_common_vals::text::{conv}),
                UNNEST(most_common_freqs) * (SELECT reltuples
                                             FROM pg_class
                                             WHERE oid = %s::regclass
                                             AND relnamespace = %s::regnamespace)
            FROM pg_stats
            WHERE tablename = %s
                AND attname = %s
                AND schemaname = %s
        """
        mcv_query = mcv_query.format(conv=attribute_converter)

        # NB: we have to repeat a few parameters here. Unfortunately, it seems that psycopg
        # does not support casts for named parameters - %(tab)s::regclass does not work
        self._db.cursor().execute(
            mcv_query, (table.full_name, schema, table.full_name, column.name, schema)
        )
        result_set = self._db.cursor().fetchall()
        assert result_set is not None

        if not result_set:
            return []
        elif k is None or k <= 0:
            return result_set
        else:
            return result_set[:k]

    def _retrieve_histogram_from_stats(
        self, column: BoundColumnReference, *, interpolation: HistogramApproximation
    ) -> Optional[Histogram]:
        attribute_converter = self._array_cast(column)
        schema = column.table.schema or "public"

        query_template = """
            SELECT UNNEST(histogram_bounds::text::{conv})
            FROM pg_stats
            WHERE schemaname = %s AND tablename = %s AND attname = %s
            """.format(conv=attribute_converter)

        self._db.cursor().execute(
            query_template, (schema, column.table.full_name, column.name)
        )
        result_set = self._db.cursor().fetchall()
        if not result_set:
            return None
        bounds = [row[0] for row in result_set]

        n_rows = self._retrieve_total_rows_from_stats(column.table)
        assert n_rows

        bucket_freq = n_rows // len(bounds)

        mcvs = self._retrieve_most_common_values_from_stats(column, k=None)

        normalized_bounds = []
        normalized_frequencies = []
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
                assert cur_mcv_val is not None
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
                assert cur_mcv_val < cur_hist
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

        self._db.cursor().execute(
            attribute_query, (column.table.full_name, column.name)
        )
        result_set = self._db.cursor().fetchone()
        assert result_set

        attribute_dtype = result_set[0]
        converter = _DTypeArrayConverters.get(attribute_dtype)
        if not converter:
            raise ValueError(
                "Cannot cast column array of type {attribute_dtype} - no converter found."
            )
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


class PostgresExplainClause(Explain):
    """A specialized *EXPLAIN* clause implementation to handle Postgres custom syntax for query plans.

    If *ANALYZE* is enabled, this also retrieves information about shared buffer usage (page hits and disk reads).

    Parameters
    ----------
    original_clause : Explain
        The actual *EXPLAIN* clause. The new explain clause acts as a decorator around the original clause.
    """

    def __init__(self, original_clause: Explain) -> None:
        super().__init__(original_clause.analyze, original_clause.target_format)

    def __str__(self) -> str:
        explain_args = "(SETTINGS, "
        if self.analyze:
            explain_args += "ANALYZE, BUFFERS, "
        explain_args += f"FORMAT {self.target_format})"
        return f"EXPLAIN {explain_args}"


class PostgresLimitClause(Limit):
    """A specialized *LIMIT* clause implementation to handle Postgres custom syntax for limits / offsets

    Parameters
    ----------
    original_clause : Limit
        The actual *LIMIT* clause. The new limit clause acts as a decorator around the original clause.
    """

    def __init__(self, original_clause: Limit) -> None:
        super().__init__(
            limit=original_clause.limit,
            offset=original_clause.offset,
            fetch_direction=original_clause.fetch_direction,
        )

    def __str__(self) -> str:
        if self.fetch_direction != "first":
            return super().__str__()

        if self.limit and self.offset:
            return f"LIMIT {self.limit} OFFSET {self.offset}"
        elif self.limit:
            return f"LIMIT {self.limit}"
        elif self.offset:
            return f"OFFSET {self.offset}"
        else:
            return ""


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
    join_order: Optional[JoinTree],
    phys_ops: Optional[PhysicalOperatorAssignment],
    plan_params: Optional[PlanParameterization],
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
                    hints.append(
                        f"Parallel({tab.identifier()} {join.parallel_workers} hard)"
                    )

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
    join_order: Optional[JoinTree],
    phys_ops: Optional[PhysicalOperatorAssignment],
    plan_params: Optional[PlanParameterization],
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
        has_dangling_worker_hints = any(
            intermediate not in phys_ops
            for intermediate in plan_params.parallel_workers
        )
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
            mode = (
                "sequential"
                if plan_params.execution_mode == "sequential"
                else "parallel"
            )
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
        hints: list[str] = ["Config(plan_mode=full)", f"JoinOrder({join_order})"]
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
                    level=level + 1,
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
        plan: Optional[QueryPlan] = None,
        *,
        join_order: Optional[JoinTree] = None,
        physical_operators: Optional[PhysicalOperatorAssignment] = None,
        plan_parameters: Optional[PlanParameterization] = None,
    ) -> SqlQuery:
        self._assert_active_backend()

        adapted_query = query
        if adapted_query.explain and not isinstance(
            adapted_query.explain, PostgresExplainClause
        ):
            adapted_query = transform.replace_clause(
                adapted_query, PostgresExplainClause(adapted_query.explain)
            )
        if adapted_query.limit_clause and not isinstance(
            adapted_query.limit_clause, PostgresLimitClause
        ):
            adapted_query = transform.replace_clause(
                adapted_query, PostgresLimitClause(adapted_query.limit_clause)
            )

        has_param = any(
            param is not None
            for param in (join_order, physical_operators, plan_parameters)
        )
        if plan is not None and has_param:
            raise ValueError(
                "Can only hint an entire query plan, or individual parts, not both."
            )

        match self._backend:
            case "pg_hint_plan":
                if plan is not None:
                    join_order = jointree_from_plan(plan)
                    physical_operators = operators_from_plan(
                        plan, include_workers=False
                    )
                    plan_parameters = parameters_from_plan(
                        plan, target_cardinality="actual", fallback_estimated=True
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

        query = transform.add_clause(adapted_query, hints)
        return query

    def format_query(self, query: SqlQuery) -> str:
        if query.explain:
            query = transform.replace_clause(
                query, PostgresExplainClause(query.explain)
            )
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
        found_pg = any(
            candidate.lower().startswith("postgres")
            for candidate in pg_candidates.stdout.split()
        )

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
                self._postgres_db.debug, "Using pg_lab hinting backend", file=sys.stderr
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
            raise ValueError(
                f"No supported hinting backend found for backend with PID {connection_pid}"
            )

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
        raw_query_plan: list = self._pg_instance.execute_query(
            query, cache_enabled=False
        )
        query_plan = PostgresExplainPlan(raw_query_plan[0])
        return query_plan.as_qep()

    @overload
    def analyze_plan(self, query: SqlQuery) -> QueryPlan: ...

    @overload
    def analyze_plan(
        self, query: SqlQuery, *, timeout: float
    ) -> Optional[QueryPlan]: ...

    @overload
    def analyze_plan(self, query: SqlQuery, *, timeout: Literal[None]) -> QueryPlan: ...

    def analyze_plan(
        self, query: SqlQuery, *, timeout: Optional[float] = None
    ) -> Optional[QueryPlan]:
        query = transform.as_explain_analyze(query)

        try:
            raw_query_plan: dict = self._pg_instance.execute_query(
                query, cache_enabled=False, timeout=timeout
            )[0]
        except TimeoutError:
            return None

        query_plan = PostgresExplainPlan(raw_query_plan)
        return query_plan.as_qep()

    def parse_plan(self, plan: Any, *, query: Optional[SqlQuery] = None) -> QueryPlan:
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
        query_plan = self._pg_instance.execute_query(query, cache_enabled=False)
        estimate: int = query_plan[0]["Plan"]["Plan Rows"]
        return Cardinality(estimate)

    def cost_estimate(self, query: SqlQuery | str) -> float:
        if isinstance(query, SqlQuery):
            query = transform.as_explain(query)
            query = self._pg_instance._hinting_backend.format_query(query)
        else:
            query = self._explainify(query)
        query_plan = self._pg_instance.execute_query(query, cache_enabled=False)
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
            raise ValueError(
                f"Cannot configure operator {operator} as it is not supported by Postgres"
            )
        status = "on" if enabled else "off"
        self._pg_instance.cursor.execute(f"SET {setting_name} TO {status}")  # type: ignore

    def uses_geqo(self, query: SqlQuery) -> bool:
        pass

    def _explainify(self, query: str) -> str:
        if not query.upper().startswith("EXPLAIN (FORMAT JSON)"):
            query = f"EXPLAIN (FORMAT JSON) {query}"
        return query


def _reconnect(name: str, *, pool: DatabasePool) -> PostgresInterface:
    """Fetches a connection from the database pool.

    If the connection is in a bad state (e.g. because the user called close() before), it is re-established.

    Parameters
    ----------
    name : str
        The name of the database connection in the pool.
    pool : DatabasePool
        The current pool.
    """
    current_instance = pool.retrieve_database(name)
    assert isinstance(current_instance, PostgresInterface)

    status = current_instance._connection.info.status
    if status != psycopg.pq.ConnStatus.OK:
        # Actually there are a lot of other ConnStatus values beyond OK and Bad
        # We could handle them explicitly here, or we might just defined anything that is not OK as Bad.
        # The latter seems much simpler so let's just do this for now.
        current_instance.reset_connection()

    return current_instance


def _ini_config_reader(file: TextIO, *, path: Path) -> dict[str, str]:
    config = configparser.ConfigParser()
    config.read_file(file)
    if len(config.sections()) != 1:
        raise ValueError(
            f"Malformed INI file '{path}': INI config file must contain exactly one section."
        )
    section = config.sections()[0]
    return dict(config.items(section))


def _read_connection_from_file(config_file: Path) -> str:
    extension = config_file.suffix.lower()
    if extension == "":
        with open(config_file, "r") as f:
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
            raise ValueError(
                f"Unsupported config file format. Could read config file '{config_file}'."
            )

    with open(config_file, mode) as f:
        config_data = reader(f)
    parts = (f"{param} = '{value}'" for param, value in config_data.items())
    return " ".join(parts)


def connect(
    *,
    name: str = "postgres",
    application_name: str = "PostBOUND",
    connect_string: str = "",
    config_file: str | Path = "",
    encoding: str = "UTF8",
    cache_enabled: bool = False,
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
    name : str, optional
        A name to identify the current connection if multiple connections to different Postgres instances should be maintained.
        This is used to register the instance on the `DatabasePool`. Defaults to *postgres*.
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
    cache_enabled : bool, optional
        Controls the default caching behaviour of the Postgres instance. Caching of general queries is disabled by default,
        whereas queries from the statistics interface are cached by default.
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
    db_pool = DatabasePool.get_instance()
    if name in db_pool and not refresh:
        return _reconnect(name, pool=db_pool)

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
        with open(".psycopg_connection", "r") as f:
            connect_string = f.readline().strip()
    elif os.getenv("PGDATABASE"):
        warnings.warn(
            "Using environment variables to construct connection string.", stacklevel=2
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

    postgres_db = PostgresInterface(
        connect_string,
        application_name=application_name,
        system_name=name,
        client_encoding=encoding,
        cache_enabled=cache_enabled,
        debug=debug,
    )
    if not private:
        orig_name = name
        instance_idx = 2
        while name in db_pool:
            name = f"{orig_name} - {instance_idx}"
            instance_idx += 1
        db_pool.register_database(name, postgres_db)
    return postgres_db


def start(pgdata: str | Path = "", *, logfile: str | Path = "") -> None:
    """Starts a local Postgres server.

    This function assumes that *pg_ctl* is available on the system PATH and either the server's data directory is specified
    explicitly, or set via the *PGDATA* environment variable.
    """
    if os.system("which pg_ctl") != 0:
        raise ValueError("Cannot start Postgres server: pg_ctl is not on PATH")

    pgdata = pgdata or os.environ.get("PGDATA", "")
    pgdata = Path(pgdata).expanduser()
    if not pgdata:
        raise ValueError(
            "Cannot start Postgres server: Must either supply pgdata argument or set PGDATA environment variable"
        )

    args = ["pg_ctl", "-D", pgdata]
    if logfile:
        args.extend(["-l", logfile])
    args.append("start")

    subprocess.run(args, check=True)


def stop(pgdata: str | Path = "", *, raise_on_error: bool = False) -> None:
    """Stops a running (local) Postgres server.

    This function assumes that *pg_ctl* is available on the system PATH and either the server's data directory is specified
    explicitly, or set via the *PGDATA* environment variable.

    If the server cannot be stopped due to whatever reason, an error can be raised by setting the corresponding parameter.
    Otherwise, it is silently ignored.
    """
    if os.system("which pg_ctl") != 0:
        raise ValueError("Cannot stop Postgres server: pg_ctl is not on PATH")

    pgdata = pgdata or os.environ.get("PGDATA", "")
    pgdata = Path(pgdata).expanduser()
    if not pgdata:
        raise ValueError(
            "Cannot stop Postgres server: Must either supply pgdata argument or set PGDATA environment variable"
        )

    subprocess.run(["pg_ctl", "-D", pgdata, "stop"], check=raise_on_error)


def is_running(pgdata: str | Path = "") -> bool:
    """Checks, whether a local Postgres server is currently running.

    This function assumes that *pg_ctl* is available on the system PATH. A data directory can be supplied to check whether
    a server is running for the specific database. If *pgdata* is not supplied, the *PGDATA* environment variable is used as
    a fallback.
    """
    if os.system("which pg_ctl") != 0:
        raise ValueError("Cannot start Postgres server: pg_ctl is not on PATH")

    cmd = ["pg_ctl"]
    pgdata = pgdata or os.environ.get("PGDATA", "")
    if pgdata:
        cmd.extend(["-D", str(pgdata)])
    cmd.append("status")

    res = subprocess.run(cmd)
    return res.returncode == 0


def _parallel_query_initializer(
    connect_string: str, local_data: threading.local, verbose: bool = False
) -> None:
    """Internal function for the `ParallelQueryExecutor` to setup worker connections.

    Parameters
    ----------
    connect_string : str
        Connection info to establish a network connection to the Postgres instance. Delegates to Psycopg
    local_data : threading.local
        Data object to store the opened connection
    verbose : bool, optional
        Whether to print logging information, by default *False*

    References
    ----------

    .. Psyopg v3: https://www.psycopg.org/psycopg3/ This is used internally by the Postgres interface to interact with the
       database
    """
    log = util.make_logger(verbose)
    tid = threading.get_ident()
    connection = psycopg.connect(
        connect_string, application_name=f"PostBOUND parallel worker ID {tid}"
    )
    connection.autocommit = True
    local_data.connection = connection
    log(f"[worker id={tid}, ts={util.timestamp()}] Connected")


def _parallel_query_worker(
    query: str | SqlQuery,
    local_data: threading.local,
    timeout: Optional[int] = None,
    verbose: bool = False,
) -> tuple[SqlQuery | str, Any]:
    """Internal function for the `ParallelQueryExecutor` to run individual queries.

    Parameters
    ----------
    query : str | SqlQuery
        The query to execute. The parallel executor does not make use of caching whatsoever, so no additional parameters are
        required.
    local_data : threading.local
        Data object that contains the database connection to use. This should have been initialized by
        `_parallel_query_initializer`
    timeout : Optional[int], optional
        The number of seconds to wait until the calculation is aborted. Defaults to *None*, which indicates no timeout. In
        case of timeout, *None* is returned.
    verbose : bool, optional
        Whether to print logging information, by default *False*

    Returns
    -------
    tuple[SqlQuery | str, Any]
        A tuple of the original query and the (simplified) result set. See `Database.execute_query` for an outline of the
        simplification process. This method applies the same rules. The query is also provided to distinguish the different
        result sets that arrive in parallel.
    """
    log = util.make_logger(verbose)
    connection: psycopg.connection.Connection = local_data.connection
    connection.rollback()
    cursor = connection.cursor()
    if timeout:
        cursor.execute(f"SET statement_timeout = '{timeout}s';")

    log(
        f"[worker id={threading.get_ident()}, ts={util.timestamp()}] Now executing query {query}"
    )
    try:
        cursor.execute(str(query))
        log(
            f"[worker id={threading.get_ident()}, ts={util.timestamp()}] Executed query {query}"
        )
    except psycopg.errors.QueryCanceled as e:
        if "canceling statement due to statement timeout" in e.args:
            log(
                f"[worker id={threading.get_ident()}, ts={util.timestamp()}] Query {query} timed out"
            )
            return query, None
        else:
            raise e

    result_set = cursor.fetchall()
    cursor.close()

    return query, result_set


class ParallelQueryExecutor:
    """The ParallelQueryExecutor provides mechanisms to conveniently execute queries in parallel.

    The parallel execution happens by maintaining a number of worker threads that execute the incoming queries.
    The number of input queries can exceed the worker pool size, potentially by a large margin. If that is the case,
    input queries will be buffered until a worker is available.

    This parallel executor has nothing to do with the Database interface and acts entirely independently and
    Postgres-specific.

    Parameters
    ----------
    connect_string : str
        Connection info to establish a network connection to the Postgres instance. Delegates to Psycopg
    n_threads : Optional[int], optional
        The maximum number of parallel workers to use. If this is not specified, uses ``os.cpu_count()`` many workers.
    timeout : Optional[int], optional
        The number of seconds to wait until an individual query is aborted. Timeouts do not affect other queries (both those
        running in parallel or those running afterwards on the same worker). In case of a timeout, the query's entry in the
        result set will be *None*.
    verbose : bool, optional
        Whether to print logging information during the query execution. This is off by default.

    See Also
    --------
    Database
    PostgresInterface

    References
    ----------

    .. Psyopg v3: https://www.psycopg.org/psycopg3/ This is used internally by the Postgres interface to interact with the
       database
    """

    def __init__(
        self,
        connect_string: str,
        n_threads: Optional[int] = None,
        *,
        timeout: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self._n_threads = (
            n_threads if n_threads is not None and n_threads > 0 else os.cpu_count()
        )
        self._connect_string = connect_string
        self._timeout = timeout
        self._verbose = verbose

        self._thread_data = threading.local()
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._n_threads,
            initializer=_parallel_query_initializer,
            initargs=(
                self._connect_string,
                self._thread_data,
            ),
        )
        self._tasks: list[concurrent.futures.Future] = []
        self._results: list[Any] = []
        self._queries: dict[concurrent.futures.Future, SqlQuery | str] = {}

    def queue_query(self, query: SqlQuery | str) -> None:
        """Adds a new query to the queue, to be executed as soon as possible.

        If a timeout was specified when creating the executor, this timeout will be applied to the query.

        Parameters
        ----------
        query : SqlQuery | str
            The query to execute
        """
        future = self._thread_pool.submit(
            _parallel_query_worker,
            query,
            self._thread_data,
            self._timeout,
            self._verbose,
        )
        self._tasks.append(future)
        self._queries[future] = query

    def drain_queue(
        self,
        timeout: Optional[float] = None,
        *,
        callback: Optional[Callable[[SqlQuery | str, ResultSet | None], None]] = None,
    ) -> None:
        """Blocks, until all queries currently queued have terminated.

        Parameters
        ----------
        timeout : Optional[float], optional
            The number of seconds to wait until the calculation is aborted. Defaults to *None*, which indicates no timeout,
            i.e. wait forever. Note that in contrast to the timeout specified when creating the executor, this timeout
            applies to the entire queue and not to individual queries. For example, one can set the per-query timeout to 1s
            which means that each query can be executed for at most 1 second. If an additional timeout of 10s is specified
            on the queue, the entire queue will be aborted if it takes longer than 10 seconds to complete.
        callback : Optional[Callable[[SqlQuery | str, ResultSet | None], None]], optional
            A callback to be executed with each query that completes. The callback receives the query that was executed and
            the corresponding (raw) result set as arguments. If the query ran into a timeout, the result set is *None*.

        Raises
        ------
        TimeoutError or concurrent.futures.TimeoutError
            If some queries have not completed after the given `timeout`.
        """
        for future in concurrent.futures.as_completed(self._tasks, timeout=timeout):
            result_set = future.result()
            self._results.append(result_set)

            if not callback:
                continue

            query = self._queries[future]
            callback(query, result_set)

    def result_set(self) -> dict[str | SqlQuery, ResultSet | None]:
        """Provides the results of all queries that have terminated already, mapping query -> result set

        Returns
        -------
        dict[str | SqlQuery, ResultSet | None]
            The query results. The raw result sets are provided without any simplification. If the query timed out, the result
            set is *None* (in contrast to empty result sets like `[]`).
        """
        return dict(self._results)

    def close(self) -> None:
        """Terminates all worker threads. The executor is essentially useless afterwards."""
        self._thread_pool.shutdown()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        running_workers = [future for future in self._tasks if future.running()]
        completed_workers = [future for future in self._tasks if future.done()]

        return (
            f"Concurrent query pool of {self._n_threads} workers, {len(self._tasks)} tasks "
            f"(run={len(running_workers)} fin={len(completed_workers)})"
        )


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
    def ok(result_set: ResultSet, exec_time: float) -> _ResultEvent:
        return _ResultEvent("success", result_set, exec_time)

    @staticmethod
    def timeout(duration: float) -> _ResultEvent:
        return _ResultEvent("timeout", None, duration)

    @staticmethod
    def failed(error: Exception) -> _ResultEvent:
        return _ResultEvent("failure", None, math.nan, error=error)


def _timeout_worker_ctl(
    status_pipe: mp_conn.Connection, *, timeout: float, executor: TimeoutQueryExecutor
) -> _ResultEvent:
    event = status_pipe.recv()
    match event:
        case _BackendConnectedEvent(pid):
            return _timeout_worker_prep(
                status_pipe, timeout=timeout, backend_pid=pid, executor=executor
            )
        case _WorkerErrorEvent(e):
            return _ResultEvent.failed(e)
        case _:
            raise StateError("Unexpected event", event)


def _timeout_worker_prep(
    status_pipe: mp_conn.Connection,
    *,
    timeout: float,
    backend_pid: int,
    executor: TimeoutQueryExecutor,
) -> _ResultEvent:
    event = status_pipe.recv()
    match event:
        case _QueryReadyEvent():
            return _timeout_worker_run_query(
                status_pipe, timeout=timeout, backend_pid=backend_pid, executor=executor
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
    executor: TimeoutQueryExecutor,
) -> _ResultEvent:
    query_finished = status_pipe.poll(timeout)
    if not query_finished:
        return _timeout_worker_timeout(
            timeout, backend_pid=backend_pid, executor=executor
        )

    event = status_pipe.recv()
    match event:
        case _QueryFinishedEvent():
            return _timeout_worker_await_result(
                status_pipe, backend_pid=backend_pid, executor=executor
            )
        case _WorkerErrorEvent(e):
            _timeout_worker_abort(e, backend_pid, executor=executor)
            return _ResultEvent.failed(e)
        case _:
            raise StateError("Unexpected event", event)


def _timeout_worker_await_result(
    status_pipe: mp_conn.Connection, *, backend_pid: int, executor: TimeoutQueryExecutor
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


def _timeout_worker_timeout(
    timeout: float, *, backend_pid: int, executor: TimeoutQueryExecutor
) -> _ResultEvent:
    executor._abort_backend(backend_pid)
    return _ResultEvent.timeout(timeout)


def _timeout_worker_abort(
    error: Exception, backend_pid: int, *, executor: TimeoutQueryExecutor
) -> _ResultEvent:
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
        cache_enabled = pg_config.get("cache_enabled", False)
        pg_instance = PostgresInterface(
            connect_string,
            application_name="PostBOUND Timeout Worker",
            cache_enabled=cache_enabled,
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
            raise DatabaseServerError(msg, e)
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
            raise DatabaseUserError(msg, e)

        runtime = (end_time - start_time) / 10**9
        pg_instance._last_query_runtime = runtime

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


class TimeoutQueryExecutor:
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

    def __init__(self, postgres_instance: Optional[PostgresInterface] = None) -> None:
        self._pg_instance: PostgresInterface = (
            postgres_instance
            if postgres_instance is not None
            else DatabasePool.get_instance().current_database()
        )
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
        cached_query: bool = (
            kwargs.get("cache_enabled", False)
            and query in self._pg_instance._query_cache
        )
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
            "cache_enabled": self._pg_instance.cache_enabled,
            "config": self._pg_instance.current_configuration(
                runtime_changeable_only=True
            ),
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


PostgresExplainJoinNodes = {
    "Nested Loop": JoinOperator.NestedLoopJoin,
    "Hash Join": JoinOperator.HashJoin,
    "Merge Join": JoinOperator.SortMergeJoin,
}
"""A mapping from Postgres EXPLAIN node names to the corresponding join operators."""

PostgresExplainScanNodes = {
    "Seq Scan": ScanOperator.SequentialScan,
    "Index Scan": ScanOperator.IndexScan,
    "Index Only Scan": ScanOperator.IndexOnlyScan,
    "Bitmap Heap Scan": ScanOperator.BitmapScan,
}
"""A mapping from Postgres EXPLAIN node names to the corresponding scan operators."""

PostgresExplainIntermediateNodes = {
    "Materialize": IntermediateOperator.Materialize,
    "Memoize": IntermediateOperator.Memoize,
    "Sort": IntermediateOperator.Sort,
}
"""A mapping from Postgres EXPLAIN node names to the corresponding intermediate operators."""


NodeType = Literal[
    "Result",
    "ProjectSet",
    "ModifyTable",
    "Append",
    "Merge Append",
    "Recursive Union",
    "BitmapAnd",
    "BitmapOr",
    "Nested Loop",
    "Merge Join",
    "Hash Join",
    "Seq Scan",
    "Sample Scan",
    "Gather",
    "Gather Merge",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Index Scan",
    "Bitmap Heap Scan",
    "Tid Scan",
    "Tid Range Scan",
    "Subquery Scan",
    "Function Scan",
    "Table Function Scan",
    "Values Scan",
    "CTE Scan",
    "Named Tuplestore Scan",
    "WorkTable Scan",
    "Foreign Scan",
    "Custom Scan",
    "Materialize",
    "Memoize",
    "Sort",
    "Incremental Sort",
    "Group",
    "Aggregate",
    "WindowAgg",
    "Unique",
    "SetOp",
    "LockRows",
    "Limit",
    "Hash",
]
"""All different nodes that can be created by Postgres.

This has been extracted directly from ExplainNode() in explain.c from the Postgres source code.
"""


class PostgresExplainNode:
    """Simplified model of a plan node as provided by Postgres' *EXPLAIN* output in JSON format.

    Generally speaking, a node stores all the information about the plan node that we currently care about. This is mostly
    focused on optimizer statistics, along with some additional data. Explain nodes form a hierarchichal structure with each
    node containing an arbitrary number of child nodes. Notice that this model is very loose in the sense that no constraints
    are enforced and no sanity checking is performed. For example, this means that nodes can contain more than two children
    even though this can never happen in a real *EXPLAIN* plan. Similarly, the correspondence between filter predicates and
    the node typse (e.g. join filter for a join node) is not checked.

    All relevant data from the explain node is exposed as attributes on the objects. Even though these are mutable, they should
    be thought of as read-only data objects.

    Parameters
    ----------
    explain_data : dict
        The JSON data of the current explain node. This is parsed and prepared as part of the *__init__* method.

    Attributes
    ----------
    node_type : NodeType | None, default None
        The node type. This should never be empty or *None*, even though it is technically allowed.
    cost : float, default NaN
        The optimizer's cost estimation for this node. This includes the cost of all child nodes as well. This should normally
        not be *NaN*, even though it is technically allowed.
    cardinality_estimate : float, default NaN
        The optimizer's estimation of the number of tuples that will be *produced* by this operator. This should normally not
        be *NaN*, even though it is technically allowed.
    execution_time : float, default NaN
        For *EXPLAIN ANALYZE* plans, this is the actual total execution time of the node in seconds. For pure *EXPLAIN*
        plans, this is *NaN*
    true_cardinality : float, default NaN
        For *EXPLAIN ANALYZE* plans, this is the average of the number of tuples that were actually produced for each loop of
        the node. For pure *EXPLAIN* plans, this is *NaN*
    loops : int, default 1
        For *EXPLAIN ANALYZE* plans, this is the number of times the operator was invoked. The number of invocations can mean
        a number of different things: for parallel operators, this normally matches the number of parallel workers. For scans,
        this matches the number of times a new tuple was requested (e.g. for an index nested-loop join the number of loops of
        the index scan part indicates how many times the index was probed).
    relation_name : str | None, default None
        The name of the relation/table that is processed by this node. This should be defined on scan nodes, but could also
        be present on other nodes.
    relation_alias : str | None, default None
        The alias of the relation/table under which the relation was accessed in th equery plan. See `relation_name`.
    index_name : str | None, default None
        The name of the index that was probed. This should be defined on index scans and index-only scans, but could also be
        present on other nodes.
    filter_condition : str | None, default None
        A post-processing filter that is applied to all rows emitted by this operator. This is most important for scan
        operations with an attached filter predicate, but can also be present on some joins.
    index_condition : str | None, default None
        The condition that is used to locate the matching tuples in an index scan or index-only scan
    join_filter : str | None, default None
        The condition that is used to determine matching tuples in a join
    hash_condition : str | None, default None
        The condition that is used to determine matching tuples in a hash join
    recheck_condition : str | None, default None
        For lossy bitmap scans or bitmap scans based on lossy indexes, this is post-processing check for whether the produced
        tuples actually match the filter condition
    parent_relationship : str | None, default None
        Describes the role that this node plays in relation to its parent. Common values are *inner* which denotes that
        this is the inner child of a join and *outer* which denotes the opposite.
    parallel_workers : int | float, default NaN
        For parallel operators in *EXPLAIN ANALYZE* plans, this is the actual number of worker processes that were started.
        Notice that in total there is one additional worker. This process takes care of spawning the other workers and
        managing them, but can also take part in the input processing.
    sort_keys : list[str]
        The columns that are used to sort the tuples that are produced by this node. This is most important for sort nodes,
        but can also be present on other nodes.
    shared_blocks_read : float, default NaN
        For *EXPLAIN ANALYZE* plans with *BUFFERS* enabled, this is the number of blocks/pages that where retrieved from
        disk while executing this node, including the reads of all its child nodes.
    shared_blocks_buffered : float, default NaN
        For *EXPLAIN ANALYZE* plans with *BUFFERS* enabled, this is the number of blocks/pages that where retrieved from
        the shared buffer while executing this node, including the hits of all its child nodes.
    temp_blocks_read : float, default NaN
        For *EXPLAIN ANALYZE* blocks with *BUFFERS* enabled, this is the number of short-term data structures (e.g. hash
        tables, sorts) that where read by this node, including reads of all its child nodes.
    temp_blocks_written : float, default NaN
        For *EXPLAIN ANALYZE* blocks with *BUFFERS* enabled, this is the number of short-term data structures (e.g. hash
        tables, sorts) that where written by this node, including writes of all its child nodes.
    plan_width : float, default NaN
        The average width of the tuples that are produced by this node.
    children : list[PostgresExplainNode]
        All child / input nodes for the current node
    """

    @staticmethod
    def all_node_types() -> frozenset[NodeType]:
        """All node types that are currently recognized by PostBOUND."""
        return frozenset(get_args(NodeType))

    def __init__(self, explain_data: dict) -> None:
        self.node_type: NodeType = explain_data["Node Type"]

        self.cost: float = explain_data.get("Total Cost", math.nan)
        self.cardinality_estimate: float = explain_data.get("Plan Rows", math.nan)
        self.execution_time: float = (
            explain_data.get("Actual Total Time", math.nan) / 1000
        )

        # true_cardinality is accessed as a property to add a warning for BitmapAnd/Or nodes
        self._true_card: float = explain_data.get("Actual Rows", math.nan)

        self.loops: float = explain_data.get("Actual Loops", 1)

        self.relation_name: str | None = explain_data.get("Relation Name", None)
        self.relation_alias: str | None = explain_data.get("Alias", None)
        self.index_name: str | None = explain_data.get("Index Name", None)
        self.subplan_name: str | None = explain_data.get("Subplan Name", None)
        self.cte_name: str | None = explain_data.get("CTE Name", None)

        self.filter_condition: str | None = explain_data.get("Filter", None)
        self.index_condition: str | None = explain_data.get("Index Cond", None)
        self.join_filter: str | None = explain_data.get("Join Filter", None)
        self.hash_condition: str | None = explain_data.get("Hash Cond", None)
        self.recheck_condition: str | None = explain_data.get("Recheck Cond", None)
        self.parent_relationship: str | None = explain_data.get(
            "Parent Relationship", None
        )
        self.launched_workers: int = explain_data.get("Workers Launched", 0)
        self.planned_workers: int = explain_data.get("Workers Planned", 0)
        self.sort_keys: str = explain_data.get("Sort Key", "")

        self.shared_blocks_read: int = explain_data.get("Shared Read Blocks", math.nan)
        self.shared_blocks_cached: int = explain_data.get("Shared Hit Blocks", math.nan)
        self.temp_blocks_read: int = explain_data.get("Temp Read Blocks", math.nan)
        self.temp_blocks_written: int = explain_data.get(
            "Temp Written Blocks", math.nan
        )
        self.plan_width: int = explain_data.get("Plan Width", math.nan)
        self.children = [
            PostgresExplainNode(child) for child in explain_data.get("Plans", [])
        ]

        self.explain_data: dict = explain_data
        self._hash_val = hash(
            (
                self.node_type,
                self.relation_name,
                self.relation_alias,
                self.index_name,
                self.subplan_name,
                self.cte_name,
                self.filter_condition,
                self.index_condition,
                self.join_filter,
                self.hash_condition,
                self.recheck_condition,
                self.parent_relationship,
                self.launched_workers,
                tuple(self.children),
            )
        )

    @property
    def true_cardinality(self) -> float:
        if self.node_type in {"BitmapAnd", "BitmapOr"}:
            # For BitmapAnd/BitmapOr nodes, the actual number of rows is always 0.
            # This is due to limitations in the Postgres implementation.
            warnings.warn(
                "Postgres does not report the actual number of rows for bitmap nodes correctly. Returning NaN.",
                stacklevel=2,
            )
            return math.nan
        return self._true_card

    def is_scan(self) -> bool:
        """Checks, whether the current node corresponds to a scan node.

        For Bitmap index scans, which are multi-level scan operators, this is true for the heap scan part that takes care of
        actually reading the tuples according to the bitmap provided by the bitmap index scan operators.

        Returns
        -------
        bool
            Whether the node is a scan node
        """
        return self.node_type in PostgresExplainScanNodes

    def is_join(self) -> bool:
        """Checks, whether the current node corresponds to a join node.

        Returns
        -------
        bool
            Whether the node is a join node
        """
        return self.node_type in PostgresExplainJoinNodes

    def is_analyze(self) -> bool:
        """Checks, whether this *EXPLAIN* plan is an *EXPLAIN ANALYZE* plan or a pure *EXPLAIN* plan.

        The analyze variant does not only obtain the plan, but actually executes it. This enables the comparison of the
        optimizer's estimates to the actual values. If a plan is an *EXPLAIN ANALYZE* plan, some attributes of this node
        receive actual values. These include `execution_time`, `true_cardinality`, `loops` and `parallel_workers`.


        Returns
        -------
        bool
            Whether the node represents part of an *EXPLAIN ANALYZE* plan
        """
        return not math.isnan(self.execution_time)

    def filter_conditions(self) -> dict[str, str]:
        """Collects all filter conditions that are defined on this node

        Returns
        -------
        dict[str, str]
            A dictionary mapping the type of filter condition (e.g. index condition or join filter) to the actual filter value.
        """
        conditions: dict[str, str] = {}
        if self.filter_condition is not None:
            conditions["Filter"] = self.filter_condition
        if self.index_condition is not None:
            conditions["Index Cond"] = self.index_condition
        if self.join_filter is not None:
            conditions["Join Filter"] = self.join_filter
        if self.hash_condition is not None:
            conditions["Hash Cond"] = self.hash_condition
        if self.recheck_condition is not None:
            conditions["Recheck Cond"] = self.recheck_condition
        return conditions

    def inner_outer_children(self) -> Sequence[PostgresExplainNode]:
        """Provides the children of this node in a sequence of inner, outer if applicable.

        For all nodes where this structure is not meaningful (e.g. intermediate nodes that operate on a single relation or
        scan nodes), the child nodes are returned as-is (e.g. as a list of a single child or an empty list).

        Returns
        -------
        Sequence[PostgresExplainNode]
            The children of the current node in a unified format
        """
        if len(self.children) < 2:
            return self.children
        assert len(self.children) == 2

        first_child, second_child = self.children
        inner_child = (
            first_child if first_child.parent_relationship == "Inner" else second_child
        )
        outer_child = first_child if second_child == inner_child else second_child
        return (inner_child, outer_child)

    def parse_table(self) -> Optional[TableReference]:
        """Provides the table that is processed by this node.

        Returns
        -------
        Optional[TableReference]
            The table being scanned. For non-scan nodes, or nodes where no table can be inferred, *None* will be returned.
        """
        if not self.relation_name:
            return None
        alias = (
            self.relation_alias
            if self.relation_alias is not None
            and self.relation_alias != self.relation_name
            else ""
        )
        return TableReference(self.relation_name, alias)

    def as_qep(self) -> QueryPlan:
        """Transforms the postgres-specific plan to a standardized `QueryPlan` instance.

        Notice that this transformation is lossy since not all information from the Postgres plan can be represented in query
        execution plan instances. Furthermore, this transformation can be problematic for complicated queries that use
        special Postgres features. Most importantly, for queries involving subqueries, special node types and parent
        relationships can be contained in the plan, that cannot be represented by other parts of PostBOUND. If this method
        and the resulting query execution plans should be used on complex workloads, it is advisable to check the plans twice
        before continuing.

        Returns
        -------
        QueryPlan
            The equivalent query execution plan for this node

        Raises
        ------
        ValueError
            If the node contains more than two children.
        """
        return self._generate_qep()

    def _generate_qep(self, *, card_adjustment: float = 1.0) -> QueryPlan:
        child_nodes = []
        inner_child, outer_child, subplan_child = None, None, None

        # planned workers is 0 for sequential execution, otherwise it contains the number of additional workers
        # if we already have a card adjustment, we are alrady in a parallel subplan so we re-use the existing adjustment
        # otherwise, the total adjustment is planned_workers + 1 to account for the main process
        child_adjustment = (
            card_adjustment if card_adjustment > 1 else (self.planned_workers + 1)
        )

        for child in self.children:
            parent_rel = child.parent_relationship
            qep_child = child._generate_qep(card_adjustment=child_adjustment)

            match parent_rel:
                case "Inner":
                    inner_child = qep_child
                case "Outer":
                    outer_child = qep_child
                case "SubPlan" | "InitPlan" | "Subquery":
                    subplan_child = qep_child
                case "Member":
                    child_nodes.append(qep_child)
                case _:
                    raise ValueError(
                        f"Unknown parent relationship '{parent_rel}' for child {child}"
                    )

        if inner_child and outer_child:
            child_nodes = [outer_child, inner_child] + child_nodes
        elif outer_child:
            child_nodes.insert(0, outer_child)
        elif inner_child:
            child_nodes.insert(0, inner_child)

        table = self.parse_table()
        subplan_name = self.subplan_name or self.cte_name

        true_card = self.true_cardinality * self.loops
        estimated_card = self.cardinality_estimate * card_adjustment

        if self.is_scan():
            operator = PostgresExplainScanNodes.get(self.node_type, None)
        elif self.is_join():
            operator = PostgresExplainJoinNodes.get(self.node_type, None)
        else:
            operator = PostgresExplainIntermediateNodes.get(self.node_type, None)

        sort_keys = (
            self._parse_sort_keys()
            if self.sort_keys
            else self._infer_sorting_from_children()
        )
        shared_hits = (
            None if math.isnan(self.shared_blocks_cached) else self.shared_blocks_cached
        )
        shared_misses = (
            None if math.isnan(self.shared_blocks_read) else self.shared_blocks_read
        )

        if self.launched_workers > 0:
            par_workers = self.launched_workers
        elif self.planned_workers > 0:
            par_workers = self.planned_workers
        else:
            par_workers = None

        plan = QueryPlan(
            self.node_type,
            base_table=table,
            operator=operator,
            children=child_nodes,
            parallel_workers=par_workers,
            index=self.index_name,
            sort_keys=sort_keys,
            estimated_cost=self.cost,
            estimated_cardinality=Cardinality(estimated_card),
            actual_cardinality=Cardinality(true_card),
            execution_time=self.execution_time,
            cache_hits=shared_hits,
            cache_misses=shared_misses,
            subplan_root=subplan_child,
            subplan_name=subplan_name,
        )

        return plan

    def inspect(self, *, _indentation: int = 0) -> str:
        """Provides a pretty string representation of the *EXPLAIN* sub-plan that can be printed.

        Parameters
        ----------
        _indentation : int, optional
            This parameter is internal to the method and ensures that the correct indentation is used for the child nodes
            of the plan. When inspecting the root node, this value is set to its default value of `0`.

        Returns
        -------
        str
            A string representation of the *EXPLAIN* sub-plan.
        """
        if self.parent_relationship in ("InitPlan", "SubPlan"):
            padding = " " * (max(_indentation - 2, 0))
            cte_name = self.subplan_name if self.subplan_name else ""
            own_inspection = [f"{padding}{self.parent_relationship}: {cte_name}"]
        else:
            own_inspection = []
        padding = " " * _indentation
        prefix = f"{padding}<- " if padding else ""
        own_inspection += [prefix + str(self)]
        child_inspections = [
            child.inspect(_indentation=_indentation + 2) for child in self.children
        ]
        return "\n".join(own_inspection + child_inspections)

    def _infer_sorting_from_children(self) -> list[SortKey]:
        # TODO: Postgres is a cruel mistress. Even if output is sorted, it might not be marked as such.
        # For example, in index scans, this is implictly encoded in the index condition, somethimes even nested in other
        # expressions. We first need a reliable way to parse the expressions into a PostBOUND-compatible format.
        # See _parse_sort_keys for a start.
        return []

    def _parse_sort_keys(self) -> list[SortKey]:
        # TODO implementation
        return []

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.node_type == other.node_type
            and self.relation_name == other.relation_name
            and self.relation_alias == other.relation_alias
            and self.children == other.children
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        analyze_content = (
            f" (actual time={self.execution_time}s rows={self.true_cardinality} loops={self.loops})"
            if self.is_analyze()
            else ""
        )
        explain_content = f"(cost={self.cost} rows={self.cardinality_estimate})"
        conditions = " ".join(
            f"{condition}: {value}"
            for condition, value in self.filter_conditions().items()
        )
        conditions = " " + conditions if conditions else ""
        if self.is_scan():
            tab = self.parse_table()
            assert tab is not None
            scan_info = f" on {tab.identifier()}"
        elif self.cte_name:
            scan_info = f" on {self.cte_name}"
        else:
            scan_info = ""
        return (
            self.node_type + scan_info + explain_content + analyze_content + conditions
        )


class PostgresExplainPlan:
    """Models an entire *EXPLAIN* plan produced by Postgres

    In contrast to `PostgresExplainNode`, this includes additional parameters (planning time and execution time) for the entire
    plan, rather than just portions of it.

    This class supports all methods that are specified on the general `QueryPlan` and returns the correct data for its actual
    plan.

    Parameters
    ----------
    explain_data : dict | list[dict]
        The JSON data of the entire explain plan. This is parsed and prepared as part of the *__init__* method.


    Attributes
    ----------
    planning_time : float
        The time in seconds that the optimizer spent to build the plan
    execution_time : float
        The time in seconds the query execution engine needed to calculate the result set of the query. This does not account
        for network time to transmit the result set.
    query_plan : PostgresExplainNode
        The actual plan
    """

    def __init__(self, explain_data: dict | list[dict]) -> None:
        self.explain_data = (
            explain_data[0] if isinstance(explain_data, list) else explain_data
        )
        if not (isinstance(self.explain_data, dict) and "Plan" in self.explain_data):
            raise ValueError(
                f"Invalid explain data: missing 'Plan' key: {explain_data}"
            )

        self.planning_time: float = (
            self.explain_data.get("Planning Time", math.nan) / 1000
        )
        self.execution_time: float = (
            self.explain_data.get("Execution Time", math.nan) / 1000
        )
        self.query_plan = PostgresExplainNode(self.explain_data["Plan"])
        self._normalized_plan = self.query_plan.as_qep()

    @property
    def root(self) -> PostgresExplainNode:
        """Gets the root node of the actual query plan."""
        return self.query_plan

    def is_analyze(self) -> bool:
        """Checks, whether this *EXPLAIN* plan is an *EXPLAIN ANALYZE* plan or a pure *EXPLAIN* plan.

        The analyze variant does not only obtain the plan, but actually executes it. This enables the comparison of the
        optimizer's estimates to the actual values. If a plan is an *EXPLAIN ANALYZE* plan, some attributes of this node
        receive actual values. These include `execution_time`, `true_cardinality`, `loops` and `parallel_workers`.


        Returns
        -------
        bool
            Whether the plan represents an *EXPLAIN ANALYZE* plan
        """
        return self.query_plan.is_analyze()

    def as_qep(self) -> QueryPlan:
        """Provides the actual explain plan as a normalized query execution plan instance

        For notes on pecularities of this method, take a look at the *See Also* section

        Returns
        -------
        QueryPlan
            The query execution plan

        See Also
        --------
        PostgresExplainNode.as_qep
        """
        return self._normalized_plan

    def inspect(self) -> str:
        """Provides a pretty string representation of the actual plan.

        Returns
        -------
        str
            A string representation of the plan

        See Also
        --------
        PostgresExplainNode.inspect
        """
        return self.query_plan.inspect()

    def __json__(self) -> Any:
        return self.explain_data

    def __getattribute__(self, name: str) -> Any:
        # All methods that are not defined on the Postgres plan delegate to the default DB plan
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            root_plan_node = object.__getattribute__(self, "query_plan")
            try:
                return root_plan_node.__getattribute__(name)
            except AttributeError:
                normalized_plan = object.__getattribute__(self, "_normalized_plan")
                return normalized_plan.__getattribute__(name)

    def __hash__(self) -> int:
        return hash(self.query_plan)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.query_plan == other.query_plan

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.is_analyze():
            prefix = f"EXPLAIN ANALYZE (plan time={self.planning_time}, exec time={self.execution_time})"
        else:
            prefix = "EXPLAIN"

        return f"{prefix} root: {self.query_plan}"


class WorkloadShifter:
    """The shifter provides simple means to manipulate the current contents of a database.

    Currently, such means only include the deletion of specific rows, but other tools could be added in the future.

    Parameters
    ----------
    pg_instance : PostgresInterface
        The database to manipulate
    """

    def __init__(self, pg_instance: PostgresInterface) -> None:
        self.pg_instance = pg_instance

    def remove_random(
        self,
        table: TableReference | str,
        *,
        n_rows: Optional[int] = None,
        row_pct: Optional[float] = None,
        vacuum: bool = False,
    ) -> None:
        """Deletes tuples from a specific tables at random.

        Parameters
        ----------
        table : TableReference | str
            The table from which to delete
        n_rows : Optional[int], optional
            The absolute number of rows to delete. Defaults to *None* in which case the `row_pct` is used.
        row_pct : Optional[float], optional
            The share of rows to delete. Value should be in range (0, 1). Defaults to *None* in which case the `n_rows` is
            used.
        vacuum : bool, optional
            Whether the database should be vacuumed after deletion. This optimizes the page layout by compacting the pages and
            forces a refresh of all statistics.

        Raises
        ------
        ValueError
            If no correct `n_rows` or `row_pct` values have been given.

        Warnings
        --------
        Notice that deletions in the given table can trigger further deletions in other tables through cascades in the schema.
        """
        table_name = table.full_name if isinstance(table, TableReference) else table
        n_rows = self._determine_row_cnt(table_name, n_rows, row_pct)
        pk_column = self.pg_instance.schema().primary_key_column(table_name)
        if pk_column is None:
            raise ValueError(
                f"Cannot perform random removal on table '{table_name}': No primary key found"
            )

        removal_template = textwrap.dedent("""
                                           WITH delete_samples AS (
                                               SELECT {col} AS sample_id, RANDOM() AS _pb_rand_val
                                               FROM {table}
                                               ORDER BY _pb_rand_val
                                               LIMIT {cnt}
                                           )
                                           DELETE FROM {table}
                                           WHERE EXISTS (SELECT 1 FROM delete_samples WHERE sample_id = {col})
                                           """)
        removal_query = removal_template.format(
            table=table_name, col=pk_column.name, cnt=n_rows
        )
        self._perform_removal(removal_query, vacuum)

    def remove_ordered(
        self,
        column: ColumnReference | str,
        *,
        n_rows: Optional[int] = None,
        row_pct: Optional[float] = None,
        ascending: bool = True,
        null_placement: Optional[Literal["first", "last"]] = None,
        vacuum: bool = False,
    ) -> None:
        """Deletes the smallest/largest tuples from a specific table.

        Parameters
        ----------
        column : ColumnReference | str
            The column to infer the deletion order. Can be either a proper column reference including the containing table, or
            a fully-qualified column string such as _table.column_ .
        n_rows : Optional[int], optional
            The absolute number of rows to delete. Defaults to *None* in which case the `row_pct` is used.
        row_pct : Optional[float], optional
            The share of rows to delete. Value should be in range (0, 1). Defaults to *None* in which case the `n_rows` is
            used.
        ascending : bool, optional
            Whether the first or the last rows should be deleted. *NULL* values are according to `null_placement`.
        null_placement : Optional[Literal["first", "last"]], optional
            Where to put *NULL* values in the order. Using the default value of *None* treats *NULL* values as being the
            largest values possible.
        vacuum : bool, optional
            Whether the database should be vacuumed after deletion. This optimizes the page layout by compacting the pages and
            forces a refresh of all statistics.

        Raises
        ------
        ValueError
            If no correct `n_rows` or `row_pct` values have been given.

        Warnings
        --------
        Notice that deletions in the given table can trigger further deletions in other tables through cascades in the schema.
        """

        if isinstance(column, str):
            table_name, col_name = column.split(".")
        elif isinstance(column, ColumnReference) and column.table is not None:
            table_name, col_name = column.table.full_name, column.name
        elif isinstance(column, ColumnReference):
            raise UnboundColumnError(column)
        else:
            raise TypeError("Unknown column type: " + str(column))
        n_rows = self._determine_row_cnt(table_name, n_rows, row_pct)
        pk_column = self.pg_instance.schema().primary_key_column(table_name)
        if pk_column is None:
            raise ValueError(
                f"Cannot perform ordered removal on table '{table_name}': No primary key found"
            )

        order_direction = "ASC" if ascending else "DESC"
        null_vals = "" if null_placement is None else f"NULLS {null_placement.upper()}"
        removal_template = textwrap.dedent("""
                                           WITH delete_entries AS (
                                               SELECT {pk_col}
                                               FROM {table}
                                               ORDER BY {order_col} {order_dir} {nulls}, {pk_col} ASC
                                               LIMIT {cnt}
                                           )
                                           DELETE FROM {table} t
                                           WHERE EXISTS (SELECT 1 FROM delete_entries
                                                         WHERE delete_entries.{pk_col} = t.{pk_col})
                                           """)
        removal_query = removal_template.format(
            table=table_name,
            pk_col=pk_column.name,
            order_col=col_name,
            order_dir=order_direction,
            nulls=null_vals,
            cnt=n_rows,
        )
        self._perform_removal(removal_query, vacuum)

    def generate_marker_table(
        self,
        target_table: str,
        marker_pct: float = 0.5,
        *,
        target_column: str = "id",
        marker_table: Optional[str] = None,
        marker_column: Optional[str] = None,
    ) -> None:
        """Generates a new table that can be used to store rows that should be deleted at a later point in time.

        The marker table will be created if it does not exist already. It contains exactly two columns: one column for the
        marker index (an ascending integer value) and another column that stores the primary keys of rows that should be
        deleted from the target table. If the marker table exists already, all current markings (but not the marked rows
        themselves) are removed. Afterwards, the new rows to delete are selected at random.

        By default, only the target table is a required parameter. All other parameters have default values or can be inferred
        from the target table. The marker index column is *marker_idx*.

        Parameters
        ----------
        target_table : str
            The table from which rows should be removed
        marker_pct : float
            The percentage of rows that should be included in the marker table. Allowed range is *[0, 1]*.
        target_column : str, optional
            The column that contains the values used to identify the rows to be deleted in the target table. Defaults to *id*.
        marker_table : Optional[str], optional
            The name of the marker table that should store the row identifiers. Defaults to
            *<target table name>_delete_markers*.
        marker_column : Optional[str], optional
            The name of the column in the marker table that should contain the target column values. Defaults to
            *<target table name>_<target column name>*.

        See Also
        --------
        remove_marked
        export_marker_table
        """
        marker_table = (
            f"{target_table}_delete_marker" if marker_table is None else marker_table
        )
        marker_column = (
            f"{target_table}_{target_column}"
            if marker_column is None
            else marker_column
        )
        target_col_ref = ColumnReference(target_column, TableReference(target_table))
        target_column_type = self.pg_instance.schema().datatype(target_col_ref)
        marker_create_query = textwrap.dedent(f"""
                                              CREATE TABLE IF NOT EXISTS {marker_table} (
                                                  marker_idx BIGSERIAL PRIMARY KEY,
                                                  {marker_column} {target_column_type}
                                              );
                                              """)
        marker_pct = round(marker_pct * 100)
        marker_inflate_query = textwrap.dedent(f"""
                                               INSERT INTO {marker_table}({marker_column})
                                               SELECT {target_column}
                                               FROM {target_table} TABLESAMPLE BERNOULLI ({marker_pct});
                                               """)
        with self.pg_instance.obtain_new_local_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(marker_create_query)  # type: ignore
            cursor.execute(f"DELETE FROM {marker_table};")  # type: ignore
            cursor.execute(marker_inflate_query)  # type: ignore

    def export_marker_table(
        self,
        *,
        target_table: Optional[str] = None,
        marker_table: Optional[str] = None,
        out_file: Optional[str | Path] = None,
    ) -> None:
        """Stores a marker table in a CSV file on disk.

        This allows the marker table to be re-imported later on.

        Parameters
        ----------
        target_table : Optional[str], optional
            The name of the target table for which the marker has been created. This can be used to infer the name of the
            marker table if the defaults have been used.
        marker_table : Optional[str], optional
            The name of the marker table. Can be omitted if the default name has been used and `target_table` is specified.
        out_file : Optional[str | Path], optional
            The name and path of the output CSV file to create. If omitted, the name will be `<marker table name>.csv` and the
            file will be placed in the current working directory. If specified, an absolute path must be used.

        Raises
        ------
        ValueError
            If neither `target_table` nor `marker_table` are given.

        See Also
        --------
        import_marker_table
        remove_marked
        """
        if target_table is None and marker_table is None:
            raise ValueError("Either marker table or target table are required!")
        marker_table = (
            f"{target_table}_delete_marker" if marker_table is None else marker_table
        )
        out_file = (
            Path(f"{marker_table}.csv").absolute() if out_file is None else out_file
        )
        self.pg_instance.cursor().execute(
            f"COPY {marker_table} TO '{out_file}' DELIMITER ',' CSV HEADER;"
        )  # type: ignore

    def import_marker_table(
        self,
        *,
        target_table: Optional[str] = None,
        marker_table: Optional[str] = None,
        target_column: str = "id",
        marker_column: Optional[str] = None,
        target_column_type: Optional[str] = None,
        in_file: Optional[str | Path] = None,
    ) -> None:
        """Loads the contents of a marker table from a CSV file from disk.

        The table will be created if it does not exist already. If the marker table exists already, all current markings (but
        not the marked rows themselves) are removed. Afterwards, the new markings are imported.

        Parameters
        ----------
        target_table : Optional[str], optional
            The name of the target table for which the marker has been created. This can be used to infer the name of the
            marker table if the defaults have been used.
        marker_table : Optional[str], optional
            The name of the marker table. Can be omitted if the default name has been used and `target_table` is specified.
        target_column : str, optional
            The column that contains the values used to identify the rows to be deleted in the target table. Defaults to *id*.
        marker_table : Optional[str], optional
            The name of the marker table that should store the row identifiers. Defaults to
            *<target table name>_delete_markers*.
        target_column_type : Optional[str], optional
            The datatype of the target column. If this parameter is not given, `target_table` has to be specified to infer the
            proper datatype from the schema metadata.
        in_file : Optional[str | Path], optional
            The name and path of the CSV file to read. If omitted, the name will be `<marker table name>.csv` and the
            file will be loaded in the current working directory. If specified, an absolute path must be used.

        Raises
        ------
        ValueError
            If neither `target_table` nor `marker_table` are given.

        See Also
        --------
        export_marker_table
        remove_marked
        """
        if not target_table and not marker_table:
            raise ValueError("Either marker table or target table are required!")
        marker_table = (
            f"{target_table}_delete_marker" if marker_table is None else marker_table
        )
        marker_column = (
            f"{target_table}_{target_column}"
            if marker_column is None
            else marker_column
        )
        in_file = Path(f"{marker_table}.csv").absolute() if in_file is None else in_file

        if target_column_type is None:
            assert target_table is not None
            target_col_ref = ColumnReference(
                target_column, TableReference(target_table)
            )
            target_column_type = self.pg_instance.schema().datatype(target_col_ref)

        marker_create_query = textwrap.dedent(f"""
                                              CREATE TABLE IF NOT EXISTS {marker_table} (
                                                  marker_idx BIGSERIAL PRIMARY KEY,
                                                  {marker_column} {target_column_type}
                                              );
                                              """)
        marker_import_query = textwrap.dedent(f"""
                                              COPY {marker_table}(marker_idx, {marker_column})
                                              FROM '{in_file}'
                                              DELIMITER ','
                                              CSV HEADER;
                                              """)
        with self.pg_instance.obtain_new_local_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(marker_create_query)  # type: ignore
            cursor.execute(f"DELETE FROM {marker_table}")  # type: ignore
            cursor.execute(marker_import_query)  # type: ignore

    def remove_marked(
        self,
        target_table: str,
        *,
        target_column: str = "id",
        marker_table: Optional[str] = None,
        marker_column: Optional[str] = None,
        vacuum: bool = False,
    ) -> None:
        """Deletes rows according to their primary keys stored in a marker table.

        Parameters
        ----------
        target_table : str
            The table from which the rows should be removed.
        target_column : str, optional
            A column of the target table that is used to identify rows matching the marked rows to remove. Defaults to *id*.
        marker_table : Optional[str], optional
            A table containing marks of the rows to delete. Defaults to *<target table>_delete_markers*.
        marker_column : Optional[str], optional
            A column of the marker table that contains the values of the columns to remove. Defaults to
            *<target table>_<target column>*.
        vacuum : bool, optional
            Whether the database should be vacuumed after deletion. This optimizes the page layout by compacting the pages and
            forces a refresh of all statistics.

        See Also
        --------
        generate_marker_table
        """
        # TODO: align parameter types with TableReference and ColumnReference
        marker_table = (
            f"{target_table}_delete_marker" if marker_table is None else marker_table
        )
        marker_column = (
            f"{target_table}_{target_column}"
            if marker_column is None
            else marker_column
        )
        removal_query = textwrap.dedent(f"""
                                        DELETE FROM {target_table}
                                        WHERE EXISTS (SELECT 1 FROM {marker_table}
                                                WHERE {marker_table}.{marker_column} = {target_table}.{target_column})""")
        self._perform_removal(removal_query, vacuum)

    def _perform_removal(self, removal_query: str, vacuum: bool) -> None:
        """Executes a specific removal query and optionally cleans up the storage system.

        Parameters
        ----------
        removal_query : str
            The query that describes the desired delete operation.
        vacuum : bool
            Whether the database should be vacuumed after deletion. This optimizes the page layout by compacting the pages and
            forces a refresh of all statistics.
        """
        with self.pg_instance.obtain_new_local_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(removal_query)  # type: ignore
        if vacuum:
            # We can't use the with-syntax here because VACUUM cannot be executed inside a transaction
            conn = self.pg_instance.obtain_new_local_connection()
            conn.autocommit = True
            cursor = conn.cursor()
            # We really need a full vacuum due to cascading deletes
            cursor.execute("VACUUM FULL ANALYZE;")
            cursor.close()
            conn.close()

    def _determine_row_cnt(
        self, table: str, n_rows: Optional[int], row_pct: Optional[float]
    ) -> int:
        """Calculates the absolute number of rows to delete while also performing sanity checks.

        Parameters
        ----------
        table : str
            The table from which rows should be deleted. This is necessary to determine the current row count.
        n_rows : Optional[int]
            The absolute number of rows to delete.
        row_pct : Optional[float]
            The fraction in (0, 1) of rows to delete.

        Returns
        -------
        int
            The absolute number rows to delete. This is equal to `n_rows` if that parameter was given. Otherwise, the number is
            inferred from the `row_pct` and the current number of tuples in the table.

        Raises
        ------
        ValueError
            If either both or neither `n_rows` and `row_pct` was given or any of the parameters is outside of the allowed
            range.
        """
        if n_rows is None and row_pct is None:
            raise ValueError(
                "Either absolute number of rows or row percentage must be given"
            )
        if n_rows is not None and row_pct is not None:
            raise ValueError(
                "Cannot use both absolute number of rows and row percentage"
            )

        if n_rows is not None and not n_rows > 0:
            raise ValueError("Not a valid row count: " + str(n_rows))
        elif n_rows is not None and n_rows > 0:
            return n_rows

        assert row_pct is not None, "Row percentage or n_rows must be given"
        if not 0.0 < row_pct < 1.0:
            raise ValueError("Not a valid row percentage: " + str(row_pct))

        total_n_rows = self.pg_instance.statistics().total_rows(
            TableReference(table), cache_enabled=False, emulated=True
        )
        if total_n_rows is None:
            raise StateError(
                "Could not determine total number of rows for table " + table
            )
        return round(row_pct * total_n_rows)
