from __future__ import annotations

import collections
from collections.abc import Iterable, Sequence

SignificantPostgresSettings = {
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

RuntimeChangeablePostgresSettings = {setting for setting in SignificantPostgresSettings} - {
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
        return PostgresConfiguration(list(args) + [PostgresSetting(key, val) for key, val in kwargs.items()])

    def __init__(self, settings: Iterable[PostgresSetting]) -> None:
        self._settings: dict[str, PostgresSetting] = {setting.parameter: setting for setting in settings}
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
        parameter = setting.parameter if isinstance(setting, PostgresSetting) else setting
        target_settings = dict(self._settings)
        target_settings.pop(parameter, None)
        return PostgresConfiguration(target_settings.values())

    def update(self, setting: PostgresSetting | str, value: object) -> PostgresConfiguration:
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
