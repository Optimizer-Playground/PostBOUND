from __future__ import annotations

import concurrent.futures
import json
import math
import textwrap
import time
import warnings
from collections import UserString
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, overload

import quacklab

from . import qal, transform
from ._core import (
    BoundColumnReference,
    Cardinality,
    ColumnReference,
    Cost,
    JoinOperator,
    PhysicalOperator,
    ScanOperator,
    TableReference,
    UnboundColumnError,
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
from ._qep import QueryPlan
from .db import (
    Database,
    DatabasePool,
    DatabaseSchema,
    DatabaseStatistics,
    HintService,
    Histogram,
    HistogramApproximation,
    OptimizerInterface,
    ResultSet,
    UnsupportedDatabaseFeatureError,
    simplify_result_set,
)
from .postgres import PostgresLimitClause
from .qal import SqlQuery
from .util import Version, dicts, jsondict, stats


class DuckDBInterface(Database):
    def __init__(
        self,
        db: Path,
        *,
        system_name: str = "DuckDB",
        cache_enabled: bool = False,
    ) -> None:
        super().__init__(system_name=system_name, cache_enabled=cache_enabled)

        self._dbfile = db

        self._cur = quacklab.connect(db)
        self._last_query_runtime = math.nan

        self._stats = DuckDBStatistics(self)
        self._schema = DuckDBSchema(self)
        self._optimizer = DuckDBOptimizer(self)
        self._hinting = DuckDBHintService(self)

        self._timeout_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def schema(self) -> DuckDBSchema:
        return self._schema

    def statistics(self) -> DatabaseStatistics:
        return self._stats

    def hinting(self) -> HintService:
        return self._hinting

    def optimizer(self) -> OptimizerInterface:
        return self._optimizer

    def execute_query(
        self,
        query: SqlQuery | str,
        *,
        cache_enabled: Optional[bool] = None,
        raw: bool = False,
        timeout: Optional[float] = None,
    ) -> Any:
        if isinstance(query, SqlQuery) and query.hints and query.hints.preparatory_statements:
            for preparatory_statement in query.hints.preparatory_statements:
                self._cur.execute(preparatory_statement)
            query = transform.drop_hints(query, preparatory_statements_only=True)
        if isinstance(query, SqlQuery):
            query = self._hinting.format_query(query)
        elif isinstance(query, UserString):
            query = str(query)

        if cache_enabled:
            cached_res = self._query_cache.get(query)
            if cached_res is not None:
                return cached_res if raw else simplify_result_set(cached_res)

        if timeout is not None:
            raw_result = self.execute_with_timeout(query, timeout=timeout)
            if raw_result is None:
                raise TimeoutError(query)
        else:
            start_time = time.perf_counter_ns()
            self._cur.execute(query)
            end_time = time.perf_counter_ns()

            raw_result = self._cur.fetchall()
            self._last_query_runtime = (end_time - start_time) / 10**9  # convert to seconds

        if cache_enabled:
            self._query_cache[query] = raw_result

        return raw_result if raw else simplify_result_set(raw_result)

    def execute_with_timeout(self, query: SqlQuery | str, *, timeout: float = 60.0) -> Optional[ResultSet]:
        if isinstance(query, SqlQuery):
            query = self._hinting.format_query(query)

        promise = self._timeout_executor.submit(self._execute_worker, query)
        try:
            promise.result(timeout=timeout)
            result_set = self._cur.fetchall()
            return result_set
        except concurrent.futures.TimeoutError:
            self._cur.interrupt()
            promise.result()

            # Make sure to update the last query runtime just now, the worker might terminate while we try to cancel and update
            # the runtime itself. This way we overwrite any measurement that the worker might have produced.
            self._last_query_runtime = math.inf
            return None

    def last_query_runtime(self) -> float:
        return self._last_query_runtime

    def time_query(self, query: SqlQuery, *, timeout: Optional[float] = None) -> float:
        self.execute_query(query, cache_enabled=False, raw=True, timeout=timeout)
        return self.last_query_runtime()

    def database_name(self) -> str:
        self._cur.execute("SELECT CURRENT_DATABASE();")
        result_set = self._cur.fetchone()
        assert result_set is not None
        db_name = result_set[0]
        return db_name

    def database_system_version(self) -> Version:
        self._cur.execute("SELECT version();")
        result_set = self._cur.fetchone()
        assert result_set is not None
        raw_ver: str = result_set[0]
        raw_ver = raw_ver.removeprefix("v")
        raw_ver = raw_ver.split("-")[0]  # remove the build information
        return Version(raw_ver)

    def cursor(self) -> quacklab.DuckDBPyConnection:
        return self._cur.cursor()

    def close(self) -> None:
        self._cur.close()

    def reconnect(self) -> None:
        self._cur = quacklab.connect(self._dbfile)

    def reset_connection(self) -> None:
        try:
            self.close()
        except Exception:
            pass

        self._cur = quacklab.connect(self._dbfile)

    def describe(self) -> jsondict:
        base_info: dict[str, object] = {
            "system_name": self.database_system_name(),
            "system_version": self.database_system_version(),
            "database": self.database_name(),
        }

        schema_info: list[jsondict] = []
        for table in self._schema.tables():
            column_info: list[jsondict] = []

            for column in self._schema.columns(table):
                column_info.append(
                    {
                        "column": str(column),
                        "indexed": self._schema.has_index(column),
                        "foreign_keys": self._schema.foreign_keys_on(column),
                    }
                )

            schema_info.append(
                {
                    "table": str(table),
                    "n_rows": self.statistics().total_rows(table, emulated=True),
                    "columns": column_info,
                    "primary_key": self._schema.primary_key_column(table),
                }
            )

        base_info["schema_info"] = schema_info
        return base_info

    def _execute_worker(self, query: str) -> None:
        start_time = time.perf_counter_ns()
        self._cur.execute(query)
        end_time = time.perf_counter_ns()
        self._last_query_runtime = (end_time - start_time) / 10**9  # convert to seconds


class DuckDBSchema(DatabaseSchema):
    def __init__(self, db: DuckDBInterface) -> None:
        super().__init__(db, prep_placeholder="?")
        self._tables: set[TableReference] = set()

    def tables(
        self, *, catalog: str = "", schema: str = "", include_system_tables: bool = False
    ) -> set[TableReference]:
        if self._tables:
            return self._tables

        catalog_placeholder = "?" if catalog else "current_database()"
        schema_placeholder = "?" if schema else "current_schema()"
        query_template = f"""
            SELECT table_name
            FROM duckdb_tables()
            WHERE lower(database_name) = lower({catalog_placeholder})
                AND lower(schema_name) = lower({schema_placeholder})
        """

        params: list[str] = []
        if catalog:
            params.append(catalog)
        if schema:
            params.append(schema)

        cur = self._db.cursor()
        cur.execute(query_template, parameters=params)
        result_set = cur.fetchall()
        assert result_set is not None

        tables = {TableReference(row[0], catalog=catalog, schema=schema) for row in result_set}
        self._tables = tables
        return self._tables

    def has_secondary_index(self, column: ColumnReference) -> bool:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)

        catalog_placeholder = "?" if column.table.catalog else "current_database()"
        schema_placeholder = "?" if column.table.schema else "current_schema()"
        query_template = textwrap.dedent(f"""
            SELECT ddbi.index_name
            FROM duckdb_indexes() ddbi
            WHERE lower(ddbi.table_name) = lower(?)
                AND lower(ltrim(rtrim(ddbi.expressions, ']'), '[')) = lower(?)
                AND lower(ddbi.database_name) = lower({catalog_placeholder})
                AND lower(ddbi.schema_name) = lower({schema_placeholder})
            """)

        params = [column.table.full_name, column.name]
        if column.table.catalog:
            params.append(column.table.catalog)
        if column.table.schema:
            params.append(column.table.schema)

        cur = self._db.cursor()
        cur.execute(query_template, parameters=params)
        result_set = cur.fetchone()

        return result_set is not None

    def indexes_on(self, column: ColumnReference) -> set[str]:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)

        catalog_placeholder = "?" if column.table.catalog else "current_database()"
        schema_placeholder = "?" if column.table.schema else "current_schema()"

        # The query template is much more complicated here, due to the different semantics of the constraint_column_usage
        # view. For UNIQUE constraints, the column is the column that is constrained. However, for foreign keys, the column
        # is the column that is being referenced.
        # Notice that this template is different from the vanilla template provided by the default implementation: in the
        # first part, we query from duckdb_indexes() instead of information_schema.key_column_usage!

        query_template = textwrap.dedent(f"""
            SELECT ddbi.index_name
            FROM duckdb_indexes() ddbi
            WHERE lower(ddbi.table_name) = lower(?)
                AND lower(ltrim(rtrim(ddbi.expressions, ']'), '[')) = lower(?)
                AND lower(ddbi.database_name) = lower({catalog_placeholder})
                AND lower(ddbi.schema_name) = lower({schema_placeholder})
            UNION
            SELECT tc.constraint_name
            FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu
                    ON tc.constraint_name = ccu.constraint_name
                    AND tc.table_name = ccu.table_name
                    AND tc.table_schema = ccu.table_schema
                    AND tc.table_catalog = ccu.table_catalog
            WHERE tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
                AND lower(ccu.table_name) = lower(?)
                AND lower(ccu.column_name) = lower(?)
                AND lower(ccu.table_catalog) = lower({catalog_placeholder})
                AND lower(ccu.table_schema) = lower({schema_placeholder})
            """)

        # Due to the UNION query, we need to repeat the placeholders. While the implementation is definitely not elegant,
        # this solution is arguably better than relying on named parameters which might or might not be supported by the
        # target database.
        params = [column.table.full_name, column.name]
        if column.table.catalog:
            params.append(column.table.catalog)
        if column.table.schema:
            params.append(column.table.schema)
        params *= 2  # repeat the parameters for the second part of the query

        cur = self._db.cursor()
        cur.execute(query_template, params)
        result_set = cur.fetchall()
        assert result_set is not None

        return {row[0] for row in result_set}

    def foreign_keys_on(self, column: ColumnReference) -> set[BoundColumnReference]:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)

        catalog_placeholder = "?" if column.table.catalog else "current_database()"
        schema_placeholder = "?" if column.table.schema else "current_schema()"
        query_template = f"""
            SELECT referenced_table, referenced_column_names
            FROM duckdb_constraints
            WHERE lower(table_name) = lower(?)
                AND lower(constraint_column_names) = lower(array_value(?))
                AND lower(database_name) = lower({catalog_placeholder})
                AND lower(schema_name) = lower({schema_placeholder})
                AND constraint_type = 'FOREIGN KEY';
        """

        params = [column.table.full_name, column.name]
        if column.table.catalog:
            params.append(column.table.catalog)
        if column.table.schema:
            params.append(column.table.schema)

        cur = self._db.cursor()
        cur.execute(query_template, params)
        result_set = cur.fetchall()
        assert result_set is not None

        foreign_keys: set[BoundColumnReference] = set()
        for table_name, column_names in result_set:
            if len(column_names) > 1:
                warnings.warn(
                    f"Column {column} is part of a multi-column foreign key constraint. "
                    "This is currently not fully supported, only the first column will be returned.",
                    stacklevel=2,
                )
            if not column_names:
                warnings.warn(
                    f"Column {column} is part of a foreign key constraint, but no referenced columns could be identified.",
                    stacklevel=2,
                )
                continue
            column_names = column_names[:1]
            referenced_column = column_names[0]
            table = TableReference(table_name, schema=column.table.schema)
            column = BoundColumnReference(referenced_column, table)
            foreign_keys.add(column)

        return foreign_keys


class DuckDBStatistics(DatabaseStatistics):
    def __init__(
        self,
        db: DuckDBInterface,
        *,
        emulated: bool = False,
        enable_emulation_fallback: bool = True,
        cache_enabled: Optional[bool] = True,
    ) -> None:
        super().__init__(
            db,
            emulated=emulated,
            enable_emulation_fallback=enable_emulation_fallback,
            cache_enabled=cache_enabled,
        )

    def _retrieve_total_rows_from_stats(self, table: TableReference) -> Optional[int]:
        catalog_placeholder = "?" if table.catalog else "current_database()"
        schema_placeholder = "?" if table.schema else "current_schema()"

        query_template = textwrap.dedent(f"""
            SELECT estimated_size
            FROM duckdb_tables()
            WHERE lower(table_name) = lower(?)
                AND lower(database_name) = lower({catalog_placeholder})
                AND lower(schema_name) = lower({schema_placeholder})
            """)

        params = [table.full_name]
        if table.catalog:
            params.append(table.catalog)
        if table.schema:
            params.append(table.schema)

        cur = self._db.cursor()
        cur.execute(query_template, parameters=params)
        result_set = cur.fetchone()

        if not result_set:
            return None

        return result_set[0] if result_set[0] is not None else None

    def _retrieve_distinct_values_from_stats(self, column: BoundColumnReference) -> Optional[int]:
        if self.enable_emulation_fallback:
            return self._calculate_distinct_values(column)
        raise UnsupportedDatabaseFeatureError(self._db, "distinct value count statistics.")

    def _retrieve_min_max_values_from_stats(self, column: BoundColumnReference) -> Optional[tuple[Any, Any]]:
        if self.enable_emulation_fallback:
            return self._calculate_min_max_values(column)
        raise UnsupportedDatabaseFeatureError(self._db, "min/max value statistics.")

    def _retrieve_most_common_values_from_stats(
        self, column: BoundColumnReference, k: int | None
    ) -> Sequence[tuple[Any, int]]:
        if self.enable_emulation_fallback:
            return self._calculate_most_common_values(column, k)
        raise UnsupportedDatabaseFeatureError(self._db, "most common value statistics.")

    def _retrieve_histogram_from_stats(
        self, column: BoundColumnReference, *, interpolation: HistogramApproximation
    ) -> Optional[Histogram]:
        if self.enable_emulation_fallback:
            return self._calculate_histogram(column, n_bins=100, interpolation=interpolation)
        raise UnsupportedDatabaseFeatureError(self._db, "histogram statistics.")


def _bind_node_to_table(scan_node: dict, *, query: SqlQuery) -> Optional[TableReference]:
    tables = query.tables()
    relname = scan_node.get("extra_info", {}).get("Table", "")
    if not relname:
        return None

    candidate_tables = [tab for tab in tables if tab.full_name == relname]
    if not candidate_tables:
        return None
    elif len(candidate_tables) == 1:
        # all table names are unique, we can directly match on the full name
        return candidate_tables[0]

    # some tables are referenced multiple times, we need to determine the correct alias
    # to avoid parsing the filter predicates backwards, we use a simple trigram-based heuristic here:
    # for each base table we extract the filter predicates from the query and compare them with the filters
    # that are applied in the scan node. Than, we select the table whith the highest overlap.
    query_filters = {candidate: str(query.filters_for(candidate)) for candidate in candidate_tables}
    query_filters = {candidate: "".join(pred.split()) for candidate, pred in query_filters.items()}

    plan_filter: str = scan_node.get("extra_info", {}).get("Filters", "")
    plan_filter = "".join(plan_filter.split())

    node_trigram = set(stats.trigrams(plan_filter))
    filter_trigrams = {candidate: set(stats.trigrams(pred)) for candidate, pred in query_filters.items()}

    scores = {
        candidate: stats.jaccard(node_trigram, pred_trigram) for candidate, pred_trigram in filter_trigrams.items()
    }
    score_ranking = sorted(scores.values(), reverse=True)
    if 0.99 * score_ranking[0] <= score_ranking[1]:
        warnings.warn(
            f"Could not unambiguously bind scan node to a table. Candidates are {scores}.",
        )
        return None

    best_match = dicts.argmax(scores)
    return best_match


def parse_duckdb_plan(raw_plan: dict | str, *, query: Optional[SqlQuery] = None) -> QueryPlan:
    """Parses a DuckDB query plan from its JSON representation.

    The query can be supplied to help with binding scan nodes to their corresponding tables (see Notes).

    Notes
    -----
    Once central limitation of DuckDB's EXPLAIN plans is that they do not contain table aliases. Thus, if a table is
    referenced multiple times in a query, we cannot unambiguously bind scan nodes to their corresponding table. We employ
    some heuristics (e.g. trying to match filter predicates) to resolve this ambiguity, but there is no guarantee that this
    always works correctly.
    """
    if isinstance(raw_plan, str):
        raw_plan = json.loads(raw_plan)

    node_type = raw_plan.get("name") or raw_plan.get("operator_name")
    if not node_type:
        assert len(raw_plan["children"]) == 1, "Expected a single child for the root operator"
        return parse_duckdb_plan(raw_plan["children"][0], query=query)

    if node_type == "EXPLAIN" or node_type == "EXPLAIN_ANALYZE":
        assert len(raw_plan["children"]) == 1, "Expected a single child for EXPLAIN operator"
        return parse_duckdb_plan(raw_plan["children"][0], query=query)

    extras: dict = raw_plan.get("extra_info", {})
    match node_type:
        case "HASH_JOIN":
            node_type = JoinOperator.HashJoin
        case (
            "SEQ_SCAN" | "SEQ_SCAN "  # DuckDB has a weird typo in the SEQ_SCAN label
        ) if extras.get("Type", "") == "Sequential Scan":
            node_type = ScanOperator.SequentialScan
        case (
            "SEQ_SCAN" | "SEQ_SCAN "  # DuckDB has a weird typo in the SEQ_SCAN label
        ) if extras.get("Type", "") == "Index Scan":
            node_type = ScanOperator.IndexScan
        case "PROJECTION" | "FILTER" | "UNGROUPED_AGGREGATE" | "PERFECT_HASH_GROUP_BY":
            pass
        case _:
            warnings.warn(f"Unknown node type: {node_type}, ({extras})")
            pass

    if node_type in ScanOperator:
        base_table = _bind_node_to_table(raw_plan, query=query) if query is not None else None
    else:
        base_table = None

    card_est = float(
        extras.get("Estimated Cardinality", math.nan)
    )  # Estimated Cardinality is a string for some reason..
    card_act = raw_plan.get("operator_cardinality", math.nan)

    children = [parse_duckdb_plan(child, query=query) for child in raw_plan.get("children", [])]

    own_runtime = raw_plan.get("operator_timing", math.nan)
    total_runtime = own_runtime + sum(child.execution_time for child in children)

    return QueryPlan(
        node_type,
        children=children,
        base_table=base_table,
        estimated_cardinality=Cardinality(card_est),
        actual_cardinality=card_act,
        execution_time=total_runtime,
    )


class DuckDBOptimizer(OptimizerInterface):
    def __init__(self, db: DuckDBInterface) -> None:
        self._db = db

    def query_plan(self, query: SqlQuery | str) -> QueryPlan:
        if isinstance(query, SqlQuery):
            explain_query = transform.as_explain(query)
            explain_query = self._db.hinting().format_query(explain_query)
        elif not query.strip().upper().startswith("EXPLAIN"):
            explain_query = f"EXPLAIN (FORMAT JSON) {query}"
        else:
            explain_query = query

        cur = self._db.cursor()
        cur.execute(explain_query)
        result_set = cur.fetchone()
        assert result_set is not None and len(result_set) == 2

        raw_explain = result_set[1]
        parsed = json.loads(raw_explain)
        return parse_duckdb_plan(parsed[0], query=query if isinstance(query, SqlQuery) else None)

    @overload
    def analyze_plan(self, query: SqlQuery) -> QueryPlan: ...

    @overload
    def analyze_plan(self, query: SqlQuery, *, timeout: float) -> Optional[QueryPlan]: ...

    def analyze_plan(self, query: SqlQuery, *, timeout: Optional[float] = None) -> Optional[QueryPlan]:
        query = transform.as_explain_analyze(query)

        try:
            result_set = self._db.execute_query(query, cache_enabled=False, raw=True, timeout=timeout)[0]
        except TimeoutError:
            return None
        assert len(result_set) == 2

        raw_explain = result_set[1]
        parsed = json.loads(raw_explain)
        return parse_duckdb_plan(parsed[0], query=query)

    def parse_plan(self, plan: Any, *, query: Optional[SqlQuery] = None) -> QueryPlan:
        # Similar to the Postgres implementation of parse_plan(), we try to be gracefull
        # and accept both simplified and raw versions of the execute_query() output.
        # In essence, we always try to break the input down to a plain dictionary and
        # unwrap the lists and tuples that come with a raw result set.
        if isinstance(plan, list):
            plan = plan[0]
        if isinstance(plan, tuple):
            plan = plan[1]
        return parse_duckdb_plan(plan, query=query)

    def cardinality_estimate(self, query: SqlQuery | str) -> Cardinality:
        plan = self.query_plan(query)
        if "AGGREGATE" in plan.node_type:
            warnings.warn(
                "Plan could have an aggregate node as root. DuckDB does not estimate cardinalities for aggregations.",
                stacklevel=2,
            )
        return plan.estimated_cardinality

    def cost_estimate(self, query: SqlQuery | str) -> Cost:
        raise UnsupportedDatabaseFeatureError(self._db, "cost estimates")


@dataclass
class HintParts:
    """Models the different kinds of optimizer hints that are supported by Postgres.

    HintParts are designed to conveniently collect all kinds of hints in order to prepare the generation of a proper
    `Hint` clause.

    See Also
    --------
    Hint
    """

    settings: list[str]
    """Settings are global to the current database connection and influence the selection of operators for all queries.

    Typical examples include ``SET enable_nestloop = 'off'``, which disables the usage of nested loop joins for all
    queries.
    """

    hints: list[str]
    """Hints are supplied by the *quack_lab* extension and influence optimizer decisions on a per-query basis.

    Typical examples include the selection of a specific join order as well as the assignment of join operators to
    individual joins.
    """

    @staticmethod
    def empty() -> HintParts:
        """Creates a new hint parts object without any contents.

        Returns
        -------
        HintParts
            A fresh plain hint parts object
        """
        return HintParts([], [])

    def add(self, hint: str) -> None:
        """Adds a new hint.

        This modifies the current object.

        Parameters
        ----------
        hint : str
            The hint to add
        """
        self.hints.append(hint)

    def merge_with(self, other: HintParts) -> HintParts:
        """Combines the hints that are contained in this hint parts object with all hints in the other object.

        This constructs new hint parts and leaves the current objects unmodified.

        Parameters
        ----------
        other : HintParts
            The additional hints to incorporate

        Returns
        -------
        HintParts
            A new hint parts object that contains the hints from both source objects
        """
        merged_settings = self.settings + [setting for setting in other.settings if setting not in self.settings]
        merged_hints = self.hints + [""] + [hint for hint in other.hints if hint not in self.hints]
        return HintParts(merged_settings, merged_hints)

    def __bool__(self) -> bool:
        return bool(self.settings or self.hints)


class DuckDBHintService(HintService):
    def __init__(self, db: DuckDBInterface) -> None:
        self._db = db

    def generate_hints(
        self,
        query: SqlQuery,
        plan: Optional[QueryPlan] = None,
        *,
        join_order: Optional[JoinTree] = None,
        physical_operators: Optional[PhysicalOperatorAssignment] = None,
        plan_parameters: Optional[PlanParameterization] = None,
    ) -> SqlQuery:
        adapted_query = query
        if adapted_query.limit_clause and not isinstance(adapted_query.limit_clause, PostgresLimitClause):
            adapted_query = transform.replace_clause(adapted_query, PostgresLimitClause(adapted_query.limit_clause))

        has_partial_hints = any(param is not None for param in (join_order, physical_operators, plan_parameters))
        if plan is not None and has_partial_hints:
            raise ValueError("Can only hint an entire query plan, or individual parts, not both.")

        if plan is not None:
            join_order = jointree_from_plan(plan)
            physical_operators = operators_from_plan(plan)
            plan_parameters = parameters_from_plan(plan)

        if join_order is not None:
            hint_parts = self._generate_join_order_hint(join_order)
        else:
            hint_parts = HintParts.empty()

        hint_parts = hint_parts if hint_parts else HintParts.empty()

        if physical_operators:
            operator_hints = self._generate_operator_hints(physical_operators)
            hint_parts = hint_parts.merge_with(operator_hints)

        if plan_parameters:
            plan_hints = self._generate_parameter_hints(plan_parameters)
            hint_parts = hint_parts.merge_with(plan_hints)

        if hint_parts:
            adapted_query = self._add_hint_block(adapted_query, hint_parts)

        return adapted_query

    def format_query(self, query: SqlQuery) -> str:
        # DuckDB uses the Postgres SQL dialect, so this part is easy..
        return qal.format_quick(query, flavor="postgres")

    def supports_hint(self, hint: PhysicalOperator | HintType) -> bool:
        return hint in {
            ScanOperator.SequentialScan,
            ScanOperator.IndexScan,
            JoinOperator.NestedLoopJoin,
            JoinOperator.HashJoin,
            JoinOperator.SortMergeJoin,
            HintType.LinearJoinOrder,
            HintType.BushyJoinOrder,
            HintType.Cardinality,
            HintType.Operator,
        }

    def _generate_join_order_hint(self, join_tree: JoinTree) -> HintParts:
        if len(join_tree) < 3:
            # we can't force the join direction anyway, so there's no point in generating a hint if there is just a single join
            return HintParts.empty()

        def recurse(join_tree: JoinTree) -> str:
            if join_tree.is_scan():
                return join_tree.base_table.identifier()

            lhs = recurse(join_tree.outer_child)
            rhs = recurse(join_tree.inner_child)

            return f"({lhs} {rhs})"

        join_order = recurse(join_tree)
        hint_parts = HintParts([], [f"JoinOrder({join_order})"])
        return hint_parts

    def _generate_operator_hints(self, ops: PhysicalOperatorAssignment) -> HintParts:
        if not ops:
            return HintParts.empty()

        hints: list[str] = []
        for tab, scan in ops.scan_operators.items():
            match scan.operator:
                case ScanOperator.SequentialScan:
                    op_txt = "SeqScan"
                case ScanOperator.IndexScan:
                    op_txt = "IdxScan"
                case _:
                    raise UnsupportedDatabaseFeatureError(self._db, scan.operator)
            tab_txt = tab.identifier()
            hints.append(f"{op_txt}({tab_txt})")

        for intermediate, join in ops.join_operators.items():
            match join.operator:
                case JoinOperator.NestedLoopJoin:
                    op_txt = "NestLoop"
                case JoinOperator.HashJoin:
                    op_txt = "HashJoin"
                case JoinOperator.SortMergeJoin:
                    op_txt = "MergeJoin"
                case _:
                    raise UnsupportedDatabaseFeatureError(self._db, join.operator)

            intermediate_txt = self._intermediate_to_hint(intermediate)
            hints.append(f"{op_txt}({intermediate_txt})")

        if ops.intermediate_operators:
            raise UnsupportedDatabaseFeatureError(
                self._db,
                "intermediate operators",
            )

        for param, val in ops.global_settings.items():
            match param:
                case ScanOperator.SequentialScan:
                    param_txt = "enable_seqscan"
                case ScanOperator.IndexScan:
                    param_txt = "enable_indexscan"
                case JoinOperator.NestedLoopJoin:
                    param_txt = "enable_nestloop"
                case JoinOperator.HashJoin:
                    param_txt = "enable_hashjoin"
                case JoinOperator.SortMergeJoin:
                    param_txt = "enable_mergejoin"
                case _:
                    raise UnsupportedDatabaseFeatureError(self._db, param)

            val_txt = "'on'" if val else "'off'"
            hints.append(f"Set({param_txt} = {val_txt})")

        return HintParts([], hints)

    def _generate_parameter_hints(self, parameters: PlanParameterization) -> HintParts:
        if not parameters:
            return HintParts.empty()

        hints: list[str] = []
        for intermediate, card in parameters.cardinalities.items():
            if not card.is_valid():
                continue
            intermediate_txt = self._intermediate_to_hint(intermediate)
            hints.append(f"Card({intermediate_txt} #{card})")

        if parameters.parallel_workers:
            raise UnsupportedDatabaseFeatureError(self._db, "parallel worker hints")

        global_settings: list[str] = []
        for param, val in parameters.system_settings.items():
            if isinstance(val, str):
                val_txt = f"'{val}'"
            else:
                val_txt = str(val)

            global_settings.append(f"{param} = {val_txt};")

        return HintParts(global_settings, hints)

    def _add_hint_block(self, query: SqlQuery, hint_parts: HintParts) -> SqlQuery:
        if not hint_parts:
            return query

        local_hints = ["/*=quack_lab="]
        for local_hint in hint_parts.hints:
            local_hints.append(f"  {local_hint}")
        local_hints.append(" */")

        hints = qal.Hint("\n".join(hint_parts.settings), "\n".join(local_hints))
        return transform.add_clause(query, hints)

    def _intermediate_to_hint(self, intermediate: Iterable[TableReference]) -> str:
        """Convert an iterable of TableReferences to a string representation."""
        return " ".join(table.identifier() for table in intermediate)


def _reconnect(name: str, *, pool: DatabasePool) -> DuckDBInterface:
    current_conn = pool.retrieve_database(name)
    assert isinstance(current_conn, DuckDBInterface)

    try:
        # check if the connection is still active
        current_conn.execute_query("SELECT 1", cache_enabled=False, raw=True)
    except quacklab.ConnectionException as e:
        # otherwise re-establish the connection
        if "Connection Error: Connection already closed!" in e.args:
            current_conn.reconnect()
        else:
            raise e

    return current_conn


def connect(
    db: str | Path,
    *,
    name: str = "duckdb",
    refresh: bool = False,
    private: bool = False,
) -> DuckDBInterface:
    """Connects to a DuckDB database file.

    Parameters
    ----------
    name : str, optional
        A name to identify the current connection if multiple connections to different DuckDB instances should be
        maintained. This is used to register the instance on the `DatabasePool`. Defaults to *duckdb*.
    refresh : bool, optional
        If true, a new connection to the database will always be established, even if a connection to the same database is
        already pooled. The registration key will be suffixed to prevent collisions. By default, the current connection is
        re-used. If that is the case, no further information (e.g. config strings) is read and only the `name` is accessed.
    private : bool, optional
        If true, skips registration of the new instance on the `DatabasePool`. Registration is performed by default.
    """
    db_pool = DatabasePool.get_instance()
    if name in db_pool and not refresh:
        return _reconnect(name, pool=db_pool)

    db = Path(db)
    duckdb_instance = DuckDBInterface(db)

    if not private:
        db_pool.register_database(name, duckdb_instance)

    return duckdb_instance
