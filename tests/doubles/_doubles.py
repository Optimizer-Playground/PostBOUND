"""Implementation of the database test doubles. See `tests.doubles` for the rationale."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from postbound import Cardinality, Cost, QueryPlan, util
from postbound._core import BoundColumnReference, ColumnReference, PhysicalOperator, TableReference
from postbound._hints import HintType, JoinTree, PhysicalOperatorAssignment, PlanParameterization
from postbound.db import (
    Cursor,
    Database,
    DatabaseSchema,
    HintService,
    MostCommonValues,
    OptimizerInterface,
    ResultSet,
    StatisticsCatalog,
)
from postbound.db._db import Histogram, HistogramApproximation
from postbound.qal import SqlQuery
from postbound.util import jsondict


class UnexpectedQueryError(AssertionError):
    """Raised when a double is asked for something it was not told about.

    This is deliberately an error rather than an empty result. A double that silently returns nothing for an
    unrecognised query turns a genuine failure into a passing test that asserts on empty data -- the exact
    hazard that makes recorded fixtures go stale without anyone noticing.
    """


def _normalize(sql: str) -> str:
    """Collapses whitespace and case so that SQL can be matched without depending on formatting."""
    return re.sub(r"\s+", " ", sql).strip().rstrip(";").lower()


@dataclass(frozen=True)
class ExpectedQuery:
    """One recorded call made against a `ScriptedCursor`.

    Attributes
    ----------
    operation : str
        The SQL exactly as the code under test passed it, without normalization.
    parameters : tuple
        The bound parameters, normalized to a tuple so that a list and a tuple of the same values compare
        equal. Empty when the call passed none.
    """

    operation: str
    parameters: tuple = ()

    @property
    def normalized(self) -> str:
        """The operation with whitespace collapsed and case folded."""
        return _normalize(self.operation)

    def mentions(self, *fragments: str) -> bool:
        """Whether every fragment occurs in the normalized operation.

        Convenience for asserting on generated SQL without pinning the exact formatting::

            assert cursor.calls[0].mentions("information_schema.columns", "table_name")
        """
        return all(_normalize(fragment) in self.normalized for fragment in fragments)


class ScriptedCursor(Cursor):
    """A `Cursor` that answers from a script and records everything it was asked.

    This is the highest-leverage double in the suite. `DatabaseSchema` declares **no abstract methods** -- all
    18 of its lookups are default implementations that issue *information_schema* SQL through
    ``self._db.cursor()``. Scripting the cursor therefore exercises that code for real, rather than replacing
    it, and the recording lets a test assert on the SQL that was generated. That is how placeholder bugs
    (``%s`` vs. DuckDB's ``?``) and parameter-count mistakes become visible.

    Responses are matched by substring against the normalized SQL, in insertion order, so a script only has
    to name the distinguishing fragment of each expected query.

    Parameters
    ----------
    responses : dict[str, ResultSet | None], optional
        Maps a SQL fragment to the rows to return for any query containing it.
    strict : bool, optional
        When *True* (the default), a query matching no fragment raises `UnexpectedQueryError`. Set to *False*
        to return an empty result set instead, for tests that only care about the SQL being generated.

    Examples
    --------
    >>> cursor = ScriptedCursor({"information_schema.tables": [("title",), ("movie_info",)]})
    >>> cursor.execute("SELECT table_name FROM information_schema.tables WHERE ...")
    >>> cursor.fetchall()
    [('title',), ('movie_info',)]
    >>> cursor.calls[0].mentions("information_schema.tables")
    True
    """

    def __init__(self, responses: dict[str, ResultSet | None] | None = None, *, strict: bool = True) -> None:
        self._responses: dict[str, ResultSet | None] = {}
        for fragment, rows in (responses or {}).items():
            self._responses[_normalize(fragment)] = rows
        self._strict = strict
        self._pending: ResultSet | None = None
        self.calls: list[ExpectedQuery] = []
        self.closed = False

    def expect(self, fragment: str, rows: ResultSet | None) -> ScriptedCursor:
        """Registers an additional response. Returns *self* so that calls can be chained."""
        self._responses[_normalize(fragment)] = rows
        return self

    def execute(self, operation: str, parameters: dict | Sequence | None = None) -> Cursor:
        bound: tuple
        if parameters is None:
            bound = ()
        elif isinstance(parameters, dict):
            bound = tuple(sorted(parameters.items()))
        else:
            bound = tuple(parameters)
        self.calls.append(ExpectedQuery(operation, bound))

        normalized = _normalize(operation)
        for fragment, rows in self._responses.items():
            if fragment in normalized:
                self._pending = rows
                return self

        if self._strict:
            known = "\n  ".join(sorted(self._responses)) or "(nothing scripted)"
            raise UnexpectedQueryError(
                f"ScriptedCursor received an unscripted query:\n  {operation.strip()}\nScripted fragments:\n  {known}"
            )
        self._pending = []
        return self

    def fetchall(self) -> ResultSet | None:
        return self._pending

    def fetchone(self):
        if not self._pending:
            return None
        return self._pending[0]

    def close(self) -> None:
        self.closed = True

    @property
    def executed_sql(self) -> list[str]:
        """The normalized SQL of every call, in order."""
        return [call.normalized for call in self.calls]

    def calls_mentioning(self, *fragments: str) -> list[ExpectedQuery]:
        """Every recorded call whose SQL contains all of `fragments`."""
        return [call for call in self.calls if call.mentions(*fragments)]


# Intentionally lower-cased: this reads as a declarative literal at call sites, not as a class.
@dataclass(frozen=True)
class column_spec:
    """Declarative description of one column, for use with `StaticSchema`.

    Attributes
    ----------
    name : str
        The column name.
    datatype : str, optional
        The SQL type, defaulting to ``integer``.
    nullable : bool, optional
        Whether the column accepts *NULL*. Defaults to *True*.
    primary_key : bool, optional
        Whether the column is the table's primary key. Implies an index.
    indexed : bool, optional
        Whether a secondary index exists on the column.
    references : str, optional
        ``"table.column"`` that this column has a foreign key to.
    """

    name: str
    datatype: str = "integer"
    nullable: bool = True
    primary_key: bool = False
    indexed: bool = False
    references: str | None = None


class StaticSchema(DatabaseSchema):
    """A `DatabaseSchema` built from a declarative spec instead of a live catalog.

    Only the leaf lookups are overridden. Everything composed on top of them -- ``__getitem__``, ``has_index``,
    ``as_graph``, ``join_equivalence_keys``, ``join_equivalence_classes``, ``lookup_column`` -- runs the real
    implementation, so those stay genuinely under test.

    This generalizes the ad-hoc ``MockSchemaLookup`` that `tests/test_qal.py` defines inline.

    Parameters
    ----------
    tables : dict[str, Iterable[column_spec | str]]
        Maps a table name to its columns. A bare string is shorthand for a nullable integer column.

    Examples
    --------
    >>> schema = StaticSchema({
    ...     "title": [column_spec("id", primary_key=True), "production_year"],
    ...     "movie_info": ["id", column_spec("movie_id", references="title.id")],
    ... })
    >>> sorted(col.name for col in schema.columns("title"))
    ['id', 'production_year']
    >>> schema.has_index(ColumnReference("id", TableReference("title")))
    True
    """

    def __init__(self, tables: dict[str, Iterable[column_spec | str]]) -> None:
        super().__init__(_SchemaOnlyDatabase(self))
        self._spec: dict[str, list[column_spec]] = {}
        for table_name, columns in tables.items():
            self._spec[table_name] = [column_spec(col) if isinstance(col, str) else col for col in columns]

    # -- helpers -------------------------------------------------------------------------------------

    def _table_name(self, table: TableReference | str) -> str:
        return table if isinstance(table, str) else table.full_name

    def _lookup(self, column: ColumnReference) -> column_spec:
        table = self._table_name(column.table) if column.table is not None else None
        if table is None or table not in self._spec:
            raise KeyError(f"Unknown table for column '{column}'")
        for spec in self._spec[table]:
            if spec.name == column.name:
                return spec
        raise KeyError(f"Unknown column '{column}'")

    # -- leaf lookups --------------------------------------------------------------------------------

    def tables(
        self, *, catalog: str = "", schema: str = "", include_system_tables: bool = False
    ) -> set[TableReference]:
        return {TableReference(name) for name in self._spec}

    def columns(self, table: TableReference | str) -> Sequence[BoundColumnReference]:
        name = self._table_name(table)
        reference = table if isinstance(table, TableReference) else TableReference(name)
        return [ColumnReference(spec.name, reference) for spec in self._spec.get(name, [])]

    def is_view(self, table: TableReference | str) -> bool:
        return False

    def is_primary_key(self, column: ColumnReference) -> bool:
        return self._lookup(column).primary_key

    def primary_key_column(self, table: TableReference | str) -> BoundColumnReference | None:
        # A separate leaf lookup rather than a composition over `is_primary_key`: the production
        # implementation issues its own information_schema query against table_constraints.
        name = self._table_name(table)
        reference = table if isinstance(table, TableReference) else TableReference(name)
        primary_keys = [spec for spec in self._spec.get(name, []) if spec.primary_key]
        if not primary_keys:
            return None
        if len(primary_keys) > 1:
            raise ValueError(f"Table {reference} has multiple primary key columns: {primary_keys}")
        return ColumnReference(primary_keys[0].name, reference)

    def has_secondary_index(self, column: ColumnReference) -> bool:
        return self._lookup(column).indexed

    def indexes_on(self, column: ColumnReference) -> set[str]:
        spec = self._lookup(column)
        table = self._table_name(column.table)
        if spec.primary_key:
            return {f"{table}_pkey"}
        return {f"{table}_{spec.name}_idx"} if spec.indexed else set()

    def datatype(self, column: ColumnReference, *, raw: bool = False) -> str:
        return self._lookup(column).datatype

    def is_nullable(self, column: ColumnReference) -> bool:
        return self._lookup(column).nullable

    def foreign_keys_on(self, column: ColumnReference) -> set[BoundColumnReference]:
        spec = self._lookup(column)
        if not spec.references:
            return set()
        table_name, column_name = spec.references.split(".")
        return {ColumnReference(column_name, TableReference(table_name))}


class _SchemaOnlyDatabase(Database):
    """Placeholder `Database` satisfying `DatabaseSchema.__init__`.

    `StaticSchema` answers every lookup from its spec and never reaches for a cursor, so this raises on any
    access rather than pretending to be usable.
    """

    def __init__(self, schema: DatabaseSchema) -> None:
        super().__init__("static")
        self._schema = schema

    def schema(self) -> DatabaseSchema:
        return self._schema

    def _unsupported(self, what: str):
        raise UnexpectedQueryError(
            f"StaticSchema answers from its spec and has no database behind it, but {what} was requested. "
            "Use FakeDatabase if the code under test needs a real Database handle."
        )

    def statistics(self) -> StatisticsCatalog:
        return self._unsupported("statistics()")

    def hinting(self) -> HintService:
        return self._unsupported("hinting()")

    def optimizer(self) -> OptimizerInterface:
        return self._unsupported("optimizer()")

    def execute_query(self, query: SqlQuery | str, *, raw: bool = False) -> Any:
        return self._unsupported(f"execute_query({query!s:.60})")

    def cursor(self) -> Cursor:
        return self._unsupported("cursor()")

    def database_name(self) -> str:
        return "static"

    def dbms_version(self) -> util.Version:
        return util.Version("0.0.0")

    def describe(self) -> jsondict:
        return {"interface-type": "static-schema"}

    def reset_connection(self) -> Any:
        return None

    def close(self) -> None:
        return None

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        return self is other


class FakeStatistics(StatisticsCatalog):
    """A `StatisticsCatalog` answering from dictionaries.

    All six lookups are abstract on the real interface, so every one is implemented here. A statistic that was
    not supplied returns *None*, which is the interface's "the system maintains this but has no value for this
    object" signal -- it does **not** trigger the emulation fallback.
    """

    def __init__(
        self,
        *,
        total_rows: dict[str, int] | None = None,
        num_distinct: dict[str, int] | None = None,
        null_frac: dict[str, float] | None = None,
        min_max: dict[str, tuple[Any, Any]] | None = None,
        most_common_values: dict[str, MostCommonValues] | None = None,
        histograms: dict[str, Histogram] | None = None,
    ) -> None:
        super().__init__()
        self._total_rows = total_rows or {}
        self._num_distinct = num_distinct or {}
        self._null_frac = null_frac or {}
        self._min_max = min_max or {}
        self._mcv = most_common_values or {}
        self._histograms = histograms or {}

    @staticmethod
    def _key(column: ColumnReference) -> str:
        table = column.table.full_name if column.table is not None else "?"
        return f"{table}.{column.name}"

    def total_rows(self, table: TableReference) -> Cardinality | None:
        rows = self._total_rows.get(table.full_name)
        return None if rows is None else Cardinality(rows)

    def num_distinct(self, column: ColumnReference) -> int | None:
        return self._num_distinct.get(self._key(column))

    def null_frac(self, column: ColumnReference) -> float | None:
        return self._null_frac.get(self._key(column))

    def min_max(self, column: ColumnReference) -> tuple[Any, Any] | None:
        return self._min_max.get(self._key(column))

    def most_common_values(self, column: ColumnReference) -> MostCommonValues | None:
        return self._mcv.get(self._key(column))

    def histogram(
        self, column: ColumnReference, *, interpolation: HistogramApproximation = "approx-uni"
    ) -> Histogram | None:
        return self._histograms.get(self._key(column))

    def describe(self) -> jsondict:
        return {"kind": "fake"}


class FakeHintService(HintService):
    """A `HintService` that records what it was asked to hint and returns the query unchanged.

    Useful for asserting that a pipeline handed the expected join order and operator assignment to the hint
    layer, without depending on any particular hint dialect.

    Parameters
    ----------
    supported : set[PhysicalOperator | HintType], optional
        What `supports_hint` should report. Defaults to supporting everything, since most callers use it only
        to filter an operator list.
    """

    def __init__(self, supported: set[PhysicalOperator | HintType] | None = None) -> None:
        self._supported = supported
        self.requests: list[tuple[SqlQuery, QueryPlan | None, JoinTree | None, PhysicalOperatorAssignment | None]] = []

    def generate_hints(
        self,
        query: SqlQuery,
        plan: QueryPlan | None = None,
        *,
        join_order: JoinTree | None = None,
        physical_operators: PhysicalOperatorAssignment | None = None,
        plan_parameters: PlanParameterization | None = None,
    ) -> SqlQuery:
        self.requests.append((query, plan, join_order, physical_operators))
        return query

    def format_query(self, query: SqlQuery) -> str:
        return str(query)

    def supports_hint(self, hint: PhysicalOperator | HintType) -> bool:
        return True if self._supported is None else hint in self._supported


class FakeOptimizer(OptimizerInterface):
    """An `OptimizerInterface` returning canned plans and estimates.

    Parameters
    ----------
    plan : QueryPlan, optional
        Returned by both `query_plan` and `analyze_plan`.
    cardinality : Cardinality, optional
        Returned by `cardinality_estimate`.
    cost : Cost, optional
        Returned by `cost_estimate`.
    """

    def __init__(
        self,
        *,
        plan: QueryPlan | None = None,
        cardinality: Cardinality | None = None,
        cost: Cost | None = None,
    ) -> None:
        self._plan = plan
        self._cardinality = cardinality if cardinality is not None else Cardinality.unknown()
        self._cost = cost if cost is not None else Cost(0.0)
        self.requests: list[SqlQuery | str] = []

    def _require_plan(self) -> QueryPlan:
        if self._plan is None:
            raise UnexpectedQueryError("FakeOptimizer was asked for a plan but none was configured.")
        return self._plan

    def query_plan(self, query: SqlQuery | str) -> QueryPlan:
        self.requests.append(query)
        return self._require_plan()

    def analyze_plan(self, query: SqlQuery) -> QueryPlan:
        self.requests.append(query)
        return self._require_plan()

    def parse_plan(self, plan: Any, *, query: SqlQuery | None = None) -> QueryPlan:
        if isinstance(plan, QueryPlan):
            return plan
        return self._require_plan()

    def cardinality_estimate(self, query: SqlQuery | str) -> Cardinality:
        self.requests.append(query)
        return self._cardinality

    def cost_estimate(self, query: SqlQuery | str) -> Cost:
        self.requests.append(query)
        return self._cost


@dataclass
class FakeDatabase(Database):
    """A minimal in-memory `Database`, for code that needs a handle rather than a server.

    `ResultCache` in ``postbound/db/_cache.py`` is the template: it is already a `Database` that delegates
    everything and intercepts one method. This double follows the same shape but answers from dictionaries.

    Register it on the pool via the ``fake_db`` fixture in ``tests/conftest.py`` rather than by hand, so that
    it is always removed again -- `DatabasePool` is a process-global that nothing resets.

    Parameters
    ----------
    schema_spec : dict[str, Iterable[column_spec | str]], optional
        Passed to `StaticSchema`. An empty schema is used when omitted.
    results : dict[str, ResultSet], optional
        Maps a SQL fragment to the raw result set `execute_query` should return for a matching query.
    cursor : ScriptedCursor, optional
        The cursor handed out by `cursor()`. One is created automatically when omitted.
    statistics_catalog, hint_service, optimizer_interface
        Override the corresponding sub-service. Fakes are created on demand otherwise.
    strict : bool, optional
        Whether an unrecognised `execute_query` raises (default) or returns an empty result set.

    Examples
    --------
    >>> db = FakeDatabase(schema_spec={"title": ["id", "production_year"]},
    ...                   results={"count(*)": [(42,)]})
    >>> db.execute_query("SELECT count(*) FROM title")
    42
    >>> db.executed_queries
    ['select count(*) from title']
    """

    schema_spec: dict[str, Iterable[column_spec | str]] | None = None
    results: dict[str, ResultSet] | None = None
    cursor_double: ScriptedCursor | None = None
    statistics_catalog: StatisticsCatalog | None = None
    hint_service: HintService | None = None
    optimizer_interface: OptimizerInterface | None = None
    strict: bool = True
    name: str = "fake"
    version: str = "1.0.0"

    executed_queries: list[str] = field(default_factory=list, init=False)
    closed: bool = field(default=False, init=False)
    reset_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        super().__init__(self.name)
        self._schema = StaticSchema(self.schema_spec or {})
        self._statistics = self.statistics_catalog or FakeStatistics()
        self._hinting = self.hint_service or FakeHintService()
        self._optimizer = self.optimizer_interface or FakeOptimizer()
        self._cursor = self.cursor_double or ScriptedCursor(strict=self.strict)
        self._results = {_normalize(k): v for k, v in (self.results or {}).items()}

    def schema(self) -> DatabaseSchema:
        return self._schema

    def statistics(self) -> StatisticsCatalog:
        return self._statistics

    def hinting(self) -> HintService:
        return self._hinting

    def optimizer(self) -> OptimizerInterface:
        return self._optimizer

    def expect_result(self, fragment: str, rows: ResultSet) -> FakeDatabase:
        """Registers the result set to return for any query containing `fragment`.

        Returns *self* so that calls can be chained.
        """
        self._results[_normalize(fragment)] = rows
        return self

    def set_schema(self, tables: dict[str, Iterable[column_spec | str]]) -> FakeDatabase:
        """Replaces the schema with one built from `tables`. Returns *self*."""
        self._schema = StaticSchema(tables)
        return self

    def execute_query(self, query: SqlQuery | str, *, raw: bool = False) -> Any:
        from postbound.db import simplify_result_set

        stringified = str(query)
        self.executed_queries.append(_normalize(stringified))

        normalized = _normalize(stringified)
        for fragment, rows in self._results.items():
            if fragment in normalized:
                return rows if raw else simplify_result_set(rows)

        if self.strict:
            known = "\n  ".join(sorted(self._results)) or "(nothing scripted)"
            raise UnexpectedQueryError(
                f"FakeDatabase received an unscripted query:\n  {stringified.strip()}\nScripted fragments:\n  {known}"
            )
        return []

    def cursor(self) -> Cursor:
        return self._cursor

    def database_name(self) -> str:
        return self.name

    def dbms_version(self) -> util.Version:
        return util.Version(self.version)

    def describe(self) -> jsondict:
        return {"interface-type": "fake", "database": self.name}

    def reset_connection(self) -> Any:
        self.reset_count += 1
        return None

    def close(self) -> None:
        self.closed = True

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        return self is other
