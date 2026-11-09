"""Implementation of the database test doubles. See `tests.doubles` for the rationale."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, override

from postbound import Cardinality, Cost, QueryPlan, util
from postbound._core import (
    BoundColumnReference,
    ColumnReference,
    PhysicalOperator,
    TableReference,
    UnboundColumnError,
)
from postbound._hints import HintType, JoinTree, PhysicalOperatorAssignment, PlanParameterization
from postbound.db import (
    Cursor,
    Database,
    DatabaseSchema,
    ForeignKeyRef,
    HintService,
    MostCommonValues,
    OptimizerInterface,
    PrimaryKeyRef,
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

    @override
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

    @override
    def fetchall(self) -> ResultSet | None:
        return self._pending

    @override
    def fetchone(self):
        if not self._pending:
            return None
        return self._pending[0]

    @override
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

    The flags describe *single-column* properties. Several columns of the same table that carry ``primary_key=True``
    together form a composite primary key. Composite UNIQUE constraints, foreign keys and indexes are declared on
    the `StaticSchema` itself.

    Attributes
    ----------
    name : str
        The column name.
    datatype : str, optional
        The SQL type, defaulting to ``integer``.
    nullable : bool, optional
        Whether the column accepts *NULL*. Defaults to *True*.
    primary_key : bool, optional
        Whether the column is (part of) the table's primary key. Implies an index.
    indexed : bool, optional
        Whether a secondary index exists on the column.
    unique : bool, optional
        Whether the column carries a UNIQUE constraint of its own. Implies an index.
    references : str, optional
        ``"table.column"`` that this column has a foreign key to.
    """

    name: str
    datatype: str = "integer"
    nullable: bool = True
    primary_key: bool = False
    indexed: bool = False
    unique: bool = False
    references: str | None = None


@dataclass(frozen=True)
class foreign_key_spec:
    """Declarative description of a (possibly composite) foreign key, for use with `StaticSchema`.

    Attributes
    ----------
    table : str
        The referencing table.
    columns : Sequence[str]
        The foreign key columns of `table`.
    referenced_table : str
        The referenced table.
    referenced_columns : Sequence[str]
        The referenced columns. The i-th referenced column matches the i-th foreign key column.
    """

    table: str
    columns: Sequence[str]
    referenced_table: str
    referenced_columns: Sequence[str]


class StaticSchema(DatabaseSchema):
    """A `DatabaseSchema` built from a declarative spec instead of a live catalog.

    Only the leaf lookups (``tables``, ``columns``, ``is_view``, ``is_primary_key``, ``primary_key_of``,
    ``is_unique``, ``all_foreign_keys_on``, ``all_indexes_on``, ``datatype``, ``is_nullable``) are overridden.
    Everything composed on top of them -- ``__getitem__``, ``foreign_keys_on``, ``has_index``,
    ``has_secondary_index``, ``indexes_on``, ``as_graph``, ``join_equivalence_keys``,
    ``join_equivalence_classes``, ``lookup_column`` -- runs the real implementation, so those stay genuinely under
    test.

    The double mimics a typical Postgres catalog: primary keys and UNIQUE constraints are backed by an index
    (``<table>_pkey``, ``<table>_<cols>_key``), foreign keys are *not* (unless an index is declared explicitly).
    Like the real implementations, the leaf lookups are insensitive to table aliases and the order of columns passed
    to the multi-column predicates (``is_primary_key``, ``is_unique``).

    Parameters
    ----------
    tables : dict[str, Iterable[column_spec | str]]
        Maps a table name to its columns. A bare string is shorthand for a nullable integer column.
    primary_keys : dict[str, Sequence[str]], optional
        Composite primary keys, keyed by table. The column order is the order of the key definition. Must not be
        combined with ``primary_key=True`` flags on the columns of the same table.
    foreign_keys : Iterable[foreign_key_spec], optional
        Foreign keys (typically composite ones) in addition to the ``references`` of the individual columns.
    unique : dict[str, Iterable[Sequence[str]]], optional
        Composite UNIQUE constraints, keyed by table.
    indexes : dict[str, dict[str, Sequence[str]]], optional
        Additional named secondary indexes (typically composite ones), keyed by table and index name.

    Raises
    ------
    ValueError
        If the constraints refer to unknown tables or columns, or if a table has two primary key definitions.

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

    def __init__(
        self,
        tables: dict[str, Iterable[column_spec | str]],
        *,
        primary_keys: dict[str, Sequence[str]] | None = None,
        foreign_keys: Iterable[foreign_key_spec] = (),
        unique: dict[str, Iterable[Sequence[str]]] | None = None,
        indexes: dict[str, dict[str, Sequence[str]]] | None = None,
    ) -> None:
        super().__init__(
            _SchemaOnlyDatabase(self), prep_placeholder="%s", default_schema="public", default_catalog="static"
        )
        self._spec: dict[str, list[column_spec]] = {}
        for table_name, columns in tables.items():
            self._spec[table_name] = [column_spec(col) if isinstance(col, str) else col for col in columns]

        self._pks: dict[str, tuple[str, ...]] = {}
        self._uniques: dict[str, list[tuple[str, ...]]] = {name: [] for name in self._spec}
        self._fks: dict[str, list[foreign_key_spec]] = {name: [] for name in self._spec}
        self._indexes: dict[str, dict[str, tuple[str, ...]]] = {name: {} for name in self._spec}

        for table_name, specs in self._spec.items():
            flagged = tuple(spec.name for spec in specs if spec.primary_key)
            if flagged:
                self._pks[table_name] = flagged
            for spec in specs:
                if spec.unique:
                    self._uniques[table_name].append((spec.name,))
                if spec.indexed:
                    self._indexes[table_name][f"{table_name}_{spec.name}_idx"] = (spec.name,)
                if spec.references:
                    target_table, target_column = spec.references.split(".")
                    self._fks[table_name].append(
                        foreign_key_spec(table_name, [spec.name], target_table, [target_column])
                    )

        for table_name, pk_columns in (primary_keys or {}).items():
            self._require_columns(table_name, pk_columns)
            if table_name in self._pks:
                raise ValueError(f"Table '{table_name}' has multiple primary key definitions")
            self._pks[table_name] = tuple(pk_columns)
        for table_name, constraints in (unique or {}).items():
            for unique_columns in constraints:
                self._require_columns(table_name, unique_columns)
                self._uniques[table_name].append(tuple(unique_columns))
        for table_name, named_indexes in (indexes or {}).items():
            for index_name, index_columns in named_indexes.items():
                self._require_columns(table_name, index_columns)
                self._indexes[table_name][index_name] = tuple(index_columns)
        for fk in (*foreign_keys, *(fk for fks in self._fks.values() for fk in fks)):
            self._require_columns(fk.table, fk.columns)
            self._require_columns(fk.referenced_table, fk.referenced_columns)
            if len(fk.columns) != len(fk.referenced_columns):
                raise ValueError(f"Foreign key on '{fk.table}' has mismatching column counts")
        for fk in foreign_keys:
            self._fks[fk.table].append(fk)

    # -- helpers -------------------------------------------------------------------------------------

    def _require_columns(self, table: str, columns: Iterable[str]) -> None:
        if table not in self._spec:
            raise ValueError(f"Unknown table '{table}'")
        known = {spec.name for spec in self._spec[table]}
        for column in columns:
            if column not in known:
                raise ValueError(f"Unknown column '{table}.{column}'")

    def _table_name(self, table: TableReference | str) -> str:
        return table if isinstance(table, str) else table.full_name

    def _known_table(self, table: TableReference | str) -> tuple[str, TableReference]:
        """Normalizes `table` (dropping any alias) and checks that it exists."""
        reference = table.drop_alias() if isinstance(table, TableReference) else TableReference(table)
        if reference.full_name not in self._spec:
            raise KeyError(f"Unknown table '{reference}'")
        return reference.full_name, reference

    def _lookup(self, column: ColumnReference) -> column_spec:
        table = self._table_name(column.table) if column.table is not None else None
        if table is None or table not in self._spec:
            raise KeyError(f"Unknown table for column '{column}'")
        for spec in self._spec[table]:
            if spec.name == column.name:
                return spec
        raise KeyError(f"Unknown column '{column}'")

    def _column_set(self, column: ColumnReference | Iterable[ColumnReference]) -> tuple[str, set[str]]:
        """Applies the argument checks of the production multi-column predicates and returns table and names."""
        columns = [column] if isinstance(column, ColumnReference) else list(column)
        if not columns:
            raise ValueError("Column list cannot be empty.")
        if not ColumnReference.all_bound(columns):
            unbound = [col for col in columns if not col.is_bound()]
            raise UnboundColumnError(f"Unbound columns detected: {unbound}")
        if len({col.table.drop_alias() for col in columns}) > 1:
            raise ValueError(f"Columns must belong to the same table. Got columns from tables: {columns}")
        for col in columns:
            self._lookup(col)  # strict: unknown columns fail loudly
        return columns[0].table.full_name, {col.name for col in columns}

    # -- leaf lookups --------------------------------------------------------------------------------

    @override
    def tables(
        self, *, catalog: str = "", schema: str = "", include_system_tables: bool = False
    ) -> set[TableReference]:
        return {TableReference(name) for name in self._spec}

    @override
    def columns(self, table: TableReference | str) -> Sequence[BoundColumnReference]:
        reference = table.drop_alias() if isinstance(table, TableReference) else TableReference(table)
        return [ColumnReference(spec.name, reference) for spec in self._spec.get(reference.full_name, [])]

    @override
    def is_view(self, table: TableReference | str) -> bool:
        return False

    @override
    def is_primary_key(self, column: ColumnReference | Iterable[ColumnReference]) -> bool:
        table, names = self._column_set(column)
        return set(self._pks.get(table, ())) == names

    @override
    def primary_key_of(self, table: TableReference | str) -> PrimaryKeyRef | None:
        name, reference = self._known_table(table)
        if name not in self._pks:
            return None
        return PrimaryKeyRef(ColumnReference(col, reference) for col in self._pks[name])

    @override
    def is_unique(self, column: ColumnReference | Iterable[ColumnReference]) -> bool:
        table, names = self._column_set(column)
        return any(set(cols) == names for cols in self._uniques[table])

    @override
    def all_foreign_keys_on(self, table: TableReference | str) -> set[ForeignKeyRef]:
        name, reference = self._known_table(table)
        return {
            ForeignKeyRef(
                [ColumnReference(col, reference) for col in fk.columns],
                [ColumnReference(col, TableReference(fk.referenced_table)) for col in fk.referenced_columns],
            )
            for fk in self._fks[name]
        }

    @override
    def all_indexes_on(self, table: TableReference | str) -> Mapping[str, set[BoundColumnReference]]:
        name, reference = self._known_table(table)
        indexes: dict[str, tuple[str, ...]] = dict(self._indexes[name])
        if name in self._pks:
            indexes[f"{name}_pkey"] = self._pks[name]
        for cols in self._uniques[name]:
            indexes[f"{name}_{'_'.join(cols)}_key"] = cols
        return {index_name: {ColumnReference(col, reference) for col in cols} for index_name, cols in indexes.items()}

    @override
    def datatype(self, column: ColumnReference, *, raw: bool = False) -> str:
        return self._lookup(column).datatype

    @override
    def is_nullable(self, column: ColumnReference) -> bool:
        return self._lookup(column).nullable


class _SchemaOnlyDatabase(Database):
    """Placeholder `Database` satisfying `DatabaseSchema.__init__`.

    `StaticSchema` answers every lookup from its spec and never reaches for a cursor, so this raises on any
    access rather than pretending to be usable.
    """

    def __init__(self, schema: DatabaseSchema) -> None:
        super().__init__("static")
        self._schema = schema

    @override
    def schema(self) -> DatabaseSchema:
        return self._schema

    def _unsupported(self, what: str):
        raise UnexpectedQueryError(
            f"StaticSchema answers from its spec and has no database behind it, but {what} was requested. "
            "Use FakeDatabase if the code under test needs a real Database handle."
        )

    @override
    def statistics(self) -> StatisticsCatalog:
        return self._unsupported("statistics()")

    @override
    def hinting(self) -> HintService:
        return self._unsupported("hinting()")

    @override
    def optimizer(self) -> OptimizerInterface:
        return self._unsupported("optimizer()")

    @override
    def execute_query(self, query: SqlQuery | str, *, raw: bool = False) -> Any:
        return self._unsupported(f"execute_query({query!s:.60})")

    @override
    def cursor(self) -> Cursor:
        return self._unsupported("cursor()")

    @override
    def database_name(self) -> str:
        return "static"

    @override
    def dbms_version(self) -> util.Version:
        return util.Version("0.0.0")

    @override
    def describe(self) -> jsondict:
        return {"interface-type": "static-schema"}

    @override
    def reset_connection(self) -> Any:
        return None

    @override
    def close(self) -> None:
        return None

    @override
    def __hash__(self) -> int:
        return hash(id(self))

    @override
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

    @override
    def total_rows(self, table: TableReference) -> Cardinality | None:
        rows = self._total_rows.get(table.full_name)
        return None if rows is None else Cardinality(rows)

    @override
    def num_distinct(self, column: ColumnReference) -> int | None:
        return self._num_distinct.get(self._key(column))

    @override
    def null_frac(self, column: ColumnReference) -> float | None:
        return self._null_frac.get(self._key(column))

    @override
    def min_max(self, column: ColumnReference) -> tuple[Any, Any] | None:
        return self._min_max.get(self._key(column))

    @override
    def most_common_values(self, column: ColumnReference) -> MostCommonValues | None:
        return self._mcv.get(self._key(column))

    @override
    def histogram(
        self, column: ColumnReference, *, interpolation: HistogramApproximation = "approx-uni"
    ) -> Histogram | None:
        return self._histograms.get(self._key(column))

    @override
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

    @override
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

    @override
    def format_query(self, query: SqlQuery) -> str:
        return str(query)

    @override
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

    @override
    def query_plan(self, query: SqlQuery | str) -> QueryPlan:
        self.requests.append(query)
        return self._require_plan()

    @override
    def analyze_plan(self, query: SqlQuery) -> QueryPlan:
        self.requests.append(query)
        return self._require_plan()

    @override
    def parse_plan(self, plan: ResultSet, *, query: SqlQuery | None = None) -> QueryPlan | None:
        if isinstance(plan, QueryPlan):
            return plan
        return self._require_plan()

    @override
    def cardinality_estimate(self, query: SqlQuery | str) -> Cardinality:
        self.requests.append(query)
        return self._cardinality

    @override
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

    @override
    def schema(self) -> DatabaseSchema:
        return self._schema

    @override
    def statistics(self) -> StatisticsCatalog:
        return self._statistics

    @override
    def hinting(self) -> HintService:
        return self._hinting

    @override
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

    @override
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

    @override
    def cursor(self) -> Cursor:
        return self._cursor

    @override
    def database_name(self) -> str:
        return self.name

    @override
    def dbms_version(self) -> util.Version:
        return util.Version(self.version)

    @override
    def describe(self) -> jsondict:
        return {"interface-type": "fake", "database": self.name}

    @override
    def reset_connection(self) -> Any:
        self.reset_count += 1
        return None

    @override
    def close(self) -> None:
        self.closed = True

    @override
    def __hash__(self) -> int:
        return hash(id(self))

    @override
    def __eq__(self, other: object) -> bool:
        return self is other
