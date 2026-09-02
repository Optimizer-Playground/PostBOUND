"""Contains the MySQL implementation of the Database interface.

The current implementation has a number of limitations. Some are caused by fundamental restrictions of how MySQL
optimizes and executes queries, while others are caused by the sheer implementation effort that would have to be
invested to implement the corresponding feature in MySQL.

The most important restrictions are as follows:

No support for parsing EXPLAIN ANALYZE plans. Calling the corresponding MysqlOptimizer.analyze_plan method raises a
``NotImplementedError``. This is because MySQL currently (i.e. as of version 8.0) only provides EXPLAIN ANALYZE plans
in TREE output format, which is not exhaustively documented and appears fairly irregular. This makes parsing the
output fairly hard.

Restrictions of the query hint generation: query execution in MySQL differs fundamentally from the way queries are
executed in more traditional systems such as PostgreSQL or Oracle. MySQL makes heavy usage of clustered indexes,
meaning that all tuples in a table are automatically stored in a B-Tree according to the primary key index. As a
consequence, MySQL strongly favors the usage of (Index-) Nested Loop Joins during query execution and rarely resorts to
other operators. In fact, the only fundamentally different join operator available is the Hash Join. This operator is
only used if a equality join should be executed between columns that do not have an index available. Therefore, it is
not possible to disable Nested Loop Joins entirely, nor can the usage of Hash Joins be enforced. Instead, query hints
can only disable the usage of Hash Joins, or *recommend* their usage. But whether or not they are actually applied is
up to the MySQL query optimizer. A similar thing happens for the join order: although MySQL provides a number of hints
related to the join order optimization, these hints are not always enforced. More specifically, to the best of our
knowledge, it is not possible to enforce the branches in the join order and MySQL heavily favors left-deep query plans.
Therefore, the generation of join order hints only works for linear join orders for now.
"""

from __future__ import annotations

import configparser
import dataclasses
import json
import math
import numbers
import os
import textwrap
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

import mysql.connector

from . import db, qal, transform, util
from ._core import (
    BoundColumnReference,
    Cardinality,
    ColumnReference,
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
from ._qep import QueryPlan
from .db import (
    Cursor,
    Database,
    DatabasePool,
    DatabaseSchema,
    DatabaseStatistics,
    HintService,
    Histogram,
    HistogramApproximation,
    MostCommonValues,
    OptimizerInterface,
    PreciseStatistics,
    UnsupportedDatabaseFeatureError,
    simplify_result_set,
)
from .qal import (
    CastExpression,
    Explain,
    Hint,
    SqlExpression,
    SqlQuery,
    StaticValueExpression,
)
from .util import Version

warnings.warn(
    "The MySQL interface is experimental and may be incomplete. Use it at your own risk.",
    UserWarning,
    stacklevel=2,
)


@dataclasses.dataclass(frozen=True)
class MysqlConnectionArguments:
    """Captures all relevant parameters that customize the way the connection to a MySQL instance is establised.

    The only required parameters are the user that should connect to the database and the name of the database to
    connect to.
    See [1]_ for the different parameters' meaning.

    References
    ----------

    .. [1] https://dev.mysql.com/doc/connector-python/en/connector-python-connectargs.html
    """

    user: str
    database: str
    password: str = ""
    host: str = "127.0.0.1"
    port: int = 3306
    use_unicode: bool = True
    charset: str = "utf8mb4"
    autocommit: bool = True
    sql_mode: str = "ANSI"

    def parameters(self) -> dict[str, str | int | bool]:
        """Provides all arguments in one neat ``dict``.

        Returns
        -------
        dict[str, str | int | bool]
            A mapping from parameter name to parameter value.
        """
        return dataclasses.asdict(self)


class MysqlInterface(Database):
    """MySQL-specific implementation of the general `Database` interface."""

    def __init__(
        self,
        connection_args: MysqlConnectionArguments,
        system_name: str = "MySQL",
    ) -> None:
        """Generates a new database interface and establishes a connection to the specified database server.

        Parameters
        ----------
        connection_args : MysqlConnectionArguments
            Configuration and required information to establish a connection to some MySQL instance.
        system_name : str, optional
            The name of the current database. Typically, this can be used to query the `DatabasePool` for this very
            instance. Defaults to ``"MySQL"``.
        cache_enabled : bool, optional
            Whether or not caching of complicated database queries should be enabled by default. Defaults to ``True``.
        """
        self.connection_args = connection_args
        self._cnx = mysql.connector.connect(**connection_args.parameters())
        self._cur = self._cnx.cursor(buffered=True)

        self._db_schema = MysqlSchemaInterface(self)
        self._db_stats = MysqlStatisticsInterface(self)
        super().__init__(system_name)

    def schema(self) -> MysqlSchemaInterface:
        return self._db_schema

    def statistics(self) -> MysqlStatisticsInterface:
        return self._db_stats

    def hinting(self) -> HintService:
        return MysqlHintService(self)

    def execute_query(
        self,
        query: SqlQuery | str,
        *,
        raw: bool = False,
    ) -> Any:
        query = self._prepare_query_execution(query)

        self._cur.execute(query)
        query_result = self._cur.fetchall()

        return query_result if raw else simplify_result_set(query_result)  # type: ignore - mysql-connector shenanigans

    def optimizer(self) -> OptimizerInterface:
        return MysqlOptimizer(self)

    def database_name(self) -> str:
        self._cur.execute("SELECT DATABASE();")
        result_set = self._cur.fetchone()
        assert result_set is not None
        return result_set[0]  # type: ignore - mysql-connector shenanigans

    def dbms_version(self) -> Version:
        self._cur.execute("SELECT VERSION();")
        result_set = self._cur.fetchone()
        assert result_set is not None
        return Version(result_set[0])  # type: ignore - mysql-connector shenanigans

    def server_mode(self) -> str:
        """Provides the current settings in the ``sql_mode`` MySQL variable.

        Returns
        -------
        str
            The ``sql_mode`` value, exactly as it is returned by the server. Typically, this is a list of
            comma-separated features.
        """
        self._cur.execute("SELECT @@session.sql_mode")
        result_set = self._cur.fetchone()
        assert result_set is not None
        return result_set[0]  # type: ignore - mysql-connector shenanigans

    def describe(self) -> dict:
        base_info = {
            "system_name": self.dbms_name(),
            "system_version": self.dbms_version(),
            "database": self.database_name(),
            "statistics": self.statistics().describe(),
        }
        self._cur.execute("SHOW VARIABLES")
        system_config = self._cur.fetchall()
        base_info["system_settings"] = dict(system_config)  # type: ignore - mysql-connector shenanigans
        return base_info

    def reset_connection(self) -> None:
        self._cur.close()
        self._cnx.cmd_reset_connection()
        self._cur = self._cnx.cursor()

    def cursor(self) -> Cursor:
        # the MySQL client is not entirely DB API 2.0 compatible because it violates (for the most part) unimportant
        # details of the execute() method
        return self._cur  # type: ignore

    def close(self) -> None:
        self._cur.close()
        self._cnx.close()

    def _prepare_query_execution(self, query: SqlQuery | str, *, drop_explain: bool = False) -> str:
        """Provides the query in a unified format, taking care of preparatory statements as necessary.

        `drop_explain` can be used to remove any EXPLAIN clauses from the query. Note that all actions that require
        the "semantics" of the query to be known (e.g. EXPLAIN modifications or query hints) and are therefore only
        executed for instances of the qal queries.
        """
        if not isinstance(query, SqlQuery):
            return query

        if drop_explain:
            query = transform.drop_clause(query, Explain)
        if query.hints and query.hints.preparatory_statements:
            self._cur.execute(query.hints.preparatory_statements)
            query = transform.drop_hints(query, preparatory_statements_only=True)
        return str(query)

    def _obtain_query_plan(self, query: str) -> dict:
        if not query.startswith("EXPLAIN FORMAT = JSON"):
            query = "EXPLAIN FORMAT = JSON " + query
        self._cur.execute(query)
        result_set = self._cur.fetchone()
        assert result_set is not None

        return json.loads(result_set[0])  # type: ignore - mysql-connector shenanigans


class MysqlSchemaInterface(DatabaseSchema):
    def __init__(self, mysql_db: MysqlInterface):
        super().__init__(mysql_db)

    def lookup_column(
        self,
        column: ColumnReference | str,
        candidate_tables: Iterable[TableReference],
        *,
        expect_match: bool = False,
    ) -> Optional[TableReference]:
        column = column.name if isinstance(column, ColumnReference) else column

        for table in candidate_tables:
            table_columns = self._fetch_columns(table)
            if column in table_columns:
                return table

        if not expect_match:
            return None
        candidate_strs = [tab.full_name for tab in candidate_tables]
        raise ValueError(f"Column {column} not found in candidate tables {candidate_strs}")

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
        query_template = (
            "SELECT index_name FROM information_schema.statistics WHERE table_name = %s AND column_name = %s"
        )
        self._db.cursor().execute(query_template, (column.table.full_name, column.name))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None
        return {index[0] for index in result_set}

    def foreign_keys_on(self, column: ColumnReference) -> set[BoundColumnReference]:
        if not column.table:
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)
        query_template = (
            "SELECT referenced_table_name, referenced_column_name "
            "FROM information_schema.key_column_usage "
            "WHERE table_name = %s AND column_name = %s AND referenced_column_name IS NOT NULL"
        )
        self._db.cursor().execute(query_template, (column.table.full_name, column.name))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None
        return {BoundColumnReference(col, TableReference(table)) for table, col in result_set}

    def datatype(self, column: ColumnReference, *, raw: bool = False) -> str:
        if not column.table:
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)
        query_template = "SELECT column_type FROM information_schema.columns WHERE table_name = %s AND column_name = %s"
        self._db.cursor().execute(query_template, (column.table.full_name, column.name))
        result_set = self._db.cursor().fetchone()
        assert result_set is not None

        dtype = result_set[0]
        return dtype if raw else dtype.lower()

    def is_nullable(self, column) -> bool:
        if not column.table:
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)
        query_template = "SELECT is_nullable FROM information_schema.columns WHERE table_name = %s AND column_name = %s"
        self._db.cursor().execute(query_template, (column.table.full_name, column.name))
        result_set = self._db.cursor().fetchone()
        assert result_set is not None
        return result_set[0] == "YES"

    def _fetch_columns(self, table: TableReference) -> list[str]:
        query_template = "SELECT column_name FROM information_schema.columns WHERE table_name = %s"
        self._db.cursor().execute(query_template, (table.full_name,))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None
        return [col[0] for col in result_set]

    def _fetch_indexes(self, table: TableReference) -> dict[str, bool]:
        index_query = textwrap.dedent("""
            SELECT column_name, column_key = 'PRI'
            FROM information_schema.columns
            WHERE table_name = %s AND column_key <> ''
        """)
        self._db.cursor().execute(index_query, table.full_name)
        result_set = self._db.cursor().fetchall()
        assert result_set is not None
        index_map = dict(result_set)
        return index_map


class MysqlStatisticsInterface(DatabaseStatistics):
    def __init__(self, mysql_db: MysqlInterface):
        super().__init__()
        self._db = mysql_db

    def total_rows(self, table: TableReference) -> Optional[Cardinality]:
        if table.virtual:
            raise VirtualTableError(table)

        count_query = "SELECT table_rows FROM information_schema.tables WHERE table_name = %s"
        self._db.cursor().execute(count_query, table.full_name)
        result_set = self._db.cursor().fetchone()
        assert result_set is not None
        count = result_set[0]
        return Cardinality.of(count)

    def num_distinct(self, column: ColumnReference) -> Optional[int]:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)

        stats_query = "SELECT cardinality FROM information_schema.statistics WHERE table_name = %s AND column_name = %s"
        self._db.cursor().execute(stats_query, (column.table.full_name, column.name))
        distinct_vals = self._db.cursor().fetchone()
        return distinct_vals[0] if distinct_vals is not None else None

    def null_frac(self, column: ColumnReference) -> Optional[float]:
        if not db.enable_emulation_fallback:
            raise UnsupportedDatabaseFeatureError(
                self._db, "null fraction statistics. Set db.enable_emulation_fallback to activate."
            )
        return PreciseStatistics(self._db).null_frac(column)

    def min_max_(self, column: ColumnReference) -> tuple[Any, Any]:
        if not db.enable_emulation_fallback:
            raise UnsupportedDatabaseFeatureError(
                self._db, "min/max value statistics. Set db.enable_emulation_fallback to activate."
            )
        return PreciseStatistics(self._db).min_max(column)

    def most_common_values(self, column: ColumnReference) -> MostCommonValues:
        if not db.enable_emulation_fallback:
            raise UnsupportedDatabaseFeatureError(
                self._db, "most common values statistics. Set db.enable_emulation_fallback to activate."
            )
        return PreciseStatistics(self._db).most_common_values(column, k=100)

    def histogram(self, column: ColumnReference, *, interpolation: HistogramApproximation = "approx-uni") -> Histogram:
        if not db.enable_emulation_fallback:
            raise UnsupportedDatabaseFeatureError(
                self._db, "null_fraction statistics. Set db.enable_emulation_fallback to activate."
            )
        return PreciseStatistics(self._db).histogram(column, n_bins=100, interpolation=interpolation)


MysqlJoinHints = {JoinOperator.HashJoin, JoinOperator.NestedLoopJoin}
MysqlScanHints = {ScanOperator.IndexScan, ScanOperator.SequentialScan}
MysqlPlanHints = {HintType.LinearJoinOrder, HintType.Operator}


class _MysqlExplainClause(Explain):
    def __init__(self, original_clause: Explain):
        super().__init__(original_clause.analyze, original_clause.target_format)

    def __str__(self) -> str:
        explain_body = ""
        if self.analyze:
            explain_body += " ANALYZE"
        if self.target_format:
            explain_body += f" FORMAT={self.target_format}"
        return "EXPLAIN" + explain_body


class _MysqlStaticValueExpression(StaticValueExpression):
    def __init__(self, original_expression: StaticValueExpression) -> None:
        super().__init__(original_expression.value)

    def __str__(self) -> str:
        return f"{self.value}" if isinstance(self.value, numbers.Number) else f'"{self.value}"'


class _MysqlCastExpression(CastExpression):
    def __init__(self, original_expression: CastExpression) -> None:
        if original_expression.array_type:
            warnings.warn("MySQL does not support array types in CAST expressions. Ignoring array type.")
        super().__init__(original_expression.casted_expression, original_expression.target_type)

    def __str__(self) -> str:
        return f"CAST({self.casted_expression} AS {self.target_type})"


def _replace_static_vals(e: SqlExpression) -> SqlExpression:
    return _MysqlStaticValueExpression(e) if isinstance(e, StaticValueExpression) else e


def _replace_casts(e: SqlExpression) -> SqlExpression:
    return _MysqlCastExpression(e) if isinstance(e, CastExpression) else e


def _generate_join_order_hint(join_order: Optional[JoinTree]) -> str:
    if not join_order:
        return ""

    join_order_text = ", ".join(table.identifier() for table in join_order.itertables())
    return f"  JOIN_ORDER({join_order_text})"


MysqlOptimizerHints = {
    JoinOperator.NestedLoopJoin: "NO_BNL",
    JoinOperator.HashJoin: "BNL",
    ScanOperator.SequentialScan: "NO_INDEX",
    ScanOperator.IndexScan: "INDEX",
    ScanOperator.IndexOnlyScan: "INDEX",
    ScanOperator.BitmapScan: "INDEX_MERGE",
}
"""See https://dev.mysql.com/doc/refman/8.0/en/optimizer-hints.html"""


def _generate_operator_hints(
    physical_operators: Optional[PhysicalOperatorAssignment],
) -> str:
    if not physical_operators:
        return ""
    hints = []

    for table, scan_assignment in physical_operators.scan_operators.items():
        table_key = table.identifier()
        operator = MysqlOptimizerHints[scan_assignment.operator]
        hints.append(f"  {operator}({table_key})")

    for join, join_assignment in physical_operators.join_operators.items():
        join_key = ", ".join(tab.identifier() for tab in join)
        operator = MysqlOptimizerHints[join_assignment.operator]
        hints.append(f"  {operator}({join_key})")

    if physical_operators.intermediate_operators:
        warnings.warn("Cannot generate intermediate operator hints for MySQL.")

    return "\n".join(hints)


MysqlSwitchableOptimizations: Mapping[PhysicalOperator, str] = {JoinOperator.HashJoin: "block_nested_loop"}
"""See https://dev.mysql.com/doc/refman/8.0/en/switchable-optimizations.html"""


def _escape_setting(setting) -> str:
    """Transforms the setting variable into a string that can be used in an SQL query."""
    if isinstance(setting, float) or isinstance(setting, int):
        return str(setting)
    elif isinstance(setting, bool):
        return "TRUE" if setting else "FALSE"
    return f"'{setting}'"


def _generate_prep_statements(
    physical_operators: Optional[PhysicalOperatorAssignment],
    plan_parameters: Optional[PlanParameterization],
) -> str:
    statements = []
    if physical_operators:
        switchable_optimizations = []
        for operator, enabled in physical_operators.global_settings.items():
            value = "on" if enabled else "off"
            switchable_optimizations.append(f"{MysqlSwitchableOptimizations[operator]}={value}")
        if switchable_optimizations:
            optimizer_switch = ",".join(switchable_optimizations)
            statements.append(f"SET @@optimizer_switch='{optimizer_switch}';")

    if plan_parameters:
        for setting, value in plan_parameters.system_settings.items():
            statements.append(f"SET {setting}={_escape_setting(value)};")

    return "\n".join(statements) if statements else ""


class MysqlHintService(HintService):
    def __init__(self, mysql_instance: MysqlInterface) -> None:
        super().__init__()
        self._mysql_instance = mysql_instance

    def generate_hints(
        self,
        query: SqlQuery,
        plan: Optional[QueryPlan] = None,
        *,
        join_order: Optional[JoinTree] = None,
        physical_operators: Optional[PhysicalOperatorAssignment] = None,
        plan_parameters: Optional[PlanParameterization] = None,
    ) -> SqlQuery:
        if join_order and not join_order.is_linear():
            raise UnsupportedDatabaseFeatureError(
                self._mysql_instance,
                "Can only enforce join order for linear join trees for now",
            )

        if plan is not None:
            join_order = jointree_from_plan(plan)
            physical_operators = operators_from_plan(plan)
            plan_parameters = parameters_from_plan(plan)

        join_order_hint = _generate_join_order_hint(join_order)
        operator_hint = _generate_operator_hints(physical_operators)
        prep_statements = _generate_prep_statements(physical_operators, plan_parameters)

        if not join_order_hint and not operator_hint:
            return query

        final_hint_block = "/*+\n" + "\n".join(hint for hint in (join_order_hint, operator_hint) if hint) + "\n*/"
        hint_clause = Hint(prep_statements, final_hint_block)
        return transform.add_clause(query, hint_clause)

    def format_query(self, query: SqlQuery) -> str:
        updated_query = query

        if updated_query.is_explain():
            assert query.explain is not None
            transform.replace_clause(query, _MysqlExplainClause(query.explain))

        if "ANSI_QUOTES" not in self._mysql_instance.server_mode():
            updated_query = transform.replace_expressions(updated_query, _replace_static_vals)
        updated_query = transform.replace_expressions(updated_query, _replace_casts)

        return qal.format_quick(updated_query, inline_hint_block=True)

    def supports_hint(self, hint: PhysicalOperator | HintType) -> bool:
        return hint in MysqlJoinHints | MysqlScanHints | MysqlPlanHints


class MysqlOptimizer(OptimizerInterface):
    def __init__(self, mysql_instance: MysqlInterface) -> None:
        self._mysql_instance = mysql_instance

    def query_plan(self, query: SqlQuery | str) -> QueryPlan:
        if isinstance(query, SqlQuery):
            prepared_query = self._mysql_instance._prepare_query_execution(query, drop_explain=True)
            query_for_plan = query
        else:
            prepared_query = query
            query_for_plan = None
        raw_query_plan = self._mysql_instance._obtain_query_plan(prepared_query)
        query_plan = parse_mysql_explain_plan(query_for_plan, raw_query_plan)
        return query_plan.as_qep()

    def analyze_plan(self, query: SqlQuery) -> QueryPlan:
        raise NotImplementedError("MySQL interface does not support ANALYZE plans yet")

    def parse_plan(self, plan: Any, *, query: Optional[SqlQuery] = None) -> QueryPlan:
        mysql_plan = MysqlExplainPlan(plan)
        return mysql_plan.as_qep()

    def cardinality_estimate(self, query: SqlQuery | str) -> Cardinality:
        return self.query_plan(query).estimated_cardinality

    def cost_estimate(self, query: SqlQuery | str) -> float:
        return self.query_plan(query).estimated_cost


def _parse_mysql_connection(config_file: str) -> MysqlConnectionArguments:
    config = configparser.ConfigParser()
    config.read(config_file)
    if "MYSQL" not in config:
        raise ValueError("Malformed MySQL config file: no [MYSQL] section found.")
    mysql_config = config["MYSQL"]

    if "User" not in mysql_config or "Database" not in mysql_config:
        raise ValueError("Malformed MySQL config file: 'User' and 'Database' keys are required in the [MYSQL] section.")
    user = mysql_config["User"]
    database = mysql_config["Database"]

    optional_settings: dict[str, Any] = {}
    for key in [
        "Password",
        "Host",
        "Port",
        "UseUnicode",
        "Charset",
        "AutoCommit",
        "SqlMode",
    ]:
        if key not in mysql_config:
            continue
        optional_settings[util.camel_case2snake_case(key)] = mysql_config[key]
    return MysqlConnectionArguments(user, database, **optional_settings)


def connect(
    *,
    name: str = "mysql",
    connection_args: Optional[MysqlConnectionArguments] = None,
    config_file: str = ".mysql_connection.config",
    private: bool = False,
) -> MysqlInterface:
    db_pool = DatabasePool.get_instance()
    if config_file and not connection_args:
        if not os.path.exists(config_file):
            raise ValueError("Config file was given, but does not exist: " + config_file)
        connection_args = _parse_mysql_connection(config_file)
    elif not connection_args:
        raise ValueError("Connect string or config file are required to connect to MySQL")

    mysql_db = MysqlInterface(connection_args, system_name=name)
    if not private:
        db_pool.register_database(name, mysql_db)
    return mysql_db


# The next several functions are concerned with MySQL EXPLAIN query plans. Although in theory MySQL offers some great
# tools to inspect query plans produced by the optimizer (having 3 different output formats: tabular, human-readable
# plan trees and JSON data), these output formats differ in the information they provide. Only the JSON format provides
# all the details that we are interested in (and makes them harder to access then when using the tree output for
# example).
# Sadly, the JSON output is not available when using EXPLAIN ANALYZE to match the optimizer's expectation
# with the reality encoutered upon query execution. Since parsing the EXPLAIN trees is quite difficult, we restrict
# ourselves to plain EXPLAIN plans for now and maybe integrate EXPLAIN ANALYZE plans in the future along with a
# dedicated parser for its structure.
# What makes the situation with the JSON-formatted EXPLAIN plans pretty bad is the fact that the structure of the
# provided JSON document is barely documented and seems incosistent at best (see
# https://mariadb.com/kb/en/explain-format-json/ for example). Therefore, our JSON-based parser strongly follows a
# similar implementation, namely the "visual explain" feature of the MySQL Workbench. They also need to traverse the
# JSON-based EXPLAIN plans, but this time to generate a graphical representation of the information. Still, the
# traversal and attribute access logic can be re-used by a great deal. It is even implemented in Python! Nevertheless,
# the code there is often barely documented so a lot of guesswork is still left for us to do. See
# https://github.com/mysql/mysql-workbench/blob/8.0/plugins/wb.query.analysis/explain_renderer.py for the Workbench
# implementation that our code is based on. The best explanation of how the different attributes in the JSON document
# should be interpreted is contained in the MySQL worklog entry to implement parts of the JSON output:
# https://dev.mysql.com/worklog/task/?id=6510


def _lookup_table(alias: str, candidate_tables: Iterable[TableReference]) -> TableReference:
    """Searches for a specific table in a list of candidate tables.

    If no candidate table has the given `alias`, the full names are used instead. If still no table matches, a
    `KeyError` is raised.

    This function is necessary, because MySQL does not contain the complete table names in the output. If that were
    the case, we could construct our `TableReference` objects directly based on this information. Instead, MySQL
    provides the "identifier" of the tables, i.e. the alias if the tables was aliased or the full name otherwise. In
    order to build the correct `TableReference` objects that also line up with the tables contained in the `SqlQuery`
    object for the same query, we need to take this detour and lookup the correct tables.

    Parameters
    ----------
    alias : str
        The table alias to search for. This does not have to be an alias, but could be a full table name just as well.
    candidate_tables : Iterable[TableReference]
        The tables that could potentially have the given alias. `_lookup_table` assumes that at least one of the
        candidates matches.

    Returns
    -------
    TableReference
        The table with the given alias or full name.
    """
    table_map = {tab.full_name: tab for tab in candidate_tables}

    # alias takes precedence over full_name in case of conflicts
    table_map |= {tab.alias: tab for tab in candidate_tables}
    return table_map[alias]


_MysqlExplainNodeTypes = {
    "nested_loop",
    "table",
    "optimized_away_subqueries",
    "grouping_operation",
    "ordering_operation",
    "duplicate_removal",
    "union_result",
    "buffer_result",
    "select_list_subqueries",
}
"""The different nodes that can occurr in the MySQL EXPLAIN output which correspond to actual operators.

Derived from ExplainContext.handle_query_block in mysql_renderer.py
"""

_MysqlMetadataNodes = {
    "cost_info",
    "rows_examined_per_scan",
    "rows_produced_per_join",
    "filtered",
}
"""The metadata contained in the MySQL EXPLAIN output that we are interested in.

For some reason, the MySQL authors decided that it was a good idea to merge this information with the normal operator
nodes and not denote the operator tree in any special way.
"""

_Cost, _IdxLookup, _IdxMerge, _TabScan = (
    "Const",
    "Index Lookup",
    "Index Merge",
    "Table Scan",
)

_MysqlJoinSourceTypes = {
    "system": _Cost,
    "const": _Cost,
    "eq_ref": _IdxLookup,
    "ref": _IdxLookup,
    "fulltext": _IdxLookup,
    "ref_or_null": _IdxLookup,
    "index_merge": _IdxMerge,
    "unique_subquery": _IdxLookup,
    "index_subquery": _IdxLookup,
    "range": _IdxLookup,
    "index": _IdxLookup,
    "ALL": _TabScan,
}
"""The different ways (Nested Loop) joins can be executed with a single input table.

See https://dev.mysql.com/doc/refman/8.0/en/explain-output.html#explain-join-types for details
"""


_MysqlJoinTypes = {
    "Block Nested Loop": "Block Nested Loop",
    "Batched Key Access": "Batched Key Access",
    "Batched Key Access (unique)": "Batched Key Access",
    "hash join": "Hash Join",  # the lower-case is intentional and not a bug..
}
"""The different join algorithms supported by MySQL.

See https://dev.mysql.com/doc/refman/8.0/en/explain-output.html#explain-extra-information for the listing.
"""


def _parse_cost_info(explain_data: dict) -> tuple[float, float]:
    """Extracts the relevant cost information from a MySQL EXPLAIN node.

    Parameters
    ----------
    explain_data : dict
        The current EXPLAIN node. Nodes without cost information are handled gracefully.

    Returns
    -------
    tuple[float, float]
        A tuple of ``(scan cost, join cost)``. Remember that MySQL merges join nodes and scan nodes in the JSON-based
        EXPLAIN output. If the node does not contain any cost information, a ``NaN`` tuple will be returned instead.
    """
    if "cost_info" not in explain_data:
        return math.nan, math.nan
    cost_info: dict = explain_data["cost_info"]

    read_cost = cost_info.get("read_cost", "")
    read_cost = float(read_cost) if read_cost else 0

    eval_cost = cost_info.get("eval_cost", "")
    eval_cost = float(eval_cost) if eval_cost else 0

    scan_cost = read_cost + eval_cost
    scan_cost = scan_cost if scan_cost else math.nan

    join_cost = cost_info.get("prefix_cost", "")
    join_cost = float(join_cost) if join_cost else math.nan
    return scan_cost, join_cost


def _parse_cardinality_info(explain_data: dict) -> tuple[float, float]:
    """Extracts the relevant cardinality information from a MySQL EXPLAIN node.

    Parameters
    ----------
    explain_data : dict
        The current EXPLAIN node. Nodes without cardinality information are handled gracefully.

    Returns
    -------
    tuple[float, float]
        A tuple of ``(scan cardinality, join cardinality)``. Remember that MySQL merges join nodes and scan nodes in
        the JSON-based EXPLAIN output. The scan cardinality accounts for all filter predicates. If no scan or join
        cardinality can be determined, a ``NaN`` is used instead.
    """
    table_cardinality = explain_data.get("rows_examined_per_scan", "")
    table_cardinality = float(table_cardinality) if table_cardinality else math.nan

    filtered = explain_data.get("filtered")
    filtered = float(filtered) if filtered else math.nan
    selectivity = filtered / 100
    scan_cardinality = selectivity * table_cardinality

    join_cardinality = explain_data.get("rows_produced_per_join", "")
    join_cardinality = float(join_cardinality) if join_cardinality else math.nan
    return scan_cardinality, join_cardinality


def _determine_join_type(explain_data: dict) -> str:
    if "using_join_buffer" not in explain_data:
        return "Nested Loop"
    return _MysqlJoinTypes[explain_data["using_join_buffer"]]


def _parse_mysql_join_node(query: Optional[SqlQuery], node_name: str, explain_data: list) -> Optional[MysqlExplainNode]:
    first_table, *remaining_tables = explain_data
    first_node = _parse_next_mysql_explain_node(query, first_table)
    if first_node is None:
        raise ValueError(f"Could not parse first table in join node {first_table}")

    current_node = first_node
    for next_table in remaining_tables:
        next_node = _parse_next_mysql_explain_node(query, next_table)
        if next_node is None:
            raise ValueError(f"Could not parse next table in join node {next_table}")

        current_node.next_node = next_node
        current_node = next_node

    return first_node


def _parse_mysql_table_node(
    query: Optional[SqlQuery], node_name: str, explain_data: dict
) -> Optional[MysqlExplainNode]:
    scanned_table = _lookup_table(explain_data["table_name"], query.tables()) if query is not None else None
    scan_type = _MysqlJoinSourceTypes[explain_data["access_type"]]  # tables are mostly scanned as part of a join
    join_type = _determine_join_type(explain_data)
    scan_cost, join_cost = _parse_cost_info(explain_data)
    scan_card, join_card = _parse_cardinality_info(explain_data)

    subquery = (
        _parse_next_mysql_explain_node(query, explain_data["materialized_from_subquery"])
        if "materialized_from_subquery" in explain_data
        else None
    )
    table_node = MysqlExplainNode(
        scan_type,
        join_type,
        table=scanned_table,
        scan_cost=scan_cost,
        join_cost=join_cost,
        scan_cardinality_estimate=scan_card,
        join_cardinality_estimate=join_card,
        subquery_node=subquery,
    )
    return table_node


def _parse_mysql_wrapper_node(
    query: Optional[SqlQuery], node_name: str, explain_data: dict
) -> Optional[MysqlExplainNode]:
    scan_cost, join_cost = _parse_cost_info(explain_data)
    scan_card, join_card = _parse_cardinality_info(explain_data)
    source_node = _parse_next_mysql_explain_node(query, explain_data)
    pretty_node_name = node_name.replace("_", " ").title()  # "grouping_operation" -> "Grouping Operation"
    return MysqlExplainNode(
        subquery_node=source_node,
        node_type=pretty_node_name,
        scan_cost=scan_cost,
        join_cost=join_cost,
        scan_cardinality_estimate=scan_card,
        join_cardinality_estimate=join_card,
    )


def _parse_mysql_explain_node(
    query: Optional[SqlQuery], node_name: str, explain_data: dict | list
) -> Optional[MysqlExplainNode]:
    if not explain_data:
        return None

    if node_name == "nested_loop":
        assert isinstance(explain_data, list)
        return _parse_mysql_join_node(query, node_name, explain_data)
    elif node_name == "table":
        assert isinstance(explain_data, dict)
        return _parse_mysql_table_node(query, node_name, explain_data)
    else:
        assert isinstance(explain_data, dict)
        explain_data = explain_data["query_block"] if "query_block" in explain_data else explain_data
        return _parse_mysql_wrapper_node(query, node_name, explain_data)


def _parse_next_mysql_explain_node(query: Optional[SqlQuery], explain_data: dict) -> Optional[MysqlExplainNode]:
    for info_key, node_data in explain_data.items():
        if info_key in _MysqlExplainNodeTypes:
            return _parse_mysql_explain_node(query, info_key, node_data)
    raise ValueError("No known node found: " + str(explain_data))


def parse_mysql_explain_plan(query: Optional[SqlQuery], explain_data: dict) -> MysqlExplainPlan:
    explain_data = explain_data["query_block"]
    query_cost = explain_data.get("cost_info", {}).get("query_cost", math.nan)

    # the EXPLAIN plan should only have a single root node, but we do not know which operator it is (the JSON document
    # contains the nodes directly as keys, not under a normalized name, remember?). Therefore, we simply iterate over
    # all entries in the JSON document and check if the current key is a valid operator name. This is exactly, what
    # _parse_next_mysql_explain_node does.
    plan_root = _parse_next_mysql_explain_node(query, explain_data)
    assert plan_root is not None
    return MysqlExplainPlan(plan_root, query_cost)


_MysqlExplainScanNodes = {
    _IdxLookup: ScanOperator.IndexScan,
    _IdxMerge: ScanOperator.BitmapScan,
    _TabScan: ScanOperator.SequentialScan,
}


_MysqlExplainJoinNodes = {
    "Block Nested Loop": JoinOperator.NestedLoopJoin,
    "Batched Key Access": JoinOperator.NestedLoopJoin,
    "Hash Join": JoinOperator.HashJoin,
}


def _node_sequence_to_qep(nodes: Sequence[MysqlExplainNode]) -> QueryPlan:
    if not nodes:
        raise ValueError("Cannot convert an empty node sequence to a query plan.")

    if len(nodes) == 1:
        return nodes[0]._make_qep_node_for_scan()

    if len(nodes) == 2:
        final_table, first_table = nodes
        final_qep = final_table._make_qep_node_for_scan()
        first_qep = first_table._make_qep_node_for_scan()
        join_operator = _MysqlExplainJoinNodes.get(final_table.join_type, JoinOperator.NestedLoopJoin)
        join_node = QueryPlan(
            final_table.join_type,
            operator=join_operator,
            children=[first_qep, final_qep],
            estimated_cost=final_table.join_cost,
            estimated_cardinality=Cardinality(final_table.join_cardinality_estimate),
        )
        return join_node

    if len(nodes) > 2:
        final_table, *former_tables = nodes
        former_qep = _node_sequence_to_qep(former_tables)
        final_qep = final_table._make_qep_node_for_scan()

        join_operator = _MysqlExplainJoinNodes.get(final_table.join_type, JoinOperator.NestedLoopJoin)
        join_node = QueryPlan(
            final_table.join_type,
            operator=join_operator,
            children=[former_qep, final_qep],
            estimated_cost=final_table.join_cost,
            estimated_cardinality=Cardinality(final_table.join_cardinality_estimate),
        )
        return join_node

    assert False, "Should never reach this point, because all cases are handled above."


class MysqlExplainNode:
    def __init__(
        self,
        scan_type: str = "",
        join_type: str = "",
        next_node: Optional[MysqlExplainNode] = None,
        *,
        node_type: Optional[str] = None,
        table: Optional[TableReference] = None,
        scan_cost: float = math.nan,
        join_cost: float = math.nan,
        scan_cardinality_estimate: float = math.nan,
        join_cardinality_estimate: float = math.nan,
        subquery_node: Optional[MysqlExplainNode] = None,
    ) -> None:
        self.scan_type = scan_type
        self.join_type = join_type
        self.node_type = node_type
        self.next_node = next_node
        self.table = table
        self.scan_cost = scan_cost
        self.join_cost = join_cost
        self.scan_cardinality_estimate = scan_cardinality_estimate
        self.join_cardinality_estimate = join_cardinality_estimate
        self.subquery = subquery_node

    def as_qep(self) -> QueryPlan:
        if self.node_type is not None:
            subquery_plan = [self.subquery.as_qep()] if self.subquery is not None else []
            own_node = QueryPlan(
                self.node_type,
                base_table=self.table,
                children=subquery_plan,
                estimated_cost=self.join_cost,
                estimated_cardinality=Cardinality(self.join_cardinality_estimate),
            )
            return own_node

        if not self.next_node:
            return self._make_qep_node_for_scan()

        node_sequence = self._collect_node_sequence()
        return _node_sequence_to_qep(node_sequence)

    def inspect(self, *, _indendation: int = 0) -> str:
        prefix = " " * _indendation + "-> " if _indendation else ""
        own_str = f"{prefix}{self}" if prefix else self._scan_str()

        if self.subquery is not None:
            subquery_str = self.subquery.inspect(_indendation=_indendation)
            return "\n".join((own_str, subquery_str))

        if self.next_node is None:
            return own_str

        next_str = self.next_node.inspect(_indendation=_indendation + 2)
        return "\n".join((own_str, next_str))

    def _collect_node_sequence(self) -> list[MysqlExplainNode]:
        if not self.next_node:
            return [self]
        return self.next_node._collect_node_sequence() + [self]

    def _make_qep_node_for_scan(self) -> QueryPlan:
        return QueryPlan(
            self.scan_type,
            base_table=self.table,
            operator=_MysqlExplainScanNodes.get(self.scan_type),
            estimated_cost=self.scan_cost,
            estimated_cardinality=Cardinality(self.scan_cardinality_estimate),
        )

    def _join_str(self) -> str:
        if self.node_type is not None:
            join_str = (
                f"Join[cost={self.join_cost}, cardinality={self.join_cardinality_estimate}]"
                if not math.isnan(self.join_cost) or not math.isnan(self.join_cardinality_estimate)
                else ""
            )
        else:
            join_str = f"{self.join_type} [cost={self.join_cost}, cardinality={self.join_cardinality_estimate}]"
        return join_str

    def _scan_str(self) -> str:
        if self.node_type is not None:
            scan_str = (
                f"Scan[cost={self.scan_cost}, cardinality={self.scan_cardinality_estimate}]"
                if not math.isnan(self.scan_cost) or not math.isnan(self.scan_cardinality_estimate)
                else ""
            )
        else:
            scan_str = f"{self.scan_type} [cost={self.scan_cost}, cardinality={self.scan_cardinality_estimate}]"
            if self.table is not None:
                scan_str += f" ON {self.table}"
        return scan_str

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        join_str, scan_str = self._join_str(), self._scan_str()
        if self.node_type is not None:
            node_str = str(self.node_type)
            if join_str:
                node_str += " " + join_str
            if scan_str:
                node_str += " " + scan_str
            return node_str

        return f"{join_str} USING {scan_str}"


class MysqlExplainPlan:
    def __init__(self, root: MysqlExplainNode, total_cost: Optional[float] = None) -> None:
        self.root = root
        self.total_cost = total_cost

    def as_qep(self) -> QueryPlan:
        return self.root.as_qep()

    def inspect(self) -> str:
        return self.root.inspect()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Plan cost={self.total_cost}, Root={self.root}"
