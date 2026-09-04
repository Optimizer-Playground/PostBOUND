"""Tests for PostBOUND's query abstraction layer.

The tests here mainly act as regression tests to ensure that SqlQuery objects provide the correct information if
queried for joins, etc.
"""

from __future__ import annotations

import os
import pathlib
import textwrap
import unittest
from collections.abc import Iterable, Sequence
from typing import ClassVar

import postbound as pb
from postbound import BoundColumnReference, ColumnReference, TableReference

from . import regression_suite

pg_connect_dir = "."


class SqlQueryTests(unittest.TestCase):
    def test_dependent_subqueries(self) -> None:
        dependent_query = "SELECT * FROM R WHERE EXISTS (SELECT * FROM S WHERE R.a = S.b)"
        parsed_query = pb.parse_query(dependent_query)
        subqueries = parsed_query.subqueries()
        self.assertTrue(len(subqueries) == 1, "Should detect 1 subquery")
        subquery = pb.util.simplify(subqueries)
        self.assertTrue(subquery.is_dependent(), "Dependent query not recognized as dependent")

        independent_query = "SELECT * FROM R WHERE R.a = 42"
        parsed_query = pb.parse_query(independent_query)
        self.assertFalse(parsed_query.is_dependent(), "Independent query recognized as dependent")

    def test_join_detection(self) -> None:
        simple_query = "SELECT * FROM R, S WHERE R.a = 42 AND R.b = S.c"
        parsed_query = pb.parse_query(simple_query)
        self.assertTrue(len(parsed_query.predicates().joins()) == 1, "Should detect 1 join")

        subquery_join_query = "SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S WHERE R.c = S.d)"
        parsed_query = pb.parse_query(subquery_join_query)
        self.assertTrue(
            len(parsed_query.predicates().joins()) == 0,
            "Should treat dependent subqueries as filters",
        )

    def test_filter_detection(self) -> None:
        simple_query = "SELECT * FROM R, S WHERE R.a = 42 AND R.b = S.c"
        parsed_query = pb.parse_query(simple_query)
        self.assertTrue(len(parsed_query.predicates().filters()) == 1, "Should detect 1 filter")

        subquery_join_query = "SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S WHERE R.c = S.d)"
        parsed_query = pb.parse_query(subquery_join_query)
        self.assertTrue(
            len(parsed_query.predicates().filters()) == 1,
            "Should detect filters for dependent subquery",
        )

        independent_subquery_query = "SELECT * FROM R WHERE R.a = (SELECT MIN(S.b) FROM S)"
        parsed_query = pb.parse_query(independent_subquery_query)
        self.assertTrue(
            len(parsed_query.predicates().filters()) == 1,
            "Should detect 1 filter for independent subquery",
        )


class PredicateTests(unittest.TestCase):
    def test_binary_predicate_joins(self) -> None:
        query = "SELECT * FROM R, S WHERE R.a = S.b"
        with self.subTest("Direct equi join", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a < S.b"
        with self.subTest("Direct non-equi join", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a = 42"
        with self.subTest("Direct filter", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE some_udf(R.a, S.b) = 42"
        with self.subTest("Join in UDF", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a = some_udf(S.b)"
        with self.subTest("Join with UDF result", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE some_udf(R.a) = 42"
        with self.subTest("Filter with UDF", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE R.a = S.b::integer"
        with self.subTest("Join with casted value", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a = (SELECT MIN(T.c) FROM T)"
        with self.subTest("Filter with subquery", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE R.a = (SELECT MIN(T.c) FROM T WHERE T.c = S.b)"
        with self.subTest("Filter with dependent subquery", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

    def test_between_predicate(self) -> None:
        query = "SELECT * FROM R, S WHERE R.a BETWEEN 24 AND 42"
        with self.subTest("Direct BETWEEN filter", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE R.a BETWEEN 24 AND S.b"
        with self.subTest("Direct BETWEEN join, end", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a BETWEEN S.b AND S42"
        with self.subTest("Direct BETWEEN join, start", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a BETWEEN S.b AND S.b + 42"
        with self.subTest("Direct BETWEEN join, both ends", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a BETWEEN 24 AND (SELECT MIN(T.c) FROM T)"
        with self.subTest("BETWEEN filter with subquery", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

    def test_in_predicate(self) -> None:
        query = "SELECT * FROM R, S WHERE R.a IN (1, 2, 3)"
        with self.subTest("Direct IN filter", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE R.a IN (1, S.b, 3)"
        with self.subTest("Join in values list", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a IN (SELECT T.b FROM T WHERE T.c = 42)"
        with self.subTest("IN filter for independent subquery", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

            query = "SELECT * FROM R, S WHERE R.a IN (SELECT T.b FROM T WHERE T.c = T.d)"
            with self.subTest("IN filter for dependent subquery", query=query):
                parsed = pb.parse_query(query)
                self.assertTrue(len(parsed.predicates().joins()) == 0)
                self.assertTrue(len(parsed.predicates().filters()) == 1)

    def test_unary_predicate(self) -> None:
        query = "SELECT * FROM R WHERE EXISTS (SELECT * FROM S WHERE R.a = S.b)"
        with self.subTest("EXISTS for dependent subquery", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R WHERE NOT EXISTS (SELECT * FROM S WHERE R.a = S.b)"
        with self.subTest("NOT EXISTS for dependent subquery", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE my_udf(R.a)"
        with self.subTest("Unary UDF filter", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE my_udf(R.a, S.b)"
        with self.subTest("Unary UDF join", query=query):
            parsed = pb.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

    def test_join_false_positives(self) -> None:
        query = textwrap.dedent("""
                                SELECT * FROM R
                                WHERE R.a = R.b
                                    AND my_udf_pred1(R.a)
                                    AND my_udf_pred2(R.a, R.c)
                                    AND R.a IN (24, R.b, 42)
                                """)
        parsed = pb.parse_query(query)
        self.assertFalse(parsed.predicates().joins())
        self.assertTrue(len(parsed.predicates().filters()) == 4)

    def test_nested_conjunction_disjunction_predicates(self) -> None:
        query = textwrap.dedent("""
                                SELECT *
                                FROM title t
                                    JOIN movie_info mi ON t.id = mi.movie_id
                                WHERE t.production_year < 2021
                                  OR t.season_nr > 10
                                  AND mi.info LIKE '%cartoon%'
                                  """)
        parsed = pb.parse_query(query)

        self.assertIsNotNone(parsed.where_clause)
        root_predicate = parsed.where_clause.root  # type: ignore

        self.assertIsInstance(
            root_predicate,
            pb.qal.OrPredicate,
            "OR predicate should come first",
        )


class MockSchemaLookup(pb.db.DatabaseSchema):
    LookupData: ClassVar[dict[str, TableReference]] = {
        "a": TableReference("r"),
        "b": TableReference("r"),
        "c": TableReference("s"),
    }

    def __init__(self) -> None:
        super().__init__(None)

    def tables(
        self, *, catalog: str = "", schema: str = "", include_system_tables: bool = False
    ) -> set[TableReference]:
        return set(MockSchemaLookup.LookupData.values())

    def lookup_column(
        self, column: ColumnReference | str, candidate_tables: Iterable[TableReference], *, expect_match: bool = False
    ) -> TableReference:
        column = column.name if isinstance(column, ColumnReference) else column
        return MockSchemaLookup.LookupData[column]

    def columns(self, table: TableReference | str) -> Sequence[BoundColumnReference]:
        table = TableReference(table) if isinstance(table, str) else table
        if table == TableReference("r"):
            return [ColumnReference("a", table), ColumnReference("b", table)]
        elif table == TableReference("s"):
            return [ColumnReference("c", table)]
        else:
            raise ValueError(f"Unknown table {table}")


class TransformationTests(unittest.TestCase):
    def test_column_binding(self) -> None:
        """Column binding happens automatically during parsing.

        Therefore we parse normally and test the binding results for normal queries explicitly in this method.
        """
        tab_r = TableReference("R")
        col_r_a = ColumnReference("a", tab_r)
        col_r_b = ColumnReference("b", tab_r)

        tab_s = TableReference("S")
        col_s_c = ColumnReference("c", tab_s)

        query = "SELECT R.a FROM R WHERE R.b = 42"
        with self.subTest("Simple binding through full name", query=query):
            parsed = pb.parse_query(query, bind_columns=False)  # bind_columns refers to live binding

            self.assertSetEqual(parsed.select_clause.columns(), {col_r_a})

            assert parsed.where_clause is not None  # also narrows the Optional for the type checker
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_b})

            self.assertSetEqual(parsed.columns(), {col_r_a, col_r_b})
            self.assertSetEqual(parsed.tables(), {tab_r})

        query = "SELECT S.c FROM R, S WHERE R.a = S.c AND R.b = 42"
        with self.subTest("Simple binding in join", query=query):
            parsed = pb.parse_query(query, bind_columns=False)  # bind_columns refers to live binding

            self.assertSetEqual(parsed.select_clause.columns(), {col_s_c})

            assert parsed.where_clause is not None  # also narrows the Optional for the type checker
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_a, col_r_b, col_s_c})

            self.assertSetEqual(parsed.columns(), {col_r_a, col_r_b, col_s_c})
            self.assertSetEqual(parsed.tables(), {tab_r, tab_s})

    def test_alias_binding(self) -> None:
        """Column binding happens automatically during parsing.

        Therefore we parse normally and test the binding results for aliased queries explicitly in this method.
        """
        tab_r = TableReference("R", "r_tab")
        col_r_a = ColumnReference("a", tab_r)
        col_r_b = ColumnReference("b", tab_r)

        tab_s = TableReference("S")  # no alias!
        col_s_c = ColumnReference("c", tab_s)

        query = "SELECT r_tab.a FROM R r_tab WHERE r_tab.b = 42"
        with self.subTest("Simple binding through alias", query=query):
            parsed = pb.parse_query(query, bind_columns=False)  # bind_columns refers to live binding

            self.assertSetEqual(parsed.select_clause.columns(), {col_r_a})

            assert parsed.where_clause is not None  # also narrows the Optional for the type checker
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_b})

            self.assertSetEqual(parsed.columns(), {col_r_a, col_r_b})
            self.assertSetEqual(parsed.tables(), {tab_r})

        query = "SELECT S.c FROM R r_tab, S WHERE r_tab.a = S.c AND r_tab.b = 42"
        with self.subTest("Simple binding in join", query=query):
            parsed = pb.parse_query(query, bind_columns=False)  # bind_columns refers to live binding

            self.assertSetEqual(parsed.select_clause.columns(), {col_s_c})

            assert parsed.where_clause is not None  # also narrows the Optional for the type checker
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_a, col_r_b, col_s_c})

            self.assertSetEqual(parsed.columns(), {col_r_a, col_r_b, col_s_c})
            self.assertSetEqual(parsed.tables(), {tab_r, tab_s})

    def test_schema_binding(self) -> None:
        """Column binding happens automatically during parsing.

        Therefore we parse normally and test the binding results explicitly in this method. In contrast to the other
        binding tests, these tests here focus on the binding that happens for unbound columns via the database schema.
        """
        tab_r = TableReference("R")
        col_r_a = ColumnReference("a", tab_r)
        col_r_b = ColumnReference("b", tab_r)

        tab_s = TableReference("S")
        col_s_c = ColumnReference("c", tab_s)

        query = "SELECT a FROM R WHERE b = 42"
        with self.subTest("Simple binding through full name", query=query):
            parsed = pb.parse_query(query, db_schema=MockSchemaLookup(), bind_columns=True)

            self.assertSetEqual(parsed.select_clause.columns(), {col_r_a})

            assert parsed.where_clause is not None  # also narrows the Optional for the type checker
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_b})

            self.assertSetEqual(parsed.columns(), {col_r_a, col_r_b})
            self.assertSetEqual(parsed.tables(), {tab_r})

        query = "SELECT c FROM R, S WHERE a = S.c AND R.b = 42"
        with self.subTest("Simple binding in join", query=query):
            parsed = pb.parse_query(query, db_schema=MockSchemaLookup(), bind_columns=True)

            self.assertSetEqual(parsed.select_clause.columns(), {col_s_c})

            assert parsed.where_clause is not None  # also narrows the Optional for the type checker
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_a, col_r_b, col_s_c})

            self.assertSetEqual(parsed.columns(), {col_r_a, col_r_b, col_s_c})
            self.assertSetEqual(parsed.tables(), {tab_r, tab_s})

    def test_subquery_binding(self) -> None:
        tab_r = TableReference("R")
        col_r_a = ColumnReference("a", tab_r)
        col_r_b = ColumnReference("b", tab_r)

        tab_s = TableReference("S")
        col_s_b = ColumnReference("b", tab_s)
        col_s_c = ColumnReference("c", tab_s)

        query = "SELECT SUM(R.b) FROM R WHERE R.a < (SELECT MIN(S.c) FROM S)"
        all_cols = {col_r_a, col_r_b, col_s_c}
        with self.subTest("Column binding in anonymous subquery", query=query):
            parsed = pb.parse_query(query)
            self.assertSetEqual(parsed.columns(), all_cols)

        query = "SELECT SUM(R.b) FROM R, (SELECT S.b, MIN(S.c) FROM S GROUP BY S.b) sub_s WHERE R.a = sub_s.b"
        sub_s = TableReference.create_virtual("sub_s")
        col_sub_s_b = ColumnReference("b", sub_s)
        all_cols = {col_r_a, col_r_b, col_s_b, col_s_c, col_sub_s_b, col_sub_s_b}
        with self.subTest("Column binding in virtual subquery", query=query):
            parsed = pb.parse_query(query)
            self.assertSetEqual(parsed.columns(), all_cols)

        query = "WITH s_cte AS (SELECT MIN(S.c) FROM S) SELECT SUM(R.b) FROM R WHERE R.a = s_cte.c"
        cte_s = TableReference.create_virtual("s_cte")
        col_cte_s_c = ColumnReference("c", cte_s)
        all_cols = {col_r_a, col_r_b, col_s_c, col_cte_s_c}
        with self.subTest("Column binding in CTE", query=query):
            parsed = pb.parse_query(query)
            self.assertSetEqual(parsed.columns(), all_cols)

    def test_end_to_end_binding(self) -> None:
        query = """
            WITH cte_a AS (SELECT R.r_a FROM R WHERE R.r_a < 42),
                cte_b AS (SELECT SUM(S.s_b) FROM S GROUP BY S.s_a)
            SELECT AVG(A.a_y * B.b_y)
            FROM A, B, cte_a, (SELECT M.m_a + N.n_a AS total FROM M, N WHERE M.m_b = N.n_b) AS sq_m_n
            WHERE A.a_x <> 100
                AND A.a_b = B.b_b
                AND B.b_a = cte_a.r_a
                AND A.a_t = sq_m_n.total
            GROUP BY A.a_z
            """
        parsed = pb.parse_query(query)
        expected_tables = {
            TableReference("R"),
            TableReference("S"),  # WITH clauses
            TableReference("A"),
            TableReference("B"),  # FROM clauses
            TableReference("M"),
            TableReference("N"),  # subquery tables
            TableReference.create_virtual("cte_a"),  # CTEs
            TableReference.create_virtual("cte_b"),
            TableReference.create_virtual("sq_m_n"),  # subqueries
        }  # subquery aliases
        self.assertEqual(parsed.tables(), expected_tables)


class ParserTests(regression_suite.QueryTestCase):
    def test_parse_simple(self) -> None:
        query = "SELECT * FROM R, S WHERE R.a = S.b AND R.c = 42"
        pb.parse_query(query)

    def test_select_star(self) -> None:
        query = "SELECT * FROM R"
        parsed = pb.parse_query(query)
        self.assertSetEqual(parsed.columns(), set(), "* should not be considered a column")

    def test_parse_subquery_without_predicates(self) -> None:
        query = "SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S)"
        parsed = pb.parse_query(query)
        self.assertTrue(len(parsed.subqueries()) == 1, "Should detect 1 subquery")

    def test_is_predicate(self) -> None:
        query = "SELECT * FROM R WHERE R.a IS NULL"
        parsed = pb.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format IS NULL correctly.")
        self.assertTrue(
            len(parsed.predicates().filters()) == 1,
            "Should detect 1 filter for IS NULL",
        )

        query = "SELECT * FROM R WHERE R.a IS NOT NULL"
        parsed = pb.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format IS NOT NULL correctly.")
        self.assertTrue(
            len(parsed.predicates().filters()) == 1,
            "Should detect 1 filter for IS NOT NULL",
        )

    def test_unary_udf_filter(self) -> None:
        query = "SELECT * FROM R WHERE my_udf(R.a)"
        parsed = pb.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format unary UDF filter correctly.")
        self.assertTrue(
            len(parsed.predicates().filters()) == 1,
            "Should detect 1 filter for unary UDF filter",
        )

    def test_implicit_from_clause(self) -> None:
        query = "SELECT * FROM R, S, T WHERE R.a = S.b AND S.b = T.c"
        parsed = pb.parse_query(query)
        self.assertTrue(parsed.has_simple_from(), "Query should be recognized as a simple query")

        query = "SELECT * FROM R r, S s, T t WHERE r.a = s.b AND s.b = t.c"
        parsed = pb.parse_query(query)
        self.assertTrue(parsed.has_simple_from(), "Query should be recognized as a simple query")

    def test_explicit_from_clause(self) -> None:
        query = "SELECT * FROM R JOIN S ON R.a = S.b WHERE R.c LIKE '%42%' AND R.c < S.b"
        parsed = pb.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format explicit FROM clause correctly.")
        self.assertTrue(
            len(parsed.predicates().filters()) == 1,
            "Should detect 1 filter in WHERE clause",
        )
        self.assertTrue(
            len(parsed.predicates().joins()) == 2,
            "Should detect 2 joins in WHERE clause",
        )
        self.assertFalse(parsed.has_simple_from(), "Query should be parsed as explicit query")

    def test_from_clause_subquery(self) -> None:
        query = "SELECT * FROM R, (SELECT * FROM S WHERE S.d < 42) AS s WHERE R.a = s.b"
        parsed = pb.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format FROM clause with subquery correctly.")
        self.assertTrue(
            len(parsed.predicates().filters()) == 1,
            "Should detect 1 filter in WHERE clause",
        )
        self.assertTrue(
            len(parsed.predicates().joins()) == 1,
            "Should detect 1 join in WHERE clause",
        )

    def test_single_cte(self) -> None:
        query = "WITH cte_r AS (SELECT * FROM R) SELECT * FROM cte_r JOIN R ON cte_r.id = R.id"
        parsed = pb.parse_query(query)

        self.assertQueriesEqual(query, parsed, "Did not parse/format CTE query correctly")

        self.assertIsNotNone(parsed.cte_clause)
        self.assertEqual(len(parsed.cte_clause.queries), 1, "Did not recognize CTE correctly")  # type: ignore

        self.assertEqual(len(parsed.tables()), 2, "Did not resolve tables correctly")

    def test_multiple_cte(self) -> None:
        query = "WITH cte_r AS (SELECT * FROM R), cte_s AS (SELECT MIN(S.c) FROM S WHERE S.c < 42) SELECT * FROM cte_r, cte_s"
        parsed = pb.parse_query(query)

        self.assertQueriesEqual(query, parsed, "Did not parse/format CTE query correctly")

        self.assertIsNotNone(parsed.cte_clause)
        self.assertTrue(len(parsed.cte_clause.queries) == 2, "Did not recognize CTEs correctly")  # type: ignore

    def test_set_ops(self) -> None:
        query = "SELECT * FROM R UNION SELECT * FROM S"
        parsed = pb.parse_query(query)

        self.assertTrue(parsed.is_set_query(), "Did not recognize UNION query as set operation")
        self.assertIsInstance(parsed, pb.qal.SetQuery)

        set_query = pb.parse_query("SELECT * FROM S")
        self.assertEqual(parsed.rhs, set_query, "Did not parse/set subquery correctly")  # type: ignore

    def test_values_list(self) -> None:
        query = "SELECT * FROM (VALUES (1, 2), (3, 4)) AS t (a, b)"
        values_table = TableReference.create_virtual("t")
        parsed = pb.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format VALUES list correctly")
        self.assertIn(values_table, parsed.tables(), "Did not recognize VALUES list as table")

    def test_table_function(self) -> None:
        query = "SELECT * FROM my_table_function(42) as foo"
        table_fn = TableReference.create_virtual("foo")
        parsed = pb.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format table function correctly")
        self.assertIn(table_fn, parsed.tables(), "Did not recognize table function as table")

    def test_fetch_next(self) -> None:
        # we cannot test FETCH PRIOR or FETCH LAST, because they are not supported by the internal PostgreSQL parser
        query = "SELECT * FROM R FETCH NEXT 10 ROWS ONLY"
        parsed = pb.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format FETCH NEXT correctly")


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_job")
class JobWorkloadTests(regression_suite.DatabaseTestCase):
    def setUp(self):
        self.db = pb.postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_job")

    def test_resultset_equivalence(self) -> None:
        skip = os.environ.get("SKIP_ONLINE", "true") == "true"
        if skip:
            self.skipTest("Skipping online workload tests due to SKIP_ONLINE=true")

        _ = pb.workloads.job()  #  dummy load to ensure that the workload is available
        workload_base_dir = pathlib.Path.home() / ".postbound" / "workloads" / "job"

        for query_file in workload_base_dir.glob("*.sql"):
            with self.subTest(query=query_file.stem):
                raw_query = query_file.read_text()
                parsed_query = pb.parse_query(raw_query)
                original_result = self.db.execute_query(parsed_query)
                parsed_result = self.db.execute_query(parsed_query)
                self.assertResultSetsEqual(original_result, parsed_result, ordered=parsed_query.is_ordered())


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_stats")
class StatsWorkloadTests(regression_suite.DatabaseTestCase):
    def setUp(self):
        self.db = pb.postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_stats")

    def test_resultset_equivalence(self) -> None:
        skip = os.environ.get("SKIP_ONLINE", "true") == "true"
        if skip:
            self.skipTest("Skipping online workload tests due to SKIP_ONLINE=true")

        _ = pb.workloads.stats()  #  dummy load to ensure that the workload is available
        workload_base_dir = pathlib.Path.home() / ".postbound" / "workloads" / "stats"

        for query_file in workload_base_dir.glob("*.sql"):
            with self.subTest(query=query_file.stem):
                raw_query = query_file.read_text()
                parsed_query = pb.parse_query(raw_query)
                original_result = self.db.execute_query(parsed_query)
                parsed_result = self.db.execute_query(parsed_query)
                self.assertResultSetsEqual(original_result, parsed_result, ordered=parsed_query.is_ordered())
