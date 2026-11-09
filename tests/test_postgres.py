import unittest
from typing import override

import pytest

import postbound as pb
from postbound.postgres import PostgresConfiguration, PostgresSetting
from tests import regression_suite

pg_connect_dir = "."

# pg_hint_plan is more limited in scope than pg_lab. On a practical level, this means that we cannot force certain operators
# with pg_hint_plan reliably, e.g., memoize nodes. The PG optimizer is free to insert such operators as it sees fit. This
# creates problems for our unit tests because the optimizer might (or might not) use a memoize node for the native plan,
# but decide on the opposite for the hinted plan (because pg_hint_plan modifies the internal cost estimates).
# This applies to memoize nodes, materialize nodes, and parallel plans in general.
# In the end, we might end up with different plans and there is nothing we can do aboout it. Therefore, we take a more
# radical approach and disable all problematic PG features beforehand.
PGHintPlanRestrictions = PostgresConfiguration(
    [
        PostgresSetting("max_parallel_workers_per_gather", 0),
        PostgresSetting("enable_material", "off"),
        PostgresSetting("enable_memoize", "off"),
    ]
)


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_stats")
@pytest.mark.live_db
class QueryExecutionTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.pg_instance = pb.postgres.connect(
            config_file=f"{pg_connect_dir}/.psycopg_connection_stats",
            private=True,
        )
        self.stats = pb.workloads.stats()

    def test_execute_query(self) -> None:
        query = self.stats["q-1"]
        result = self.pg_instance.execute_query(query)
        self.assertEqual(result, 79851)

    def test_execute_raw_query(self) -> None:
        query = self.stats["q-1"]
        result_set = self.pg_instance.execute_query(query, raw=True)
        self.assertEqual(len(result_set), 1)

        result = result_set[0][0]
        self.assertEqual(result, 79851)

    def test_execute_timeout(self) -> None:
        query = self.stats["q-1"]
        self.pg_instance.execute_query(query)
        vanilla_runtime = self.pg_instance.last_query_runtime()

        timeout = 0.5 * vanilla_runtime
        result_set = self.pg_instance.execute_with_timeout(query, timeout=timeout)
        self.assertIs(result_set, None)

        result_set = self.pg_instance.execute_with_timeout(query, timeout=2 * vanilla_runtime)
        self.assertIsNot(result_set, None)
        assert result_set is not None

        self.assertEqual(len(result_set), 1)
        result = result_set[0][0]
        self.assertEqual(result, 79851)


@pytest.mark.live_db
class StatsSchemaTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.pg_instance = pb.postgres.connect(
            config_file=f"{pg_connect_dir}/.psycopg_connection_stats",
            private=True,
        )
        self.stats = pb.workloads.stats()

    # TODO


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_job")
@pytest.mark.live_db
class JobHintingTests(regression_suite.PlanTestCase):
    @override
    def setUp(self) -> None:
        self.pg_instance = pb.postgres.connect(
            config_file=f"{pg_connect_dir}/.psycopg_connection_job",
            private=True,
        )
        self.job = pb.workloads.job()

    @pytest.mark.slow
    def test_pglab_backend(self) -> None:
        if self.pg_instance.hinting().backend != "pg_lab":
            self.skipTest("pg_lab is not available")

        for label, query in self.job.entries():
            with self.subTest("Query", label=label):
                self.pg_instance.reset_connection()
                native_plan = self.pg_instance.optimizer().query_plan(query)
                hinted_query = self.pg_instance.hinting().generate_hints(query, native_plan)
                explicit_plan = self.pg_instance.optimizer().query_plan(hinted_query)
                self.assertEqual(native_plan, explicit_plan)

    @pytest.mark.slow
    def test_pg_hint_plan_backend(self) -> None:
        if self.pg_instance.hinting().backend != "pg_hint_plan":
            self.skipTest("pg_hint_plan is not available")

        for label, query in self.job.entries():
            with self.subTest("Query", label=label):
                self.pg_instance.reset_connection()
                self.pg_instance.apply_configuration(PGHintPlanRestrictions)
                native_plan = self.pg_instance.optimizer().query_plan(query)
                hinted_query = self.pg_instance.hinting().generate_hints(query, native_plan)
                explicit_plan = self.pg_instance.optimizer().query_plan(hinted_query)
                self.assertEqual(native_plan, explicit_plan)


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_stats")
@pytest.mark.live_db
class StatsHintingTests(regression_suite.PlanTestCase):
    @override
    def setUp(self) -> None:
        self.pg_instance = pb.postgres.connect(
            config_file=f"{pg_connect_dir}/.psycopg_connection_stats",
            private=True,
        )
        self.stats = pb.workloads.stats()

    @pytest.mark.slow
    def test_pglab_backend(self) -> None:
        if self.pg_instance.hinting().backend != "pg_lab":
            self.skipTest("pg_lab is not available")

        for label, query in self.stats.entries():
            with self.subTest("Query", label=label):
                self.pg_instance.reset_connection()
                native_plan = self.pg_instance.optimizer().query_plan(query)
                hinted_query = self.pg_instance.hinting().generate_hints(query, native_plan)
                explicit_plan = self.pg_instance.optimizer().query_plan(hinted_query)
                self.assertEqual(native_plan, explicit_plan)

    @pytest.mark.slow
    def test_pg_hint_plan_backend(self) -> None:
        if self.pg_instance.hinting().backend != "pg_hint_plan":
            self.skipTest("pg_hint_plan is not available")

        for label, query in self.stats.entries():
            with self.subTest("Query", label=label):
                self.pg_instance.reset_connection()
                self.pg_instance.apply_configuration(PGHintPlanRestrictions)
                native_plan = self.pg_instance.optimizer().query_plan(query)
                hinted_query = self.pg_instance.hinting().generate_hints(query, native_plan)
                explicit_plan = self.pg_instance.optimizer().query_plan(hinted_query)
                self.assertEqual(native_plan, explicit_plan)

    def test_pglab_parallel(self) -> None:
        if self.pg_instance.hinting().backend != "pg_lab":
            self.skipTest("pg_lab is not available")

        posts = pb.TableReference("posts", "p")
        users = pb.TableReference("users", "u")
        query = pb.parse_query("SELECT * FROM posts p JOIN users u ON u.id = p.owneruserid")
        plan = pb.QueryPlan(
            "Nested Loop",
            operator=pb.JoinOperator.NestedLoopJoin,
            children=[
                pb.QueryPlan(
                    "Gather",
                    parallel_workers=2,
                    children=pb.QueryPlan(
                        "Seq Scan",
                        operator=pb.ScanOperator.SequentialScan,
                        base_table=posts,
                    ),
                ),
                pb.QueryPlan(
                    "Index Scan",
                    operator=pb.ScanOperator.IndexScan,
                    base_table=users,
                ),
            ],
        )

        hinted_plan = self.pg_instance.explain(query, plan=plan)
        self.assertQueryExecutionPlansEqual(plan, hinted_plan)


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_stats")
@pytest.mark.live_db
class RegressionTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.pg_instance = pb.postgres.connect(
            config_file=f"{pg_connect_dir}/.psycopg_connection_stats",
            private=True,
        )
        self.stats = pb.workloads.stats()

    def test_execute_no_result_set(self) -> None:
        res = self.pg_instance.execute_query("SET geqo TO off")
        self.assertIsNone(res, "Queries that do not provide a result set should return None")


# -- regression tests --------------------------------------------------------------------------------------


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_stats")
@pytest.mark.live_db
@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("votes", "bountyamount"),
        ("posts", "answercount"),
        ("posts", "owneruserid"),
        ("comments", "userid"),
        ("posts", "closeddate"),
    ],
    ids=["mcvs-inside-buckets", "mcvs-below-histogram", "many-mcvs", "large-mcv-share", "timestamp"],
)
def test_postgres_histogram_merges_the_mcvs_into_a_consistent_histogram(table: str, column: str) -> None:
    """Regression guard for the histogram reconstruction of `PostgresStatistics`: the MCVs were merged in frequency
    rather than value order, which produced non-monotone bounds, and the MCV rows were counted twice (comments.userid
    had 254,600 histogram rows for 171,470 non-NULL rows). Timestamp columns could not be read at all.

    The histogram must be sorted and cover all non-NULL rows. Since the catalog is sampled, the row count is checked
    against the catalog's own estimate of the non-NULL rows rather than the true count.
    """
    pg_instance = pb.postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_stats", private=True)
    col = pb.ColumnReference(column, pb.TableReference(table))
    stats = pg_instance.statistics()

    hist = stats.histogram(col)

    assert hist is not None
    assert hist.bounds == sorted(set(hist.bounds))
    assert hist.lower <= hist.bounds[0]
    n_rows, null_frac = stats.total_rows(col.table), stats.null_frac(col)
    assert n_rows is not None and null_frac is not None
    assert hist.n_rows == pytest.approx((1 - null_frac) * int(n_rows), rel=0.01)
