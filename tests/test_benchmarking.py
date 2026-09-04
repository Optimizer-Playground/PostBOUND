import random
import tempfile
import unittest
from collections.abc import Iterable

import pandas as pd

import postbound as pb
from tests import regression_suite

pg_connect_dir = "."


class DummyJoinOrdering(pb.JoinOrderOptimization):
    def optimize_join_order(self, query: pb.SqlQuery) -> pb.JoinTree:
        join_tree = pb.JoinTree()
        for table in query.tables():
            join_tree = join_tree.join_with(table)
        return join_tree


class DummyOperatorSelection(pb.PhysicalOperatorSelection):
    def select_physical_operators(
        self, query: pb.SqlQuery, join_order: pb.JoinTree | None
    ) -> pb.PhysicalOperatorAssignment:
        assignment = pb.PhysicalOperatorAssignment()
        for table in query.tables():
            assignment.add(pb.ScanOperator.SequentialScan, [table])
        return assignment


class DummyPlanParameterization(pb.ParameterGeneration):
    def generate_plan_parameters(
        self,
        query: pb.SqlQuery,
        join_order: pb.JoinTree | None,
        operator_assignment: pb.PhysicalOperatorAssignment | None,
    ) -> pb.PlanParameterization:
        params = pb.PlanParameterization()
        for table in query.tables():
            params.add_cardinality([table], pb.Cardinality(random.randint(42, 1000)))
        return params


class DummyCardinalityEstimator(pb.CardinalityEstimator):
    def calculate_estimate(
        self,
        query: pb.SqlQuery,
        intermediate: pb.TableReference | Iterable[pb.TableReference],
    ) -> pb.Cardinality:
        intermediate = pb.util.enlist(intermediate)
        cardinality = len(intermediate) * random.randint(42, 1000)
        return pb.Cardinality(cardinality)


class FailingCardinalityEstimator(pb.CardinalityEstimator):
    def calculate_estimate(
        self,
        query: pb.SqlQuery,
        intermediate: pb.TableReference | Iterable[pb.TableReference],
    ) -> pb.Cardinality:
        raise RuntimeError("Cardinality estimation failed")


class DataDrivenOptimizer(pb.IncrementalOptimizationStep):
    def __init__(self) -> None:
        super().__init__()
        self.received_data_training = False

    def optimize_query(self, query: pb.SqlQuery, current_plan: pb.QueryPlan) -> pb.QueryPlan:
        return current_plan

    def fit_database(self, database) -> pb.train.TrainingMetrics:
        self.received_data_training = True
        return {}

    def database_fit_completed(self) -> bool:
        return self.received_data_training


class WorkloadDrivenOptimizer(pb.IncrementalOptimizationStep):
    def __init__(self) -> None:
        super().__init__()
        self.received_workload_training = False

    def optimize_query(self, query: pb.SqlQuery, current_plan: pb.QueryPlan) -> pb.QueryPlan:
        return current_plan

    def fit_workload(self, queries: pb.Workload, database: pb.Database) -> pb.train.TrainingMetrics:
        self.received_workload_training = True
        return {}

    def workload_fit_completed(self) -> bool:
        return self.received_workload_training


class OfflineLearnedOptimizer(pb.IncrementalOptimizationStep):
    def __init__(self) -> None:
        super().__init__()
        self.received_offline_training = False

    def optimize_query(self, query: pb.SqlQuery, current_plan: pb.QueryPlan) -> pb.QueryPlan:
        return current_plan

    def fit_samples(self, samples: pb.train.TrainingData) -> pb.train.TrainingMetrics:
        self.received_offline_training = True
        return {}

    def sample_spec(self) -> pb.train.TrainingSpec:
        return pb.train.TrainingSpec("query")

    def sample_fit_completed(self) -> bool:
        return self.received_offline_training


class OnlineLearnedOptimizer(pb.IncrementalOptimizationStep):
    def __init__(self) -> None:
        super().__init__()
        self.received_online_training = False

    def optimize_query(self, query: pb.SqlQuery, current_plan: pb.QueryPlan) -> pb.QueryPlan:
        return current_plan

    def learn_from_feedback(
        self, query: pb.SqlQuery, result_set: pb.db.ResultSet, *, exec_time: pb.TimeMs
    ) -> pb.train.TrainingMetrics:
        self.received_online_training = True
        return {}


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_stats")
class StatsBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pg_instance = pb.postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_stats")
        self.stats = pb.workloads.stats()

    def test_native_execution(self) -> None:
        results = pb.bench.execute_workload(self.stats.first(3), on=self.pg_instance)
        self.assertEqual(len(results), 3)
        self.assertTrue((results["status"] == "ok").all())

    def test_cardinality_estimation(self) -> None:
        estimator = DummyCardinalityEstimator()
        results = pb.bench.execute_workload(self.stats.first(3), on=estimator)
        self.assertEqual(len(results), 3)
        self.assertTrue((results["status"] == "ok").all())

    def test_multistage_pipeline(self) -> None:
        pipeline = (
            pb.MultiStageOptimizationPipeline(self.pg_instance)
            .use(DummyJoinOrdering())
            .use(DummyOperatorSelection())
            .use(DummyPlanParameterization())
            .build()
        )
        results = pb.bench.execute_workload(self.stats.first(3), on=pipeline)
        self.assertEqual(len(results), 3)
        self.assertTrue((results["status"] == "ok").all())

    def test_timeouts(self) -> None:
        native_results = pb.bench.execute_workload(self.stats.first(3), on=self.pg_instance)

        min_runtime = native_results["exec_time"].min()
        timeout_results = pb.bench.execute_workload(
            self.stats.first(3),
            on=self.pg_instance,
            timeout=0.5 * min_runtime,
        )
        self.assertEqual(len(timeout_results), 3)
        self.assertTrue((timeout_results["status"] == "timeout").any())

    def test_explain(self) -> None:
        results = pb.bench.execute_workload(
            self.stats.first(3),
            on=self.pg_instance,
            query_preparation={"output": "explain_analyze"},
        )
        self.assertEqual(len(results), 3)
        self.assertTrue((results["status"] == "ok").all())

    def test_result_streaming(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as out_file:
            pb.bench.execute_workload(
                self.stats.first(3),
                on=self.pg_instance,
                progressive_output=out_file.name,
            )

    def test_optimization_error(self) -> None:
        estimator = FailingCardinalityEstimator()
        results = pb.bench.execute_workload(self.stats.first(3), on=estimator)
        self.assertEqual(len(results), 3)
        self.assertTrue((results["status"] == "optimization-error").all())

    def test_learned_estimators(self) -> None:
        dummy_training_data = pb.train.TrainingData(
            pd.DataFrame(columns=["query"]),  # type: ignore
            feature_map={"query": "query"},
        )

        data_driven_opt = DataDrivenOptimizer()
        workload_driven_opt = WorkloadDrivenOptimizer()
        offline_opt = OfflineLearnedOptimizer()
        online_opt = OnlineLearnedOptimizer()
        pipeline = (
            pb.IncrementalOptimizationPipeline(self.pg_instance)
            .use(data_driven_opt)
            .use(workload_driven_opt)
            .use(offline_opt)
            .use(online_opt)
        )

        pb.bench.execute_workload(self.stats.first(3), on=pipeline, training_data=dummy_training_data)

        self.assertTrue(
            data_driven_opt.database_fit_completed(),
            "Data-driven optimizer did not receive database",
        )
        self.assertTrue(
            workload_driven_opt.workload_fit_completed(),
            "Workload-driven optimizer did not receive workload",
        )
        self.assertTrue(
            offline_opt.sample_fit_completed(),
            "Offline learned optimizer did not receive training samples",
        )
        self.assertTrue(
            online_opt.received_online_training,
            "Online learned optimizer did not receive feedback from query execution",
        )
