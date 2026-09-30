import argparse
import time
from collections.abc import Iterable
from pathlib import Path

import postbound as pb


class DataDrivenCardinalityEstimator(pb.CardinalityEstimator):
    def __init__(self) -> None:
        super().__init__()
        self._rel_sizes: dict[pb.TableReference, pb.Cardinality] = {}

    def calculate_estimate(
        self, query: pb.SqlQuery, intermediate: pb.TableReference | Iterable[pb.TableReference]
    ) -> pb.Cardinality:
        intermediate = [intermediate] if isinstance(intermediate, pb.TableReference) else list(intermediate)
        if len(intermediate) != 1:
            return pb.Cardinality.unknown()

        base_table = intermediate[0]
        return self._rel_sizes.get(base_table.drop_alias(), pb.Cardinality.unknown())

    def generate_plan_parameters(
        self,
        query: pb.SqlQuery,
        join_order: pb.JoinTree | None,
        operator_assignment: pb.PhysicalOperatorAssignment | None,
    ) -> pb.PlanParameterization:
        params = pb.PlanParameterization()

        for tab in query.tables():
            estimate = self._rel_sizes.get(tab.drop_alias())
            if estimate is None:
                continue
            params.add_cardinality([tab], estimate)

        return params

    def fit_database(self, database: pb.Database) -> pb.train.TrainingMetrics:
        train_start = time.perf_counter_ns()

        for table in database.schema():
            base_card = database.statistics().total_rows(table)
            base_card = base_card or pb.Cardinality.unknown()
            self._rel_sizes[table] = base_card

        train_end = time.perf_counter_ns()
        train_duration_ms = (train_end - train_start) / 1e6
        return {"training_duration_ms": train_duration_ms}

    def database_fit_completed(self) -> bool:
        return len(self._rel_sizes) > 0


class QueryDrivenCardinalityEstimator(pb.CardinalityEstimator):
    def __init__(self) -> None:
        super().__init__()
        self._samples: dict[pb.SqlQuery, pb.Cardinality] = {}

    def calculate_estimate(
        self, query: pb.SqlQuery, intermediate: pb.TableReference | Iterable[pb.TableReference]
    ) -> pb.Cardinality:
        subquery = pb.transform.extract_subquery(query, intermediate)

        closest_dist = float("inf")
        closest_estimate = pb.Cardinality.unknown()

        query_hash = hash(subquery)
        for sample, cardinality in self._samples.items():
            sample_hash = hash(sample)
            current_dist = abs(query_hash - sample_hash)
            if current_dist > closest_dist:
                continue

            closest_dist = current_dist
            closest_estimate = cardinality

        return closest_estimate

    def fit_workload(self, queries: pb.Workload, database: pb.Database) -> pb.train.TrainingMetrics:
        train_start = time.perf_counter_ns()

        samples = queries.pick_random(30)
        self._samples = {query: database.optimizer().cardinality_estimate(query) for query in samples}

        train_end = time.perf_counter_ns()
        train_duration_ms = (train_end - train_start) / 1e6
        return {"training_duration_ms": train_duration_ms, "n_samples": len(self._samples)}

    def workload_fit_completed(self) -> bool:
        return len(self._samples) > 0


class ReinforcementOperatorSelection(pb.OperatorSelection):
    def __init__(self, database: pb.Database) -> None:
        super().__init__()
        self._database = database
        self._operator_runtimes: dict[pb.PhysicalOperator, float] = {}

    def select_physical_operators(
        self, query: pb.SqlQuery, join_order: pb.JoinTree | None
    ) -> pb.PhysicalOperatorAssignment:
        if not self._operator_runtimes:
            return pb.PhysicalOperatorAssignment()

        best_operator = pb.util.argmin(self._operator_runtimes)

        assignment = pb.PhysicalOperatorAssignment()
        assignment.add(best_operator, query.tables())

        return assignment

    def learn_from_feedback(
        self, query: pb.SqlQuery, result_set: pb.db.ResultSet, *, exec_time: pb.TimeMs
    ) -> pb.train.TrainingMetrics:
        train_start = time.perf_counter_ns()

        plan = self._database.optimizer().parse_plan(result_set)
        if not plan:
            return {}

        final_join = plan.find_first_node(lambda node: node.is_join())
        if final_join is None or final_join.operator is None:
            return {}

        current_runtime = self._operator_runtimes.get(final_join.operator, 0.0)
        self._operator_runtimes[final_join.operator] = exec_time + 0.5 * current_runtime

        train_end = time.perf_counter_ns()
        train_duration_ms = (train_end - train_start) / 1e6
        return {"training_duration_ms": train_duration_ms}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Basic demo for a custom join ordering algorithm. Requires Postgres with IMDB to be set up."
    )
    parser.add_argument("--pg-config", "-c", type=Path, help="Postgres connect file")
    parser.add_argument("--out-dir", "-o", type=Path, help="Output directory for benchmark results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    logger = pb.util.standard_logger(args.verbose)

    out_dir: Path = args.out_dir
    if out_dir.exists() and not out_dir.is_dir():
        parser.error(f"Output path {out_dir} is not a directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger("Connecting to database")
    pg_instance = pb.postgres.connect(args.pg_config)
    query_prep = pb.bench.QueryPreparation(
        projection="count_star", output="explain_analyze", prewarm=True, preparatory_statements=["SET geqo TO off;"]
    )

    logger("Loading JOB queries")
    job = pb.workloads.job()

    logger("Benchmarking data-driven optimizer")
    data_driven_opt = (
        pb.MultiStageOptimizationPipeline(pg_instance)
        .use(DataDrivenCardinalityEstimator())
        .build()
    )  # fmt: skip

    data_driven_results = pb.bench.execute_workload(
        job,
        on=data_driven_opt,
        query_preparation=query_prep,
        timeout=60,
        logger="tqdm",
        progressive_output=out_dir / "results-pg-job-data-driven.csv",
    )
    logger("Run completed. Total query runtime:", data_driven_results["exec_time"].sum())

    logger("Benchmarking query-driven optimizer")
    query_driven_opt = (
        pb.MultiStageOptimizationPipeline(pg_instance)
        .use(QueryDrivenCardinalityEstimator())
        .build()
    )  # fmt: skip

    query_driven_results = pb.bench.execute_workload(
        job,
        on=query_driven_opt,
        query_preparation=query_prep,
        timeout=60,
        logger="tqdm",
        progressive_output=out_dir / "results-pg-job-query-driven.csv",
    )
    logger("Run completed. Total query runtime:", query_driven_results["exec_time"].sum())

    logger("Benchmarking reinforcement learning optimizer")
    reinforcement_opt = (
        pb.MultiStageOptimizationPipeline(pg_instance)
        .use(ReinforcementOperatorSelection(pg_instance))
        .build()
    )  # fmt: skip

    reinforcement_results = pb.bench.execute_workload(
        job,
        on=reinforcement_opt,
        query_preparation=query_prep,
        timeout=60,
        logger="tqdm",
        progressive_output=out_dir / "results-pg-job-reinforcement.csv",
    )
    logger("Run completed. Total query runtime:", reinforcement_results["exec_time"].sum())


if __name__ == "__main__":
    main()
