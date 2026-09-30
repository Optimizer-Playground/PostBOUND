import argparse
from collections.abc import Iterable
from pathlib import Path

import postbound as pb


class VerySmartCardinalityEstimator(pb.CardinalityEstimator):
    def __init__(self, target_db: pb.Database | None) -> None:
        super().__init__()
        self._target_db = target_db or pb.db.current_database()

    def calculate_estimate(
        self, query: pb.SqlQuery, intermediate: pb.TableReference | Iterable[pb.TableReference]
    ) -> pb.Cardinality:
        subquery = pb.transform.extract_subquery(query, intermediate)
        return self._target_db.optimizer().cardinality_estimate(subquery)


class PrettyStupidCardinalityEstimator(pb.CardinalityEstimator):
    def __init__(self) -> None:
        super().__init__()

    def calculate_estimate(
        self, query: pb.SqlQuery, intermediate: pb.TableReference | Iterable[pb.TableReference]
    ) -> pb.Cardinality:
        intermediate = [intermediate] if isinstance(intermediate, pb.TableReference) else list(intermediate)
        return pb.Cardinality(42 * len(intermediate))


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic benchmarking demo. Requires Postgres with IMDB to be set up.")
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

    logger("Loading JOB workload")
    job = pb.workloads.job()
    query_prep = pb.bench.QueryPreparation(
        projection="count_star", output="explain_analyze", prewarm=True, preparatory_statements=["SET geqo TO off;"]
    )

    logger("Running native baseline")
    native_results = pb.bench.execute_workload(
        job,
        on=pg_instance,
        query_preparation=query_prep,
        workload_repetitions=3,
        timeout=60,
        shuffled=True,
        logger="tqdm",
        progressive_output=out_dir / "results-pg-job-native.csv",
    )
    logger("Run completed. Total query runtime:", native_results["exec_time"].sum())

    logger("Evaluating very smart estimator")
    smart_results = pb.bench.execute_workload(
        job,
        on=VerySmartCardinalityEstimator(pg_instance),
        query_preparation=query_prep,
        workload_repetitions=3,
        timeout=60,
        shuffled=True,
        logger="tqdm",
        progressive_output=out_dir / "results-pg-job-smart.csv",
    )
    logger("Run completed. Total query runtime:", smart_results["exec_time"].sum())

    logger("Evaluating pretty stupid estimator")
    stupid_results = pb.bench.execute_workload(
        job,
        on=PrettyStupidCardinalityEstimator(),
        query_preparation=query_prep,
        workload_repetitions=3,
        timeout=60,
        shuffled=True,
        logger="tqdm",
        progressive_output=out_dir / "results-pg-job-stupid.csv",
    )
    logger("Run completed. Total query runtime:", stupid_results["exec_time"].sum())


if __name__ == "__main__":
    main()
