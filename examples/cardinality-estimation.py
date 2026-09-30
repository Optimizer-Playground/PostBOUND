import argparse
import random
from collections.abc import Iterable
from pathlib import Path

from tqdm import tqdm

import postbound as pb


class RandomEstimator(pb.CardinalityEstimator):
    def __init__(self, database: pb.Database, *, deviation: float = 0.5) -> None:
        super().__init__()
        self._database = database
        self._optimizer = self._database.optimizer()
        self._dev = deviation

    def calculate_estimate(
        self, query: pb.SqlQuery, intermediate: pb.TableReference | Iterable[pb.TableReference]
    ) -> pb.Cardinality:
        subquery = pb.transform.extract_subquery(query, intermediate)
        native_estimate = float(self._optimizer.cardinality_estimate(subquery))

        lo, hi = self._dev * native_estimate, (1 + self._dev) * native_estimate

        final_estimate = random.uniform(lo, hi)
        return pb.Cardinality.of(final_estimate)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Basic demo for a custom cardinality estimator. Requires Postgres with IMDB to be set up."
    )
    parser.add_argument("--pg-config", "-c", type=Path, help="Postgres connect file")
    parser.add_argument("--deviation", "-d", type=float, default=0.5, help="Allowed underestimation/overestimation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    logger = pb.util.standard_logger(args.verbose)

    logger("Connecting to database")
    pg_instance = pb.postgres.connect(args.pg_config)

    logger("Loading JOB queries")
    job = pb.workloads.job()

    logger("Creating optimizer")
    estimator = RandomEstimator(pg_instance, deviation=args.deviation)
    optimizer = (
        pb.MultiStageOptimizationPipeline(pg_instance)
        .use(estimator)
        .build()
    )  # fmt: skip

    logger(""""Optimizing" JOB queries""")
    for label, query in tqdm(job.entries()):
        hinted_query = optimizer.optimize_query(query)
        print(label)
        print(pb.qal.format_quick(hinted_query))


if __name__ == "__main__":
    main()
