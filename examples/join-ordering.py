import argparse
import random
from pathlib import Path

from tqdm import tqdm

import postbound as pb


class RandomJoinOrder(pb.JoinOrdering):
    def optimize_join_order(self, query: pb.SqlQuery) -> pb.JoinTree:
        if not pb.qal.is_select_query(query):
            raise pb.OptimizationError("Expected a SELECT query")

        initial_table, *free_tables = random.sample(list(query.tables()), k=len(query.tables()))
        join_tree = pb.JoinTree(base_table=initial_table)

        while free_tables:
            candidates = [table for table in free_tables if query.joins_between(table, join_tree.tables())]
            next_table = random.choice(candidates)
            join_tree = join_tree.join_with(next_table)
            free_tables.remove(next_table)

        return join_tree

    def pre_check(self) -> pb.validation.OptimizationPreCheck:
        return pb.validation.SPJCheck()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Basic demo for a custom join ordering algorithm. Requires Postgres with IMDB to be set up."
    )
    parser.add_argument("--pg-config", "-c", type=Path, help="Postgres connect file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    logger = pb.util.standard_logger(args.verbose)

    logger("Connecting to database")
    pg_instance = pb.postgres.connect(args.pg_config)

    logger("Loading JOB queries")
    job = pb.workloads.job()

    logger("Creating optimizer")
    estimator = RandomJoinOrder()
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
