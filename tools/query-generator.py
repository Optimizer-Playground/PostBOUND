from __future__ import annotations

import argparse
import pathlib
import re
import sys
import textwrap

import pandas as pd

import postbound as pb
from postbound.experiments import querygen


def _resolve_ignore_tables(
    table_specifiers: list[str], *, pg_instance: pb.postgres.PostgresInterface
) -> set[pb.TableReference]:
    if not table_specifiers:
        return set()

    candidate_tables = pg_instance.schema().tables()

    ignored: set[pb.TableReference] = set()
    for spec in table_specifiers:
        if "*" not in spec:
            ignored.add(pb.TableReference(spec))
            continue

        table_pattern = re.compile(spec.replace("*", ".*"))
        matching_tables = [tab for tab in candidate_tables if table_pattern.match(tab.full_name)]
        ignored.update(matching_tables)

    return ignored


def _prewarm_tables(tables: set[pb.TableReference], *, pg_instance: pb.postgres.PostgresInterface) -> None:
    if not pg_instance.has_extension("pg_prewarm"):
        print("pg_prewarm seems to be unavailable. Skipping prewarming.", file=sys.stderr)
        return

    # We start the prewarming with the smallest tables because they are cheap to re-load if the shared buffer is not large
    # enough to fit all tables. Ideally, this enables us to keep the large tables mostly in the buffer.
    tables_by_size = sorted(tables, key=lambda tab: pg_instance.statistics().total_rows(tab))
    for tab in tables_by_size:
        pg_instance.prewarm_tables(tab)


def main() -> None:
    description = textwrap.dedent("""
                                  Generate a workload of random queries.

                                  The generator selects a random subset of (connected) tables from the schema graph of the
                                  target database and builds a random number of random filter predicates on the columns of the
                                  selected tables.

                                  The schema graph contains edges between tables that are connected via foreign key
                                  constraints. Hence, the generator is focused on star-queries.
                                  """)
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--db-connect",
        "-c",
        help="Postgres connection string. If omitted, the connection must be specified via --db-config.",
    )
    parser.add_argument(
        "--db-config",
        help="Path to a Postgres configuration file. If omitted, the connection must be specified via --db-connect.",
    )
    parser.add_argument(
        "--n-queries", "-n", type=int, default=100, help="The number of queries to generate. Defaults to 100."
    )
    parser.add_argument(
        "--min-tables",
        type=int,
        default=1,
        help="The minimum number of tables to include in a query. The default is 1.",
    )
    parser.add_argument(
        "--max-tables",
        type=int,
        default=None,
        help="The maximum number of tables to include in a query. If omitted, all available tables can be used.",
    )
    parser.add_argument(
        "--min-filters",
        type=int,
        default=0,
        help="Them minimum number of filter predicates to generate per query. The default is 0.",
    )
    parser.add_argument(
        "--max-filters",
        type=int,
        default=None,
        help="The maximum number of filter predicates to generate per query. "
        "If omitted, each column from the selected tables could receive a filter predicate.",
    )
    parser.add_argument(
        "--ignore-tables",
        nargs="+",
        default=[],
        help="Tables that should not be included in the queries. "
        "Tables can be identified by their full name, or wildcard patterns can be used. For example, "
        "'movie_*' will ignore all tables starting with 'movie_'. In addition, all Postgres-internal tables "
        "starting with 'pg_' are ignored automatically.",
    )
    parser.add_argument(
        "--projection",
        choices=["star", "countstar"],
        default="star",
        help="Whether the generated queries should feature SELECT * or SELECT COUNT(*) clauses.",
    )
    parser.add_argument(
        "--filter-keys",
        action="store_true",
        help="Whether primary key and foreign key columns should be included in the filter predicates.",
    )
    parser.add_argument(
        "--numeric-filters",
        action="store_true",
        help="Whether only numeric columns should be used as filter predicates.",
    )
    parser.add_argument(
        "--query-prefix", default="q-", help="How to name the generated queries. Queries are labelled <prefix><number>."
    )
    parser.add_argument(
        "--output-mode",
        choices=["plain", "csv"],
        default="plain",
        help="How to export the generated queries. In plain mode, each query is written to a separate file, "
        "named according to the query prefix. In CSV mode, all queries are written to a single CSV file. "
        "Each query goes to a separate row, with a label column according to the query prefix and a "
        "query column.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress information")
    parser.add_argument(
        "out_path",
        help="File path to write the generated queries to. For plain-mode, this is treated as a "
        "directory. For CSV-mode, this is treated as an actual file.",
    )

    args = parser.parse_args()
    if not args.db_connect and not args.db_config:
        parser.error("Either --db-connect or --db-config must be provided.")

    pg_instance = (
        pb.postgres.connect(connect_string=args.db_connect)
        if args.db_connect
        else pb.postgres.connect(config_file=args.db_config)
    )

    ignored_tables = _resolve_ignore_tables(args.ignore_tables, pg_instance=pg_instance)
    count_star = args.projection == "countstar"
    _prewarm_tables(pg_instance.schema().tables() - ignored_tables, pg_instance=pg_instance)

    query_hashes: set[int] = set()
    generated_queries: list[tuple[str, pb.SqlQuery]] = []
    qgen = querygen.generate_query(
        pg_instance,
        count_star=count_star,
        ignore_tables=ignored_tables,
        min_tables=args.min_tables,
        max_tables=args.max_tables,
        min_filters=args.min_filters,
        max_filters=args.max_filters,
        filter_key_columns=args.filter_keys,
        numeric_filters=args.numeric_filters,
    )
    for i in range(1, args.n_queries + 1):
        if args.verbose and i > 0 and i % 10 == 0:
            print(i, file=sys.stderr)

        label = f"{args.query_prefix}{i}"
        query = next(qgen)
        while hash(query) in query_hashes:
            query = next(qgen)

        query_hashes.add(hash(query))
        generated_queries.append((label, query))

    match args.output_mode:
        case "plain":
            out_dir = pathlib.Path(args.out_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            for label, query in generated_queries:
                out_path = pathlib.Path(args.out_path) / f"{label}.sql"
                out_path.write_text(pb.qal.format_quick(query), encoding="utf-8")

        case "csv":
            out_path = pathlib.Path(args.out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                [(label, pb.qal.format_quick(query)) for label, query in generated_queries], columns=["label", "query"]
            )
            df.to_csv(out_path, index=False)

        case _:
            parser.error(f"Unknown output mode: {args.output_mode}")


if __name__ == "__main__":
    main()
