import argparse
import os
import pathlib

from postbound import postgres
from postbound.experiments import ceb


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a CEB-based workload.")
    parser.add_argument(
        "--queries-per-template",
        "-n",
        type=int,
        required=True,
        help="Number of queries to generate per template.",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default=".",
        help="Directory to write the generated queries to.",
    )
    parser.add_argument(
        "--template-pattern",
        "-p",
        type=str,
        default="*.toml",
        help="File name GLOB that all templates must match.",
    )
    parser.add_argument(
        "--with-subdirs",
        action="store_true",
        help="Whether queries should be written into different subdirectories for each template.",
    )
    parser.add_argument(
        "--db-config",
        type=str,
        required=False,
        help="Path to the Postgres config file used to obtain a "
        "database connection. The database is used to determine valid candidate values for the different "
        "templates. The connect file has to be supported by psycopg. See postgres.connect() for more details.",
    )
    parser.add_argument("template_dir", type=str, help="Directory containing the templates to use.")

    args = parser.parse_args()
    if not os.path.isdir(args.template_dir):
        raise FileNotFoundError(f"Template directory '{args.template_dir}' does not exist.")
    pg_instance = postgres.connect(config_file=args.db_config) if args.db_config else postgres.connect()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if not args.with_subdirs:
        queries = ceb.generate_raw_workload(
            args.template_dir,
            queries_per_template=args.queries_per_template,
            template_pattern=args.template_pattern,
            db_connection=pg_instance,
        )
        ceb.persist_workload(args.out_dir, queries)
        return

    for template_file in pathlib.Path(args.template_dir).glob(args.template_pattern):
        local_glob = template_file.name
        queries = ceb.generate_workload(
            args.template_dir,
            queries_per_template=args.queries_per_template,
            template_pattern=local_glob,
            db_connection=pg_instance,
        )
        out_dir = pathlib.Path(args.out_dir) / template_file.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        ceb.persist_workload(out_dir, queries)


if __name__ == "__main__":
    main()
