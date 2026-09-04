#!/usr/bin/env python3

import argparse
import pathlib
import sys

import natsort
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Utility to create CSV workload files from multiple source files")
    parser.add_argument(
        "directory",
        action="store",
        default=".",
        nargs="?",
        help="Directory containing the source files (working directory per default).",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        action="store",
        default="*.sql",
        help="Filename pattern to match source files. '*.sql' per default.",
    )
    parser.add_argument(
        "--generate-labels",
        action="store_true",
        default=False,
        help="Use the filenames as query labels. Adds a 'label' column.",
    )
    parser.add_argument(
        "--query-col", action="store", default="query", help="CSV column to write the queries to ('query' per default)."
    )
    parser.add_argument(
        "--out", "-o", action="store", default=sys.stdout, help="File to write results to (stdout by default)."
    )

    args = parser.parse_args()

    root = pathlib.Path(args.directory)
    query_files = list(root.glob(args.pattern))

    queries = []
    labels = []

    for query_file in query_files:
        labels.append(query_file.stem)
        with open(query_file, encoding="utf-8") as raw_query:
            query_text = "".join(raw_query.readlines())
            queries.append(query_text)

    csv_df = pd.DataFrame({args.query_col: queries})
    if args.generate_labels:
        csv_df["label"] = labels
        csv_df.sort_values(by="label", key=lambda _: np.argsort(natsort.index_natsorted(csv_df["label"])), inplace=True)
        csv_df = csv_df[["label", args.query_col]].copy()

    csv_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
