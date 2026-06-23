"""Utilities to work with Pandas data frames"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .jsonize import to_json


def read_df(path: Path | str, **kwargs) -> pd.DataFrame:
    """Reads a Pandas `DataFrame` from a file, inferring the file format from the file extension.

    All additional arguments are passed directly to the respective Pandas read function.

    For SQLite databases, you must specify the table name using the 'table' or
    'relation' argument, e.g. ``read_df("data.db", table="my_table")``.
    """
    path = Path(path)
    match path.suffix.lower():
        case ".csv":
            return pd.read_csv(path, **kwargs)
        case ".parquet":
            return pd.read_parquet(path, **kwargs)
        case ".xlsx" | ".xls":
            return pd.read_excel(path, **kwargs)
        case ".json":
            return pd.read_json(path, **kwargs)
        case ".hdf" | ".h5" | ".hdf5":
            return pd.read_hdf(path, **kwargs)
        case ".feather":
            return pd.read_feather(path, **kwargs)
        case ".orc":
            return pd.read_orc(path, **kwargs)
        case ".db" | ".sqlite" | ".sqlite3":
            import sqlite3

            if "table" not in kwargs and "relation" not in kwargs:
                raise ValueError(
                    "For SQLite databases, you must specify the table name using the 'table' or 'relation' argument."
                )
            tab = kwargs.pop("table", None) or kwargs.pop("relation", None)
            with sqlite3.connect(path) as conn:
                return pd.read_sql_table(tab, conn, **kwargs)
        case _:
            raise ValueError(f"Unsupported file format: {path.suffix}")


def write_df(
    df: pd.DataFrame,
    path: Path | str,
    *,
    index: bool = False,
    append: bool = False,
    **kwargs,
) -> None:
    """Writes a Pandas `DataFrame` to a file, inferring the file format from the file extension.

    In addition to the automatic dispatch based on the file type, this function performs the following preprocessing
    steps:

    1. It ensures that the parent directory of the target file exists, creating it if necessary.
    2. It transforms all complex objects in the data frame into their JSON representation

    All additional arguments are passed directly to the respective Pandas write function.

    Please note that append-export is currently only supported for CSV files. We are working on adding support for other
    file formats in the future.
    """
    path = Path(path)
    if append and path.suffix != ".csv":
        raise ValueError("Appending is currently only supported for CSV files.")

    path.parent.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        if df[col].dtype != "object":
            continue
        if all(t is str for t in df[col].map(type)):
            continue
        df[col] = df[col].map(to_json)

    match path.suffix.lower():
        case ".csv":
            if append and path.exists():
                df.to_csv(path, index=index, mode="a", **kwargs)
            df.to_csv(path, index=index, **kwargs)
        case ".parquet":
            df.to_parquet(path, **kwargs)
        case ".xlsx" | ".xls":
            df.to_excel(path, index=index, **kwargs)
        case ".json":
            df.to_json(path, index=index, **kwargs)
        case ".hdf" | ".h5" | ".hdf5":
            df.to_hdf(path, **kwargs)
        case ".feather":
            df.to_feather(path, **kwargs)
        case ".orc":
            df.to_orc(path, **kwargs)
        case ".db" | ".sqlite" | ".sqlite3":
            import sqlite3

            with sqlite3.connect(path) as conn:
                df.to_sql("data", conn, index=index, **kwargs)
        case _:
            raise ValueError(f"Unsupported file format: {path.suffix}")
