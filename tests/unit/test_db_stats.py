"""Tests for `postbound.db._stats` -- `PreciseStatistics`, which computes every statistic with a live SQL query.

`PreciseStatistics` only talks to its database through `execute_query`, so a `FakeDatabase` answering from scripted
result sets is a complete seam: the tests run the real query generation and the real result-set simplification.

Whether the generated SQL is actually valid on a given system is not covered here; that needs a live database.
Currently this module only holds the tests around NULL handling in `num_distinct`.
"""

from __future__ import annotations

from postbound._core import ColumnReference, TableReference
from postbound.db import PreciseStatistics
from tests.doubles import FakeDatabase

R = TableReference("r")
R_A = ColumnReference("a", R)


# -- regression tests --------------------------------------------------------------------------------------


def test_num_distinct_counts_null_as_a_distinct_value() -> None:
    """Regression guard for the NULL handling in `PreciseStatistics.num_distinct`: it only ran
    ``COUNT(DISTINCT col)``, which ignores NULLs, while the native Postgres catalog counts NULL as one additional
    distinct value. The two catalogs disagreed for every column containing NULLs.
    """
    db = FakeDatabase(results={"count(distinct": [(2,)], "is null": [(None, 7)]})

    n_distinct = PreciseStatistics(db).num_distinct(R_A)

    assert n_distinct == 3


def test_num_distinct_does_not_count_null_for_a_column_without_nulls() -> None:
    db = FakeDatabase(results={"count(distinct": [(2,)], "is null": []})

    n_distinct = PreciseStatistics(db).num_distinct(R_A)

    assert n_distinct == 2


def test_num_distinct_catches_the_null_of_a_single_column_table() -> None:
    db = FakeDatabase(results={"count(distinct": [(2,)], "is null": [(None,)]})

    n_distinct = PreciseStatistics(db).num_distinct(R_A)

    assert n_distinct == 3
