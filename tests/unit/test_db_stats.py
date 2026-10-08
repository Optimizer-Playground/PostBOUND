"""Tests for `postbound.db._stats` -- `PreciseStatistics`, which computes every statistic with a live SQL query.

`PreciseStatistics` only talks to its database through `execute_query`, so a `FakeDatabase` answering from scripted
result sets is a complete seam: the tests run the real query generation and the real result-set simplification.

Whether the generated SQL is actually valid on a given system is not covered here; that needs a real engine. For
`histogram` this is checked against an in-memory DuckDB in `tests/test_duckdb.py`, which is also where the shape of
the scripted result sets below comes from (one ``(value, COUNT(col))`` row per distinct non-NULL value, ascending).
The estimates of the resulting `Histogram` are covered in `test_db_histogram.py`.

Currently this module holds the tests around NULL handling in `num_distinct` and the histogram construction.
"""

from __future__ import annotations

import pytest

from postbound import UnboundColumnError, VirtualTableError
from postbound._core import ColumnReference, TableReference
from postbound.db import Histogram, PreciseStatistics
from tests.doubles import FakeDatabase

R = TableReference("r")
R_A = ColumnReference("a", R)


def histogram_db(value_counts: list[tuple[object, int]]) -> FakeDatabase:
    """A strict database answering only the histogram's GROUP BY query, with `value_counts`."""
    return FakeDatabase(results={"group by": value_counts})


# -- histogram ----------------------------------------------------------------------------------------------


def test_histogram_groups_the_non_null_column_values_in_ascending_order() -> None:
    db = histogram_db([(1, 5), (2, 5), (3, 5), (4, 5)])

    PreciseStatistics(db).histogram(R_A, n_bins=2)

    assert len(db.executed_queries) == 1
    value_query = db.executed_queries[0]
    assert "count(r.a)" in value_query
    assert "where r.a is not null" in value_query
    assert "group by r.a" in value_query
    assert "order by r.a" in value_query


def test_histogram_builds_equi_depth_buckets() -> None:
    db = histogram_db([(1, 5), (2, 5), (3, 5), (4, 5)])

    hist = PreciseStatistics(db).histogram(R_A, n_bins=2)

    assert hist == Histogram([2, 4], [10, 10], lower=1, bucket_interpolation="approx-uni")


def test_histogram_uses_uniform_approximation_by_default() -> None:
    db = histogram_db([(1, 5), (2, 5)])

    assert PreciseStatistics(db).histogram(R_A, n_bins=2).bucket_interpolation == "approx-uni"


def test_histogram_passes_the_interpolation_strategy_on() -> None:
    db = histogram_db([(1, 5), (2, 5)])

    assert (
        PreciseStatistics(db).histogram(R_A, n_bins=2, interpolation="bound-lower").bucket_interpolation
        == "bound-lower"
    )


def test_histogram_puts_every_value_into_its_own_bucket_if_there_are_fewer_rows_than_bins() -> None:
    db = histogram_db([(1, 2), (2, 1), (3, 1)])

    hist = PreciseStatistics(db).histogram(R_A, n_bins=100)

    assert list(hist) == [(1, 2), (2, 1), (3, 1)]
    assert hist.lower == 1


def test_histogram_lets_a_frequent_value_overflow_its_bucket() -> None:
    """A value is never split across buckets, so the buckets are only approximately equi-depth: value 2 alone fills
    more than two buckets' worth of rows, and only 2 of the 3 requested buckets are created.
    """
    db = histogram_db([(1, 1), (2, 8), (3, 1), (4, 3)])

    hist = PreciseStatistics(db).histogram(R_A, n_bins=3)

    assert list(hist) == [(2, 9), (4, 4)]


@pytest.mark.parametrize("n_bins", [None, -1])
def test_histogram_requires_a_positive_bin_count(n_bins: int | None) -> None:
    db = FakeDatabase()

    with pytest.raises(ValueError, match="n_bins must be a positive number"):
        PreciseStatistics(db).histogram(R_A, n_bins=n_bins)

    assert db.executed_queries == []


def test_histogram_rejects_unbound_columns() -> None:
    with pytest.raises(UnboundColumnError):
        PreciseStatistics(FakeDatabase()).histogram(ColumnReference("a"))


def test_histogram_rejects_columns_of_virtual_tables() -> None:
    column = ColumnReference("a", TableReference.create_virtual("sq"))

    with pytest.raises(VirtualTableError):
        PreciseStatistics(FakeDatabase()).histogram(column)


def test_histogram_of_a_column_without_values_raises() -> None:
    """An empty table or an all-NULL column has no values to build buckets from."""
    db = histogram_db([])

    with pytest.raises(ValueError, match="empty frequency list"):
        PreciseStatistics(db).histogram(R_A)


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


def test_histogram_closes_a_final_bucket_for_the_remaining_values() -> None:
    """Regression guard for the unreleased histogram fixes: `_infer_histogram_bounds` only emitted a bucket once its
    cumulative frequency reached ``n_rows // n_bins`` and discarded the rows still accumulated when the loop ended.
    Here values 3 and 4 (7 of 17 rows) vanished: the histogram ended at 2 instead of the maximum 4, with 10 rows.
    """
    db = histogram_db([(1, 5), (2, 5), (3, 5), (4, 2)])

    hist = PreciseStatistics(db).histogram(R_A, n_bins=2)

    assert list(hist) == [(2, 10), (4, 7)]
    assert hist.upper == 4
    assert hist.n_rows == 17


def test_histogram_of_a_column_with_a_single_distinct_value() -> None:
    """Regression guard for the unreleased histogram fixes: `PreciseStatistics.histogram` called ``execute_query``
    without ``raw=True``, so the one-row result ``[(7, 2)]`` was simplified to the tuple ``(7, 2)``, and
    `_infer_histogram_bounds` crashed trying to unpack the integer 7 as a ``(value, freq)`` pair.
    """
    db = histogram_db([(7, 2)])

    hist = PreciseStatistics(db).histogram(R_A)

    assert hist == Histogram([7], [2], lower=7, bucket_interpolation="approx-uni")


def test_histogram_bucket_size_ignores_null_rows() -> None:
    """Regression guard for the unreleased histogram fixes: the buckets were sized with `total_rows`, which counts NULL
    rows that are not part of the histogram. 4 non-NULL values in a table with 4 more NULL rows got bucket size
    ``8 // 4 = 2`` and only 2 of the 4 requested buckets. The NULL group itself (``(None, 0)``, sorted last) could even
    become the upper bound; that part is guarded on DuckDB in `tests/test_duckdb.py`, since the WHERE clause now keeps
    it out of the scripted result.
    """
    db = histogram_db([(1, 1), (2, 1), (3, 1), (4, 1)])

    hist = PreciseStatistics(db).histogram(R_A, n_bins=4)

    assert list(hist) == [(1, 1), (2, 1), (3, 1), (4, 1)]


def test_histogram_rejects_zero_bins() -> None:
    """Regression guard for the unreleased histogram fixes: only ``n_bins=None`` was validated, so ``n_bins=0``
    reached ``n_rows // n_bins`` in `_infer_histogram_bounds` and failed with a `ZeroDivisionError`.
    """
    db = FakeDatabase()

    with pytest.raises(ValueError, match="n_bins must be a positive number"):
        PreciseStatistics(db).histogram(R_A, n_bins=0)
