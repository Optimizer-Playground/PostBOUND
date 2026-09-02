from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from .._core import Cardinality, ColumnReference, TableReference, UnboundColumnError, VirtualTableError
from ..qal import (
    BaseProjection,
    From,
    GroupBy,
    Limit,
    OrderBy,
    OrderByExpression,
    Select,
    Where,
    as_predicate,
    as_query,
)
from ..util import jsondict
from ._cache import ResultCache
from ._db import Database, DatabaseStatistics, Histogram, HistogramApproximation, MostCommonValues


def _infer_histogram_bounds[T](
    frequencies: Sequence[tuple[T, int]], *, n_bins: int, n_rows: int
) -> tuple[T, Sequence[T], Sequence[int]]:
    """Infer the bucket bounds and frequencies for a histogram from a list of (value, frequency) pairs."""
    if not frequencies:
        raise ValueError("Cannot infer histogram bounds from empty frequency list")

    bucket_size = n_rows // n_bins

    bounds: list[T] = []
    buckets: list[int] = []
    cumulative_freq = 0
    for value, freq in frequencies:
        cumulative_freq += freq
        if cumulative_freq < bucket_size:
            continue
        bounds.append(value)
        buckets.append(cumulative_freq)
        cumulative_freq = 0

    return frequencies[0][0], bounds, buckets


class PreciseStatistics(DatabaseStatistics):
    @staticmethod
    def create_cached(db: Database, *, offline_cache: Optional[Path] = None) -> PreciseStatistics:
        cached = ResultCache.create_cache(db, offline_cache=offline_cache)
        return PreciseStatistics(cached)

    def __init__(self, db: Database) -> None:
        self._db = db

    def total_rows(self, table: TableReference) -> Cardinality:
        if table.virtual:
            raise VirtualTableError(table)

        select_clause = Select.count_star()
        from_clause = From.create_for(table)
        sql = as_query(select_clause, from_clause)

        n_rows = self._db.execute_query(sql)
        return Cardinality.of(n_rows)

    def num_distinct(self, column: ColumnReference) -> int:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)

        select_clause = Select(BaseProjection.create_count(column, distinct=True))
        from_clause = From.create_for(column.table)
        sql = as_query(select_clause, from_clause)

        return self._db.execute_query(sql)

    def null_frac(self, column: ColumnReference) -> float:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)

        n_rows = self.total_rows(column.table)
        if n_rows.is_zero():
            return 0

        select_clause = Select.count_star()
        from_clause = From.create_for(column.table)
        where_clause = Where(as_predicate(column, "IS NULL"))
        sql = as_query(select_clause, from_clause, where_clause)

        n_nulls: int = self._db.execute_query(sql)

        return n_nulls / int(n_rows)

    def min_max(self, column: ColumnReference) -> tuple[object, object]:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)

        select_clause = Select([BaseProjection.create_min(column), BaseProjection.create_max(column)])
        from_clause = From.create_for(column.table)
        sql = as_query(select_clause, from_clause)

        return self._db.execute_query(sql)

    def most_common_values(self, column: ColumnReference, *, k: int | None = 100) -> MostCommonValues:
        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)

        select_clause = Select([BaseProjection.column(column), BaseProjection.create_count(column, target_name="n")])
        from_clause = From.create_for(column.table)
        group_clause = GroupBy.create_for(column)
        order_clause = OrderBy(
            [OrderByExpression.create_for(ColumnReference("n"), ascending=False), OrderByExpression.create_for(column)]
        )
        limit_clause = Limit(limit=k) if k is not None else None
        sql = as_query(select_clause, from_clause, group_clause, order_clause, limit_clause)

        result_set = self._db.execute_query(sql)
        return MostCommonValues(result_set)

    def histogram(
        self, column: ColumnReference, *, n_bins: int | None = 100, interpolation: HistogramApproximation = "approx-uni"
    ) -> Histogram:
        if n_bins is None:
            raise ValueError("n_bins must be set for emulated histogram")

        if not ColumnReference.assert_bound(column):
            raise UnboundColumnError(column)
        if column.table.virtual:
            raise VirtualTableError(column.table)

        select_clause = Select([BaseProjection.column(column), BaseProjection.create_count(column, target_name="n")])
        from_clause = From.create_for(column.table)
        group_clause = GroupBy.create_for(column)
        order_clause = OrderBy.create_for(column)
        sql = as_query(select_clause, from_clause, group_clause, order_clause)

        result_set = self._db.execute_query(sql)
        n_rows = self.total_rows(column.table)
        lo, bounds, buckets = _infer_histogram_bounds(result_set, n_bins=n_bins, n_rows=int(n_rows))

        return Histogram(
            bounds,
            buckets,
            lower=lo,
            bucket_interpolation=interpolation,
        )

    def describe(self) -> jsondict:
        return {"kind": "precise-emulated"}
