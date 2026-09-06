"""Provides `PreciseStatistics`, a `StatisticsCatalog` that computes all statistics by issuing live SQL queries.

This is the successor of the emulation mode that used to be built directly into the statistics interface: rather than
being a special mode of every `StatisticsCatalog` implementation, emulation is now a standalone, database-independent
implementation of that interface. It can be used whenever a database system's native statistics catalog does not
maintain a specific statistic (see `enable_emulation_fallback` in `postbound.db`), or to force exact, up-to-date
statistics for reproducible experiments regardless of what the native catalog provides.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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
from ._db import Database, Histogram, HistogramApproximation, MostCommonValues, StatisticsCatalog


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


class PreciseStatistics(StatisticsCatalog):
    """A statistics catalog that computes all statistics on live data instead of reading a native catalog.

    Each statistic is derived by issuing an equivalent SQL query against the database. For example, the number of
    distinct values of a column is obtained by running a *SELECT COUNT(DISTINCT column) FROM table* query. Since these
    queries operate on the actual data, the resulting statistics are always exact and up-to-date -- in contrast to a
    native statistics catalog, which is typically based on samples and can be stale.

    That exactness is a double-edged sword: it makes experiments reproducible, but it also means that an optimizer
    running on `PreciseStatistics` might perform better than the same optimizer running on the native catalog of the
    same system, simply because it receives better input. Keep this in mind when comparing across database systems.

    Because computing statistics this way can be expensive, consider using `create_cached` (or wrapping `db` in a
    `ResultCache` manually) so that repeated requests for the same statistic do not re-execute the underlying query.

    Parameters
    ----------
    db : Database
        The database on which the statistics queries should be executed.

    See Also
    --------
    PreciseStatistics.create_cached : Constructs a `PreciseStatistics` on top of a cached database.
    postbound.db.enable_emulation_fallback : Controls whether native catalogs fall back to this implementation.
    """

    @staticmethod
    def create_cached(db: Database, *, offline_cache: Path | None = None) -> PreciseStatistics:
        """Constructs a `PreciseStatistics` instance that caches the results of its statistics queries.

        This is a shorthand for wrapping `db` in a `ResultCache` and building the statistics catalog on top of that
        cache. Since the underlying queries are expensive and PostBOUND assumes the database to be immutable during a
        run, this is usually the preferred way to obtain precise statistics.

        Parameters
        ----------
        db : Database
            The database on which the statistics queries should be executed.
        offline_cache : Path, optional
            A JSON file to persist the cached results to. See `ResultCache` for details.

        Returns
        -------
        PreciseStatistics
            The statistics catalog, backed by a (potentially shared) `ResultCache`.
        """
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
        """Provides the `k` most frequent values of a column, computed on live data.

        In addition to `StatisticsCatalog.most_common_values`, this implementation allows to customize how many values
        should be retrieved via `k`. Pass *None* or a non-positive value to retrieve the frequencies of all values.
        """
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
        limit_clause = Limit(limit=k) if k is not None and k > 0 else None
        sql = as_query(select_clause, from_clause, group_clause, order_clause, limit_clause)

        result_set = self._db.execute_query(sql)
        return MostCommonValues(result_set)

    def histogram(
        self, column: ColumnReference, *, n_bins: int | None = 100, interpolation: HistogramApproximation = "approx-uni"
    ) -> Histogram:
        """Provides an equi-depth histogram of the column's value distribution, computed on live data.

        In addition to `StatisticsCatalog.histogram`, this implementation allows to customize the number of buckets via
        `n_bins`. Since the histogram has to be constructed from scratch, `n_bins` is required and passing *None*
        raises a `ValueError`.
        """
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
        return {"kind": "precise"}
