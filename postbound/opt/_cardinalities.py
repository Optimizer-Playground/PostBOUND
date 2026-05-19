from __future__ import annotations

import json
import random
from collections.abc import Iterable
from typing import Literal, Optional

import pandas as pd

from .. import parser, transform, util
from .._core import Cardinality, TableReference
from .._stages import (
    CardinalityEstimator,
)
from ..db import Database, DatabasePool
from ..qal import SqlQuery
from ..workloads import Workload


class PreciseCardinalities(CardinalityEstimator):
    """Cardinality "estimator" that calculates exact cardinalities.

    These cardinalities are determined by actually executing the intermediate query plan and counting the number of result
    tuples. To speed up this potentially very costly computation, the estimator can store already calculated cardinalities in
    an intermediate cache. Notice that this cache is different from the query cache provided by the `Database` interface. The
    reason for this distinction is simple: the query result cache assumes static databases. If it connects to the same logical
    database at two different points in time (potentially after a data shift), the cached results will be out-of-date. On the
    other hand, the cardinality cache is transient and local to each estimator. Therefore, it will always calculate the current
    results, even when a data shift is simulated. Even when the same estimator is used while simulating a data shift, the cache
    can be reset manually without impacting caching of all other queries.

    Parameters
    ----------
    database : Optional[Database], optional
        The database for which the estimates should be calculated. If omitted, the database system is inferred from the
        database pool.
    enable_cache : bool, optional
        Whether cardinalities of intermediates should be cached *for the lifetime of the estimator object*. Defaults to
        *False*.
    allow_cross_products : bool, optional
        Whether cardinality estimates for arbitrary cross products should be included.
    """

    def __init__(
        self,
        database: Optional[Database] = None,
        *,
        enable_cache: bool = False,
        allow_cross_products: bool = False,
    ) -> None:
        super().__init__(allow_cross_products=allow_cross_products)
        self.database = (
            database
            if database is not None
            else DatabasePool.get_instance().current_database()
        )
        self.cache_enabled = enable_cache
        self._cardinality_cache: dict[SqlQuery, Cardinality] = {}

    def describe(self) -> dict:
        return {"name": "true-cards", "database": self.database.describe()}

    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        intermediate = util.enlist(intermediate)
        subquery = transform.extract_subquery(query, intermediate)
        subquery = transform.as_count_star_query(subquery)
        if subquery in self._cardinality_cache:
            return self._cardinality_cache[subquery]
        cardinality = Cardinality(self.database.execute_query(subquery))
        self._cardinality_cache[subquery] = cardinality
        return cardinality

    def reset_cache(self) -> None:
        self._cardinality_cache.clear()


def _parse_tables(tabs: str) -> set[TableReference]:
    """Utility to load tables from their JSON representation.

    Parameters
    ----------
    tabs : str
        The raw JSON data

    Returns
    -------
    set[TableReference]
        The corresponding tables
    """
    return {parser.load_table_json(t) for t in json.loads(tabs)}


class PreComputedCardinalities(CardinalityEstimator):
    """Re-uses existing cardinalities from an external data source.

    The cardinalities have to be stored in a CSV file which follows a certain structure. Some details can be customized (e.g.
    column names). Most importantly, queries have to be identified via their labels. See parameters for details.

    Parameters
    ----------
    workload : workloads.Workload
        The workload which was used to calculate the cardinalities. This is required to determine the query label based on an
        input query. Each hint generator can only support a specific workload.
    lookup_table_path : str
        The file path to the CSV file containing the cardinalities.
    include_cross_products : bool, optional
        Whether cardinality estimates for arbitrary cross products are contained in the CSV file and hence can be used during
        estimation. By default this is disabled.
    default_cardinality : Optional[Cardinality], optional
        In case no cardinality estimate exists for a specific intermediate, a default cardinality can be used instead. In case
        no default value has been specified, an error would be raised. Notice that a ``None`` value unsets the default. If the
        client should handle this situation instead, another value (e.g. ``Cardinality.unknown()`` has to be used).
        The default cardinality is not used if live fallback is enabled (see below).
    label_col : str, optional
        The column in the CSV file that contains the query labels. Defaults to *label*.
    tables_col : str, optional
        The column in the CSV file that contains the (JSON serialized) tables that form the current intermediate result of the
        current query. Defaults to *tables*.
    cardinality_col : str, optional
        The column in the CSV file that contains the actual cardinalities. Defaults to *cardinality*.
    live_fallback : bool, optional
        Whether to fall back to a live database in case no cardinality estimate is found in the CSV file. This is off by
        default.
    error_on_missing_card : bool, optional
        If live fallback is disabled and we did not find a cardinality estimate for a specific intermediate, we will raise an
        error by default. If this is not desired and missing values can be handled by the client, this behavior can be disabled
        with this parameter.
    live_fallback_style : Literal["actual", "estimated"], optional
        In case the fallback is enabled, this customizes the calculation strategy. "actual" will calculate the true cardinality
        of the intermediate in question, whereas "estimated" (the default) will use the native optimizer to estimate the
        cardinality.
    live_db : Optional[Database], optional
        The database system that should be used in case of a live fallback. If omitted, the database system is inferred from
        the database pool.
    save_live_fallback_results : bool, optional
        Whether the cardinalities computed by the live fallback should be stored in the original file containing the lookup
        table. This is only used if live fallback is active and enabled by default.
    """

    def __init__(
        self,
        workload: Workload,
        lookup_table_path: str,
        *,
        include_cross_products: bool = False,
        default_cardinality: Optional[Cardinality] = None,
        label_col: str = "label",
        tables_col: str = "tables",
        cardinality_col: str = "cardinality",
        live_fallback: bool = False,
        error_on_missing_card: bool = True,
        live_db: Optional[Database] = None,
        live_fallback_style: Literal["actual", "estimated"] = "estimated",
        save_live_fallback_results: bool = True,
    ) -> None:
        super().__init__(allow_cross_products=include_cross_products)
        self._workload = workload
        self._label_col = label_col
        self._tables_col = tables_col
        self._card_col = cardinality_col
        self._default_card = default_cardinality
        self._lookup_df_path = lookup_table_path

        self._error_on_missing_card = error_on_missing_card
        self._live_db: Optional[Database] = None
        if live_fallback:
            self._live_db = (
                DatabasePool.get_instance().current_database()
                if live_db is None
                else live_db
            )
        else:
            self._live_db = None
        self._live_fallback_style = live_fallback_style
        self._save_life_fallback = save_live_fallback_results

        true_card_df = pd.read_csv(
            lookup_table_path, converters={tables_col: _parse_tables}
        )
        self._df_cols = set(true_card_df.columns)
        self._cards: dict[tuple[str, frozenset[TableReference]], Cardinality] = {}
        for _, row in true_card_df.iterrows():
            label = row[label_col]
            tables = row[tables_col]
            card = row[cardinality_col]
            self._cards[(label, frozenset(tables))] = Cardinality(card)

    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        intermediate = frozenset(util.enlist(intermediate))
        label = self._workload.label_of(query)
        card = self._cards.get((label, intermediate))
        if card is not None:
            return card
        return self._use_default(query, intermediate)

    def describe(self) -> dict:
        return {
            "name": "pre-computed-cards",
            "location": self._lookup_df_path,
            "workload": self._workload.name,
        }

    def _use_default(
        self, query: SqlQuery, intermediate: frozenset[TableReference]
    ) -> Cardinality:
        if self._live_db is not None:
            return self._use_live_fallback(query, intermediate)

        if self._default_card is not None:
            return self._default_card

        if self._error_on_missing_card:
            raise ValueError(
                f"No cardinality estimate found for query '{query}' and tables '{intermediate}' and no default value specified."
            )
        return Cardinality.unknown()

    def _use_live_fallback(
        self, query: SqlQuery, intermediate: frozenset[TableReference]
    ) -> Cardinality:
        assert self._live_db is not None
        query_fragment = transform.extract_subquery(query, intermediate)

        match self._live_fallback_style:
            case "actual":
                true_card_query = transform.as_count_star_query(query_fragment)
                cardinality = Cardinality(self._live_db.execute_query(true_card_query))
            case "estimated":
                cardinality = self._live_db.optimizer().cardinality_estimate(
                    query_fragment
                )

        if self._save_life_fallback:
            self._dump_fallback_estimate(query, intermediate, cardinality)
        return cardinality

    def _dump_fallback_estimate(
        self,
        query: SqlQuery,
        tables: frozenset[TableReference],
        cardinality: Cardinality,
    ) -> None:
        """Stores a newly computed cardinality estimate in the lookup table.

        Parameters
        ----------
        query : SqlQuery
            The query for which the cardinality was estimated
        tables : frozenset[TableReference]
            The tables that form the current intermediate
        cardinality : int
            The computed cardinality
        """
        result_row = {}
        result_row[self._label_col] = [self._workload.label_of(query)]
        result_row[self._tables_col] = [util.to_json(tables)]

        if "query" in self._df_cols:
            result_row["query"] = [str(query)]
        if "query_fragment" in self._df_cols:
            result_row["query_fragment"] = [
                str(transform.extract_query_fragment(query, tables))
            ]

        result_row[self._card_col] = [cardinality]
        result_df = pd.DataFrame(result_row)
        result_df.to_csv(self._lookup_df_path, index=False, mode="a", header=False)


class CardinalityDistortion(CardinalityEstimator):
    """Decorator to simulate errors during cardinality estimation.

    The distortion service uses cardinality estimates produced by an actual estimator and mofifies its estimations to simulate
    the effect of deviations and misestimates.

    Behavior regarding cross products is inferred based on the behavior of the actual estimator.

    Parameters
    ----------
    estimator : CardinalityEstimator
        The actual estimator that calculates the "correct" cardinalities.
    distortion_factor : float
        How much the cardinalities are allowed to deviate from the original estimations. Values > 1 simulate overestimation
        whereas values < 1 simulate underestimation. For example, a distortion factor of 0.5 means that the final estimates can
        deviate at most half of the original cardinalities, pr a factor of 1.3 allows an overestimation of up to 30%.
    distortion_strategy : Literal["fixed", "random"], optional
        How the estimation errors should be calculated. The default *fixed* strategy always applies the exact distrotion factor
        to the cardinalities. For example, an estimate of 1000 tuples would always become 1300 tuples with a distrotion factor
        of 1.3. On the other hand the *random* strategy allows any error between 1 and the desired factor and selects the
        specific distortion at random. For example, an estimate of 100 could become any cardinality between 50 and 100 tuples
        with a distortion factor of 0.5.
    """

    def __init__(
        self,
        estimator: CardinalityEstimator,
        distortion_factor: float,
        *,
        distortion_strategy: Literal["fixed", "random"] = "fixed",
    ) -> None:
        super().__init__(allow_cross_products=estimator.allow_cross_products)
        self.estimator = estimator
        self.distortion_factor = distortion_factor
        self.distortion_strategy = distortion_strategy

    def describe(self) -> dict:
        return {
            "name": "cardinality-distortion",
            "estimator": "distortion",
            "distortion_factor": self.distortion_factor,
            "distortion_strategy": self.distortion_strategy,
        }

    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        card_est = self.estimator.calculate_estimate(query, intermediate)
        if not card_est.is_valid():
            return Cardinality.unknown()
        if self.distortion_strategy == "fixed":
            distortion_factor = self.distortion_factor
        elif self.distortion_strategy == "random":
            distortion_factor = random.uniform(
                min(self.distortion_factor, 1.0), max(self.distortion_factor, 1.0)
            )
        else:
            raise ValueError(
                f"Unknown distortion strategy: '{self.distortion_strategy}'"
            )
        return round(card_est * distortion_factor)
