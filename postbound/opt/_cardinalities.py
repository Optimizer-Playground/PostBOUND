from __future__ import annotations

import atexit
import collections
import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .. import PlanParameterization, parser, transform
from .._core import Cardinality, TableReference, TimeS
from .._stages import (
    CardinalityEstimator,
)
from ..db import Database, DatabasePool, TimeoutSupport
from ..qal import SqlQuery
from ..util import Logger, jsondict, make_logger


class PerfectCardinalities(CardinalityEstimator):
    """Perfect (true) cardinalities are computed in an online fashion by executing COUNT(\\*) queries on a database.

    Re-computing cardinalities can be pretty expensive, so it is probably a good idea to combine this estimator with
    the `OfflineCardinalities` or at least the `CardinalityCache`. `cached` functions as an easy entry point to
    create such a combination.

    Parameters
    ----------
    database : Database, optional
        The database to use for cardinality estimation. If not provided, the current database from the DatabasePool
        will be used.
    allow_cross_products : bool, optional
        Whether to allow cross products in the query plan. If enabled, the estimator will compute cardinalities for
        cross products as part of `estimate_cardinalities` or `generate_plan_parameters`. Since this can be incredibly
        expensive, it is disabled by default.
    timeout : TimeS, optional
        If the database provides timeout support, this is the maximum number of seconds to wait for a cardinality
        "estimate" (computation). If the query is not completed within this amount of time, an unknown cardinality will
        be returned.
    """

    @staticmethod
    def cached(
        database: Database | None = None,
        *,
        offline_file: Path | str,
        allow_cross_products: bool = False,
        timeout: TimeS | None = None,
    ) -> CardinalityEstimator:
        estimator = PerfectCardinalities(database, allow_cross_products=allow_cross_products, timeout=timeout)
        offline = OfflineCardinalities(offline_file, fallback=estimator)
        return offline

    def __init__(
        self, database: Database | None = None, *, allow_cross_products: bool = False, timeout: TimeS | None = None
    ) -> None:
        super().__init__(allow_cross_products=allow_cross_products)
        self.database = database if database is not None else DatabasePool.get_instance().current_database()
        if not self.database.provides(TimeoutSupport) and timeout is not None:
            raise ValueError(f"Database {self.database} does not provide timeout support")
        self._timeout = timeout

    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        subquery = transform.extract_subquery(query, intermediate)
        subquery = transform.as_count_star_query(subquery)

        if self._timeout is not None:
            assert isinstance(self.database, TimeoutSupport)
            card = self.database.execute_with_timeout(subquery, timeout=self._timeout)
        else:
            card = self.database.execute_query(subquery, raw=True)

        if card is None:
            return Cardinality.unknown()

        return Cardinality(card[0][0])

    def describe(self) -> jsondict:
        return {"name": "perfect-cardinalities", "database": self.database.describe()}


@dataclasses.dataclass
class _CardinalityCacheEntry:
    complete: PlanParameterization | None
    intermediate: dict[frozenset[TableReference], Cardinality]

    @staticmethod
    def create() -> _CardinalityCacheEntry:
        return _CardinalityCacheEntry(None, {})


class CardinalityCache(CardinalityEstimator):
    """Wraps another cardinality estimator and caches its results in memory.

    Parameters
    ----------
    estimator : CardinalityEstimator
        The underlying estimator to use for cardinality estimation. This estimator will be called if a specific
        intermediate is not yet cached.

    See Also
    --------
    OfflineCardinalities : if the cache should be persisted to disk
    """

    def __init__(self, estimator: CardinalityEstimator) -> None:
        super().__init__(allow_cross_products=estimator.allow_cross_products)
        self.estimator = estimator
        self._cache = collections.defaultdict(_CardinalityCacheEntry.create)

    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        intermediate = frozenset([intermediate] if isinstance(intermediate, TableReference) else intermediate)
        cache_line = self._cache[query]

        cached = cache_line.intermediate.get(intermediate)
        if cached is not None:
            return cached

        card = self.estimator.calculate_estimate(query, intermediate)
        cache_line.intermediate[intermediate] = card
        return card

    def estimate_cardinalities(self, query: SqlQuery) -> PlanParameterization:
        cache_line = self._cache[query]
        if cache_line.complete is not None:
            return cache_line.complete

        params = self.estimator.estimate_cardinalities(query)
        cache_line.complete = params
        for intermediate, card in params.cardinalities.items():
            cache_line.intermediate[intermediate] = card

        return params

    def describe(self) -> jsondict:
        return {"name": "cardinality-cache", "estimator": self.estimator.describe()}


def _load_offline_json(path: Path) -> dict[SqlQuery, Cardinality]:
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)
    return {parser.parse_query(query): Cardinality(card) for query, card in raw_data.items()}


def _load_offline_csv(path: Path) -> dict[SqlQuery, Cardinality]:
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    return {parser.parse_query(row["query"]): Cardinality(row["cardinality"]) for _, row in df.iterrows()}


def _load_offline_parquet(path: Path) -> dict[SqlQuery, Cardinality]:
    if not path.exists():
        return {}

    df = pd.read_parquet(path)
    return {parser.parse_query(row["query"]): Cardinality(row["cardinality"]) for _, row in df.iterrows()}


def _dump_offline_json(cardinalities: dict[SqlQuery, Cardinality], to: Path) -> None:
    normalized = {str(query): card.value for query, card in cardinalities.items()}
    with open(to, "w", encoding="utf-8") as f:
        json.dump(normalized, f)


def _dump_offline_csv(cardinalities: dict[SqlQuery, Cardinality], to: Path) -> None:
    normalized = {"query": [], "cardinality": []}
    for query, card in cardinalities.items():
        normalized["query"].append(str(query))
        normalized["cardinality"].append(card.value)
    df = pd.DataFrame(normalized)
    df.to_csv(to, index=False)


def _dump_offline_parquet(cardinalities: dict[SqlQuery, Cardinality], to: Path) -> None:
    normalized = {"query": [], "cardinality": []}
    for query, card in cardinalities.items():
        normalized["query"].append(str(query))
        normalized["cardinality"].append(card.value)
    df = pd.DataFrame(normalized)
    df.to_parquet(to, index=False)


class _NativeCardinalities(CardinalityEstimator):
    def __init__(self, database: Database) -> None:
        super().__init__(allow_cross_products=False)
        self._database = database

    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        subquery = transform.extract_subquery(query, intermediate)

        return self._database.optimizer().cardinality_estimate(subquery)


class OfflineCardinalities(CardinalityEstimator):
    """Loads pre-computed cardinalities from disk.

    The offline cardinalities can be combined with a fallback estimator to use if a specific intermediate is not found
    in the offline file. In this case, the estimator will automatically update the offline file at the end of the
    program execution.

    Currently, the following file formats are supported:

    - JSON: each intermediate query functions as a key and the corresponding cardinality is the value
    - CSV: two columns, one for the intermediate query and one for the corresponding cardinality
    - Parquet: two columns, one for the intermediate query and one for the corresponding cardinality

    If the file does not exist, a fallback estimator must be provided. In this case, the file will be created at the
    end of the program execution with all cardinalities that have been computed in the meantime (using the fallback
    estimator).

    The two most-commonly used estimators are perfect (true) cardinalities and native cardinalities. Both can be created
    directly using dedicated factory methods.

    Parameters
    ----------
    source : Path | str
        The offline file containing the pre-computed cardinalities.
    fallback : CardinalityEstimator, optional
        The fallback estimator to use if a specific intermediate is not found in the offline file. If not provided, a
        KeyError will be raised when an intermediate is not found.
    log : Logger, optional
        A logger to document when the fallback estimator is used.
    """

    @staticmethod
    def fallback_native(
        source: Path | str, *, database: Database | None = None, log: Logger | None = None
    ) -> OfflineCardinalities:
        """Creates an OfflineCardinalities estimator with a native fallback estimator.

        Parameters
        ----------
        source : Path | str
            The offline file containing the pre-computed cardinalities.
        database : Database, optional
            The database to use for the native fallback estimator. If not provided, the current database from the
            DatabasePool will be used.
        log : Logger, optional
                A logger to document when the native estimator is used.
        """
        database = database or DatabasePool.get_instance().current_database()
        fallback = _NativeCardinalities(database)
        return OfflineCardinalities(source, fallback=fallback, log=log)

    @staticmethod
    def fallback_perfect(
        source: Path | str, *, database: Database | None = None, timeout: TimeS | None = None, log: Logger | None = None
    ) -> OfflineCardinalities:
        """Creates an OfflineCardinalities estimator with a perfect fallback estimator.

        Parameters
        ----------
        source : Path | str
            The offline file containing the pre-computed cardinalities.
        database : Database, optional
            The database to use for the perfect fallback estimator. If not provided, the current database from the
            DatabasePool will be used.
        timeout : TimeS, optional
            If the database provides timeout support, this is the maximum number of seconds to wait for a cardinality
            "estimate" (computation). If the query is not completed within this amount of time, an unknown cardinality
            will be returned.
        log : Logger, optional
                A logger to document when the perfect estimator is used.
        """
        database = database or DatabasePool.get_instance().current_database()
        fallback = PerfectCardinalities(database, timeout=timeout)
        return OfflineCardinalities(source, fallback=fallback, log=log)

    def __init__(
        self, source: Path | str, *, fallback: CardinalityEstimator | None = None, log: Logger | None = None
    ) -> None:
        super().__init__(allow_cross_products=fallback.allow_cross_products if fallback is not None else False)

        source = Path(source)
        match source.suffix:
            case ".json":
                self._cardinalities = _load_offline_json(source)
            case ".csv":
                self._cardinalities = _load_offline_csv(source)
            case ".parquet":
                self._cardinalities = _load_offline_parquet(source)
            case _:
                raise ValueError(f"Unsupported file type: '{source.suffix}'")

        self._fallback = fallback
        self._dump_requested = False
        self._source = source

        self._complete_cache: dict[SqlQuery, PlanParameterization] = {}

        self._log = log if log is not None else make_logger(False)

    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        subquery = transform.extract_subquery(query, intermediate)
        subquery = transform.as_count_star_query(subquery)

        card = self._cardinalities.get(subquery)
        if card is not None:
            return card

        if self._fallback is None:
            raise KeyError(f"No cardinality estimate found for query '{subquery}' and no fallback estimator specified.")

        self._log(f"Using fallback estimator for query '{subquery}'")
        card = self._fallback.calculate_estimate(query, intermediate)
        self._cardinalities[subquery] = card

        if self._dump_requested:
            return card

        atexit.register(self._dump_cardinalities)
        self._dump_requested = True
        return card

    def estimate_cardinalities(self, query: SqlQuery) -> PlanParameterization:
        cached = self._complete_cache.get(query)
        if cached is not None:
            return cached

        params = super().estimate_cardinalities(query)
        self._complete_cache[query] = params
        return params

    def describe(self) -> jsondict:
        return {
            "name": "offline-cardinalities",
            "source": str(self._source),
            "fallback": self._fallback.describe() if self._fallback is not None else None,
        }

    def _dump_cardinalities(self) -> None:
        match self._source.suffix:
            case ".json":
                _dump_offline_json(self._cardinalities, to=self._source)
            case ".csv":
                _dump_offline_csv(self._cardinalities, to=self._source)
            case ".parquet":
                _dump_offline_parquet(self._cardinalities, to=self._source)
            case _:
                raise ValueError(f"Unsupported file type: '{self._source.suffix}'")
