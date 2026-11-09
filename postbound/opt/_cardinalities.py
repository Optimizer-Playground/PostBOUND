from __future__ import annotations

import atexit
import collections
import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, override

import pandas as pd

from .. import parser, transform
from .._core import Cardinality, TableReference, TimeS
from .._hints import PlanParameterization
from .._stages import (
    CardinalityEstimator,
)
from ..db import Database, DatabasePool, TimeoutSupport
from ..qal import SqlQuery
from ..util import Logger, jsondict, make_logger, read_df
from ..workloads import Workload


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

    @override
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

    @override
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

    @override
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

    @override
    def estimate_cardinalities(self, query: SqlQuery) -> PlanParameterization:
        cache_line = self._cache[query]
        if cache_line.complete is not None:
            return cache_line.complete

        params = self.estimator.estimate_cardinalities(query)
        cache_line.complete = params
        for intermediate, card in params.cardinalities.items():
            cache_line.intermediate[intermediate] = card

        return params

    @override
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

    @override
    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        subquery = transform.extract_subquery(query, intermediate)

        return self._database.optimizer().cardinality_estimate(subquery)


class _DummyCardinalities(CardinalityEstimator):
    def __init__(self) -> None:
        super().__init__(allow_cross_products=False)

    @override
    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        return Cardinality.unknown()


class _StaticCardinalities(CardinalityEstimator):
    def __init__(self, value: float | Cardinality) -> None:
        self._value = Cardinality.of(value)

    @override
    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        return self._value


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
    update_source : bool, optional
        Whether to update the offline file with newly computed cardinalities. If set to False, fallback cardinalities
        are only retained for the lifetime of the estimator.
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
        return OfflineCardinalities(source, fallback=fallback, update_source=False, log=log)

    @staticmethod
    def fallback_perfect(
        source: Path | str,
        *,
        database: Database | None = None,
        timeout: TimeS | None = None,
        update_source: bool = True,
        log: Logger | None = None,
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
        updated_source : bool, optional
            Whether to update the offline file with newly computed cardinalities
        log : Logger, optional
                A logger to document when the perfect estimator is used.
        """
        database = database or DatabasePool.get_instance().current_database()
        fallback = PerfectCardinalities(database, timeout=timeout)
        return OfflineCardinalities(source, fallback=fallback, update_source=update_source, log=log)

    @staticmethod
    def fallback_dummy(source: Path | str, *, log: Logger | None = None) -> OfflineCardinalities:
        """Creates an OfflineCardinalities estimator that ignores all missing cardinalities.

        Parameters
        ----------
        source : Path | str
            The offline file containing the pre-computed cardinalities.
        log : Logger, optional
                A logger to document when estimates are missing.
        """
        return OfflineCardinalities(source, fallback=_DummyCardinalities(), update_source=False, log=log)

    @staticmethod
    def fallback_static(
        source: Path | str, value: float | Cardinality, *, log: Logger | None = None
    ) -> OfflineCardinalities:
        """Creates an OfflineCardinalities estimator that returns a fixed value for all missing cardinalities.

        Parameters
        ----------
        source : Path | str
            The offline file containing the pre-computed cardinalities.
        value : float | Cardinality
            The static value to return for all missing cardinalities.
        log : Logger, optional
                A logger to document when estimates are missing.
        """
        return OfflineCardinalities(source, fallback=_StaticCardinalities(value), update_source=False, log=log)

    def __init__(
        self,
        source: Path | str,
        *,
        fallback: CardinalityEstimator | None = None,
        update_source: bool = True,
        log: Logger | None = None,
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
        self._update_source = update_source

        self._complete_cache: dict[SqlQuery, PlanParameterization] = {}

        self._log = log or make_logger(False)

    @override
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

        if self._dump_requested or not self._update_source:
            return card

        atexit.register(self._dump_cardinalities)
        self._dump_requested = True
        return card

    @override
    def estimate_cardinalities(self, query: SqlQuery) -> PlanParameterization:
        cached = self._complete_cache.get(query)
        if cached is not None:
            return cached

        params = super().estimate_cardinalities(query)
        self._complete_cache[query] = params
        return params

    @override
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


class WorkloadSpecificCardinalities(CardinalityEstimator):
    """Loads pre-computed cardinalities for a specific workload from a DataFrame.

    This estimator differs from `OfflineCardinalities` in several ways:

    1. Cardinalities are stored in a DataFrame.
    2. The DataFrame can be loaded from disk (see `from_file`) or passed directly.
    3. Subqueries are identified by their workload label and intermediate, rather than by the full SQL query.

    The third point is particularly useful to detect matching cardinalities in an order-agnostic way: Consider a
    situation where during cardinality computation the subquery was
    ``SELECT * FROM R, S WHERE R.id = S.id AND R.a < 42``, but during workload evaluation it is
    ``SELECT * FROM R, S WHERE R.a < 42 AND R.id = S.id``. `OfflineCardinalities` uses a direct hash-based lookup and
    would miss the cardinality. On the other hand, `WorkloadSpecificCardinalities` uses the query and the intermediate
    tables to identify the subquery, independent of any ordering within the intermediate. This allows it to find the
    matching cardinality even if the subquery is written in a different order.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the pre-computed cardinalities. See the ``*_col`` parameters below for required columns
        and their expected contents.
        **Mind that the estimator will assume full control over the DataFrame. This includes potential in-place
        modifications. If this is not acceptable, make sure to pass a copy!**
    workload : Workload[str, SqlQuery]
        The workload to which the cardinalities belong.
    label_col : str, optional
        The name of the column in the DataFrame that contains the workload label for each query.
        Defaults to ``"label"``.
    intermediate_col : str, optional
        The name of the column in the DataFrame that contains the intermediate tables for each query.
        Defaults to ``"intermediate"``.
        The intermediates must either be a pre-generated iterable of table references, or a JSON-encoded string of such
        an iterable.
    cardinality_col : str, optional
        The name of the column in the DataFrame that contains the cardinality for each intermediate.
        Defaults to ``"cardinality"``.
    fallback : CardinalityEstimator, optional
        The fallback estimator to use if a specific intermediate is not found in the DataFrame.
        If not provided, a KeyError will be raised when an intermediate is not found.
    log : Logger
        A logger to document when the fallback estimator is used.
    """

    @staticmethod
    def from_file(
        path: Path | str,
        *,
        workload: Workload[str, SqlQuery],
        label_col: str = "label",
        intermediate_col: str = "intermediate",
        cardinality_col: str = "cardinality",
        fallback: CardinalityEstimator | Literal["dummy"] | None = None,
    ) -> WorkloadSpecificCardinalities:
        """Creates a WorkloadSpecificCardinalities estimator from a file.

        Most parameters are the same as for the constructor. `path` is the path to the file containing the pre-computed
        cardinalities. The `fallback` also accepts the special ``"dummy"`` keyword. In this case, *unknown*  is returned
        for all missing cardinalities.
        """
        fallback = _DummyCardinalities() if fallback == "dummy" else fallback
        df = read_df(path)
        return WorkloadSpecificCardinalities(
            df,
            workload=workload,
            label_col=label_col,
            intermediate_col=intermediate_col,
            cardinality_col=cardinality_col,
            fallback=fallback,
        )

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        workload: Workload[str, SqlQuery],
        label_col: str = "label",
        intermediate_col: str = "intermediate",
        cardinality_col: str = "cardinality",
        fallback: CardinalityEstimator | None = None,
        log: Logger | None = None,
    ) -> None:
        super().__init__(allow_cross_products=False)

        self._log = log or make_logger(False)
        self._workload = workload

        sample = df.iloc[0]
        sample_intermediate = sample[intermediate_col]
        if isinstance(sample_intermediate, str):
            self._log("Intermediates appear to still be JSON-encoded. Parsing them now. This may take a while.")
            df[intermediate_col] = (
                df[intermediate_col]
                .map(json.loads)
                .map(lambda jsonized: frozenset(parser.load_table_json(entry) for entry in jsonized))
            )

        self._cardinalities: dict[tuple[SqlQuery, frozenset[TableReference]], Cardinality] = {}
        for _, row in df.iterrows():
            query = workload[row[label_col]]
            intermediate = frozenset(row[intermediate_col])
            card = Cardinality.of(row[cardinality_col])
            self._cardinalities[(query, intermediate)] = card

        self._complete_cache: dict[SqlQuery, PlanParameterization] = {}
        self._fallback = fallback

    @override
    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        intermediate = frozenset([intermediate] if isinstance(intermediate, TableReference) else intermediate)
        card = self._cardinalities.get((query, intermediate))
        if card is not None:
            return card

        label = self._workload.label_of(query)
        if self._fallback is None:
            raise KeyError(
                f"No cardinality estimate found for query '{label}' and intermediate '{intermediate}' and no fallback estimator specified."
            )

        self._log(f"Using fallback estimator for query '{label}' with intermediate '{intermediate}'")
        card = self._fallback.calculate_estimate(query, intermediate)
        self._cardinalities[(query, intermediate)] = card
        return card

    @override
    def estimate_cardinalities(self, query: SqlQuery) -> PlanParameterization:
        cached = self._complete_cache.get(query)
        if cached is not None:
            return cached

        params = super().estimate_cardinalities(query)
        self._complete_cache[query] = params
        return params

    @override
    def describe(self) -> jsondict:
        return {
            "name": "workload-specific-cardinalities",
            "workload": self._workload.name,
            "fallback": self._fallback.describe() if self._fallback is not None else None,
        }
