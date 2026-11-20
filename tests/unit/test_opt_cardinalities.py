"""Tests for `postbound.opt._cardinalities` -- cardinality estimators that wrap or replace a native estimator.

All tests are tier 0. `OfflineCardinalities` reads and writes its files below pytest's `tmp_path`; the files are
written the way a user would produce them (`json.dump` / `DataFrame.to_csv` / `DataFrame.to_parquet`) with plain SQL
strings as keys. Updating the source file is deferred to program exit via `atexit.register`, so every test that lets the
estimator update its source replaces `atexit.register` with a recorder (through `monkeypatch`), both to observe the
registration and to avoid leaving a dump callback behind in the test process.

Currently this module only covers `OfflineCardinalities`. `PerfectCardinalities` and the native fallback need a
database and are not covered here; `CardinalityCache` and `WorkloadSpecificCardinalities` are not covered yet.
"""

from __future__ import annotations

import atexit
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from postbound import parser
from postbound._core import Cardinality, TableReference
from postbound.opt import OfflineCardinalities
from postbound.opt._cardinalities import _StaticCardinalities

pytestmark = pytest.mark.usefixtures("no_column_binding")

R = TableReference("r")
S = TableReference("s")

QUERY = parser.parse_query("SELECT * FROM r, s WHERE r.a = s.b", bind_columns=False)

FORMATS = ["json", "csv", "parquet"]


def write_offline_file(path: Path, cardinalities: dict[str, int]) -> Path:
    """Writes `cardinalities` (SQL string -> cardinality) in the format indicated by the suffix of `path`."""
    match path.suffix:
        case ".json":
            path.write_text(json.dumps(cardinalities), encoding="utf-8")
        case ".csv":
            pd.DataFrame({"query": list(cardinalities), "cardinality": list(cardinalities.values())}).to_csv(
                path, index=False
            )
        case ".parquet":
            pd.DataFrame({"query": list(cardinalities), "cardinality": list(cardinalities.values())}).to_parquet(
                path, index=False
            )
        case _:
            raise ValueError(f"Unsupported test file type: '{path.suffix}'")
    return path


@pytest.fixture
def atexit_callbacks(monkeypatch: pytest.MonkeyPatch) -> list[Callable[[], None]]:
    """Records the callbacks that are registered with `atexit` instead of running them at interpreter exit."""
    callbacks: list[Callable[[], None]] = []
    monkeypatch.setattr(atexit, "register", callbacks.append)
    return callbacks


# -- OfflineCardinalities: loading -------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_offline_cardinalities_finds_an_entry_stored_as_count_star_query(tmp_path: Path, fmt: str) -> None:
    """``SELECT COUNT(*)`` is the format the estimator itself dumps, so it must be found again after reloading."""
    source = write_offline_file(tmp_path / f"cards.{fmt}", {"SELECT COUNT(*) FROM r, s WHERE r.a = s.b": 42})
    estimator = OfflineCardinalities(source)

    card = estimator.calculate_estimate(QUERY, {R, S})

    assert card == Cardinality(42)


def test_offline_cardinalities_looks_up_each_intermediate_by_its_own_subquery(tmp_path: Path) -> None:
    source = write_offline_file(
        tmp_path / "cards.json",
        {"SELECT * FROM r": 100, "SELECT * FROM s": 200, "SELECT * FROM r, s WHERE r.a = s.b": 42},
    )
    estimator = OfflineCardinalities(source)

    assert estimator.calculate_estimate(QUERY, R) == Cardinality(100)
    assert estimator.calculate_estimate(QUERY, S) == Cardinality(200)
    assert estimator.calculate_estimate(QUERY, {R, S}) == Cardinality(42)


def test_offline_cardinalities_without_a_fallback_raises_for_a_missing_intermediate(tmp_path: Path) -> None:
    source = write_offline_file(tmp_path / "cards.json", {"SELECT * FROM r": 100})
    estimator = OfflineCardinalities(source)

    with pytest.raises(KeyError, match="No cardinality estimate found"):
        estimator.calculate_estimate(QUERY, {R, S})


def test_offline_cardinalities_treats_a_missing_source_file_as_empty(tmp_path: Path) -> None:
    estimator = OfflineCardinalities(tmp_path / "does-not-exist.json")

    with pytest.raises(KeyError, match="No cardinality estimate found"):
        estimator.calculate_estimate(QUERY, R)


def test_offline_cardinalities_rejects_an_unsupported_file_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Unsupported file type: '\.txt'"):
        OfflineCardinalities(tmp_path / "cards.txt")


# -- OfflineCardinalities: fallback and updating the source ------------------------------------------------


def test_offline_cardinalities_prefers_the_offline_value_over_the_fallback(tmp_path: Path) -> None:
    source = write_offline_file(tmp_path / "cards.json", {"SELECT * FROM r": 100})
    estimator = OfflineCardinalities(source, fallback=_StaticCardinalities(7), update_source=False)

    assert estimator.calculate_estimate(QUERY, R) == Cardinality(100)


def test_offline_cardinalities_uses_the_fallback_for_a_missing_intermediate(tmp_path: Path) -> None:
    source = write_offline_file(tmp_path / "cards.json", {"SELECT * FROM r": 100})
    estimator = OfflineCardinalities(source, fallback=_StaticCardinalities(7), update_source=False)

    assert estimator.calculate_estimate(QUERY, {R, S}) == Cardinality(7)


def test_offline_cardinalities_without_update_source_never_schedules_a_dump(
    tmp_path: Path, atexit_callbacks: list[Callable[[], None]]
) -> None:
    estimator = OfflineCardinalities(tmp_path / "cards.json", fallback=_StaticCardinalities(7), update_source=False)

    estimator.calculate_estimate(QUERY, {R, S})

    assert atexit_callbacks == []
    assert not (tmp_path / "cards.json").exists()


def test_offline_cardinalities_schedules_a_single_dump_for_several_fallback_estimates(
    tmp_path: Path, atexit_callbacks: list[Callable[[], None]]
) -> None:
    estimator = OfflineCardinalities(tmp_path / "cards.json", fallback=_StaticCardinalities(7))

    estimator.calculate_estimate(QUERY, R)
    estimator.calculate_estimate(QUERY, S)

    assert len(atexit_callbacks) == 1


@pytest.mark.parametrize("fmt", FORMATS)
def test_offline_cardinalities_reloads_its_own_dump(
    tmp_path: Path, fmt: str, atexit_callbacks: list[Callable[[], None]]
) -> None:
    source = tmp_path / f"cards.{fmt}"
    estimator = OfflineCardinalities(source, fallback=_StaticCardinalities(7))
    estimator.calculate_estimate(QUERY, R)
    estimator.calculate_estimate(QUERY, {R, S})

    (dump,) = atexit_callbacks
    dump()
    reloaded = OfflineCardinalities(source)

    assert reloaded.calculate_estimate(QUERY, R) == Cardinality(7)
    assert reloaded.calculate_estimate(QUERY, {R, S}) == Cardinality(7)


# -- OfflineCardinalities: factory methods -----------------------------------------------------------------


def test_offline_cardinalities_fallback_dummy_returns_an_unknown_cardinality(tmp_path: Path) -> None:
    estimator = OfflineCardinalities.fallback_dummy(tmp_path / "cards.json")

    card = estimator.calculate_estimate(QUERY, {R, S})

    assert card.is_unknown()


# -- regression tests --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_offline_cardinalities_finds_an_entry_stored_as_select_star_query(tmp_path: Path, fmt: str) -> None:
    """Regression guard for 2b54b03: lookups use the ``SELECT COUNT(*)`` form of the intermediate subquery, but the
    loaders stored every query exactly as written in the file. An entry written as ``SELECT *`` could therefore never be
    found, and every lookup fell through to the fallback estimator (or raised a `KeyError` without one).
    """
    source = write_offline_file(tmp_path / f"cards.{fmt}", {"SELECT * FROM r, s WHERE r.a = s.b": 42})
    estimator = OfflineCardinalities(source)

    card = estimator.calculate_estimate(QUERY, {R, S})

    assert card == Cardinality(42)


def test_offline_cardinalities_loads_the_cardinality_column_of_a_csv_file(tmp_path: Path) -> None:
    """Regression guard for the CSV loader fix in v0.23.5: 2b54b03 passed ``Cardinality`` as a pandas converter for the
    *cardinality* column. pandas hands converters the raw cell *string*, so ``Cardinality("42")`` failed with a
    `TypeError` in its ``value < 0`` check, and every non-empty CSV source crashed the estimator's constructor.
    """
    source = write_offline_file(tmp_path / "cards.csv", {"SELECT * FROM r": 100})

    estimator = OfflineCardinalities(source)

    assert estimator.calculate_estimate(QUERY, R) == Cardinality(100)


def test_offline_cardinalities_fallback_static_returns_the_static_value(tmp_path: Path) -> None:
    """Regression guard for the `_StaticCardinalities` fix in v0.23.5: its ``__init__`` never called
    ``CardinalityEstimator.__init__``, so the ``allow_cross_products`` attribute was missing. `OfflineCardinalities`
    reads that attribute from its fallback, hence `fallback_static` raised an `AttributeError` before it could return.
    """
    estimator = OfflineCardinalities.fallback_static(tmp_path / "cards.json", 7)

    card = estimator.calculate_estimate(QUERY, {R, S})

    assert card == Cardinality(7)
