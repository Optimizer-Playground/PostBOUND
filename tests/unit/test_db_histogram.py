"""Tests for `postbound.db.Histogram` -- the bucketized value distribution used for range-predicate estimates.

`Histogram` is a pure value object, so everything here is tier 0 and built directly through the constructor. Most
tests use a hand-computable equi-depth histogram (`equi_depth`): three buckets ``[0, 10]``, ``(10, 20]`` and
``(20, 30]`` holding 10 rows each, so the true answer of every estimate can be derived by hand under the uniform
assumption -- e.g. 15 rows are ``<= 15`` and 15 rows are ``> 15``.

The three strategies have different contracts, which the tests check separately:

- "bound-lower" leaves out the bucket containing the value and never over-estimates;
- "bound-upper" counts that bucket entirely and never under-estimates;
- "approx-uni" interpolates within the bucket and satisfies ``frequency_below(v) + frequency_above(v) == n_rows``,
  since ``<=`` and ``>`` partition the rows;
- the deprecated "bound" is an alias for "bound-upper" that emits a `DeprecationWarning` at the caller.

All of them are exact on bucket bounds. The bounding contracts are checked against a concrete dataset
(`DATASET`), not only against the equi-depth histogram. Every strategy can be the histogram's default or be passed
per call via `interpolation`; the tests mostly use the per-call override and check separately that both routes agree.

How the histograms are *built* by the statistics catalogs is covered in `test_db_stats.py` (`PreciseStatistics`) and
the DuckDB tests in `tests/test_duckdb.py`.
"""

from __future__ import annotations

import warnings
from datetime import date

import pytest

from postbound.db import Histogram, HistogramApproximation
from postbound.util import jsonize

# -- fixtures -----------------------------------------------------------------------------------------------


def equi_depth(strategy: HistogramApproximation = "bound-upper") -> Histogram:
    """Buckets [0, 10], (10, 20], (20, 30] with 10 rows each. A fresh object, since the strategy is mutable."""
    return Histogram([10, 20, 30], [10, 10, 10], lower=0, bucket_interpolation=strategy)


STRATEGIES: list[HistogramApproximation] = ["bound-lower", "bound-upper", "approx-uni"]

# The rows behind `DATASET_HISTOGRAM`: buckets [1, 2], (2, 7], (7, 9].
DATASET = [1] * 3 + [2] * 5 + [4] * 2 + [7] * 6 + [9] * 4
DATASET_HISTOGRAM = Histogram([2, 7, 9], [8, 8, 4], lower=1)
DATASET_PROBES = [v / 2 for v in range(0, 22)]  # 0, 0.5, ..., 10.5: every bucket, every bound, and beyond


def true_below(value: float) -> int:
    return sum(1 for row in DATASET if row <= value)


def true_above(value: float) -> int:
    return sum(1 for row in DATASET if row > value)


# -- construction -------------------------------------------------------------------------------------------


def test_histogram_rejects_empty_bounds() -> None:
    with pytest.raises(ValueError, match="Bounds cannot be empty"):
        Histogram([], [], lower=0)


def test_histogram_rejects_bounds_and_frequencies_of_different_length() -> None:
    with pytest.raises(ValueError, match="Got 2 bounds and 3 frequencies"):
        Histogram([10, 20], [1, 2, 3], lower=0)


def test_histogram_defaults_to_the_approx_uni_strategy() -> None:
    assert Histogram([10], [5], lower=0).bucket_interpolation == "approx-uni"


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_histogram_accepts_every_supported_strategy(strategy: HistogramApproximation) -> None:
    assert equi_depth(strategy).bucket_interpolation == strategy


def test_histogram_derives_its_summary_properties() -> None:
    hist = Histogram([10, 20, 30], [4, 5, 7], lower=1)

    assert hist.n_rows == 16
    assert hist.n_buckets == 3
    assert hist.freq_per_bucket == 5  # 16 // 3, an integer average
    assert hist.lower == 1
    assert hist.upper == 30


def test_histogram_bounds_is_a_copy() -> None:
    hist = equi_depth()

    bounds = hist.bounds
    assert isinstance(bounds, list)
    bounds.append(40)

    assert hist.bounds == [10, 20, 30]


def test_histogram_exposes_its_buckets_as_bound_frequency_pairs() -> None:
    hist = Histogram([10, 20, 30], [4, 5, 7], lower=0)

    assert len(hist) == 3
    assert list(hist) == [(10, 4), (20, 5), (30, 7)]
    assert hist[1] == (20, 5)


# -- bucket_interpolation setter ----------------------------------------------------------------------------


def test_bucket_interpolation_setter_switches_the_strategy() -> None:
    hist = equi_depth("bound-upper")

    hist.bucket_interpolation = "bound-lower"

    assert hist.bucket_interpolation == "bound-lower"
    assert hist.frequency_below(15) == 10


def test_bucket_interpolation_setter_rejects_unknown_strategies() -> None:
    hist = equi_depth()

    with pytest.raises(ValueError):
        hist.bucket_interpolation = "linear"  # ty: ignore[invalid-assignment]

    assert hist.bucket_interpolation == "bound-upper"


# -- frequency_below ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_frequency_below_is_zero_below_the_lower_bound(strategy: HistogramApproximation) -> None:
    assert equi_depth().frequency_below(-5, interpolation=strategy) == 0


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("value", [30, 31, 1000], ids=["at-upper", "just-above", "far-above"])
def test_frequency_below_counts_every_row_from_the_upper_bound_on(strategy: HistogramApproximation, value: int) -> None:
    assert equi_depth().frequency_below(value, interpolation=strategy) == 30


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(("value", "expected"), [(10, 10), (20, 20)], ids=["first-bound", "second-bound"])
def test_frequency_below_is_exact_on_bucket_bounds(strategy: HistogramApproximation, value: int, expected: int) -> None:
    """``<=`` includes the bucket that ends at `value`."""
    assert equi_depth().frequency_below(value, interpolation=strategy) == expected


@pytest.mark.parametrize(("value", "expected"), [(0, 0), (5, 0), (12, 10), (15, 10), (19, 10), (25, 20)])
def test_frequency_below_with_lower_bound_strategy_leaves_out_the_containing_bucket(value: int, expected: int) -> None:
    """This includes `lower` itself: it is part of the first bucket (e.g. the column minimum for `PreciseStatistics`)."""
    assert equi_depth().frequency_below(value, interpolation="bound-lower") == expected


@pytest.mark.parametrize(("value", "expected"), [(0, 10), (5, 10), (12, 20), (15, 20), (19, 20), (25, 30)])
def test_frequency_below_with_upper_bound_strategy_counts_the_containing_bucket(value: int, expected: int) -> None:
    """This includes `lower` itself: it is part of the first bucket (e.g. the column minimum for `PreciseStatistics`)."""
    assert equi_depth().frequency_below(value, interpolation="bound-upper") == expected


@pytest.mark.parametrize("value", [0, 5, 12, 15, 25, 29.5])
def test_frequency_below_with_uniform_approximation_interpolates_within_the_bucket(value: float) -> None:
    """For this histogram, the uniform assumption puts exactly one row per unit, so the estimate equals `value`."""
    assert equi_depth().frequency_below(value, interpolation="approx-uni") == pytest.approx(value)


@pytest.mark.parametrize(
    ("strategy", "expected"), [("bound-lower", 10), ("bound-upper", 20)], ids=["bound-lower", "bound-upper"]
)
def test_frequency_below_with_bound_strategies_works_for_non_numeric_values(
    strategy: HistogramApproximation, expected: int
) -> None:
    hist = Histogram(["f", "p", "z"], [10, 10, 10], lower="a")

    assert hist.frequency_below("p", interpolation=strategy) == 20
    assert hist.frequency_below("k", interpolation=strategy) == expected


def test_frequency_below_with_uniform_approximation_interpolates_dates_like_numbers() -> None:
    """Dates are interpolated via their `timedelta` differences, so a histogram over days 10/20/30 of a month must
    estimate exactly like the same histogram over the integers 10/20/30.
    """
    dates = Histogram(
        [date(2024, 1, 10), date(2024, 1, 20), date(2024, 1, 30)],
        [10, 10, 10],
        lower=date(2024, 1, 1),
        bucket_interpolation="approx-uni",
    )
    ints = Histogram([10, 20, 30], [10, 10, 10], lower=1, bucket_interpolation="approx-uni")

    assert dates.frequency_below(date(2024, 1, 25)) == ints.frequency_below(25) == 25
    assert dates.frequency_below(date(2024, 1, 5)) == ints.frequency_below(5)


# -- frequency_above ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_frequency_above_counts_every_row_below_the_lower_bound(strategy: HistogramApproximation) -> None:
    assert equi_depth().frequency_above(-5, interpolation=strategy) == 30


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("value", [30, 31, 1000], ids=["at-upper", "just-above", "far-above"])
def test_frequency_above_is_zero_from_the_upper_bound_on(strategy: HistogramApproximation, value: int) -> None:
    assert equi_depth().frequency_above(value, interpolation=strategy) == 0


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(("value", "expected"), [(10, 20), (20, 10)], ids=["first-bound", "second-bound"])
def test_frequency_above_is_exact_on_bucket_bounds(strategy: HistogramApproximation, value: int, expected: int) -> None:
    """``>`` excludes the bucket that ends at `value`."""
    assert equi_depth().frequency_above(value, interpolation=strategy) == expected


@pytest.mark.parametrize(("value", "expected"), [(0, 20), (5, 20), (12, 10), (15, 10), (25, 0)])
def test_frequency_above_with_lower_bound_strategy_leaves_out_the_containing_bucket(value: int, expected: int) -> None:
    assert equi_depth().frequency_above(value, interpolation="bound-lower") == expected


@pytest.mark.parametrize(("value", "expected"), [(0, 30), (5, 30), (12, 20), (15, 20), (25, 10)])
def test_frequency_above_with_upper_bound_strategy_counts_the_containing_bucket(value: int, expected: int) -> None:
    assert equi_depth().frequency_above(value, interpolation="bound-upper") == expected


@pytest.mark.parametrize("value", [0, 5, 12, 15, 25, 29.5])
def test_frequency_above_with_uniform_approximation_interpolates_within_the_bucket(value: float) -> None:
    assert equi_depth().frequency_above(value, interpolation="approx-uni") == pytest.approx(30 - value)


@pytest.mark.parametrize(
    ("strategy", "expected"), [("bound-lower", 10), ("bound-upper", 20)], ids=["bound-lower", "bound-upper"]
)
def test_frequency_above_with_bound_strategies_works_for_non_numeric_values(
    strategy: HistogramApproximation, expected: int
) -> None:
    hist = Histogram(["f", "p", "z"], [10, 10, 10], lower="a")

    assert hist.frequency_above("p", interpolation=strategy) == 10
    assert hist.frequency_above("k", interpolation=strategy) == expected
    assert hist.frequency_above("zz", interpolation=strategy) == 0


# -- strategy contracts -------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", DATASET_PROBES)
def test_lower_bound_strategy_never_overestimates_the_true_frequency(value: float) -> None:
    assert DATASET_HISTOGRAM.frequency_below(value, interpolation="bound-lower") <= true_below(value)
    assert DATASET_HISTOGRAM.frequency_above(value, interpolation="bound-lower") <= true_above(value)


@pytest.mark.parametrize("value", DATASET_PROBES)
def test_upper_bound_strategy_never_underestimates_the_true_frequency(value: float) -> None:
    assert DATASET_HISTOGRAM.frequency_below(value, interpolation="bound-upper") >= true_below(value)
    assert DATASET_HISTOGRAM.frequency_above(value, interpolation="bound-upper") >= true_above(value)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("value", [2, 7, 9])
def test_every_strategy_is_exact_on_the_bucket_bounds_of_a_dataset(
    strategy: HistogramApproximation, value: int
) -> None:
    assert DATASET_HISTOGRAM.frequency_below(value, interpolation=strategy) == true_below(value)
    assert DATASET_HISTOGRAM.frequency_above(value, interpolation=strategy) == true_above(value)


@pytest.mark.parametrize("value", [-5, 0, 5, 10, 12, 15, 20, 25, 29.5, 30, 31])
def test_frequency_estimates_with_uniform_approximation_partition_the_rows(value: float) -> None:
    hist = equi_depth("approx-uni")

    assert hist.frequency_below(value) + hist.frequency_above(value) == pytest.approx(hist.n_rows)


@pytest.mark.parametrize("value", DATASET_PROBES)
def test_bound_strategies_are_each_others_complement(value: float) -> None:
    hist = DATASET_HISTOGRAM

    assert (
        hist.frequency_below(value, interpolation="bound-lower")
        + hist.frequency_above(value, interpolation="bound-upper")
        == hist.n_rows
    )
    assert (
        hist.frequency_below(value, interpolation="bound-upper")
        + hist.frequency_above(value, interpolation="bound-lower")
        == hist.n_rows
    )


# -- per-call interpolation override ------------------------------------------------------------------------


@pytest.mark.parametrize("default", STRATEGIES)
@pytest.mark.parametrize("override", STRATEGIES)
def test_interpolation_override_matches_a_histogram_with_that_default(
    default: HistogramApproximation, override: HistogramApproximation
) -> None:
    hist = equi_depth(default)
    reference = equi_depth(override)

    for value in [5, 12, 25]:
        assert hist.frequency_below(value, interpolation=override) == reference.frequency_below(value)
        assert hist.frequency_above(value, interpolation=override) == reference.frequency_above(value)


def test_interpolation_override_applies_to_a_single_estimate_only() -> None:
    hist = equi_depth("bound-lower")

    assert hist.frequency_below(15, interpolation="bound-upper") == 20
    assert hist.frequency_above(15, interpolation="approx-uni") == pytest.approx(15)

    assert hist.bucket_interpolation == "bound-lower"
    assert hist.frequency_below(15) == 10
    assert hist.frequency_above(15) == 10


# -- deprecated "bound" alias ------------------------------------------------------------------------------

DEPRECATION_MESSAGE = "'bound' interpolation strategy is deprecated and will be removed in version 0.23.0"


def test_deprecated_bound_strategy_is_stored_as_upper_bound_in_the_constructor() -> None:
    hist = Histogram([10, 20, 30], [10, 10, 10], lower=0, bucket_interpolation="bound")

    assert hist.bucket_interpolation == "bound-upper"
    assert hist == equi_depth("bound-upper")


def test_deprecated_bound_strategy_is_stored_as_upper_bound_by_the_setter() -> None:
    hist = equi_depth("approx-uni")
    hist.bucket_interpolation = "bound"

    assert hist.bucket_interpolation == "bound-upper"


@pytest.mark.parametrize("estimate", ["frequency_below", "frequency_above"])
@pytest.mark.parametrize("value", DATASET_PROBES)
def test_deprecated_bound_override_estimates_like_the_upper_bound(estimate: str, value: float) -> None:
    hist = Histogram([2, 7, 9], [8, 8, 4], lower=1, bucket_interpolation="bound-lower")

    deprecated = getattr(hist, estimate)(value, interpolation="bound")

    assert deprecated == getattr(hist, estimate)(value, interpolation="bound-upper")


def _deprecated_bound_via_constructor() -> None:
    Histogram([10], [5], lower=0, bucket_interpolation="bound")


def _deprecated_bound_via_setter() -> None:
    equi_depth().bucket_interpolation = "bound"


def _deprecated_bound_via_frequency_below() -> None:
    equi_depth().frequency_below(15, interpolation="bound")


def _deprecated_bound_via_frequency_above() -> None:
    equi_depth().frequency_above(15, interpolation="bound")


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_supported_strategies_do_not_warn(strategy: HistogramApproximation) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        hist = equi_depth(strategy)
        hist.bucket_interpolation = strategy
        hist.frequency_below(15, interpolation=strategy)
        hist.frequency_above(15, interpolation=strategy)


# -- equality and serialization -----------------------------------------------------------------------------


def test_histograms_with_the_same_buckets_are_equal() -> None:
    assert equi_depth("bound-upper") == equi_depth("bound-upper")


@pytest.mark.parametrize(
    "other",
    [
        Histogram([10, 20, 31], [10, 10, 10], lower=0),
        Histogram([10, 20, 30], [10, 10, 11], lower=0),
        Histogram([10, 20, 30], [10, 10, 10], lower=1),
        Histogram([10, 20, 30], [10, 10, 10], lower=0, bucket_interpolation="bound-lower"),
    ],
    ids=["bounds", "frequencies", "lower", "strategy"],
)
def test_histograms_differing_in_any_field_are_unequal(other: Histogram) -> None:
    assert equi_depth("bound-upper") != other


def test_histogram_json_contains_all_fields() -> None:
    hist = Histogram([10, 20], [3, 4], lower=1, bucket_interpolation="bound-lower")

    assert jsonize.to_json(hist) == (
        '{"bounds": [10, 20], "frequencies": [3, 4], "lower": 1, "n_rows": 7, "bucket_interpolation": "bound-lower"}'
    )


# -- regression tests --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "expected"), [(5, 5), (10, 10), (12, 12), (15, 15), (20, 20), (25, 25)])
def test_frequency_below_with_uniform_approximation_counts_all_complete_buckets(value: int, expected: int) -> None:
    """Regression guard for af3c87c (v0.22.2): `Histogram.frequency_below` added the in-bucket fraction to
    ``upper_idx - 1`` complete buckets instead of the ``upper_idx`` buckets that lie entirely below the value, so
    every estimate inside the histogram was one bucket short (``frequency_below(15)`` was 5). The first bucket was
    never interpolated at all (``frequency_below(5)`` was 0), since `lower` was not used by any estimate.
    """
    assert equi_depth("approx-uni").frequency_below(value) == pytest.approx(expected)


@pytest.mark.parametrize(("value", "expected"), [(10, 20), (12, 20), (20, 10), (25, 10), (30, 0)])
def test_frequency_above_with_bound_strategy_counts_the_buckets_above_the_value(value: int, expected: int) -> None:
    """Regression guard for af3c87c (v0.22.2): `Histogram.frequency_above` with the then "bound" strategy returned
    ``bisect_left(bounds, value) * freq_per_bucket``, the number of rows in the buckets *below* the value -- a copy of
    `frequency_below` that was never mirrored (``frequency_above(25)`` was 20). ``frequency_above(10)`` also counted
    the first bucket, although none of its rows is ``> 10``. The strategy is now called "bound-upper".
    """
    assert equi_depth("bound-upper").frequency_above(value) == expected


@pytest.mark.parametrize(("value", "expected"), [(21, 9), (25, 5), (30, 0)])
def test_frequency_above_with_uniform_approximation_handles_the_last_bucket(value: int, expected: int) -> None:
    """Regression guard for af3c87c (v0.22.2): for a value in the last bucket, `Histogram.frequency_above` read
    ``bounds[lower_idx + 1]`` past the end of the list and raised an `IndexError`.
    """
    assert equi_depth("approx-uni").frequency_above(value) == pytest.approx(expected)


def test_frequency_above_with_uniform_approximation_interpolates_in_the_bucket_containing_the_value() -> None:
    """Regression guard for af3c87c (v0.22.2): `Histogram.frequency_above` interpolated in the bucket *after* the one
    containing the value (with a negative fraction). With 3 buckets the errors cancelled out exactly at the midpoint,
    hence the 4 buckets here: 25 rows are ``> 15``, but the estimate was 15.
    """
    hist = Histogram([10, 20, 30, 40], [10, 10, 10, 10], lower=0, bucket_interpolation="approx-uni")

    assert hist.frequency_above(15) == pytest.approx(25)
    assert hist.frequency_above(12) == pytest.approx(28)


def test_frequency_estimates_use_the_bucket_frequencies() -> None:
    """Regression guard for af3c87c (v0.22.2): all estimates multiplied bucket counts with the average
    `freq_per_bucket` and ignored the stored per-bucket frequencies, so ``<= 10`` was estimated as 50 rows here
    instead of 90. Both producers create unequal buckets (`PostgresStatistics` merges the MCVs as single-value buckets,
    `PreciseStatistics` lets frequent values overflow their bucket).
    """
    hist = Histogram([10, 20], [90, 10], lower=0)

    assert hist.frequency_below(10, interpolation="bound-lower") == 90
    assert hist.frequency_below(10, interpolation="bound-upper") == 90
    assert hist.frequency_above(10, interpolation="bound-upper") == 10
    assert hist.frequency_below(15, interpolation="approx-uni") == pytest.approx(95)
    assert hist.frequency_above(5, interpolation="approx-uni") == pytest.approx(55)


def test_histogram_constructor_rejects_an_unknown_strategy() -> None:
    """Regression guard for af3c87c (v0.22.2): `Histogram.__init__` stored the strategy unchecked, so an invalid one
    only surfaced on the first in-bucket estimate as a bare ``assert`` (silently interpolating under ``python -O``).
    It now raises the same `ValueError` as the setter.
    """
    with pytest.raises(ValueError, match="Unsupported bucket interpolation strategy: linear"):
        Histogram([10, 20, 30], [10, 10, 10], lower=0, bucket_interpolation="linear")
