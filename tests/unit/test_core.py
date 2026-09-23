"""Tests for `postbound._core.Cardinality` -- the NaN/inf-aware wrapper around row counts.

`Cardinality` is a pure value object (immutable, no I/O), so everything here is tier 0. The focus is the
numeric protocol (`Number` subclass): every arithmetic and comparison operator, combined with plain `int`/
`float` operands as well as other `Cardinality` instances, with particular attention to how the two invalid
states -- NaN ("unknown") and inf ("prohibitively large") -- propagate or short-circuit.

`tests/test_core.py` (legacy `unittest.TestCase`) already covers a handful of `Cardinality` smoke cases; this
module is the exhaustive pass and does not duplicate that file's tests. `TableReference` / `ColumnReference`
are out of scope here -- they are already covered there.

Writing this surfaced several real bugs (see the `# -- regression tests --` section at the bottom); per
`TESTING.md` §6.5 they are pinned, not fixed, and also recorded in `CHANGELOG.md`.
"""

from __future__ import annotations

import math

import pytest

from postbound import Cardinality
from postbound.util._errors import StateError
from postbound.util.jsonize import to_json

# -- fixtures -------------------------------------------------------------------------------------------

VALID = Cardinality(5)
ZERO = Cardinality(0)
NAN = Cardinality.unknown()
INF = Cardinality.infinite()


# -- construction and factories --------------------------------------------------------------------------


def test_of_wraps_plain_numbers() -> None:
    assert Cardinality.of(5) == Cardinality(5)
    assert Cardinality.of(5.0) == Cardinality(5)


def test_of_returns_the_same_instance_for_a_cardinality_input() -> None:
    assert Cardinality.of(VALID) is VALID


def test_of_none_returns_unknown() -> None:
    result = Cardinality.of(None)

    assert result.isnan()


def test_unknown_factory_produces_nan_state() -> None:
    assert Cardinality.unknown().isnan()
    assert not Cardinality.unknown().is_valid()


def test_infinite_factory_produces_inf_state() -> None:
    assert Cardinality.infinite().isinf()
    assert not Cardinality.infinite().is_valid()


def test_zero_factory_produces_a_valid_zero() -> None:
    zero = Cardinality.zero()

    assert zero.is_valid()
    assert zero.is_zero()
    assert zero.value == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(42.0, 42), (42.4, 42), (42.6, 43), (42.5, 42), (43.5, 44), (-0.4, 0)],
    ids=["exact", "round-down", "round-up", "half-to-even-down", "half-to-even-up", "negative-rounds-to-zero"],
)
def test_constructor_rounds_the_value_like_builtin_round(raw: float, expected: int) -> None:
    """`round()` uses banker's rounding (round-half-to-even), and the constructor inherits that."""
    assert Cardinality(raw).value == expected


def test_constructor_accepts_nan_and_inf_directly() -> None:
    assert Cardinality(math.nan).isnan()
    assert Cardinality(math.inf).isinf()


# -- state inspection -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("card", "expected_isnan", "expected_isinf", "expected_is_unknown", "expected_is_valid"),
    [
        (VALID, False, False, False, True),
        (ZERO, False, False, False, True),
        (NAN, True, False, True, False),
        (INF, False, True, False, False),
    ],
    ids=["valid", "zero", "nan", "inf"],
)
def test_state_predicates_agree_for_each_state(
    card: Cardinality, expected_isnan: bool, expected_isinf: bool, expected_is_unknown: bool, expected_is_valid: bool
) -> None:
    assert card.isnan() is expected_isnan
    assert card.isinf() is expected_isinf
    assert card.is_unknown() is expected_is_unknown
    assert card.is_valid() is expected_is_valid


@pytest.mark.parametrize("card", [NAN, INF], ids=["nan", "inf"])
def test_value_raises_for_invalid_states(card: Cardinality) -> None:
    with pytest.raises(StateError, match="Not a valid cardinality"):
        _ = card.value


def test_value_returns_the_rounded_int_for_a_valid_cardinality() -> None:
    assert VALID.value == 5
    assert isinstance(VALID.value, int)


@pytest.mark.parametrize(
    ("card", "expected"),
    [(VALID, 5.0), (ZERO, 0.0), (INF, math.inf)],
    ids=["valid", "zero", "inf"],
)
def test_raw_value_never_raises(card: Cardinality, expected: float) -> None:
    assert card.raw_value == expected


def test_raw_value_of_nan_is_nan() -> None:
    assert math.isnan(NAN.raw_value)


@pytest.mark.parametrize(
    ("card", "expected"),
    [(VALID, 5.0), (INF, math.inf)],
    ids=["valid", "inf"],
)
def test_get_matches_float_conversion(card: Cardinality, expected: float) -> None:
    assert card.get() == expected


def test_get_of_nan_is_nan() -> None:
    assert math.isnan(NAN.get())


@pytest.mark.parametrize(
    ("card", "expected"),
    [(ZERO, True), (Cardinality(0.4), True), (VALID, False), (NAN, False), (INF, False)],
    ids=["zero", "rounds-to-zero", "nonzero", "nan", "inf"],
)
def test_is_zero_is_only_true_for_a_valid_zero(card: Cardinality, expected: bool) -> None:
    assert card.is_zero() is expected


@pytest.mark.parametrize(
    ("card", "expected"),
    [(VALID, True), (ZERO, True), (NAN, False), (INF, False)],
    ids=["valid", "zero", "nan", "inf"],
)
def test_bool_reflects_validity_not_the_numeric_value(card: Cardinality, expected: bool) -> None:
    """`bool(Cardinality(0))` is `True`: `__bool__` tracks validity, not truthiness of the count."""
    assert bool(card) is expected


# -- arithmetic: addition and subtraction -----------------------------------------------------------------


def test_add_with_a_cardinality_operand() -> None:
    assert VALID + Cardinality(3) == Cardinality(8)


@pytest.mark.parametrize("other", [3, 3.0], ids=["int", "float"])
def test_add_with_plain_number_operands(other: int | float) -> None:
    assert VALID + other == Cardinality(8)


@pytest.mark.parametrize("other", [3, 3.0], ids=["int", "float"])
def test_radd_with_plain_number_operands(other: int | float) -> None:
    assert other + VALID == Cardinality(8)


@pytest.mark.parametrize(
    ("lhs", "rhs", "expected_isnan", "expected_isinf"),
    [
        (VALID, NAN, True, False),
        (NAN, VALID, True, False),
        (VALID, INF, False, True),
        (INF, VALID, False, True),
        (NAN, INF, True, False),
        (INF, NAN, True, False),
        (NAN, NAN, True, False),
        (INF, INF, False, True),
    ],
    ids=["valid+nan", "nan+valid", "valid+inf", "inf+valid", "nan+inf", "inf+nan", "nan+nan", "inf+inf"],
)
def test_add_propagates_invalid_states(
    lhs: Cardinality, rhs: Cardinality, expected_isnan: bool, expected_isinf: bool
) -> None:
    result = lhs + rhs

    assert result.isnan() is expected_isnan
    assert result.isinf() is expected_isinf


def test_sub_with_a_cardinality_operand() -> None:
    assert Cardinality(8) - Cardinality(3) == Cardinality(5)


@pytest.mark.parametrize("other", [3, 3.0], ids=["int", "float"])
def test_sub_with_plain_number_operands(other: int | float) -> None:
    assert Cardinality(8) - other == Cardinality(5)


@pytest.mark.parametrize("other", [3, 3.0], ids=["int", "float"])
def test_rsub_with_plain_number_operands(other: int | float) -> None:
    assert other - Cardinality(2) == Cardinality(1)


def test_sub_with_nan_or_inf_operand_is_unknown_or_infinite() -> None:
    assert (VALID - NAN).isnan()
    assert (VALID - INF).isinf()
    assert (INF - VALID).isinf()


def test_sub_of_infinite_from_infinite_is_unknown() -> None:
    """inf - inf is mathematically indeterminate; the wrapped float arithmetic produces NaN."""
    assert (INF - INF).isnan()


# -- arithmetic: multiplication and division --------------------------------------------------------------


def test_mul_with_a_cardinality_operand() -> None:
    assert Cardinality(4) * Cardinality(3) == Cardinality(12)


@pytest.mark.parametrize("other", [3, 3.0], ids=["int", "float"])
def test_mul_with_plain_number_operands(other: int | float) -> None:
    assert Cardinality(4) * other == Cardinality(12)


@pytest.mark.parametrize("other", [3, 3.0], ids=["int", "float"])
def test_rmul_with_plain_number_operands(other: int | float) -> None:
    assert other * Cardinality(4) == Cardinality(12)


def test_mul_propagates_nan_and_inf() -> None:
    assert (VALID * NAN).isnan()
    assert (VALID * INF).isinf()


def test_mul_of_zero_and_infinite_is_unknown() -> None:
    """0 * inf is mathematically indeterminate; the wrapped float arithmetic produces NaN."""
    assert (ZERO * INF).isnan()


def test_truediv_with_a_cardinality_operand() -> None:
    assert Cardinality(10) / Cardinality(2) == Cardinality(5)


@pytest.mark.parametrize("other", [2, 2.0], ids=["int", "float"])
def test_truediv_with_plain_number_operands(other: int | float) -> None:
    assert Cardinality(10) / other == Cardinality(5)


@pytest.mark.parametrize("other", [2, 2.0], ids=["int", "float"])
def test_rtruediv_with_plain_number_operands(other: int | float) -> None:
    assert other / Cardinality(2) == Cardinality(1)


def test_truediv_propagates_nan_and_inf() -> None:
    assert (VALID / NAN).isnan()
    assert (VALID / INF).is_zero()
    assert (INF / VALID).isinf()


@pytest.mark.parametrize(
    "compute",
    [lambda: VALID / ZERO, lambda: VALID / 0],
    ids=["cardinality-zero", "int-zero"],
)
def test_truediv_by_zero_raises(compute) -> None:
    with pytest.raises(ZeroDivisionError):
        compute()


# -- arithmetic: exponentiation ----------------------------------------------------------------------------


def test_pow_with_a_cardinality_operand() -> None:
    assert Cardinality(2) ** Cardinality(3) == Cardinality(8)


@pytest.mark.parametrize("other", [3, 3.0], ids=["int", "float"])
def test_pow_with_plain_number_operands(other: int | float) -> None:
    assert Cardinality(2) ** other == Cardinality(8)


@pytest.mark.parametrize("other", [2, 2.0], ids=["int", "float"])
def test_rpow_with_plain_number_operands(other: int | float) -> None:
    assert other ** Cardinality(3) == Cardinality(8)


def test_pow_propagates_nan() -> None:
    assert (VALID**NAN).isnan()
    assert (NAN**VALID).isnan()


def test_pow_of_infinite_base_with_zero_exponent_is_one() -> None:
    assert Cardinality(1) == INF**0


# -- unary no-ops: abs / trunc / ceil / floor / round --------------------------------------------------------


@pytest.mark.parametrize("card", [VALID, ZERO, NAN, INF], ids=["valid", "zero", "nan", "inf"])
def test_abs_returns_self_unchanged(card: Cardinality) -> None:
    """Cardinalities are always non-negative by contract, so `abs()` is documented as a no-op."""
    assert abs(card) is card


@pytest.mark.parametrize("card", [VALID, ZERO, NAN, INF], ids=["valid", "zero", "nan", "inf"])
def test_trunc_ceil_floor_return_self_unchanged(card: Cardinality) -> None:
    assert math.trunc(card) is card
    assert math.ceil(card) is card
    assert math.floor(card) is card


@pytest.mark.parametrize("card", [VALID, ZERO, NAN, INF], ids=["valid", "zero", "nan", "inf"])
def test_round_returns_self_unchanged_regardless_of_ndigits(card: Cardinality) -> None:
    assert round(card) is card
    assert round(card, 3) is card


# -- divmod / floordiv / mod --------------------------------------------------------------------------------


def test_divmod_with_a_cardinality_operand() -> None:
    assert divmod(Cardinality(7), Cardinality(2)) == (3, 1)


def test_divmod_with_a_plain_number_operand() -> None:
    assert divmod(Cardinality(7), 2) == (3, 1)


@pytest.mark.parametrize("card", [NAN, INF], ids=["nan", "inf"])
def test_divmod_when_self_is_invalid_returns_a_nan_pair(card: Cardinality) -> None:
    quotient, remainder = divmod(card, Cardinality(2))

    assert math.isnan(quotient)
    assert math.isnan(remainder)


def test_divmod_when_the_other_cardinality_is_nan_returns_a_nan_pair() -> None:
    quotient, remainder = divmod(VALID, NAN)

    assert math.isnan(quotient)
    assert math.isnan(remainder)


def test_divmod_when_the_other_cardinality_is_infinite_returns_zero_and_self() -> None:
    assert divmod(VALID, INF) == (0, VALID.value)


def test_rdivmod_with_a_valid_cardinality() -> None:
    assert divmod(10, Cardinality(3)) == (3.0, 1.0)


def test_rdivmod_when_self_is_nan_returns_a_nan_pair() -> None:
    quotient, remainder = divmod(10, NAN)

    assert math.isnan(quotient)
    assert math.isnan(remainder)


def test_rdivmod_when_self_is_infinite() -> None:
    assert divmod(10, INF) == (0.0, 10.0)


def test_mod_with_a_cardinality_operand() -> None:
    assert Cardinality(7) % Cardinality(2) == Cardinality(1)


@pytest.mark.parametrize("other", [2, 2.0], ids=["int", "float"])
def test_mod_with_plain_number_operands(other: int | float) -> None:
    assert Cardinality(7) % other == Cardinality(1)


@pytest.mark.parametrize("card", [NAN, INF], ids=["nan", "inf"])
def test_mod_when_self_is_invalid_returns_unknown(card: Cardinality) -> None:
    assert (card % Cardinality(2)).isnan()


def test_mod_when_the_other_cardinality_is_nan_returns_unknown() -> None:
    assert (VALID % NAN).isnan()


def test_mod_when_the_other_cardinality_is_infinite_returns_self() -> None:
    assert VALID % INF == VALID


@pytest.mark.parametrize("other", [math.nan, math.inf], ids=["nan", "inf"])
def test_mod_when_the_other_plain_number_is_nan_or_inf(other: float) -> None:
    result = VALID % other

    if math.isnan(other):
        assert result.isnan()
    else:
        assert result == VALID


def test_rmod_with_a_valid_cardinality() -> None:
    assert 10 % Cardinality(3) == Cardinality(1)


def test_rmod_when_self_is_nan_returns_unknown() -> None:
    assert (10 % NAN).isnan()


def test_rmod_when_self_is_infinite_returns_the_left_operand() -> None:
    assert Cardinality(10) == 10 % INF


def test_floordiv_with_plain_valid_operands() -> None:
    """Uses an evenly divisible pair to sidestep the rounding bug pinned below."""
    assert Cardinality(8) // Cardinality(2) == 4
    assert 8 // Cardinality(2) == 4
    assert Cardinality(8) // 2 == 4


def test_floordiv_when_the_other_cardinality_is_infinite_is_zero() -> None:
    assert VALID // INF == 0


# -- comparisons: lt / le / gt / ge, Cardinality operands -----------------------------------------------

# (self, other, lt, le, gt, ge) -- verified against the actual implementation; equality for mixed
# valid/nan/inf Cardinality pairs is covered separately below, since several of those combinations raise
# (see the regression test for that bug).
CARD_COMPARISON_CASES = [
    (VALID, VALID, False, True, False, True),
    (VALID, ZERO, False, False, True, True),
    (VALID, NAN, False, False, False, False),
    (VALID, INF, True, True, False, False),
    (ZERO, VALID, True, True, False, False),
    (ZERO, ZERO, False, True, False, True),
    (ZERO, NAN, False, False, False, False),
    (ZERO, INF, True, True, False, False),
    (NAN, VALID, False, False, False, False),
    (NAN, ZERO, False, False, False, False),
    (NAN, NAN, False, False, False, False),
    (NAN, INF, False, False, False, False),
    (INF, VALID, False, False, True, True),
    (INF, ZERO, False, False, True, True),
    (INF, NAN, False, False, True, True),
    (INF, INF, False, True, False, True),
]
CARD_COMPARISON_IDS = [
    "valid-valid",
    "valid-zero",
    "valid-nan",
    "valid-inf",
    "zero-valid",
    "zero-zero",
    "zero-nan",
    "zero-inf",
    "nan-valid",
    "nan-zero",
    "nan-nan",
    "nan-inf",
    "inf-valid",
    "inf-zero",
    "inf-nan",
    "inf-inf",
]


@pytest.mark.parametrize(("self_card", "other", "lt", "le", "gt", "ge"), CARD_COMPARISON_CASES, ids=CARD_COMPARISON_IDS)
def test_comparisons_against_a_cardinality_operand(
    self_card: Cardinality, other: Cardinality, lt: bool, le: bool, gt: bool, ge: bool
) -> None:
    """inf is documented as an unconditional upper bound: `inf >= x` is `True` even for `x` being NaN."""
    assert (self_card < other) is lt
    assert (self_card <= other) is le
    assert (self_card > other) is gt
    assert (self_card >= other) is ge


# -- comparisons: lt / le / gt / ge / eq, plain number operands ------------------------------------------

NUM_COMPARISON_CASES = [
    (VALID, 5, False, True, False, True, True),
    (VALID, 0, False, False, True, True, False),
    (VALID, math.nan, False, False, False, False, False),
    (VALID, math.inf, True, True, False, False, False),
    (ZERO, 5, True, True, False, False, False),
    (ZERO, 0, False, True, False, True, True),
    (ZERO, math.nan, False, False, False, False, False),
    (ZERO, math.inf, True, True, False, False, False),
    (NAN, 5, False, False, False, False, False),
    (NAN, math.nan, False, False, False, False, True),
    (NAN, math.inf, False, False, False, False, False),
    (INF, 5, False, False, True, True, False),
    (INF, math.nan, False, False, True, True, False),
    (INF, math.inf, False, True, False, True, True),
]
NUM_COMPARISON_IDS = [
    "valid-5",
    "valid-0",
    "valid-nan",
    "valid-inf",
    "zero-5",
    "zero-0",
    "zero-nan",
    "zero-inf",
    "nan-5",
    "nan-nan",
    "nan-inf",
    "inf-5",
    "inf-nan",
    "inf-inf",
]


@pytest.mark.parametrize(
    ("self_card", "other", "lt", "le", "gt", "ge", "eq"), NUM_COMPARISON_CASES, ids=NUM_COMPARISON_IDS
)
def test_comparisons_against_a_plain_number_operand(
    self_card: Cardinality, other: float, lt: bool, le: bool, gt: bool, ge: bool, eq: bool
) -> None:
    """Unlike the Cardinality-Cardinality case, comparing against a raw NaN/inf float never raises."""
    assert (self_card < other) is lt
    assert (self_card <= other) is le
    assert (self_card > other) is gt
    assert (self_card >= other) is ge
    assert (self_card == other) is eq


def test_lt_against_an_unrelated_type_raises_type_error() -> None:
    with pytest.raises(TypeError):
        _ = VALID < "5"  # type: ignore - the point of this test is that this raises at runtime


# -- conversions: float / int / complex ------------------------------------------------------------------


def test_float_conversion_for_every_state() -> None:
    assert float(VALID) == 5.0
    assert math.isnan(float(NAN))
    assert math.isinf(float(INF))


def test_int_conversion_for_a_valid_cardinality() -> None:
    assert int(VALID) == 5


@pytest.mark.parametrize("card", [NAN, INF], ids=["nan", "inf"])
def test_int_conversion_raises_for_invalid_states(card: Cardinality) -> None:
    with pytest.raises(StateError, match="Not a valid cardinality"):
        int(card)


def test_complex_conversion_for_every_state() -> None:
    assert complex(VALID) == complex(5)
    assert complex(INF) == complex(math.inf)

    nan_complex = complex(NAN)
    assert math.isnan(nan_complex.real)
    assert nan_complex.imag == 0


# -- equality and hashing ---------------------------------------------------------------------------------


def test_eq_reflexive_for_every_state() -> None:
    assert VALID == VALID
    assert NAN == NAN
    assert INF == INF


def test_eq_between_equal_valid_cardinalities() -> None:
    assert Cardinality(5) == Cardinality(5)


def test_eq_against_equal_plain_numbers() -> None:
    assert VALID == 5
    assert VALID == 5.0


def test_eq_against_unequal_plain_numbers() -> None:
    assert VALID != 6
    assert VALID != 6.0


def test_nan_equals_a_raw_nan_float() -> None:
    assert math.nan == NAN


def test_infinite_equals_a_raw_inf_float() -> None:
    assert math.inf == INF


def test_hash_is_stable_and_consistent_with_eq_for_cardinality_operands() -> None:
    a, b = Cardinality(5), Cardinality(5)

    assert a == b
    assert hash(a) == hash(b)
    assert hash(a) == hash(a)


def test_hash_differs_for_cardinalities_with_different_states() -> None:
    assert hash(NAN) != hash(INF)
    assert hash(VALID) != hash(Cardinality(6))


def test_hash_differs_between_nan_and_a_valid_zero() -> None:
    """`_value` alone is not enough to distinguish states: NaN/inf both store `-1` internally."""
    assert hash(NAN) != hash(ZERO)


# -- serialization ------------------------------------------------------------------------------------------


def test_json_dunder_returns_a_plain_float() -> None:
    assert VALID.__json__() == 5.0
    assert math.isnan(NAN.__json__())
    assert math.isinf(INF.__json__())


def test_to_json_round_trips_through_a_plain_float() -> None:
    assert to_json(VALID) == "5.0"


# -- repr / str ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("card", "expected"),
    [(VALID, "5"), (NAN, "NaN"), (INF, "inf")],
    ids=["valid", "nan", "inf"],
)
def test_str_and_repr_for_every_state(card: Cardinality, expected: str) -> None:
    assert str(card) == expected
    assert repr(card) == expected


# -- pattern matching ---------------------------------------------------------------------------------------


def test_match_binds_the_single_positional_pattern_to_raw_value() -> None:
    match VALID:
        case Cardinality(bound):
            assert bound == 5.0
        case _:
            pytest.fail("match should have bound the Cardinality pattern")


# -- regression tests ----------------------------------------------------------------------------------------


def test_neg_returns_the_notimplemented_sentinel_instead_of_raising() -> None:
    """Documents a real bug, not the intended behavior.

    `__neg__` unconditionally `return NotImplemented`. For binary operators that sentinel triggers the
    reflected-method protocol, but `__neg__` is unary -- there is no other operand to hand it to, so Python
    does not intercept it. `-VALID` therefore silently evaluates to the `NotImplemented` singleton itself
    rather than raising `TypeError` (the usual outcome for an unsupported unary operator) or returning a
    sensible value. The comment on `__neg__` ("What's a negative cardinality supposed to be?") suggests
    raising `TypeError` was the intent.
    """
    result = -VALID

    assert result is NotImplemented


def test_subtraction_can_silently_produce_an_invalid_negative_cardinality() -> None:
    """Documents a real bug, not the intended behavior.

    The class docstring states "a valid cardinality can be any non-negative integer", but the constructor
    never enforces that: `__sub__` just wraps `self.get() - other.get()` in `Cardinality(...)` with no bounds
    check, so subtracting a larger cardinality from a smaller one produces a Cardinality that reports
    `is_valid() is True` and a negative `.value`.
    """
    result = Cardinality(3) - Cardinality(5)

    assert result.is_valid() is True
    assert result.value == -2


@pytest.mark.parametrize(
    ("lhs", "rhs"),
    [
        (VALID, NAN),
        (NAN, VALID),
        (VALID, INF),
        (INF, VALID),
        (NAN, INF),
        (INF, NAN),
    ],
    ids=["valid-nan", "nan-valid", "valid-inf", "inf-valid", "nan-inf", "inf-nan"],
)
def test_eq_between_differently_invalid_cardinalities_raises_instead_of_returning_false(
    lhs: Cardinality, rhs: Cardinality
) -> None:
    """Documents a real bug, not the intended behavior.

    `__eq__`'s `Cardinality` branch only special-cases the case where *both* operands share the same invalid
    state (both NaN, or both inf); every other combination -- including one valid operand paired with one
    invalid one -- falls through to `self.value == other.value`, which raises `StateError` because `.value`
    is undefined for NaN/inf. Comparing against a raw `float('nan')`/`float('inf')` instead of a `Cardinality`
    does not hit this path and correctly returns `False` (see
    `test_comparisons_against_a_plain_number_operand`), so the crash is specific to comparing two
    `Cardinality` instances.
    """
    with pytest.raises(StateError, match="Not a valid cardinality"):
        _ = lhs == rhs


def test_hash_is_inconsistent_with_equality_against_a_plain_int() -> None:
    """Documents a real bug, not the intended behavior.

    `Cardinality(5) == 5` is `True` (see `test_eq_against_equal_plain_numbers`), but `__hash__` hashes the
    tuple `(self._nan, self._inf, self._value)` rather than delegating to `hash(float(self))`. This violates
    the invariant that equal objects must hash equal, and would silently break lookups in a `dict`/`set` that
    mixes `Cardinality` and plain `int`/`float` keys of the same numeric value.
    """
    assert VALID == 5
    assert hash(VALID) != hash(5)


def test_floordiv_with_a_nan_operand_raises_instead_of_propagating_unknown() -> None:
    """Documents a real bug, not the intended behavior.

    `__mod__`/`__divmod__` explicitly special-case NaN/inf operands and propagate them gracefully (see
    `test_mod_when_self_is_invalid_returns_unknown` and friends). `__floordiv__`/`__rfloordiv__` have no such
    special-casing: they just do `math.floor(float(self / other))`, and `math.floor(nan)` raises `ValueError`
    instead of returning an "unknown" result.
    """
    with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
        _ = VALID // NAN

    with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
        _ = NAN // VALID

    with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
        _ = 10 // NAN


def test_floordiv_rounds_to_the_nearest_int_before_flooring_instead_of_truncating() -> None:
    """Documents a real bug, not the intended behavior.

    `__floordiv__` is `math.floor(float(self / other))`, and `self / other` (`__truediv__`) already rounds
    its quotient to the nearest int via the `Cardinality(...)` constructor before `__floordiv__` ever floors
    it. So `11 // 4` -- true floor division is `2` -- first computes `11 / 4 = 2.75`, rounds that to `3`
    inside the intermediate `Cardinality`, and only then floors, landing on `3`.
    """
    result = Cardinality(11) // Cardinality(4)

    assert result == 3
    assert result != 11 // 4


def test_floordiv_with_an_infinite_dividend_raises_instead_of_propagating_infinite() -> None:
    """Documents a real bug, not the intended behavior.

    Same root cause as `test_floordiv_with_a_nan_operand_raises_instead_of_propagating_unknown`:
    `math.floor(math.inf)` raises `OverflowError` rather than `__floordiv__` returning something like `inf`.
    """
    with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
        _ = INF // VALID


def test_match_pattern_does_not_match_the_documented_is_valid_value_shape() -> None:
    """Documents a real bug, not the intended behavior.

    The class docstring says cardinalities "match the following pattern: *(is_valid, value)*", implying a
    two-element positional pattern. In reality `__match_args__ = ("raw_value",)` is a single-element tuple,
    so a `case Cardinality(is_valid, value):` as documented raises `TypeError` at match time instead of
    matching -- see `test_match_binds_the_single_positional_pattern_to_raw_value` for what the pattern
    actually binds.
    """
    with pytest.raises(TypeError, match=r"accepts 1 positional sub-pattern \(2 given\)"):
        match VALID:
            case Cardinality(_is_valid, _value):  # type: ignore - the point of this test is that this raises at runtime
                pass
