"""Tests for `postbound._hints` -- the partial optimization decisions handed to the hint layer.

Query plans are built directly through the `QueryPlan` constructor: they are PostBOUND's own model and the functions
under test only read their estimates and measurements, so no EXPLAIN output is involved.

Currently this module only holds regression tests. The hint generation of the individual backends is covered in
`test_postgres_hints.py`.
"""

from __future__ import annotations

import pytest

from postbound._core import Cardinality, ScanOperator, TableReference
from postbound._hints import HintType, parameters_from_plan
from postbound._qep import QueryPlan

R = TableReference("r")

ESTIMATED = Cardinality(10)
ACTUAL = Cardinality(99)
SCAN_WITH_ESTIMATE_AND_MEASUREMENT = QueryPlan(
    "Seq Scan",
    operator=ScanOperator.SequentialScan,
    base_table=R,
    estimated_cardinality=ESTIMATED,
    actual_cardinality=ACTUAL,
)


# -- regression tests --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"target_cardinality": "estimates"}],
    ids=["default", "explicit"],
)
def test_parameters_from_plan_uses_the_estimates_when_asked_for_estimates(kwargs: dict) -> None:
    """Regression guard for cacf542: the `target_cardinality` literal was renamed from ``"estimated"`` to
    ``"estimates"`` (also the new default), but the body still compared against ``"estimated"``. Every request for
    estimates therefore fell through to the fallback branch, which prefers the actual cardinality whenever one exists.
    """
    params = parameters_from_plan(SCAN_WITH_ESTIMATE_AND_MEASUREMENT, **kwargs)

    assert params.cardinalities == {frozenset([R]): ESTIMATED}


def test_join_order_hint_type_has_a_clean_label() -> None:
    """Regression guard for 8ac3b52: merging the linear and bushy join order hint types introduced the value
    ``"Join orderˆˆ"`` with two stray characters, which leaked into `str()` output and serialized pre-check results.
    """
    assert HintType.JoinOrder.value == "Join order"
