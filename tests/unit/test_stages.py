"""Tests for `postbound._stages` -- the user-facing optimization stage interfaces.

The stages are abstract; the tests implement the single abstract hook of each stage with a tiny in-test subclass and
exercise the default template logic around it. Queries come from the real parser and need no database, since the
default `CardinalityEstimator` intermediates are derived purely from the query's join graph.

Currently this module only holds regression tests. The pipelines that drive the stages are not covered here.
"""

from __future__ import annotations

from collections.abc import Iterable

from postbound import parser
from postbound._core import Cardinality, TableReference
from postbound._stages import CardinalityEstimator
from postbound.qal import SqlQuery

R = TableReference("r")
S = TableReference("s")


def parse(sql: str) -> SqlQuery:
    return parser.parse_query(sql, bind_columns=False)


class _CannedEstimator(CardinalityEstimator):
    """Answers from a fixed `{intermediate: cardinality}` table and reports unknown for anything else."""

    def __init__(self, estimates: dict[frozenset[TableReference], Cardinality]) -> None:
        super().__init__()
        self._estimates = estimates

    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        key = frozenset([intermediate]) if isinstance(intermediate, TableReference) else frozenset(intermediate)
        return self._estimates.get(key, Cardinality.unknown())


# -- regression tests --------------------------------------------------------------------------------------


def test_estimate_cardinalities_keeps_valid_estimates_and_drops_unknown_ones() -> None:
    """Regression guard for 73dcbf4: replacing ``not math.isnan(estimate)`` with ``estimate.is_valid()`` lost the
    negation the wrong way round (``if not estimate.is_valid()``), so `estimate_cardinalities` stored only the
    *unknown* estimates and discarded every valid one.
    """
    query = parse("SELECT * FROM r, s WHERE r.a = s.b")
    estimator = _CannedEstimator({frozenset([R]): Cardinality(10), frozenset([R, S]): Cardinality(42)})

    parameterization = estimator.estimate_cardinalities(query)

    assert parameterization.cardinalities == {
        frozenset([R]): Cardinality(10),
        frozenset([R, S]): Cardinality(42),
    }
