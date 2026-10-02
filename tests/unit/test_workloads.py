"""Tests for `postbound.workloads` -- the `Workload` container and the workload loaders.

Workloads are built in memory from parsed queries. The ready-made loaders (`job()`, `stats()`, ...) download their
queries on first use and are therefore not covered at tier 0.

Currently this module only holds regression tests.
"""

from __future__ import annotations

from postbound import parser
from postbound.qal import SqlQuery
from postbound.workloads import Workload


def parse(sql: str) -> SqlQuery:
    return parser.parse_query(sql, bind_columns=False)


OWN_QUERY = parse("SELECT * FROM r")
OTHER_QUERY = parse("SELECT * FROM s")
EXTRA_QUERY = parse("SELECT * FROM t")


# -- regression tests --------------------------------------------------------------------------------------


def test_union_operator_keeps_its_own_query_on_a_label_conflict() -> None:
    """Regression guard for 828cc52: when `Workload` became a read-only `Mapping`, `__or__` copied its own entries
    first and then applied ``update(other)``, so the right-hand operand overwrote conflicting labels. v0.21.6 (and the
    inline comment) retain the left-hand query.
    """
    own = Workload({"q1": OWN_QUERY})
    other = Workload({"q1": OTHER_QUERY, "q2": EXTRA_QUERY})

    merged = own | other

    assert dict(merged) == {"q1": OWN_QUERY, "q2": EXTRA_QUERY}
