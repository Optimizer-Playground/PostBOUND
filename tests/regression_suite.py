from __future__ import annotations

import abc
import os
import unittest
from collections.abc import Callable
from typing import TypeVar

import pytest

from postbound import QueryPlan

ResultSet = list[tuple[object]]

_T = TypeVar("_T")


def _rebuild_result_set(result_set: ResultSet) -> ResultSet:
    """Transforms a possibly simplified result set as returned by the Database interface back to a normalized one."""
    result_set = _rebuild_result_set_from_cache(result_set)

    # case 1: result set of a single row -- ('foo', 42) becomes [('foo', 42)]
    if isinstance(result_set, tuple):
        return [result_set]

    # case 2: result set of a single row of a single value -- 42 becomes [(42,)]
    if not isinstance(result_set, list):
        return [(result_set,)]

    first_row = result_set[0]
    # case 3: result set of multiple rows of multiple values is left as-is
    if isinstance(first_row, tuple):
        return result_set

    # case 4: result set of multiple rows of single values -- ['foo', 'bar'] becomes [('foo',), ('bar',)]
    return [(value,) for value in result_set]


def _rebuild_result_set_from_cache(
    result_set: ResultSet,
) -> ResultSet:
    """Transforms a result set as stored in the database cache into the same format as provided by a cursor.

    If the given result set does not need such adaptions, none will be performed.
    """
    if not isinstance(result_set, list) or not len(result_set) or not isinstance(result_set[0], list):
        # result sets stored in a cached are a list of nested lists of values
        # each entry in the outer list corresponds to a row and each value in the nested lists corresponds to a
        # column value
        return result_set
    return [tuple(row) for row in result_set]


def _stringify_result_set(result_set: ResultSet) -> str:
    """Transforms a (possibly simplified) result set into a string representation.

    Since result sets can become quite large, this cuts the result set to only contain the first 5 values if
    necessary.
    """
    if "__len__" not in dir(result_set):
        return str(result_set)
    if len(result_set) > 5:
        shortened_result_set = result_set[:5]
        return f"result set ({len(result_set)} elements total) :: first 5 = {shortened_result_set}"
    return f"result set ({len(result_set)} elements) :: contents = {result_set}"


def _assert_default_result_sets_equal(first_set: ResultSet, second_set: ResultSet) -> None:
    """Compares two unordered result sets and makes sure they contain exactly the same rows."""
    first_set_set = set(first_set)
    second_set_set = set(second_set)
    if first_set_set != second_set_set:
        first_set_str = _stringify_result_set(first_set)
        second_set_str = _stringify_result_set(second_set)
        raise AssertionError(f"Result sets differ: {first_set_str} vs. {second_set_str}")


def _assert_ordered_result_sets_equal(first_set: ResultSet, second_set: ResultSet) -> None:
    """Compares two ordered results sets and makes sure they contain exactly the same rows in exactly the same order."""
    for cursor, row in enumerate(first_set):
        comparison_row = second_set[cursor]
        if row != comparison_row:
            first_set_str = _stringify_result_set(first_set)
            second_set_str = _stringify_result_set(second_set)
            raise AssertionError(f"Result sets differ: {first_set_str} vs {second_set_str}")


class DatabaseTestCase(unittest.TestCase, abc.ABC):
    """Abstract test case that provides assertions on result sets of actually executed database queries."""

    def assertResultSetsEqual(self, first_set: ResultSet, second_set: ResultSet, *, ordered: bool = False) -> None:
        """Assertion that fails if the two result sets differ.

        Ordering can be accounted for by the `ordered` argument. By default, result sets are assumed to be unordered.
        """
        if type(first_set) is not type(second_set):
            error_msg = "Result sets have different types: "
            first_set_str = _stringify_result_set(first_set)
            second_set_str = _stringify_result_set(second_set)
            error_msg += f"{first_set_str} ({type(first_set)}) and {second_set_str} ({type(second_set)})"
            raise AssertionError(error_msg)

        first_set = _rebuild_result_set(first_set)
        second_set = _rebuild_result_set(second_set)
        if len(first_set) != len(second_set):
            first_set_str = _stringify_result_set(first_set)
            second_set_str = _stringify_result_set(second_set)
            raise AssertionError(f"Result sets have different length: {first_set_str} and {second_set_str}")

        if ordered:
            _assert_ordered_result_sets_equal(first_set, second_set)
        else:
            _assert_default_result_sets_equal(first_set, second_set)


class QueryTestCase(unittest.TestCase, abc.ABC):
    """Abstract test case that provides assertions on the structure of database queries."""

    def assertQueriesEqual(self, first_query: object, second_query: object, message: str = "") -> None:
        """Assertion that fails if the two queries differ in a _significant_ way.

        This method is heavily heuristic and compares the two query strings according to the following rules:

        - leading/trailing whitespace is ignored
        - a trailing semicolon is ignored
        - upper/lowercase is ignored throughout the query

        Each other difference results in failure of the assertion. This includes optional parentheses as well as
        insignificant whitespace or comments within the query.
        """
        first_query = str(first_query).strip().removesuffix(";").lower()
        second_query = str(second_query).strip().removesuffix(";").lower()
        return self.assertEqual(first_query, second_query, message)


class PlanTestCase(unittest.TestCase, abc.ABC):
    def assertQueryExecutionPlansEqual(
        self,
        first_plan: QueryPlan,
        second_plan: QueryPlan,
        message: str = "",
        *,
        _cur_level: int = 0,
    ) -> None:
        if first_plan.node_type != second_plan.node_type:
            default_msg = f"Different operators at level {_cur_level}: {first_plan} vs. {second_plan}"
            raise AssertionError(default_msg if not message else f"{message} :: {default_msg}")
        elif len(first_plan.children) != len(second_plan.children):
            default_msg = f"Different number of child nodes at level {_cur_level}: {first_plan} vs. {second_plan}"
            raise AssertionError(default_msg if not message else f"{message} :: {default_msg}")

        for left_child, right_child in zip(first_plan.children, second_plan.children, strict=False):
            self.assertQueryExecutionPlansEqual(left_child, right_child, message=message, _cur_level=_cur_level + 1)


def skip_if_no_db(config_file: str | os.PathLike) -> Callable[[_T], _T]:
    """Decorator that skips a test class or method when its database cannot be reached.

    The connectivity probe is **deferred**: this decorator only attaches a marker recording which connection
    file the test needs, and ``tests/conftest.py`` resolves availability at collection time (memoized per
    file). That matters for two reasons:

    - An offline run never opens a connection. The previous implementation connected eagerly at decoration
      time, i.e. during module import, so merely *collecting* the suite opened one throwaway connection per
      decorated class even when every test in it was about to be deselected.
    - A missing or malformed connection file no longer aborts collection of the surrounding module. The
      earlier version caught only `psycopg.OperationalError`, so an authentication failure or an unparseable
      config file propagated out of the import.

    Relative paths are resolved against the repository root rather than the working directory.

    Parameters
    ----------
    config_file : str | os.PathLike
        The config file describing the connection, in any format accepted by `postgres.connect`.

    Returns
    -------
    Callable
        A decorator attaching the deferred-skip marker.
    """
    return pytest.mark.requires_db(str(config_file))
