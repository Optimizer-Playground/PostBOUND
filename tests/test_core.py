"""Tests for PostBOUND's core data structures."""

from __future__ import annotations

import copy
import math
import pickle
import unittest

import postbound as pb
from postbound._core import VirtualTableError, normalize, quote
from postbound.util._errors import StateError


class RegressionTests(unittest.TestCase):
    def test_column_copy(self) -> None:
        posts = pb.TableReference("posts")
        score = pb.ColumnReference("score", posts)

        cpy = copy.copy(score)
        self.assertEqual(score, cpy)

        deep_cpy = copy.deepcopy(score)
        self.assertEqual(score, deep_cpy)

    def test_column_pickle(self) -> None:
        posts = pb.TableReference("posts")
        score = pb.ColumnReference("score", posts)

        serialized = pickle.dumps(score)
        reloaded = pickle.loads(serialized)
        self.assertEqual(score, reloaded)
        self.assertIs(type(reloaded), pb.BoundColumnReference)

        unbound = pb.ColumnReference("id")
        serialized = pickle.dumps(unbound)
        reloaded = pickle.loads(serialized)
        self.assertEqual(unbound, reloaded)
        self.assertIs(type(reloaded), pb.ColumnReference)


class CardinalityTests(unittest.TestCase):
    def test_value_access_and_casting(self) -> None:
        card = pb.Cardinality.of(42.7)

        self.assertTrue(card.is_valid())
        self.assertEqual(card.value, 43)
        self.assertEqual(int(card), 43)
        self.assertAlmostEqual(float(card), 43.0)
        self.assertTrue(card)

    def test_invalid_states_raise_state_error(self) -> None:
        unknown = pb.Cardinality.unknown()
        infinite = pb.Cardinality.infinite()

        self.assertFalse(unknown)
        self.assertFalse(infinite)
        with self.assertRaises(StateError):
            _ = unknown.value
        with self.assertRaises(StateError):
            _ = infinite.value
        self.assertTrue(math.isnan(unknown.get()))
        self.assertTrue(math.isinf(infinite.get()))

    def test_unknown_propagates_through_arithmetic(self) -> None:
        result = pb.Cardinality.unknown() + pb.Cardinality.of(5)

        self.assertTrue(result.isnan())
        self.assertEqual(result, float("nan"))

    def test_modulo_with_infinite_operand(self) -> None:
        card = pb.Cardinality.of(10)

        self.assertEqual(card % math.inf, card)
        self.assertEqual(math.inf % card, pb.Cardinality(math.inf % card.value))

    def test_divmod_with_nan_cardinality(self) -> None:
        quotient, remainder = divmod(pb.Cardinality.unknown(), pb.Cardinality.of(5))

        self.assertTrue(math.isnan(quotient))
        self.assertTrue(math.isnan(remainder))

    def test_comparison_with_special_floats(self) -> None:
        card = pb.Cardinality.of(5)

        self.assertTrue(card < float("inf"))
        self.assertFalse(card > float("inf"))
        self.assertTrue(pb.Cardinality.unknown() == float("nan"))


class TableReferenceTests(unittest.TestCase):
    def test_requires_identifier(self) -> None:
        with self.assertRaises(ValueError):
            pb.TableReference("")

    def test_schema_requires_full_name(self) -> None:
        with self.assertRaises(ValueError):
            pb.TableReference("", schema="public")

    def test_identifier_prefers_alias_and_str_output(self) -> None:
        table = pb.TableReference("posts", alias="p")

        self.assertEqual(table.identifier(), "p")
        self.assertEqual(str(table), "posts AS p")

    def test_qualified_name_on_virtual_table_raises(self) -> None:
        virtual_table = pb.TableReference.create_virtual("subq")

        with self.assertRaises(VirtualTableError):
            virtual_table.qualified_name()

    def test_drop_alias_on_virtual_table_raises(self) -> None:
        virtual_table = pb.TableReference.create_virtual("subq")

        with self.assertRaises(StateError):
            virtual_table.drop_alias()

    def test_drop_alias_returns_physical_table(self) -> None:
        table = pb.TableReference("posts", alias="p")
        without_alias = table.drop_alias()

        self.assertEqual(without_alias.alias, "")
        self.assertEqual(without_alias.full_name, "posts")

    def test_case_insensitive_equality_and_sorting(self) -> None:
        lhs = pb.TableReference("Posts", alias="T", schema="Public")
        rhs = pb.TableReference("posts", alias="t", schema="public")

        self.assertEqual(lhs, rhs)
        other = pb.TableReference("accounts", alias="acc")
        unsorted = [rhs, other]
        sorted_refs = sorted(unsorted)
        self.assertEqual(sorted_refs[0].identifier(), "acc")

    def test_with_alias_requires_physical_table(self) -> None:
        alias_only = pb.TableReference("", alias="tmp", virtual=True)

        with self.assertRaises(StateError):
            alias_only.with_alias("new_tmp")

    def test_make_virtual_preserves_alias_and_schema(self) -> None:
        table = pb.TableReference("posts", alias="p", schema="public")
        virtual_copy = table.make_virtual()

        self.assertTrue(virtual_copy.virtual)
        self.assertEqual(virtual_copy.alias, "p")
        self.assertEqual(virtual_copy.schema, "public")

    def test_update_creates_new_reference_with_changes(self) -> None:
        table = pb.TableReference("posts", alias="p")
        updated = table.update(alias="posts_alias", virtual=True)

        self.assertEqual(updated.alias, "posts_alias")
        self.assertTrue(updated.virtual)


class ColumnReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = pb.TableReference("posts", alias="p")

    def test_requires_name(self) -> None:
        with self.assertRaises(ValueError):
            pb.ColumnReference("")

    def test_bound_instances_are_created_automatically(self) -> None:
        col = pb.ColumnReference("score", self.table)

        self.assertIsInstance(col, pb.BoundColumnReference)
        self.assertTrue(col.is_bound())
        self.assertEqual(col.table, self.table)
        self.assertTrue(pb.ColumnReference.assert_bound(col))

    def test_assert_bound_returns_false_for_unbound_columns(self) -> None:
        col = pb.ColumnReference("score")

        self.assertFalse(col.is_bound())
        self.assertFalse(pb.ColumnReference.assert_bound(col))

    def test_bind_to_creates_new_bound_reference(self) -> None:
        col = pb.ColumnReference("score")
        other_table = pb.TableReference("archived_posts")
        bound = col.bind_to(other_table)

        self.assertIsInstance(bound, pb.BoundColumnReference)
        self.assertEqual(bound.table, other_table)
        self.assertIsNot(bound, col)

    def test_as_unbound_removes_binding(self) -> None:
        bound = pb.ColumnReference("score", self.table)
        unbound = bound.as_unbound()

        self.assertIsInstance(unbound, pb.ColumnReference)
        self.assertFalse(unbound.is_bound())

    def test_drop_table_alias_normalizes_reference(self) -> None:
        col = pb.ColumnReference("score", self.table)
        normalized = col.drop_table_alias()

        self.assertIsNot(normalized.table, None)
        self.assertEqual(normalized.table.alias, "")
        self.assertEqual(normalized.table.full_name, "posts")

    def test_drop_table_alias_on_virtual_table_raises(self) -> None:
        virtual_table = pb.TableReference.create_virtual("subq")
        col = pb.ColumnReference("projected", virtual_table)

        with self.assertRaises(StateError):
            col.drop_table_alias()

    def test_ordering_prefers_unbound_columns(self) -> None:
        unbound = pb.ColumnReference("alpha")
        bound = pb.ColumnReference("beta", self.table)

        self.assertLess(unbound, bound)

    def test_json_representation_contains_column_and_table(self) -> None:
        col = pb.ColumnReference("score", self.table)

        self.assertEqual(col.__json__(), {"column": "score", "table": self.table})


class IdentifierUtilitiesTests(unittest.TestCase):
    def test_quote_returns_identifier_for_valid_name(self) -> None:
        self.assertEqual(quote("movie_id"), "movie_id")

    def test_quote_wraps_keywords_and_uppercase(self) -> None:
        self.assertEqual(quote("select"), '"select"')
        self.assertEqual(quote("Movie"), '"Movie"')

    def test_quote_handles_empty_and_special_characters(self) -> None:
        self.assertEqual(quote(""), "")
        self.assertEqual(quote("movie title"), '"movie title"')

    def test_normalize_lowercases_identifiers(self) -> None:
        self.assertEqual(normalize("MovieID"), "movieid")
        self.assertEqual(normalize(""), "")
