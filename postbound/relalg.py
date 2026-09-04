"""relalg provides fundamental building blocks of relational algebra and a converter from SQL to algebra.

The central component of our algebra implementation is the `RelNode` class. All relational operators inherit from this abstract
class. Following the design of the expression and predicate models, all algebraic trees are immutable data structures. Once a
tree has been generated, it can no longer be modified.

One important aspect of our relational algebra design is how to model arbitrary expressions and projections, mappings, etc. on
these expressions. Some systems introduce temporary variables for mapping targets, e.g. ``arg0 <- R.a + 42`` and then base all
further accesses on these temporary variables. In this school of thought, an algebra tree for the query
``SELECT R.a + 42 FROM R`` would like this:

.. math:: \\pi_{arg_0}(\\chi_{arg_0 \\leftarrow R.a + 42}(R))

This representation is especially usefull for a code-generation or physical optimization scenario because it enables a
straightforward creation of additional (temporary) columns. At the same time, it makes the translation of SQL queries to
relational algebra more challenging, since re-writes have to be applied during parsing. Since we are not concerned with
code-generation in our algebra representation and focus more on structural properties, we take a different approach: all
expressions (as defined in the `expressions` module) are contained as-is in the operators. However, we make sure that necessary
pre-processing actions are included as required. For example, if a complex expression is included in a predicate or a
projection, we generate the appropriate mapping operation beforehand and use it as an input for the consuming operator.

In addition to the conventional operators of relational algebra, we introduce a couple of additional operators that either
mimic features from SQL, or that make working with the algebra much easier from a technical point-of-view. The first category
includes operators such as `Sort` or `DuplicateElimination` and `Limit`, whereas the second category includes the
`SubqueryScan`.

Notice that while most algebraic expressions correspond to tree structures, there might be cases where a directed, acyclic
graph is generated. This is especially the case when a base relation is used as part of subqueries. Nevertheless, there will
always be only one root (sink) node.
"""

from __future__ import annotations

import abc
import collections
import copy
import dataclasses
import enum
import functools
import operator
import typing
from collections.abc import Generator, Iterable, Mapping, Sequence

from . import transform, util
from ._core import ColumnReference, TableReference
from .qal import (
    AbstractPredicate,
    AndPredicate,
    ArrayExpression,
    BetweenPredicate,
    BinaryPredicate,
    CaseExpression,
    CastExpression,
    ColumnExpression,
    CompoundOperator,
    CompoundPredicate,
    DirectTableSource,
    ExpressionCollector,
    FunctionExpression,
    InPredicate,
    JoinTableSource,
    MathExpression,
    NotPredicate,
    OrderByExpression,
    OrPredicate,
    PredicateVisitor,
    QuantifierExpression,
    SelectStatement,
    SetOperator,
    SetQuery,
    SqlExpression,
    SqlExpressionVisitor,
    SqlQuery,
    StarExpression,
    StaticValueExpression,
    SubqueryExpression,
    SubqueryTableSource,
    TableSource,
    UnaryOperator,
    UnaryPredicate,
    WindowExpression,
)

# TODO: the creation and mutation of different relnodes should be handled by a dedicated factory class. This solves all issues
# with mutatbility/immutability and automated linking to parent nodes.


class RelNode(abc.ABC):
    """Models a fundamental operator in relation algebra. All specific operators like selection or theta join inherit from it.

    Parameters
    ----------
    parent_node : Optional[RelNode]
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.

    See Also
    --------
    parse_relalg
    """

    def __init__(self, parent_node: RelNode | None) -> None:
        self._parent = parent_node
        self._sideways_pass: set[RelNode] = set()
        self._node_type = type(self).__name__
        self._hash_val = hash((self._node_type, self._recalc_hash_val()))
        self._maintain_child_links()

    @property
    def node_type(self) -> str:
        """Get the current operator as a string.

        Returns
        -------
        str
            The operator name
        """
        return self._node_type

    @property
    def parent_node(self) -> RelNode | None:
        """Get the parent node of the current operator, if it exists.

        Returns
        -------
        Optional[RelNode]
            The parent is the operator that receives the output relation of the current operator. If the current operator is
            the root and (currently) does not have a parent, *None* is returned.
        """
        return self._parent

    @property
    def sideways_pass(self) -> frozenset[RelNode]:
        """Get all nodes that receive the output of the current operator in addition to the parent node.

        Returns
        -------
        frozenset[RelNode]
            The sideways pass nodes
        """
        return frozenset(self._sideways_pass)

    def root(self) -> RelNode:
        """Traverses the algebra tree upwards until the root node is found.

        Returns
        -------
        RelNode
            The root node of the algebra expression. Can be the current node if it does not have a parent.
        """
        if self._parent is None:
            return self
        return self._parent.root()

    def leaf(self) -> RelNode | None:
        """Traverses the algebra tree downwards until a unique leaf node is found.

        Returns
        -------
        Optional[RelNode]
            The leaf node. If multiple leaf nodes exist (e.g. for join nodes), *None* is returned.
        """
        children = self.children()
        if not children:
            return self
        return children[0] if len(children) == 1 else None

    @abc.abstractmethod
    def children(self) -> Sequence[RelNode]:
        """Provides all input nodes of the current operator.

        Returns
        -------
        Sequence[RelNode]
            The input nodes. For leave nodes such as table scans, the sequence will be usually empty (except for subquery
            aliases), otherwise the children are provided from left to right.
        """
        raise NotImplementedError

    def tables(self, *, ignore_subqueries: bool = False) -> frozenset[TableReference]:
        """Provides all relations that are contained in the current node.

        Consider the following algebraic expression: *π(⋈(σ(R), S))*. This expression contains two relations: *R* and *S*.

        Parameters
        ----------
        ignore_subqueries : bool, optional
            Whether relations that are only referenced in subquery subtrees should be excluded. Off by default.

        Returns
        -------
        frozenset[TableReference]
            The tables
        """
        return frozenset(util.set_union(child.tables(ignore_subqueries=ignore_subqueries) for child in self.children()))

    def provided_expressions(self) -> frozenset[SqlExpression]:
        """Collects all expressions that are available to parent nodes.

        These expressions will contain all expressions that are provided by child nodes as well as all expressions that are
        calculated by the current node itself.

        Returns
        -------
        frozenset[expressions.SqlExpression]
            The expressions
        """
        return frozenset(util.set_union(child.provided_expressions() for child in self.children()))

    @abc.abstractmethod
    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        """Enables processing of the current algebraic expression by an expression visitor.

        Parameters
        ----------
        visitor : RelNodeVisitor[VisitorResult]
            The visitor
        """
        raise NotImplementedError

    def dfs_walk(self) -> Generator[RelNode, None, None]:
        """Performs a depth-first search on the algebraic expression.

        This produces the subtree induced by the current node. The current node is also included in the output.

        Yields
        ------
        Generator[RelNode, None, None]
            All nodes of the subtree induced by the current node.
        """
        yield self
        for child in self.children():
            yield from child.dfs_walk()

    def mutate(self, *, as_root: bool = False, **kwargs) -> RelNode:
        """Creates a new instance of the current operator with modified attributes.

        The specific parameters depend on the concrete operator type. Calling `mutate()` on *any* node generates a copy of the
        *entire* tree, so make sure to use this operation sparingly.

        See *Notes* for important remarks regarding the mutate implementation for custom node types.

        Parameters
        ----------
        as_root : bool, optional
            Whether the current node should become the new root node of the tree. Defaults to *False*, which leaves the node
            at its current position in the tree.
        **kwargs
            The attributes to modify. The specific attributes depend on the concrete operator type.

        Returns
        -------
        RelNode
            The modified node

        Notes
        -----
        Concrete node types should implement their own `mutate()` function which accepts parameters specific to the node type
        (in addition to the required `as_root` parameter). Other than the function prototype, the methods can share the same
        default implementation:

        .. code-block:: python

            params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
            return super().mutate(**params)

        In order for the method to work properly, all field-specific parameters *must* use the common property/attribute
        conventions. For example, to update a property `input_node` of the operator, its value must be stored in a "private"
        attribute `_input_node`. At the same time, the `mutate()` method must accept a parameter `input_node` (without the
        leading underscore). The method will then automatically update the internal attribute with the new value.

        See the implementation of `Selection` for an example.
        """
        update_service = _RelNodeUpdateManager(self.root(), initiator=self)
        relalg_copy = update_service.make_relalg_copy(as_root=as_root, **kwargs)
        return relalg_copy.updated_initiator

    def clone(self) -> RelNode:
        """Obtains a 1:1 copy of the current node.

        Returns
        -------
        RelNode
            The cloned node
        """
        copied = copy.copy(self)
        copied._sideways_pass = set(copied.sideways_pass)
        return copied

    def inspect(self, *, _indentation: int = 0) -> str:
        """Provides a nice hierarchical string representation of the algebraic expression.

        The representation typically spans multiple lines and uses indentation to separate parent nodes from their
        children.

        Parameters
        ----------
        indentation : int, optional
            Internal parameter to the `inspect` function. Should not be modified by the user. Denotes how deeply
            recursed we are in the plan tree. This enables the correct calculation of the current indentation level.
            Defaults to 0 for the root node.

        Returns
        -------
        str
            A string representation of the algebraic expression
        """
        padding = " " * _indentation
        prefix = f"{padding}<- " if padding else ""
        inspections = [prefix + str(self)]
        for child in self.children():
            inspections.append(child.inspect(_indentation=_indentation + 2))
        return "\n".join(inspections)

    def _maintain_child_links(self, *, recursive: bool = False) -> None:
        """Ensures that all child nodes of the current node *A* have *A* set as their parent.

        Parameters
        ----------
        recursive : bool, optional
            Whether the child links should be maintained recursively. Defaults to *False*.
        """
        for child in self.children():
            if child._parent is None:
                child._parent = self
                continue
            child._sideways_pass.add(self)

        if not recursive:
            return
        for child in self.children():
            child._maintain_child_links(recursive=True)

    def _update_child_nodes(self, children: Sequence[RelNode]) -> None:
        """Updates the child nodes of the current operator.

        This method uses two heuristics: children of nodes with a single input should be called `input_node` (with its internal
        attribute `_input_node`), whereas nodes with two inputs should be called `left_input` and `right_input` (with internal
        attributes `_left_input` and `_right_input`, respectively).

        If none of these fields are found, a `NotImplementedError` is raised.

        Parameters
        ----------
        children : Sequence[RelNode]
            The new children

        Raises
        ------
        ValueError
            If the number of supplied children does not match the expected number of child nodes.
        NotImplementedError
            If the current node does not have the required attributes to update its child nodes.
        """
        self._assert_correct_update_child_count(children)
        attrs = vars(self)
        if "_input_node" in attrs and len(children) == 1:
            attrs["_input_node"] = children[0]
        elif "_left_input" in attrs and "_right_input" in attrs and len(children) == 2:
            attrs["_left_input"] = children[0]
            attrs["_right_input"] = children[1]
        else:
            raise NotImplementedError(
                f"Cannot use the default implementation of _update_child_nodes for node '{self}'."
            )

    def _clear_parent_links(self, *, recursive: bool = False) -> None:
        """Removes all references to parent nodes from the current operator.

        Parameters
        ----------
        recursive : bool, optional
            Whether the parent links should be cleared recursively. Defaults to *False*.
        """
        self._parent = None
        self._sideways_pass.clear()

        if not recursive:
            return
        for child in self.children():
            child._clear_parent_links(recursive=True)

    def _rebuild_linkage(self) -> None:
        """Ensures that the subtree induced by the current node is correctly linked."""
        self._clear_parent_links(recursive=True)
        self._maintain_child_links(recursive=True)

    def _rehash(self) -> None:
        """Re-calculates the hash value of the current node and all its children."""
        self._hash_val = hash((self._node_type, self._recalc_hash_val()))
        for child in self.children():
            child._rehash()

    def _assert_correct_update_child_count(self, children: Sequence[RelNode]) -> None:
        """Ensures that the correct number of child nodes is supplied for updating the current node.

        Parameters
        ----------
        children : Sequence[RelNode]
            The new children

        Raises
        ------
        ValueError
            If the number of supplied children does not match the expected number of child nodes.
        """
        n_current_children = len(self.children())
        n_new_children = len(children)
        if n_current_children != n_new_children:
            raise ValueError(
                f"Cannot update a node containing {n_current_children} child nodes with {n_new_children} children."
            )

    @abc.abstractmethod
    def _recalc_hash_val(self) -> int:
        """Calculates the hash value of the current node.

        This method only needs to consider attributes that are unique to the node. The final hash value will incorporate
        additional information that is shared between nodes, such as the node type.

        Returns
        -------
        int
            The current hash value
        """
        raise NotImplementedError

    def __hash__(self) -> int:
        return self._hash_val

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        child_reprs = ", ".join(repr(child) for child in self.children())
        return f"{self.node_type}({child_reprs})"

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class Selection(RelNode):
    """A selection filters the input relation based on an arbitrary predicate.

    Parameters
    ----------
    input_node : RelNode
        The tuples to filter
    predicate : AbstractPredicate
        The predicate that must be satisfied by all output tuples
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.

    Notes
    -----
    A selection is defined as

    .. math:: \\sigma_\\theta(R) := \\{ r \\in R | \\theta(r) \\}
    """

    def __init__(
        self,
        input_node: RelNode,
        predicate: AbstractPredicate,
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        self._input_node = input_node
        self._predicate = predicate
        super().__init__(parent_node)

    @property
    def input_node(self) -> RelNode:
        """Get the input relation that should be filtered.

        Returns
        -------
        RelNode
            A relation
        """
        return self._input_node

    @property
    def predicate(self) -> AbstractPredicate:
        """Get the predicate that must be satisfied by the output tuples.

        Returns
        -------
        AbstractPredicate
            The filter condition
        """
        return self._predicate

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_selection(self)

    def mutate(
        self,
        *,
        input_node: RelNode | None = None,
        predicate: AbstractPredicate | None = None,
        as_root: bool = False,
    ) -> Selection:
        """Creates a new selection with modified attributes.

        Parameters
        ----------
        input_node : Optional[RelNode], optional
            The new input node to use. If *None*, the current input node is re-used.
        predicate : Optional[AbstractPredicate], optional
            The new predicate to use. If *None*, the current predicate is re-used.
        as_root : bool, optional
            Whether the selection should become the new root node of the tree. This overwrites any value passed to
            `parent`.

        Returns
        -------
        Selection
            The modified selection node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash((self._input_node, self._predicate))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._input_node == other._input_node
            and self._predicate == other._predicate
        )

    def __str__(self) -> str:
        return f"σ ({self._predicate})"


class CrossProduct(RelNode):
    """A cross product calculates the cartesian product between tuples from two relations.

    Parameters
    ----------
    left_input : RelNode
        Relation containing the first set of tuples
    right_input : RelNode
        Relation containing the second set of tuples
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.

    Notes
    -----
    A cross product is defined as

    .. math:: R \\times S := \\{ r \\circ s | r \\in R, s \\in S \\}
    """

    def __init__(
        self,
        left_input: RelNode,
        right_input: RelNode,
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        self._left_input = left_input
        self._right_input = right_input
        super().__init__(parent_node)

    @property
    def left_input(self) -> RelNode:
        """Get the operator providing the first set of tuples.

        Returns
        -------
        RelNode
            A relation
        """
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        """Get the operator providing the second set of tuples.

        Returns
        -------
        RelNode
            A relation
        """
        return self._right_input

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_cross_product(self)

    def mutate(
        self,
        *,
        left_input: RelNode | None = None,
        right_input: RelNode | None = None,
        as_root: bool = False,
    ) -> CrossProduct:
        """Creates a new cross product with modified attributes.

        Parameters
        ----------
        left_input : Optional[RelNode], optional
            The new left child node to use. If *None*, the current left input node is re-used.
        right_input : Optional[RelNode], optional
            The new right child node to use. If *None*, the current right input node is re-used.
        as_root : bool, optional
            Whether the cross product should become the new root node of the tree. This overwrites any value passed to
            `parent`.

        Returns
        -------
        CrossProduct
            The modified cross product node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash((self._left_input, self._right_input))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._left_input == other._left_input
            and self._right_input == other._right_input
        )

    def __str__(self) -> str:
        return "⨯"


class Union(RelNode):
    """A union combines the tuple sets of two relations into a single output relation.

    In order for a union to work, both relations must have the same structure.

    Parameters
    ----------
    left_input : RelNode
        The first relation
    right_input : RelNode
        The second relation
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.

    Notes
    -----
    The union is defined as

    .. math:: R \\cup S := \\{ t | t \\in R \\lor t \\in S \\}
    """

    def __init__(
        self,
        left_input: RelNode,
        right_input: RelNode,
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        self._left_input = left_input
        self._right_input = right_input
        super().__init__(parent_node)

    @property
    def left_input(self) -> RelNode:
        """Get the operator providing the first relation's tuples.

        Returns
        -------
        RelNode
            A relation
        """
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        """Get the operator providing the second relation's tuples.

        Returns
        -------
        RelNode
            A relation
        """
        return self._right_input

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_union(self)

    def mutate(
        self,
        *,
        left_input: RelNode | None = None,
        right_input: RelNode | None = None,
        as_root: bool = False,
    ) -> Union:
        """Creates a new union with modified attributes.

        Parameters
        ----------
        left_input : Optional[RelNode], optional
            The new left child node to use. If *None*, the current left input node is re-used.
        right_input : Optional[RelNode], optional
            The new right child node to use. If *None*, the current right input node is re-used.
        as_root : bool, optional
            Whether the union should become the new root node of the tree. This overwrites any value passed to `parent`.

        Returns
        -------
        Union
            The modified union node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash((self._left_input, self._right_input))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._left_input == other._left_input
            and self._right_input == other._right_input
        )

    def __str__(self) -> str:
        return "∪"


class Intersection(RelNode):
    """An intersection provides all tuples that are contained in both of its input operators.

    In order for an intersection to work, both relations must have the same structure.

    Parameters
    ----------
    left_input : RelNode
        The first relation.
    right_input : RelNode
        The second relation.
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.

    Notes
    -----
    The difference is defined as

    .. math:: R \\cap S := \\{ t | t \\in R \\land t \\in S \\}
    """

    def __init__(
        self,
        left_input: RelNode,
        right_input: RelNode,
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        self._left_input = left_input
        self._right_input = right_input
        super().__init__(parent_node)

    @property
    def left_input(self) -> RelNode:
        """Get the operator providing the first relation's tuples.

        Returns
        -------
        RelNode
            A relation
        """
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        """Get the operator providing the second relation's tuples.

        Returns
        -------
        RelNode
            A relation
        """
        return self._right_input

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_intersection(self)

    def mutate(
        self,
        *,
        left_input: RelNode | None = None,
        right_input: RelNode | None = None,
        as_root: bool = False,
    ) -> Intersection:
        """Creates a new intersection with modified attributes.

        Parameters
        ----------
        left_input : Optional[RelNode], optional
            The new left child node to use. If *None*, the current left input node is re-used.
        right_input : Optional[RelNode], optional
            The new right child node to use. If *None*, the current right input node is re-used.
        as_root : bool, optional
            Whether the intersection should become the new root node of the tree. This overwrites any value passed to
            `parent`.

        Returns
        -------
        Intersection
            The modified intersection node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash((self._left_input, self._right_input))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._left_input == other._left_input
            and self._right_input == other._right_input
        )

    def __str__(self) -> str:
        return "∩"


class Difference(RelNode):
    """An intersection returns all tuples from one relation, that are not present in another relation.

    In order for the difference to work, both input relations must share the same structure.

    Parameters
    ----------
    left_input : RelNode
        The first relation. This is the relation to remove tuples from.
    right_input : RelNode
        The second relation. This is the relation containing the tuples that should be removed from the `left_input`.
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.

    Notes
    -----
    The difference is defined as

    .. math:: R \\setminus S := \\{ r \\in R | r \\notin S \\}
    """

    def __init__(
        self,
        left_input: RelNode,
        right_input: RelNode,
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        self._left_input = left_input
        self._right_input = right_input
        super().__init__(parent_node)

    @property
    def left_input(self) -> RelNode:
        """Get the operator providing the relation to remove tuples from.

        Returns
        -------
        RelNode
            A relation
        """
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        """Get the operator providing the tuples to remove.

        Returns
        -------
        RelNode
            A relation
        """
        return self._right_input

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_difference(self)

    def mutate(
        self,
        *,
        left_input: RelNode | None = None,
        right_input: RelNode | None = None,
        as_root: bool = False,
    ) -> Difference:
        """Creates a new difference with modified attributes.

        Parameters
        ----------
        left_input : Optional[RelNode], optional
            The new left child node to use. If *None*, the current left input node is re-used.
        right_input : Optional[RelNode], optional
            The new right child node to use. If *None*, the current right input node is re-used.
        as_root : bool, optional
            Whether the difference should become the new root node of the tree. This overwrites any value passed to
            `parent`.

        Returns
        -------
        Difference
            The modified difference node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash((self._left_input, self._right_input))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._left_input == other._left_input
            and self._right_input == other._right_input
        )

    def __str__(self) -> str:
        return "\\"


class Relation(RelNode):
    """A relation provides the tuples ("rows") contained in a table.

    Each relation can correspond to a physical table contained in some relational schema, or it can represent the result of a
    subquery operation.

    Parameters
    ----------
    table : TableReference
        The table that is represented by this relation.
    provided_columns : Iterable[ColumnReference  |  ColumnExpression]
        The columns that are contained in the table.
    subquery_input : Optional[RelNode], optional
        For subquery relations, this is the algebraic expression that computes the results of the subquery. Relations that
        correspond to base tables do not have this attribute set.
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.
    """

    def __init__(
        self,
        table: TableReference,
        provided_columns: Iterable[ColumnReference | ColumnExpression],
        *,
        subquery_input: RelNode | None = None,
        parent_node: RelNode | None = None,
    ) -> None:
        self._table = table
        self._provided_cols = frozenset(
            col if isinstance(col, ColumnExpression) else ColumnExpression(col) for col in provided_columns
        )

        self._subquery_input = subquery_input if subquery_input is not None else None
        if self._subquery_input is not None:
            # We need to set the parent node explicitly here in order to prevent infinite recursion
            self._subquery_input._parent = self

        super().__init__(parent_node)

    @property
    def table(self) -> TableReference:
        """Get the table that is represented by this relation.

        Returns
        -------
        TableReference
            A table. Usually this will correpond to an actual physical database table, but for subqueries this might also be a
            virtual table.
        """
        return self._table

    @property
    def subquery_input(self) -> RelNode | None:
        """Get the root node of the subquery that produces the input tuples for this relation.

        Returns
        -------
        Optional[RelNode]
            The root node if it exists, or *None* for actual base table relations.
        """
        return self._subquery_input

    def children(self) -> Sequence[RelNode]:
        return [self._subquery_input] if self._subquery_input else []

    def tables(self, *, ignore_subqueries: bool = False) -> frozenset[TableReference]:
        if ignore_subqueries:
            return frozenset((self._table,))
        return super().tables() | {self._table}

    def provided_expressions(self) -> frozenset[SqlExpression]:
        return super().provided_expressions() | self._provided_cols

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_base_relation(self)

    def mutate(
        self,
        *,
        table: TableReference | None = None,
        provided_columns: Iterable[ColumnReference | ColumnExpression] | None = None,
        subquery_input: RelNode | None = None,
        as_root: bool = False,
    ) -> Relation:
        """Creates a new relation with modified attributes.

        Parameters
        ----------
        table : Optional[TableReference], optional
            The new table to use. If *None*, the current table is re-used.
        provided_columns : Optional[Iterable[ColumnReference | ColumnExpression]], optional
            The new columns to use. If *None*, the current columns are re-used.
        subquery_input : Optional[RelNode], optional
            The new subquery input to use. If *None*, the current subquery input is re-used.
        as_root : bool, optional
            Whether the relation should become the new root node of the tree. This overwrites any value passed to `parent`.

        Returns
        -------
        Relation
            The modified relation node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _update_child_nodes(self, children: Sequence[RelNode]) -> None:
        self._assert_correct_update_child_count(children)
        if self._subquery_input:
            self._subquery_input = children[0]

    def _recalc_hash_val(self) -> int:
        return hash((self._table, self._subquery_input))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._table == other._table

    def __repr__(self) -> str:
        return self._table.identifier()

    def __str__(self) -> str:
        return self._table.identifier()


class ThetaJoin(RelNode):
    """A theta joins combines individual tuples from two input relations if they match a specific predicate.

    Parameters
    ----------
    left_input : RelNode
        Relation containing the first set of tuples.
    right_input : RelNode
        Relation containing the second set of tuples.
    predicate : AbstractPredicate
        A predicate that must be satisfied by all joined tuples.
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.

    Notes
    -----
    A theta join is defined as

    .. math:: \\bowtie_\\theta(R, S) := \\{ r \\circ s | r \\in R \\land s \\in S \\land \\theta(r, s) \\}
    """

    def __init__(
        self,
        left_input: RelNode,
        right_input: RelNode,
        predicate: AbstractPredicate,
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        self._left_input = left_input
        self._right_input = right_input
        self._predicate = predicate
        super().__init__(parent_node)

    @property
    def left_input(self) -> RelNode:
        """Get the operator providing the first set of tuples.

        Returns
        -------
        RelNode
            A relation
        """
        return self._left_input

    @property
    def right_input(self) -> RelNode:
        """Get the operator providing the second set of tuples.

        Returns
        -------
        RelNode
            A relation
        """
        return self._right_input

    @property
    def predicate(self) -> AbstractPredicate:
        """Get the condition that must be satisfied by the input tuples.

        Returns
        -------
        AbstractPredicate
            A predicate
        """
        return self._predicate

    def children(self) -> Sequence[RelNode]:
        return [self._left_input, self._right_input]

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_theta_join(self)

    def mutate(
        self,
        *,
        left_input: RelNode | None = None,
        right_input: RelNode | None = None,
        predicate: AbstractPredicate | None = None,
        as_root: bool = False,
    ) -> ThetaJoin:
        """Creates a new theta join with modified attributes.

        Parameters
        ----------
        left_input : Optional[RelNode], optional
            The new left child node to use. If *None*, the current left input node is re-used.
        right_input : Optional[RelNode], optional
            The new right child node to use. If *None*, the current right input node is re-used.
        predicate : Optional[AbstractPredicate], optional
            The new predicate to use. If *None*, the current predicate is re-used.
        as_root : bool, optional
            Whether the theta join should become the new root node of the tree. This overwrites any value passed to
            `parent`.

        Returns
        -------
        ThetaJoin
            The modified theta join node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash((self._left_input, self._right_input, self._predicate))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._left_input == other._left_input
            and self._right_input == other._right_input
            and self._predicate == other._predicate
        )

    def __str__(self) -> str:
        return f"⋈ ϴ=({self._predicate})"


class Projection(RelNode):
    """A projection selects individual attributes from the tuples of an input relation.

    The output relation will contain exactly the same tuples as the input, but each tuple will potentially contain less
    attributes.

    Parameters
    ----------
    input_node : RelNode
        The tuples to process
    targets : Sequence[SqlExpression]
        The attributes that should still be contained in the output relation
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.
    """

    def __init__(
        self,
        input_node: RelNode,
        targets: Sequence[SqlExpression],
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        self._input_node = input_node
        self._targets = tuple(targets)
        super().__init__(parent_node)

    @property
    def input_node(self) -> RelNode:
        """Get the operator providing the tuples to project.

        Returns
        -------
        RelNode
            A relation
        """
        return self._input_node

    @property
    def columns(self) -> Sequence[SqlExpression]:
        """Provides the attributes that should be included in the output relation's tuples.

        Returns
        -------
        Sequence[SqlExpression]
            The projected attributes.
        """
        return self._targets

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_projection(self)

    def mutate(
        self,
        *,
        input_node: RelNode | None = None,
        targets: Sequence[SqlExpression] | None = None,
        as_root: bool = False,
    ) -> Projection:
        """Creates a new projection with modified attributes.

        Parameters
        ----------
        input_node : Optional[RelNode], optional
            The new input node to use. If *None*, the current input node is re-used.
        targets : Optional[Sequence[SqlExpression]], optional
            The new targets to use. If *None*, the current targets are re-used.
        as_root : bool, optional
            Whether the projection should become the new root node of the tree. This overwrites any value passed to
            `parent`.

        Returns
        -------
        Projection
            The modified projection node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash((self._input_node, self._targets))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self)) and self._input_node == other._input_node and self._targets == other._targets
        )

    def __str__(self) -> str:
        col_str = ", ".join(str(col) for col in self._targets)
        return f"π ({col_str})"


class Grouping(RelNode):
    """Grouping partitions input tuples according to specific attributes and calculates aggregated values.

    Parameters
    ----------
    input_node : RelNode
        The tuples to process
    group_columns : Sequence[SqlExpression]
        The expressions that should be used to partition the input tuples. Can be empty if only aggregations over all input
        tuples should be computed.
    aggregates : Optional[dict[frozenset[SqlExpression], frozenset[FunctionExpression]]], optional
        The aggregates that should be computed. This is a mapping from the input expressions to the desired aggregate. Can be
        empty if only a grouping should be performed. In this case, the grouping operates as a duplicate-elimination mechanism.
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.
    """

    def __init__(
        self,
        input_node: RelNode,
        group_columns: Sequence[SqlExpression],
        *,
        aggregates: dict[frozenset[SqlExpression], frozenset[FunctionExpression]] | None = None,
        parent_node: RelNode | None = None,
    ) -> None:
        if not group_columns and not aggregates:
            raise ValueError("Either group columns or aggregation functions must be specified!")
        self._input_node = input_node
        self._group_columns = tuple(group_columns)
        self._aggregates: util.frozendict[frozenset[SqlExpression], frozenset[FunctionExpression]] = util.frozendict(
            aggregates
        )
        super().__init__(parent_node)

    @property
    def input_node(self) -> RelNode:
        """Get the operator that provides the tuples to group.

        Returns
        -------
        RelNode
            A relation
        """
        return self._input_node

    @property
    def group_columns(self) -> Sequence[SqlExpression]:
        """Get the expressions that should be used to partition the input tuples.

        Returns
        -------
        Sequence[SqlExpression]
            The group columns. Can be empty if only aggregations over all input tuples should be computed.
        """
        return self._group_columns

    @property
    def aggregates(self) -> util.frozendict[frozenset[SqlExpression], frozenset[FunctionExpression]]:
        """Get the aggregates that should be computed.

        Aggregates map from the input expressions to the desired aggregation function.

        Returns
        -------
        util.frozendict[frozenset[SqlExpression], frozenset[FunctionExpression]]
            The aggregations. Can be empty if only a grouping should be performed.
        """
        return self._aggregates

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def provided_expressions(self) -> frozenset[SqlExpression]:
        aggregate_expressions = util.set_union(self._aggregates.values())
        return frozenset(set(self._group_columns) | aggregate_expressions)

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_grouping(self)

    def mutate(
        self,
        *,
        input_node: RelNode | None = None,
        group_columns: Sequence[SqlExpression] | None = None,
        aggregates: dict[frozenset[SqlExpression], frozenset[FunctionExpression]] | None = None,
        parent: RelNode | None = None,
        as_root: bool = False,
    ) -> Grouping:
        """Creates a new group by with modified attributes.

        Parameters
        ----------
        input_node : Optional[RelNode], optional
            The new input node to use. If *None*, the current input node is re-used.
        group_columns : Optional[Sequence[SqlExpression]], optional
            The new group columns to use. If *None*, the current group columns are re-used.
        aggregates : Optional[dict[frozenset[SqlExpression], frozenset[FunctionExpression]]], optional
            The new aggregates to use. If *None*, the current aggregates are re-used.
        as_root : bool, optional
            Whether the group by should become the new root node of the tree. This overwrites any value passed to `parent`.

        Returns
        -------
        Grouping
            The modified group by node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash((self._input_node, self._group_columns, self._aggregates))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._input_node == other._input_node
            and self._group_columns == other._group_columns
            and self._aggregates == other._aggregates
        )

    def __str__(self) -> str:
        pretty_aggregations: dict[str, str] = {}
        for cols, agg_funcs in self._aggregates.items():
            if len(cols) == 1:
                col_str = str(util.simplify(cols))
            else:
                col_str = "(" + ", ".join(str(c) for c in cols) + ")"
            if len(agg_funcs) == 1:
                agg_str = str(util.simplify(agg_funcs))
            else:
                agg_str = "(" + ", ".join(str(agg) for agg in agg_funcs) + ")"
            pretty_aggregations[col_str] = agg_str

        agg_str = ", ".join(f"{col}: {agg_func}" for col, agg_func in pretty_aggregations.items())
        if not self._group_columns:
            return f"γ ({agg_str})"
        group_str = ", ".join(str(col) for col in self._group_columns)
        return f"{group_str} γ ({agg_str})"


class Rename(RelNode):
    """Rename remaps column names to different names.

    Parameters
    ----------
    input_node : RelNode
        The tuples to modify
    mapping : dict[ColumnReference, ColumnReference]
        A map from current column name to new column name.
    parent_node : Optional[RelNode]
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.

    Warnings
    --------
    This node is currently not used since we do not support natural joins.
    """

    def __init__(
        self,
        input_node: RelNode,
        mapping: dict[ColumnReference, ColumnReference],
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        # TODO: check types + add provided / required expressions method
        self._input_node = input_node
        self._mapping = util.frozendict(mapping)
        super().__init__(parent_node)

    @property
    def input_node(self) -> RelNode:
        """Get the operator that provides the tuples to modify

        Returns
        -------
        RelNode
            A relation
        """
        return self._input_node

    @property
    def mapping(self) -> util.frozendict[ColumnReference, ColumnReference]:
        """Get the performed renamings.

        Returns
        -------
        util.frozendict[ColumnReference, ColumnReference]
            A map from current column name to new column name.
        """
        return self._mapping

    def provided_expressions(self) -> frozenset[SqlExpression]:
        return frozenset(
            transform.rename_columns_in_expression(child_expr, self._mapping)
            for child_expr in self._input_node.provided_expressions()
        )

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_rename(self)

    def mutate(
        self,
        *,
        input_node: RelNode | None = None,
        mapping: dict[ColumnReference, ColumnReference] | None = None,
        as_root: bool = False,
    ) -> Rename:
        """Creates a new rename with modified attributes.

        Parameters
        ----------
        input_node : Optional[RelNode], optional
            The new input node to use. If *None*, the current input node is re-used.
        mapping : Optional[dict[ColumnReference, ColumnReference]], optional
            The new mapping to use. If *None*, the current mapping is re-used.
        as_root : bool, optional
            Whether the rename should become the new root node of the tree. This overwrites any value passed to `parent`.

        Returns
        -------
        Rename
            The modified rename node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash((self._input_node, self._mapping))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self)) and self._input_node == other._input_node and self._mapping == other._mapping
        )

    def __str__(self) -> str:
        map_str = ", ".join(f"{col}: {target}" for col, target in self._mapping.items())
        return f"ϱ ({map_str})"


SortDirection = typing.Literal["asc", "desc"]
"""Describes whether tuples should be sorted in ascending or descending order."""


class Sort(RelNode):
    """Sort modifies the order in which tuples are provided.

    Parameters
    ----------
    input_node : RelNode
        The tuples to order
    sorting : Sequence[tuple[SqlExpression, SortDirection]  |  SqlExpression]
        The expressions that should be used to determine the sorting. For expressions that do not specify any particular
        direction, ascending order is assumed. Later expressions are used to solve ties among tuples with the same expression
        values in the first couple of expressions.
    parent_node : Optional[RelNode], optional
        _description_, by default None

    Notes
    -----
    Strictly speaking, this operator is not part of traditional relational algebra. This is because the algebra uses
    set-semantics which do not supply any ordering. However, due to SQL's *ORDER BY* clause, most relational algebra dialects
    support ordering nevertheless.

    However, we do not support special placement of *NULL* column values, i.e. no *ORDER BY R.a NULLS LAST*, etc.
    """

    # TODO: support NULLS FIRST/NULLS LAST
    def __init__(
        self,
        input_node: RelNode,
        sorting: Sequence[tuple[SqlExpression, SortDirection] | SqlExpression],
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        self._input_node = input_node
        self._sorting = tuple(
            [sort_item if isinstance(sort_item, tuple) else (sort_item, "asc") for sort_item in sorting]
        )
        super().__init__(parent_node)

    @property
    def input_node(self) -> RelNode:
        """Get the operator providing the tuples to sort.

        Returns
        -------
        RelNode
            A relation
        """
        return self._input_node

    @property
    def sorting(self) -> Sequence[tuple[SqlExpression, SortDirection]]:
        """Get the desired ordering.

        Later expressions are used to solve ties among tuples with the same expression values in the first couple of
        expressions.

        Returns
        -------
        Sequence[tuple[SqlExpression, SortDirection]]
            The expressions to order, most signifcant orders coming first.
        """
        return self._sorting

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_sort(self)

    def mutate(
        self,
        *,
        input_node: RelNode | None = None,
        sorting: Sequence[tuple[SqlExpression, SortDirection] | SqlExpression] | None = None,
        as_root: bool = False,
    ) -> Sort:
        """Creates a new sort with modified attributes.

        Parameters
        ----------
        input_node : Optional[RelNode], optional
            The new input node to use. If *None*, the current input node is re-used.
        sorting : Optional[Sequence[tuple[SqlExpression, SortDirection] | SqlExpression]], optional
            The new sorting to use. If *None*, the current sorting is re-used.
        as_root : bool, optional
            Whether the sort should become the new root node of the tree. This overwrites any value passed to `parent`.

        Returns
        -------
        Sort
            The modified sort node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash((self._input_node, self._sorting))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self)) and self._input_node == other._input_node and self._sorting == other._sorting
        )

    def __str__(self) -> str:
        sorting_str = ", ".join(
            f"{sort_col}{'↓' if sort_dir == 'asc' else '↑'}" for sort_col, sort_dir in self._sorting
        )
        return f"τ ({sorting_str})"


class Map(RelNode):
    """Mapping computes new expressions from the currently existing ones.

    For example, the expression *R.a + 42* can be computed during a mapping operation based on the input from a relation
    node *R*.

    Parameters
    ----------
    input_node : RelNode
        The tuples to process
    mapping : dict[frozenset[SqlExpression  |  ColumnReference], frozenset[SqlExpression]]
        The expressions to compute. Maps from the arguments to the target expressions. The arguments themselves can be computed
        during the very same mapping operation. Alternatively, they can be supplied by the `input_node`.
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.
    """

    def __init__(
        self,
        input_node: RelNode,
        mapping: Mapping[frozenset[SqlExpression | ColumnReference], frozenset[SqlExpression]],
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        self._input_node = input_node
        self._mapping = util.frozendict(
            {
                ColumnExpression(expression) if isinstance(expression, ColumnReference) else expression: target
                for expression, target in mapping.items()
            }
        )
        super().__init__(parent_node)

    @property
    def input_node(self) -> RelNode:
        """Get the operator that provides the tuples to map.

        Returns
        -------
        RelNode
            A relation
        """
        return self._input_node

    @property
    def mapping(
        self,
    ) -> util.frozendict[frozenset[SqlExpression], frozenset[SqlExpression]]:
        """Get the expressions to compute. Maps from the arguments to the target expressions.

        The arguments themselves can be computed during the very same mapping operation. Alternatively, they can be supplied by
        the input node.

        Returns
        -------
        util.frozendict[frozenset[SqlExpression], frozenset[SqlExpression]]
            The expressions
        """
        return self._mapping

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def provided_expressions(self) -> frozenset[SqlExpression]:
        return super().provided_expressions() | util.set_union(map_target for map_target in self._mapping.values())

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_map(self)

    def mutate(
        self,
        *,
        input_node: RelNode | None = None,
        mapping: dict[frozenset[SqlExpression | ColumnReference], frozenset[SqlExpression]] | None = None,
        as_root: bool = False,
    ) -> Map:
        """Creates a new map with modified attributes.

        Parameters
        ----------
        input_node : Optional[RelNode], optional
            The new input node to use. If *None*, the current input node is re-used.
        mapping : Optional[dict[frozenset[SqlExpression | ColumnReference], frozenset[SqlExpression]]], optional
            The new mapping to use. If *None*, the current mapping is re-used.
        as_root : bool, optional
            Whether the map should become the new root node of the tree. This overwrites any value passed to `parent`.

        Returns
        -------
        Map
            The modified map node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash((self._input_node, self._mapping))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self)) and self._input_node == other._input_node and self._mapping == other._mapping
        )

    def __str__(self) -> str:
        pretty_mapping: dict[str, str] = {}
        for target_col, expression in self._mapping.items():
            if len(target_col) == 1:
                target_col = util.simplify(target_col)
                target_str = str(target_col)
            else:
                target_str = "(" + ", ".join(str(t) for t in target_col) + ")"
            if len(expression) == 1:
                expression = util.simplify(expression)
                expr_str = str(expression)
            else:
                expr_str = "(" + ", ".join(str(e) for e in expression) + ")"
            pretty_mapping[target_str] = expr_str

        mapping_str = ", ".join(f"{target_col}: {expr}" for target_col, expr in pretty_mapping.items())
        return f"χ ({mapping_str})"


class DuplicateElimination(RelNode):
    """Duplicate elimination ensures that all attribute combinations of all tuples are unique.

    Parameters
    ----------
    input_node : RelNode
        The tuples that should be unique
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.

    Notes
    -----
    Strictly speaking, this operator is not part of traditional relational algebra. This is because the algebra uses
    set-semantics which do not supply any ordering. However, due to SQL's usage of multi-sets which allow duplicates, most
    relational algebra dialects support ordering nevertheless.
    """

    def __init__(self, input_node: RelNode, *, parent_node: RelNode | None = None) -> None:
        self._input_node = input_node
        super().__init__(parent_node)

    @property
    def input_node(self) -> RelNode:
        return self._input_node

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_duplicate_elim(self)

    def mutate(self, *, input_node: RelNode | None = None, as_root: bool = False) -> DuplicateElimination:
        """Creates a new duplicate elimination with modified attributes.

        Parameters
        ----------
        input_node : Optional[RelNode], optional
            The new input node to use. If *None*, the current input node is re-used.
        as_root : bool, optional
            Whether the duplicate elimination should become the new root node of the tree. This overwrites any value passed to
            `parent`.

        Returns
        -------
        DuplicateElimination
            The modified duplicate elimination node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash(self._input_node)

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._input_node == other._input_node

    def __str__(self) -> str:
        return "δ"


class SemiJoin(RelNode):
    """A semi join provides all tuples from one relation with a matching partner tuple from another relation.

    Parameters
    ----------
    input_node : RelNode
        The tuples to "filter"
    subquery_node : SubqueryScan
        The relation that provides all tuples that have to match tuples in the `input_node`.
    predicate : Optional[AbstractPredicate], optional
        An optional predicate that is used to determine a match.
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.

    Notes
    -----
    A semi join is defined as

    .. math:: ⋉_\\theta(R, S) := \\{ r | r \\in R \\land s \\in S \\land \\theta(r, s) \\}
    """

    def __init__(
        self,
        input_node: RelNode,
        subquery_node: SubqueryScan,
        predicate: AbstractPredicate | None = None,
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        # TODO: dependent iff predicate is None
        self._input_node = input_node

        self._subquery_node: SubqueryScan = subquery_node.mutate()
        self._subquery_node._parent = self  # we need to set the parent manually to prevent infinite recursion

        self._predicate = predicate

        super().__init__(parent_node)

    @property
    def input_node(self) -> RelNode:
        """Get the operator providing the tuples to filter.

        Returns
        -------
        RelNode
            A relation
        """
        return self._input_node

    @property
    def subquery_node(self) -> SubqueryScan:
        """Get the operator providing the filtering tuples.

        Returns
        -------
        SubqueryScan
            A relation
        """
        return self._subquery_node

    @property
    def predicate(self) -> AbstractPredicate | None:
        """Get the match condition to determine the join partners.

        If there is no dedicated predicate, tuples from the `input_node` match, if any tuple is emitted by the
        `subquery_node`.

        Returns
        -------
        Optional[AbstractPredicate]
            The condition
        """
        return self._predicate

    def is_dependent(self) -> bool:
        """Checks, whether the subquery relation is depdent (sometimes also called correlated) with the input relation.

        Returns
        -------
        bool
            Whether the subquery is correlated with the input query

        See Also
        --------
        SqlQuery.is_depedent
        """
        return self._predicate is None

    def children(self) -> Sequence[RelNode]:
        return [self._input_node, self._subquery_node]

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_semijoin(self)

    def mutate(
        self,
        *,
        input_node: RelNode | None = None,
        subquery_node: SubqueryScan | None = None,
        predicate: AbstractPredicate | None = None,
        as_root: bool = False,
    ) -> SemiJoin:
        """Creates a new semi join with modified attributes.

        Parameters
        ----------
        input_node : Optional[RelNode], optional
            The new input node to use. If *None*, the current input node is re-used.
        subquery_node : Optional[SubqueryScan], optional
            The new subquery node to use. If *None*, the current subquery node is re-used.
        predicate : Optional[AbstractPredicate], optional
            The new predicate to use. If *None*, the current predicate is re-used.
        as_root : bool, optional
            Whether the semi join should become the new root node of the tree. This overwrites any value passed to `parent`.

        Returns
        -------
        SemiJoin
            The modified semi join node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _update_child_nodes(self, children: Sequence[RelNode]) -> None:
        assert isinstance(children[1], SubqueryScan)
        self._assert_correct_update_child_count(children)
        self._input_node = children[0]
        self._subquery_node = children[1]

    def _recalc_hash_val(self) -> int:
        return hash((self._input_node, self._subquery_node, self._predicate))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._input_node == other._input_node
            and self._subquery_node == other._subquery_node
            and self._predicate == other._predicate
        )

    def __str__(self) -> str:
        return "⋉" if self._predicate is None else f"⋉ ({self._predicate})"


class AntiJoin(RelNode):
    """An anti join provides all tuples from one relation with no matching partner tuple from another relation.

    Parameters
    ----------
    input_node : RelNode
        The tuples to "filter"
    subquery_node : SubqueryScan
        The relation that provides all tuples that have to match tuples in the `input_node`.
    predicate : Optional[AbstractPredicate], optional
        An optional predicate that is used to determine a match.
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.

    Notes
    -----
    An anti join is defined as

    .. math:: ▷_\\theta(R, S) := \\{ r | r \\in R \\land s \\in S \\land \\theta(r, s) \\}

    """

    def __init__(
        self,
        input_node: RelNode,
        subquery_node: SubqueryScan,
        predicate: AbstractPredicate | None = None,
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        # TODO: dependent iff predicate is None
        self._input_node = input_node

        self._subquery_node: SubqueryScan = subquery_node.mutate()
        self._subquery_node._parent = self  # we need to set the parent manually to prevent infinite recursion

        self._predicate = predicate

        super().__init__(parent_node)

    @property
    def input_node(self) -> RelNode:
        """Get the operator providing the tuples to filter.

        Returns
        -------
        RelNode
            A relation
        """
        return self._input_node

    @property
    def subquery_node(self) -> SubqueryScan:
        """Get the operator providing the filtering tuples.

        Returns
        -------
        SubqueryScan
            A relation
        """
        return self._subquery_node

    @property
    def predicate(self) -> AbstractPredicate | None:
        """Get the match condition to determine the join partners.

        If there is no dedicated predicate, tuples from the `input_node` match, if any tuple is emitted by the
        `subquery_node`.

        Returns
        -------
        Optional[AbstractPredicate]
            The condition
        """
        return self._predicate

    def is_dependent(self) -> bool:
        """Checks, whether the subquery relation is depdent (sometimes also called correlated) with the input relation.

        Returns
        -------
        bool
            Whether the subquery is correlated with the input query

        See Also
        --------
        SqlQuery.is_depedent
        """
        return self._predicate is None

    def children(self) -> Sequence[RelNode]:
        return [self._input_node, self._subquery_node]

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_antijoin(self)

    def mutate(
        self,
        *,
        input_node: RelNode | None = None,
        subquery_node: SubqueryScan | None = None,
        predicate: AbstractPredicate | None = None,
        as_root: bool = False,
    ) -> AntiJoin:
        """Creates a new anti join with modified attributes.

        Parameters
        ----------
        input_node : Optional[RelNode], optional
            The new input node to use. If *None*, the current input node is re-used.
        subquery_node : Optional[SubqueryScan], optional
            The new subquery node to use. If *None*, the current subquery node is re-used.
        predicate : Optional[AbstractPredicate], optional
            The new predicate to use. If *None*, the current predicate is re-used.
        as_root : bool, optional
            Whether the anti join should become the new root node of the tree. This overwrites any value passed to `parent`.

        Returns
        -------
        AntiJoin
            The modified anti join node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _update_child_nodes(self, children: Sequence[RelNode]) -> None:
        assert isinstance(children[1], SubqueryScan)
        self._assert_correct_update_child_count(children)
        self._input_node = children[0]
        self._subquery_node = children[1]

    def _recalc_hash_val(self) -> int:
        return hash((self._input_node, self._subquery_node, self._predicate))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._input_node == other._input_node
            and self._subquery_node == other._subquery_node
            and self._predicate == other._predicate
        )

    def __str__(self) -> str:
        return "▷" if self._predicate is None else f"▷ ({self._predicate})"


class SubqueryScan(RelNode):
    """A meta node to designate a subtree that originated from a subquery.

    Parameters
    ----------
    input_node : RelNode
        The relation that identifies the subquery result
    subquery : SqlQuery
        The query that actually calculates the subquery
    parent_node : Optional[RelNode], optional
        The parent node of the operator, if one exists. The parent is the operator that receives the output relation of the
        current operator. If the current operator is the root and (currently) does not have a parent, *None* can be used.

    Notes
    -----
    This node is not part of traditional relational algebra, nor do many other systems make use of it. For our purposes it
    serves as a marker node to quickly designate subqueries and to operate on the original queries or their algebraic
    representation in a convenient manner.
    """

    def __init__(
        self,
        input_node: RelNode,
        subquery: SelectStatement,
        *,
        parent_node: RelNode | None = None,
    ) -> None:
        self._input_node = input_node
        self._subquery = subquery
        super().__init__(parent_node)

    @property
    def input_node(self) -> RelNode:
        """Get the result node of the subquery

        Returns
        -------
        RelNode
            A relation
        """
        return self._input_node

    @property
    def subquery(self) -> SelectStatement:
        """Get the actual subquery.

        Returns
        -------
        SqlQuery
            A query
        """
        return self._subquery

    def tables(self, *, ignore_subqueries: bool = False) -> frozenset[TableReference]:
        return frozenset() if ignore_subqueries else super().tables(ignore_subqueries=ignore_subqueries)

    def children(self) -> Sequence[RelNode]:
        return [self._input_node]

    def provided_expressions(self) -> frozenset[SqlExpression]:
        return frozenset({SubqueryExpression(self._subquery)} | super().provided_expressions())

    def accept_visitor(self, visitor: RelNodeVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_subquery(self)

    def mutate(
        self,
        *,
        input_node: RelNode | None = None,
        subquery: SelectStatement | None = None,
        as_root: bool = False,
    ) -> SubqueryScan:
        """Creates a new subquery scan with modified attributes.

        Parameters
        ----------
        input_node : Optional[RelNode], optional
            The new input node to use. If *None*, the current input node is re-used.
        subquery : Optional[SqlQuery], optional
            The new subquery to use. If *None*, the current subquery is re-used.
        as_root : bool, optional
            Whether the subquery scan should become the new root node of the tree. This overwrites any value passed to
            `parent`.

        Returns
        -------
        SubqueryScan
            The modified subquery scan node

        See Also
        --------
        RelNode.mutate : for safety considerations and calling conventions
        """
        params = {param: val for param, val in locals().items() if param != "self" and not param.startswith("__")}
        return super().mutate(**params)

    def _recalc_hash_val(self) -> int:
        return hash((self._input_node, self._subquery))

    __hash__ = RelNode.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._input_node == other._input_node
            and self._subquery == other._subquery
        )

    def __str__(self) -> str:
        return "<<Scalar Subquery Scan>>" if self._subquery.is_scalar() else "<<Subquery Scan>>"


VisitorResult = typing.TypeVar("VisitorResult")
"""Result type of visitor processes."""


class RelNodeVisitor(abc.ABC, typing.Generic[VisitorResult]):
    """Basic visitor to operator on arbitrary relational algebra trees.

    See Also
    --------
    RelNode

    References
    ----------

    .. Visitor pattern: https://en.wikipedia.org/wiki/Visitor_pattern
    """

    @abc.abstractmethod
    def visit_selection(self, selection: Selection) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_cross_product(self, cross_product: CrossProduct) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_union(self, union: Union) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_intersection(self, intersection: Intersection) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_difference(self, difference: Difference) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_base_relation(self, base_table: Relation) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_theta_join(self, join: ThetaJoin) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_projection(self, projection: Projection) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_grouping(self, grouping: Grouping) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_rename(self, rename: Rename) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_sort(self, sorting: Sort) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_map(self, mapping: Map) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_duplicate_elim(self, duplicate_elim: DuplicateElimination) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_semijoin(self, join: SemiJoin) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_antijoin(self, join: AntiJoin) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_subquery(self, subquery: SubqueryScan) -> VisitorResult:
        raise NotImplementedError


def _collect_leaf_nodes(root: RelNode) -> set[Relation]:
    """Provides all leaf nodes of a relational algebra tree.

    For relation nodes with subquery input, only the leaf nodes of the subquery are returned.

    Parameters
    ----------
    root : RelNode
        The root node of the tree

    Returns
    -------
    set[Relation]
        The leaf nodes
    """
    if isinstance(root, Relation) and not root.subquery_input:
        nodes = {root}
    else:
        nodes = util.set_union(_collect_leaf_nodes(child) for child in root.children())
    return nodes


@dataclasses.dataclass
class _RelTreeUpdateSet:
    """Holds the root node of the updated tree and the node that asked for the tree update in the first place."""

    updated_root: RelNode
    updated_initiator: RelNode


class _RelNodeUpdateManager:
    """Handles the process of copying and modifying an entire relational algebra tree.

    Parameters
    ----------
    root : RelNode
        The current root node of the tree
    initiator : RelNode
        The node that asked for the tree update in the first place
    """

    def __init__(self, root: RelNode, *, initiator: RelNode) -> None:
        self._root = root
        self._initiator = initiator
        self._updated_nodes: dict[RelNode, RelNode] = {}
        self._node_working_set: list[RelNode] = list(_collect_leaf_nodes(root))

    def make_relalg_copy(self, **kwargs) -> _RelTreeUpdateSet:
        """Creates a copy of the relational algebra tree and applies modifications to it.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments to apply to the tree nodes. These are the arguments to the `mutate` call.

        Returns
        -------
        _RelTreeUpdateSet
            The updated relalg tree.
        """
        as_root: bool = kwargs.get("as_root", False)

        updated_root: RelNode
        updated_initiator: RelNode

        while self._node_working_set:
            current_node = self._node_working_set.pop(0)
            self._update_node_working_set(current_node)

            already_updated = current_node in self._updated_nodes
            pending_child_update = any(child not in self._updated_nodes for child in current_node.children())
            if already_updated or pending_child_update:
                continue

            updated_node = current_node.clone()
            updated_node._update_child_nodes([self._updated_nodes[child] for child in current_node.children()])
            updated_node._clear_parent_links()

            if current_node == self._initiator:
                self._perform_node_update(updated_node, **kwargs)
                updated_initiator = updated_node
                if as_root:
                    updated_root = updated_node
                    break

            self._updated_nodes[current_node] = updated_node

            if not as_root and current_node == self._root:
                updated_root = updated_node

        assert updated_initiator is not None
        assert updated_root is not None

        updated_root._rebuild_linkage()
        updated_root._rehash()
        self._root._rebuild_linkage()
        self._root._rehash()

        return _RelTreeUpdateSet(updated_root, updated_initiator)

    def _perform_node_update(self, node: RelNode, **kwargs) -> None:
        """Updates the initiator of a tree mutation with node-specific arguments.

        Parameters
        ----------
        node : RelNode
            The node to update. Must be the *copied* version of the initiator node.
        **kwargs
            The arguments to apply to the node
        """
        fields: set[str] = set(vars(node).keys())
        for attr, value in kwargs.items():
            if value is None:
                continue
            actual_attr_name = f"_{attr}"
            if actual_attr_name not in fields:
                continue

            if isinstance(value, RelNode):
                value = self._merge_new_child(value)
            setattr(node, actual_attr_name, value)

    def _merge_new_child(self, child_node: RelNode) -> RelNode:
        """Merges a new child node into the tree update process.

        This ensures that all child nodes of the new node correctly reference their updated counterparts.

        Parameters
        ----------
        child_node : RelNode
            The new child node to merge. Must be the *original* version of the child node.

        Returns
        -------
        RelNode
            The updated child node
        """
        internal_node_working_set: list[RelNode] = list(_collect_leaf_nodes(child_node))

        updated_child: RelNode

        while internal_node_working_set:
            current_node = internal_node_working_set.pop(0)
            already_updated = current_node in self._updated_nodes
            if already_updated and current_node == child_node:
                updated_child = self._updated_nodes[current_node]
                break
            if already_updated:
                self._update_node_working_set(current_node, working_set=internal_node_working_set)
                continue
            if any(child not in self._updated_nodes for child in current_node.children()):
                continue

            updated_node = current_node.clone()
            updated_node._update_child_nodes([self._updated_nodes[child] for child in current_node.children()])
            updated_node._clear_parent_links()

            self._updated_nodes[current_node] = updated_node
            if current_node == child_node:
                updated_child = updated_node
                break

            self._update_node_working_set(current_node, working_set=internal_node_working_set)

        assert updated_child is not None
        return updated_child

    def _update_node_working_set(self, node: RelNode, *, working_set: list[RelNode] | None = None) -> None:
        """Utility method to quickly include all relevant parent nodes into a node working set.

        Parameters
        ----------
        node : RelNode
            The node whose upwards links should be added to the working set
        working_set : Optional[list[RelNode]], optional
            The working set to update. If *None*, the internal working set of the update manager is used.
        """
        working_set = self._node_working_set if working_set is None else working_set
        if node.parent_node:
            working_set.append(node.parent_node)
        working_set.extend(node.sideways_pass)


def _is_aggregation(expression: SqlExpression) -> bool:
    """Utility to check whether an arbitrary SQL expression is an aggregation function.

    Parameters
    ----------
    expression : SqlExpression
        The expression to check

    Returns
    -------
    bool
        *True* if the expression is an aggregation or *False* otherwise
    """
    return isinstance(expression, FunctionExpression) and expression.is_aggregate()


def _requires_aggregation(expression: SqlExpression) -> bool:
    """Checks, whether the current expression or any of its nested children aggregate input tuples.

    Parameters
    ----------
    expression : SqlExpression
        The expression to check

    Returns
    -------
    bool
        *True* if an aggregation was detected or *False* otherwise.
    """
    return any(
        _is_aggregation(child_expr) or _requires_aggregation(child_expr) for child_expr in expression.iterchildren()
    )


def _needs_mapping(expression: SqlExpression) -> bool:
    """Checks, whether an expression has to be calculated via a mapping or can be supplied directly by the execution engine.

    The latter case basically only applies to static values. Direct column expressions are still considered as requiring a
    mapping.

    Parameters
    ----------
    expression : SqlExpression
        The expression to check

    Returns
    -------
    bool
        *True* if the expression has to be mapped, *False* otherwise.
    """
    return not isinstance(expression, (StaticValueExpression, StarExpression))


def _generate_expression_mapping_dict(
    expressions: list[SqlExpression],
) -> dict[frozenset[SqlExpression], frozenset[SqlExpression]]:
    """Determines all required expressions and maps them to their dervied expressions.

    Consider an expression *CAST(R.a + 42 AS int)*. In order to evaluate the *CAST* statement, *R.a + 42* has to be calculated
    first. This knowledge is encoded in the mapping dictionary, which would look like ``{R.a + 42: CAST(...)}`` in this case.

    Notice that this process is not recursive, i.e. nested expressions in the child expressions are not considered. The
    reasoning behind this is that these expressions should have been computed already via earlier mappings. Continuing the
    above example, there is no entry ``R.a: R.a + 42`` in the mapping, if the term *R.a + 42* itself is not part of the
    arguments.

    Parameters
    ----------
    expressions : list[SqlExpression]
        The expressions to resolve

    Returns
    -------
    dict[frozenset[SqlExpression], frozenset[SqlExpression]]
        A map from arguments to target expressions. If the same set of arguments is used to derive multiple expressions, all
        these target expressions are contained in the dictionary value.
    """
    mapping: dict[frozenset[SqlExpression], set[SqlExpression]] = collections.defaultdict(set)
    for expression in expressions:
        child_expressions = frozenset(
            child_expr for child_expr in expression.iterchildren() if _needs_mapping(child_expr)
        )
        mapping[child_expressions].add(expression)
    return {
        child_expressions: frozenset(derived_expressions) for child_expressions, derived_expressions in mapping.items()
    }


class EvaluationPhase(enum.IntEnum):
    """Indicates when a specific expression or predicate can be evaluated at the earliest."""

    BaseTable = enum.auto()
    """Evaluation is possible using only the tuples from the base table, e.g. base table filters."""

    Join = enum.auto()
    """Evaluation is possible while joining the required base tables, e.g. join predicates."""

    PostJoin = enum.auto()
    """Evaluation is possible once all base tables have been joined, e.g. mappings over joined columns."""

    PostAggregation = enum.auto()
    """Evaluation is possible once all aggregations have been performed, e.g. filters over aggregated columns."""


@dataclasses.dataclass(frozen=True)
class _SubquerySet:
    """More expressive wrapper to collect subqueries from SQL queries.

    Two subquery sets can be merged using the addition operator. Boolean tests on the subquery set succeed, if the set contains
    at least one subquery.

    Attributes
    ----------
    subqueries : frozenset[SqlQuery]
        The subqueries that are currently in the set. Can be empty if there are no subqueries.
    """

    subqueries: frozenset[SelectStatement]

    @staticmethod
    def empty() -> _SubquerySet:
        """Generates a new subquery set without any entries."""
        return _SubquerySet(frozenset())

    @staticmethod
    def of(subqueries: SelectStatement | Iterable[SelectStatement]) -> _SubquerySet:
        """Generates a new subquery set containing specific subqueries.

        This factory handles the generation of an appropriate frozenset.
        """
        subqueries = util.enlist(subqueries)
        return _SubquerySet(frozenset(subqueries))

    def __add__(self, other: _SubquerySet) -> _SubquerySet:
        if not isinstance(other, type(self)):
            return NotImplemented
        return _SubquerySet(self.subqueries | other.subqueries)

    def __bool__(self) -> bool:
        return bool(self.subqueries)


class _SubqueryDetector(SqlExpressionVisitor[_SubquerySet], PredicateVisitor[_SubquerySet]):
    """Collects all subqueries from SQL expressions or predicates."""

    def visit_and_predicate(self, predicate: AndPredicate, *args, **kwargs) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_or_predicate(self, predicate: OrPredicate, *args, **kwargs) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_not_predicate(self, predicate: NotPredicate, *args, **kwargs) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_binary_predicate(self, predicate: BinaryPredicate, *args, **kwargs) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_between_predicate(self, predicate: BetweenPredicate, *args, **kwargs) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_in_predicate(self, predicate: InPredicate, *args, **kwargs) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_unary_predicate(self, predicate: UnaryPredicate, *args, **kwargs) -> _SubquerySet:
        return self._traverse_predicate_expressions(predicate)

    def visit_static_value_expr(self, expression: StaticValueExpression, *args, **kwargs) -> _SubquerySet:
        return _SubquerySet.empty()

    def visit_column_expr(self, expression: ColumnExpression, *args, **kwargs) -> _SubquerySet:
        return _SubquerySet.empty()

    def visit_cast_expr(self, expression: CastExpression, *args, **kwargs) -> _SubquerySet:
        return self._traverse_nested_expressions(expression)

    def visit_function_expr(self, expression: FunctionExpression, *args, **kwargs) -> _SubquerySet:
        return self._traverse_nested_expressions(expression)

    def visit_math_expr(self, expression: MathExpression, *args, **kwargs) -> _SubquerySet:
        return self._traverse_nested_expressions(expression)

    def visit_star_expr(self, expression: StarExpression, *args, **kwargs) -> _SubquerySet:
        return _SubquerySet.empty()

    def visit_subquery_expr(self, expression: SubqueryExpression, *args, **kwargs) -> _SubquerySet:
        return _SubquerySet.of(expression.query)

    def visit_window_expr(self, expression: WindowExpression, *args, **kwargs) -> _SubquerySet:
        return self._traverse_nested_expressions(expression)

    def visit_case_expr(self, expression: CaseExpression, *args, **kwargs) -> _SubquerySet:
        return self._traverse_nested_expressions(expression)

    def visit_quantifier_expr(self, expression: QuantifierExpression, *args, **kwargs) -> _SubquerySet:
        return self._traverse_nested_expressions(expression)

    def visit_array_expr(self, expression: ArrayExpression, *args, **kwargs) -> _SubquerySet:
        return self._traverse_nested_expressions(expression)

    def visit_predicate_expr(self, expression: AbstractPredicate, *args, **kwargs) -> _SubquerySet:
        return self._traverse_nested_expressions(expression)

    def _traverse_predicate_expressions(self, predicate: AbstractPredicate) -> _SubquerySet:
        """Handler to collect subqueries from predicates."""
        return functools.reduce(
            operator.add,
            [expression.accept_visitor(self) for expression in predicate.iterexpressions()],
        )

    def _traverse_nested_expressions(self, expression: SqlExpression) -> _SubquerySet:
        """Handler to collect subqueries from SQL expressions."""
        return functools.reduce(
            operator.add,
            [nested_expression.accept_visitor(self) for nested_expression in expression.iterchildren()],
        )


class _BaseTableLookup(SqlExpressionVisitor[TableReference | None], PredicateVisitor[TableReference]):
    """Handler to determine the base table in an arbitrarily deep predicate or expression hierarchy.

    This service is designed to traverse filter predicates or expressions operating on a single base table and provides exactly
    this table. As a special case, it also traverses dependent subqueries. In this case, the base table is the outer table.

    The lookup may be started directly by calling the instantiated service with a predicate or an expression as argument.

    Notes
    -----
    In cases where multiple applicable base tables are detected, a ``ValueError`` is raised. Therefore, using the lookup on a
    predicate is always guaranteed to provide a table (or raise an error) since all predicates operate on tables. On the other
    hand, checking an arbitrary SQL expression may or may not contain a base table (e.g. *CAST(42 AS float)*). Hence, an
    optional is returned for expressions.
    """

    def visit_and_predicate(self, predicate: AndPredicate, *args, **kwargs) -> TableReference:
        base_tables = {child_pred.accept_visitor(self) for child_pred in predicate.children}
        return self._fetch_valid_base_tables(base_tables)

    def visit_or_predicate(self, predicate: OrPredicate, *args, **kwargs) -> TableReference:
        base_tables = {child_pred.accept_visitor(self) for child_pred in predicate.children}
        return self._fetch_valid_base_tables(base_tables)

    def visit_not_predicate(self, predicate: NotPredicate, *args, **kwargs) -> TableReference:
        return predicate.child.accept_visitor(self)

    def visit_binary_predicate(self, predicate: BinaryPredicate, *args, **kwargs) -> TableReference:
        base_tables = (
            predicate.lhs.accept_visitor(self),
            predicate.rhs.accept_visitor(self),
        )
        return self._fetch_valid_base_tables(set(base_tables))

    def visit_between_predicate(self, predicate: BetweenPredicate, *args, **kwargs) -> TableReference:
        base_tables = (
            predicate.column.accept_visitor(self),
            predicate.lower.accept_visitor(self),
            predicate.upper.accept_visitor(self),
        )
        return self._fetch_valid_base_tables(set(base_tables))

    def visit_in_predicate(self, predicate: InPredicate, *args, **kwargs) -> TableReference:
        base_tables = {predicate.column.accept_visitor(self)}
        base_tables |= {val.accept_visitor(self) for val in predicate.values}
        return self._fetch_valid_base_tables(base_tables)

    def visit_unary_predicate(self, predicate: UnaryPredicate, *args, **kwargs) -> TableReference:
        return predicate.expression.accept_visitor(self)

    def visit_static_value_expr(self, expression: StaticValueExpression, *args, **kwargs) -> TableReference | None:
        return None

    def visit_column_expr(self, expression: ColumnExpression, *args, **kwargs) -> TableReference | None:
        return expression.column.table

    def visit_cast_expr(self, expression: CastExpression, *args, **kwargs) -> TableReference | None:
        return expression.casted_expression.accept_visitor(self)

    def visit_function_expr(self, expression: FunctionExpression, *args, **kwargs) -> TableReference | None:
        referenced_tables = {argument.accept_visitor(self) for argument in expression.arguments}
        return self._fetch_valid_base_tables(referenced_tables, accept_empty=True)

    def visit_math_expr(self, expression: MathExpression, *args, **kwargs) -> TableReference | None:
        base_tables = {child.accept_visitor(self) for child in expression.iterchildren()}
        return self._fetch_valid_base_tables(base_tables)

    def visit_star_expr(self, expression: StarExpression, *args, **kwargs) -> TableReference | None:
        return expression.from_table

    def visit_subquery_expr(self, expression: SubqueryExpression, *args, **kwargs) -> TableReference | None:
        subquery = expression.query
        if not subquery.is_dependent():
            return None
        dependent_tables = subquery.unbound_tables()
        return self._fetch_valid_base_tables(dependent_tables, accept_empty=True)

    def visit_window_expr(self, expression: WindowExpression, *args, **kwargs) -> TableReference | None:
        # base tables can only appear in predicates and window functions are limited to SELECT statements
        return None

    def visit_case_expr(self, expression: CaseExpression, *args, **kwargs) -> TableReference | None:
        referenced_tables = {child.accept_visitor(self) for child in expression.iterchildren()}
        return self._fetch_valid_base_tables(referenced_tables, accept_empty=True)

    def visit_quantifier_expr(self, expr: QuantifierExpression, *args, **kwargs) -> TableReference | None:
        return expr.expression.accept_visitor(self)

    def visit_array_expr(self, expression: ArrayExpression, *args, **kwargs) -> TableReference | None:
        referenced_tables = {element.accept_visitor(self) for element in expression.elements}
        return self._fetch_valid_base_tables(referenced_tables, accept_empty=True)

    def visit_predicate_expr(self, expression: AbstractPredicate, *args, **kwargs) -> TableReference:
        return expression.accept_visitor(self)

    def _fetch_valid_base_tables(
        self, base_tables: set[TableReference | None], *, accept_empty: bool = False
    ) -> TableReference | None:
        """Handler to extract the actual base table from a set of candidate tables.

        Parameters
        ----------
        base_tables : set[TableReference  |  None]
            The candidate tables. Usually, this should be a set containing exactly one base table and potentially an
            additional *None* value. In all other situations an error is raised (see below)
        accept_empty : bool, optional
            Whether an empty set of actual candidate tables (i.e. excluding *None* values) is an acceptable argument. If that
            is the case, *None* is returned in such a situation. Empty candidate sets raise an error by default.

        Returns
        -------
        Optional[TableReference]
            The base table

        Raises
        ------
        ValueError
            If the candidate tables either contain more than one actual table instance, or if empty candidate sets are not
            accepted and the set does not contain any *non-None* enties.
        """
        if None in base_tables:
            base_tables.remove(None)
        if len(base_tables) != 1 or (accept_empty and not base_tables):
            raise ValueError(f"Expected exactly one base predicate but found {base_tables}")
        return util.simplify(base_tables) if base_tables else None

    def __call__(self, elem: AbstractPredicate | SqlExpression) -> TableReference:
        if isinstance(elem, AbstractPredicate) and elem.is_join():
            raise ValueError(f"Cannot determine base table for join predicate '{elem}'")
        tables = elem.tables()
        if len(tables) == 1:
            return util.simplify(tables)
        base_table = elem.accept_visitor(self)
        if base_table is None:
            raise ValueError(f"No base table found in '{elem}'")
        return base_table


def _collect_all_expressions(
    expression: SqlExpression, *, traverse_aggregations: bool = False
) -> frozenset[SqlExpression]:
    """Provides all expressions in a specific expression tree, including the root expression.

    Parameters
    ----------
    expression : SqlExpression
        The root expression
    traverse_aggregations : bool, optional
        Whether expressions nested in aggregation functions should be included. Disabled by default.

    Returns
    -------
    frozenset[SqlExpression]
        The expression as well as all child expressions, including deeply nested children.
    """
    if not traverse_aggregations and isinstance(expression, FunctionExpression) and expression.is_aggregate():
        return frozenset({expression})
    child_expressions = util.set_union(_collect_all_expressions(child_expr) for child_expr in expression.iterchildren())
    all_expressions = frozenset({expression} | child_expressions)
    return frozenset({expression for expression in all_expressions if _needs_mapping(expression)})


def _determine_expression_phase(expression: SqlExpression) -> EvaluationPhase:
    """Calculates the evaluation phase during which an expression can be evaluated at the earliest."""
    match expression:
        case ColumnExpression() | StaticValueExpression():
            return EvaluationPhase.BaseTable
        case FunctionExpression() if expression.is_aggregate():
            return EvaluationPhase.PostAggregation
        case (
            FunctionExpression()
            | MathExpression()
            | CastExpression()
            | CaseExpression()
            | QuantifierExpression()
            | ArrayExpression()
            | AbstractPredicate()
        ):
            own_phase = EvaluationPhase.Join if len(expression.tables()) > 1 else EvaluationPhase.BaseTable
            child_phase = max(_determine_expression_phase(child_expr) for child_expr in expression.iterchildren())
            return max(own_phase, child_phase)
        case SubqueryExpression():
            return EvaluationPhase.BaseTable if len(expression.query.unbound_tables()) < 2 else EvaluationPhase.PostJoin
        case StarExpression() | StaticValueExpression():
            # TODO: should we rather raise an error in this case?
            return EvaluationPhase.BaseTable
        case WindowExpression():
            # these expressions can currently only appear within SELECT clauses
            return EvaluationPhase.PostAggregation
        case _:
            raise ValueError(f"Unknown expression type: '{expression}'")


def _determine_predicate_phase(predicate: AbstractPredicate) -> EvaluationPhase:
    """Calculates the evaluation phase during which a predicate can be evaluated at the earliest.

    See Also
    --------
    _determine_expression_phase
    """
    nested_subqueries = predicate.accept_visitor(_SubqueryDetector())
    subquery_tables = len(util.set_union(subquery.bound_tables() for subquery in nested_subqueries.subqueries))
    n_tables = len(predicate.tables()) - subquery_tables
    if n_tables == 1:
        # It could actually be that the number of tables is negative. E.g. HAVING count(*) < (SELECT min(r_a) FROM R)
        # Therefore, we only check for exactly 1 table
        return EvaluationPhase.BaseTable
    if subquery_tables:
        # If there are subqueries and multiple base tables present, we encoutered a predicate like
        # R.a = (SELECT min(S.b) FROM S WHERE S.b = T.c) with a dependent subquery on T. By default, such predicates should be
        # executed after the join phase.
        return EvaluationPhase.PostJoin

    expression_phase = max(
        _determine_expression_phase(expression)
        for expression in predicate.iterexpressions()
        if type(expression) not in {StarExpression, StaticValueExpression}
    )
    if expression_phase > EvaluationPhase.Join:
        return expression_phase

    return EvaluationPhase.Join if isinstance(predicate, BinaryPredicate) else EvaluationPhase.PostJoin


def _filter_eval_phase(predicate: AbstractPredicate, expected_eval_phase: EvaluationPhase) -> AbstractPredicate | None:
    """Provides all parts of predicate that can be evaluated during a specific logical query execution phase.

    The following rules are used to determine matching (sub-)predicates:

    - For base predicates, either the entire matches the expected evaluation phase, or none at all.
    - For conjunctive predicates, the matching parts are combined into a smaller conjunction. If no part matches, *None* is
      returned.
    - For disjunctive predicates or negations, the same rules as for base predicates are used: either the entire predicate
      matches, or nothing does.

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate to check
    expected_eval_phase : EvaluationPhase
        The desired evaluation phase

    Returns
    -------
    Optional[AbstractPredicate]
        A predicate composed of the matching (sub-) predicates, or *None* if there is no match whatsoever.

    See Also
    --------
    _determine_predicate_phase
    """
    eval_phase = _determine_predicate_phase(predicate)
    if eval_phase < expected_eval_phase:
        return None

    if isinstance(predicate, AndPredicate):
        child_predicates = [
            child for child in predicate.children if _determine_predicate_phase(child) == expected_eval_phase
        ]
        return CompoundPredicate.create_and(child_predicates) if child_predicates else None

    return predicate if eval_phase == expected_eval_phase else None


class _ImplicitRelalgParser:
    """Parser to generate a `RelNode` tree from `SqlQuery` instances.

    Parameters
    ----------
    query : SqlQuery
        The query to parse
    provided_base_tables : Optional[dict[TableReference, RelNode]], optional
        When parsing subqueries, these are the tables that are provided by the outer query and their corresponding relational
        algebra fragments.

    Notes
    -----
    Our parser operates in four strictly sequential stages. These stages approximately correspond to the logical evaluation
    stages of an SQL query. For reference, see [eder-sql-eval-order]_. At the same time, the entire process is loosely oriented
    on the query execution strategy applied by PostgreSQL.

    During the initial stage, all base tables are processed. This includes all filters that may be executed directly on the
    base table, as well as mappings that enable the filters, e.g. for predicates such as *CAST(R.a AS int) = 42*. As a special
    case, we also handle *EXISTS* and *MISSING* predicates during this stage. The end result of the initial stage is a
    dictionary that maps each base table to a relational algebra fragment corresponds to the scan of the base table as well as
    all filters, etc..

    Secondly, all joins are processed. This process builds on the initial dictionary of base tables and iteratively combines
    pairs of fragments according to their join predicate. In the end, usually just a single fragment remains. This fragment has
    exactly one root node that corresponds to the final joined relation. If there are multiple fragments (and hence root nodes)
    remaining, we need to use cross products to ensure that we just have a single relation in the end. A post-processing step
    applies all filter predicates that were not recognized as joins but that require columns from multiple base tables, e.g.
    *R.a + S.b < 42*.

    The third stage is concerned with grouping and aggregation. If the query contains aggregates or a *GROUP BY* clause, these
    operations are now inserted. Since we produced just a single root node during the second stage, our grouping uses this
    node as input. Once again, during a post-processing step all filter predicates from the *HAVING* clause are inserted as
    selections.

    The fourth and final phase executes all "cleanup" actions such as sorting, duplicate removal and projection.

    Notice that during each stage, additional mapping steps can be required if some expression requires input that has not
    yet been calculated.

    References
    ----------

    .. [eder-sql-eval-oder]_ https://blog.jooq.org/a-beginners-guide-to-the-true-order-of-sql-operations/

    """

    def __init__(
        self,
        query: SelectStatement,
        *,
        provided_base_tables: dict[TableReference, RelNode] | None = None,
    ) -> None:
        self._query = query
        self._base_table_fragments: dict[TableReference, RelNode] = {}
        self._required_columns: dict[TableReference, set[ColumnReference]] = collections.defaultdict(set)
        self._provided_base_tables: dict[TableReference, RelNode] = provided_base_tables if provided_base_tables else {}

        if query:
            query_cols = self._query.columns()
            util.collections.foreach(query_cols, lambda col: self._required_columns[col.table].add(col))

    def generate_relnode(self) -> RelNode:
        """Produces a relational algebra tree for the current query.

        Returns
        -------
        RelNode
            Root node of the algebraic expression
        """
        if isinstance(self._query, SetQuery):
            return self._parse_set_query(self._query)

        # TODO: robustness: query without FROM clause

        if self._query.cte_clause:
            for cte in self._query.cte_clause.queries:
                cte_root = self._add_subquery(cte.query)
                self._add_table(cte.target_table, input_node=cte_root)

        # we add the WHERE clause before all explicit JOIN statements to make sure filters are already present and we can
        # stitch together the correct fragments in OUTER JOINs
        # Once the explicit JOINs have been processed, we continue with all remaining implicit joins
        # TODO: since the implementation of JOIN statements is currently undergoing a major rework, we don't process such
        # statements at all

        util.collections.foreach(self._query.from_clause.items, self._add_table_source)

        if self._query.where_clause:
            self._add_predicate(self._query.where_clause.root, eval_phase=EvaluationPhase.BaseTable)

        final_fragment = self._generate_initial_join_order()

        if self._query.where_clause:
            # add all post-join filters here
            final_fragment = self._add_predicate(
                self._query.where_clause.root,
                input_node=final_fragment,
                eval_phase=EvaluationPhase.PostJoin,
            )

        final_fragment = self._add_aggregation(final_fragment)
        if self._query.having_clause:
            final_fragment = self._add_predicate(
                self._query.having_clause.condition,
                input_node=final_fragment,
                eval_phase=EvaluationPhase.PostAggregation,
            )

        if self._query.orderby_clause:
            final_fragment = self._add_ordering(self._query.orderby_clause.expressions, input_node=final_fragment)

        final_fragment = self._add_final_projection(final_fragment)
        return final_fragment

    def _update_query(self, query: SqlQuery) -> None:
        self._query = query
        query_cols = self._query.columns()
        util.collections.foreach(query_cols, lambda col: self._required_columns[col.table].add(col))

    def _parse_set_query(self, query: SetQuery) -> RelNode:
        """Handler method to translate a set query into a relational algebra fragment.

        Parameters
        ----------
        query : SetQuery
            _description_

        Returns
        -------
        RelNode
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        parser = _ImplicitRelalgParser(None)

        if query.cte_clause:
            for cte in query.cte_clause.queries:
                parser._update_query(cte.query)
                parser.generate_relnode()  # we don't care about the result, we just want to add the CTE to the base tables

        parser._update_query(query.lhs)
        left_relalg = parser.generate_relnode()
        parser._update_query(query.rhs)
        right_relalg = parser.generate_relnode()

        match query.set_operation:
            case SetOperator.Union:
                final_fragment = Union(left_relalg, right_relalg)
                final_fragment = DuplicateElimination(final_fragment)
            case SetOperator.UnionAll:
                final_fragment = Union(left_relalg, right_relalg)
            case SetOperator.Intersect:
                final_fragment = Intersection(left_relalg, right_relalg)
            case SetOperator.Except:
                final_fragment = Difference(left_relalg, right_relalg)
            case _:
                raise ValueError(f"Unknown set operation: '{query.set_operation}'")

        if query.orderby_clause:
            final_fragment = self._add_ordering(query.orderby_clause.expressions, input_node=final_fragment)

        return final_fragment

    def _resolve(self, table: TableReference) -> RelNode:
        """Provides the algebra fragment for a specific base table, resorting to outer query tables if necessary."""
        if table in self._base_table_fragments:
            return self._base_table_fragments[table]
        return self._provided_base_tables[table]

    def _add_table(self, table: TableReference, *, input_node: RelNode | None = None) -> RelNode:
        """Generates and stores a new base table relation node for a specific table.

        The relation will be stored in `self._base_table_fragments`.

        Parameters
        ----------
        table : TableReference
            The base table
        input_node : Optional[RelNode], optional
            If the base table corresponds to a subquery or CTE target, this is the root node of the fragment that computes the
            actual subquery.

        Returns
        -------
        RelNode
            A relational algebra fragment
        """
        required_cols = self._required_columns[table]
        table_node = Relation(table, required_cols, subquery_input=input_node)
        self._base_table_fragments[table] = table_node
        return table_node

    def _add_table_source(self, table_source: TableSource) -> RelNode:
        """Generates the appropriate algebra fragment for a specific table source.

        The fragment will be stored in `self._base_table_fragments`.
        """
        match table_source:
            case DirectTableSource():
                if table_source.table.virtual:
                    # Virtual tables in direct table sources are only created through references to CTEs. However, these CTEs
                    # have already been included in the base table fragments.
                    return self._base_table_fragments[table_source.table]
                return self._add_table(table_source.table)
            case SubqueryTableSource(subquery, alias):
                if alias is None:
                    raise ValueError(f"Subqueries without an alias cannot be converted: {self._query}")
                subquery_root = self._add_subquery(subquery)
                self._base_table_fragments[alias] = subquery_root
                return self._add_table(alias, input_node=subquery_root)
            case JoinTableSource():
                raise ValueError(f"Explicit JOIN syntax is currently not supported: '{table_source}'")
            case _:
                raise ValueError(f"Unknown table source: '{table_source}'")

    def _generate_initial_join_order(self) -> RelNode:
        """Combines all base table fragments to generate a single root relation corresponding to the join of all fragments.

        If necessary, fragments are combined via cross products.

        Returns
        -------
        RelNode
            The root operator corresponding to the relation that joins all base table fragments.
        """
        # TODO: figure out the interaction between implicit and explicit joins, especially regarding their timing

        joined_tables: set[TableReference] = set()
        for table_source in self._query.from_clause.items:
            # TODO: determine correct join partners for explicit JOINs
            joined_tables |= table_source.tables()

        if self._query.where_clause:
            self._add_predicate(self._query.where_clause.root, eval_phase=EvaluationPhase.Join)

        head_nodes = set(self._base_table_fragments.values())
        if len(head_nodes) == 1:
            return util.simplify(head_nodes)

        current_head, *remaining_nodes = head_nodes
        for remaining_node in remaining_nodes:
            current_head = CrossProduct(current_head, remaining_node)
        return current_head

    def _add_aggregation(self, input_node: RelNode) -> RelNode:
        """Generates all necesssary aggregation operations for the SQL query.

        If there are no necessary aggregations, the current algebra tree is returned unmodified.

        Parameters
        ----------
        input_node : RelNode
            The root of the current algebra tree

        Returns
        -------
        RelNode
            The algebra tree, potentially expanded by grouping, mapping and selection nodes.
        """
        aggregation_collector = ExpressionCollector(lambda e: isinstance(e, FunctionExpression) and e.is_aggregate())
        aggregation_functions: set[FunctionExpression] = util.set_union(
            select_expr.accept_visitor(aggregation_collector)
            for select_expr in self._query.select_clause.iterexpressions()
        )

        if self._query.having_clause:
            aggregation_functions |= util.set_union(
                having_expr.accept_visitor(aggregation_collector)
                for having_expr in self._query.having_clause.iterexpressions()
            )
        if not self._query.groupby_clause and not aggregation_functions:
            return input_node

        aggregation_arguments: set[SqlExpression] = set()
        for agg_func in aggregation_functions:
            aggregation_arguments |= util.set_union(
                _collect_all_expressions(arg, traverse_aggregations=True) for arg in agg_func.arguments
            )
        missing_expressions = aggregation_arguments - input_node.provided_expressions()
        if missing_expressions:
            input_node = Map(input_node, _generate_expression_mapping_dict(missing_expressions))

        group_cols = self._query.groupby_clause.group_columns if self._query.groupby_clause else []
        aggregates: dict[frozenset[SqlExpression], set[FunctionExpression]] = collections.defaultdict(set)
        for agg_func in aggregation_functions:
            aggregates[agg_func.arguments].add(agg_func)
        groupby_node = Grouping(
            input_node,
            group_columns=group_cols,
            aggregates={agg_input: frozenset(agg_funcs) for agg_input, agg_funcs in aggregates.items()},
        )
        return groupby_node

    def _add_final_projection(self, input_node: RelNode) -> RelNode:
        """Generates all necessary output preparation nodes.

        Such nodes include the final projection, sorting, duplicate elimination or limit.

        Parameters
        ----------
        input_node : RelNode
            The root of the current algebra tree

        Returns
        -------
        RelNode
            The algebra tree, potentially expanded by some final nodes.
        """
        # TODO: Sorting, Duplicate elimination, limit
        if self._query.select_clause.is_star():
            return input_node
        required_expressions = util.set_union(
            _collect_all_expressions(target.expression) for target in self._query.select_clause.targets
        )
        missing_expressions = required_expressions - input_node.provided_expressions()
        final_node = (
            Map(input_node, _generate_expression_mapping_dict(missing_expressions))
            if missing_expressions
            else input_node
        )
        return Projection(
            final_node,
            [target.expression for target in self._query.select_clause.targets],
        )

    def _add_ordering(self, ordering: Sequence[OrderByExpression], *, input_node: RelNode) -> RelNode:
        sorting: list[tuple[SqlExpression, SortDirection]] = []
        final_fragment = input_node

        for order in ordering:
            final_fragment = self._add_expression(order.column, input_node=final_fragment)
            sorting.append((order.column, "asc" if order.ascending else "desc"))

        final_fragment = Sort(final_fragment, sorting)
        return final_fragment

    def _add_predicate(
        self,
        predicate: AbstractPredicate,
        *,
        input_node: RelNode | None = None,
        eval_phase: EvaluationPhase = EvaluationPhase.BaseTable,
    ) -> RelNode:
        """Inserts a selection into the corresponding relational algebra fragment.

        Parameters
        ----------
        predicate : AbstractPredicate
            The entire selection. Notice that only those parts of the predicate will be included in the selection, that match
            the expected `eval_phase`.
        input_node : Optional[RelNode], optional
            The current fragment. For the base table evaluation phase this can be *None*. The actual base table will be
            inferred from the predicate. For all other evaluation phases this parameter is required.
        eval_phase : EvaluationPhase, optional
            The current evaluation phase, by default `EvaluationPhase.BaseTable`

        Returns
        -------
        RelNode
            The complete fragment. The return value is mostly only interesting for evaluation phases later than the base table
            evaluation. For the base table evaluation, the result will also be stored in `self._base_table_fragments`.

        See Also
        --------
        _filter_eval_phase
        """
        predicate = _filter_eval_phase(predicate, eval_phase)
        if predicate is None:
            return input_node

        match eval_phase:
            case EvaluationPhase.BaseTable:
                for base_table, base_pred in self._split_filter_predicate(predicate).items():
                    base_table_fragment = self._convert_predicate(
                        base_pred, input_node=self._base_table_fragments[base_table]
                    )
                    self._base_table_fragments[base_table] = base_table_fragment
                return base_table_fragment
            case EvaluationPhase.Join:
                for join_predicate in self._split_join_predicate(predicate):
                    join_node = self._convert_join_predicate(join_predicate)
                    for outer_table in join_node.tables():
                        self._base_table_fragments[outer_table] = join_node
                return join_node
            case EvaluationPhase.PostJoin | EvaluationPhase.PostAggregation:
                assert input_node is not None
                # Generally speaking, when consuming a post-join predicate, all required tables should be available by now.
                # However, there is one caveat: a complex predicate that can currently only be executed after the join phase
                # (e.g. disjunctions of join predicates) could contain a correlated scalar subquery. In this case, some of the
                # required tables might not be available yet (more precisely, the native tables from the depedent subquery
                # are not available). We solve this situation by introducing cross products between the dependent subquery
                # and the outer table _within the subquery_. This is because the subquery needs to reference the outer table
                # in its join predicate.
                if not predicate.tables().issubset(input_node.tables()):
                    missing_tables = predicate.tables() - input_node.tables()
                    for outer_table in missing_tables:
                        if outer_table not in self._provided_base_tables:
                            # the table will be supplied by a subquery
                            continue
                        input_node = CrossProduct(input_node, self._provided_base_tables[outer_table])
                return self._convert_predicate(predicate, input_node=input_node)
            case _:
                raise ValueError(f"Unknown evaluation phase '{eval_phase}' for predicate '{predicate}'")

    def _convert_predicate(self, predicate: AbstractPredicate, *, input_node: RelNode) -> RelNode:
        """Generates the appropriate selection nodes for a specific predicate.

        Depending on the specific predicate, operations other than a plain old selection might be required. For example,
        for disjunctions involving subqueries, a union is necessary. Therefore, the conversion might actually force a deviation
        from pure algebra trees and require an directed, acyclic graph instead.

        Likewise, applying a predicate can make a preparatory mapping operation necessary, if the expressions required by the
        predicate are not yet produced by the input node.

        Parameters
        ----------
        predicate : AbstractPredicate
            The predicate that should be converted
        input_node : RelNode
            The operator after which the predicate is required. It is assumed that the input node is actually capable of
            producing the required attributes in order to apply the predicate. For example, if the predicate consumes
            attributes from multiple base relations, it is assumed that the input node provides tuples that already contain
            these nodes.

        Returns
        -------
        RelNode
            The algebra fragment
        """
        contains_subqueries = _SubqueryDetector()
        final_fragment = input_node

        if isinstance(predicate, UnaryPredicate) and not predicate.accept_visitor(contains_subqueries):
            final_fragment = self._ensure_predicate_applicability(predicate, final_fragment)
            final_fragment = Selection(final_fragment, predicate)
            return final_fragment
        elif isinstance(predicate, UnaryPredicate):
            subquery_target = "semijoin" if predicate.operator == UnaryOperator.Exists else "antijoin"
            return self._add_expression(
                predicate.expression,
                input_node=final_fragment,
                subquery_target=subquery_target,
            )

        if isinstance(predicate, BetweenPredicate) and not predicate.accept_visitor(contains_subqueries):
            final_fragment = self._ensure_predicate_applicability(predicate, final_fragment)
            final_fragment = Selection(final_fragment, predicate)
            return final_fragment
        elif isinstance(predicate, BetweenPredicate):
            # BETWEEN predicate with scalar subquery
            final_fragment = self._add_expression(predicate.column, input_node=final_fragment)
            final_fragment = self._add_expression(predicate.lower, input_node=final_fragment)
            final_fragment = self._add_expression(predicate.upper, input_node=final_fragment)
            final_fragment = Selection(final_fragment, predicate)
            return final_fragment

        if isinstance(predicate, InPredicate) and not predicate.accept_visitor(contains_subqueries):
            # we need to determine the required expressions due to IN predicates like "r_a + 42 IN (1, 2, 3)"
            # or "r_a IN (r_b + 42, 42)"
            final_fragment = self._ensure_predicate_applicability(predicate, final_fragment)
            final_fragment = Selection(final_fragment, predicate)
            return final_fragment
        elif isinstance(predicate, InPredicate):
            # TODO: test weird IN predicates like r_a IN (1, 2, (SELECT min(...)), 4)
            # or even r_a IN ((SELECT r_a FROM ...) + (SELECT min(...)))
            pure_in_values: list[SqlExpression] = []
            subquery_in_values: list[tuple[SqlExpression, _SubquerySet]] = []
            for value in predicate.values:
                detected_subqueries = value.accept_visitor(contains_subqueries)
                if detected_subqueries and not all(subquery.is_scalar() for subquery in detected_subqueries.subqueries):
                    subquery_in_values.append((value, detected_subqueries))
                else:
                    final_fragment = self._add_expression(value, input_node=final_fragment)
                    pure_in_values.append(value)
            final_fragment = self._add_expression(predicate.column, input_node=final_fragment)
            if pure_in_values:
                reduced_predicate = InPredicate(predicate.column, pure_in_values)
                final_fragment = Selection(final_fragment, reduced_predicate)
            for subquery_value, _detected_subqueries in subquery_in_values:
                final_fragment = self._add_expression(
                    subquery_value,
                    input_node=final_fragment,
                    subquery_target="in",
                    in_column=predicate.column,
                )
            return final_fragment

        if isinstance(predicate, BinaryPredicate) and not predicate.accept_visitor(contains_subqueries):
            final_fragment = self._ensure_predicate_applicability(predicate, final_fragment)
            final_fragment = Selection(final_fragment, predicate)
            return final_fragment
        elif isinstance(predicate, BinaryPredicate):
            if predicate.lhs.accept_visitor(contains_subqueries):
                final_fragment = self._add_expression(
                    predicate.lhs,
                    input_node=final_fragment,
                    subquery_target="scalar",
                )
            if predicate.rhs.accept_visitor(contains_subqueries):
                final_fragment = self._add_expression(
                    predicate.rhs,
                    input_node=final_fragment,
                    subquery_target="scalar",
                )
            final_fragment = self._ensure_predicate_applicability(predicate, final_fragment)
            final_fragment = Selection(final_fragment, predicate)
            return final_fragment

        match predicate:
            case AndPredicate() | OrPredicate():
                regular_predicates: list[AbstractPredicate] = []
                subquery_predicates: list[AbstractPredicate] = []
                for child_pred in predicate.iterchildren():
                    if child_pred.accept_visitor(contains_subqueries):
                        subquery_predicates.append(child_pred)
                    else:
                        regular_predicates.append(child_pred)
                if regular_predicates:
                    simplified_composite = CompoundPredicate.create(predicate.operation, regular_predicates)
                    final_fragment = self._ensure_predicate_applicability(simplified_composite, final_fragment)
                    final_fragment = Selection(final_fragment, simplified_composite)
                for subquery_pred in subquery_predicates:
                    if predicate.operation == CompoundOperator.And:
                        final_fragment = self._convert_predicate(subquery_pred, input_node=final_fragment)
                        continue
                    subquery_branch = self._convert_predicate(subquery_pred, input_node=input_node)
                    final_fragment = Union(final_fragment, subquery_branch)
                return final_fragment

            case NotPredicate():
                if not predicate.child.accept_visitor(contains_subqueries):
                    final_fragment = self._ensure_predicate_applicability(predicate, final_fragment)
                    final_fragment = Selection(final_fragment, predicate)
                    return final_fragment
                subquery_branch = self._convert_predicate(predicate.child, input_node=input_node)
                final_fragment = Difference(final_fragment, subquery_branch)
                return final_fragment

            case _:
                raise ValueError(f"Unknown predicate type: '{predicate}'")

    def _convert_join_predicate(self, predicate: AbstractPredicate) -> RelNode:
        """Generates the appropriate join nodes for a specific predicate.

        Most of the implementation is structurally similar to `_convert_predicate`, so take a look at its documentation for
        details.

        See Also
        --------
        _ImplicitRelalgParser._convert_predicate
        """
        contains_subqueries = _SubqueryDetector()
        nested_subqueries = predicate.accept_visitor(contains_subqueries)
        subquery_tables = util.set_union(subquery.bound_tables() for subquery in nested_subqueries.subqueries)
        table_fragments = {self._resolve(join_partner) for join_partner in predicate.tables() - subquery_tables}
        if len(table_fragments) == 1:
            input_node = util.simplify(table_fragments)
            provided_expressions = self._collect_provided_expressions(input_node)
            required_expressions = util.set_union(_collect_all_expressions(e) for e in predicate.iterexpressions())
            missing_expressions = required_expressions - provided_expressions
            if missing_expressions:
                final_fragment = Map(input_node, _generate_expression_mapping_dict(missing_expressions))
            else:
                final_fragment = input_node
            return Selection(final_fragment, predicate)
        if len(table_fragments) != 2:
            raise ValueError(
                "Expected exactly two base table fragments for join predicate "
                f"'{predicate}', but found {table_fragments}"
            )

        required_expressions = util.set_union(_collect_all_expressions(e) for e in predicate.iterexpressions())
        if isinstance(predicate, BinaryPredicate):
            first_input, second_input = table_fragments
            first_arg, second_arg = predicate.lhs, predicate.rhs
            if first_arg.tables() <= first_input.tables(
                ignore_subqueries=True
            ) and second_arg.tables() <= second_input.tables(ignore_subqueries=True):
                left_input, right_input = first_input, second_input
            elif first_arg.tables() <= second_input.tables(
                ignore_subqueries=True
            ) and second_arg.tables() <= first_input.tables(ignore_subqueries=True):
                left_input, right_input = second_input, first_input
            else:
                raise ValueError(f"Unsupported join predicate '{predicate}'")

            left_input = self._add_expression(first_arg, input_node=left_input)
            right_input = self._add_expression(second_arg, input_node=right_input)

            provided_expressions = self._collect_provided_expressions(left_input, right_input)
            missing_expressions = required_expressions - provided_expressions
            left_mappings: list[SqlExpression] = []
            right_mappings: list[SqlExpression] = []
            for missing_expr in missing_expressions:
                if missing_expr.tables() <= left_input.tables():
                    left_mappings.append(missing_expr)
                elif missing_expr.tables() <= right_input.tables():
                    right_mappings.append(missing_expr)
                else:
                    raise ValueError(
                        "Cannot calculate expression on left or right input: "
                        f"'{missing_expr}' for predicate '{predicate}'"
                    )
            if left_mappings:
                left_input = Map(left_input, _generate_expression_mapping_dict(left_mappings))
            if right_mappings:
                right_input = Map(right_input, _generate_expression_mapping_dict(right_mappings))
            return ThetaJoin(left_input, right_input, predicate)

        match predicate:
            case AndPredicate() | OrPredicate():
                regular_predicates: list[AbstractPredicate] = []
                subquery_predicates: list[AbstractPredicate] = []
                for child_pred in predicate.children:
                    if predicate.accept_visitor(contains_subqueries):
                        subquery_predicates.append(child_pred)
                    else:
                        regular_predicates.append(child_pred)
                if regular_predicates:
                    simplified_composite = CompoundPredicate(predicate.operation, regular_predicates)
                    final_fragment = self._convert_join_predicate(simplified_composite)
                else:
                    first_input, second_input = table_fragments
                    final_fragment = CrossProduct(first_input, second_input)
                for subquery_pred in subquery_predicates:
                    final_fragment = self._convert_predicate(subquery_pred, input_node=final_fragment)
                return final_fragment

            case NotPredicate():
                pass

            case _:
                raise ValueError(
                    f"Unsupported join predicate '{predicate}'. Perhaps this should be a post-join filter?"
                )

    def _add_expression(
        self,
        expression: SqlExpression,
        *,
        input_node: RelNode,
        subquery_target: typing.Literal["semijoin", "antijoin", "scalar", "in"] = "scalar",
        in_column: SqlExpression | None = None,
    ) -> RelNode:
        """Generates the appropriate algebra fragment to execute a specific expression.

        Depending on the specific expression, simple mappings or even join nodes might be included in the fragment. If the
        expression is already provided by the input fragment, the fragment will be returned unmodified.

        Parameters
        ----------
        expression : SqlExpression
            The expression to include
        input_node : RelNode
            The operator that provides the input tuples for the expression
        subquery_target : typing.Literal["semijoin", "antijoin", "scalar", "in"], optional
            How the subquery results should be handled. This parameter is only used if the expression actually contains
            subqueries and depends on the context in which the expression is used, such as the owning predicate. *semijoin*
            and *antijoin* correspond to *MISSING* and *EXISTS* predicates. *in* corresponds to *IN* predicates that could
            either contain scalar subqueries that produce just a single value, or subqueries that produce an entire column of
            values. The appropriate handling is determined by this method automatically. Lastly, *scalar* indicates that the
            subquery is scalar and should produce just a single value, for usage e.g. in binary predicates or *SELECT* clauses.
        in_column : Optional[SqlExpression], optional
            For *IN* predicates that contain a subquery producing multiple rows, e.g. *R.a IN (SELECT S.b FROM S)*, this is the
            column that is compared to the subquery tuples (*R.a* in the example). For all other cases, this parameter is
            ignored.

        Returns
        -------
        RelNode
            The expanded algebra fragment
        """
        if expression in input_node.provided_expressions():
            return input_node

        match expression:
            case ColumnExpression() | StaticValueExpression():
                return input_node
            case SubqueryExpression():
                subquery_root = self._add_subquery(expression.query)
                match subquery_target:
                    case "semijoin":
                        return SemiJoin(input_node, subquery_root)
                    case "antijoin":
                        return AntiJoin(input_node, subquery_root)
                    case "scalar":
                        return CrossProduct(input_node, subquery_root)
                    case "in" if expression.query.is_scalar():
                        return CrossProduct(input_node, subquery_root)
                    case "in" if not expression.query.is_scalar():
                        unwrapped_scan = subquery_root.input_node
                        assert isinstance(unwrapped_scan, Projection) and len(unwrapped_scan.columns) == 1
                        assert in_column is not None, (
                            "IN predicate with subquery must provide the column to compare against"
                        )
                        in_predicate = BinaryPredicate.create_equal(in_column, unwrapped_scan.columns[0])
                        return SemiJoin(input_node, subquery_root, in_predicate)
            case CastExpression() | FunctionExpression() | MathExpression():
                return self._ensure_expression_applicability(expression, input_node)
            case WindowExpression() | CaseExpression():
                return self._ensure_expression_applicability(expression, input_node)
            case _:
                raise ValueError(f"Did not expect expression '{expression}'")

    def _add_subquery(self, subquery: SelectStatement) -> SubqueryScan:
        """Generates the appropriate algebra fragment to include a subquery in the current algebra tree."""
        subquery_parser = _ImplicitRelalgParser(subquery, provided_base_tables=self._base_table_fragments)
        subquery_root = subquery_parser.generate_relnode()
        self._required_columns = util.dicts.merge(subquery_parser._required_columns, self._required_columns)
        # We do not include the subquery base tables in our _base_table_fragments since the subquery base tables are already
        # processed completely and this would contradict the interpretation of the _base_table_fragments in
        # _generate_initial_join_order()
        return SubqueryScan(subquery_root, subquery)

    def _split_filter_predicate(self, pred: AbstractPredicate) -> dict[TableReference, AbstractPredicate]:
        """Extracts applicable filter predicates for varios base tables.

        This method splits conjunctive filters consisting of individual predicates for multiple base tables into an explicit
        mapping from base table to its filters.

        For example, consider a predicate *R.a = 42 AND S.b < 101 AND S.c LIKE 'foo%'*. The split would provide a dictionary
        ``{R: R.a < 42, S: S.b < 101 AND S.c LIKE 'foo%'}``

        Warnings
        --------
        The behavior of this method is undefined if the supplied predicate is not a filter predicate that can be evaluated
        during the base table evaluation phase.
        """
        if not pred.is_filter():
            raise ValueError(f"Not a filter predicate: {pred}")

        match pred:
            case AndPredicate(children):
                raw_predicate_components: dict[TableReference, set[AbstractPredicate]] = collections.defaultdict(set)
                for child_pred in children:
                    child_split = self._split_filter_predicate(child_pred)
                    for tab, pred in child_split.items():
                        raw_predicate_components[tab].add(pred)
                return {
                    base_table: CompoundPredicate.create_and(list(predicates))
                    for base_table, predicates in raw_predicate_components.items()
                }

            case OrPredicate() | NotPredicate():
                return {_BaseTableLookup()(pred): pred}

            case _:
                return {_BaseTableLookup()(pred): pred}

    def _split_join_predicate(self, predicate: AbstractPredicate) -> set[AbstractPredicate]:
        """Provides all individual join predicates that have to be evaluated.

        For conjunctive predicates, these are the actual components of the conjunction, all other predicates are returned
        as-is.
        """
        if not predicate.is_join():
            raise ValueError(f"Not a join predicate: '{predicate}'")
        if isinstance(predicate, AndPredicate):
            return set(predicate.children)
        return {predicate}

    def _ensure_predicate_applicability(self, predicate: AbstractPredicate, input_node: RelNode) -> RelNode:
        """Computes all required mappings that have to be execute before a predicate can be evaluated.

        If such mappings exist, the input relation is expanded with a new mapping operation, otherwise the relation is provided
        as-is.

        Parameters
        ----------
        predicate : AbstractPredicate
            The predicate to evaluate
        input_node : RelNode
            An operator providing the expressions that are already available.

        Returns
        -------
        RelNode
            An algebra fragment
        """
        provided_expressions = self._collect_provided_expressions(input_node)
        required_expressions = util.set_union(
            _collect_all_expressions(expression) for expression in predicate.iterexpressions()
        )
        missing_expressions = required_expressions - provided_expressions
        if missing_expressions:
            return Map(input_node, _generate_expression_mapping_dict(missing_expressions))
        return input_node

    def _ensure_expression_applicability(self, expression: SqlExpression, input_node: RelNode) -> RelNode:
        """Computes all required mappings that have to be execute before an expression can be evaluated.

        This is pretty much the equivalent to `_ensure_predicate_applicability` but for expressions.

        Parameters
        ----------
        expression : SqlExpression
            The expression to evaluate
        input_node : RelNode
            An operator providing the expressions that are already available.

        Returns
        -------
        RelNode
            An algebra fragment
        """
        provided_expressions = self._collect_provided_expressions(input_node)
        required_expressions = util.set_union(
            _collect_all_expressions(child_expr) for child_expr in expression.iterchildren()
        )
        missing_expressions = required_expressions - provided_expressions
        if missing_expressions:
            return Map(input_node, _generate_expression_mapping_dict(missing_expressions))
        return input_node

    def _collect_provided_expressions(self, *nodes: RelNode) -> set[SqlExpression]:
        """Collects all expressions that are provided by a set of algebra nodes."""
        outer_table_expressions = util.set_union(
            base_table.provided_expressions() for base_table in self._provided_base_tables.values()
        )
        return util.set_union(node.provided_expressions() for node in nodes) | outer_table_expressions


def parse_relalg(query: SelectStatement) -> RelNode:
    """Converts an SQL query to a representation in relational algebra.

    Parameters
    ----------
    query : SqlQuery
        The query to convert

    Returns
    -------
    RelNode
        The root node of the relational algebra tree. Notice that in some cases the algebraic expression might not be a tree
        but a directed, acyclic graph instead. However, in this case there still is a single root node.
    """
    raw_relnode = _ImplicitRelalgParser(query).generate_relnode()

    # We perform a final mutation to ensure that sideways passes are generated correctly. This removes redundant subtrees that
    # could be left over from the initial parsing.
    return raw_relnode.mutate()
