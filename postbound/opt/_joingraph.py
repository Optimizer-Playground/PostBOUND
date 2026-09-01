"""Provides an implementation of a dynamic join graph, as well as some related objects."""

from __future__ import annotations

import collections
import copy
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, Optional, cast

import networkx as nx

from .. import transform, util
from .._core import ColumnReference, TableReference
from ..db import DatabasePool, DatabaseSchema
from ..qal import (
    AbstractPredicate,
    BinaryPredicate,
    CompoundPredicate,
    SelectStatement,
    determine_join_equivalence_classes,
    generate_predicates_for_equivalence_classes,
)


@dataclass(frozen=True)
class JoinPath:
    """A join path models the join between two tables.

    Usually a path represents a join where one table is part of an intermediate result, whereas the other table is already part
    of an intermediate result. However, this is not required.

    Attributes
    ----------
    start_table : TableReference
        The first join partner involved in the join. This is the table that is already part of the intermediate result of the
        query
    target_table : TableReference
        The second join partner involved in the join. This is the table that is not yet part of any intermediate result. Thus,
        this is the table that should be joined next
    join_condition : Optional[AbstractPredicate], optional
        The predicate that is used to actually join the `target_table` with the current intermediate result. Usually the
        predicate is restricted to the the join between `start_table` and `target_table`, but can also include additional join
        predicates over other tables in the intermediate results.
    """

    start_table: TableReference
    target_table: TableReference
    join_condition: Optional[AbstractPredicate] = None

    def tables(self) -> tuple[TableReference, TableReference]:
        """Provides the tables that are joined.

        Returns
        -------
        tuple[TableReference, TableReference]
            The tables

        Warnings
        --------
        The definition of this methods differs slightly from other definitions of the tables method that can be found in the
        query abstraction layer. The tables method for join paths really only focuses on `start_table` and `target_table`. If
        additional tables appear as part of the `join_condition`, they are ignored.
        """
        return self.start_table, self.target_table

    def spans_table(self, table: TableReference) -> bool:
        """Checks, whether a specific table is either the start, or the target table in this path.

        Parameters
        ----------
        table : TableReference
            The table to check

        Returns
        -------
        bool
            Whether the table is part of the join path. Notice that this check does not consider tables that are part of the
            `join_condition`.
        """
        return table == self.start_table or table == self.target_table

    def flip_direction(self) -> JoinPath:
        """Creates a new join path with the start and target tables reversed.

        Returns
        -------
        JoinPath
            The new join path
        """
        return JoinPath(self.target_table, self.start_table, join_condition=self.join_condition)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.start_table} ⋈ {self.target_table}"


class IndexInfo:
    """This class captures relevant information about the availability of per-column indexes and their status.

    The lifecycle of an index can be managed using the `invalidate` method. This indicates that an index can no longer be used
    for a specific join, for example because its column has become part of an intermediate result already. In contrast to many
    other types in PostBOUND, index information is a mutable structure and can be changed in-place.

    The current implementation is only focused on indexes over a single column, multi-column indexes are not supported. Another
    limitation is that the specific type (i.e. data structure) of the index is not captured. If this information is important,
    it has to be maintained by the user.

    Parameters
    ----------
    column : ColumnReference
        The column fr which the index is created
    index_type : Literal["primary", "secondary", "none"]
        The kind of index that is maintained. ``"none"`` indicates that there is no index on the column. This is a different
        concept from an index that exists, but cannot be used. The latter case is indicated via the `invalid` parameter
    invalid : bool, optional
        Whether the index can still be used during query execution. Typically, this is true for relations that have not been
        included in any intermediate result and false afterwards.
    """

    @staticmethod
    def primary_index(column: ColumnReference) -> IndexInfo:
        """Creates index information for a primary key index.

        Parameters
        ----------
        column : ColumnReference
            The indexed column

        Returns
        -------
        IndexInfo
            The index information. The index is initialized as a valid index.
        """
        return IndexInfo(column, "primary")

    @staticmethod
    def secondary_index(column: ColumnReference) -> IndexInfo:
        """Creates index information for a secondary index.

        Foreign key indexes are often defined this way.

        Parameters
        ----------
        column : ColumnReference
            The indexed column

        Returns
        -------
        IndexInfo
            The index information. The index is initialized as a valid index.
        """
        return IndexInfo(column, "secondary")

    @staticmethod
    def no_index(column: ColumnReference) -> IndexInfo:
        """Creates index information that indicates the absence of an index.

        Parameters
        ----------
        column : ColumnReference
            A column that does not have any index

        Returns
        -------
        IndexInfo
            The index information
        """
        return IndexInfo(column, "none")

    @staticmethod
    def generate_for(column: ColumnReference, db_schema: DatabaseSchema) -> IndexInfo:
        """Determines available indexes for a specific column.

        Parameters
        ----------
        column : ColumnReference
            The column. It has to be connected to a valid, non-virtual table reference
        db_schema : DatabaseSchema
            The schema of the database to which the column belongs.

        Returns
        -------
        IndexInfo
            _description_

        Raises
        ------
        base.UnboundColumnError
            If the column is not associated with any table
        """
        if db_schema.is_primary_key(column):
            return IndexInfo.primary_index(column)
        elif db_schema.has_secondary_index(column):
            return IndexInfo.secondary_index(column)
        else:
            return IndexInfo.no_index(column)

    def __init__(
        self,
        column: ColumnReference,
        index_type: Literal["primary", "secondary", "none"],
        invalid: bool = False,
    ) -> None:
        self._column = column
        self._index_type = index_type
        self._is_invalid = invalid

    @property
    def column(self) -> ColumnReference:
        """Get the column to which the index information belongs.

        Returns
        -------
        ColumnReference
            The column
        """
        return self._column

    @property
    def index_type(self) -> Literal["primary", "secondary", "none"]:
        """Get the kind of index that is in principle available on the column.

        The index type does not contain any information about whether an index is actually usable for a specific join. It
        merely states whether an index has been defined.

        Returns
        -------
        str
            The index type. Can be *primary*, *secondary* or *none*.
        """
        return self._index_type

    @property
    def is_invalid(self) -> bool:
        """Get whether the index is actually usable.

        To determine whether an index can be used right now, this property has to be combined with the `index_type` value.
        If there never was an index on the column, `is_valid` might have been true from the get-go. To make this check easier,
        a number of utility methods exist.

        Returns
        -------
        bool
            Whether the index is usable if it exists. If there is no index on the column, the index cannot be interpreted in
            any meaningful way.
        """
        return self._is_invalid

    def is_primary(self) -> bool:
        """Checks, whether this is a valid primary index.

        Returns
        -------
        bool
            Whether this is a primary key index and ensures that it is still valid.
        """
        return not self._is_invalid and self._index_type == "primary"

    def is_secondary(self) -> bool:
        """Checks, whether this is a valid secondary index.

        Returns
        -------
        bool
            Whether this is a secondary index and ensures that it is still valid.
        """
        return not self._is_invalid and self._index_type == "secondary"

    def is_indexed(self) -> bool:
        """Checks, whether there is any valid index defined for the column.

        This check does not differentiate between primary key indexes and secondary indexes.

        Returns
        -------
        bool
            Whether this is a primary key or secondary index and ensures that it is still valid.
        """
        return self.is_primary() or self.is_secondary()

    def can_pk_fk_join(self, other: IndexInfo) -> bool:
        """Checks, whether two columns can be joined via a primary key/foreign key join.

        This method does not restrict the direction of such a join, i.e. each column could act as the primary key or
        foreign key. Likewise, no datatype checks are performed and it is assumed that a database system would be
        able to actually join the two columns involved.

        If indexes on any of the columns are no longer available, this check fails.

        Parameters
        ----------
        other : IndexInfo
            The index information of the other column that should participate in the join

        Returns
        -------
        bool
            Whether a primary key/foreign key join could be executed between the columns.
        """
        if not self.is_indexed() or not other.is_indexed():
            return False

        if self.is_secondary() and other.is_secondary():
            return False

        # all other cases have at least one primary key index available
        return True

    def invalidate(self) -> None:
        """Marks the index as invalid.

        Once a table is included in an intermediate join result, the index structures of its columns most likely
        become invalid, and it is no longer possible to use the index to query for specific tuples (because the
        occurrences of the individual tuples are multiplied when executing the join). This method can be used to model
        the lifecycle of index structures within the course of the execution of a single query.
        """
        self._is_invalid = True

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        invalid_state = " INVALID" if self._is_invalid else ""
        if self._index_type == "none":
            return f"NO INDEX({self._column})"
        elif self._index_type == "primary":
            return f"PRIMARY INDEX({self._column}){invalid_state}"
        else:
            return f"SECONDARY INDEX({self._column}){invalid_state}"


@dataclass(frozen=True)
class TableInfo:
    """This class captures information about the state of tables in the join graph.

    Attributes
    ----------
    free : bool
        Whether the table is still *free*, i.e. is not a part of any intermediate join result.
    index_info : Collection[IndexInfo]
        Information about the indexes of all columns that belong to the table. If a column does not appear in this collection,
        it does not have any indexes, or the column is not relevant in the current join graph (i.e. because it does not
        appear in any join predicate)
    """

    free: bool
    index_info: Collection[IndexInfo]


_PredicateMap = collections.defaultdict[frozenset[TableReference], list[AbstractPredicate]]
"""Type alias for an (internally used) predicate map"""


class JoinGraph(Mapping[TableReference, TableInfo]):
    """The join graph models the connection between different tables in a query.

    All tables that are referenced in the query are represented as the nodes in the graph. If two tables are joined via a join
    predicate in the SQL query, they will be linked with an edge in the join graph. This graph is further annotated by the join
    predicate. Additionally, the join graph stores index information for each of the relevant columns in the query.

    In contrast to many other types in PostBOUND, a join graph is a mutable structure. It also models the current state of the
    optimizer once specific tables have been included in an intermediate join result.

    The wording of the different join graph methods distinguishes three states of joins (and correspondingly tables):

    - a join might be *free*, if at least one of the corresponding tables have not been marked as joined, yet
    - a join might be *available*, if it is *free* and one of the tables is already included in some intermediate join
    - a join is *consumed*, if its no longer *free*. This occurs once the partner tables have both been marked as joined.

    A further distinction is made between n:m joins and primary key/foreign key joins. Information about the available joins of
    each type can be queried easily and many methods are available in two variants: one that includes all possible joins and
    one that is only focused on primary key/foreign key joins. To determine the precise join types, the join graph needs to
    access the database schema. A n:m join is one were the column values of both join partners can appear an arbitrary number
    of times, corresponding to an n:m relation between the two tables.

    By calling the `mark_joined` method,  the state of individual joins and their corresponding tables might change. This also
    means that former primary key/foreign key joins might become n:m joins (which is the case exactly when the primary key
    table is inserted into an intermediate join result).

    Parameters
    ----------
    query : SqlQuery
        The query for which the join graph should be generated
    db_schema : Optional[DatabaseSchema], optional
        The schema of the database on which the query should be executed. If this is ``None``, the database schema is inferred
        based on the `DatabasePool`.
    include_predicate_equivalence_classes : bool, optional
        Whether predicates from the same equivalence class should be added to the join graph as well, even if they do not exist
        in the original query. Consider two join conditions *a = b* and *b = c*. From these conditions, it follows that *a = c*
        by the transitive property. If `include_predicate_equivalence_classes` is true, that last predicate will also be added
        to the join graph.

    Warnings
    --------
    If predicate equivalence classes are used, the optimization algorithm can potentially generate queries that contain
    additional predicates that were not present in the original query.
    """

    def __init__(
        self,
        query: SelectStatement,
        db_schema: Optional[DatabaseSchema] = None,
        *,
        include_predicate_equivalence_classes: bool = False,
    ) -> None:
        if db_schema is None:
            db_schema = DatabasePool.get_instance().current_database().schema()

        if any(not col.is_bound() for col in query.columns()):
            raise ValueError("Cannot create a join graph for a query with unbound columns")

        self.query = query
        self._db_schema = db_schema
        self._index_structures: dict[ColumnReference, IndexInfo] = {}

        graph = nx.Graph()
        graph.add_nodes_from(query.tables(), free=True)
        edges = []
        predicate_map: _PredicateMap = collections.defaultdict(list)
        join_predicates = query.predicates().joins()
        if any(not isinstance(pred, BinaryPredicate) for pred in join_predicates):
            raise ValueError("Join graph can only be created for queries with binary join predicates")
        join_predicates = cast(list[BinaryPredicate], join_predicates)

        if include_predicate_equivalence_classes:
            join_equivalence_classes = determine_join_equivalence_classes(join_predicates)
            equivalence_class_predicates = generate_predicates_for_equivalence_classes(join_equivalence_classes)
            join_predicates = set(join_predicates) | equivalence_class_predicates
        for join_predicate in join_predicates:
            if len(join_predicate.columns()) != 2:
                continue
            first_col, second_col = join_predicate.columns()
            predicate_map[frozenset([first_col.table, second_col.table])].append(join_predicate)  # type: ignore - all columns are bound

        for tables, joins in predicate_map.items():
            first_tab, second_tab = tables
            join_predicate = CompoundPredicate.create_and(joins)
            edges.append((first_tab, second_tab, {"predicate": join_predicate}))
            for column in join_predicate.columns():
                self._index_structures[column] = IndexInfo.generate_for(column, db_schema)

        graph.add_edges_from(edges)
        self._graph = graph

    def initial(self) -> bool:
        """Checks, whether the join graph has already been modified.

        Returns
        -------
        bool
            ``True`` indicates that the join graph is still in its initial state, i.e. no table has been marked as joined, yet.
        """
        return all(is_free for __, is_free in self._graph.nodes.data("free"))

    def contains_cross_products(self) -> bool:
        """Checks, whether there are any cross products in the input query.

        A cross product is a join between tables without a restricting join predicate. Note that this is only the case
        if those tables are also not linked via a sequence of join predicates with other tables.

        Returns
        -------
        bool
            Whether the join graph contains at least one cross product.
        """
        return not nx.is_connected(self._graph)

    def contains_free_tables(self) -> bool:
        """Checks, whether there is at least one more free tables remaining in the graph.

        Returns
        -------
        bool
            Whether there are still free tables in the join graph
        """
        return any(is_free for __, is_free in self._graph.nodes.data("free"))

    def contains_free_n_m_joins(self) -> bool:
        """Checks, whether there is at least one more free n:m join remaining in the graph.

        Returns
        -------
        bool
            Whether there are still n:m joins with at least one free table.
        """
        is_first_join = self.initial()
        for first_tab, second_tab, predicate in self._graph.edges.data("predicate"):
            if not self.is_available_join(first_tab, second_tab) and not is_first_join:
                continue
            for first_col, second_col in predicate.join_partners():
                if not self._index_structures[first_col].can_pk_fk_join(self._index_structures[second_col]):
                    return True
        return False

    def count_consumed_tables(self) -> int:
        """Determines the number of tables that have been joined already.

        This number might be 1 if only the initial tables has been selected, or 0 if the join graph is still in its initial
        state.

        Returns
        -------
            int
                The number of joined tables
        """
        return len([is_free for __, is_free in self._graph.nodes.data("free") if not is_free])

    def join_components(self) -> Iterable[JoinGraph]:
        """Provides all components of the join graph.

        A component is a subgraph of the original join graph, such that the subgraph is connected but there was no
        edge between nodes from different sub-graphs. This corresponds to the parts of the query that have to be joined
        via a cross product.

        Returns
        -------
        Iterable[JoinGraph]
            The components of the join graph, each as its own full join graph object
        """
        components = []
        for component in nx.connected_components(self._graph):
            component_query = transform.extract_subquery(self.query, component)
            components.append(JoinGraph(component_query, self._db_schema))
        return components

    def joined_tables(self) -> frozenset[TableReference]:
        """Provides all non-free tables in the join graph.

        Returns
        -------
        frozenset[TableReference]
            The tables that have already been joined / consumed.
        """
        return frozenset(table for table in self if not self.is_free_table(table))

    def free_tables(self) -> frozenset[TableReference]:
        """Provides all tables that have not been joined, yet.

        Returns
        -------
        frozenset[TableReference]
            The tables that are not consumed
        """
        return frozenset(table for table in self if self.is_free_table(table))

    def all_joins(self) -> Iterable[tuple[TableReference, TableReference]]:
        """Provides all edges in the join graph, no matter whether they are available or not.

        Returns
        -------
        Iterable[tuple[TableReference, TableReference]]
            The possible joins in the graph. The assignment to the first or second component of the tuple is arbitrary
        """
        return list(self._graph.edges)

    def available_join_paths(self, *, both_directions_on_initial: bool = False) -> Iterable[JoinPath]:
        """Provides all joins that can be executed in the current join graph.

        The precise output of this method depends on the current state of the join graph: If the graph is still in its initial
        state (i.e. none of the tables is joined yet), all joins are provided. Otherwise, only those join paths are considered
        available, where one table is already joined, and the join partner is still free. The free table will be the target
        table in the join path whereas the joined table will be the start table.

        Parameters
        ----------
        both_directions_on_initial : bool, optional
            Whether to include the join path *R* -> *S* as well as *S* -> *R* for initial join graphs, assuming there is a join
            between *R* and *S* in the graph.

        Returns
        -------
        Iterable[JoinPath]
            All possible joins in the current graph.
        """
        join_paths = []
        if self.initial():
            for join_edge in self._graph.edges.data("predicate"):
                source_table, target_table, join_condition = join_edge
                current_join_path = JoinPath(source_table, target_table, join_condition)
                join_paths.append(current_join_path)
                if both_directions_on_initial:
                    join_paths.append(current_join_path.flip_direction())
            return join_paths

        for join_edge in self._graph.edges.data("predicate"):
            source_table, target_table, join_condition = join_edge
            if self.is_free_table(source_table) and self.is_free_table(target_table):
                # both tables are still free -> no path
                continue
            elif not self.is_free_table(source_table) and not self.is_free_table(target_table):
                # both tables are already joined -> no path
                continue

            if self.is_free_table(source_table):
                # fix directionality
                source_table, target_table = target_table, source_table
            join_paths.append(JoinPath(source_table, target_table, join_condition))

        return join_paths

    def available_n_m_join_paths(self, *, both_directions_on_initial: bool = False) -> Iterable[JoinPath]:
        """Provides exactly those join paths from `available_join_paths` that correspond to n:m joins.

        The logic for initial and "dirty" join graphs is inherited from `available_join_paths` and can be further customized
        via the `both_directions_on_initial` parameter.

        Parameters
        ----------
        both_directions_on_initial : bool, optional
            Whether to include the join path *R* -> *S* as well as *S* -> *R* for initial join graphs if *R* ⨝ *S* is an n:m
            join.

        Returns
        -------
        Iterable[JoinPath]
            The available n:m joins
        """
        n_m_paths = []
        for join_path in self.available_join_paths():
            start_table, target_table = join_path.start_table, join_path.target_table
            if not self.is_pk_fk_join(start_table, target_table) and not self.is_pk_fk_join(target_table, start_table):
                n_m_paths.append(join_path)
                if both_directions_on_initial and self.initial():
                    n_m_paths.append(join_path.flip_direction())
        return n_m_paths

    def available_join_paths_for(self, table: TableReference) -> Iterable[JoinPath]:
        """Returns all possible joins of a specific table.

        What constitutes a possible join depends on the state of the join graph: for an initial join graph, the only
        requirement is a valid join predicate between the tables. In all other cases, exactly one of the tables has to be free
        and the other table has to be consumed.

        Parameters
        ----------
        table : TableReference
            The table that should be joined

        Returns
        -------
        Iterable[JoinPath]
            All possible join paths for the table. This includes n:m joins as well as primary key/foreign key joins. In
            each join path the specified table will be the start table and the join partner will be the target table.
        """
        self._assert_contains_table(table)
        return [
            JoinPath(table, partner_table, join_edge["predicate"])
            for partner_table, join_edge in self._graph.adj[table].items()
            if self.is_available_join(table, partner_table)
        ]

    def nx_graph(self) -> nx.Graph:
        """Provides the underlying graph object for this join graph.

        Returns
        -------
        nx.Graph
            A deep copy of the raw join graph
        """
        return copy.deepcopy(self._graph)

    def is_free_table(self, table: TableReference) -> bool:
        """Checks, whether a specific table is still free in this join graph.

        If the table is not part of the join graph, an error is raised.

        Parameters
        ----------
        table : TableReference
            The table to check

        Returns
        -------
        bool
            Whether the given table is still free
        """
        return self._graph.nodes[table]["free"]

    def joins_tables(self, first_table: TableReference, second_table: TableReference) -> bool:
        """Checks, whether the join graph contains an edge between specific tables.

        This check does not require the join in question to be available (this is what `is_available_join` is for).

        Parameters
        ----------
        first_table : TableReference
            The first join partner
        second_table : TableReference
            The second join partner

        Returns
        -------
        bool
            Whether there is any join predicate between the given tables. The direction or availability does not matter for
            this check
        """
        return (first_table, second_table) in self._graph.edges

    def is_available_join(self, first_table: TableReference, second_table: TableReference) -> bool:
        """Checks, whether the join between two tables is still available.

        For initial join graphs, this check passes as long as there is a valid join predicate between the two given tables. In
        all other cases, one of the join partners has to be consumed, whereas the other partner has to be free.

        Parameters
        ----------
        first_table : TableReference
            The first join partner
        second_table : TableReference
            The second join partner

        Returns
        -------
        bool
            Whether there is a valid join between the given tables and whether this join is still available. The join direction
            and join type do not matter.
        """
        first_free, second_free = (
            self._graph.nodes[first_table]["free"],
            self._graph.nodes[second_table]["free"],
        )
        valid_join = self.joins_tables(first_table, second_table)
        available_join = (first_free and not second_free) or (not first_free and second_free) or self.initial()
        return valid_join and available_join

    def is_pk_fk_join(self, fk_table: TableReference, pk_table: TableReference) -> bool:
        """Checks, whether the join between the supplied tables is a primary key/foreign key join.

        This check does not require the indicated join to be available.

        Parameters
        ----------
        fk_table : TableReference
            The foreign key table
        pk_table : TableReference
            The primary key table

        Returns
        -------
        bool
            Whether the join between the given tables is a primary key/foreign key join with the correct direction

        Warnings
        --------
        In the current implementation, this check only works for (conjunctions of) binary join predicates. An error is raised
        for joins between multiple columns
        """

        if not self.joins_tables(fk_table, pk_table):
            return False

        predicate: AbstractPredicate = self._graph.edges[fk_table, pk_table]["predicate"]
        for base_predicate in predicate.base_predicates():
            fk_col = util.simplify(base_predicate.columns_of(fk_table))
            pk_col = util.simplify(base_predicate.columns_of(pk_table))
            if self._index_structures[fk_col].is_indexed() and self._index_structures[pk_col].is_primary():
                return True
        return False

    def is_n_m_join(self, first_table: TableReference, second_table: TableReference) -> bool:
        """Checks, whether the join between the supplied tables is an n:m join.

        This check does not require the indicated join to be available.

        Parameters
        ----------
        first_table : TableReference
            The first join partner
        second_table : TableReference
            The second join partner

        Returns
        -------
        bool
            Whether the join between the given tables is an n:m join

        Warnings
        --------
        In the current implementation, this check only works for (conjunctions of) binary join predicates. An error is raised
        for joins between multiple columns
        """
        return (
            self.joins_tables(first_table, second_table)
            and not self.is_pk_fk_join(first_table, second_table)
            and not self.is_pk_fk_join(second_table, first_table)
        )

    def available_pk_fk_joins_for(self, fk_table: TableReference) -> Iterable[JoinPath]:
        """Provides all available primary key/foreign key joins with a specific foreign key table.

        This method does not restrict itself to available joins, but requires that at least one of the join parts is free.

        Parameters
        ----------
        fk_table : TableReference
            The foreign key table. This will be the start table in all join paths.

        Returns
        -------
        Iterable[JoinPath]
            All matching join paths. The start table of the path will be the foreign key table, whereas the primary key table
            will be the target table.
        """
        self._assert_contains_table(fk_table)
        return [
            JoinPath(fk_table, pk_table, join_edge["predicate"])
            for pk_table, join_edge in self._graph.adj[fk_table].items()
            if self.is_pk_fk_join(fk_table, pk_table) and (self.is_free_table(fk_table) or self.is_free_table(pk_table))
        ]

    def available_deep_pk_join_paths_for(
        self,
        fk_table: TableReference,
        ordering: Callable[[TableReference, dict], int] | None = None,
    ) -> Iterable[JoinPath]:
        """Provides all available pk/fk joins with the given table, as well as follow-up pk/fk joins.

        In contrast to the `available_pk_fk_joins_for` method, this method does not only return direct joins between the
        foreign key table, but augments its output in the following way: suppose the foreign key table is pk/fk joined with a
        primary key table *t*. Then, this method also includes all joins of *t* with additional tables *t'*, such
        that *t* ⋈ *t'* is once again a primary key/foreign key join, but this time with *t* acting as the foreign key
        and *t'* as the primary key. This procedure is repeated for all *t'* tables recursively until no more primary
        key/foreign key joins are available.

        Essentially, this is equivalent to performing a breadth-first search on all (directed) primary key/foreign key
        joins, starting at the foreign key table. The sequence in which joins on the same level are placed into the resulting
        iterable can be customized via the `ordering` parameter. This callable receives the current primary key table
        and the edge data as input and produces a numerical position weight as output (smaller values meaning earlier
        placement). The provided edge data contains the join predicate under the ``"predicate"`` key. Using the join predicate,
        the join partner (i.e. the foreign key table) can be retrieved.

        Parameters
        ----------
        fk_table : TableReference
            The foreign key table at which the search should be anchored.
        ordering : Callable[[TableReference, dict], int] | None, optional
            How to sort different primary key join partners on the same level. Lower values mean earlier positioning. This
            defaults to ``None``, in which case an arbitrary ordering is used.

        Returns
        -------
        Iterable[JoinPath]
            All deep primary key/foreign key join paths, starting at the `fk_table`
        """
        self._assert_contains_table(fk_table)
        available_joins = util.nx.nx_bfs_tree(self._graph, fk_table, self._check_pk_fk_join, node_order=ordering)
        join_paths = []
        for join in available_joins:
            current_pk_table: TableReference = join[0]
            join_predicate: AbstractPredicate = join[1]["predicate"]
            current_fk_table: TableReference = util.simplify(
                {column.table for column in join_predicate.join_partners_of(current_pk_table)}
            )  # type: ignore - all columns are bound
            join_paths.append(JoinPath(current_fk_table, current_pk_table, join_predicate))
        return join_paths

    def join_partners_from(
        self, table: TableReference, candidate_tables: Iterable[TableReference]
    ) -> set[TableReference]:
        """Provides exactly those tables from a set of candidates that are joined with a specific given table.

        This check does not require the joins in question to be available. The existence of a join edge is sufficient.

        Parameters
        ----------
        table : TableReference
            The table that should be joined with the candidates
        candidate_tables : Iterable[TableReference]
            Possible join partners for the `table`. Join type and direction do not matter

        Returns
        -------
        set[TableReference]
            Those tables of the `candidate_tables` that can be joined with the partner table.
        """
        candidate_tables = set(candidate_tables)
        return set(neighbor for neighbor in self._graph.adj[table].keys() if neighbor in candidate_tables)

    def join_predicates_between(
        self,
        first_tables: TableReference | Iterable[TableReference],
        second_tables: Optional[TableReference | Iterable[TableReference]] = None,
    ) -> Collection[AbstractPredicate]:
        """Provides all join predicates between sets of tables.

        This method operates in two modes: if only one set of tables is given, all join predicates for tables within that set
        are collected. If two sets are given, all join predicates for tables from both sets are collected, but not predicates
        from tables within the same set.

        The status of the tables, as well as the join type, do not play a role in this check.

        Parameters
        ----------
        first_tables : TableReference | Iterable[TableReference]
            The first set of candidate tables. Can optionally also be a single table, in which case the check is only
            performed for this table and the partner set
        second_tables : Optional[TableReference  |  Iterable[TableReference]], optional
            The second set of candidate tables. By default this is ``None``, which results in collecting only join predicates
            for tables from `first_tables`. Can also be a single table, in which case the check is only performed for this
            table and the partner set

        Returns
        -------
        Collection[AbstractPredicate]
            All join predicates
        """
        first_tables = util.enlist(first_tables)
        second_tables = util.enlist(second_tables) if second_tables else first_tables
        matching_predicates = set()

        for first_table in first_tables:
            for second_table in second_tables:
                join_predicate = self._fetch_join_predicate(first_table, second_table)
                if join_predicate:
                    matching_predicates.add(join_predicate)

        return matching_predicates

    def mark_joined(self, table: TableReference, join_edge: Optional[AbstractPredicate] = None) -> None:
        """Updates the join graph to include a specific table in the intermediate result.

        This procedure also changes the available index structures according to the kind of join that was executed.
        This is determined based on the current state of the join graph, the index structures, as well as the supplied join
        predicate. If no join predicate is supplied, it is inferred from the query predicates.

        Parameters
        ----------
        table : TableReference
            The tables that becomes part of an intermediate result
        join_edge : Optional[AbstractPredicate], optional
            The condition that is used to carry out the join. Defaults to ``None``, in which case the predicate is inferred
            from the predicates that have been supplied by the initial query.
        """

        # TODO: check, if we actually need to handle transient index updates here as well
        # TODO: do we still need the join_edge parameter if we infer it from the predicates anyway?

        self._graph.nodes[table]["free"] = False
        if len(self.joined_tables()) == 1:
            return

        join_edge = join_edge if join_edge else self.query.predicates().joins_between(table, self.joined_tables())
        if not join_edge:
            # We still need this check even though we already know that there are at least two tables joined, since
            # these two tables might have nothing to do with each other (e.g. different components in the join graph)
            return

        partner_tables: set[TableReference] = {col.table for col in join_edge.join_partners_of(table)}  # type: ignore - all columns are bound
        for partner_table in partner_tables:
            pk_fk_join = self.is_pk_fk_join(table, partner_table)
            fk_pk_join = self.is_pk_fk_join(partner_table, table)

            if pk_fk_join and fk_pk_join:  # PK/PK join
                continue

            for col1, col2 in join_edge.join_partners():
                joined_col, partner_col = (col1, col2) if col1.table == table else (col2, col1)
                if pk_fk_join:
                    self._index_structures[partner_col].invalidate()
                elif fk_pk_join:
                    self._index_structures[joined_col].invalidate()
                else:
                    self._index_structures[partner_col].invalidate()
                    self._index_structures[joined_col].invalidate()

            if pk_fk_join:
                continue

            for table, is_free in self._graph.nodes.data("free"):
                if is_free or table == partner_table:
                    continue
                self._invalidate_indexes_on(table)

    def clone(self) -> JoinGraph:
        """Provides a deep copy of the current join graph.

        Returns
        -------
        JoinGraph
            The copy. It can be safely modified without affecting the original join graph.
        """
        cloned = JoinGraph(self.query, self._db_schema)
        cloned._graph = self.nx_graph()
        cloned._index_structures = copy.deepcopy(self._index_structures)
        return cloned

    def _assert_contains_table(self, table: TableReference) -> None:
        """Raises an error if a specific table is not part of the join graph.

        Parameters
        ----------
        table : TableReference
            The table to check

        Raises
        ------
        ValueError
            If the table is not a node in the join graph
        """
        if table not in self:
            raise ValueError(f"Join graph does not contain table {table}")

    def _check_pk_fk_join(self, pk_table: TableReference, edge_data: dict) -> bool:
        """Checks, whether a specific table acts as a primary key in the join as indicated by a join graph edge.

        Parameters
        ----------
        pk_table : TableReference
            The table to check
        edge_data : dict
            The join that should be performed. This has to be contained in the ``"predicate"`` key.

        Returns
        -------
        bool
            Whether the `pk_table` actually acts as the primary key in the given join edge.
        """
        join_predicate: AbstractPredicate = edge_data["predicate"]
        for base_predicate in join_predicate.base_predicates():
            fk_table: TableReference = util.simplify(
                {column.table for column in base_predicate.join_partners_of(pk_table)}
            )  # type: ignore - all columns are bound
            if self.is_pk_fk_join(fk_table, pk_table):
                return True
        return False

    def _invalidate_indexes_on(self, table: TableReference) -> None:
        """Invalidates all indexes on all columns that belong to the given table.

        Parameters
        ----------
        table : TableReference
            The table for which the invalidation should take place
        """
        for column, index in self._index_structures.items():
            if column.table == table:
                index.invalidate()

    def _fetch_join_predicate(
        self, first_table: TableReference, second_table: TableReference
    ) -> Optional[AbstractPredicate]:
        """Provides the join predicate between specific tables if there is one.

        Parameters
        ----------
        first_table : TableReference
            The first join partner
        second_table : TableReference
            The second join partner

        Returns
        -------
        Optional[AbstractPredicate]
            The join predicate if exists or ``None`` otherwise. The status of the join partners does not matter
        """
        if (first_table, second_table) not in self._graph.edges:
            return None
        return self._graph.edges[first_table, second_table]["predicate"]

    def _index_info_for(self, table: TableReference) -> Collection[IndexInfo]:
        """Provides all index info for a specific table (i.e. for each column that belongs to the table).

        Parameters
        ----------
        table : TableReference
            The table to retrieve the index info for

        Returns
        -------
        Collection[IndexInfo]
            The index info of each column of the table. If no information for a specific column is contained in this
            collection, this indicates that the column is not important for the join graph's query.
        """
        return [info for info in self._index_structures.values() if info._column.belongs_to(table)]

    def __len__(self) -> int:
        return len(self._graph)

    def __iter__(self) -> Iterator[TableReference]:
        return iter(self._graph.nodes)

    def __contains__(self, x: object) -> bool:
        return x in self._graph.nodes

    def __getitem__(self, key: TableReference) -> TableInfo:
        if key not in self:
            raise KeyError(f"Table {key} is not part of the join graph")
        free = self.is_free_table(key)
        index_info = self._index_info_for(key)
        return TableInfo(free, index_info)
