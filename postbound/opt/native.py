"""Native strategies obtain execution plans from actual database management systems.

Instead of performing optimizations on their own, the native stages delegate all decisions to a specific database system.
Afterwards, they analyze the query plan and encode the relevant information in a stage-specific format.

Notes
-----
By combining native stages with different target database systems, the optimizers of the respective systems can be combined.
For example, combining a join ordering stage with an Oracle backend and an operator selection stage with a Postgres backend
would provide a combined query optimizer with Oracle's join ordering algorithm and Postgres' operator selection.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from typing import Optional

from .. import transform, util
from .._core import (
    Cardinality,
    ColumnReference,
    Cost,
    IntermediateOperator,
    JoinOperator,
    ScanOperator,
    TableReference,
)
from .._hints import (
    JoinTree,
    PhysicalOperatorAssignment,
    PlanParameterization,
    jointree_from_plan,
    operators_from_plan,
    parameters_from_plan,
)
from .._qep import QueryPlan
from .._stages import (
    CardinalityEstimator,
    CompleteOptimizationAlgorithm,
    CostModel,
    JoinOrderOptimization,
    ParameterGeneration,
    PhysicalOperatorSelection,
)
from ..db import Database, DatabaseServerError, DatabaseUserError
from ..postgres import PostgresInterface
from ..qal import ColumnExpression, OrderBy, SqlQuery
from ..util import jsondict


class CostEstimationWarning(UserWarning):
    pass


class NativeCostModel(CostModel):
    """Obtains the cost of a query plan by using the cost model of an actual database system.

    Parameters
    ----------
    raise_on_error : bool
        Whether the cost model should raise an error if anything goes wrong during the estimation. For example, this can
        happen if the query plan cannot be executed on the target database system. If this is off (the default), failure
        results in an infinite cost.
    verbose : bool, optional
        Whether the cost model should issue warnings if anything goes wrong during the estimation. This includes cases
        where the cost of some operators cannot be estimated by the target database system.
    """

    def __init__(self, *, raise_on_error: bool = False, verbose: bool = False) -> None:
        super().__init__()
        self.target_db: Optional[Database] = None
        self._raise_on_error = raise_on_error
        self._verbose = verbose

    def estimate_cost(self, query: SqlQuery, plan: QueryPlan) -> Cost:
        assert self.target_db is not None

        matching_tables = query.tables() == plan.tables()
        intermediate_op = plan.operator in {
            IntermediateOperator.Materialize,
            IntermediateOperator.Memoize,
        }
        if intermediate_op and matching_tables:
            raise ValueError("Cannot estimate the cost of intermediate operators as final operator in a plan.")
        if not intermediate_op and not matching_tables:
            query = transform.extract_subquery(query, plan.tables())

        match plan.operator:
            case ScanOperator.IndexScan | ScanOperator.IndexOnlyScan:
                return self._cost_index_op(query, plan)
            case IntermediateOperator.Materialize:
                return self._cost_materialize_op(query, plan)
            case IntermediateOperator.Memoize:
                return self._cost_memoize_op(query, plan)
            case IntermediateOperator.Sort:
                return self._cost_sort_op(query, plan)
            case _:
                # No action needed, processing starts below
                pass

        hinted_query = self.target_db.hinting().generate_hints(query, plan.with_actual_card())
        if self._raise_on_error:
            cost = self.target_db.optimizer().cost_estimate(hinted_query)
        else:
            try:
                cost = self.target_db.optimizer().cost_estimate(hinted_query)
            except (DatabaseServerError, DatabaseUserError):
                cost = math.inf
        return cost

    def describe(self) -> jsondict:
        return {
            "name": "native",
            "database_system": self.target_db.describe() if self.target_db is not None else None,
        }

    def initialize(self, target_db: Database, query: SqlQuery) -> None:
        self.target_db = target_db

    def _cost_index_op(self, query: SqlQuery, plan: QueryPlan) -> Cost:
        """Try to estimate the cost of an index scan or index-only scan at the root of a specific query plan.

        This method purely exists to keep the rather complex logic of index costing out of the main cost estimation method.
        """
        assert self.target_db is not None

        plan = plan.with_actual_card()
        original_query = self.target_db.hinting().generate_hints(query, plan)
        try:
            cost = self.target_db.optimizer().cost_estimate(original_query)
            return cost
        except (DatabaseServerError, DatabaseUserError):
            pass

        # This did not work, let's try a COUNT(*) query instead:
        # Some database systems (including Postgres) only use indexes for plain queries if they can do something useful with
        # it. We have to trick them.

        count_query = transform.as_count_star_query(query)
        count_query = self.target_db.hinting().generate_hints(count_query, plan)
        try:
            count_plan = self.target_db.optimizer().query_plan(count_query)
            cost = math.nan
        except (DatabaseServerError, DatabaseUserError):
            cost = math.inf

        if math.isinf(cost) and self._raise_on_error:
            raise DatabaseServerError(f"Could not estimate the cost of index plan {plan}.")
        elif math.isinf(cost):
            return cost

        index_node = count_plan.outermost_scan()
        count_plan_matches_original_plan = index_node and index_node.operator == plan.operator
        if not count_plan_matches_original_plan and self._raise_on_error:
            raise DatabaseServerError(f"Could not estimate the cost of index plan {plan}.")
        elif not count_plan_matches_original_plan:
            return math.inf

        return index_node.estimated_cost

    def _cost_materialize_op(self, query: SqlQuery, plan: QueryPlan) -> Cost:
        """Try to estimate the cost of a materialize node at the root of a specific query plan.

        Parameters
        ----------
        query : SqlQuery
            The **entire** query that should be optimized (not just the fragment that should be estimated right now). This
            query has to include additional tables that are not part of the plan. Ensuring that this is actually the case is
            the responsibility of the caller.
        plan : QueryPlan
            The plan that should be estimated. The root node of this plan is expected to be a materialize node.

        Returns
        -------
        Cost
            The estimated cost or *inf* if costing did not work.
        """

        # It is quite difficult to estimate the cost of a materialize node based on the subplan because materialization only
        # happens within a plan and never at the top level. Therefore, we have to improvise a bit here:
        # Our strategy is to create a plan that uses the materialize operator as an inner child and then extract the cost of
        # that node.
        # Since materialization only really happens in front of nested-loop joins, we just construct one of those.
        # Now, to build such an additional join, we need to determine a suitable join partner and and construct a meaningful
        # query for it, which makes up for the lion's share of this method.
        #
        # Since we are going to push a new join node on top of our current plan, we will call the additional join partner and
        # the resulting plan "topped".
        #
        # Our entire strategy is closely aligned with the Postgres planning and execution model. Therefore, we are going to
        # restrict this cost function to Postgres backends.

        if not isinstance(self.target_db, PostgresInterface):
            warnings.warn(
                "Can only estimate the cost of materialize operators for Postgres.",
                category=CostEstimationWarning,
            )
            return math.inf

        # Our join partner has to be a table that is not already part of the plan. Based on these tables, we need to determine
        # all tables that have a suitable join condition with the tables that are already part of the plan.
        free_tables = query.tables() - plan.tables()
        candidate_joins = query.predicates().joins_between(free_tables, plan.tables())
        if not candidate_joins:
            warnings.warn(
                "Could not find a suitable consumer of the materialized table. Returning infinite costs.",
                category=CostEstimationWarning,
            )
            return math.inf
        candidate_tables = free_tables & candidate_joins.tables()

        # Materialization of the child node should always cost the same, no matter what we join afterwards. Therefore, it does
        # not matter which table we choose here.
        topped_table = util.collections.get_any(candidate_tables)

        # Now that we have a table to join with, we can build the updated plan.
        topped_scan = QueryPlan(ScanOperator.SequentialScan, base_table=topped_table)
        topped_plan = QueryPlan(JoinOperator.NestedLoopJoin, children=[topped_scan, plan])

        # Based on the plan we need to construct a suitable query and retrieve its execution plan.
        query_fragment = transform.extract_subquery(query, plan.tables() | {topped_table})
        query_fragment = transform.as_star_query(query_fragment)
        topped_query = self.target_db.hinting().generate_hints(query_fragment, topped_plan)
        try:
            topped_explain = self.target_db.optimizer().query_plan(topped_query)
        except (DatabaseServerError, DatabaseUserError):
            warnings.warn(
                f"Could not estimate the cost of materialize plan {plan}. Returning infinite costs.",
                category=CostEstimationWarning,
            )
            return math.inf

        # Finally, we need to extract the cost estimate of the materialize node.
        intermediate_node = topped_explain.find_first_node(
            lambda node: node.node_type == IntermediateOperator.Materialize and node.tables == plan.tables()
        )
        if not intermediate_node:
            warnings.warn(
                f"Could not estimate cost of materialize plan {plan}. Returning infinite costs.",
                category=CostEstimationWarning,
            )
            return math.inf
        return intermediate_node.estimated_cost

    def _cost_memoize_op(self, query: SqlQuery, plan: QueryPlan) -> Cost:
        """Try to estimate the cost of a memoize node at the root of a specific query plan.

        Parameters
        ----------
        query : SqlQuery
            The **entire** query that should be optimized (not just the fragment that should be estimated right now). This
            query has to include additional tables that are not part of the plan. Ensuring that this is actually the case is
            the responsibility of the caller.
        plan : QueryPlan
            The plan that should be estimated. The root node of this plan is expected to be a memoize node.

        Returns
        -------
        Cost
            The estimated cost or *inf* if costing did not work.
        """

        # It is quite difficult to estimate the cost of a memoize node based on the subplan because memoization only
        # happens within a plan and never at the top level. Therefore, we have to improvise a bit here:
        # Our strategy is to create a plan that uses the memoize operator as an inner child and then extract the cost of
        # that node.
        # Since memoization only really happens in front of nested-loop joins, we just construct one of those.
        # Now, to build such an additional join, we need to determine a suitable join partner and and construct a meaningful
        # query for it. This makes up for the lion's share of this method, even though we can use the plan's lookup key to
        # guide our process.
        #
        # Since we are going to push a new join node on top of our current plan, we will call the additional join partner and
        # the resulting plan "topped".
        #
        # Our entire strategy is closely aligned with the Postgres planning and execution model. Therefore, we are going to
        # restrict this cost function to Postgres backends.

        if not isinstance(self.target_db, PostgresInterface):
            warnings.warn(
                "Can only estimate the cost of memoize operators for Postgres. Returning infinte costs.",
                category=CostEstimationWarning,
            )
            return math.inf

        cache_key = plan.lookup_key
        if not cache_key:
            raise ValueError("Cannot estimate the cost of memoize operators without a lookup key.")
        if not isinstance(cache_key, ColumnExpression):
            warnings.warn(
                "Can only estimate the cost of memoize for single column cache keys. Returning infinite costs.",
                category=CostEstimationWarning,
            )
            return math.inf

        column = cache_key.column
        if not ColumnReference.assert_bound(column):
            warnings.warn(
                "Can only estimate the cost of memoize for cache keys that are bound to a table. Returning infinite costs.",
                category=CostEstimationWarning,
            )
            return math.inf

        # Our join partner has to be a table that is not already part of the plan. Based on these tables, we need to determine
        # all tables that have a suitable join condition with our cache key.
        free_tables = query.tables() - plan.tables()
        candidate_joins = query.predicates().joins_between(free_tables, column.table)
        if not candidate_joins:
            warnings.warn(
                "Could not find a suitable consumer of the materialized table. Returning infinite costs.",
                category=CostEstimationWarning,
            )
            return math.inf
        candidate_tables = free_tables & candidate_joins.tables()

        # Memoization of the child node should always cost the same as long as the same cache key to construct the lookup table
        # is used. Since we enforce this based on the lookup_key it does not matter which table we choose here.
        topped_table = util.collections.get_any(candidate_tables)

        # Now that we have a table to join with, we can build the updated plan.
        topped_scan = QueryPlan(ScanOperator.SequentialScan, base_table=topped_table)
        topped_plan = QueryPlan(JoinOperator.NestedLoopJoin, children=[topped_scan, plan])

        # Based on the plan we need to construct a suitable query and retrieve its execution plan.
        query_fragment = transform.extract_subquery(query, plan.tables() | {topped_table})
        query_fragment = transform.as_star_query(query_fragment)
        topped_query = self.target_db.hinting().generate_hints(query_fragment, topped_plan)
        try:
            topped_explain = self.target_db.optimizer().query_plan(topped_query)
        except (DatabaseServerError, DatabaseUserError):
            warnings.warn(
                f"Could not estimate the cost of memoize plan {plan}. Returning infinite costs.",
                category=CostEstimationWarning,
            )
            return math.inf

        # Finally, we need to extract the cost estimate of the materialize node.
        intermediate_node = topped_explain.find_first_node(
            lambda node: node.node_type == IntermediateOperator.Memoize and node.tables == plan.tables()
        )
        if not intermediate_node:
            warnings.warn(
                f"Could not estimate cost of memoize plan {plan}. Returning infinite costs.",
                category=CostEstimationWarning,
            )
            return math.inf
        return intermediate_node.estimated_cost

    def _cost_sort_op(self, query: SqlQuery, plan: QueryPlan) -> Cost:
        """Try to estimate the cost of a sort node at the root of a specific query plan.

        Parameters
        ----------
        query : SqlQuery
            The query should be estimated. This can be the entire query being optimized or just the part that should be costed
            right now. We don't really care since we are going to extract the relevant bits anyway.
        plan : QueryPlan
            The plan that should be estimated. The root node of this plan is expected to be a sort node.
        """
        assert self.target_db is not None

        # Estimating the cost of a sort node is a bit tricky but not too difficult compared with costing memoize or materialize
        # nodes. The trick is to determine the cost of a modified ORDER BY query which encodes the desired sort order.
        # We just need to be a bit careful because the sort column might not be referenced in the plan, yet, nor must it be
        # present in the query (e.g. for cheap merge joins).

        query_fragment = transform.extract_subquery(query, plan.tables())
        query_fragment = transform.as_star_query(query_fragment)
        target_columns: set[ColumnReference] = util.set_union(
            [self.target_db.schema().columns(tab) for tab in plan.tables()]
        )

        orderby_cols: list[ColumnReference] = []
        for sort_key in plan.sort_keys:
            col = next(
                (
                    col.column
                    for col in sort_key.equivalence_class
                    if isinstance(col, ColumnExpression) and col.column in target_columns
                )
            )
            orderby_cols.append(col)
        orderby_clause = OrderBy.create_for(orderby_cols)
        query_fragment = transform.add_clause(query_fragment, orderby_clause)

        assert plan.input_node
        return self.estimate_cost(query_fragment, plan.input_node)

    def _warn(self, msg: str) -> None:
        if not self._verbose:
            return
        warnings.warn(msg, category=CostEstimationWarning)


class NativeCardinalityEstimator(CardinalityEstimator):
    """Obtains the cardinality of a query plan by using the cardinality estimator of an actual database system."""

    def __init__(self, target_db: Optional[Database] = None) -> None:
        super().__init__(allow_cross_products=True)
        self._target_db: Optional[Database] = target_db

    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        assert self._target_db is not None

        intermediate = util.enlist(intermediate)
        subquery = transform.extract_subquery(query, intermediate)
        subquery = transform.as_star_query(subquery)
        return self._target_db.optimizer().cardinality_estimate(subquery)

    def describe(self) -> jsondict:
        return {
            "name": "native",
            "database_system": self._target_db.describe() if self._target_db is not None else None,
        }

    def initialize(self, target_db: Database, query: SqlQuery) -> None:
        self._target_db = target_db


class NativeJoinOrderOptimizer(JoinOrderOptimization):
    """Obtains the join order for an input query by using the optimizer of an actual database system.

    Parameters
    ----------
    db_instance : db.Database
        The target database whose optimization algorithm should be used.
    """

    def __init__(self, db_instance: Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def optimize_join_order(self, query: SqlQuery) -> Optional[JoinTree]:
        query_plan = self.db_instance.optimizer().query_plan(query)
        return jointree_from_plan(query_plan)

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self.db_instance.describe()}


class NativePhysicalOperatorSelection(PhysicalOperatorSelection):
    """Obtains the physical operators for an input query by using the optimizer of an actual database system.

    Since this process normally is the second stage in the optimization pipeline, the operators are selected according to a
    specific join order. If no such order exists, it is also determined by the database system.

    Parameters
    ----------
    db_instance : db.Database
        The target database whose optimization algorithm should be used.
    """

    def __init__(self, db_instance: Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def select_physical_operators(self, query: SqlQuery, join_order: Optional[JoinTree]) -> PhysicalOperatorAssignment:
        if join_order:
            query = self.db_instance.hinting().generate_hints(query, join_order=join_order)
        query_plan = self.db_instance.optimizer().query_plan(query)
        return operators_from_plan(query_plan)

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self.db_instance.describe()}


class NativePlanParameterization(ParameterGeneration):
    """Obtains the plan parameters for an inpuit querry by using the optimizer of an actual database system.

    This process determines the parameters according to a join order and physical operators. If no such information exists, it
    is also determined by the database system.

    Parameters
    ----------
    db_instance : db.Database
        The target database whose optimization algorithm should be used.
    """

    def __init__(self, db_instance: Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def generate_plan_parameters(
        self,
        query: SqlQuery,
        join_order: Optional[JoinTree],
        operator_assignment: Optional[PhysicalOperatorAssignment],
    ) -> PlanParameterization:
        if join_order or operator_assignment:
            query = self.db_instance.hinting().generate_hints(
                query, join_order=join_order, physical_operators=operator_assignment
            )
        query_plan = self.db_instance.optimizer().query_plan(query)
        return parameters_from_plan(query_plan)

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self.db_instance.describe()}


class NativeOptimizer(CompleteOptimizationAlgorithm):
    """Obtains a complete query execution plan by using the optimizer of an actual database system.

    Parameters
    ----------
    db_instance : db.Database
        The target database whose optimization algorithm should be used.
    """

    def __init__(self, db_instance: Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def optimize_query(self, query: SqlQuery) -> QueryPlan:
        return self.db_instance.optimizer().query_plan(query)

    def describe(self) -> jsondict:
        return {"name": "native", "database_system": self.db_instance.describe()}
