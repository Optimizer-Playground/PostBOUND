from __future__ import annotations

import collections
import itertools
import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Optional

from .. import transform, util
from .._core import (
    Cardinality,
    ColumnReference,
    IntermediateOperator,
    JoinOperator,
    PhysicalOperator,
    ScanOperator,
    TableReference,
)
from .._qep import QueryPlan, SortKey
from .._stages import (
    CardinalityEstimator,
    CostModel,
    OptimizationPreCheck,
    PlanEnumerator,
)
from ..db import Database, DatabasePool, DatabaseSchema, DatabaseServerError
from ..postgres import PostgresInterface, PostgresJoinHints, PostgresScanHints
from ..qal import (
    AbstractPredicate,
    AndPredicate,
    ColumnExpression,
    QueryPredicates,
    SelectStatement,
)
from ..util import LogicError, jsondict
from ..validation import (
    CrossProductPreCheck,
    EquiJoinPreCheck,
    InnerJoinPreCheck,
    SetOperationsPreCheck,
    SubqueryPreCheck,
    VirtualTablesPreCheck,
    merge_checks,
)
from . import native

DPTable = dict[frozenset[TableReference], QueryPlan]


def _calc_plan_estimates(
    query: SelectStatement,
    plan: QueryPlan,
    *,
    cost_model: CostModel,
    cardinality_estimator: CardinalityEstimator,
) -> QueryPlan:
    """Handler method to update the cost and cardinality estimates of a given plan."""
    card_est = cardinality_estimator.calculate_estimate(query, plan.tables())
    plan = plan.with_estimates(cardinality=card_est)
    cost_est = cost_model.estimate_cost(query, plan)
    return plan.with_estimates(cost=cost_est)


def _collect_used_columns(query: SelectStatement, table: TableReference, *, schema: DatabaseSchema) -> set[ColumnReference]:
    columns = query.columns_of(table)
    for star_expression in query.select_clause.star_expressions():
        if table not in star_expression.tables():
            continue

        columns.update(schema.columns(table))
    return columns


class DynamicProgrammingEnumerator(PlanEnumerator):
    """A very basic dynamic programming-based plan enumerator.

    This enumerator is very basic because it does not implement any sophisticated pruning rules or traversal strategies and
    only focuses on a small subset of possible operators. It simply enumerates all possible access paths and join paths and
    picks the cheapest one. This should only serve as a starting point when lacking an actual decent enumerator implementation
    (see *Limitation* below). Its purpose is mainly to shield users that are only interested in the cost model or the
    cardinality estimator from having to implement their own enumerator in order to use the `TextBookOptimizationPipeline`.
    Notice that for experiments based on PostgreSQL, a much more sophisticated implementation is available with the
    `PostgresDynProg` enumerator (and this enumerator is automatically selected when using the textbook pipeline with a
    Postgres target database).

    Limitations
    -----------

    - Only the cheapest access paths are considered, without taking sort orders into account. This prevents free merge join
      optimizations, i.e. if an access path is more expensive but already sorted, it will be discarded in favor of a cheaper
      alternative, even though a later merge join might become much cheaper due to the sort order.
    - No optimizations to intermediates are considered, i.e. no materialization or memoization of subplans.
    - Only the basic scan and join operators are considered. For scans, this includes sequential scan, index scan, index-only
      scan and bitmap scan. For joins, this includes nested loop join, hash join and sort merge join. These can be further
      restricted through the `supported_scan_ops` and `supported_join_ops` parameters.
    - Only simple SPJ queries are supported. Importantly, the query may not contain any set operations, subqueries, CTEs etc.
      All joins must be inner equijoins and no cross products are allowed.
    - Aggregations, sorting, etc. are not considered. In this way, the enumerator is comparable to the ``join_search_hook``
      of PostgreSQL. We assume that such "technicalities" are handled when creating appropriate hints for the target database
      or when executing the query on the target database at the latest.

    Parameters
    ----------
    supported_scan_ops : Optional[set[ScanOperator]], optional
        The set of scan operators that should be considered during the enumeration. This should be a subset of the following
        operators: sequential scan, index scan, index-only scan, bitmap scan. If any other operators are included, these
        are simply never considered. By default all operators that are available on the `target_db` are allowed.
    supported_join_ops : Optional[set[JoinOperator]], optional
        The set of join operators that should be considered during the enumeration. This should be a subset of the following
        operators: nested loop join, hash join, sort merge join. If any other operators are included, these are simply never
        considered. By default all operators that are available on the `target_db` are allowed.
    target_db : Optional[Database], optional
        The target database system for which the optimization pipeline is intended. If not omitted, the database is inferred
        from the `DatabasePool`.
    """

    def __init__(
        self,
        *,
        supported_scan_ops: Optional[set[ScanOperator]] = None,
        supported_join_ops: Optional[set[JoinOperator]] = None,
        target_db: Optional[Database] = None,
    ) -> None:
        target_db = target_db if target_db is not None else DatabasePool.get_instance().current_database()

        supported_scan_ops = supported_scan_ops if supported_scan_ops is not None else set(ScanOperator)
        supported_join_ops = supported_join_ops if supported_join_ops is not None else set(JoinOperator)

        if target_db is not None:
            supported_scan_ops = {op for op in supported_scan_ops if target_db.hinting().supports_hint(op)}
            supported_join_ops = {op for op in supported_join_ops if target_db.hinting().supports_hint(op)}

        self.predicates: QueryPredicates | None = None

        self._target_db = target_db
        self._scan_ops = supported_scan_ops
        self._join_ops = supported_join_ops

    def generate_execution_plan(self, query, *, cost_model, cardinality_estimator) -> QueryPlan:
        self.predicates = query.predicates()
        cost_model.initialize(self._target_db, query)
        cardinality_estimator.initialize(self._target_db, query)

        dp_table = self._determine_base_access_paths(
            query, cost_model=cost_model, cardinality_estimator=cardinality_estimator
        )
        final_plan = self._build_join_paths(
            query,
            dp_table=dp_table,
            cost_model=cost_model,
            cardinality_estimator=cardinality_estimator,
        )

        cost_model.cleanup()
        cardinality_estimator.cleanup()
        self.predicates = None
        return final_plan

    def pre_check(self) -> OptimizationPreCheck:
        return merge_checks(
            CrossProductPreCheck(),
            VirtualTablesPreCheck(),
            EquiJoinPreCheck(),
            InnerJoinPreCheck(),
            SubqueryPreCheck(),
            SetOperationsPreCheck(),
        )

    def describe(self) -> jsondict:
        return {
            "name": "dynamic_programming",
            "flavor": "default",
            "scan_ops": [op.name for op in self._scan_ops],
            "join_ops": [op.name for op in self._join_ops],
            "database_system": self._target_db.describe(),
        }

    def _determine_base_access_paths(
        self,
        query: SelectStatement,
        *,
        cost_model: CostModel,
        cardinality_estimator: CardinalityEstimator,
    ) -> DPTable:
        """Initializes a new dynamic programming table which includes the cheapest access paths for each base table.

        The base tables are directly inferred from the query.
        """
        assert self.predicates is not None
        dp_table: DPTable = {}

        for table in query.tables():
            # We determine access paths in two phases: initially, we just gather all possible access paths to a specific table.
            # Aftewards, we evaluate these candidates according to our cost model and select the cheapest one.
            candidate_plans: list[QueryPlan] = []
            filter_condition = self.predicates.filters_for(table)

            if ScanOperator.SequentialScan in self._scan_ops:
                candidate_plans.append(
                    QueryPlan(
                        ScanOperator.SequentialScan,
                        base_table=table,
                        filter_predicate=filter_condition,
                    )
                )
            candidate_plans += self._determine_index_paths(query, table)

            candidate_plans = [
                _calc_plan_estimates(
                    query,
                    candidate,
                    cost_model=cost_model,
                    cardinality_estimator=cardinality_estimator,
                )
                for candidate in candidate_plans
            ]

            cheapest_plan = min(candidate_plans, key=lambda plan: plan.estimated_cost)
            dp_table[frozenset([table])] = cheapest_plan

        return dp_table

    def _determine_index_paths(self, query: SelectStatement, table: TableReference) -> Iterable[QueryPlan]:
        """Gathers all possible index access paths for a specific table.

        The access paths do not contain a cost or cardinality estimates, yet. These information must be added by the caller.
        """
        filter_condition = query.predicates().filters_for(table)
        required_columns = _collect_used_columns(query, table, schema=self._target_db.schema())
        can_idx_only_scan = len(required_columns) <= 1  # check for <= 1 to include cross products with select star
        candidate_indexes = {column: self._target_db.schema().indexes_on(column) for column in required_columns}

        if not candidate_indexes:
            return []

        candidate_plans: list[QueryPlan] = []
        for column, available_indexes in candidate_indexes.items():
            if not available_indexes:
                continue
            sorting = [SortKey.of(column)]

            for index in available_indexes:
                if ScanOperator.IndexScan in self._scan_ops:
                    candidate_plans.append(
                        QueryPlan(
                            ScanOperator.IndexScan,
                            base_table=table,
                            index=index,
                            sort_keys=sorting,
                            filter_predicate=filter_condition,
                        )
                    )
                if can_idx_only_scan and ScanOperator.IndexOnlyScan in self._scan_ops:
                    candidate_plans.append(
                        QueryPlan(
                            ScanOperator.IndexOnlyScan,
                            base_table=table,
                            index=index,
                            sort_keys=sorting,
                            filter_predicate=filter_condition,
                        )
                    )

        if ScanOperator.BitmapScan in self._scan_ops:
            # The target DB/cost model is responsible for figuring out good bitmap index hierarchies.
            # Since bitmap scans combine multiple indexes, we do not consider bitmap scans in the above loop.
            # Furthermore, bitmap scans are partial sequential scans and thus do not provide a sort key.
            candidate_plans.append(
                QueryPlan(
                    ScanOperator.BitmapScan,
                    base_table=table,
                    indexes=candidate_indexes,
                    filter_predicate=filter_condition,
                )
            )

        return candidate_plans

    def _build_join_paths(
        self,
        query: SelectStatement,
        *,
        dp_table: DPTable,
        cost_model: CostModel,
        cardinality_estimator: CardinalityEstimator,
    ) -> QueryPlan:
        """Main optimization loop for the dynamic programmer.

        In this loop we construct increasingly large join paths by combining the optimal access paths of their input relations.
        At the end of the loop we have just constructed the cheapest join path for the entire query.

        All access paths are stored in the `dp_table`. This method assumes that the `dp_table` already contains the cheapest
        access paths for all base relations.

        Returns
        -------
        QueryPlan
            The final query plan that represents the cheapest join path for the given query.
        """

        predicates = query.predicates()
        candidate_tables = query.tables()

        for current_level in range(2, len(candidate_tables) + 1):
            # The current level describes how large the intermediate join paths that we are considering next are going to be.
            # For each potential intermediate that matches the current level, we determine the cheapest access path. This path
            # is going to re-use the cheapest access paths that we determined as part of an earlier iteration.

            current_intermediates = itertools.combinations(candidate_tables, current_level)
            access_paths = {
                frozenset(join): self._determine_cheapest_path(
                    query,
                    join,
                    dp_table=dp_table,
                    cost_model=cost_model,
                    cardinality_estimator=cardinality_estimator,
                )
                for join in current_intermediates
                if predicates.joins_tables(join)  # we do not consider cross products
            }
            dp_table.update(access_paths)

        return dp_table[frozenset(candidate_tables)]

    def _determine_cheapest_path(
        self,
        query: SelectStatement,
        intermediate: Iterable[TableReference],
        *,
        dp_table: DPTable,
        cost_model: CostModel,
        cardinality_estimator: CardinalityEstimator,
    ) -> QueryPlan:
        """DP subroutine that selects the cheapest access path for a specific intermediate."""
        intermediate = frozenset(intermediate)
        candidate_plans: list[QueryPlan] = []

        # We determine the cheapest access path to our intermediate by checking all potential join partners that could possibly
        # be used to construct this intermediate. This works by splitting the intermediate into an outer relation and an inner
        # one. To guarantee that we test each possible split, we generate the entire power set of the intermediate.
        # By basing our algorithm on the power set we can solve two important problems: first, we can easily generate bushy
        # plans (each time the outer plan has at least two tables and leaves more than one table for the inner plan results in
        # a bushy plan). Second, we can also generate plans that invert the role of inner and outer relation just as easy.
        # This is because the power set will eventually visit the set of all tables in the (former) inner relation, which will
        # then become the outer relation for the current iteration.
        #
        # Once again, we first gather all possible join paths and then evaluate the costs for each of them in order to select
        # the cheapest one.

        for outer in util.collections.powerset(intermediate):
            if not outer or len(outer) == len(intermediate):
                # Skip the empty set and the full set because we would lack a join partner.
                continue
            outer = frozenset(outer)

            # All tables of our intermediate that are not part of the outer relation have to become part of the inner relation
            inner = intermediate - outer

            outer_plan, inner_plan = dp_table.get(outer), dp_table.get(inner)
            if not outer_plan or not inner_plan:
                # If we do not find the access paths for one of our inputs, it means that this is constructed using a cross
                # product. Since we do not consider cross products, we can skip this split.
                continue

            join_condition = query.predicates().joins_between(outer, inner)

            if JoinOperator.NestedLoopJoin in self._join_ops:
                candidate_plans.append(
                    QueryPlan(
                        JoinOperator.NestedLoopJoin,
                        children=[outer_plan, inner_plan],
                        join_condition=join_condition,
                    )
                )

            if JoinOperator.HashJoin in self._join_ops:
                candidate_plans.append(
                    QueryPlan(
                        JoinOperator.HashJoin,
                        children=[outer_plan, inner_plan],
                        join_condition=join_condition,
                    )
                )

            if JoinOperator.SortMergeJoin in self._join_ops:
                # The target DB is utimately responsible for figuring out whether it needs explicit sorts or whether it can
                # just merge directly.
                candidate_plans.append(
                    QueryPlan(
                        JoinOperator.SortMergeJoin,
                        children=[outer_plan, inner_plan],
                        join_condition=join_condition,
                    )
                )

        candidate_plans = [
            _calc_plan_estimates(
                query,
                candidate,
                cost_model=cost_model,
                cardinality_estimator=cardinality_estimator,
            )
            for candidate in candidate_plans
        ]

        return min(candidate_plans, key=lambda plan: plan.estimated_cost)


@dataclass
class RelOptInfo:
    """Simplified model of the RelOptInfo from the Postgres planner.

    We only specify the fields that we truly care about (and that are not covered by other parts of PostBOUND), the rest is
    omitted.

    For example, we don't need to worry about equivalence classes, because the Postgres enumerator is responsible for
    expanding the query with all EQ predicates. Aftwards, we can use the query abstraction to determine available joins.
    """

    intermediate: frozenset[TableReference]
    """The relation that is represented by this RelOptInfo.

    This is simply the set of all tables that are part of the relation.
    """

    pathlist: list[QueryPlan]
    """All access paths that can be used to compute this relation (that we know of and care about).

    In contrast to the original PG implementation, we don't care about sorting this list. Retaining the sort order is mainly
    an implementation detail and optimization of PG.
    """

    partial_paths: list[QueryPlan]
    """All access paths that can be used to compute this relation with parallel workers. Otherwise the same as `paths`."""

    cheapest_path: Optional[QueryPlan]
    """The cheapest access path that we have found.

    Notice that this is only set after all paths for the RelOpt have been collected.
    """

    cheapest_partial_path: Optional[QueryPlan]
    """The cheapest access path that we have found for parallel execution.

    Notice that this is only set after all paths for the RelOpt have been collected.
    """

    cardinality: Cardinality
    """The estimated number of rows that are produced by this relation."""

    def __contains__(self, item: object) -> bool:
        match item:
            case RelOptInfo():
                item = item.intermediate
            case TableReference():
                item = frozenset([item])
            case _:
                raise TypeError(f"Cannot check membership of {type(item)} in RelOptInfo.")

        return item < self.intermediate

    def __hash__(self) -> int:
        return hash(self.intermediate)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RelOptInfo) and self.intermediate == other.intermediate

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        tables = ", ".join(sorted(t.identifier() for t in self.intermediate))
        return f"{{{tables}}}"


Sorting = Sequence[SortKey]
"""A specific sort order for some relation."""

Level = int
"""The current level in the dynamic programming table."""

JoinRelLevel = dict[Level, list[RelOptInfo]]
"""Alias for our dynamic programming table."""

AddPathHook = Callable[["PostgresDynProg", RelOptInfo, QueryPlan, bool], None]
"""Hook method for users to get control in Postgres' *add_path()* method.

The method is reponsible for storing a new candidate path in its `RelOptInfo`. It can decide, whether the path is actually
worth storing or not. Furthermore, the method can also prune existing paths from the `pathlist` that are dominated by the
new path.

All of these actions should be performed in-place by modifying the `RelOptInfo` object. No return value is expected.

The last boolean parameter indicates whether the path is a partial path (i.e. a path for parallel execution) or not.
If it is, the path should be stored in the `partial_paths` list, instead of the standard `pathlist`. Likewise, all checks
should be performed against the `partial_paths` list. If the hook should not handle partial paths, it can simply delegate
to the standard implementation on the enumerator.

If need be, the method can also access the current state of the dynamic programmer. Specifically, the enumerator provides
access to the query and database, the selected cost model and cardinality estimator, as well as to the current `JoinRelLevel`.
Finally, the method is allowed to invoke the default path addition logic by calling the `standard_add_path()` method on the
enumerator.
"""


class DPWarning(UserWarning):
    pass


class PostgresDynProg(PlanEnumerator):
    """Dynamic programming-based plan enumeration strategy that mimics the behavior of the Postgres query optimizer.

    Postgres-style dynamic programming means two things: first, we use the Postgres pruning rules to reduce the search space.
    Second, we apply the same opinionated traversal rules. Most importantly, this concerns when we consider materialization or
    memoization of subplans. If some of the related operators are not allowed, the traversal rules are adjusted accordingly.

    The implementation is based on a translation of the actual Postgres source code.

    Parameters
    ----------
    supported_scan_ops : Optional[set[ScanOperators]], optional
        The scan operators that the enumerator is allowed to use. If omitted, all scan operators that are supported by the
        target database are used.
    supported_join_ops : Optional[set[JoinOperators]], optional
        The join operators that the enumerator is allowed to use. If omitted, all join operators supported by the target
        database are used.
    enable_materialize : bool, optional
        Whether the optimizer is allowed to insert materialization operators into the query plan. This is enabled by default.
    enable_memoize : bool, optional
        Whether the optimizer is allowed to insert memoization operators into the query plan. This is enabled by default.
    enable_sort : bool, optional
        Whether the optimizer is allowed to perform explicit sorts in the query plan. Notice that setting this to *False* only
        prevents optional sorts. For example, if the query contains an *ORDER BY* clause, the optimizer will still perform the
        required sorting. However, it will not perform any merge joins that require a different kind of sorting.
        This is enabled by default.
    add_path_hook : Optional[AddPathHook], optional
        Optional function to implement custom path addition logic. See documentation on `AddPathHook` for more details.
    target_db : Optional[PostgresInterface], optional
        The database on which the plans should be executed. This has to be a Postgres instance. If omitted, the database is
        inferred from the `DatabasePool`.
    verbose : bool, optional
        Whether the enumerator should issue warnings if it encounters unexpected situations, e.g. if it rejects a path
        because it is illegal. This includes cases where the cost of some operators cannot be estimated by the target
        database system.
    """

    def __init__(
        self,
        *,
        supported_scan_ops: Optional[set[ScanOperator]] = None,
        supported_join_ops: Optional[set[JoinOperator]] = None,
        enable_materialize: bool = True,
        enable_memoize: bool = True,
        enable_sort: bool = True,
        max_parallel_workers: Optional[int] = None,
        add_path_hook: Optional[AddPathHook] = None,
        target_db: Optional[PostgresInterface] = None,
        verbose: bool = False,
    ) -> None:
        if target_db is None:
            fallback = DatabasePool.get_instance().current_database()
            target_db = fallback if isinstance(fallback, PostgresInterface) else None

        if not isinstance(target_db, PostgresInterface):
            raise LogicError(
                "The PostgresDynProg enumerator can only be used with a Postgres database. "
                "(but you can execute the plans on any database that supports the required hints)."
            )

        supported_scan_ops = supported_scan_ops if supported_scan_ops is not None else PostgresScanHints
        supported_join_ops = supported_join_ops if supported_join_ops is not None else PostgresJoinHints

        if target_db is not None:
            supported_scan_ops = {op for op in supported_scan_ops if target_db.hinting().supports_hint(op)}
            supported_join_ops = {op for op in supported_join_ops if target_db.hinting().supports_hint(op)}

        self.query: SelectStatement | None = None
        self.predicates: QueryPredicates | None = None
        self.cost_model: CostModel | None = None
        self.cardinality_estimator: CardinalityEstimator | None = None
        self.join_rel_level: JoinRelLevel | None = None
        self.target_db = target_db

        self._scan_ops = supported_scan_ops
        self._join_ops = supported_join_ops
        self._enable_materialize = enable_materialize
        self._enable_memoize = enable_memoize
        self._enable_sort = enable_sort
        self._max_workers = max_parallel_workers if max_parallel_workers else 0
        self._add_path_hook = add_path_hook

        self._verbose = verbose

    def infer_settings(self) -> None:
        """Sets all allowed operators according to the configuration of the target database.

        For example, if the target database has index scans disabled, the enumerator will disable them as well.
        """
        allowed_scan_ops: list[ScanOperator] = []
        if self.target_db.config["enable_seqscan"] == "on":
            allowed_scan_ops.append(ScanOperator.SequentialScan)
        if self.target_db.config["enable_indexscan"] == "on":
            allowed_scan_ops.append(ScanOperator.IndexScan)
        if self.target_db.config["enable_indexonlyscan"] == "on":
            allowed_scan_ops.append(ScanOperator.IndexOnlyScan)
        if self.target_db.config["enable_bitmapscan"] == "on":
            allowed_scan_ops.append(ScanOperator.BitmapScan)
        self._scan_ops = set(allowed_scan_ops)

        allowed_join_ops: list[JoinOperator] = []
        if self.target_db.config["enable_nestloop"] == "on":
            allowed_join_ops.append(JoinOperator.NestedLoopJoin)
        if self.target_db.config["enable_hashjoin"] == "on":
            allowed_join_ops.append(JoinOperator.HashJoin)
        if self.target_db.config["enable_mergejoin"] == "on":
            allowed_join_ops.append(JoinOperator.SortMergeJoin)
        self._join_ops = set(allowed_join_ops)

        self._enable_materialize = self.target_db.config["enable_material"] == "on"
        self._enable_memoize = self.target_db.config["enable_memoize"] == "on"
        self._enable_sort = self.target_db.config["enable_sort"] == "on"

        self._max_workers = int(self.target_db.config["max_parallel_workers_per_gather"])

    def generate_execution_plan(self, query, *, cost_model, cardinality_estimator) -> QueryPlan:
        self.query = transform.add_ec_predicates(query)
        self.predicates = self.query.predicates()

        cardinality_estimator.initialize(self.target_db, query)
        cost_model.initialize(self.target_db, query)
        self.cardinality_estimator = cardinality_estimator
        self.cost_model = cost_model

        base_rels = self._init_base_rels()
        self._set_base_rel_pathlists(base_rels)

        final_rel = self._standard_join_search(initial_rels=base_rels)
        assert final_rel.cheapest_path is not None, "No valid plan found for the given query."

        if self._is_pg_cost(cost_model):
            # This seems weird at first, so let's explain what is going on here:
            # If we use the actual PG cost model, we can be make better decisions about what the optimal plan is.
            # The reason is that PG might parallelize a portion of the upper operators (anything that happens after the final
            # join such as aggregations). With these upper rel parallelizations, a partial plan might become cheaper than its
            # counterpart that executes the last join in parallel (and which we compute). Consequently, that very path might
            # become cheaper than the cheapest path that we have computed so far.
            #
            # Therefore, we need to explicitly consider all partial paths as well as the sequential ones and retrieve the
            # final cost estimates for them.
            # But this only works if know we are using the PG execution engine and cost model.

            plans = final_rel.pathlist + self._generate_pseudo_gather_paths(final_rel)
            plan_costs = {plan: self._pg_cost_estimate(plan) for plan in plans}
            cheapest_plan = util.argmin(plan_costs)
        else:
            cheapest_plan = final_rel.cheapest_path

        cost_model.cleanup()
        cardinality_estimator.cleanup()
        self.query = None
        self.predicates = None
        self.cost_model = None
        self.cardinality_estimator = None
        return cheapest_plan

    def describe(self) -> jsondict:
        return {
            "name": "dynamic_programming",
            "flavor": "postgres",
            "scan_ops": [op.name for op in self._scan_ops],
            "join_ops": [op.name for op in self._join_ops],
            "database_system": self.target_db.describe(),
        }

    def pre_check(self) -> OptimizationPreCheck:
        return merge_checks(
            CrossProductPreCheck(),
            EquiJoinPreCheck(),
            InnerJoinPreCheck(),
            VirtualTablesPreCheck(),
            SetOperationsPreCheck(),
        )

    def standard_add_path(self, rel: RelOptInfo, path: QueryPlan, *, is_partial: bool = False) -> None:
        """Checks, whether a specific path is worthy of further consideration. If it is, the path is stored in the pathlist.

        This method's naming is exceptionally bad, but this the way it is named in the PG source code, so we stick with it.

        On an abstract level, this method implements the following logic:

        For each existing path in the pathlist, we check whether the new path dominates the existing one.
        If it does, we evict the existing path. It the existing path is better, we keep it and discard the new path.

        To determine, whether one path dominates another, we compare the paths' costs and sort orders. For one path to
        dominate the other one, it must be cheaper and at least as good sorted.
        If the paths are sorted differently, we keep them both.

        This basic logic is executed for both regular paths and partial paths. Those are paths that can be executed in parallel
        and will eventually be merged with the main paths (once the parallel portion is finished).
        While "normal" paths (completely sequential ones or parallel paths that have finished their parallel portion) are
        stored in the `pathlist`, partial paths are stored in the `partial_paths` list.
        """
        current_paths = rel.partial_paths if is_partial else rel.pathlist
        if not current_paths:
            current_paths.append(path)
            return

        result_paths: list[QueryPlan] = []
        keep_new = True  # we assume that we want to keep the new path to handle new sort orders correctly
        new_cost = path.estimated_cost

        for i, old_path in enumerate(current_paths):
            if not self._sorting_subsumes(path.sort_keys, other=old_path.params.sort_keys):
                result_paths.append(old_path)
                continue

            # Postgres uses a fuzzy cost comparison (compare_path_costs_fuzzily() from pathnode.c) and evicts old paths even
            # if there cost is slightly better than the new path, if the new path is better sorted.
            old_cost = old_path.estimated_cost
            new_dominates = (
                new_cost < old_cost
                if self._same_sorting(path.sort_keys, other=old_path.sort_keys)
                else new_cost <= 1.01 * old_cost
            )

            if new_dominates:
                # The new path is better (or at least equally) sorted and cheaper, we can evict the old path
                keep_new = True  # strictly speaking, this is not necessary, but it makes our intention clearer
                continue  # don't break here, we need to check the remaining paths
            else:
                # The existing path is better (or at least equally) sorted and cheaper, we don't need the new path.
                # This also means that we can stop checking the remaining paths. The new one won't get added and the old ones
                # successfully competed against the path that just beat our new one. We can keep them all.
                result_paths.extend(current_paths[i:])
                keep_new = False
                break

        if keep_new:
            result_paths.append(path)

        if is_partial:
            rel.partial_paths = result_paths
        else:
            rel.pathlist = result_paths

    def _init_base_rels(self) -> list[RelOptInfo]:
        """Creates and initializes the RelOptInfos for all tables in the query, without computing any access paths."""

        # Combines logic from make_one_rel() and set_base_rel_sizes()
        assert self.query is not None and self.cardinality_estimator is not None
        initial_rels: list[RelOptInfo] = []

        for base_rel in self.query.tables():
            intermediate = frozenset([base_rel])
            cardinality = self.cardinality_estimator.calculate_estimate(self.query, intermediate)

            initial_rels.append(
                RelOptInfo(
                    intermediate=intermediate,
                    pathlist=[],
                    partial_paths=[],
                    cheapest_path=None,
                    cardinality=cardinality,
                    cheapest_partial_path=None,
                )
            )

        return initial_rels

    def _set_base_rel_pathlists(self, initial_rels: list[RelOptInfo]) -> None:
        """Adds access paths to the base relations.

        The specific paths depend on the available operators and the current schema.
        """
        # This function leads to a larger chain of function calls which in end culminate in set_plain_rel_pathlist()
        # We implement the behavior of that function here

        for rel in initial_rels:
            self._create_sequential_paths(rel)
            self._create_index_paths(rel)
            self._create_bitmap_path(rel)

            self._generate_gather_paths(rel)
            self._set_cheapest(rel)

    def _standard_join_search(self, initial_rels: list[RelOptInfo]) -> RelOptInfo:
        """Main entry point into the dynamic programming join search.

        The implementation assumes that the `initial_rels` have already been initialized. Therefore, the dynamic programmer
        is only concerned with building join rels.
        """
        levels_needed = len(initial_rels)
        self.join_rel_level: JoinRelLevel = collections.defaultdict(list)
        self.join_rel_level[1] = initial_rels

        for level in range(2, levels_needed + 1):
            self._join_search_one_level(level)

            for rel in self.join_rel_level[level]:
                self._generate_gather_paths(rel)
                self._set_cheapest(rel)

        assert len(self.join_rel_level[levels_needed]) == 1, "Final join rel level should only contain one relation."
        final_rel = self.join_rel_level[levels_needed][0]
        self.join_rel_level = None
        return final_rel

    def _join_search_one_level(self, level: Level) -> None:
        """Handler method to construct all intermediates of a current level for the DP join search.

        Parameters
        ----------
        level : int
            The number of base tables that should be contained in each intermediate relation that we will construct.
        """
        assert self.join_rel_level is not None and self.predicates is not None

        # First, consider left-deep plans
        for rel1 in self.join_rel_level[level - 1]:
            # the body of this loop implements the logic of make_join_rel() (which is called by make_rels_by_clause_joins())

            for rel2 in self.join_rel_level[1]:
                if len(rel1.intermediate & rel2.intermediate) > 0:
                    # don't join anything that we have already joined
                    continue
                if not self.predicates.joins_between(rel1.intermediate, rel2.intermediate):
                    # don't consider cross products
                    continue

                # functionality of build_join_rel()
                intermediate = rel1.intermediate | rel2.intermediate
                join_rel = self._build_join_rel(intermediate)

                # functionality of populate_joinrel_with_paths()
                self._add_paths_to_joinrel(join_rel, outer_rel=rel1, inner_rel=rel2)
                self._add_paths_to_joinrel(join_rel, outer_rel=rel2, inner_rel=rel1)

        for outer_size in range(2, level // 2 + 1):
            inner_size = level - outer_size
            if inner_size < 2:
                continue

            for rel1 in self.join_rel_level[outer_size]:
                for rel2 in self.join_rel_level[inner_size]:
                    if len(rel1.intermediate & rel2.intermediate) > 0:
                        # don't join anything that we have already joined
                        continue
                    if not self.predicates.joins_between(rel1.intermediate, rel2.intermediate):
                        # don't consider cross products
                        continue

                    # functionality of build_join_rel()
                    intermediate = rel1.intermediate | rel2.intermediate
                    join_rel = self._build_join_rel(intermediate)

                    # functionality of populate_joinrel_with_paths()
                    self._add_paths_to_joinrel(join_rel, outer_rel=rel1, inner_rel=rel2)
                    self._add_paths_to_joinrel(join_rel, outer_rel=rel2, inner_rel=rel1)

    def _build_join_rel(self, intermediate: frozenset[TableReference]) -> RelOptInfo:
        """Constructs and initializes a new RelOptInfo for a specific intermediate. No access paths are added, yet."""
        assert self.join_rel_level is not None and self.cardinality_estimator is not None and self.query is not None

        # This function integrates the logic of find_join_rel() and build_join_rel()
        level = len(intermediate)
        for rel in self.join_rel_level[level]:
            if rel.intermediate == intermediate:
                return rel

        cardinality = self.cardinality_estimator.calculate_estimate(self.query, intermediate)
        join_rel = RelOptInfo(
            intermediate=intermediate,
            pathlist=[],
            partial_paths=[],
            cheapest_path=None,
            cardinality=cardinality,
            cheapest_partial_path=None,
        )

        self.join_rel_level[level].append(join_rel)
        return join_rel

    def _add_paths_to_joinrel(self, join_rel: RelOptInfo, *, outer_rel: RelOptInfo, inner_rel: RelOptInfo) -> None:
        """Builds all possible access paths for a specific join relation.

        The build process adheres to the assignment of join directions from the parameters, i.e. the `outer_rel` will always be
        the outer relation and the `inner_rel` will always be the inner relation. If it does not matter, what the specific
        assignment is, this method has to be called twice with inversed parameters.
        """
        if JoinOperator.NestedLoopJoin in self._join_ops:
            self._match_unsorted_outer(join_rel, outer_rel=outer_rel, inner_rel=inner_rel)
        if JoinOperator.SortMergeJoin in self._join_ops:
            self._sort_inner_outer(join_rel, outer_rel=outer_rel, inner_rel=inner_rel)
        if JoinOperator.HashJoin in self._join_ops:
            self._hash_inner_outer(join_rel, outer_rel=outer_rel, inner_rel=inner_rel)

    def _sort_inner_outer(self, join_rel: RelOptInfo, *, outer_rel: RelOptInfo, inner_rel: RelOptInfo) -> None:
        """Constructs all potential merge join paths for a specific intermediate.

        This method assumes that merge joins are actually enabled.
        """

        # The implementation of this function is loosely based on sort_inner_outer() of the PG source code.
        # However, since the original function is tightly coupled with Postgres' internal query and planner representation, we
        # deviate a bit more than usual from the original implementation.
        #
        # Specifically, our implementation performs the following high-level algorithm:
        # For each potential join key between the input relations, we check whether they are already sorted based on the join
        # key. If they are not, we introduce an explicit sort operator for the path. Afterwards, we create a merge join based
        # on the candidate paths.
        #
        # As a consequence, this function essentially also implements the merge-join specific behavior of
        # match_unsorted_outer(). In our implementation, that function only handles nested-loop joins.
        #
        # Notice that Postgres does not consider materialization or memoization of subpaths for merge joins, so neither do we.

        join_keys = self._determine_join_keys(outer_rel=outer_rel, inner_rel=inner_rel)

        # How often can we nest? Quite often! But this should still be fairly readable.
        # We simply loop over all join keys. For each join key, we try each combination of inner and outer relations and
        # see if we end up with a decent merge join path.
        for join_key in join_keys:
            outer_col, inner_col = self._extract_join_columns(join_key, outer_rel=outer_rel, inner_rel=inner_rel)
            if not outer_col or not inner_col:
                continue

            for outer_path in outer_rel.pathlist:
                if not self._is_sorted_by(outer_path, outer_col) and not self._enable_sort:
                    # If the path is not already sorted and we are not allowed to sort it ourselves, there is no point in
                    # merge joining. Just skip the path.
                    continue

                outer_path = (
                    outer_path
                    if self._is_sorted_by(outer_path, outer_col)
                    else self._create_sort_path(outer_path, sort_key=outer_col)
                )

                for inner_path in inner_rel.pathlist:
                    if not self._is_sorted_by(inner_path, inner_col) and not self._enable_sort:
                        # If the path is not already sorted and we are not allowed to sort it ourselves, there is no point in
                        # merge joining. Just skip the path.
                        continue

                    inner_path = (
                        inner_path
                        if self._is_sorted_by(inner_path, inner_col)
                        else self._create_sort_path(inner_path, sort_key=inner_col)
                    )

                    merge_path = self._create_mergejoin_path(join_rel, outer_path=outer_path, inner_path=inner_path)
                    self._add_path(join_rel, merge_path)

            for outer_partial in outer_rel.partial_paths:
                # same as above, just as for partial paths - we try each combination of partial outer with regular inner
                if not self._is_sorted_by(outer_partial, outer_col) and not self._enable_sort:
                    continue

                outer_path = (
                    outer_path
                    if self._is_sorted_by(outer_partial, outer_col)
                    else self._create_sort_path(outer_partial, sort_key=outer_col)
                )

                for inner_path in inner_rel.pathlist:
                    if not self._is_sorted_by(inner_path, inner_col) and not self._enable_sort:
                        continue

                    inner_path = (
                        inner_path
                        if self._is_sorted_by(inner_path, inner_col)
                        else self._create_sort_path(inner_path, sort_key=inner_col)
                    )

                    merge_path = self._create_mergejoin_path(join_rel, outer_path=outer_path, inner_path=inner_path)
                    self._add_path(join_rel, merge_path, is_partial=True)

    def _match_unsorted_outer(self, join_rel: RelOptInfo, *, outer_rel: RelOptInfo, inner_rel: RelOptInfo) -> None:
        """Constructs all potential nested loop-join paths for a specific intermediate.

        This also includes adding paths with memoization or materialization if they are allowed and appear useful.

        This method assumes that nested loop joins are actually enabled.
        """
        assert self.predicates is not None

        # as outlined in _sort_inner_outer(), we only handle nested-loop joins here
        # Nested-loop joins are inherently unsorted, so we only care about the cheapest access paths to the input relations
        # here.

        outer_path, inner_path = outer_rel.cheapest_path, inner_rel.cheapest_path
        if not outer_path or not inner_path:
            raise LogicError("No cheapest paths set")

        # Try plain NLJ first, variations (memoization/materialization) afterwards
        nlj_path = self._create_nestloop_path(join_rel, outer_path=outer_path, inner_path=inner_path)
        self._add_path(join_rel, nlj_path)

        if self._enable_memoize:
            # For memoization, we attempt to cache each potential join key for the inner relation. Since there might be
            # multiple such keys, especially for larger intermediates, we need to check multiple

            join_predicate = self.predicates.joins_between(outer_rel.intermediate, inner_rel.intermediate)
            if join_predicate is None:
                raise LogicError(
                    "Cross product detected. This should never happen so deep down in the optimization process. "
                    f"Intermediates {outer_rel} and {inner_rel}"
                )

            for first_col, second_col in join_predicate.join_partners():
                cache_key = first_col if first_col.table in inner_rel.intermediate else second_col
                if cache_key.table not in inner_rel.intermediate:
                    raise LogicError(
                        "Cache key must be part of the inner relation.",
                        f"Key was {cache_key}, relation was {inner_rel}",
                    )

                memo_inner = self._create_memoize_path(inner_path, cache_key=cache_key)
                memo_nlj = self._create_nestloop_path(join_rel, outer_path=outer_path, inner_path=memo_inner)
                self._add_path(join_rel, memo_nlj)

        if self._enable_materialize:
            mat_path = self._create_materialize_path(inner_path)
            mat_nlj = self._create_nestloop_path(join_rel, outer_path=outer_path, inner_path=mat_path)
            self._add_path(join_rel, mat_nlj)

        outer_partial = outer_rel.cheapest_partial_path
        if not outer_partial:
            return
        par_nlj = self._create_nestloop_path(join_rel, outer_path=outer_partial, inner_path=inner_path)
        self._add_path(join_rel, par_nlj, is_partial=True)

        if self._enable_memoize:
            # same as above
            join_predicate = self.predicates.joins_between(outer_rel.intermediate, inner_rel.intermediate)
            if join_predicate is None:
                raise LogicError(
                    "Cross product detected. This should never happen so deep down in the optimization process. "
                    f"Intermediates {outer_rel} and {inner_rel}"
                )

            for first_col, second_col in join_predicate.join_partners():
                cache_key = first_col if first_col.table in inner_rel.intermediate else second_col
                if cache_key.table not in inner_rel.intermediate:
                    raise LogicError(
                        "Cache key must be part of the inner relation.",
                        f"Key was {cache_key}, relation was {inner_rel}",
                    )

                memo_inner = self._create_memoize_path(inner_path, cache_key=cache_key)
                par_nlj = self._create_nestloop_path(join_rel, outer_path=outer_partial, inner_path=memo_inner)
                self._add_path(join_rel, par_nlj, is_partial=True)

    def _hash_inner_outer(self, join_rel: RelOptInfo, *, outer_rel: RelOptInfo, inner_rel: RelOptInfo) -> None:
        """Constructs the hash join path for a specific intermediate.

        In contrast to merge joins and nested loop joins, there is really only one way to perform a hash join.

        This method assumes that hash joins scans are actually enabled.
        """

        # Hash joins are inherently unsorted, so we only care about the cheapest access paths to the input relations here.
        # Notice that Postgres does not consider materialization or memoization of subpaths for hash joins, so neither do we.

        outer_path, inner_path = outer_rel.cheapest_path, inner_rel.cheapest_path
        if not outer_path or not inner_path:
            raise LogicError("No cheapest paths set")
        hash_path = self._create_hashjoin_path(join_rel, outer_path=outer_path, inner_path=inner_path)
        self._add_path(join_rel, hash_path)

        outer_partial = outer_rel.cheapest_partial_path
        if not outer_partial:
            return
        par_hash = self._create_hashjoin_path(join_rel, outer_path=outer_partial, inner_path=inner_path)
        self._add_path(join_rel, par_hash, is_partial=True)

    def _add_path(self, rel: RelOptInfo, path: QueryPlan, *, is_partial: bool = False) -> None:
        """Checks, whether a specific path is worthy of further consideration. If it is, the path is stored in the pathlist.

        This method's naming is exceptionally bad, but this the way it is named in the PG source code, so we stick with it.

        If an `_add_path_hook` has been specified, this hook takes control after checking for illegal paths. The normal
        path adding logic is skipped in this case. Otherwise, we call the standard path adding logic from
        `_standard_add_path()`.

        The method handles both regular paths and partial paths. Which path it is can be switched using the `is_partial`
        parameter.
        """

        if math.isinf(path.estimated_cost):
            # The cost model returns infinite costs for illegal query plans.
            self._warn(f"Rejecting illegal path {path}")
            return

        if self._add_path_hook:
            self._add_path_hook(self, rel, path, is_partial)
        else:
            self.standard_add_path(rel, path, is_partial=is_partial)

    def _set_cheapest(self, rel: RelOptInfo) -> None:
        """Determines the cheapest path in terms of costs from the pathlist and partial pathlist."""
        if rel.pathlist:
            rel.pathlist.sort(key=lambda path: path.estimated_cost)
            cheapest_path = rel.pathlist[0]
            rel.cheapest_path = cheapest_path
        else:
            raise LogicError("No valid paths for relation found.")

        if rel.partial_paths:
            rel.partial_paths.sort(key=lambda path: path.estimated_cost)
            cheapest_partial = rel.partial_paths[0]
            rel.cheapest_partial_path = cheapest_partial
        else:
            # Empty partial pathlist is allowed
            pass

    def _generate_gather_paths(self, rel: RelOptInfo) -> None:
        """Scans a RelOpt for partial paths that might be gathered into regular paths."""
        assert self.query is not None and self.cost_model is not None

        for partial_path in rel.partial_paths:
            n_workers = partial_path.get("estimated_workers", None)
            if n_workers is None:
                raise LogicError(f"Partial path does not have estimated workers: {partial_path}")
            gather_path = partial_path.parallelize(n_workers)
            query_fragment = transform.extract_subquery(self.query, gather_path.tables())
            cost_estimate = self.cost_model.estimate_cost(query_fragment, gather_path)
            gather_path = gather_path.with_estimates(cost=cost_estimate)

            # It is important to not delete the partial path after gathering it. The path might still be useful in an
            # upper-level partial path!
            self._add_path(rel, gather_path)

    def _generate_pseudo_gather_paths(self, rel: RelOptInfo) -> list[QueryPlan]:
        """Creates partial paths that execute the entire plan in parallel, rather than all joins."""
        gather_paths: list[QueryPlan] = []

        for partial_path in rel.partial_paths:
            n_workers = partial_path.get("estimated_workers", None)
            if n_workers is None:
                raise LogicError(f"Partial path does not have estimated workers: {partial_path}")

            pseudo_node = QueryPlan(node_type="Gather", children=partial_path, parallel_workers=n_workers)
            gather_paths.append(pseudo_node)

        return gather_paths

    def _create_seqscan_path(self, rel: RelOptInfo) -> QueryPlan:
        """Constructs and initializes a sequential scan path for a specific relation.

        This method assumes that sequential scans are actually enabled.
        """
        assert self.predicates is not None and self.query is not None and self.cost_model is not None

        baserel = util.simplify(rel.intermediate)
        filter_condition = self.predicates.filters_for(baserel)
        workers = self._estimate_workers(baserel)
        path = QueryPlan(
            ScanOperator.SequentialScan,
            children=[],
            base_table=baserel,
            estimated_cardinality=rel.cardinality,
            filter_predicate=filter_condition,
            estimated_workers=workers,
        )

        # Instead of running the cost estimation on the full query, we transform the into a SELECT * query
        # This prevents the query optimizer from inserting additional operators that would artifically increase the cost of the
        # scan.
        # For example, consider the following JOB query: SELECT min(k.keyword), ... FROM keyword k JOIN ... WHERE ...
        # If we run the cost estimation without the star transformation, the query fragment for k would end up to be
        # SELECT min(k.keyword) FROM keyword k WHERE ...
        # Executing this query would require an additional aggregation step which should no be performed already at this point.
        # By transforming the query into a star query, we get the plain result set without any additional operators.
        star_query = transform.as_star_query(self.query)
        cost = self.cost_model.estimate_cost(star_query, path)
        path = path.with_estimates(cost=cost)
        return path

    def _create_sequential_paths(self, rel: RelOptInfo) -> None:
        """Builds all sequential paths for a specific relation.

        This includes the regular sequential scan as well as the partial version.
        """
        if ScanOperator.SequentialScan not in self._scan_ops:
            return
        seq_path = self._create_seqscan_path(rel)
        self._add_path(rel, seq_path)

        if seq_path["estimated_workers"]:
            self._add_path(rel, seq_path, is_partial=True)

    def _create_index_paths(self, rel: RelOptInfo) -> None:
        """Builds all index scan paths for a specific relation.

        This method considers each index on the relation that spans columns from the query. If this is just a single column,
        it tries to create an index-only scan instead.

        If both kinds of index scans are disabled, this method does nothing.
        """
        assert self.predicates is not None and self.query is not None and self.cost_model is not None

        if ScanOperator.IndexScan not in self._scan_ops and ScanOperator.IndexOnlyScan not in self._scan_ops:
            return

        base_table = util.simplify(rel.intermediate)
        filter_condition = self.predicates.filters_for(base_table)
        required_columns = self.query.columns_of(base_table)
        idx_only_scan = ScanOperator.IndexOnlyScan in self._scan_ops and len(required_columns) <= 1
        candidate_indexes = {column: self.target_db.schema().indexes_on(column) for column in required_columns}

        index_paths: list[QueryPlan] = []
        for column, available_indexes in candidate_indexes.items():
            if not available_indexes:
                continue
            sorting = [SortKey.of(column)]

            for index in available_indexes:
                workers = self._estimate_workers(index)
                if ScanOperator.IndexScan in self._scan_ops:
                    idx_path = QueryPlan(
                        ScanOperator.IndexScan,
                        base_table=base_table,
                        index=index,
                        sort_keys=sorting,
                        filter_predicate=filter_condition,
                        estimated_workers=workers,
                    )
                    index_paths.append(idx_path)
                if idx_only_scan:
                    idx_path = QueryPlan(
                        ScanOperator.IndexOnlyScan,
                        base_table=base_table,
                        index=index,
                        sort_keys=sorting,
                        filter_predicate=filter_condition,
                        estimated_workers=workers,
                    )
                    index_paths.append(idx_path)

        # See comment in _create_seqscan_path() for explanation of the star query. We do the same here.
        star_query = transform.as_star_query(self.query)
        for path in index_paths:
            cost_estimate = self.cost_model.estimate_cost(star_query, path)
            path = path.with_estimates(cost=cost_estimate)
            self._add_path(rel, path)

            if path["estimated_workers"]:
                self._add_path(rel, path, is_partial=True)

    def _create_bitmap_path(self, rel: RelOptInfo) -> None:
        """Constructs and initializes a bitmap scan path for a specific relation.

        Since we don't model bitmap index scans, bitmap ANDs, etc. explicitly, we only need to create a single bitmap path.
        Afterwards, we let the hinting backend figure out how to perform the scan precisely.

        If bitmap scans are disabled, this method does nothing.
        """
        assert self.query is not None and self.predicates is not None and self.cost_model is not None

        if ScanOperator.BitmapScan not in self._scan_ops:
            return

        # We deviate from the vanilla PG implementation and only consider the cheapest bitmap path. Since they are all unsorted
        # anyway (due to the final sequential scan), this should be fine.
        #
        # Notice that the hinting backend is responsible for selecting the appropriate bitmap index hierarchies.

        base_table = util.simplify(rel.intermediate)
        required_columns = self.query.columns_of(base_table)
        candidate_indexes = {column: self.target_db.schema().indexes_on(column) for column in required_columns}
        if not candidate_indexes:
            # We only check if there are no candidate indexes at all and explicitly accept the case where there is a single
            # candidate index. This is because we can still perform an index lookup followed by a sequential scan of the pages.
            # This might be better than doing a full sequential scan or a full random I/O index scan.
            return

        filter_condition = self.predicates.filters_for(base_table)

        # bitmap scans parallelize the scan portion, not the index portion
        workers = self._estimate_workers(base_table)

        bitmap_path = QueryPlan(
            ScanOperator.BitmapScan,
            base_table=base_table,
            indexes=candidate_indexes,
            filter_predicate=filter_condition,
            estimated_workers=workers,
        )

        # See comment in _create_seqscan_path() for explanation of the star query. We do the same here.
        star_query = transform.as_star_query(self.query)
        cost_estimate = self.cost_model.estimate_cost(star_query, bitmap_path)
        bitmap_path = bitmap_path.with_estimates(cost=cost_estimate)

        self._add_path(rel, bitmap_path)
        if workers:
            self._add_path(rel, bitmap_path, is_partial=True)

    def _create_memoize_path(self, path: QueryPlan, *, cache_key: ColumnReference) -> QueryPlan:
        """Constructs and initializes a memo path for a specific relation.

        The `cache_key` is the column that identifies different entries in the memo table.

        This method assumes that memoization is actually enabled.
        """
        assert self.query is not None and self.cost_model is not None

        workers = path.get("estimated_workers", None)

        memo_path = QueryPlan(
            IntermediateOperator.Memoize,
            children=path,
            lookup_key=ColumnExpression(cache_key),
            estimated_cardinality=path.estimated_cardinality,
            estimated_workers=workers,
        )

        if self._is_pg_cost(self.cost_model):
            # Cost estimation happens as part of the parent join, so we just use a dummy here
            # see comment in generate_execution_plan() for details. We use the same reasoning here.
            return memo_path

        cost_estimate = self.cost_model.estimate_cost(self.query, memo_path)
        memo_path = memo_path.with_estimates(cost=cost_estimate)
        return memo_path

    def _create_materialize_path(self, path: QueryPlan) -> QueryPlan:
        """Constructs and initializes a materialize path for a specific relation.

        This method assumes that materialization is actually enabled.
        """
        assert self.cost_model is not None and self.query is not None

        workers = path.get("estimated_workers", None)

        mat_path = QueryPlan(
            IntermediateOperator.Materialize,
            children=path,
            estimated_cardinality=path.estimated_cardinality,
            estimated_workers=workers,
        )

        if self._is_pg_cost(self.cost_model):
            # Cost estimation happens as part of the parent join, so we just use a dummy here
            # see comment in generate_execution_plan() for details. We use the same reasoning here.
            return mat_path

        cost_estimate = self.cost_model.estimate_cost(self.query, mat_path)
        mat_path = mat_path.with_estimates(cost=cost_estimate)
        return mat_path

    def _create_sort_path(self, path: QueryPlan, *, sort_key: ColumnReference) -> QueryPlan:
        """Constructs and initializes a sort path for a specific relation on a specific column.

        The column to sort by is specified by `sort_key`. Notice that the sort path will always be created, even if the path
        is already sorted by the key.

        This method assumes that sorting is actually enabled.
        """
        assert self.cost_model is not None and self.query is not None

        workers = path.get("estimated_workers", None)

        sort_path = QueryPlan(
            IntermediateOperator.Sort,
            children=path,
            sort_keys=[SortKey.of(sort_key)],
            estimated_cardinality=path.estimated_cardinality,
            estimated_workers=workers,
        )

        cost_estimate = self.cost_model.estimate_cost(self.query, sort_path)
        sort_path = sort_path.with_estimates(cost=cost_estimate)
        return sort_path

    def _create_nestloop_path(self, join_rel: RelOptInfo, *, outer_path: QueryPlan, inner_path: QueryPlan) -> QueryPlan:
        """Constructs and initializes a nested loop join path for a specific intermediate.

        This method assumes that nested loop joins are actually enabled.

        Parameters
        ----------
        query : SqlQuery
            The query that is currently being optimized.
        join_rel : _RelOptInfo
            The RelOptInfo of the join to construct
        outer_path : QueryPlan
            The access path for the outer relation in the join
        inner_path : QueryPlan
            The access path for the inner relation in the join
        cost_model : CostModel
            The cost model to evaluate the new path
        """
        assert self.query is not None and self.predicates is not None and self.cost_model is not None

        join_condition = self.predicates.joins_between(outer_path.tables(), inner_path.tables())
        workers = outer_path.get("estimated_workers", None)

        nlj_path = QueryPlan(
            JoinOperator.NestedLoopJoin,
            children=[outer_path, inner_path],
            estimated_cardinality=join_rel.cardinality,
            filter_predicate=join_condition,
            estimated_workers=workers,
        )

        # We need to be very hacky here. Let's explain why:
        # PG is rather picky about which operators can be used for which queries. For example, it is generally not possible to
        # perform an index scan for an arbitrary SELECT * FROM relation query. Instead, PG requires the existence of a filter
        # condition on the indexed column. Likewise, Memoize and Materialize can never be the final operator in a plan, and
        # must be part of a nested-loop join.
        # Therefore, it might not be possible to correctly estimate the cost of these operators on their own, but only in
        # conjunction with a nested-loop join.
        # To account for this, we check whether we have any of these special/"weird" operators on the inner side of our NLJ.
        # If we do, and our target cost model is actually PG itself, we ask PG to estimate the cost for the entire NLJ plan
        # and just extract the costs that we require.

        weird_ops: set[PhysicalOperator] = {
            ScanOperator.IndexScan,
            ScanOperator.IndexOnlyScan,
            IntermediateOperator.Memoize,
            IntermediateOperator.Materialize,
        }
        if self._is_pg_cost(self.cost_model) and inner_path.operator in weird_ops:
            native_plan = self._pg_plan(nlj_path)

            if native_plan:
                nlj_node = native_plan.find_first_node(lambda n: n.operator == JoinOperator.NestedLoopJoin)
                assert nlj_node is not None, "No NLJ node found on NLJ plan"

                weird_node = native_plan.find_first_node(lambda n: n.operator == inner_path.operator, direction="inner")
                inner_cost = weird_node.estimated_cost if weird_node else math.inf
                nlj_cost = nlj_node.estimated_cost if not math.isinf(inner_cost) else math.inf

                updated_inner = inner_path.with_estimates(cost=inner_cost)
                nlj_path = QueryPlan(
                    JoinOperator.NestedLoopJoin,
                    children=[outer_path, updated_inner],
                    estimated_cardinality=join_rel.cardinality,
                    estimated_cost=nlj_cost,
                    filter_predicate=join_condition,
                    estimated_workers=workers,
                )
            else:
                cost_estimate = self.cost_model.estimate_cost(self.query, nlj_path)
                nlj_path = nlj_path.with_estimates(cost=cost_estimate)

        else:
            cost_estimate = self.cost_model.estimate_cost(self.query, nlj_path)
            nlj_path = nlj_path.with_estimates(cost=cost_estimate)

        return nlj_path

    def _create_mergejoin_path(
        self, join_rel: RelOptInfo, *, outer_path: QueryPlan, inner_path: QueryPlan
    ) -> QueryPlan:
        """Constructs and initializes a merge join path for a specific intermediate.

        This method assumes that merge joins are actually enabled and that the input paths are already sorted by the join key.
        However, we take a conservative approach and only assume sorting by the first sort key of each path.

        Parameters
        ----------
        query : SqlQuery
            The query that is currently being optimized.
        join_rel : _RelOptInfo
            The RelOptInfo of the join to construct
        outer_path : QueryPlan
            The access path for the outer relation in the join
        inner_path : QueryPlan
            The access path for the inner relation in the join
        cost_model : CostModel
            The cost model to evaluate the new path
        """
        assert self.query is not None and self.predicates is not None and self.cost_model is not None

        # This function assumes that outer_path and inner_path are already sorted appropriately.
        merge_key = outer_path.sort_keys[0].merge_with(inner_path.sort_keys[0])
        join_condition = self.predicates.joins_between(outer_path.tables(), inner_path.tables())
        workers = outer_path.get("estimated_workers", None)

        merge_path = QueryPlan(
            JoinOperator.SortMergeJoin,
            children=[outer_path, inner_path],
            sort_keys=[merge_key],
            estimated_cardinality=join_rel.cardinality,
            filter_predicate=join_condition,
            estimated_workers=workers,
        )

        cost_estimate = self.cost_model.estimate_cost(self.query, merge_path)
        merge_path = merge_path.with_estimates(cost=cost_estimate)
        return merge_path

    def _create_hashjoin_path(self, join_rel: RelOptInfo, *, outer_path: QueryPlan, inner_path: QueryPlan) -> QueryPlan:
        """Constructs and initializes a hash join path for a specific intermediate.

        This method assumes that hash joins are actually enabled.

        Parameters
        ----------
        query : SqlQuery
            The query that is currently being optimized.
        join_rel : _RelOptInfo
            The RelOptInfo of the join to construct
        outer_path : QueryPlan
            The access path for the outer relation in the join
        inner_path : QueryPlan
            The access path for the inner relation in the join
        cost_model : CostModel
            The cost model to evaluate the new path
        """
        assert self.query is not None and self.predicates is not None and self.cost_model is not None

        join_condition = self.predicates.joins_between(outer_path.tables(), inner_path.tables())
        workers = outer_path.get("estimated_workers", None)

        hash_path = QueryPlan(
            JoinOperator.HashJoin,
            children=[outer_path, inner_path],
            estimated_cardinality=join_rel.cardinality,
            filter_predicate=join_condition,
            estimated_workers=workers,
        )

        cost_estimate = self.cost_model.estimate_cost(self.query, hash_path)
        hash_path = hash_path.with_estimates(cost=cost_estimate)
        return hash_path

    def _estimate_workers(self, relation: TableReference | str) -> int:
        """Computes the number of worker processes to use for specific base relation or index.

        Parameters
        ----------
        relation : TableReference | str
            The base table or index to estimate. Indexes should be represented by their name whereas base tables can be
            supplied either as a `TableReference` or by name.
        """
        if not self._max_workers:
            return 0
        n_pages = self.target_db.statistics().n_pages(relation)
        workers = math.log(n_pages, 3)
        if self._max_workers is None:
            return round(workers)
        return min(self._max_workers, round(workers))

    def _pg_cost_estimate(self, path: QueryPlan) -> float:
        assert self.query is not None

        query_fragment = transform.extract_subquery(self.query, path.tables())
        hinted_query = self.target_db.hinting().generate_hints(query_fragment, path)
        try:
            cost = self.target_db.optimizer().cost_estimate(hinted_query)
            return cost
        except DatabaseServerError:
            return math.inf

    def _is_pg_cost(self, cost_model: CostModel) -> bool:
        return isinstance(cost_model, native.NativeCostModel) and cost_model.target_db == self.target_db

    def _pg_plan(self, path: QueryPlan) -> Optional[QueryPlan]:
        assert self.query is not None

        query_fragment = transform.extract_subquery(self.query, path.tables())
        hinted_query = self.target_db.hinting().generate_hints(query_fragment, path)
        try:
            native_plan = self.target_db.optimizer().query_plan(hinted_query)
            return native_plan
        except DatabaseServerError:
            return None

    def _determine_join_keys(self, *, outer_rel: RelOptInfo, inner_rel: RelOptInfo) -> list[AbstractPredicate]:
        """Determines all available join predicates between two relations.

        The predicates are implicitly ANDed together.
        """
        assert self.predicates is not None

        join_predicates = self.predicates.joins_between(outer_rel.intermediate, inner_rel.intermediate)
        if not join_predicates:
            raise LogicError(
                "Cross product detected. This should never happen so deep down in the "
                "optimization process. Intermediates are "
                f"{outer_rel} and {inner_rel}"
            )

        match join_predicates:
            case AndPredicate(children):
                join_keys = list(children)
            case _:
                join_keys = [join_predicates]

        return join_keys

    def _extract_join_columns(
        self,
        join_key: AbstractPredicate,
        *,
        outer_rel: RelOptInfo,
        inner_rel: RelOptInfo,
    ) -> tuple[ColumnReference, ColumnReference] | tuple[None, None]:
        """Provides the join columns that are joined together in the format (outer_col, inner_col).

        This method assumes that we indeed only perform a binary equi-join and will break otherwise.
        """
        partners = join_key.join_partners()
        if len(partners) != 2:
            # TODO: in all further processing, we ignore the case where a path might be sorted by more than one join key
            # already this should occur very rarely, but it might provide a decent performance boost in those situations
            return None, None

        partner: tuple[ColumnReference, ColumnReference] = util.simplify(partners)
        first_col, second_col = partner

        if first_col.table in outer_rel.intermediate and second_col.table in inner_rel.intermediate:
            return first_col, second_col
        elif first_col.table in inner_rel.intermediate and second_col.table in outer_rel.intermediate:
            return second_col, first_col
        else:
            raise LogicError()

    def _is_sorted_by(self, path: QueryPlan, column: ColumnReference) -> bool:
        """Checks, whether a specific path is sorted by some column.

        The column has to be the dominating part of the ordering, i.e. it is not sufficient that the column appears somewhere
        on the join path, it has to be the first column.
        """
        if not path.sort_keys:
            return False

        primary_sorting = path.sort_keys[0]
        return primary_sorting.is_compatible_with(column)

    def _sorting_subsumes(self, sorting: Sorting, *, other: Sorting) -> bool:
        """Checks, whether some sorting is "included" in another sorting.

        We define subsumption as follows:
        - If both sortings are equal, they subsume each other
        - If one sorting is longer than the other, but the shorter one is a prefix of the larger one, the larger one subsumes
          the smaller one

        Parameters
        ----------
        sorting : Sorting
            The sorting which should subsume the `other` sorting
        other : Sorting
            The sorting being subsumed
        """
        if sorting and not other:
            # we should always be able to evict other paths if we are sorted (and cheaper) and the other path is not
            # notice that we only check for sorting here, costs are handled elsewhere
            return True
        if other and not sorting:
            # we should never evict if we are not sorted but the other path is
            return False

        if len(other) > len(sorting):
            # we should never evict if the other path is more precise
            return False

        for i, key in enumerate(sorting):
            if i >= len(other):
                # Our current path has more sort keys than the other path (i.e. it is more specific) and so far all sort keys
                # have been equivalent. The other sorting is subsumed by our sorting.
                return True

            other_key = other[i]
            if not key.is_compatible_with(other_key):
                return False

        return True

    def _same_sorting(self, sorting: Sorting | None, *, other: Sorting | None) -> bool:
        """Checks, whether two sort orders are exactly equivalent."""
        if sorting is None and other is None:
            return True
        if sorting is None or other is None:
            return False

        if len(sorting) != len(other):
            return False

        for key, other_key in zip(sorting, other):
            if not key.is_compatible_with(other_key):
                return False

        return True

    def _warn(self, msg: str) -> None:
        if not self._verbose:
            return
        warnings.warn(msg, category=DPWarning, stacklevel=2)
