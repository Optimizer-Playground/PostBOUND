Implementing an Optimizer Prototype
===================================

Developing a new query optimization algorithm with PostBOUND revolves around optimization stages and optimization
pipelines. In short, an optimization pipeline is a mental model for a specific optimizer architecture.
For example, there exists a :class:`~postbound.TextBookOptimizationPipeline` that implements the traditional interplay of
plan enumeration, cost modelling, and cardinality estimation. Each pipeline in turn consists of one or multiple optimization
stages. Those stages function as hooks into the optimization process, allowing you to implement your own logic at specific
points of the optimization process.

Implementing a novel optimization prototype boils down to mapping the idea to the best-fitting optimization stage(s) and
combining them into the correponding optimization pipeline. One central design principle of the optimization pipelines
is that they do not require you to implement all of their stages. Instead, you can implement only those stages that are
relevant for your idea. PostBOUND will then either select reasonable default implementation for the remaining stages,
or skip them entirely (which usually forces the native optimizer of the target database to step in).

The remainder of this document provides a high-level overview of the available optimization stages.
To learn more about optimization pipelines, check the :doc:`dedicated documentation <core/optimization>`.


Basic Optimization Scenarios
----------------------------

All optimization stages share a common :class:`~postbound.OptimizationStage` base class. Depending on the specific hook,
these stages specify one more additional methods that need to be implemented. Check the individual documentation of the
stages for details. In general, it is good practice to do the following:

1. Make sure to call ``super().__init__()`` in the constructor of your implementation to properly initialize the base class.
2. Implement the ``describe()`` method to allow introspection of the optimizer configuration. The default implementation
   only provides the name of the optimizer hook.
3. Implement the ``pre_check()`` method to ensure that the optimization stage is only called for supported queries. The
   default implementation allows all queries to be optimized.

Currently, PostBOUND supports the following optimization stages out of the box:


.. _stage-card-est:

Cardinality Estimation
^^^^^^^^^^^^^^^^^^^^^^

Cardinality estimators are implemented using the :class:`~postbound.CardinalityEstimator` interface. It requires a single
method, :meth:`~postbound.CardinalityEstimator.calculate_estimate`, which receives the full query currently being optimized
and the specific intermediate (a subset of all relations in the query) for which the cardinality estimate should be
calculated.

As an example, consider a cardinality estimator that consistently overestimates the native cardinality estimate of a
database system by a fixed factor:

.. code-block:: python

    import postbound as pb

    class CardinalityOverestimation(pb.CardinalityEstimator):
        def __init__(self, target_db: pb.Database, *, overestimation_factor: float = 10.0):
            super().__init__()
            self._native_optimizer = target_db.optimizer()
            self._overestimation_factor = overestimation_factor

        def calculate_estimate(
            self,
            query: pb.SqlQuery,
            intermediate: pb.TableReference | Iterable[pb.TableReference]
        ) -> pb.Cardinality:
            subquery = pb.transform.extract_query_fragment(query, intermediate)
            if subquery is None:
                return pb.Cardinality.unknown()
            native_estimate = self._native_optimizer.cardinality_estimate(subquery)
            return self._overestimation_factor * native_estimate

The estimator can be used in the :class:`~postbound.TextBookOptimizationPipeline` as well as the
:class:`~postbound.MultiStageOptimizationPipeline`. In the latter case, it acts as a plan parameterization stage.
The article on :doc:`optimization pipelines <core/optimization>` provides more insight into which
pipeline to use under which circumstances. In short, use the :class:`~postbound.MultiStageOptimizationPipeline` if you
"only" want to develop a novel cardinality estimator and leave the rest of the optimizer as-is.
See the documentation of :class:`~postbound.CardinalityEstimator` for additional methods that a cardinality estimator
provides.


.. _stage-cost-model:

Cost Models
^^^^^^^^^^^

Custom cost models are supported via the :class:`~postbound.CostModel` interface. It requires implementing the
:meth:`~postbound.CostModel.estimate_cost` method, which receives the full query currently being optimized and the subplan
that should be evaluated.

The following example demonstrates a cost model that doubles the cost of all nested loop joins and leaves all other costs
as they are:

.. code-block:: python

    import postbound as pb

    class ExpensiveNLJ(pb.CostModel):
        def __init__(self, target_db: pb.Database):
            super().__init__()
            self._target_db = target_db

        def estimate_cost(self, query: pb.SqlQuery, plan: pb.QueryPlan) -> pb.Cost:
            subquery = pb.transform.extract_query_fragment(query, plan.tables())
            if subquery is None:
                raise ValueError("Could not extract a valid subquery for the given plan.")

            # extract_query_fragment keeps all projections if they are compatible with the intermediate. This is not
            # useful for our case, since we cannot assume that the database system can push down those projections.
            # Therefore, we opt for a more conservative approach and treat the subquery as a "SELECT *" query.
            subquery = pb.transform.as_star_query(subquery)
            subquery = self._target_db.hinting().generate_hints(subquery, plan)

            native_cost = self._target_db.optimizer().cost_estimate(subquery)
            if plan.operator == pb.JoinOperators.NestedLoopJoin:
                return 2 * native_cost
            return native_cost

Cost models can currently only be used in the :class:`~postbound.TextBookOptimizationPipeline`. In this pipeline, the cost
model is called by the enumerator whenever it needs to assign a cost to a given subplan. The cost model is not responsible
for setting the cost on the given subplan. This is the enumerators job. The reasoning behind this design is that a) the cost
model should be a "pure" function that does not have side effects (at least from PostBOUND's point-of-view) and b) it
allows the enumerator to process the subplan and its cost further (e.g., by pruning a plan alltogether).
See the documentation of :class:`~postbound.CostModel` for additional methods that a cost model provides.


.. _stage-plan-enum:

Plan Enumeration
^^^^^^^^^^^^^^^^

Plan enumerators are responsible for generating full candidate plans for a given query. Each plan includes the join order
to use, as well as the physical operators to use for each join and scan operator. However, the target database is allowed
to insert additional intermediate operators if these are required for the correctness of the plan. For example, a database
might insert an additional sort operator before a merge join.

Enumerators are defined using the :class:`~postbound.PlanEnumerator` interface. It requires implementing the
:meth:`~postbound.PlanEnumerator.generate_execution_plan` method. In turn, this method receives the full query currently
being optimized, as well as the selected cost model and cardinality estimator. The enumerator is free to use these
components in any way it sees fit (including not at all). The only requirement is that the enumerator needs to return a
valid execution plan for the given query. This plan can be computed all at once, or by stitching together multiple
intermediate plans.

Currently, plan enumeration has no explicit support for logical preprocessing of the query, such as transformations in
relational algebra. However, the enumerator is free to process the query before it starts generating plans. For example, it
can interact with the :mod:`~postbound.relalg` module for relational algebra support.

Since implementing an entire plan enumerator is a complex undertaking, we do not show a full example here. Instead, you can
check out the implementation of the :class:`~postbound.opt.dynprog.DynamicProgrammingEnumerator` for a textbook-style dynamic
programming plan enumerator
(`reference <https://github.com/Optimizer-Playground/PostBOUND/blob/main/postbound/opt/dynprog.py#L79>`__).

Plan enumerators can currently only be used in the :class:`~postbound.TextBookOptimizationPipeline`. As part of this
pipeline, the cardinality estimator and cost model can also be selected. The pipeline than takes care of passing these on
to the enumerator.
See the documentation of :class:`~postbound.PlanEnumerator` for additional methods that a plan enumerator provides.
If you do not need to implement a full plan enumerator, there are two "small cousins" of the plan enumerator that allow
you to implement only the join ordering or physical operator selection logic. These are described in the following
sections.


.. _stage-join-order:

Join Ordering
^^^^^^^^^^^^^

Join ordering is supported via the :class:`~postbound.JoinOrderOptimization` stage. In contrast to the full
:class:`~postbound.PlanEnumerator`, this stage is only responsible for generating a join order for the given query, without
selecting physical operators. To use this interface, the :meth:`~postbound.JoinOrderOptimization.optimize_join_order` method
needs to be implemented. It simply receives the full query currently being optimized.

As an example, the following implementation generates (linear) join order by always joining the smallest available relation:

.. code-block:: python

    import random
    import postbound as pb

    class GreedyJoinOrder(pb.JoinOrderOptimization):
        def __init__(self, target_db: pb.Database) -> None:
            super().__init__()
            self._target_db = target_db

        def optimize_join_order(self, query: pb.SqlQuery) -> pb.JoinTree:
            join_order = pb.JoinTree()
            available_relations = list(query.tables())

            # we use the statistics catalog to figure out the relation sizes
            available_relations = sorted(
                available_relations, key=self._target_db.statistics().total_rows
            )

            while available_relations:
                candidate = next(
                    (
                        rel
                        for rel in available_relations
                        if not join_order or query.joins_between(rel, join_order.tables())
                    ),
                    None,
                )
                if candidate is None:
                    # cross product - let's try again
                    continue

                join_order = join_order.join_with(candidate)
                available_relations.remove(candidate)

            return join_order

In this example, we would end up in an infinite loop if the query actually contains a cross product. Therefore, we would
need to also implement a ``pre_check()`` to forbid the optimizer from being called on such a query.

Join ordering algorithms can only be used in the :class:`~postbound.MultiStageOptimizationPipeline` where it is the very
first step. If you also want to select the physical operators which compute the join order, you can either implement a full
:class:`~postbound.PlanEnumerator` or use the physical operator selection stage described in the next section.

Currently, PostBOUND does not have dedicated support to pass a cardinality estimator or cost model to a join ordering stage.
If your logic needs any of those components, you can just pass them to the constructor of your optimizer.
See the documentation of :class:`~postbound.JoinOrderOptimization` for additional details on the interface.


.. _stage-op-selection:

Physical Operator Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The operator selection stage is responsible for assigning physical operators for scans and joins in a given query.
It is defined in the :class:`~postbound.PhysicalOperatorSelection` interface and forms the second step in the
:class:`~postbound.MultiStageOptimizationPipeline`. While this interface only requires the implementation of the
:meth:`~postbound.PhysicalOperatorSelection.select_physical_operators` method, it operates in two distinct modes, depending
on the overall pipeline configuration: if a join order optimization stage is present, the operator selection receives the
final join order as input. In this case, it only needs to assign physical operators to (a subset of) the joins and scans in
the join order. Otherwise, the operator selection can assign physical operators to any intermediate in the query. This
functions as a restriction of the search space of the native optimizer, which has to come up with its own join order.
If your operator selection does not support one of the two modes, the best strategy is currently to raise an error if that
mode is encoutered.

In the following example, we simply execute all joins as hash joins and all scans as sequential scans:

.. code-block:: python

    import postbound as pb

    class BasicOperatorSelection(pb.PhysicalOperatorSelection):
        def __init__(self) -> None:
            super().__init__()

        def select_physical_operators(
            self,
            query: pb.SqlQuery,
            join_order: Optional[pb.JoinTree]
        ) -> pb.PhysicalOperatorAssignment:
            if join_order is None:
                joins = (intermediate for intermediate in pb.util.powerset(query.tables()) if len(intermediate) > 1)
            else:
                joins = (node.tables() for node in join_order.iterjoins())

            assignment = pb.PhysicalOperatorAssignment()
            for tab in query.tables():
                assignment.add(pb.ScanOperator.SequentialScan, tab)
            for join in joins:
                assignment.add(pb.JoinOperator.HashJoin, join)
            return assignment

Operator selection algorithms can only be used in the :class:`~postbound.MultiStageOptimizationPipeline` where it is the
second step. If you also want to compute the join order, you can either implement a full :class:`~postbound.PlanEnumerator`
or use the physical operator selection stage described in the previous section.

Currently, PostBOUND does not have dedicated support to pass a cardinality estimator or cost model to an operator selection.
If your logic needs any of those components, you can just pass them to the constructor of your optimizer.
See the documentation of :class:`~postbound.PhysicalOperatorSelection` for additional details on the interface.


Plan Parameterization
^^^^^^^^^^^^^^^^^^^^^

The plan parameterization takes care of all aspects of query plans that do not directly map to the join order or the
physical operators. This includes three kinds of information: cardinality estimates, parallelization info, and optimizer
settings that are specific to the target database.
To supply plan parameters, the :class:`~postbound.ParameterGeneration` interface and its
:meth:`~postbound.ParameterGeneration.generate_plan_parameters` need to be implemented.

A parameterization can only be used in the :class:`~postbound.MultiStageOptimizationPipeline` (with one exception, see
below). In this pipeline, it acts as the final stage. Consequently, it serves two distinct purposes:
First, parameters can supply information about the plan which neither the :ref:`stage-join-order` nor the
:ref:`stage-op-selection` stage provided. This is particularly useful for choosing the number of parallel workers a subplan
should use.
Second, if the :ref:`stage-join-order` and :ref:`stage-op-selection` stages are not implemented, the parameterization can be used to
supply cardinality estimates. For example, if you skip the operator selection stage, the native optimizer would use the
calculated join order, but perform its normal logic for operator selection. However, this selection happens with a twist:
instead of calculating its own cardinality estimates, the cost model would use the estimates supplied by the
parameterization.

As an extreme case, you could skip both join ordering and operator selection and only implement the parameterization stage
to compute cardinality estimates. In this scenario, you essentially overwrite the native cardinality estimator of the
target database system. This is exactly what the :class:`~postbound.CardinalityEstimator` does when used in the
:class:`~postbound.MultiStageOptimizationPipeline` (see :ref:`above <stage-card-est>`).

In the following example, we use the parameterization to execute the final join of a query in parallel:

.. code-block:: python

    import postbound as pb

    class ParallelPlans(pb.ParameterGeneration):
        def __init__(self, *, n_workers: int = 4) -> None:
            super().__init__()
            self._n_workers = n_workers

        def generate_plan_parameters(
            self,
            query: pb.SqlQuery,
            join_order: Optional[pb.JoinTree],
            operator_assignment: Optional[pb.PhysicalOperatorAssignment]
        ) -> pb.PlanParameterization:
            params = pb.PlanParameterization()
            final_join = frozenset(query.tables())
            params.set_workers(final_join, self._n_workers)
            return params

See the documentation of :class:`~postbound.ParameterGeneration` for more details on the interface.


.. _stage-incremental-transform:

Plan Transformation
^^^^^^^^^^^^^^^^^^^

A plan transformation receives a complete execution plan as input and in turn produces a complete execution plan as output.
Multiple plan transformations can be chained together in an :class:`~postbound.IncrementalOptimizationPipeline` to model
complex optimization scenarios.
For example, earlier stages could perform subquery unnesting and join reordering, while later stages take care of operator
selection and plan parameterization.
Each individual transformation must implement the :class:`~postbound.IncrementalOptimizationStep` interface and its
:meth:`~postbound.IncrementalOptimizationStep.optimize_query` method.

.. note::

    Plan transformations are currently not widely used. You will notice that the overall developer experience for modifying
    existing plans is not very comfortable. If you have a specific use case for plan transformation and are struggling
    with the current API, please reach out to us on `GitHub <https://github.com/Optimizer-Playground/PostBOUND/issues>`_. We
    are happy to improve the API but need more diverse use cases to do so.


.. _stage-complete-opt:

Complete Plan Generation
^^^^^^^^^^^^^^^^^^^^^^^^

If you want to compute the entire execution plan for a given query in a black-box fashion, you can implement a
:class:`~postbound.CompleteOptimizationAlgorithm`. For this optimization stage, you do not need to interact with other
outside optimization stages. The central :meth:`~postbound.CompleteOptimizationAlgorithm.optimize_query` method receives
the full query and returns the complete execution plan for that query.

This otimization stage is particularly useful when you receive the plan from an external component. For example, some
learned optimizers obtain candidate plans from the target database system and then use a learned cost model to select the
best plan among those candidates.


Advanced Optimization Scenarios
-------------------------------

Optimization stages are the basic building blocks for implementing optimization algorithms in PostBOUND.
If your optimization idea does not perfectly fit into a single pre-defined optimization stage, there are four main
strategies to still implement it in PostBOUND (in increasing order of complexity):

1. Combine multiple pre-defined optimization stages in an optimization pipeline. For example, in addition to implementing
   join ordering, you might also implement cardinality estimation and combine both in a multi-stage pipeline. Since Python
   allows for multiple inheritance, you can even implement them in a single class and allow for easy sharing of internal
   state. In general, this is the prefered and intended strategy.
2. Use a more powerful optimization stage. For example, if you are researching join ordering but you also need to know
   about the physical operators, you might move away from the multi-stage pipeline and implement a full plan enumerator for
   the textbook pipeline. However, this strategy requires you to implement more complex logic that might be beyond the scope
   of the specific optimization idea you want to implement.
3. Implement a new subclass of the :class:`~postbound.OptimizationPipeline` with its own custom subclasses of
   :class:`~postbound.OptimizationStage` (see :ref:`stages-beyond-built-in` below).
4. Move away from the pipeline-based optimization paradigm. Even in this extreme case, you can still use PostBOUND to
   benefit from query abstraction, database interaction, and hinting utilities. The :doc:`cookbook` contains some examples
   of how to perform hinting manually, etc. In our own research, we opted for this strategy when we wanted to obtain
   multiple different query plans for a single query.


Learned Algorithms
------------------

Optimization algorithms that learn from training samples, raw data, or even online during query optimization are
increasingly popular in the research community. Therefore, PostBOUND provides dedicated support for optimization stages
that need some kind of learning component. This could be as simple as checking the database schema to know which kinds of
statistics to build, or as complex as training a deep learning model on pre-computed features.

No matter the specific learning strategy, the core steps to integrate it into PostBOUND are always the same: the
optimization provides a number of hooks that are called for different training tasks (similar to how pipelines have stages
for different optimization tasks). All you need to do is implement the corresponding methods for these hooks. Afterwards,
the benchmarking utilities of PostBOUND will automatically detect the presence of these hooks and call them at the
appropriate time.

For example, consider a data-driven cardinality estimator:

.. code-block:: python

    import postbound as pb
    import time

    class DataDrivenCardinalityEstimator(pb.CardinalityEstimator):
        def __init__(self):
            super().__init__()
            self._model = create_my_model()

        def calculate_estimate(
            self,
            query: pb.SqlQuery,
            intermediate: pb.TableReference | Iterable[pb.TableReference]
        ) -> pb.CardinalityEstimate:
            subquery = pb.transform.extract_query_fragment(query, intermediate)
            if subquery is None:
                return pb.Cardinality.unknown()
            return self._model.predict(subquery)

        def fit_database(self, database: pb.Database) -> pb.train.TrainingMetrics:
            start = time.perf_counter_ns()
            for table in database.schema():
                self._model.train_on(table)
            end = time.perf_counter_ns()
            train_time = end - start
            return {"train_time_ns": train_time}

        def database_fit_completed(self) -> bool:
            return self._model.is_trained()

The ``fit_database()`` and ``database_fit_completed()`` methods are the hooks that PostBOUND provides. See the
documentation on :class:`~postbound.OptimizationStage` for more details on the training process.


.. _stages-beyond-built-in:

Beyond Built-in Hooks
---------------------

If your optimization algorithm does not fit into any of the pre-defined optimization stages, you have to implement a custom
subclass of the :class:`~OptimizationPipeline`. This pipeline could then implement the desired optimization logic directly,
or rely its own :class:`~OptimizationStage` subclasses. This would allow you to still plug your optimizer into the
benchmarking utilities, etc. of PostBOUND.

.. note::

    We envisioned PostBOUND to allow prototyping of vastly different optimization ideas. If you indeed need to implement a
    custom pipeline, please reach out to us on GitHub. We are happy to incorporate your use case into the core framework
    if that makes sense.


Pre-defined Optimization Stages & Utilities
-------------------------------------------

PostBOUND ships a number of simple optimization stages out-of-the-box in the :mod:`postbound.opt` module. These can be
used as building blocks for your own optimizers or as a reference for implementing your own stages.
The separete :doc:`advanced/existing-strategies` documentation provides an overview of the provided stages.
