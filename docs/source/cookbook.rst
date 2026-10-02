Cookbook
========

The cookbook demonstrates how to perform certain, frequently used patterns.
Throughout the examples, we use the following setup:

.. ipython:: python
    :okwarning:

    import postbound as pb
    pg_instance = pb.postgres.connect(config_file=".psycopg_connection")
    stats = pb.workloads.stats()


.. _cookbook-cardinality-estimation:

Cardinality estimation
----------------------

**TLDR: use the :class:`~postbound.MultiStageOptimizationPipeline` to implement a new cardinality estimator.**

PostBOUND provides two main :doc:`optimization pipelines <core/optimization>` to implement new cardinality estimators,
the :class:`~postbound.TextBookOptimizationPipeline` and the :class:`~postbound.MultiStageOptimizationPipeline`.
The textbook pipeline models the traditional interplay between plan enumerator, cost model, and cardinality estimator.
In contrast, the multi-stage pipeline follows a sequential approach that consists of join ordering, operator selection,
and plan parameterization. This third stage includes optional cardinality estimates.

At first glance, the :class:`~postbound.TextBookOptimizationPipeline` might seem to be the better suited one.
However, if the sole concern is cardinality estimation, there are some caveats when using this pipeline.
Specifically, in the textbook architecture, it is the plan enumerator's responsibility to request cost estimates for
its candidate plans, and it is the cost model's responsibility to request cardinality estimates for its plans.
Therefore, the cardinalities to estimate ultimately depend on which plans the enumerator generates.
If a textbook pipeline does not have an enumerator specified, PostBOUND must "fill the gaps" and come up with a way
to generate plans and invoke cost model and cardinality estimator.

Currently, PostBOUND cannot interactively cooperate with the optimizer of the target database system to ask questions
such as "What plan would you inspect next, given these cardinalities/costs?". This leaves simulation as the only viable
option: if no enumerator is specified, PostBOUND uses a very simple, dynamic-programming based enumerator to generate
candidate plans. But this is only a simulation and the generated plans will differ from the plans that the actual
optimizer would produce. See the documentation of the :ref:`default plan enumerator selection <default-enumerator>` for
more details.

In contrast, when a :class:`~postbound.MultiStageOptimizationPipeline` contains just a cardinality estimator as plan
parameterization (and no join ordering nor operator selection), PostBOUND simply ships all estimated cardinalities to
the target database system. This allows the native optimizer to be invoked, with the just the given cardinalities being
overwritten. For all missing cardinalities, the join order, and the selected operators, the optimizer must use its
own strategies.
In fact, the default implementation when using a cardinality estimator in this context generates estimates for all
sensible intermediates. While this can be quiet a lot of candidates for more complex queries (e.g., the more complex
JOB queries contain several thousand intermediates) we have not observed a significant performance impact.

**Therefore, the recommended way to implement a new cardinality estimator is to use the**
:class:`~postbound.CardinalityEstimator` **within a** :class:`~postbound.MultiStageOptimizationPipeline`. If you should
need to switch to a textbook pipeline later, you can do so without changing your estimator implementation. This is
because the estimator class already provides default implementations to satisfy both interfaces.


.. _cookbook-partial-hinting:

Manual hinting
--------------

You can easily generate hinted queries and execution plans using any of the fundamental
:ref:`optimizer data structures <optimizer-data-structures>` by talking directly to the
:class:`~postbound.db.HintService` of your target :class:`~postbound.Database`:

.. ipython:: python

    query = stats["q-10"]
    print(pb.qal.format_quick(query))
    operators = pb.PhysicalOperatorAssignment()
    operators.add(pb.JoinOperator.HashJoin, query.tables())
    operators

    hinted_query = pg_instance.hinting().generate_hints(query, physical_operators=operators)
    hinted_query

    print(pg_instance.optimizer().query_plan(hinted_query).inspect())

Combined with the :mod:`query transformation tools <postbound.transform>` this is a powerful mechanism to obtain
(partial) plans for arbitrary subqueries:

.. ipython:: python
    :okwarning:

    subquery = pb.transform.extract_query_fragment(query, pb.TableReference("posts", "p"))
    print(pb.qal.format_quick(subquery))
    cards = pb.PlanParameterization()
    cards.add_cardinality(subquery.tables(), 42)
    hinted_subquery = pg_instance.hinting().generate_hints(subquery, plan_parameters=cards)


.. _cookbook-postgres-plans:

Postgres Query Plans
--------------------

When working with Postgres, there are three basic ways to access query plans:

1. You can retrieve the raw plan JSON using a plain :meth:`~postbound.postgres.PostgresDatabase.execute_query`
2. You can parse a raw plan into a :class:`~postbound.postgres.PostgresPlan`, which is pretty
   much a 1:1 model of the raw plan with more expressive attribute access and some high-level access methods
3. You can convert an explain into a proper normalized :class:`~postbound.QueryPlan` object

The conversion between the different formats works as follows:

.. ipython:: python

    query = stats["q-10"]
    explain_query = pb.transform.as_explain(query)
    raw_plan = pg_instance.execute_query(explain_query)
    raw_plan
    postgres_plan = pb.postgres.PostgresPlan(raw_plan)
    print(postgres_plan.inspect())
    qep = postgres_plan.as_qep()
    print(qep.inspect())

Each :class:`~postbound.db.OptimizerInterface` also provides a :meth:`~postbound.db.OptimizerInterface.parse_plan`
method that combines all of the above steps.

.. _jsonize:

JSON export
-----------

To export arbitrary objets to JSON, PostBOUND provides a *jsonize* protocol. Essentially, all you need to do is a add a
``__json__`` method to your class. This class can emit arbitrary objects that can either be JSON-serialized by Python's
standard JSON dump logic, or that provide a ``__json__`` method themselves.
To make sure that this method works, use the :func:`~postbound.util.to_json` or
:func:`~postbound.util.to_json_dump` for the export. All of PostBOUND's built-in JSON export does this automatically.


Miscellaneous utilities
-----------------------

There are some general utilities that might make your life a little easier, mostly when it comes to working with one or
multiple instances of some class.

Use :func:`~postbound.util.flatten` to flatten nested lists or iterables:

.. ipython:: python

    pb.util.flatten([[1, 2], [3, 4]])

Likewise, :func:`~postbound.util.set_union` performs a union over multiple sets, thereby removing duplicates:

.. ipython:: python

    pb.util.set_union([{1, 2, 3}, {2, 3, 4}, {4, 5}])

To export data to disk, the :func:`~postbound.util.write_df` function automatically invokes the correct exporter based
on the file type. It also handles the conversion of PostBOUND objects to JSON representations if necessary.
