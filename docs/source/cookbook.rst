Cookbook
========

The cookbook demonstrates how to perform certain, frequently used patterns.
Throughout the examples, we use the following setup:

.. code-block:: ipython

    In [1]: import postbound as pb

    In [2]: pg_instance = pb.postgres.connect(config_file=".psycopg_connection")

    In [3]: stats = pb.workloads.stats()


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

.. code-block:: ipython

    In [4]: query = stats["q-10"]

    In [5]: print(pb.qal.format_quick(query))
    SELECT COUNT(*)
    FROM comments AS c, posts AS p, users AS u
    WHERE c.userid = u.id
      AND u.id = p.owneruserid
      AND c.creationdate >= CAST('2010-08-05 00:36:02' AS timestamp)
      AND c.creationdate <= CAST('2014-09-08 16:50:49' AS timestamp)
      AND p.viewcount >= 0
      AND p.viewcount <= 2897
      AND p.commentcount >= 0
      AND p.commentcount <= 16
      AND p.favoritecount >= 0
      AND p.favoritecount <= 10;

    In [6]: operators = pb.PhysicalOperatorAssignment()

    In [7]: operators.add(pb.JoinOperator.HashJoin, query.tables())
    Out[7]:
    PhysicalOperatorAssignment
      Join operators:
        +- {u, c, p}: Hash Join

    In [8]: operators
    Out[8]:
    PhysicalOperatorAssignment
      Join operators:
        +- {u, c, p}: Hash Join

    In [9]: hinted_query = pg_instance.hinting().generate_hints(query, physical_operators=operators)

    In [10]: hinted_query
    Out[10]:
    /*=pg_lab=
      Config(plan_mode=anchored)
      HashJoin(u c p)
     */
     SELECT COUNT(*) FROM comments AS c, posts AS p, users AS u WHERE c.userid = u.id AND u.id = p.owneruserid AND c.creationdate >= CAST('2010-08-05 00:36:02' AS timestamp) AND c.creationdate <= CAST('2014-09-08 16:50:49' AS timestamp) AND p.viewcount >= 0 AND p.viewcount <= 2897 AND p.commentcount >= 0 AND p.commentcount <= 16 AND p.favoritecount >= 0 AND p.favoritecount <= 10;

    In [11]: print(pg_instance.optimizer().query_plan(hinted_query).inspect())
    Aggregate
      Estimated Cardinality=1, Estimated Cost=20500.61
      ->  Gather
            Estimated Cardinality=2, Estimated Cost=20500.59
            Parallel Workers=2
            ->  Aggregate
                  Estimated Cardinality=3, Estimated Cost=19500.39
                  ->  Hash Join
                        Estimated Cardinality=30555, Estimated Cost=19474.92
                        ->  Nested Loop
                              Estimated Cardinality=4270218, Estimated Cost=14541.43
                              ->  Seq Scan(comments AS c)
                                    Estimated Cardinality=214590, Estimated Cost=7441.41
                              ->  Memoize
                                    Estimated Cardinality=3, Estimated Cost=0.8
                                    ->  Index Scan(posts AS p)
                                          Estimated Cardinality=3, Estimated Cost=0.79
                                          Index=posts_owneruserid_fkey
                        ->  Hash
                              Estimated Cardinality=71163, Estimated Cost=900.15
                              ->  Index Only Scan(users AS u)
                                    Estimated Cardinality=71163, Estimated Cost=900.15
                                    Index=users_pkey

Combined with the :mod:`query transformation tools <postbound.transform>` this is a powerful mechanism to obtain
(partial) plans for arbitrary subqueries:

.. code-block:: ipython

    In [12]: subquery = pb.transform.extract_query_fragment(query, pb.TableReference("posts", "p"))

    In [13]: print(pb.qal.format_quick(subquery))
    SELECT COUNT(*)
    FROM posts AS p
    WHERE p.viewcount >= 0
      AND p.viewcount <= 2897
      AND p.commentcount >= 0
      AND p.commentcount <= 16
      AND p.favoritecount >= 0
      AND p.favoritecount <= 10;

    In [14]: cards = pb.PlanParameterization()

    In [15]: cards.add_cardinality(subquery.tables(), 42)
    Out[15]:
    PlanParameterization
      Cardinalities:
        +- {p}: 42

    In [16]: hinted_subquery = pg_instance.hinting().generate_hints(subquery, plan_parameters=cards)


.. _cookbook-postgres-plans:

Postgres Query Plans
--------------------

When working with Postgres, there are three basic ways to access query plans:

1. You can retrieve the raw plan JSON using a plain :meth:`~postbound.postgres.PostgresDatabase.execute_query`
2. You can parse a raw plan into a :class:`~postbound.postgres.PostgresPlan`, which is pretty
   much a 1:1 model of the raw plan with more expressive attribute access and some high-level access methods
3. You can convert an explain into a proper normalized :class:`~postbound.QueryPlan` object

The conversion between the different formats works as follows:

.. code-block:: ipython

    In [17]: query = stats["q-10"]

    In [18]: explain_query = pb.transform.as_explain(query)

    In [19]: raw_plan = pg_instance.execute_query(explain_query)

    In [20]: raw_plan
    Out[20]:
    [{'Plan': {'Node Type': 'Aggregate',
       'Strategy': 'Plain',
       'Partial Mode': 'Finalize',
       'Parallel Aware': False,
       'Async Capable': False,
       'Startup Cost': 16951.55,
       'Total Cost': 16951.56,
       'Plan Rows': 1,
       'Plan Width': 8,
       'Disabled': False,
       'Output': ['count(*)'],
       'Plans': [{'Node Type': 'Gather',
         'Parent Relationship': 'Outer',
         'Parallel Aware': False,
         'Async Capable': False,
         'Startup Cost': 16951.34,
         'Total Cost': 16951.55,
         'Plan Rows': 2,
         'Plan Width': 8,
         'Disabled': False,
         'Output': ['(PARTIAL count(*))'],
         'Workers Planned': 2,
         'Single Copy': False,
         'Plans': [{'Node Type': 'Aggregate',
           'Strategy': 'Plain',
           'Partial Mode': 'Partial',
           'Parent Relationship': 'Outer',
           'Parallel Aware': False,
           'Async Capable': False,
           'Startup Cost': 15951.34,
           'Total Cost': 15951.35,
           'Plan Rows': 1,
           'Plan Width': 8,
           'Disabled': False,
           'Output': ['PARTIAL count(*)'],
           'Plans': [{'Node Type': 'Nested Loop',
             'Parent Relationship': 'Outer',
             'Parallel Aware': False,
             'Async Capable': False,
             'Join Type': 'Inner',
             'Startup Cost': 1196.96,
             'Total Cost': 15925.88,
             'Plan Rows': 10185,
             'Plan Width': 0,
             'Disabled': False,
             'Inner Unique': False,
             'Plans': [{'Node Type': 'Hash Join',
               'Parent Relationship': 'Outer',
               'Parallel Aware': True,
               'Async Capable': False,
               'Join Type': 'Inner',
               'Startup Cost': 1196.66,
               'Total Cost': 8825.86,
               'Plan Rows': 71530,
               'Plan Width': 8,
               'Disabled': False,
               'Output': ['c.userid', 'u.id'],
               'Inner Unique': True,
               'Hash Cond': '(c.userid = u.id)',
               'Plans': [{'Node Type': 'Seq Scan',
                 'Parent Relationship': 'Outer',
                 'Parallel Aware': True,
                 'Async Capable': False,
                 'Relation Name': 'comments',
                 'Schema': 'public',
                 'Alias': 'c',
                 'Startup Cost': 0.0,
                 'Total Cost': 7441.41,
                 'Plan Rows': 71530,
                 'Plan Width': 4,
                 'Disabled': False,
                 'Output': ['c.id',
                  'c.postid',
                  'c.score',
                  'c.text',
                  'c.creationdate',
                  'c.userid',
                  'c.userdisplayname'],
                 'Filter': "((c.creationdate >= '2010-08-05 00:36:02'::timestamp without time zone) AND (c.creationdate <= '2014-09-08 16:50:49'::timestamp without time zone))"},
                {'Node Type': 'Hash',
                 'Parent Relationship': 'Inner',
                 'Parallel Aware': True,
                 'Async Capable': False,
                 'Startup Cost': 900.15,
                 'Total Cost': 900.15,
                 'Plan Rows': 23721,
                 'Plan Width': 4,
                 'Disabled': False,
                 'Output': ['u.id'],
                 'Plans': [{'Node Type': 'Index Only Scan',
                   'Parent Relationship': 'Outer',
                   'Parallel Aware': True,
                   'Async Capable': False,
                   'Scan Direction': 'Forward',
                   'Index Name': 'users_pkey',
                   'Relation Name': 'users',
                   'Schema': 'public',
                   'Alias': 'u',
                   'Startup Cost': 0.29,
                   'Total Cost': 900.15,
                   'Plan Rows': 23721,
                   'Plan Width': 4,
                   'Disabled': False,
                   'Output': ['u.id']}]}]},
              {'Node Type': 'Memoize',
               'Parent Relationship': 'Inner',
               'Parallel Aware': False,
               'Async Capable': False,
               'Startup Cost': 0.3,
               'Total Cost': 0.8,
               'Plan Rows': 1,
               'Plan Width': 4,
               'Disabled': False,
               'Output': ['p.owneruserid'],
               'Cache Key': 'c.userid',
               'Cache Mode': 'logical',
               'Plans': [{'Node Type': 'Index Scan',
                 'Parent Relationship': 'Outer',
                 'Parallel Aware': False,
                 'Async Capable': False,
                 'Scan Direction': 'Forward',
                 'Index Name': 'posts_owneruserid_fkey',
                 'Relation Name': 'posts',
                 'Schema': 'public',
                 'Alias': 'p',
                 'Startup Cost': 0.29,
                 'Total Cost': 0.79,
                 'Plan Rows': 1,
                 'Plan Width': 4,
                 'Disabled': False,
                 'Output': ['p.owneruserid'],
                 'Index Cond': '(p.owneruserid = c.userid)',
                 'Filter': '((p.viewcount >= 0) AND (p.viewcount <= 2897) AND (p.commentcount >= 0) AND (p.commentcount <= 16) AND (p.favoritecount >= 0) AND (p.favoritecount <= 10))'}]}]}]}]}]},
      'Settings': {'effective_cache_size': '18GB',
       'work_mem': '96MB',
       'max_parallel_workers_per_gather': '6',
       'max_parallel_workers': '12'},
      'Planning Time': 0.736},
     {'Optimizer': {'Planner': 'Custom Hook',
       'Join Ordering': 'Dynamic Programming'}}]

    In [21]: postgres_plan = pb.postgres.PostgresPlan(raw_plan)

    In [22]: print(postgres_plan.inspect())
    Aggregate(cost=16951.56 rows=1)
      <- Gather(cost=16951.55 rows=2)
        <- Aggregate(cost=15951.35 rows=1)
          <- Nested Loop(cost=15925.88 rows=10185)
            <- Hash Join(cost=8825.86 rows=71530) Hash Cond: (c.userid = u.id)
              <- Seq Scan on c(cost=7441.41 rows=71530) Filter: ((c.creationdate >= '2010-08-05 00:36:02'::timestamp without time zone) AND (c.creationdate <= '2014-09-08 16:50:49'::timestamp without time zone))
              <- Hash(cost=900.15 rows=23721)
                <- Index Only Scan on u(cost=900.15 rows=23721)
            <- Memoize(cost=0.8 rows=1)
              <- Index Scan on p(cost=0.79 rows=1) Filter: ((p.viewcount >= 0) AND (p.viewcount <= 2897) AND (p.commentcount >= 0) AND (p.commentcount <= 16) AND (p.favoritecount >= 0) AND (p.favoritecount <= 10)) Index Cond: (p.owneruserid = c.userid)

    In [23]: qep = postgres_plan.as_qep()

    In [24]: print(qep.inspect())
    Aggregate
      Estimated Cardinality=1, Estimated Cost=16951.56
      ->  Gather
            Estimated Cardinality=2, Estimated Cost=16951.55
            Parallel Workers=2
            ->  Aggregate
                  Estimated Cardinality=3, Estimated Cost=15951.35
                  ->  Nested Loop
                        Estimated Cardinality=30555, Estimated Cost=15925.88
                        ->  Hash Join
                              Estimated Cardinality=214590, Estimated Cost=8825.86
                              ->  Seq Scan(comments AS c)
                                    Estimated Cardinality=214590, Estimated Cost=7441.41
                              ->  Hash
                                    Estimated Cardinality=71163, Estimated Cost=900.15
                                    ->  Index Only Scan(users AS u)
                                          Estimated Cardinality=71163, Estimated Cost=900.15
                                          Index=users_pkey
                        ->  Memoize
                              Estimated Cardinality=3, Estimated Cost=0.8
                              ->  Index Scan(posts AS p)
                                    Estimated Cardinality=3, Estimated Cost=0.79
                                    Index=posts_owneruserid_fkey

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

.. code-block:: ipython

    In [25]: pb.util.flatten([[1, 2], [3, 4]])
    Out[25]: [1, 2, 3, 4]

Likewise, :func:`~postbound.util.set_union` performs a union over multiple sets, thereby removing duplicates:

.. code-block:: ipython

    In [26]: pb.util.set_union([{1, 2, 3}, {2, 3, 4}, {4, 5}])
    Out[26]: {1, 2, 3, 4, 5}

To export data to disk, the :func:`~postbound.util.write_df` function automatically invokes the correct exporter based
on the file type. It also handles the conversion of PostBOUND objects to JSON representations if necessary.
