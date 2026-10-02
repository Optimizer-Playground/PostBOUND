10 Minutes to PostBOUND
=======================

Inspired by the `10 minutes to Pandas <https://pandas.pydata.org/docs/user_guide/10min.html>`__ tutorial, this is a
quick introduction to PostBOUND, geared towards new users.
It assumes that you have a basic understanding of Python and a decent understanding of relational databases, especially
query optimization.

The primary target audience for PostBOUND are database researchers that want to implement their own ideas for query
optimization and evaluate them on real-world data.
To learn how to install PostBOUND check the :doc:`Setup Guide<setup>` guide.

By convention, we use the following imports in this tutorial:

.. code-block:: ipython

    In [1]: import pandas as pd

    In [2]: import postbound as pb


Optimization Pipelines
-----------------------

Optimization pipelines are at the core of PostBOUND.
They provide a simple model for optimization workflows oriented around different optimizer architectures.
Each pipeline has different interfaces that allow to customize different parts of the optimization process (these are
*stages* in PostBOUND-lingo).
Implementing a new optimization algorithm boils down to selecting the appropriate optimization pipeline and stage.

The two most commonly used pipelines are:

The :class:`TextBookOptimizationPipeline <postbound.TextBookOptimizationPipeline>`, which is modelled after the
traditional query optimizer architecture.
It uses a :class:`PlanEnumerator <postbound.PlanEnumerator>` to produce different candidate plans and ultimately select
the best one.
To assess the quality of each candidate plan, a  :class:`CostModel <postbound.CostModel>` is used.
The cost model typically uses the :class:`CardinalityEstimator <postbound.CardinalityEstimator>` to estimate the number
of rows produced by each operator in the plan.

The :class:`MultiStageOptimizationPipeline <postbound.MultiStageOptimizationPipeline>` performs the query optimization in
multiple sequential steps.
Initially, it computes a join order using the :class:`JoinOrderOptimization <postbound.JoinOrderOptimization>` stage.
Afterwards, it selects the best physical operators for the join order in the
:class:`PhysicalOperatorSelection <postbound.PhysicalOperatorSelection>` stage.
Finally, the :class:`ParameterGeneration <postbound.ParameterGeneration>` can be used to add additional metadata to the
query plan.
This stage is especially well-suited for optimization scenarios where only part of the decisions of the native optimizer
are overridden.
For example, you can implement your own join ordering and cardinality estimator and leave the operator selection to the
database system (given your cardinality estimates).
See the documentation on :doc:`core/hinting` for how this actually works under the hood.

To learn more about optimization pipelines, take a look at the separate :doc:`core/optimization` documentation.

.. note::

    Users do not need to implement all stages of a pipeline.
    Instead, PostBOUND automatically "fills the gaps" with reasonable defaults.
    This allows users to focus only on the parts of the optimization process that are relevant for their research.
    For example, if you want to implement a new join ordering algorithm in the multi-stage pipeline, PostBOUND will
    let the target database system select the physical operators for the join order.

Query Abstraction
-----------------

PostBOUND provides a powerful query abstraction centered around :class:`~postbound.qal.SqlQuery`.
Pretty much all other parts of the framework operate on this abstraction.
You can obtain an instance of this class by parsing a raw SQL query string using
:func:`~postbound.parse_query`:

.. code-block:: ipython

    In [3]: raw_query = "SELECT p.creationdate, min(p.score) FROM posts p GROUP BY p.creationdate"

    In [4]: query = pb.parse_query(raw_query)

    In [5]: query
    Out[5]: SELECT p.creationdate, MIN(p.score) FROM posts AS p GROUP BY p.creationdate;

Alternatively, you can use the :ref:`workload functionality <10minutes-workloads>` to load an entire set of queries at
once.
Most query-related functionality is provided by the :mod:`qal <postbound.qal>` module (short for *query abstraction layer*).
For example, use :func:`qal.format_quick() <postbound.qal.format_quick>` to obtain a nicely formatted string for
a query.

Once you have a query object, you can access different properties, such as the tables that are referenced in the query,
or the raw clauses:

.. code-block:: ipython

    In [6]: query.tables()
    Out[6]: {TableReference(full_name='posts', alias='p', virtual=False, schema='', catalog='')}

    In [7]: query.select_clause
    Out[7]: SELECT p.creationdate, MIN(p.score)

For more details on the query abstraction, see the separate :doc:`core/qal` documentation.

.. note::

    The :class:`~postbound.qal.SqlQuery` is only used to represent ``SELECT`` queries and PostBOUND currently
    does not support DDL or DML queries.
    Plain ``SELECT`` queries are modelled as a :class:`~postbound.qal.SelectStatement` subclass.
    Set operations such as ``UNION`` are modelled in a separate :class:`~postbound.qal.SetQuery>` subclass.
    The reasons that are explained in the :doc:`core/qal` documentation.
    Most parts of the framework operate on the abstract :class:`~postbound.qal.SqlQuery`.
    If some functionalities are only available for set queries or plain ``SELECT``queries, this is indicated in the
    function signature.

    We might add support for DDL and DML queries in the future, if there is actual demand for it.
    For now, such statements have to be represented as raw SQL strings.
    To indicate that an interface would also accept queries beyond ``SELECT``, the
    :type:`~postbound.qal.SqlStatement` type exists.
    But this is currently rarely used.


.. _10minutes-db-connection:

Database Connection
-------------------

A key philosophy of PostBOUND is to always execute queries on real database systems instead of research prototypes or
simulated environments.
We treat the query execution time as the ultimate measure of quality for a query plan.
But, since PostBOUND is implemented as a Python framework, we cannot interact with the optimizer directly.
Instead, PostBOUND uses query hints to restrict the native optimizer of the database system and to enforce the
optimization decisions made within the framework.

As a consequence, PostBOUND requires a connection to a database system for much of its functionality.
For Postgres, you can connect to the database like so:

.. code-block:: ipython

    In [8]: pg_instance = pb.postgres.connect(config_file=".psycopg_connection")

    In [9]: pg_instance
    Out[9]: stats @ Postgres (v18.1)

Here, the ``config_file`` parameter points to a file that contains the connection parameters as a
`psycopg-compatible <https://www.psycopg.org/psycopg3/docs/api/connections.html#psycopg.Connection.connect>`__ string.

.. note::

    PostgreSQL does not provide hinting support out-of-the-box.
    Therefore, PostBOUND uses the `pg_hint_plan <https://github.com/ossc-db/pg_hint_plan>`__ extension to add query hints.
    If you set up your own Postgres instance, make sure to install the extension.
    As an alternative, you can use `pg_lab <https://github.com/rbergm/pg_lab>`__, which extends Postgres with more advanced
    hinting capabilities and additional extension points for optimizer research.

If you are using DuckDB, you can load a database using :func:`pb.duckdb.connect() <postbound.duckdb.connect>`.

See the separate :doc:`core/databases` documentation for more details on the database abstraction.


.. _10minutes-workloads:

Workload Handling
-----------------

A :class:`Workload <postbound.Workload>` is a collection of queries that can be used to benchmark the
performance of different optimization strategies.
All queries are associated with labels that are typically used to retrieve them, e.g., ``job["1a"]``.
A workload provides rich functionality to retrieve (subsets of) the queries, such as by specific properties or randomly
to obtain a test set.

Following the *batteries included* philosophy, PostBOUND already ships some of the commonly used workloads in query
optimization.
These can be accessed from the :mod:`~postbound.workloads` module.
Specifically, the Join Order Benchmark (JOB, including variations such as JOB-light), the Stats Benchmark and the
Stack Benchmark are available out-of-the-box:

.. code-block:: ipython

    In [10]: stats = pb.workloads.stats()

    In [11]: stats
    Out[11]: Workload: Stats (146 queries)

You can also load your own workloads by using :func:`~postbound.workloads.read_workload` or
:func:`~postbound.workloads.read_csv_workload`.
See the separate :doc:`core/benchmarking` documentation for more details.

.. _10minutes-benchmarking:

Benchmarking
------------

Once you have implemented you own optimization algorithm, you can benchmark it using the
:func:`~postbound.bench.execute_workload` utility.

It produces a pandas DataFrame with the results of the executed queries:

.. code-block:: ipython

    In [12]: results = pb.bench.execute_workload(stats.first(3), pg_instance)

    In [13]: results
    Out[13]:
       exec_index label  ... optimization_pipeline  optimized_query
    0           1   q-1  ...                  None             None
    1           2   q-2  ...                  None             None
    2           3   q-3  ...                  None             None

    [3 rows x 15 columns]

If you want to store the results in a file file, you can use :func:`~postbound.util.write_df`.
Even simpler, you can pass a file to the *progressive_output* parameter to automatically flush all results to disk as
soon as they arrive.

The :class:`~postbound.bench.QueryPreparation` enables you to customize the
execution of the queries.
For example, you can ensure that all queries are executed as *EXPLAIN ANALYZE* to capture their query plans, or you can
prewarm the shared buffer before execution to ensure that timing measurements are not affected by I/O activity.
See the separate :doc:`core/benchmarking` documentation for more details.

.. hint::

    Ready to get started?
    Head over to the :doc:`setup` guide to learn how to install PostBOUND.
    If you want to learn more about the different parts of PostBOUND, take a look at the :doc:`core/index` section.
