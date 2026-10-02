Core concepts
=============

This section is dedicated to the core modules of PostBOUND and how they interact with each other.
If you are a new user, it is highly recommended to read through this section to get a better understanding of the framework.
At the same time, the :doc:`../10minutes` tutorial provides a more fast-paced introduction.

Contents
--------

.. toctree::
   :maxdepth: 1

   hinting
   qal
   databases
   optimization
   benchmarking


Package structure
-----------------

==============================  =============================================================================================================================================================================================================================================================================================
Package                         Description
==============================  =============================================================================================================================================================================================================================================================================================
*root*                          The main entry point into PostBOUND as well as fundamental data structures like :class:`OptimizationPipeline <postbound.OptimizationPipeline>` or :class:`QueryPlan <postbound.QueryPlan>` are located here. See :doc:`optimization` for details.
:mod:`~postbound.qal`           Home of the query abstraction used throughout PostBOUND. See :doc:`qal` for details.
:mod:`~postbound.parser`        Contains our SQL parser
:mod:`~postbound.transform`     Collects different utilities to modify SQL queries
:mod:`~postbound.relalg`        Provides a simple relational algebra implementation to represent SQL queries.
:mod:`~postbound.db`            Contains all parts of PostBOUND that concern database interaction. See :doc:`databases` for details.
:mod:`~postbound.postgres`      Implements the database backend for PostgreSQL.
:mod:`~postbound.duckdb`        Implements the database backend for DuckDB.
:mod:`~postbound.mysql`         Provides a simple database backend implementation for MySQL. MySQL is currently provided on a best-effort basis and not an official backend. Not all features are implemented.
:mod:`~postbound.workloads`     Provides the :class:`~postbound.Workload` interface to represent query workloads and routines to load commonly-used benchmarks like JOB or Stats.
:mod:`~postbound.bench`         Contains benchmarking utilities to measure the performance of different optimizers and optimization settings. See :doc:`benchmarking` for details.
:mod:`~postbound.opt`           Provides utilities to aid with optimizer development, including optimization algorithms like :class:`~postbound.opt.DynamicProgrammingEnumerator` and additional utilities. See :doc:`../advanced/existing-strategies` for details.
:mod:`~postbound.validation`    Provides the basic definitions to check the applicability of optimizer prototypes to different queries and database systems. In addition, some commonly-used validations are implemented here.
:mod:`~postbound.util`          Utilities that are not really specific to query optimization find their home here. See the package documentation for more details.
:mod:`~postbound.vis`           Contains utilities to visualize different concepts in query optimization (join orders, join graphs, query execution plans, ...).
==============================  =============================================================================================================================================================================================================================================================================================
