Postgres Interface
==================

Postgres is the primary supported backend for PostBOUND. Historically, it was the first and only supported backend and has
influenced the overall design of the :doc:`database abstraction <databases>`. The Postgres backend was eventually
generalized into the database abstraction. Therefore, all features of the database abstraction are supported by the
Postgres backend. Additionally, the Postgres backend provides a number of features that make working with a Postgres server
easier. Since these are specific to Postgres, they are not part of the general database abstraction. This document outlines
the general usage as well as the special features of the Postgres backend.


Setup
-----

To use the Postgres backend, you need a running Postgres server. To make use of the hinting functionality, this server
must at least have the `pg_hint_plan <https://github.com/ossc-db/pg_hint_plan>`_ extension installed and enabled.
For more advanced hinting features, we recommend using `pg_lab <https://github.com/rbergm/pg_lab>`_. In addition to the
pg_hint_plan hints, pg_lab provides support for base table cardinalities and reliable support for memoize operators,
materialize operators, and parallel execution of subplans.

To connect to the Postgres server, you need to provide the connection parameters either via a connection string or
(recommended) via a config file. See the :func:`~postbound.postgres.connect` function for details. The Postgres interface
will detect the available hinting mechanism automatically.

The PostBOUND repository contains utility scripts to setup a local Postgres server with pg_hint_plan from scratch. These
are located in the ``db-support/postgres`` directory. The ``postgres-setup.sh`` script will install Postgres, create a
new database cluster, and setup a Postgres server with the pg_hint_plan extension enabled. The
``postgres-config-generator.py`` utility will generate an optimized Postgres configuration based on your hardware based on
`PGTune <https://pgtune.leopard.in.ua/>`_. Finally, the ``workloadXYZ-setup.sh`` scripts will create commonly-used
database instances such as IMDB/JOB, Stats, or Stack on your Postgres server.


Usage
-----

The Postgres backend is available from the :mod:`~postbound.postgres` module. You can create a connection to a Postgres
server using the :func:`~postbound.postgres.connect` function. The returned database instance functions like any regular
:class:`postbound.Database` interface:

.. code-block:: python

    import postbound as pb

    pg_instance = pb.postgres.connect(config_file="pg-connect.toml")
    job = pb.workloads.job()
    pg_instance.execute_query(job["1a"])


Supported Backend Features
--------------------------

+-------------------------------------------+-------------------+
| Feature                                   | Status            |
+===========================================+===================+
| Query Execution                           | fully supported   |
+-------------------------------------------+-------------------+
| Query Execution with timeouts             | fully supported   |
+-------------------------------------------+-------------------+
| Schema interface                          | fully implemented |
+-------------------------------------------+-------------------+
| Statistics interface                      | fully implemented |
+-------------------------------------------+-------------------+
| EXPLAIN parsing and query plan extraction | fully supported   |
+-------------------------------------------+-------------------+
| EXPLAIN ANALYZE plans                     | fully supported   |
+-------------------------------------------+-------------------+
| Extraction of cardinality estimates       | fully supported   |
+-------------------------------------------+-------------------+
| Extraction of cost estimates              | fully supported   |
+-------------------------------------------+-------------------+
| Plan hinting                              | fully supported   |
+-------------------------------------------+-------------------+


Advanced Backend Features
-------------------------

In addition to the standardized database abstraction features, the Postgres backend provides the following additional
features:

* cache warmup via :meth:`~postbound.postgres.PostgresInterface.prewarm_tables`
* server configuration via :meth:`~postbound.postgres.PostgresInterface.apply_configuration`
* statistics maintenance via :meth:`~postbound.postgres.PostgresStatisticsInterface.update_statistics`
* server management with :func:`~postbound.postgres.start`, :func:`~postbound.postgres.stop`, and
  :func:`~postbound.postgres.is_running`
* parallel query execution based on the :class:`~postbound.postgres.ParallelQueryExecutor`
* simple database manipulation with the :class:`~postbound.postgres.WorkloadShifter`


Query Plans
-----------

To obtain DuckDB query plans, you can either use the :meth:`~postbound.postgres.PostgresOptimizer.query_plan` method or
parse the EXPLAIN output manually using :class:`~postbound.postgres.PostgresExplainPlan`. Both options yield the same
results:

.. code-block:: python

    # obtain a query plan directly:
    plan = pg_instance.optimizer().query_plan(job["1a"])

    # this is equivalent to:
    explain_query = pb.transform.as_explain(job["1a"])
    raw_plan = pg_instance.execute_query(explain_query)
    equivalent_plan = pb.postgres.PostgresExplainPlan(raw_plan)


.. _pg-server-config:

Optimizing the Server Configuration
-----------------------------------

The PostBOUND repository contains a utility script
``db-support/postgres/postgres-config-generator.py`` that generates an optimized Postgres configuration based on your
hardware. The script uses rules from `PGTune <https://pgtune.leopard.in.ua/>`_. The script outputs an SQL file that
contains the necessary ``ALTER SYSTEM`` commands to modify the *postgresql.conf*. Please note that these are just
heuristics that might not be optimal for your workload. Furthermore, the settings are intended for a dedicated database
server and a single user, single query at-a-time scenario.

One fragile aspect of the script is figuring out whether the database is stored on an SSD or HDD. If the script misdetects
the storage type or raises an error, you can manually specify it via ``--disk-type``.


.. _postgres-pghintplan-vs-pglab:

Hinting Backends
-----------------

The :class:`~postbound.postgres.PostgresInterface` supports two different hinting backends: the widely-used
`pg_hint_plan <https://github.com/ossc-db/pg_hint_plan>`_ and the research-focused
`pg_lab <https://github.com/rbergm/pg_lab>`_. pg_lab is a fork of vanilla Postgres that adds additional extension points
to the server. These extension points allow to control optimizer internals in a fine-grained manner. The hinting extension
shipped with pg_lab uses these extension points to provide more reliable and more detailed hinting features compared to
pg_hint_plan.

Upon establishing a server connection, the Postgres interface automatically detects which hinting backend is available on
the server and adjusts its hinting dialect used in :meth:`~postbound.postgres.PostgresHintService.generate_hints`
accordingly. If for some reason you want to change the current hinting dialect, you can do so via the
:attr:`~postbound.postgres.PostgresHintService.backend` attribute available via
:meth:`~postbound.postgres.PostgresInterface.hinting`.
