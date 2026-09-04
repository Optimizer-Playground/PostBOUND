Database Abstraction
====================

Databases serve two main purposes in PostBOUND:

1. **Query execution:** databases are used to execute SQL queries and generate :doc:`hints <hinting>` for the
   :doc:`optimizer pipelines <optimization>`.
2. **Data access:** databases provide access to the underlying data schema and statistics catalog to facilitate the
   optimizer implementation.

Both use cases are described in this document. For specifics on the Postgres interaction, see the
:doc:`separate document <postgres>`.
All of the functionality is handled by the central :class:`~postbound.Database` interface. Specific database systems
implement this interface to provide connections for their respective systems. The idea behind this decision is to allow
researchers to implement their algorithms independently of the underlying DBMS since access to statistics, etc. is unified.
Each instance of the :class:`~postbound.Database` class is connected to an actual database server.

.. note::

    Naturally, some differences between the database systems cannot hidden behind an interface and some functionality
    simply is not available for all systems. In these cases, functions can require instances of specific database systems
    and database interfaces can raise an error if a specific feature is not available. However, this should only be a last
    resort and the interface is designed to be as generic as possible.

.. warning::

    Currently, the database interface assumes that the underlying database is not modified while PostBOUND is running and
    you need to be careful when you deviate from this assumption. Especially, make sure to disable the
    :ref:`query cache <query-cache>` before executing any queries.


Database Backends
-----------------

.. toctree::
    :maxdepth: 1

    postgres
    duckdb
    mysql


Query execution
---------------

Queries can be executed via the :func:`~postbound.Database.execute_query` method. This method takes an
:class:`~postbound.SqlQuery` or a raw query string as input and provides the result set of the query as output.
By default, the database tries to simplify the result set to make it easier to work with. Specifically, if the query
returns just a single row with a single column, the result is returned as a scalar value instead of a nested list. See the
method documentation for more details on the simplification logic.
This behavior can be controlled with the ``raw`` parameter.

.. tip::

    Some database systems also support timeouts during query execution. In this case, a separate
    :func:`~postbound.db.TimeoutSupport.execute_with_timeout` method is also available on the database interface.
    See :class:`~postbound.db.TimeoutSupport` for more details.

.. _query-cache:

Since the underlying database is usually assumed to be static, you can use a query cache to prevent repeated execution of
non-benchmark queries. This can be especially useful when running complex queries to calculate advanced statistics.
Caching can be controlled either globally via the :attr:`~postbound.Database.cache_enabled` property, or by setting
the parameter on :func:`~postbound.Database.execute_query` calls. See the documentation on
:class:`~postbound.Database` for more details.

If the query execution fails for some reason, a :exc:`~postbound.db.DatabaseServerError` or
:exc:`~postbound.db.DatabaseUserError` is raised - depending on the error's cause.


.. _hinting-interface:

Hint generation
----------------

The :class:`~postbound.db.HintService` is used to enforce PostBOUND's optimization decisions while executing the queries on
the actual database system. Its behavior is entirely specific to the database. Hinting does not execute any query by
itself. Instead, the hinting interface provides a transformed version of the query depending on the database system's
requirements.

The hint service of each database can be accessed via :meth:`~postbound.Database.hinting`.


Optimizer interaction
---------------------

Each database provides simple access to some core optimizer functionality as part of the
:class:`~postbound.db.OptimizerInterface`. This includes retrieving query plans or estimates for cost and cardinalities.
The optimizer functionality can be accessed by calling :meth:`~postbound.Database.optimizer`.

.. tip::

    To obtain the cost or cardinality estimate for an arbitrary query plan, combine the :ref:`hinting-interface` with the
    optimizer interface. This can be further combined with :func:`~postbound.transform.extract_query_fragment` to
    get estimates or plans for subqueries.

.. _database-infrastructure:

Schema access
-------------

Information about tables, columns, indexes, datatypes, etc. of the database are captured in the
:class:`~postbound.db.DatabaseSchema`. Use :meth:`~postbound.Database.schema` to get the schema of the current database.
Most of the schema information is accessible via dedicated methods, such as :meth:`~postbound.db.DatabaseSchema.tables` or
:meth:`~postbound.db.DatabaseSchema.datatype`. You can also access a compact representation of the schema via
:meth:`~postbound.db.DatabaseSchema.as_graph`. This method provides a
`networkx-based directed graph <https://networkx.org/>`_ with edges that correspond to primary key/foreign key
relationships in the schema.


.. _database-statistics:

Statistics catalog
------------------

The :class:`~postbound.db.DatabaseStatistics` serves as a unified statistics catalog. It is the central repository for all
base statistics that are typically maintained by database systems. The catalog can be used to retrieve table cardinalities,
most common values, etc. Use :meth:`~postbound.Database.statistics` to access the them.

One important design consideration of the statistics catalog is that different systems maintain vastly different kinds of
statistics. For example, Postgres does not keep track of minimum or maximum values for columns, but derives them from the
histograms. On the other hand, MySQL does not store most common values and pretty much entirely relies on histograms.
Such differences hinder the implementation of optimizer prototypes if they rely on a specific set of statistics.
To address this, :class:`~postbound.db.DatabaseStatistics` offer an *emulation mode*. The basic idea is that whenever a
database system does not maintain a specific statistic, an equivalent SQL query is issued that computes the same
information. For example, say you want to retrieve the most common values of a column on MySQL. Calling
:meth:`~postbound.db.DatabaseStatistics.most_common_values` will instead issue the following query:
``SELECT col, COUNT(*) FROM tab GROUP BY col ORDER BY COUNT(*) DESC LIMIT 10``.

Since these computations can be pretty expensive, the statistics catalog provides its own
:ref:`caching control <query-cache>` that is independent of the global cache setting. By switching
:attr:`~postbound.db.DatabaseStatistics.cache_enabled` on, queries from the statistics catalog are always cached, no matter
what the global cache setting is. Setting this attribute to *None* falls back to the global cache setting.

.. important::

    One downside of the emulation approach is granularity: by issuing SQL queries to emulate statistics, you always get
    perfect statistics (since they are computed on live data). However, an actual statistics catalog might be slightly
    outdated. As a consequence, database systems with emulated statistics might perform better than their counterparts with
    actual statistics.

    If the emulation leads to weird results, you can disable it via the
    :attr:`~postbound.db.DatabaseStatistics.emulation_fallback` attribute. You can also go the different route and force
    all statistics to be emulated (even if the database system actually supports them) by setting
    :attr:`~postbound.db.DatabaseStatistics.emulated`.

    See :class:`~postbound.db.DatabaseStatistics` for more details.


Utilities
---------

In addition to the core interfaces, the database module also provides some convenience functions to simplify working with
databases.

The :class:`~postbound.db.DatabasePool` is used to keep track of active database connections. It is mostly used to quickly
get :class:`~postbound.Database` instances for the currently active database system. Throughout PostBOUND's source code
you will frequently see the following pattern in function signatures: ``db: Optional[Database] = None``. If no database is
provided, the current database is inferred from the database pool. This allows you to just safe some typing.
You can also use :func:`~postbound.db.current_database` to retrieve the active database instance, provided that there is
just one (which should usually be the case).

Performance measurements can be heavily influenced by the database system's page cache. If a lot of table data is already
cached, much less I/O is required and queries appear much faster. To mitigate these issues to some extent, PostBOUND
provides means to simulate query execution on a perfectly pre-warmed database (i.e. all required pages are already in the
shared buffer). This is achieved via the :meth:`~postbound.db.PrewarmingSupport.prewarm_tables` method. Since not all
database provide this kind of functionality and it is also not a core feature of the database interface, this method is
part of an extra :class:`~postbound.db.PrewarmingSupport` protocol. Notably, the
:class:`~postbound.postgres.PostgresInterface` provides full prewarming support.
For other systems, you can use simple ``isinstance`` checks to see if the database supports prewarming.

.. tip::

    `pg_lab <https://github.com/rbergm/pg_lab>`_-based installations of Postgres also provide support for proper cold
    starts in Postgres. The :class:`~postbound.postgres.PostgresInterface` has a corresponding
    :meth:`~postbound.postgres.PostgresInterface.cooldown_tables` method.
