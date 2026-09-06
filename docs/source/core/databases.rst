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
    you need to be careful when you deviate from this assumption. Especially, make sure not to wrap the database in a
    :class:`~postbound.db.ResultCache` if the data can change while PostBOUND is running.


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

Since the underlying database is usually assumed to be static, you can wrap any database in a
:class:`~postbound.db.ResultCache` to prevent repeated execution of non-benchmark queries. This can be especially useful
when running complex queries to calculate advanced statistics. The cache behaves like any other
:class:`~postbound.Database` and only intercepts :func:`~postbound.Database.execute_query`. Use
:meth:`~postbound.db.ResultCache.create_cache` to obtain one, optionally backed by a JSON file that persists the cached
results across processes.

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

The :class:`~postbound.db.StatisticsCatalog` serves as a unified statistics catalog. It is the central repository for all
base statistics that are typically maintained by database systems. The catalog can be used to retrieve table cardinalities,
most common values, etc. Use :meth:`~postbound.Database.statistics` to access the them.

One important design consideration of the statistics catalog is that different systems maintain vastly different kinds of
statistics. For example, Postgres does not keep track of minimum or maximum values for columns, but derives them from the
histograms. On the other hand, MySQL does not store most common values and pretty much entirely relies on histograms.
Such differences hinder the implementation of optimizer prototypes if they rely on a specific set of statistics.
To address this, PostBOUND can compute the missing statistics on live data instead: whenever a database system does not
maintain a specific statistic, an equivalent SQL query is issued that computes the same information. For example, say you
want to retrieve the most common values of a column on MySQL. Calling
:meth:`~postbound.db.StatisticsCatalog.most_common_values` will instead issue the following query:
``SELECT col, COUNT(*) FROM tab GROUP BY col ORDER BY COUNT(*) DESC LIMIT 10``.

This behavior is controlled by the module-level :data:`~postbound.db.enable_emulation_fallback` flag. If it is disabled,
database systems raise an :exc:`~postbound.db.UnsupportedDatabaseFeatureError` for statistics they do not maintain
themselves.

The computation itself is implemented by :class:`~postbound.db.PreciseStatistics`, which is a full
:class:`~postbound.db.StatisticsCatalog` in its own right. You can use it directly to force *all* statistics to be
computed on live data, even for systems that do maintain them natively. Since this can be pretty expensive, prefer
:meth:`~postbound.db.PreciseStatistics.create_cached`, which puts a :class:`~postbound.db.ResultCache` underneath.

.. important::

    One downside of computing statistics this way is granularity: by issuing SQL queries, you always get perfect
    statistics (since they are computed on live data). However, an actual statistics catalog might be slightly
    outdated. As a consequence, database systems with computed statistics might perform better than their counterparts
    with actual statistics.


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
:class:`~postbound.postgres.PostgresDatabase` provides full prewarming support.
For other systems, you can use simple ``isinstance`` checks to see if the database supports prewarming.

.. tip::

    `pg_lab <https://github.com/rbergm/pg_lab>`_-based installations of Postgres also provide support for proper cold
    starts in Postgres. The :class:`~postbound.postgres.PostgresDatabase` has a corresponding
    :meth:`~postbound.postgres.PostgresDatabase.cooldown_tables` method.
