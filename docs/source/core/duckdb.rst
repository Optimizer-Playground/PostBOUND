
DuckDB Interface
================

The DuckDB backend provides support for query execution and plan hinting. It is implemented in the
:mod:`~postbound.duckdb` module. Since DuckDB provides no support for optimizer hints that we rely on, we use a fork of
DuckDB called `quacklab <https://github.com/rbergm/quacklab>`_ that adds support for the necessary hints.
Even with the necessary hints in place, some limitations remain due to internal design of DuckDB. These are outlined in
the :ref:`limitation <duckdb-limitations>` below.


Setup
-----

To setup DuckDB, use the ``duckdb-setup.sh`` script located in the ``db-support/duckdb`` directory of the PostBOUND
repository. This script will:

1. pull the most recent (or a selected) version of quacklab
2. compile DuckDB from source and create the Python wheel file
3. optionally install the wheel in the current Python environment

DuckDB uses `uv <https://docs.astral.sh/uv/>` for the build process. If this is not installed on your system, the setup
script will install it automatically into a Python virtual environment. This environment can be controlled by a CLI
parameter and will default to *.venv* in the DuckDB directory. It might be a good idea to point this to the venv that you
use for your PostBOUND installation. In addition, you need a recent (C++ 17 compliant) C++ compiler on your system. Other
dependencies such as CMake will be installed by the script as needed.

The entire setup process might take some time, so feel free to grab a cup of coffee after starting the script.

.. tip::
    Once the setup is complete, you can import the database interface in your Python code directly by doing an
    ``import quacklab``. We use *quacklab* instead of *duckdb* to distinguish the modified version from the original
    DuckDB. As a nice side effect, you can install vanilla DuckDB as well as quacklab in the same Python environment
    without conflicts. Other than the name differences, the packages have the exact same API.

Once the setup is complete, you can create commonly-used database instances such as IMDB/JOB or Stats using the
``workload-setup.py`` script. This script must be executed while the virtual environment containing the quacklab
installation is active.
The workload setup shell scripts serve the same purpose but require the DuckDB executable to be available on your *PATH*.
This is currently not the case when building quacklab due to limitations of the build system.

Usage
-----

The DuckDB backend is available from the :mod:`~postbound.duckdb` module. You can create a connection to a DuckDB database
file using the :func:`~postbound.duckdb.connect` function. The returned database instance functions like any regular
:class:`~postbound.Database` interface:

.. code-block:: python

    import postbound as pb

    duck_instance = pb.duckdb.connect("stats.duckdb")
    stats = pb.workloads.stats()
    duck_instance.execute_query(stats["q-1"])

You can also import the "raw" database interface in your Python code directly by doing an ``import quacklab``. We use
*quacklab* instead of *duckdb* to distinguish the modified version from the original DuckDB. As a nice side effect, vanilla
DuckDB and quacklab can be installed side-by-side in the same Python environment without conflicts. Other than the name
differences, the packages have the exact same API.


Supported Backend Features
--------------------------

+-------------------------------------------+--------------------------------------------------------------------+
| Feature                                   | Status                                                             |
+===========================================+====================================================================+
| Query Execution                           | fully supported                                                    |
+-------------------------------------------+--------------------------------------------------------------------+
| Query Execution with timeouts             | fully supported                                                    |
+-------------------------------------------+--------------------------------------------------------------------+
| Schema interface                          | fully implemented                                                  |
+-------------------------------------------+--------------------------------------------------------------------+
| Statistics interface                      | fully implemented [#f1]_                                           |
+-------------------------------------------+--------------------------------------------------------------------+
| EXPLAIN parsing and query plan extraction | fully supported [#f2]_                                             |
+-------------------------------------------+--------------------------------------------------------------------+
| EXPLAIN ANALYZE plans                     | fully supported [#f2]_                                             |
+-------------------------------------------+--------------------------------------------------------------------+
| Extraction of cardinality estimates       | fully supported                                                    |
+-------------------------------------------+--------------------------------------------------------------------+
| Extraction of cost estimates              | not supported (see :ref:`limitations <duckdb-limitations>`)        |
+-------------------------------------------+--------------------------------------------------------------------+
| Plan hinting                              | partially supported (see :ref:`limitations <duckdb-limitations>`)  |
+-------------------------------------------+--------------------------------------------------------------------+

To obtain DuckDB query plans, you can either use the :meth:`~postbound.duckdb.DuckDBOptimizer.query_plan` method or parse
the EXPLAIN output manually using :func:`~postbound.duckdb.parse_duckdb_plan`. Both options yield the same results:

.. code-block:: python

    # obtain a query plan directly:
    plan = duck_instance.optimizer().query_plan(stats["q-1"])

    # this is equivalent to:
    explain_query = pb.transform.as_explain(stats["q-1"])
    raw_plan = duck_instance.execute_query(explain_query)
    equivalent_plan = pb.duckdb.parse_duckdb_plan(raw_plan)


.. _duckdb-optimizer-architecture:

Optimizer Essentials
--------------------

DuckDB employs a sequential optimizer that performs complex algebraic transformations (e.g., subquery unnesting) on the
logical query plan. The join order is determined using a dynamic programming algorithm based on [Moerkotte08]_. It uses a
simple "cost model" that sums the cardinality estimates of the input relations and the output cardinality of the
intermediate. Based on this join order, physical operators are selected following a rule-based approach. Basically, hash
joins are used whenever possible and nested loop joins are only used as a last resort.


.. _duckdb-limitations:

Limitations
-----------

The current implementation of the DuckDB execution engine imposes some strict limitations on the kind of hints we use
reliably. In particular, the implementation of the physical operators is tightly coupled with their selection rules.
For example, if the optimizer selects one join operator based on some property of the input query, the implementation of
that operator will rely on this property being satisfied. Using a different operator typically leads to execution errors.
As a consequence, operator hints can be specified, but should only be used with great care if at all.
At the same time, this makes optimization approaches using a costs model much less appealing.

Based on these limitations, the following features can be used reliably with the DuckDB backend:

- Join order hints and corresponding optimization strategies
- Cardinality hints and corresponding estimation strategies


References
----------

.. [Moerkotte08]
    Guido Moerkotte and Thomas Neumann: "Dynamic Programming Strikes Back" (SIGMOD 2008)
    DOI: `10.1145/1376616.1376672 <https://doi.org/10.1145/1376616.1376672>`_


.. rubric:: footnotes

.. [#f1]
    DuckDB maintains only a minimal set of statistics (due to the large variety of input sources).
    In particular, only the number of rows is reliably available. Other statistics can only be
    :ref:`emulated <database-statistics>`.

.. [#f2]
    DuckDB query plans lack one crucial piece of information: if a table appears multiple times in a query, we cannot
    distinguish between the different instances in the plan. For example, for the query
    ``SELECT former.title, latter.title FROM movies former, movies latter WHERE former.series = latter.series AND former.year < latter.year;``,
    both instances of the *movies* table will simply appear as *movies* in the plan and the alias information is lost.
    As a consequence, we cannot fully bind the base table references in the scan nodes. To mitigate this issue, we use a
    number of heuristics, e.g., by inferring the alias based on filter predicates, etc. However, these heuristics are not
    completely reliable and might fail in some cases.
