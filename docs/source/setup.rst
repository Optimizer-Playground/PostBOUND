Setup
=====

Each installation of PostBOUND consists of two parts: the PostBOUND framework itself, as well as at least one database
instance (such as PostgreSQL or DuckDB) that is used to actually execute the optimized queries
(see :ref:`10minutes-db-connection`).
Depending on your use case, you can use an integrated setup where PostBOUND takes care of installing the framework as
well as the database instance(s), or you can manage the database instances on your own.

In the following, we describe the different installation methods for PostBOUND:

1. Installing PostBOUND via pip. This requires you to setup and manage the database instances on your own. Optionally,
   you can use the build tools shipped with PostBOUND to setup database instances.
2. A Docker-based installation that automates the entire setup of PostBOUND along with a Postgres and/or DuckDB instance.
3. A manual installation of PostBOUND. Optionally, you can use the build tools shipped with PostBOUND to setup database
   instances.

.. tip::

    Local installations of PostBOUND can be managed via `uv <https://docs.astral.sh/uv/>`_. Just update the local files
    via Git and sync the uv state. This also works within the Docker-based installation.


.. tip::

    For PostgreSQL, we support two different hinting backends: `pg_hint_plan <https://github.com/ossc-db/pg_hint_plan>`_
    and `pg_lab <https://github.com/rbergm/pg_lab>`_. While the former is widely used and easy to setup, it provides
    only basic hinting capabilities. In contrast, pg_lab allows for more fine-grained control over the optimizer's
    behavior. See :ref:`postgres-pghintplan-vs-pglab` for more details on the differences between both backends and which
    one to choose for your use case.


Pip-based installation
----------------------

This installation method is probably the easiest and fastest way to get started with PostBOUND.
Simply install the framework using pip:

.. code-block:: bash

    pip install postbound

.. tip::

    It is probably best to install PostBOUND in a new Python virtual environment to avoid dependency hell.

Afterwards, you need to setup and configure the database instances on your own.
For Postgres, this includes installing the `pg_hint_plan <https://github.com/ossc-db/pg_hint_plan>`_ extension to enable
query hinting (or `pg_lab <https://github.com/rbergm/pg_lab>`_).
Finally, create a connection file that contains the connection string to connect to your database.
See the documentation of :func:`~postbound.postgres.connect` for details.

If you want to use DuckDB, **do not install the official DuckDB Python package from PyPI**.
The reason is that DuckDB does not provide any hinting functionality out-of-the-box.
Therefore, we developed `quacklab <https://github.com/rbergm/quacklab>`_, a fork of DuckDB that adds hinting capabilities
very similar to pg_lab. It has Python bindings available via
`quacklab-python <https://github.com/rbergm/quacklab-python>`_.
By default, the latest stable release of quacklab is shipped with PostBOUND.


Manual Installation
-------------------

A manual installation requires a UNIX-based system, with Linux being the most well supported.
Windows users should use the Windows Subsystem for Linux (WSL) to run PostBOUND (which we do as well).
By default, PostBOUND is installed within a Python virtual environment using `uv <https://docs.astral.sh/uv/>`_:
in your local project, simply run ``uv add path/to/postbound/directory/``.

.. note::

    We test PostBOUND mostly on a WSL-based Ubuntu system and sporadically on MacOS.
    Therefore, these are the most well-supported platforms.
    Sadly, at the current time we cannot provide support for different setups.

A key requirement for PostBOUND is a running database server to execute queries against.
See the :ref:`hinting` description for more details on why this is necessary and what  functionality the database has to
provide.
Currently, the most well-supported DBS are PostgreSQL and DuckDB.
Limited support for MySQL is also available.
In the following, we assume that PostBOUND should interact with a Postgres server.

To ensure the smoothest interaction between PostBOUND and Postgres and to have the least configuration effort, the
Postgres server and PostBOUND should run in the same address space (i.e. on the same machine or within the same
virtualized environment). Basically, there are three different options to setup a Postgres server:

1. Installing Postgres on your own using the package manager or a binary distribution. This requires you to also install
   the `pg_hint_plan <https://github.com/ossc-db/pg_hint_plan>`_ extension to enable query hinting.
2. Installing Postgres using the `pg_lab <https://github.com/rbergm/pg_lab>`_ build tools. This provides the most
   complete hinting functionality, e.g. with support for base table cardinalities and parallel query plans.
3. Using the build tools that are shipped with PostBOUND. These create a local Postgres server by compiling from source
   and take care of setting up all required extensions (such as *pg_hint_plan*). The remainder of this section documents
   this installation method in detail.

.. admonition:: pg_lab vs. pg_hint_plan

    On the surface, *pg_hint_plan* and *pg_lab* provide very similar functionality.
    Both provide a hinting mechanism to embed optimizer decisions into the raw query text using special SQL comments.
    However, *pg_lab*  provides much more extensive hinting support, including for example cardinality hints for base
    tables, reliable hints for memoize and materialize operators and proper parallelization support.
    At the same time, *pg_lab* goes far beyond query hinting: it is an extension of Postgres that adds additional
    extension points to the query optimizer.
    These allow extensions to modify the optimizer's behavior in a fine-grained manner.
    In a way, *pg_lab* can be thought of as a low-level alternative to PostBOUND for Postgres-specific research.
    That is, if you want to learn all of the internal details of the Postgres optimizer and how to modify them.

The Postgres build tools compile a fresh server instance from source.
Therefore, you need a number of tools and libraries available on you system.
For Ubuntu-based distributions, these can be installed like so:

.. code-block:: bash

    sudo apt install \
        build-essential meson ninja-build flex bison curl pkg-config llvm clang \
        libreadline-dev libssl-dev libicu-dev liblz4-dev libossp-uuid-dev python3-dev \
        liburing-dev \
        git unzip zstd

Adjust the packages depending on your distribution.
Once the required packages are installed, you can use the build tools from the ``db-support/postgres`` directory to set
up a Postgres server.
All scripts support ``--help`` to view the available options.

.. code-block:: bash

    cd db-support/postgres
    ./postgres-setup.sh --pg-ver 18 --stop
    . ./postgres-start.sh

This will pull the latest stable release of Postgres 18, compile it from source and initialize a new database cluster
along with all required extensions (e.g., pg_hint_plan).
The server will have a default user that corresponds to your system username.
All binaries, data directory, etc. are placed under *postgres-server* in the working directory.
A central challenge of this manual setup is that the Postgres binaries, etc. are not globally available, e.g., on your
PATH.
This means that you cannot simply psql into your server.
To mitigate this situation, PostBOUND ships a number of management scripts:

- ``postgres-load-env.sh`` takes care of setting up all environment variables (PATH, C_INCLUDE_PATH, etc.) as necessary
- ``postgres-start.sh`` starts Postgres and also initializes your environment
- ``postgres-stop.sh`` shuts down a running server and undoes all changes to your environment

Since all of these scripts make changes to your global environment, they have to be sourced instead of executed.

You can also install multiple different Postgres versions side-by-side.
To do so, all scripts accept an optional installation path as parameter:

.. code-block:: bash

    ./postgres-setup.sh --pg-ver 16 --stop --dir /my/path
    . ./postgres-start.sh /my/path

.. tip::

    We recommend to always setup the Postgres server with the ``--stop`` option and not source this script directly.
    This ensures that your shell does not terminate in case the setup runs into any issues.
    Once the setup is completed, just start the server using the ``postgres-start.sh``.

After your server is setup and running, you can populate it with some well-known benchmarks, such as JOB, Stats or Stack.
PostBOUND provides simple setup scripts for these out-of-the-box:

.. code-block:: bash

    ./workload-job-setup.sh

These scripts assume that your Postgres server is running and you can simply use *psql* to connect to it.
Once again, you can use the ``--help`` option to view the available options (including ways to adapt the connection
parameters).

One last question is how to connect to the database server from within PostBOUND.
Internally, PostBOUND uses the `psycopg <https://www.psycopg.org/>`_ library to connect to Postgres.
You can use the ``postgres-psycopg-setup.sh`` script to create a connection file with the necessary parameters to connect
to the Postgres installation.
See the documentation of :func:`~postbound.postgres.connect` for more details on the config file
and alternative ways to establish a connection.

Now, you should be able to connect to the Postgres server using the following code:

.. code-block:: ipython

    In [1]: import postbound as pb

    In [2]: pg_instance = pb.postgres.connect(config_file=".psycopg_connection")

    In [3]: pg_instance
    Out[3]: stats @ Postgres (v18.1)

Putting things together, you can create an entirely new Postgres server like so:

.. code-block:: bash

    cd db-support/postgres
    ./postgres-setup.sh --pg-ver 18 --stop
    . ./postgres-start.sh
    ./workload-job-setup.sh
    ./postgres-psycopg-setup.sh job imdb
    cp .psycopg_connect_job ../..

Similar to the Postgres setup, you can also create a local DuckDB installation by compiling it from source.
However, note that since PostBOUND v0.22, quacklab/DuckDB is a first-party dependency of PostBOUND and is automatically
installed when you install PostBOUND via pip/uv.

For a manual installation, use the ``db-support/duckdb/duckdb-setup.sh`` script.
This script will automatically install the Python package into your PostBOUND virtual environment.
See the ``--help`` options for more details.

.. tip::

    DuckDB does not provide any hinting functionality out-of-the-box.
    Therefore, the setup creates a special version of DuckDB called `quacklab <https://github.com/rbergm/quacklab>`_,
    that adds basic hinting capabilities to DuckDB.
    This is also the reason why the setup compiles DuckDB from source instead of using a binary distribution.
    Sadly, the official build system of the Python extension does not include the DuckDB binary itself. If you want to
    use the DuckDB CLI with hinting capabilities, you also need to compile quacklab in addition to quacklab-python.

.. code-block:: bash

    source pb-venv/bin/activate  # make sure to activate the venv which contains quacklab
    cd db-support/duckdb
    ./duckdb-setup.sh
    ./workload-setup.py --workload imdb


Docker Installation
-------------------

The Docker-based installation essentially automates the manual installation process described above.
The resulting Docker container contains a virtual environment-based installation of PostBOUND and a Postgres (or pg_lab)
server as well as DuckDB completely configured and ready to use.
Optionally, you can also obtain an optimized Postgres server configuration and setup different benchmarks.

To create the Docker image, simply run ``docker build`` in the main PostBOUND directory.
You can specify the timezone of the image using the ``TIMEZONE`` ``--build-arg`` (see below).
You can customize the container with the following options via ``--env`` parameters (with the exception of *TIMEZONE*,
which must be specified as a `--build-arg` when creating the image).
Please note that the *run* command will invoke a lot of setup logic.
Hence, it will take a substantial amount of time to complete the installation (think hours).
This is because the container will compile a local Postgres server from source, import benchmarks, etc.
Use ``docker logs -f <container name>`` to monitor the installation progress.

+------------------------+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| Argument               | Allowed values                | Description                                                                                                                                                                                                                                                                         | Default       |
+========================+===============================+=====================================================================================================================================================================================================================================================================================+===============+
| ``TIMEZONE``           | Any valid timezone identifier | Timezone of the Docker container (and hence the Postgres server). It is probably best to just use the value of ``cat /etc/timezone``.                                                                                                                                               | ``UTC``       |
+------------------------+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``USERNAME``           | Any valid UNIX username       | The username within the Docker container. This will also be the Postgres user and password.                                                                                                                                                                                         | ``postbound`` |
+------------------------+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``SETUP_POSTGRES``     | *true* or *false*             | Whether a Postgres server should be setup. If ``USE_PGLAB`` is also set to *true*, a pg_lab server is created instead.                                                                                                                                                              | *true*        |
+------------------------+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``OPTIMIZE_PG_CONFIG`` | *true* or *false*             | Whether the Postgres configuration parameters should be automatically set based on your hardware platform. Rules are based on `PGTune <https://pgtune.leopard.in.ua/>`__ by `le0pard <https://github.com/le0pard>`__. See :ref:`pg-server-config` for more details.                 | *false*       |
+------------------------+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``PG_DISK_TYPE``.      | *SSD* or *HDD*                | In case the Postgres server is automatically configured (see ``OPTIMIZE_PG_CONFIG``) this indicates the kind of storage for the actual database. In turn, this influences the relative cost of sequential access and index-based access for the query optimizer.                    | *SSD*         |
+------------------------+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``PGVER``              | 16, 17, ...                   | The Postgres version to use. Notice that pg_lab supports fewer versions. This value is passed to the ``postgres-setup.sh`` script of the Postgres tooling (either under ``db-support`` or from pg_lab), which provides the most up to date list of supported versions.              | *18*          |
+------------------------+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``USE_PGLAB``          | *true* or *false*             | Whether to initialize a `pg_lab <https://github.com/rbergm/pg_lab>`__ server instead of a normal Postgres server. pg_lab provides advanced hinting capabilities and offers additional extension points for the query optimizer.                                                     | *false*       |
+------------------------+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``SETUP_DUCKDB``       | *true* or *false*             | Whether a local DuckDB installation should be created as part of the PostBOUND setup. This will compile DuckDB from source and install it under ``/postbound/db-support/duckdb/duckdb-server``.                                                                                     | *false*       |
+------------------------+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``SETUP_IMDB``         | *true* or *false*             | Whether an `IMDB <https://doi.org/10.14778/2850583.2850594>`__ instance should be created as part of the setup. PostBOUND can connect to the Postgres database using the ``.psycopg_connection_job`` config file. The DuckDB image will be available at ``/postbound/imdb.duckdb``. | *false*       |
+------------------------+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``SETUP_STATS``        | *true* or *false*             | Whether a `Stats <https://doi.org/10.14778/3503585.3503586>`__ instance should be created as part of the setup. PostBOUND can connect to the database using the ``.psycopg_connection_stats`` config file. The DuckDB image will be available at ``/postbound/stats.duckdb``.       | *false*       |
+------------------------+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``SETUP_STACK``        | *true* or *false*             | Whether a `Stack <https://doi.org/10.1145/3448016.3452838>`__ instance should be created as part of the setup. PostBOUND can connect to the database using the ``.psycopg_connection_stack`` config file. Note that we currently do not create a Stack image for DuckDB.            | *false*       |
+------------------------+-------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+

The Docker container makes port 5432 available to bind on the system.
This enables you to connect to the Postgres server from outside.
If you plan on using Jupyter for data analysis, consider also publishing port 8888 on the container to access the notebooks
from your client's browser.
Volumes are created at ``/postbound/`` and ``/pg_lab`` (only useful if pg_lab is actually enabled).
The PostBOUND installation itself is located at ``/postbound``.
If a vanilla Postgres server is used, it is installed at ``/postbound/db-support/postgres/postgres-server``.
pg_lab servers are installed at ``/pg_lab``.
If the pg_lab volume points to an existing (i.e. non-empty) directory, the setup assumes that this is already a valid
pg_lab installation and skips the corresponding setup.
This can be useful if multiple containers should share the same pg_lab installation.

.. danger::

    The setup script uses the existence of the target volumes as an indicator that the PostBOUND/pg_lab have already
    been set up. If you run into any troubles and need to re-run the setup script, make sure to remove the volumes or
    use a different volume path.

Once you log in to the container, the PostBOUND virtual environment will be activated automatically.
Likewise, all Postgres and DuckDB binaries are available on the *PATH*.

Putting things together, you can create a Docker container with PostBOUND running Postgres and DuckDB like so:

.. code-block:: bash

    docker build -t postbound --build-arg TIMEZONE=$(cat /etc/timezone) .

    docker run -dt \
        --shm-size 4G \
        --name postbound \
        --env SETUP_DUCKDB=true \
        --env SETUP_IMDB=true \
        --env SETUP_STATS=true \
        --env OPTIMIZE_PG_CONFIG=true \
        --env PG_DISK_TYPE=SSD \
        --env PGVER=18 \
        --env USE_PGLAB=true \
        --volume $PWD/vol-postbound:/postbound \
        --volume $PWD/vol-pglab:/pg_lab \
        --publish 5432:5432 \
        --publish 8888:8888 \
        postbound

    docker exec -it postbound /bin/bash

.. tip::

    Building the Docker container will take a while.
    This is expected and nothing to worry about.
    The build process involves downloading and compiling Postgres from source, as well as optionally setting up the
    databases for JOB, Stats and the like (which also includes downloading and importing them).
    If you also include DuckDB in the setup, this will also be compiled from source.
    During testing, we noticed that the creating an optimized build for DuckDB can take a substantial amount of time
    (around 30 to 60 minutes on a reasonably fast machine).
    You can follow the current progress via ``docker logs -f postbound`` (provided that your container is called *postbound*).
