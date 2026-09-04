# PostBOUND

![GitHub License](https://img.shields.io/github/license/Optimizer-Playground/PostBOUND)
![GitHub Release](https://img.shields.io/github/v/release/Optimizer-Playground/PostBOUND?color=blue)

<p align="center">
    <img src="https://raw.githubusercontent.com/Optimizer-Playground/PostBOUND/refs/heads/main/docs/figures/postbound-logo.svg" style="width: 150px; margin: 15px;" alt="PostBOUND logo">
</p>

PostBOUND is a Python framework for studying query optimization in database systems.
At a high level, PostBOUND has the following goals and features:

- 🏃‍♀️ **Rapid prototyping:** PostBOUND allows researchers to implement specific phases of the optimization process, such
  as the cardinality estimator or the join order optimization. Researchers can focus precisely on the parts that they are
  studying and use _reasonable_ defaults for the rest. See [🧑‍🏫 Example](#-example) for how this looks in practice.
- 🧰 **No boilerplate:** to ease the implementation of the optimizers, PostBOUND provides a large toolbox of
  support functionality, such as query parsing, join graph construction, relational algebra or database statistics.
- 📊 **Transparent benchmarks:** once a new optimizer prototype is completed, the benchmarking tools allow to compare this
  prototype against other optimization strategies in a transparent and reproducible way. As a core design principle,
  PostBOUND executes queries on an actual database system such as PostgreSQL, or DuckDB, rather than research systems or
  artificial "lab" comparisons. See [💡 Essentials](#-essentials) for more information on this.
- 🔋 **Batteries included:** in addition to the Python package, PostBOUND provides a lot of utilities to setup databases
  and load commonly used benchmarks (e.g., JOB, Stats and Stack).

| **[💻 Installation](https://postbound.readthedocs.io/en/latest/setup.html)**
| **[📖 Documentation](https://postbound.readthedocs.io/en/latest/)**
| **[🧑‍🏫 Examples](https://github.com/Optimizer-Playground/PostBOUND/tree/main/examples)** |


## ⚡️ Quick Start

An installation of PostBOUND consists of two parts: the PostBOUND framework itself, as well as a running database instance
(such as PostgreSQL or DuckDB) that is used to actually execute the optimized queries (see [💡 Essentials](#-essentials)
below).

The easiest way to install PostBOUND is via Pip:

```sh
pip install postbound
```

Afterwards, you can [connect](https://postbound.readthedocs.io/en/latest/10minutes.html#database-connection) it to a
Postgres server running [pg_hint_plan](https://github.com/ossc-db/pg_hint_plan).

If you prefer a more integrated setup, we provide a Docker image that contains PostBOUND as well as a readily-configured
Postgres server or DuckDB installation.
You can build the Docker image with the following command:

```sh
docker build -t postbound --build-arg TIMEZONE=$(cat /etc/timezone) .
```

Once the image is built, you can create any number of containers with different setups.
For example, to create a container with a local Postgres instance (using [pg_lab](https://github.com/rbergm/pg_lab)) and
setup the Stats, JOB and Stack benchmarks, use the following command:

```sh
docker run -dt \
    --shm-size 4G \
    --name postbound \
    --env USE_PGLAB=true \
    --env OPTIMIZE_PG_CONFIG=true \
    --env PG_DISK_TYPE=SSD \
    --env SETUP_DUCKDB=false \
    --env SETUP_STATS=true \
    --env SETUP_IMDB=false \
    --env SETUP_STACK=false \
    --volume $PWD/vol-postbound:/postbound \
    --volume $PWD/vol-pglab:/pg_lab \
    --publish 5432:5432 \
    --publish 8888:8888 \
    postbound
```

All supported build arguments are listed under [Docker options](#-docker-options).
See [Essentials](#-essentials) for why a database instance is necessary.
For Postgres, adjust the amount of shared memory depending on your machine.
Note that the initial start of the container will take a substantial amount of time.
This is because the container needs to compile a fresh Postgres server from source, download and import workloads, etc.
Use `docker logs -f postbound`  to monitor the startup process.

> [!TIP]
> Shared memory is used by Postgres for its internal caching and therefore paramount for good server performance.
> The general recommendation is to set it to at least 1/4 of the available RAM.

The Postgres server will be available at port 5432 from the host machine (using the user _postbound_ with the same
password).
You can also create a local DuckDB installation by setting `SETUP_DUCKDB` to _true_.
If you plan on using Jupyter for data analysis, also publish port 8888.
The volume mountpoints provide all internal files from PostBOUND and pg_lab (if used).

You can connect to the PostBOUND container using the usual

```sh
docker exec -it postbound /bin/bash
```

The shell enviroment is setup to have PostBOUND available in a fresh Python virtual environment (which is activated by
default).
Furthermore, all Postgres utilities are available on the _PATH_.

> [!TIP]
> If you want to install PostBOUND directly on your machine, the
> [documentation](https://postbound.readthedocs.io/en/latest/setup.html) provides a detailed setup guide.


## 💡 Essentials

As a central design decision, PostBOUND is not integrated into a specific database system.
Instead, it is implemented as a Python framework operating on top of a running database instance.
In the end, all query plans generated by PostBOUND should be executed on a real-world database system.
This decision was made to ensure that the optimization strategies are actually useful in practice and we treat the execution
time as the ultimate measure of optimization quality.

However, this decision means that we need a way to ensure that the optimization decisions made within the framework are
actually used when executing the query in the context of the target database.
This is achieved by using query hints which typically encode the optimization decisions in comment blocks within the query.

In the case of Postgres, this interaction roughly looks like this:

<p align="center">
  <img src="https://raw.githubusercontent.com/Optimizer-Playground/PostBOUND/refs/heads/main/docs/figures/postbound-pg-interaction.svg" style="width: 600px; margin: 15px;">
</p>

Users implement their optimization strategies in terms of
[optimization pipelines](https://postbound.readthedocs.io/en/latest/core/optimization.html).
For an incoming SQL query, the pipeline computes an abstract representation of the resulting query plan without user
intervention.
This plan is automatically converted into equivalent hints for the target database system such as Postgres or DuckDB.
Afterwards, the hints ensure that the database system executes the query plan that was originally computed in the optimization
pipeline.

Depending on the actual database system, the hints might differ in syntax as well as semantics.
Generally speaking, PostBOUND figures out which hints to use on its own, without user intervention.
In the case of PostgreSQL,  PostBOUND relies on either [pg_hint_plan](https://github.com/ossc-db/pg_hint_plan) or
[pg_lab](https://github.com/rbergm/pg_lab) to provide the necessary hinting functionality when targeting Postgres.
For DuckDB, PostBOUND uses [quacklab](https://github.com/rbergm/quacklab) hints to guide the optimizer.

> [!NOTE]
> PostBOUND's database interaction is designed to be independent of a specific system (such as PostgreSQL, Oracle, ...).
> However, the current implementation is most complete for PostgreSQL and DuckDB with limited support for MySQL.
> This is due to practical reasons, mostly our own time budget and the popularity of the two systems in the optimizer research
> community.


## 🧑‍🏫 Example

The typical end-to-end workflow using PostBOUND looks like this:

1. **Implement your new optimization strategy**. To do so, you need to figure out which parts of the optimization process you
   want to customize and what the most appropriate optimization pipeline is. In a nutshell, the optimization pipeline is
   a mental model of how the optimizer works. Commonly used pipelines are the textbook-style pipeline (i.e. using plan
   enumerator, cost model and cardinality estimator), or the multi-stage pipeline which first computes a join order and
   afterwards selects the best physical operators. The pipeline determines which interfaces can be implemented.
2. Select your **target database system** and **benchmark** that should be used for the evaluation.
3. Optionally, select different optimization strategies that you want to compare against.
4. Use the **benchmarking tools** to execute the workload against the target database system.

For example, a random join order optimizer could be implemented like this:

```python
import random

import postbound as pb


# Step 1: define our optimization strategy.
# In this example we develop a simple join order optimizer that
# selects a linear join order at random.
# We delegate most of the actual work to the pre-defined join grap
# that keeps track of free tables.
class RandomJoinOrderOptimizer(pb.JoinOrderOptimization):
    def optimize_join_order(self, query: pb.SqlQuery) -> pb.LogicalJoinTree:
        join_tree = pb.LogicalJoinTree()
        join_graph = pb.opt.JoinGraph(query)

        while join_graph.contains_free_tables():
            candidate_tables = [
                path.target_table for path in join_graph.available_join_paths()
            ]
            next_table = random.choice(candidate_tables)

            join_tree = join_tree.join_with(next_table)
            join_graph.mark_joined(next_table)

        return join_tree

    def describe(self) -> pb.util.jsondict:
        return {"name": "random-join-order"}


# Step 2: connect to the target database, load the workload and
# setup the optimization pipeline.
# In our case, we evaluate on the Join Order Benchmark on Postgres
pg_imdb = pb.postgres.connect(config_file=".psycopg_connection_job")
job = pb.workloads.job()

optimization_pipeline = (
    pb.MultiStageOptimizationPipeline(pg_imdb)
    .use(RandomJoinOrderOptimizer())
    .build()
)

# (Step 3): in this example we just compare against the native Postgres optimizer
# Therefore, we do not need to setup any additional optimizers.

# Step 4: execute the workload.
# We use the QueryPreparationService to prewarm the database buffer and run all
# queries as EXPLAIN ANALYZE.
query_prep = pb.bench.QueryPreparation(
    prewarm=True, analyze=True, preparatory_statements=["SET geqo TO off;"]
)
native_results = pb.bench.execute_workload(
    job,
    on=pg_imdb,
    query_preparation=query_prep,
    workload_repetitions=3,
    progressive_output="job-results-native.csv",
    logger="tqdm",
)
optimized_results = pb.bench.execute_workload(
    job,
    on=optimization_pipeline,
    query_preparation=query_prep,
    workload_repetitions=3,
    progressive_output="job-results-optimized.csv",
    logger="tqdm",
)

```


## 🤬 Issues

Something feels wrong or broken, or a part of PostBOUND is poorly documented or otherwise unclear?
Please don't hestitate to file an issue or open a pull request!
PostBOUND is not one-off software, but an ongoing research project.
We are always happy to improve both PostBOUND and its documentation and we feel that the user experience (specifically,
_your_ user experience) is a very important part of this.


## 🫶 Reference

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{bergmann2025elephant,
  author       = {Rico Bergmann and
                  Claudio Hartmann and
                  Dirk Habich and
                  Wolfgang Lehner},
  title        = {An Elephant Under the Microscope: Analyzing the Interaction of Optimizer
                  Components in PostgreSQL},
  journal      = {Proc. {ACM} Manag. Data},
  volume       = {3},
  number       = {1},
  pages        = {9:1--9:28},
  year         = {2025},
  url          = {https://doi.org/10.1145/3709659},
  doi          = {10.1145/3709659},
  timestamp    = {Tue, 01 Apr 2025 19:03:19 +0200},
  biburl       = {https://dblp.org/rec/journals/pacmmod/BergmannHHL25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

---


## 📖 Documentation

A detailed documentation of PostBOUND is available [here](https://postbound.readthedocs.io/en/latest/).


## 🐳 Docker options

The following options can be used when starting the Docker container as `--env` parameters (with the exception of _TIMEZONE_,
which must be specified as a `--build-arg` when creating the image).

| Argument | Allowed values | Description | Default |
|----------|----------------|-------------|---------|
| `TIMEZONE` | Any valid timezone identifier | Timezone of the Docker container (and hence the Postgres server). It is probably best to just use the value of `cat /etc/timezone` | `UTC` |
| `USERNAME` | Any valid UNIX username. | The username within the Docker container. This will also be the Postgres user and password. | `postbound` |
| `SETUP_POSTGRES` | `true` or `false` | Whether to include a Postgres server in the setup. By default, this is a vanilla Postgres server with the latest minor release. However, this can be customized with `USE_PGLAB` and `PGVER`. | `true` |
| `USE_PGLAB` | `true` or `false` | Whether to initialize a [pg_lab](https://github.com/rbergm/pg_lab) server instead of a normal Postgres server. pg_lab provides advanced hinting capabilities and offers additional extension points for the query optimizer. | `false` |
| `OPTIMIZE_PG_CONFIG` |  `true` or `false` | Whether the Postgres configuration parameters should be automatically set based on your hardware platform. Rules are based on [PGTune](https://pgtune.leopard.in.ua/) by [le0pard](https://github.com/le0pard). **If you use this setting, you should also specify the `PG_DISK_TYPE`.** Otherwise, the optimization process might fail! | `false` |
| `PG_DISK_TYPE` | `SSD` or `HDD` | In case the Postgres server is automatically configured (see `OPTIMIZE_PG_CONFIG`) this indicates the kind of storage for the actual database. In turn, this influences the relative cost of sequential access and index-based access for the query optimizer. | `SSD` |
| `PGVER` | 16, 17, ... | The Postgres version to use. Notice that pg_lab supports fewer versions. This value is passed to the `postgres-setup.sh` script of the Postgres tooling (either under `db-support` or from pg_lab), which provides the most up to date list of supported versions. | 18 |
| `SETUP_DUCKDB` | `true` or `false` | Whether DuckDB-support should be added to PostBOUND. If enabled, a [DuckDB version with hinting support](https://github.com/rbergm/quacklab) will be compiled and images for all selected benchmarks will be created. Please be aware that during testing we noticed that creating an optimized build of DuckDB takes a lot of time on some platforms (think a couple of hours). | `false` |
| `SETUP_IMDB` | `true` or `false` | Whether an [IMDB](https://doi.org/10.14778/2850583.2850594) instance should be created as part of the setup. If a Postgres server is included in the setup, PostBOUND can connect to the database using the `.psycopg_connection_job` config file. For DuckDB, the database will be located at `/postbound/imdb.duckdb`. | `false` |
| `SETUP_STATS` | `true` or `false` | Whether a [Stats](https://doi.org/10.14778/3503585.3503586) instance should be created as part of the setup. If a Postgres server is included in the setup, PostBOUND can connect to the database using the `.psycopg_connection_stats` config file. For DuckDB, the database will be located at `/postbound/stats.duckdb` | `false` |
| `SETUP_STACK` | `true` or `false`| Whether a [Stack](https://doi.org/10.1145/3448016.3452838) instance should be created as part of the Postgres setup. If a Postgres server is included in the setup, PostBOUND can connect to the database using the `.psycopg_connection_stack` config file. DuckDB is currently not supported. | `false` |

The PostBOUND source code is located at `/postbound`. If pg_lab is being used, the corresponding files are located at `/pg_lab`.
The container automatically exposes the Postgres port 5432 and provides volume mountpoints at `/postbound` and `/pg_lab`.
These mountpoints can be used as backups or to easily ingest data into the container.
If the pg_lab mountpoint points to an existing (i.e. non-empty) directory, the setup assumes that this is already a valid
pg_lab installation and skips the corresponding setup.

> [!TIP]
> pg_lab provides advanced hinting support (e.g. for materialization or cardinality hints for base tables) and offers
> additional extension points for the query optimizer (e.g. hooks for the different cost functions).
> If pg_lab is not used, the Postgres server will setup pg_hint_plan instead.


## 🛠️ Contributing

PostBOUND is developed with [uv](https://docs.astral.sh/uv/). To get set up:

```sh
uv sync                     # creates .venv with the package, all extras and the dev tools
uv run pre-commit install   # enables the formatting/linting/type-checking git hooks
```

Formatting (`ruff format`), linting (`ruff check`) and type checking (`ty`) are enforced on every
commit and configured in `pyproject.toml`, so your editor and the hooks always agree.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the details.

## 📑 Repo Structure

The repository is structured as follows.
The `postbound` directory contains the actual source code, all other folders are concerned with "supporting" aspects
(which are nevertheless important..).
Almost all of the subdirectories contain further READMEs that explain their purpose and structure in more detail.

| Folder        | Description |
| ------------- | ----------- |
| `postbound`   | Contains the source code of the PostBOUND framework |
| `docs`        | contains the high-level documentation as well as infrastructure to export the source code documentation |
| `examples`    | contains general examples for typical usage scenarios. These should be run from the root directory, e.g. as `python3 -m examples.example-01-basic-workflow` |
| `tests`       | contains the unit tests and integration tests for the framework implementatino. These should also be run from the root directory, e.g. as `python3 -m unittest tests` |
| `db-support`  | Contains utilities to setup instances of the respective database systems and contain system-specific scripts to import popular benchmarks for them |
| `workloads`   | Contains the raw SQL queries of some popular benchmarks |
| `tools`       | Provides different other utilities that are not directly concerned with specific database systems, but rather with common problems encoutered when benchmarking query optimizers |
