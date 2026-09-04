# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PostBOUND is a research framework for prototyping and benchmarking **query optimizers** of relational database systems.
It is a pure Python package (`postbound/`) that runs *on top of* a live database server (Postgres, DuckDB, limited MySQL) —
it never patches the DBMS. Optimization decisions are enforced by translating an abstract query plan into system-specific
**query hints** (pg_hint_plan / pg_lab for Postgres, quacklab for DuckDB).

## Commands

The project is managed with `uv` (see `uv.lock`, requires Python >= 3.12). Prefix everything with `uv run` — the
system Python does not have the dependencies. All commands are run **from the repository root**.

```sh
uv sync                                     # install/refresh the dev environment (incl. all extras)
uv run pre-commit install                   # enable the git hooks -- do this once after cloning
uv run python -m unittest discover -s tests -t .        # full test suite
uv run python -m unittest tests.test_qal -v             # single module
uv run python -m unittest tests.test_qal.SomeTest.test_x # single test
uv run python -m examples.example-01-basic-workflow      # examples are modules, not scripts
uv run python -m tools.ceb-generator --help              # same for tools/*.py
```

Formatting, linting and type checking (all enforced by the pre-commit hooks):

```sh
uv run ruff format .                        # formatter, line-length 120
uv run ruff check --fix .                   # linter, rules E,W,F,I,UP,B,SIM,RUF
uv run ty check                             # type checker
uv run pre-commit run --all-files           # everything the hooks would run
```

Docs (Sphinx, published to readthedocs):

```sh
cd docs && uv run --group doc sphinx-build -M html source build
```

### Tests and database connections

Most integration tests are guarded by `tests/regression_suite.skip_if_no_db(<config_file>)`, which skips the test if the
connection file is missing or the server is unreachable — so a bare run silently skips ~30 tests. Postgres connection
files live in the repo root as `.psycopg_connection_<workload>` (e.g. `.psycopg_connection_job`,
`.psycopg_connection_stats`); `tools/set-workload.sh` switches the active one. Two extra opt-in env vars gate the
slowest checks: `COMPARE_RESULT_SETS` and `CHECK_JOB_SANITY`.

Workload queries are *not* in the repo — `postbound/workloads.py` downloads them on first use into `$HOME/.postbound/`.

Database instances can be provisioned with the shell scripts in `db-support/<system>/` (`postgres-setup.sh`,
`workload-job-setup.sh`, …), or via the Dockerfile (see the README's Docker options table).

## Architecture

### The optimization flow

```
SqlQuery  --(pipeline: optimization stages)-->  QueryPlan  --(HintService)-->  hinted SqlQuery  --> real DBMS
```

- `postbound/_pipelines.py` — four `OptimizationPipeline` flavours, each a different *mental model* of an optimizer:
  `TextBookOptimizationPipeline` (enumerator + cost model + cardinality estimator), `MultiStageOptimizationPipeline`
  (join order → operator selection → plan parameters), `IntegratedOptimizationPipeline` (one algorithm computes
  everything), `IncrementalOptimizationPipeline` (successive plan rewrites). Pipelines are configured by chained
  `use(...)`/`setup_*(...)` calls followed by `build()`, which runs compatibility pre-checks.
- `postbound/_stages.py` — the user-facing extension points (`JoinOrderOptimization`, `PhysicalOperatorSelection`,
  `CardinalityEstimator`, `CostModel`, `PlanEnumerator`, `ParameterGeneration`, `IncrementalOptimizationStep`,
  `CompleteOptimizationAlgorithm`). Every stage also declares its training needs (`fit_database`, `fit_workload`,
  `fit_samples`, `learn_from_feedback`) — the pipeline and `bench` drive these automatically.
- `postbound/_qep.py` — `QueryPlan`, the central plan representation (physical operators, estimates, measurements).
- `postbound/_hints.py` — the *partial* decision containers handed to the hint layer: `JoinTree`/`LogicalJoinTree`,
  `PhysicalOperatorAssignment`, `PlanParameterization`. Anything left unset is delegated to the native optimizer.
- `postbound/_core.py` — foundational value types shared everywhere: `TableReference`, `ColumnReference`, `Cardinality`
  (an integer wrapper that also models NaN = unknown and inf), `Cost`, and the physical operator enums.
- `postbound/validation.py` — `OptimizationPreCheck`s that reject query/strategy/backend combinations up front.

### Database backends

`postbound/db/_db.py` defines the abstract `Database` and its delegated interfaces: `DatabaseSchema`,
`DatabaseStatistics`, `OptimizerInterface`, and `HintService`. `HintService.generate_hints()` is the bridge from
PostBOUND's abstractions back to executable SQL. Concrete backends are top-level modules: `postgres.py` (by far the most
complete — two hint dialects, `PostgresExplainPlan` parsing, timeout/parallel executors, `WorkloadShifter`),
`duckdb.py`, `mysql.py`. `DatabasePool` holds the "current" database so most APIs can default to it.

### Query abstraction layer (qal)

`postbound/qal/` is the SQL model: expressions → predicates → clauses → `SqlQuery` (plus `ImplicitSqlQuery`,
`ExplicitSqlQuery`, `MixedSqlQuery`, `SetQuery`). Everything is **immutable**; there is no in-place mutation anywhere.

- `postbound/parser.py` — string/JSON → qal, built on `pglast` (the real Postgres parser). It also binds columns to
  tables; the second, schema-dependent binding phase needs a DB connection and is controlled by the module-level
  `auto_bind_columns` flag.
- `postbound/transform.py` — the only sanctioned way to "modify" a query: functions that build new query objects.
- `postbound/relalg.py` — relational-algebra view (`RelNode` trees/DAGs) converted from a parsed query.
- `postbound/qal/_formatter.py` — pretty-printing back to SQL.

### Everything else

- `postbound/bench.py` — `execute_workload`, `QueryPreparation`, result frames; drives pipelines reproducibly.
- `postbound/workloads.py` — `Workload`, the generic `read_workload()` reader, and ready-made loaders `job()`,
  `job_light()`, `job_complex()`, `stats()`, `stack()`, `ssb()`.
- `postbound/opt/` — ready-made algorithms and helpers: `dynprog.py`, `enumeration.py`, `randomized.py`, `native.py`,
  `noopt.py`, plan/JSON helpers (`_helpers.py`), cardinality wrappers (`_cardinalities.py`). The `JoinGraph` abstraction
  was removed; `README.md` and `docs/source/` still reference it.
  `postbound/experiments/` has already been removed; `tools/ceb-generator.py` and `tools/query-generator.py` still
  import it and are therefore broken.
- `postbound/util/` — generic helpers (collections, dicts, `jsonize`, logging, networkx, stats).
- `postbound/train/` — training data plumbing for learned stages. `postbound/vis/` — plotting/graphviz (extra `vis`).

## Conventions

**Private modules, public packages.** Implementation lives in underscore-prefixed modules (`_qal.py`, `_core.py`,
`_pipelines.py`, `_db.py`, …). The public surface is assembled in `__init__.py`/`__init__.pyi`.

**Lazy loading via stub files.** `postbound/__init__.py`, `postbound/opt/__init__.py` and `postbound/util/__init__.py`
use `lazy_loader.attach_stub`, so the real export list lives in the adjacent `__init__.pyi`. **A new public symbol is
invisible until it is added to both the `.pyi` stub's imports and its `__all__`.** For `qal`, exports go through the
explicit imports in `qal/__init__.py`.

**Immutability + precomputed hashes.** qal, `relalg`, and plan objects are immutable data objects that use `__slots__`
and compute `self._hash_val` in `__init__`, returning it from `__hash__`. Preserve this pattern — the optimizer hot
loops depend on it. Never mutate a qal object; construct a new one (see `transform`).

**Visitors.** Traversal goes through `accept_visitor` and the `SqlExpressionVisitor` / `PredicateVisitor` /
`ClauseVisitor` / `RelNodeVisitor` hierarchies rather than isinstance chains.

**Serialization.** Objects expose `__json__()` and are dumped with `util.jsonize.to_json`. Optimization stages
additionally implement `describe() -> jsondict` so benchmark output records the exact configuration used.

**Tooling is uv-only.** `ruff` and `ty` are pinned exactly in the `dev` dependency group, and the pre-commit hooks are
local hooks that shell out to `uv run`, so `uv.lock` is the single source of truth for tool versions. Rules that are
deliberately not enforced are listed with their reasoning in `[tool.ruff.lint].ignore`; type-check suppressions live in
`[[tool.ty.overrides]]`, each with a comment saying what would let the override be deleted. `unresolved-import` and
`unresolved-attribute` are enforced everywhere without exception. See `CONTRIBUTING.md`.

**Docstrings** are NumPy-style (`Parameters` / `Returns` / `Raises` / `See Also` / `References` sections) and are the
source for the Sphinx API docs — public API additions are expected to carry them. Lines wrap at 120 columns (enforced by
`ruff format`; `E501` is not enforced, so pre-existing long docstring prose still exceeds it).

**CHANGELOG.md** is maintained by hand per release with fixed sections (New features / Updates / Fixes / Deprecations /
Known bugs); older entries are archived in `HISTORY.md`. Deprecations are announced there with the version that will
remove them, and the version in `pyproject.toml` is the single source of truth.
