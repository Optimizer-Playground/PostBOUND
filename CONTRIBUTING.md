# Contributing to PostBOUND

## Development setup

PostBOUND is managed with [uv](https://docs.astral.sh/uv/). It is required — the pinned tool
versions and the pinned Python interpreter both come from `uv.lock` and `.python-version`, which is
what keeps results identical across machines and editors.

```sh
uv sync                     # creates .venv, installs the package, all extras and the dev tools
uv run pre-commit install   # enables the git hooks (pre-commit and pre-push)
```

That is the whole setup. `uv sync` pulls in the optional backends (`vis`, `duckdb`, `mysql`) as well,
because without them the type checker cannot resolve the imports in `postbound/vis/`, `duckdb.py` and
`mysql.py`, and results would differ between developers.

Prefix everything with `uv run` — the system Python does not have the dependencies:

```sh
uv run pytest                                            # the offline test suite (tier 0)
uv run pytest --tier 2                                   # ... plus the tests needing a live database
uv run pytest tests/test_qal.py -v                        # a single module
```

## Code style

Formatting, linting and type checking are enforced by the git hooks. All three are configured in
`pyproject.toml`, so your editor, the hooks and a manual run always agree:

| Tool | What it owns | Manual invocation |
| --- | --- | --- |
| `ruff format` | formatting, line length 120 | `uv run ruff format .` |
| `ruff check` | lint rules `E,W,F,I,UP,B,SIM,RUF` | `uv run ruff check --fix .` |
| `ty` | type checking | `uv run ty check` |

Rules that are deliberately *not* enforced are listed with their reasoning in the `ignore` array of
`[tool.ruff.lint]`. Files with type-check suppressions are listed under `[[tool.ty.overrides]]`, each
with a comment saying what would allow the override to be deleted. Prefer fixing an issue over adding
to either list; if you must suppress, suppress the narrowest thing at the narrowest scope and say why.

`unresolved-import` and `unresolved-attribute` are enforced everywhere with no exceptions — those are
the diagnostics that catch a module going stale after a refactoring.

### Editors

`.editorconfig` covers indentation and line endings for any editor. Shared workspace settings are
checked in for VS Code (`.vscode/`) and Zed (`.zed/`); both are configured to format with ruff on save
and to use the project's `.venv`. VS Code users should accept the recommended extensions.

Everything else under `.vscode/` and `.zed/` is gitignored, so your personal launch configurations and
tasks stay local.

### Bulk reformats and `git blame`

Purely mechanical reformats are recorded in `.git-blame-ignore-revs`. GitHub applies this
automatically; for local `git blame`, opt in once:

```sh
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

## Tests

The testing strategy — what to test for a new module, which tier it belongs in, which doubles to use, and when the
work counts as done — is described in [TESTING.md](TESTING.md). This section only covers the mechanics.

Tests run under `pytest`. Existing `unittest.TestCase` classes are collected unchanged, but **new tests
should be written as plain functions**, because pytest fixtures and `@pytest.mark.parametrize` do not work
inside a `TestCase` subclass.

Tests are grouped into **tiers** by the environment they need. Tiers are cumulative, so `--tier 2` runs
tiers 0, 1 and 2:

```sh
uv run pytest                 # tier 0 (default) — pure code and test doubles; no server, no network
uv run pytest --tier 1        # + embedded engines (DuckDB in-memory) and recorded transcripts
uv run pytest --tier 2        # + a live database server
uv run pytest --tier 3        # + slow workload sweeps, real timeouts, prewarming
uv run pytest tests/test_qal.py -v          # a single module
uv run pytest -k "predicate and not join"   # by name
```

**Tier 0 is the only tier enforced automatically**, as a `pre-push` hook alongside `ruff` and `ty`. It must
stay fast (budget: under 10 seconds) and must never touch a database or the network. The higher tiers are
run deliberately, and should be run before opening a pull request that touches a database backend.

Requesting a tier whose environment is unavailable is an **error**, not a silent pass: if every selected
test is skipped, the run fails and says so. Every run also prints a summary of what it skipped and why.

Postgres connection files live in the repository root as `.psycopg_connection_<workload>` (for example
`.psycopg_connection_job`, `.psycopg_connection_stats`); `tools/set-workload.sh` switches the active one.
Tests needing one declare it with `tests/regression_suite.skip_if_no_db(<config_file>)`, which defers the
connectivity probe to collection time — so an offline run never opens a connection. Paths are resolved
against the repository root, not the working directory.

Workload queries are not stored in the repository — `postbound/workloads.py` downloads them on first
use into `$HOME/.postbound/`. Database instances can be provisioned with the shell scripts in
`db-support/<system>/`, or via the Dockerfile (see the README's Docker options table).

Every fixed bug gets a **dedicated regression test function** that fails if the bug is re-introduced. Put it
in a final `# -- regression tests --` section of the relevant test module, with a docstring naming the commit
or issue, and write it at the lowest tier that reproduces the bug so it stays cheap enough to run constantly.
See [TESTING.md](TESTING.md#66-regression-tests) for details.

## Documentation

```sh
cd docs && uv run --group doc sphinx-build -M html source build
```

Docstrings are NumPy-style (`Parameters` / `Returns` / `Raises` / `See Also` / `References`) and are the
source for the Sphinx API docs, so public API additions are expected to carry them.

## Conventions

These are load-bearing; see `CLAUDE.md` for the longer version.

- **Private modules, public packages.** Implementation lives in underscore-prefixed modules; the public
  surface is assembled in `__init__.py` / `__init__.pyi`. A new public symbol is invisible until it is
  added to both the stub's imports and its `__all__`.
- **Immutability.** qal, `relalg` and plan objects are immutable, use `__slots__`, and precompute
  `self._hash_val`. Never mutate a qal object — construct a new one via `postbound/transform.py`.
- **Visitors.** Traverse via `accept_visitor` and the visitor hierarchies rather than isinstance chains.
- **CHANGELOG.md** is maintained by hand per release. Deprecations are announced there together with the
  version that will remove them.
