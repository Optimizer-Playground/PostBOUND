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
uv run python -m unittest discover -s tests -t .          # the full test suite
uv run python -m unittest tests.test_qal -v               # a single module
uv run python -m examples.example-01-basic-workflow       # examples are modules, not scripts
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

Tests use `unittest`, not pytest. Many of them talk to a live database and are guarded by
`tests/regression_suite.skip_if_no_db(<config_file>)`, which skips the test when the connection file is
missing or the server is unreachable — so a bare run silently skips around 20 tests.

Postgres connection files live in the repository root as `.psycopg_connection_<workload>` (for example
`.psycopg_connection_job`, `.psycopg_connection_stats`); `tools/set-workload.sh` switches the active
one. Two extra environment variables gate the slowest checks: `COMPARE_RESULT_SETS` and
`CHECK_JOB_SANITY`.

Workload queries are not stored in the repository — `postbound/workloads.py` downloads them on first
use into `$HOME/.postbound/`. Database instances can be provisioned with the shell scripts in
`db-support/<system>/`, or via the Dockerfile (see the README's Docker options table).

**Please run the suite with a database configured before opening a pull request**, since a bare run
does not exercise any of the backend code.

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
