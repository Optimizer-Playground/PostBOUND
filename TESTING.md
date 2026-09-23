# Testing strategy

This document is the procedure to follow whenever a module of PostBOUND needs tests — whether it is brand new or has
simply never been covered. It is written so that a human developer and an LLM agent can both follow it end to end;
the intended workflow is a one-line request such as

> Implement tests for `postbound/<module>.py` in accordance with `TESTING.md`.

It describes **what** to test (§4, §5) and **how** to do it (§3, §6, §7). The mechanics of the test runner (tiers,
markers, hooks) are summarized from `CONTRIBUTING.md`, `CLAUDE.md` and `tests/conftest.py`, which remain authoritative
if they ever disagree with this file.

> [!NOTE]
> This document is LLM-generated and primarily intended for the automatic test generation.

---

## 1. Principles

These are non-negotiable. Every rule further down follows from one of them.

1. **A test must be able to fail.** A test that cannot fail is worse than no test, because it reports coverage that
   does not exist. (The JOB/Stats result-set tests once executed the _same_ query on both sides of the comparison and
   were green for years.) Before you finish, prove each test can fail — see §7.
2. **Skips are loud, silence is an error.** Never let a test pass by asserting on empty data or by being skipped
   quietly. Doubles are _strict_ by default and raise on anything they were not told about; requesting a tier whose
   environment is missing fails the run.
3. **Test at the lowest tier that can observe the behaviour.** Pure logic belongs in tier 0 even if it currently lives
   next to I/O code — find or use a seam instead of reaching for a database.
4. **Tier 0 touches nothing.** No server, no network, no file downloads, no process-global state left behind. It is
   enforced by the `pre-push` hook and must stay under **10 seconds in total** (currently ~470 tests in ~1.2 s).
5. **Test against reality, not against our assumptions.** When the input is PostBOUND's own model (qal, plans), build
   it through the real code path (the parser). When the input comes from an external system (EXPLAIN JSON, catalog
   rows), use _verbatim captured_ output, not hand-written approximations.
6. **Doubles subclass the real interfaces.** Never use `unittest.mock` for the `Database` interfaces: `Mock`
   auto-creates attributes, so a renamed abstract method would keep passing silently.
7. **Writing tests does not change production code.** A test commit touches `tests/` (and `CHANGELOG.md`) only. Bugs
   found along the way are pinned, not fixed (§6); seams that are genuinely missing are added in a separate, deliberate
   commit.

---

## 2. The test infrastructure at a glance

### Running tests

All commands run from the repository root and are prefixed with `uv run`.

```sh
uv run pytest                                   # tier 0 (default) -- what the pre-push hook runs
uv run pytest --tier 1                          # + embedded engines / recorded transcripts
uv run pytest --tier 2                          # + a live database server
uv run pytest --tier 3                          # + slow workload sweeps
uv run pytest tests/unit/test_validation.py -v  # one module
uv run pytest -k "predicate and not join"       # by name
uv run pytest --durations=10                    # find slow tests
```

`--tier` and `-m` are mutually exclusive. Prefer `--tier`: tier-3 tests carry more than one marker, so a raw
`-m live_db` would also drag in the slow sweeps.

### Tiers

Tiers are cumulative (`--tier 2` runs 0, 1 and 2). A test's tier is determined by its markers.

| Tier | Marker(s)                                                        | Environment                                        | Budget        |
| ---- | ---------------------------------------------------------------- | -------------------------------------------------- | ------------- |
| 0    | _(none)_                                                         | pure code plus test doubles; no server, no network | < 10 s total  |
| 1    | `@pytest.mark.embedded`                                          | DuckDB in-memory, or a recorded transcript         | < 60 s total  |
| 2    | `@pytest.mark.live_db` + `skip_if_no_db(...)`                    | a live database server                             | < 5 min total |
| 3    | `@pytest.mark.live_db` + `@pytest.mark.slow` (+ `skip_if_no_db`) | full workload sweeps, real timeouts, prewarming    | unbounded     |

`--strict-markers` is on, so a typo in a marker name is an error. A global `timeout = 300` exists only to turn a
deadlock (e.g. in the multiprocessing timeout code) into a failure; no test should come close to it.

### Fixtures (`tests/conftest.py`)

| Fixture                       | Use it when                                                                                                                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `isolated_database_pool`      | _autouse_ — snapshots and restores `DatabasePool` around every test. Nothing to do.                                                                                                                     |
| `fake_db`                     | the code under test calls `DatabasePool.get_instance().current_database()` (parser column binding, pipelines, `opt.enumeration`, `bench`). Registers a strict `FakeDatabase` and removes it afterwards. |
| `scripted_cursor`             | you drive the generic `DatabaseSchema` implementations (information_schema SQL).                                                                                                                        |
| `no_column_binding`           | you parse SQL with _unqualified_ columns and do not care about binding.                                                                                                                                 |
| `emulation_fallback_disabled` | the test depends on `db.enable_emulation_fallback` being off.                                                                                                                                           |
| `pg_connect_dir`              | you need the directory holding `.psycopg_connection_*` files (the repo root, CWD-independent).                                                                                                          |

Any **new process-global flag** that a test must flip gets its own save-and-restore fixture in `conftest.py`, following
`no_column_binding` / `emulation_fallback_disabled`. Never assign a module-level flag directly in a test.

### Doubles (`tests/doubles/`)

| Double            | What it is                                                                                       | Typical use                                                                                                                                                                                                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ScriptedCursor`  | a `Cursor` answering from `{sql_fragment: rows}` and recording every call in `.calls`            | the highest-leverage double: `DatabaseSchema` has **no abstract methods**, so scripting the cursor runs the real information_schema code and lets you assert on the generated SQL (`calls[0].mentions(...)`, `.parameters`). Catches `%s` vs `?` placeholder and parameter-count bugs. |
| `StaticSchema`    | a `DatabaseSchema` from a declarative `{table: [column_spec(...) \| "col"]}` spec                | anything that needs PKs, FKs, indexes, datatypes. Only leaf lookups are overridden; `has_index`, `as_graph`, `join_equivalence_keys`, `lookup_column`, … run for real.                                                                                                                 |
| `FakeStatistics`  | a `StatisticsCatalog` with canned values                                                         | cardinality estimators, cost models.                                                                                                                                                                                                                                                   |
| `FakeHintService` | records `generate_hints(...)` requests in `.requests`, returns the query unchanged               | asserting what a pipeline handed to the hint layer, without a hint dialect.                                                                                                                                                                                                            |
| `FakeOptimizer`   | canned `QueryPlan` / `Cardinality` / `Cost`; raises if asked for a plan it was not given         | code that consults the native optimizer.                                                                                                                                                                                                                                               |
| `FakeDatabase`    | a minimal `Database` composed of the above; `expect_result(fragment, rows)`, `.executed_queries` | anything that just needs a handle. Obtain it through the `fake_db` fixture when it must be the pool's current database.                                                                                                                                                                |

Anything a strict double was not told about raises `UnexpectedQueryError` — that is intentional (principle 2). Pass
`strict=False` only in tests that assert solely on the _generated_ SQL, not on results.

**Do not use doubles** for connection handling, psycopg type adapters, query cancellation, prewarming or transactions.
A double shares whatever mistaken assumption about the server the code makes and can only confirm the bug. Those
behaviours belong in tier 2.

`unittest.mock` remains fine for genuinely incidental patching (`subprocess`, `atexit`, `urllib.request.urlretrieve`,
`time`), never for the `Database` interfaces.

### Helpers for higher tiers (`tests/regression_suite.py`)

- `skip_if_no_db(config_file)` — attaches the `requires_db` marker. The connectivity probe runs at _collection_ time,
  once per connection file, and never for deselected tests, so tier 0 never opens a connection. Pass paths relative to
  the repo root, e.g. `skip_if_no_db(".psycopg_connection_stats")`.
- `DatabaseTestCase.assertResultSetsEqual`, `QueryTestCase.assertQueriesEqual`,
  `PlanTestCase.assertQueryExecutionPlansEqual` — assertion mixins used by the legacy `TestCase` modules. New
  function-style tests may call the underlying module-level helpers or write equivalent plain assertions.

---

## 3. The procedure

Follow these steps in order for the module under test (called _the module_ below).

### Step 1 — Survey before writing anything

1. Read the module in full, plus its entry in the public stub (`__init__.pyi` / `qal/__init__.py`) to learn which
   symbols are public.
2. Find its callers (`grep -rn "<symbol>" postbound/`). Callers tell you which behaviours actually matter and which
   inputs are realistic.
3. Read the existing tests of neighbouring modules (`tests/unit/`) and reuse their fixtures and idioms.
4. Write down, for every public function/class (and every private function holding non-trivial decision logic):
    - its **contract** — inputs, outputs, raised exceptions, documented edge cases (the NumPy docstring is the spec);
    - what it **depends on** — pure data, the parser, the `DatabasePool`, a cursor, the native optimizer, the file
      system, the network, a subprocess, a clock, randomness, a global flag;
    - where the **seams** are — a parameter, an attribute, or a small function through which that dependency enters.

### Step 2 — Assign every behaviour to the lowest possible tier

| The behaviour depends on…                                                                   | Tier | How                                                                                  |
| ------------------------------------------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------ |
| nothing but its inputs (qal, plans, value types, algorithms)                                | 0    | call it directly                                                                     |
| a schema / statistics / optimizer / hint service                                            | 0    | `StaticSchema`, `FakeStatistics`, `FakeOptimizer`, `FakeHintService`, `FakeDatabase` |
| SQL it sends through a cursor                                                               | 0    | `ScriptedCursor`, assert on `.calls`                                                 |
| output of an external system (EXPLAIN JSON, catalog rows)                                   | 0    | verbatim captured fixtures (§5.4)                                                    |
| one live attribute of a live object (e.g. `pg_instance.config["geqo_threshold"]`)           | 0    | a tiny stub class exposing exactly that attribute                                    |
| a pipe / process / event loop, but the _decisions_ are in small functions                   | 0    | drive those functions with a scripted fake (see `test_postgres_timeout.py`)          |
| a real SQL engine, but DuckDB in-memory suffices                                            | 1    | `@pytest.mark.embedded`                                                              |
| a real server: connections, type adapters, cancellation, transactions, hint _effectiveness_ | 2    | `@pytest.mark.live_db` + `skip_if_no_db(...)`                                        |
| an entire workload, real timeouts, prewarming                                               | 3    | additionally `@pytest.mark.slow`                                                     |

Most modules end up almost entirely in tier 0 with a handful of tier-2 tests confirming that the real backend agrees.
If a behaviour seems to need a server only because its logic is tangled with I/O, look harder for a seam before
settling for tier 2 — and if there truly is none, note it in the test module docstring as a candidate refactoring.

### Step 3 — Decide what to test

Work through the generic checklist in §4, then the category-specific list(s) in §5 that apply to the module. Prioritize
in this order:

1. decision logic with a large blast radius that has no tests yet;
2. error paths and edge cases (they are the least exercised in practice and break most often);
3. the documented contract of each public symbol;
4. invariants shared by a whole class hierarchy (hashing, immutability, serialization, visitors).

### Step 4 — Build inputs

- **qal objects**: parse real SQL with `parser.parse_query(...)`. Use explicitly qualified columns (`r.a`, not `a`) so
  the parser can bind them without a schema, or pass `bind_columns=False`, or request `no_column_binding`. Define
  frequently used queries as module-level constants (`IMPLICIT_JOIN = parse("SELECT * FROM r, s WHERE r.a = s.b")`).
  Short abstract tables (`r`, `s`, `t`) are preferred over workload tables for tier 0.
- **Value types** (`TableReference`, `ColumnReference`, `Cardinality`, operators): construct directly.
- **Plans**: build through the public constructors or parse captured EXPLAIN output; keep them small.
- **External output**: see §5.4 — capture once, paste verbatim, cite the producing query.
- **Synthetic inputs** only where reality cannot provide them, and say why in a comment — e.g. a minimal node to test
  defaults for _absent_ fields, or a deliberately corrupted copy of a real node for an error path a correct DBMS never
  produces.

### Step 5 — Write the tests

See §6 for layout and style. In short: one plain-function test per behaviour, named as a sentence, grouped into
sections, in a new module under `tests/unit/` (tier 0) or the area's `tests/test_<area>.py` (tier 1+).

### Step 6 — Handle bugs you find

See §6.5. Pin, record, do not fix. Once a bug _is_ fixed, it gets a dedicated regression test (§6.6).

### Step 7 — Verify and finish

See §7.

---

## 4. Generic checklist — applies to every module

For each public function / method:

- [ ] The **typical case** from the docstring or from the most common caller.
- [ ] **Boundaries**: empty inputs, a single element, `None` where the signature allows it, the largest structure the
      callers realistically produce.
- [ ] **Every documented `Raises`** with `pytest.raises(ExcType, match="…")` — matching on a message fragment, so the
      _right_ failure is asserted.
- [ ] **Every branch of non-trivial decision logic** (one test per branch, or one `parametrize` table).
- [ ] **Keyword options and flags**, each at least once in its non-default setting.

For PostBOUND-specific contracts, where the module defines such objects:

- [ ] **Equality and hashing**: equal objects compare equal _and_ hash equal; objects differing in any
      identity-relevant field compare unequal. Where the class precomputes `_hash_val`, check that `hash(obj)` is
      stable and consistent with `==`.
- [ ] **Immutability**: transformations return a _new_ object and leave the original unchanged (assert on the
      original after the call).
- [ ] **Serialization**: `util.jsonize.to_json(obj)` succeeds and contains the fields that matter; if a loader exists,
      `load(to_json(obj)) == obj` round-trips.
- [ ] **`describe()`** of optimization stages returns a JSON-serializable dict that identifies the configuration
      (different parameters → different descriptions).
- [ ] **Visitors**: `accept_visitor` dispatches to the correct `visit_*` method for each concrete subclass (a small
      recording visitor is enough).
- [ ] **Pattern matching**: if the class defines `__match_args__`, a positional `case Cls(a, b, …)` actually matches
      (this has silently failed before — see `test_qal_clauses.py`).
- [ ] **Public exposure**: at least one test imports each new public symbol through its _public_ path
      (`from postbound import X`, `pb.opt.X`, `from postbound.qal import X`). Because of lazy loading via `.pyi` stubs,
      a symbol missing from the stub's imports or `__all__` is otherwise invisible. Internals may be imported from
      their private module.

What **not** to test:

- the DBMS itself, or third-party libraries (pglast, networkx, psycopg) beyond how the module uses them;
- exact formatting of output that is not part of the contract — prefer round trips and fragment checks;
- iteration order of sets/frozensets — it is not part of their contract and string hashing is salted per process;
- trivial attribute accessors with no logic.

---

## 5. Category-specific guidance

Pick every category that applies to the module.

### 5.1 qal, relalg and other immutable value objects

- Build inputs through the parser (§3 step 4). The parser is the normal, correct way to construct qal; there is no
  "external shape" risk.
- Cover each concrete subclass of a hierarchy at least once, including the rarely used ones (window, case, quantifier,
  values/function table sources, set operations, CTEs, EXPLAIN, hints).
- Test aggregate queries (`tables()`, `columns()`, `joins()`, `filters()`, `subqueries()`, `bound_tables()`,
  `is_dependent()`, …) over the query shapes that tend to break them: explicit `JOIN` syntax, cross products, queries
  with no predicates at all, dependent vs. independent subqueries, set queries.
- For formatters and transformations prefer **round trips**: `parse(format(q)) == q`, or "transform, then assert on the
  properties of the result" rather than on its string.
- Model: `tests/unit/test_qal_*.py`.

### 5.2 Optimization stages and algorithms (`opt/`, `_stages.py` subclasses, `_pipelines.py`)

- Test on tiny, fully known inputs: 1, 2 and 3–4 table queries, a cross product, a star and a chain, where the correct
  answer can be derived by hand.
- Feed estimates via `FakeStatistics` / `FakeOptimizer` / `StaticSchema`, so results are deterministic.
- Assert **structural properties** of the output (every table appears exactly once in the join tree, operators are
  assigned only to existing intermediates, the plan covers the query) as well as the exact expected result on the hand
  -computable cases.
- `Cardinality` models NaN (unknown) and infinity: check how the stage handles both, and that it never emits invalid
  estimates unless documented.
- `pre_check()` / required hints: queries the stage claims to support pass; unsupported shapes are rejected with a
  meaningful `failure_reason`.
- Training hooks (`fit_database`, `fit_workload`, `fit_samples`, `learn_from_feedback`): the declared needs match what
  the stage actually uses, and the stage behaves sensibly (or raises clearly) when used untrained.
- Randomized algorithms: seed them and assert reproducibility for the same seed; assert properties, not a particular
  random outcome.
- Pipelines: use `FakeHintService` and assert on `.requests` to check what was handed to the hint layer; check that
  `build()` rejects incompatible configurations.
- Anything reaching for `DatabasePool` → use the `fake_db` fixture.

### 5.3 Database backends (`db/`, `postgres/`, `duckdb/`, `mysql.py`)

- **Tier 0**: everything that generates SQL or interprets rows. Drive it with `ScriptedCursor` / `FakeDatabase` and
  assert on the generated SQL (`calls[i].mentions(...)`), on the number and order of bound parameters, and on the
  placeholder style of the dialect. Model: `tests/unit/test_database_schema.py`.
- Check that every concrete backend class is **instantiable** (i.e. implements all abstract methods, including
  `__eq__`/`__hash__`); this has regressed after renames before. Where instantiation needs a connection, a tier-1/2
  smoke test suffices.
- **Tier 1** (DuckDB in-memory): the same behaviours against a real embedded engine, with a small synthetic schema
  created in the test.
- **Tier 2**: connection handling, type adapters, cancellation/timeouts, transactions, and that generated hints are
  _accepted and effective_ on the real server (e.g. the plan shape actually changes).
- Hint generation logic itself (pg_hint_plan / pg_lab / quacklab text) is pure: tier 0, with a minimal stub for any
  single live lookup. Hint fragments built from sets must be asserted order-independently (see
  `assert_any_ordering` in `test_postgres_hints.py`). Model: `tests/unit/test_postgres_hints.py`.

### 5.4 Parsers of external output (EXPLAIN plans, catalog dumps, workload files)

- Capture the real output **once** against a real system, paste it **verbatim** as a Python literal, and put the exact
  producing statement (and the workload/DB it ran against) in a comment above the fixture, so it can be recaptured.
- Nothing is executed at test time — the tests stay tier 0.
- Cover each node/record kind the module distinguishes, the parent/child ordering conventions of the source system
  (e.g. Postgres lists _Outer_ before _Inner_), and derived values (e.g. parallel-worker cardinality scaling).
- Synthetic input is allowed only for absent-field defaults and impossible-in-reality error paths, and is labelled as
  such. Model: `tests/unit/test_postgres_explain.py`.

### 5.5 State machines, concurrency, timeouts

- Do not test through the process/thread boundary in tier 0. Identify the small functions that decide _what to do next
  for each event_, and drive them with a scripted fake of the channel (pipe, queue) and a recording fake of the
  side-effecting collaborator.
- Cover **every transition**, especially the error and timeout paths that a live test almost never reaches.
- Keep one tier-2/3 test that exercises the real process end to end. Model: `tests/unit/test_postgres_timeout.py`.

### 5.6 Validation / pre-checks

- Each check is a pure `SqlQuery -> PreCheckResult` function: a table of passing queries and a table of failing
  queries, asserting both `passed` and the `failure_reason`. Include compound checks and merging. Model:
  `tests/unit/test_validation.py`.

### 5.7 Workloads, benchmarking, file and network I/O (`workloads.py`, `bench.py`, `util/`)

- Use `tmp_path` for anything that reads or writes files; never write into the repo or `$HOME/.postbound/` from a test.
- Patch network downloads (`urllib.request.urlretrieve`) with `unittest.mock` / `monkeypatch` in tier 0; a real
  download is at least tier 2.
- `bench.execute_workload`: run against `fake_db` with scripted results and assert on the result frame's columns and
  rows; real execution belongs in tier 2/3.

### 5.8 Visualization (`vis/`)

- Test only data preparation (the structures handed to matplotlib/graphviz) in tier 0; smoke-test that plotting
  functions run without error using a non-interactive backend. Do not compare images.

---

## 6. Layout, style and conventions

### 6.1 Where tests go

- **Tier-0 tests** for `postbound/<pkg>/<module>.py` go in a new module `tests/unit/test_<pkg>_<module>.py` (drop the
  leading underscore of private modules; e.g. `postgres/_explain.py` → `test_postgres_explain.py`). Split very large
  modules by structural layer, as `test_qal_{expressions,predicates,clauses,query,formatter}.py` do.
- **Tier-1+ tests** go in the area's top-level module `tests/test_<area>.py` (e.g. `tests/test_postgres.py`), or a new
  one if none exists.
- Reusable doubles go in `tests/doubles/` (subclass the real ABC, export via `tests/doubles/__init__.py`, give them
  NumPy docstrings). Reusable fixtures go in `tests/conftest.py`. Single-use helpers stay in the test module.
- Legacy `unittest.TestCase` classes in `tests/test_*.py` are left as they are; do not add new `TestCase` classes.

### 6.2 Module skeleton

```python
"""Tests for `postbound.<pkg>.<module>` -- <one-line description of what it does>.

<Why the tests are shaped the way they are: which tier, where the inputs come from (parsed SQL, captured
output, doubles), which seam makes the module testable offline.>

<What is deliberately NOT covered here and where it is covered instead (e.g. "connection handling: see the
live tests in tests/test_postgres.py"), plus any known quirks the tests route around.>
"""

from __future__ import annotations

import pytest

from postbound import parser
from postbound.<pkg> import <PublicSymbol>          # public path for public API
from postbound.<pkg>._<module> import _helper        # private path only for internals
from tests.doubles import FakeDatabase, StaticSchema

# -- fixtures -----------------------------------------------------------------------------------------------


def parse(sql: str):
    return parser.parse_query(sql, bind_columns=False)


CHAIN_JOIN = parse("SELECT * FROM r, s, t WHERE r.a = s.b AND s.c = t.d")


# -- <SymbolOrBehaviourGroup> --------------------------------------------------------------------------------


def test_<subject>_<expected_behaviour>() -> None:
    query = CHAIN_JOIN

    result = <PublicSymbol>(query)

    assert result.tables() == {...}


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT * FROM r", False),
        ("SELECT * FROM r, s WHERE r.a = s.b", True),
    ],
    ids=["single-table", "equi-join"],
)
def test_<subject>_<behaviour>_for_each_query_shape(sql: str, expected: bool) -> None:
    assert <PublicSymbol>(parse(sql)).<property> is expected


def test_<subject>_rejects_<bad_input>() -> None:
    with pytest.raises(ValueError, match="<message fragment>"):
        <PublicSymbol>(<bad input>)


# -- regression tests --------------------------------------------------------------------------------------


def test_<subject>_<previously_broken_behaviour>() -> None:
    """Regression guard for <commit hash or issue>: <what used to go wrong and why>."""
    ...
```

### 6.3 Style rules

- **Plain functions**, not `TestCase` classes — fixtures and `parametrize` do not work inside `TestCase`.
- **Names are sentences** describing the behaviour: `test_merge_collects_every_failure_reason`,
  `test_cross_product_check_rejects_a_disconnected_join_graph`. Someone reading only the failure report should know
  what broke.
- **One behaviour per test**; arrange / act / assert separated by blank lines. Multiple asserts are fine when they
  describe one behaviour.
- **`parametrize` with `ids`** for tables of similar cases instead of loops or `subTest`.
- Every test is annotated `-> None`; test modules pass `ruff` and `ty` like production code.
- Group tests with `# -- <Section> ---…` comment rules, in the order the symbols appear in the module.
- A test docstring is expected whenever the _why_ is not obvious from the name: a non-obvious expectation, a quirk of
  the system under test, a regression being guarded, a bug being pinned.
- Assert on specific values, not truthiness: `assert result == {R, S}`, not `assert result`.
- Order-independent comparisons for anything derived from sets.
- No sleeps, no dependence on wall-clock time, no dependence on test execution order, no randomness without a seed.

### 6.4 Global state

`DatabasePool` is reset automatically. For everything else that is process-global (`parser.auto_bind_columns`,
`db.enable_emulation_fallback`, logging configuration, module-level caches) use or add a restore fixture, or pytest's
`monkeypatch`, which restores automatically. A test must leave the process exactly as it found it, regardless of
whether it passes.

### 6.5 Bugs found while writing tests

Writing tests routinely surfaces production bugs (the first offline batch found seven). Handle them like this:

1. **Do not fix them in the test commit.** Keep the test change and the fix separately reviewable.
2. **Pin the current behaviour** with a dedicated test whose docstring starts with
   `Documents a real bug, not the intended behaviour.` and then explains the root cause (the call chain, not just the
   symptom) and what the correct behaviour would be. The test asserts on what the code _does today_ (including
   `pytest.raises` for a crash), so the fix later becomes a deliberate, visible test change rather than a silent one.
3. **Record it** under `## 🪲 Known bugs` in the unreleased section of `CHANGELOG.md`.
4. **Report it** in the commit message / final summary.
5. When the bug is fixed (separate commit): replace the pinned test with a regression test as described in §6.6, and
   move the CHANGELOG entry from _Known bugs_ to _Fixes_.

### 6.6 Regression tests

Every fixed bug gets a **dedicated regression test function** — a test that passes on the fixed code and **fails if the
bug is ever re-introduced**. This applies equally to bugs fixed on their own and to bugs that were pinned by §6.5.

- **One function per regression.** Do not fold the check into an existing, more general test, and do not rely on the
  bug being "covered somewhere". Name it after the behaviour that used to break, e.g.
  `test_sql_query_bound_tables_includes_tables_from_an_explicit_join`, so a failure report points straight at the
  regression.
- **It asserts the correct behaviour**, not the old wrong one, using the smallest input that triggered the bug.
- **Prove it catches the regression**: run it against the unfixed code (e.g. `git stash` the fix, or temporarily revert
  the relevant lines) and watch it fail, then confirm it passes with the fix. A regression test that passes on both is
  not a regression test.
- **Docstring** starts with `Regression guard for <commit hash or issue>:` and explains what used to go wrong and why
  (the root cause, not just the symptom).
- **Lowest tier** that reproduces the bug, so it is cheap enough to run on every push. A bug originally observed
  against a live server is usually reproducible at tier 0 with a double or a captured fixture.
- **Placement**: in the test module of the component that was broken, in a final `# -- regression tests --` section.
  Always a plain function — also in legacy modules that still contain `TestCase` classes; do not add new methods to
  the legacy `RegressionTests` classes.

---

## 7. Definition of done

Before declaring the work finished, all of the following hold:

1. **Each test can fail.** For every test (or at least every non-trivial one), temporarily break the behaviour it
   checks — invert a condition in the code under test, return a wrong value, or corrupt the expected value — and watch
   it fail for the right reason. Revert afterwards. A test that stays green is deleted or fixed.
2. **The module's tests pass in isolation** and in any order:
   `uv run pytest tests/unit/test_<module>.py -v`.
3. **The full tier 0 passes and stays fast**: `uv run pytest` is green, the total remains well under 10 s, and
   `uv run pytest --durations=10` shows no new test above ~100 ms. A new module should add well under a second.
4. **Higher tiers**, if you added tests there, pass against a provisioned environment
   (`uv run pytest --tier 2`, `--tier 3`). If the environment is unavailable, say so explicitly — do not claim they
   pass.
5. **Tooling is clean**: `uv run ruff format .`, `uv run ruff check --fix .`, `uv run ty check`
   (or `uv run pre-commit run --all-files`).
6. **No production code changed** (`git diff --stat -- postbound/` is empty), unless a seam was added in a separate,
   deliberate commit.
7. **Known bugs** found are pinned (§6.5) and listed in `CHANGELOG.md`; every **fixed** bug has a dedicated regression
   test that was shown to fail without the fix (§6.6).
8. **Summary / commit message** states: which surfaces are now covered, which are deliberately not (and why), which
   tier the tests live in, any bugs found, and the tier-0 test count and runtime before → after.

---

## 8. Notes for LLM agents

When asked to "implement tests for `<module>` in accordance with `TESTING.md`":

- Follow §3 step by step. Do the survey (step 1) with actual file reads and greps; do not guess signatures or
  behaviour from names.
- Treat the docstrings as the specification, but trust the code for what actually happens. Where they disagree, that
  is a finding: pin it per §6.5 and report it.
- Do not modify anything under `postbound/`. If the module is untestable at tier 0 without a seam, write the tests you
  can, and describe the missing seam in the module docstring and in your summary.
- Do not add dependencies, plugins or new pytest options.
- Reuse the doubles and fixtures from §2 before writing new ones; if you write a reusable double, put it in
  `tests/doubles/` and subclass the real interface.
- Run the checks in §7 yourself and report their actual output. Never report a tier as passing if it was skipped.
- Do not commit unless asked.
