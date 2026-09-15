"""Shared pytest configuration for the PostBOUND test suite.

This module implements the *tier* mechanism described in the testing strategy. Tests are grouped by the
environment they need to run, and a bare ``pytest`` invocation executes tier 0 only:

===== ============================ ==================================================== ================
Tier  Marker                       Environment                                          Budget
===== ============================ ==================================================== ================
0     *(none)*                     pure code plus test doubles -- no server, no network  < 10 s
1     ``embedded``                 DuckDB in-memory or a recorded transcript             < 60 s
2     ``live_db``                  a live database server                                < 5 min
3     ``slow``                     full workload sweeps, real timeouts, prewarming       unbounded
===== ============================ ==================================================== ================

Two properties matter more than the split itself:

1. **Tier 0 touches nothing.** Connectivity is probed lazily, once per connection file, and only when a test
   that actually needs a server is about to run. Collection never opens a connection.
2. **Skips are loud.** A tier that was explicitly requested but skipped wholesale is an error, not a pass.
   Silently reporting success for zero executed tests is the failure mode this suite is moving away from.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator

import pytest

from postbound import db

#: Marker attached by `regression_suite.skip_if_no_db`. Carries the connection file as its single
#: argument so that the availability probe can be deferred out of module import.
REQUIRES_DB_MARKER = "requires_db"

#: Markers that select a tier above 0, in ascending order.
TIER_MARKERS = ("embedded", "live_db", "slow")

#: Marker expression per tier. Tiers are **cumulative**: ``--tier 2`` runs tiers 0, 1 and 2, so a single
#: invocation gives a full regression run up to the chosen cost ceiling. Selecting a tier by raw marker
#: expression instead (``-m live_db``) would also drag in the tier-3 sweeps, because those carry both
#: markers -- which is exactly the mistake this option exists to prevent.
_TIER_EXPRESSIONS = {
    "0": "not embedded and not live_db and not slow",
    "1": "not live_db and not slow",
    "2": "not slow",
    "3": "",
}

#: Memoizes the connectivity probe so that a suite touching several test modules opens at most one
#: throwaway connection per distinct connection file, rather than one per decorated class.
_db_availability: dict[str, bool] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--tier",
        action="store",
        default=None,
        choices=sorted(_TIER_EXPRESSIONS),
        help=(
            "Run every tier up to and including this one. "
            "0 = offline (default), 1 = +embedded engines, 2 = +live database, 3 = +slow workload sweeps."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{REQUIRES_DB_MARKER}(config_file): needs the database described by config_file; skipped when unreachable",
    )

    tier = config.getoption("--tier")
    markexpr = config.getoption("markexpr")
    if tier is not None and markexpr:
        raise pytest.UsageError("--tier and -m are mutually exclusive; use one or the other.")
    if tier is not None:
        config.option.markexpr = _TIER_EXPRESSIONS[tier]
    elif not markexpr:
        # Neither given: default to the offline tier, which is what the pre-push hook enforces.
        config.option.markexpr = _TIER_EXPRESSIONS["0"]


def repo_root() -> pathlib.Path:
    """The repository root, resolved from this file rather than from the working directory.

    The test modules historically built connection paths from ``pg_connect_dir = "."``, which made the whole
    suite CWD-relative: running it from anywhere but the repository root silently skipped every database test
    instead of failing. Path resolution goes through here instead.
    """
    return pathlib.Path(__file__).resolve().parent.parent


def database_available(config_file: str | pathlib.Path) -> bool:
    """Checks whether the database described by `config_file` can be reached.

    The result is cached for the lifetime of the session. Import errors and malformed connection files count
    as "unavailable" rather than propagating, so that a broken local setup skips the database tiers instead of
    aborting collection of the offline ones.
    """
    resolved = pathlib.Path(config_file)
    if not resolved.is_absolute():
        resolved = repo_root() / resolved
    key = str(resolved)

    if key in _db_availability:
        return _db_availability[key]

    available = False
    if resolved.exists():
        try:
            from postbound import postgres

            instance = postgres.connect(config_file=key, private=True)
            instance.close()
            available = True
        except Exception:
            # Deliberately broad: the original helper caught only psycopg.OperationalError, so an auth
            # failure, a malformed file or a missing extension aborted collection of the entire module.
            available = False

    _db_availability[key] = available
    return available


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Applies the deferred database-availability skips.

    Doing this at collection time (rather than at decoration time, as the original `skip_if_no_db` did) means
    a tier-0 run never opens a connection: items carrying the marker are deselected by the marker expression
    before this hook ever probes them.
    """
    for item in items:
        marker = item.get_closest_marker(REQUIRES_DB_MARKER)
        if marker is None or not marker.args:
            continue
        config_file = marker.args[0]
        if not database_available(config_file):
            item.add_marker(pytest.mark.skip(reason=f"database not reachable via '{config_file}'"))


def _requested_tier(config: pytest.Config) -> str | None:
    """The tier the user explicitly asked for, or *None* for a plain offline run.

    A request is what makes an all-skipped run an error: if the user asked for tier 2, a green run that
    executed nothing is a false negative rather than a success.
    """
    tier = config.getoption("--tier")
    if tier is not None:
        return tier if tier != "0" else None

    # Expert mode: a raw -m expression that mentions a tier marker without negating it.
    markexpr: str = config.getoption("markexpr") or ""
    requested = [marker for marker in TIER_MARKERS if marker in markexpr and f"not {marker}" not in markexpr]
    return ", ".join(requested) if requested else None


def _tier_ran_nothing(config: pytest.Config) -> str | None:
    """Returns the requested tier if it was requested but executed no tests, else *None*."""
    requested = _requested_tier(config)
    if requested is None:
        return None

    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is None:
        return None

    stats = reporter.stats
    executed = len(stats.get("passed", [])) + len(stats.get("failed", []))
    return requested if executed == 0 and stats.get("skipped") else None


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    """Lists what was skipped and why, grouped by reason.

    A bare run used to hide roughly 30 silent skips behind a green "OK". Printing the reasons makes the gap
    between "the suite passed" and "the suite ran" visible without having to pass ``-rs``.
    """
    skipped = terminalreporter.stats.get("skipped", [])
    if skipped:
        reasons: dict[str, int] = {}
        for report in skipped:
            reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
            reasons[reason] = reasons.get(reason, 0) + 1
        terminalreporter.write_sep("-", "skipped tests")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            terminalreporter.write_line(f"  {count:3d}  {reason.removeprefix('Skipped: ')}")

    requested = _tier_ran_nothing(config)
    if requested is None:
        return

    terminalreporter.write_sep("=", "tier requested but not executed", red=True)
    terminalreporter.write_line(
        f"Requested tier {requested} but every selected test was skipped. "
        "The environment those tests need is not available, so this run proves nothing."
    )
    terminalreporter.write_line("Provision that environment, or run the offline tier with --tier 0.")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Turns an all-skipped run of an explicitly requested tier into a non-zero exit.

    Selecting a tier is a statement of intent. Reporting success for a run that executed nothing is exactly
    how the previous ``SKIP_ONLINE`` default kept six permanently-dead tests green, so it is made to fail
    here rather than merely warn.
    """
    if exitstatus != pytest.ExitCode.OK:
        return
    if _tier_ran_nothing(session.config) is not None:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(autouse=True)
def isolated_database_pool() -> Iterator[None]:
    """Prevents `DatabasePool` state from leaking between tests.

    The pool is a process-global that nothing resets, so a database registered by one test module stays
    visible to every later one. That is not hypothetical: it currently makes `parser.parse_query` bind columns
    against an arbitrary leaked database in ``test_relalg``, which warns and can silently produce wrong parses.

    The pool contents are snapshotted before each test and restored afterwards, so a test that registers a
    database cannot affect its neighbours regardless of how it terminates.
    """
    pool = db.DatabasePool.get_instance()
    # `DatabasePool` exposes register/retrieve/clear but no way to enumerate its contents, so the snapshot
    # has to reach for the private dict. Worth a small public accessor on the pool eventually.
    snapshot = dict(pool._pool)
    try:
        yield
    finally:
        pool.clear()
        for key, instance in snapshot.items():
            pool.register_database(key, instance)


@pytest.fixture
def fake_db():
    """A `FakeDatabase` registered as the pool's current database, removed again on teardown.

    Use this whenever the code under test reaches for `DatabasePool.get_instance().current_database()` -- the
    parser's column binding, `_pipelines`, `opt.enumeration`, `bench.execute_workload` all do. Registering by
    hand is a hazard because the pool is a process-global with no reset hook; going through the fixture
    guarantees cleanup even when the test fails.

    The returned double is *strict*: an `execute_query` it was not told about raises rather than returning an
    empty result set, so a test cannot accidentally assert on silence.

    Examples
    --------
    >>> def test_something(fake_db):
    ...     fake_db.expect_result("count(*)", [(42,)])
    """
    from tests.doubles import FakeDatabase

    pool = db.DatabasePool.get_instance()
    instance = FakeDatabase()
    pool.register_database("fake", instance)
    try:
        yield instance
    finally:
        pool.clear()


@pytest.fixture
def scripted_cursor():
    """A strict `ScriptedCursor` for driving the generic `DatabaseSchema` implementations."""
    from tests.doubles import ScriptedCursor

    return ScriptedCursor()


@pytest.fixture(scope="session")
def pg_connect_dir() -> pathlib.Path:
    """Directory holding the ``.psycopg_connection_*`` files, independent of the working directory."""
    return repo_root()


@pytest.fixture
def no_column_binding() -> Iterator[None]:
    """Disables the parser's schema-dependent column binding for the duration of a test.

    Parsing itself is pure (it runs on pglast), but resolving *unqualified* column references to their owning
    table needs a schema catalog. Offline tests that do not care about binding turn it off here rather than
    depending on whatever happens to be registered in the pool.
    """
    from postbound import parser

    previous = parser.auto_bind_columns
    parser.auto_bind_columns = False
    try:
        yield
    finally:
        parser.auto_bind_columns = previous


@pytest.fixture
def emulation_fallback_disabled() -> Iterator[None]:
    """Turns off `db.enable_emulation_fallback`, restoring it afterwards.

    The flag is process-global mutable state with no reset hook, so a test that flips it without restoring
    changes the behaviour of every statistics test that follows.
    """
    previous = db.enable_emulation_fallback
    db.enable_emulation_fallback = False
    try:
        yield
    finally:
        db.enable_emulation_fallback = previous


def pytest_report_header(config: pytest.Config) -> list[str]:
    requested = _requested_tier(config)
    header = [f"postbound: test tier {requested or '0 (offline)'}"]
    if os.environ.get("SKIP_ONLINE"):
        header.append(
            "postbound: SKIP_ONLINE is set but no longer read -- select tiers with -m instead (e.g. -m live_db)"
        )
    return header
