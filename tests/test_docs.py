"""Tests for the code examples of the high-level documentation in `docs/source/`.

The documentation shows its examples as static IPython sessions (``.. code-block:: ipython``) with captured output,
so that the Read the Docs build does not need to execute anything. The price is that nothing keeps those sessions in
sync with the code. These tests close the gap: they extract every session from the RST sources, replay each document
in a single namespace (just like the session in the docs) and check that every statement still runs and still
produces the output that the docs show.

Blocks are assigned to tiers individually. Everything before the first block that talks to a database runs in tier 0;
that block and all later ones of the same document run in tier 2 against the Stats database. The docs connect via a
``.psycopg_connection`` file in the working directory, so the live replay runs in a temporary directory that holds a
copy of ``.psycopg_connection_stats`` under that name.

Some outputs legitimately differ between environments (server versions, plan estimates, set iteration order). Those
statements are listed in `UNSTABLE_OUTPUTS` together with the reason; for them, the test only checks that they run and
produce the same *kind* of output (printed text and/or a displayed result) as shown in the docs.

Both tiers live in this module rather than being split between `tests/unit/` and `tests/`, since they share the
session extraction and replay machinery and only differ in which blocks they select.
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import pathlib
import re
import shutil
import textwrap
import traceback
import warnings
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from IPython.lib import pretty

from postbound import db
from tests import regression_suite

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs" / "source"
LIVE_CONFIG = ".psycopg_connection_stats"

#: Documents whose sessions need a database, mapped to the (1-based) number of the first block that does. All
#: blocks of a document from this one onwards are replayed in tier 2. Documents not listed here are fully offline.
FIRST_LIVE_BLOCK: dict[str, int] = {
    "10minutes.rst": 4,
    "cookbook.rst": 1,
    "setup.rst": 1,
}

#: Statements (by document and exact source) whose output depends on the environment rather than on PostBOUND, with
#: the reason why. These only need to run and produce the same kind of output as shown in the docs.
UNSTABLE_OUTPUTS: dict[tuple[str, str], str] = {
    ("core/qal.rst", 'query.predicates().filters_for(pb.TableReference("posts", "p"))'): "conjunct order is set order",
    ("core/qal.rst", "filter_pred.column"): "filter_pred is an arbitrary element of a set",
    ("core/qal.rst", "filter_pred.operation"): "filter_pred is an arbitrary element of a set",
    ("core/qal.rst", "filter_pred.value"): "filter_pred is an arbitrary element of a set",
    ("10minutes.rst", "pg_instance"): "shows the server version",
    ("setup.rst", "pg_instance"): "shows the server version",
    ("cookbook.rst", "operators.add(pb.JoinOperator.HashJoin, query.tables())"): "shows a set of tables",
    ("cookbook.rst", "operators"): "shows a set of tables",
    ("cookbook.rst", "hinted_query"): "the hint syntax depends on whether pg_lab or pg_hint_plan is installed",
    (
        "cookbook.rst",
        "print(pg_instance.optimizer().query_plan(hinted_query).inspect())",
    ): "estimates are server-specific",
    ("cookbook.rst", "raw_plan"): "estimates and settings are server-specific",
    ("cookbook.rst", "print(postgres_plan.inspect())"): "estimates are server-specific",
    ("cookbook.rst", "print(qep.inspect())"): "estimates are server-specific",
}

_BLOCK_START = re.compile(r"^(?P<indent>[ \t]*)\.\. code-block:: ipython[ \t]*$")
_INPUT = re.compile(r"^In \[(?P<number>\d+)\]: ?(?P<source>.*)$")
_CONTINUATION = re.compile(r"^ *\.\.\.:(?: (?P<source>.*))?$")
_OUTPUT = re.compile(r"^Out\[\d+\]:")


@dataclass(frozen=True)
class Statement:
    """A single ``In [n]:`` prompt of a session, together with the output the docs show for it."""

    number: int
    source: str
    expected_output: tuple[str, ...]


@dataclass(frozen=True)
class SessionBlock:
    """A ``code-block:: ipython`` of a document."""

    document: str
    index: int
    line: int
    statements: tuple[Statement, ...]

    @property
    def is_live(self) -> bool:
        first_live = FIRST_LIVE_BLOCK.get(self.document)
        return first_live is not None and self.index >= first_live

    @property
    def id(self) -> str:
        return f"{self.document}:{self.line}"


@dataclass(frozen=True)
class StatementResult:
    """The outcome of replaying a `Statement`: its output in session format, or the traceback if it failed."""

    statement: Statement
    output: tuple[str, ...]
    error: str | None


def _normalize(lines: list[str]) -> tuple[str, ...]:
    """Drops trailing whitespace, which the docs do not preserve (the pre-commit hooks strip it)."""
    stripped = [line.rstrip() for line in lines]
    while stripped and not stripped[-1]:
        stripped.pop()
    return tuple(stripped)


def _parse_statements(document: str, body: list[str]) -> tuple[Statement, ...]:
    statements: list[Statement] = []
    number: int | None = None
    source: list[str] = []
    output: list[str] = []

    def flush() -> None:
        if number is not None:
            statements.append(Statement(number, "\n".join(source), _normalize(output)))

    for line in body:
        if match := _INPUT.match(line):
            flush()
            number, source, output = int(match["number"]), [match["source"]], []
        elif number is None:
            if line.strip():
                raise ValueError(f"{document}: session block contains text before its first prompt: {line!r}")
        elif not output and (match := _CONTINUATION.match(line)):
            source.append(match["source"] or "")
        else:
            output.append(line)
    flush()
    return tuple(statements)


def parse_sessions(document: str) -> list[SessionBlock]:
    """Extracts all IPython sessions from a document, given relative to `DOCS_ROOT`."""
    lines = (DOCS_ROOT / document).read_text().split("\n")
    blocks: list[SessionBlock] = []
    i = 0
    while i < len(lines):
        match = _BLOCK_START.match(lines[i])
        if not match:
            i += 1
            continue

        start = i
        directive_indent = len(match["indent"])
        i += 1
        body: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= directive_indent:
                break
            body.append(line)
            i += 1

        statements = _parse_statements(document, textwrap.dedent("\n".join(body)).split("\n"))
        blocks.append(SessionBlock(document, len(blocks) + 1, start + 1, statements))
    return blocks


def documents() -> list[str]:
    return sorted(str(path.relative_to(DOCS_ROOT)) for path in DOCS_ROOT.rglob("*.rst"))


SESSIONS: dict[str, list[SessionBlock]] = {document: parse_sessions(document) for document in documents()}
SESSIONS = {document: blocks for document, blocks in SESSIONS.items() if blocks}
ALL_BLOCKS = [block for blocks in SESSIONS.values() for block in blocks]


def _display(value: object) -> list[str]:
    """Renders an expression result the same way the IPython displayhook does (``Out[n]:`` prompt included)."""
    text = pretty.pretty(value, max_width=79).split("\n")
    return ["", *text] if len(text) > 1 else text


def _run_statement(statement: Statement, namespace: dict[str, object]) -> StatementResult:
    stdout = io.StringIO()
    try:
        # Like IPython, a trailing expression is evaluated separately so that its value can be displayed.
        tree = ast.parse(statement.source)
        last_expression: ast.expr | None = None
        if tree.body and isinstance(last := tree.body[-1], ast.Expr):
            tree.body.pop()
            last_expression = last.value
        filename = f"<In [{statement.number}]>"
        with contextlib.redirect_stdout(stdout):
            exec(compile(tree, filename, "exec"), namespace)
            value = None
            if last_expression is not None:
                value = eval(compile(ast.Expression(last_expression), filename, "eval"), namespace)
    except Exception:
        return StatementResult(statement, (), traceback.format_exc())

    output = stdout.getvalue().rstrip("\n").split("\n") if stdout.getvalue() else []
    if value is not None:
        displayed = _display(value)
        output.append(f"Out[{statement.number}]: {displayed[0]}")
        output.extend(displayed[1:])
    return StatementResult(statement, _normalize(output), None)


@contextlib.contextmanager
def _documentation_environment(workdir: pathlib.Path, *, live: bool) -> Iterator[None]:
    """Runs a replay in `workdir` and undoes everything the documented code does to the process.

    The docs connect to Postgres with the default settings, which registers the database in the global
    `DatabasePool`. The pool is therefore restored after the replay and every database it gained is closed.
    """
    pool = db.DatabasePool.get_instance()
    snapshot = dict(pool._pool)
    previous_cwd = pathlib.Path.cwd()
    if live:
        shutil.copy(REPO_ROOT / LIVE_CONFIG, workdir / ".psycopg_connection")
    os.chdir(workdir)
    try:
        # The original ipython directives were all :okwarning:, the docs never promised a warning-free session.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        os.chdir(previous_cwd)
        for key, instance in pool._pool.items():
            if key not in snapshot:
                instance.close()
        pool.clear()
        for key, instance in snapshot.items():
            pool.register_database(key, instance)


class DocumentationReplay:
    """Replays documents lazily and caches the results, so that each document runs at most once per tier."""

    def __init__(self, workdir: pathlib.Path) -> None:
        self._workdir = workdir
        self._results: dict[tuple[str, bool], dict[int, list[StatementResult]]] = {}

    def results_for(self, block: SessionBlock) -> list[StatementResult]:
        key = (block.document, block.is_live)
        if key not in self._results:
            self._results[key] = self._replay(block.document, live=block.is_live)
        return self._results[key][block.index]

    def _replay(self, document: str, *, live: bool) -> dict[int, list[StatementResult]]:
        # An offline replay stops before the first live block, a live replay has to rebuild the whole session.
        blocks = [block for block in SESSIONS[document] if live or not block.is_live]
        namespace: dict[str, object] = {"__name__": "__main__"}
        workdir = self._workdir / f"{document.replace('/', '_')}-{'live' if live else 'offline'}"
        workdir.mkdir()

        results: dict[int, list[StatementResult]] = {}
        with _documentation_environment(workdir, live=live):
            for block in blocks:
                results[block.index] = [_run_statement(statement, namespace) for statement in block.statements]
        return results


@pytest.fixture(scope="module")
def replay(tmp_path_factory: pytest.TempPathFactory) -> DocumentationReplay:
    return DocumentationReplay(tmp_path_factory.mktemp("docs"))


def _shape(output: tuple[str, ...]) -> tuple[bool, bool]:
    """Whether an output contains printed text and whether it contains a displayed result."""
    displayed_at = next((i for i, line in enumerate(output) if _OUTPUT.match(line)), len(output))
    return displayed_at > 0, displayed_at < len(output)


def _assert_block_reproduces(block: SessionBlock, results: list[StatementResult]) -> None:
    failures: list[str] = []
    for result in results:
        statement = result.statement
        header = f"In [{statement.number}]: {statement.source}"
        if result.error is not None:
            failures.append(f"{header}\nraised an error:\n{result.error}")
        elif (block.document, statement.source) in UNSTABLE_OUTPUTS:
            if _shape(result.output) != _shape(statement.expected_output):
                failures.append(
                    f"{header}\nproduced a different kind of output.\n"
                    f"--- docs\n" + "\n".join(statement.expected_output) + "\n--- actual\n" + "\n".join(result.output)
                )
        elif result.output != statement.expected_output:
            failures.append(
                f"{header}\nproduced different output.\n"
                f"--- docs\n" + "\n".join(statement.expected_output) + "\n--- actual\n" + "\n".join(result.output)
            )

    assert not failures, f"Session block {block.id} no longer matches the docs:\n\n" + "\n\n".join(failures)


# -- session extraction --------------------------------------------------------------------------------------


def test_documentation_contains_no_executed_ipython_directives() -> None:
    """The ``ipython`` directive was removed from the Sphinx config, so its blocks would break the docs build."""
    offending = [
        f"{document}:{number}"
        for document in documents()
        for number, line in enumerate((DOCS_ROOT / document).read_text().split("\n"), start=1)
        if re.match(r"^\s*\.\. ipython::", line)
    ]

    assert offending == []


def test_every_session_block_contains_statements() -> None:
    empty = [block.id for block in ALL_BLOCKS if not block.statements]

    assert ALL_BLOCKS, "no IPython sessions found in the docs -- has the extraction broken?"
    assert empty == []


def test_session_prompts_are_numbered_consecutively_per_document() -> None:
    """Each document is a single session, so its prompts form one sequence. A gap means a statement was lost."""
    for document, blocks in SESSIONS.items():
        numbers = [statement.number for block in blocks for statement in block.statements]

        assert numbers == list(range(1, len(numbers) + 1)), document


def test_live_block_table_refers_to_existing_blocks() -> None:
    for document, first_live in FIRST_LIVE_BLOCK.items():
        assert document in SESSIONS, document
        assert 1 <= first_live <= len(SESSIONS[document]), document


def test_unstable_outputs_refer_to_existing_statements() -> None:
    """A stale entry would silently weaken the check for whatever statement later reuses that source."""
    sources = {(block.document, statement.source) for block in ALL_BLOCKS for statement in block.statements}

    assert set(UNSTABLE_OUTPUTS) - sources == set()


# -- offline sessions ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block",
    [block for block in ALL_BLOCKS if not block.is_live],
    ids=lambda block: block.id,
)
def test_offline_session_block_reproduces_the_documented_output(
    block: SessionBlock, replay: DocumentationReplay
) -> None:
    _assert_block_reproduces(block, replay.results_for(block))


# -- live sessions -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block",
    [
        pytest.param(block, marks=[pytest.mark.live_db, regression_suite.skip_if_no_db(LIVE_CONFIG)])
        for block in ALL_BLOCKS
        if block.is_live
    ],
    ids=lambda block: block.id,
)
def test_live_session_block_reproduces_the_documented_output(block: SessionBlock, replay: DocumentationReplay) -> None:
    _assert_block_reproduces(block, replay.results_for(block))
