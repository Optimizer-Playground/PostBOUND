"""Tests for `postbound.workloads` -- the `Workload` container and the workload loaders.

Everything here is tier 0. Workloads are built in memory from parsed queries, and the file-based loaders read small
query files written to pytest's `tmp_path`. Column binding against a schema uses `FakeDatabase` / `StaticSchema`
instead of a live connection.

The ready-made loaders (`job()`, `stack()`, ...) download their queries on first use. They are covered by patching
`_fetch_workload` to point at a temporary directory; `_fetch_workload` itself is covered by patching the download
(`urllib.request.urlretrieve`) and the PostBOUND home directory. Whether the real archives can still be downloaded and
parsed is only exercised by the live tests that load JOB/Stats (`tests/test_qal.py`, `tests/test_postgres.py`).
"""

from __future__ import annotations

import warnings
import zipfile
from pathlib import Path

import pytest

import postbound as pb
from postbound import parser, workloads
from postbound._core import ColumnReference, TableReference
from postbound.db import DatabasePool
from postbound.qal import SqlQuery
from postbound.workloads import (
    Workload,
    fetch_workload,
    generate_workload,
    read_batch_workload,
    read_csv_workload,
    read_workload,
    wrap_workload,
)
from tests.doubles import FakeDatabase

# -- fixtures -----------------------------------------------------------------------------------------------


def parse(sql: str) -> SqlQuery:
    return parser.parse_query(sql, bind_columns=False)


OWN_QUERY = parse("SELECT * FROM r")
OTHER_QUERY = parse("SELECT * FROM s")
EXTRA_QUERY = parse("SELECT * FROM t")

Q1 = parse("SELECT r.a FROM r")
Q2 = parse("SELECT s.b FROM s")
Q10 = parse("SELECT t.c FROM t")

# The unqualified column `a` can only be bound with the help of a schema. Both databases below disagree on which table
# owns it, so a test can tell which one the parser actually used.
UNQUALIFIED_SQL = "SELECT r.id FROM r, s WHERE a = 1 AND r.id = s.id"
A_IN_R = ColumnReference("a", TableReference("r"))
A_IN_S = ColumnReference("a", TableReference("s"))
UNBOUND_A = ColumnReference("a")


def db_with_a_in_r() -> FakeDatabase:
    return FakeDatabase(schema_spec={"r": ["a", "id"], "s": ["b", "id"]})


def db_with_a_in_s() -> FakeDatabase:
    return FakeDatabase(schema_spec={"r": ["id"], "s": ["a", "b", "id"]})


def unqualified_a(query: SqlQuery) -> ColumnReference:
    """Provides the column reference for `a` in `UNQUALIFIED_SQL`, bound or not."""
    (column,) = [col for col in query.columns() if col.name == "a"]
    return column


def sample_workload() -> Workload[str, SqlQuery]:
    return Workload({"q10": Q10, "q2": Q2, "q1": Q1}, name="sample", root=Path("/tmp/sample"))


def write_queries(directory: Path, files: dict[str, str]) -> Path:
    """Writes `files` (relative path -> SQL) below `directory` and returns `directory`."""
    for relative_path, sql in files.items():
        file = directory / relative_path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(sql)
    return directory


@pytest.fixture
def workload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points the ready-made workload loaders at an (initially empty) local directory instead of the download."""
    target = tmp_path / "workload"
    target.mkdir()
    monkeypatch.setattr(workloads, "_fetch_workload", lambda name: target)
    return target


# -- Workload: ordering and access --------------------------------------------------------------------------


def test_workload_orders_labels_naturally() -> None:
    workload = sample_workload()

    assert workload.labels() == ["q1", "q2", "q10"]
    assert workload.queries() == [Q1, Q2, Q10]
    assert workload.entries() == [("q1", Q1), ("q2", Q2), ("q10", Q10)]


def test_workload_head_provides_the_first_entry_in_natural_order() -> None:
    assert sample_workload().head() == ("q1", Q1)


def test_workload_head_of_an_empty_workload_is_none() -> None:
    assert Workload({}).head() is None


def test_workload_behaves_like_a_read_only_mapping() -> None:
    workload = sample_workload()

    assert len(workload) == 3
    assert workload["q2"] == Q2
    assert "q10" in workload
    assert set(workload) == {"q1", "q2", "q10"}
    with pytest.raises(KeyError):
        workload["q3"]


def test_workload_label_of_provides_the_label_of_a_query() -> None:
    assert sample_workload().label_of(Q2) == "q2"


def test_workload_label_of_rejects_an_unknown_query() -> None:
    with pytest.raises(KeyError, match="not part of the workload"):
        sample_workload().label_of(OWN_QUERY)


@pytest.mark.parametrize(
    ("workload", "expected"),
    [
        (Workload({"q1": Q1}, name="sample", root=Path("/tmp/root")), "Workload: sample (1 queries)"),
        (Workload({"q1": Q1}, root=Path("/tmp/root")), "Workload: root (1 queries)"),
        (Workload({"q1": Q1}), "Workload: 1 queries"),
    ],
    ids=["named", "root-only", "anonymous"],
)
def test_workload_str_prefers_name_over_root(workload: Workload, expected: str) -> None:
    assert str(workload) == expected
    assert repr(workload) == expected


# -- Workload: selections -----------------------------------------------------------------------------------


def test_workload_with_labels_keeps_only_the_requested_labels() -> None:
    selected = sample_workload().with_labels(["q10", "q1", "unknown"])

    assert selected.labels() == ["q1", "q10"]


def test_workload_selections_keep_name_and_root() -> None:
    selected = sample_workload().first(1)

    assert selected.name == "sample"
    assert str(selected) == "Workload: sample (1 queries)"


def test_workload_first_and_last_follow_the_natural_order() -> None:
    workload = sample_workload()

    assert workload.first(2).labels() == ["q1", "q2"]
    assert workload.last(2).labels() == ["q2", "q10"]


def test_workload_first_takes_everything_if_n_exceeds_the_size() -> None:
    assert sample_workload().first(10).labels() == ["q1", "q2", "q10"]


def test_workload_selections_leave_the_original_unchanged() -> None:
    workload = sample_workload()

    workload.first(1)
    workload.shuffle()
    workload.filter_by(lambda label, _: label == "q1")

    assert workload.labels() == ["q1", "q2", "q10"]
    assert dict(workload) == {"q1": Q1, "q2": Q2, "q10": Q10}


def test_workload_pick_random_selects_distinct_queries_of_the_workload() -> None:
    workload = sample_workload()

    picked = workload.pick_random(2)

    assert len(picked) == 2
    assert set(picked.labels()) <= set(workload.labels())
    assert all(picked[label] == workload[label] for label in picked)


def test_workload_pick_random_takes_everything_if_n_exceeds_the_size() -> None:
    assert sample_workload().pick_random(10).labels() == ["q1", "q2", "q10"]


def test_workload_with_prefix_filters_by_label_prefix() -> None:
    workload = Workload({"a/q1": Q1, "a/q2": Q2, "b/q1": Q10})

    assert workload.with_prefix("a/").labels() == ["a/q1", "a/q2"]


def test_workload_with_prefix_rejects_labels_without_startswith() -> None:
    workload = Workload({1: Q1})

    with pytest.raises(ValueError, match="startswith"):
        workload.with_prefix(1)


def test_workload_filter_by_receives_label_and_query() -> None:
    selected = sample_workload().filter_by(lambda label, query: label != "q2" and query != Q10)

    assert dict(selected) == {"q1": Q1}


# -- Workload: transformations ------------------------------------------------------------------------------


def test_workload_relabel_replaces_labels_and_reorders() -> None:
    relabeled = sample_workload().relabel(lambda label, _: int(label.removeprefix("q")))

    assert relabeled.entries() == [(1, Q1), (2, Q2), (10, Q10)]


def test_workload_map_replaces_queries_and_forwards_arguments() -> None:
    def replace(query: SqlQuery, replacement: SqlQuery, *, only: SqlQuery) -> SqlQuery:
        return replacement if query == only else query

    mapped = sample_workload().map(replace, OWN_QUERY, only=Q2)

    assert dict(mapped) == {"q1": Q1, "q2": OWN_QUERY, "q10": Q10}


def test_workload_transform_replaces_labels_and_queries() -> None:
    transformed = sample_workload().transform(lambda label, query, suffix: (label + suffix, query), "x")

    assert dict(transformed) == {"q1x": Q1, "q2x": Q2, "q10x": Q10}


def test_workload_shuffle_keeps_label_query_pairs_intact() -> None:
    workload = sample_workload()

    shuffled = workload.shuffle()

    assert sorted(shuffled.labels()) == sorted(workload.labels())
    assert list(zip(shuffled.labels(), shuffled.queries(), strict=True)) == shuffled.entries()
    assert all(workload[label] == query for label, query in shuffled.entries())


def test_workload_ordered_restores_the_natural_order() -> None:
    workload = sample_workload()

    # A shuffle may by chance produce the natural order, so the order is broken by hand instead.
    shuffled = workload.shuffle()
    shuffled._sorted_labels = ["q10", "q1", "q2"]
    shuffled._update_query_order()

    assert shuffled.ordered().entries() == [("q1", Q1), ("q2", Q2), ("q10", Q10)]


# -- Workload: set operations -------------------------------------------------------------------------------


def test_workload_addition_keeps_its_own_query_on_a_label_conflict() -> None:
    own = Workload({"q1": OWN_QUERY})
    other = Workload({"q1": OTHER_QUERY, "q2": EXTRA_QUERY})

    merged = own + other

    assert dict(merged) == {"q1": OWN_QUERY, "q2": EXTRA_QUERY}


def test_workload_addition_rejects_non_workloads() -> None:
    with pytest.raises(TypeError, match="add workloads"):
        Workload({"q1": OWN_QUERY}) + {"q2": OTHER_QUERY}  # ty: ignore[unsupported-operator]


@pytest.mark.parametrize(
    "other",
    [Workload({"q2": OTHER_QUERY}), ["q2"], {"q2"}],
    ids=["workload", "label-list", "label-set"],
)
def test_workload_subtraction_removes_labels(other) -> None:
    workload = Workload({"q1": OWN_QUERY, "q2": EXTRA_QUERY})

    assert dict(workload - other) == {"q1": OWN_QUERY}


@pytest.mark.parametrize(
    "other",
    [Workload({"q2": OTHER_QUERY}), ["q2"], {"q2"}],
    ids=["workload", "label-list", "label-set"],
)
def test_workload_intersection_keeps_own_queries_of_shared_labels(other) -> None:
    workload = Workload({"q1": OWN_QUERY, "q2": EXTRA_QUERY})

    assert dict(workload & other) == {"q2": EXTRA_QUERY}


@pytest.mark.parametrize("operator", ["-", "&"], ids=["subtraction", "intersection"])
def test_workload_set_operations_reject_non_iterables(operator: str) -> None:
    workload = Workload({"q1": OWN_QUERY})

    with pytest.raises(TypeError, match="Expected workload or labels"):
        if operator == "-":
            workload - 1  # ty: ignore[unsupported-operator]
        else:
            workload & 1  # ty: ignore[unsupported-operator]


def test_workload_union_accepts_plain_mappings() -> None:
    merged = Workload({"q1": OWN_QUERY}) | {"q1": OTHER_QUERY, "q2": EXTRA_QUERY}

    assert dict(merged) == {"q1": OWN_QUERY, "q2": EXTRA_QUERY}


# -- generate_workload / wrap_workload ----------------------------------------------------------------------


def test_generate_workload_uses_numeric_labels_counting_from_one() -> None:
    workload = generate_workload([Q1, Q2])

    assert dict(workload) == {1: Q1, 2: Q2}


def test_generate_workload_uses_explicit_labels() -> None:
    workload = generate_workload([Q1, Q2], labels={Q1: "first", Q2: "second"})

    assert dict(workload) == {"first": Q1, "second": Q2}


def test_generate_workload_infers_the_name_from_the_root() -> None:
    assert generate_workload([Q1], workload_root="/data/my-queries.csv").name == "my-queries"
    assert generate_workload([Q1], name="explicit", workload_root="/data/my-queries.csv").name == "explicit"


def test_wrap_workload_uses_numeric_labels() -> None:
    workload = wrap_workload([Q1, Q2], name="wrapped")

    assert dict(workload) == {1: Q1, 2: Q2}
    assert workload.name == "wrapped"


# -- read_workload ------------------------------------------------------------------------------------------


def test_read_workload_labels_queries_by_file_name(tmp_path: Path) -> None:
    write_queries(tmp_path, {"q1.sql": "SELECT r.a FROM r", "q2.sql": "SELECT s.b\nFROM s"})

    workload = read_workload(tmp_path, "files")

    assert dict(workload) == {"q1": Q1, "q2": Q2}
    assert workload.name == "files"


def test_read_workload_only_reads_files_matching_the_pattern(tmp_path: Path) -> None:
    write_queries(tmp_path, {"q1.sql": "SELECT r.a FROM r", "q2.txt": "SELECT s.b FROM s", "notes.md": "# notes"})

    assert read_workload(tmp_path).labels() == ["q1"]
    assert read_workload(tmp_path, query_file_pattern="*.txt").labels() == ["q2"]


def test_read_workload_ignores_subdirectories_by_default(tmp_path: Path) -> None:
    write_queries(tmp_path, {"q1.sql": "SELECT r.a FROM r", "sub/q2.sql": "SELECT s.b FROM s"})

    assert read_workload(tmp_path).labels() == ["q1"]


def test_read_workload_prefixes_labels_with_the_global_prefix(tmp_path: Path) -> None:
    write_queries(tmp_path, {"q1.sql": "SELECT r.a FROM r"})

    assert read_workload(tmp_path, query_label_prefix="job-").labels() == ["job-q1"]


def test_read_workload_keeps_hints_by_default(tmp_path: Path) -> None:
    write_queries(tmp_path, {"q1.sql": "/*+ SeqScan(r) */ SELECT r.a FROM r"})

    with_hints = read_workload(tmp_path)["q1"]
    without_hints = read_workload(tmp_path, include_hints=False)["q1"]

    assert with_hints.hints is not None
    assert "SeqScan(r)" in str(with_hints.hints)
    assert without_hints.hints is None


def test_read_workload_propagates_parser_errors_by_default(tmp_path: Path) -> None:
    write_queries(tmp_path, {"broken.sql": "SELEC r.a FROM r"})

    with pytest.raises(ValueError, match=r"Could not parse query from .*broken\.sql"):
        read_workload(tmp_path)


def test_read_workload_warns_about_unparsable_files(tmp_path: Path) -> None:
    write_queries(tmp_path, {"broken.sql": "SELEC r.a FROM r"})

    with pytest.warns(UserWarning, match=r"Could not parse query .*broken\.sql"):
        read_workload(tmp_path, on_error="warn")


def test_read_workload_binds_unqualified_columns_against_the_given_database(tmp_path: Path) -> None:
    write_queries(tmp_path, {"q1.sql": UNQUALIFIED_SQL})

    workload = read_workload(tmp_path, database=db_with_a_in_r())

    assert unqualified_a(workload["q1"]) == A_IN_R


def test_read_workload_prefers_the_given_database_over_the_pool(tmp_path: Path, fake_db: FakeDatabase) -> None:
    """The explicit database makes binding deterministic, even if the pool holds several (conflicting) databases."""
    fake_db.set_schema({"r": ["id"], "s": ["a", "b", "id"]})
    DatabasePool.get_instance().register_database("other", db_with_a_in_s())
    write_queries(tmp_path, {"q1.sql": UNQUALIFIED_SQL})

    with warnings.catch_warnings():
        warnings.simplefilter("error", parser.ParserWarning)
        workload = read_workload(tmp_path, database=db_with_a_in_r())

    assert unqualified_a(workload["q1"]) == A_IN_R


def test_read_workload_falls_back_to_the_pool_without_a_database(tmp_path: Path, fake_db: FakeDatabase) -> None:
    fake_db.set_schema({"r": ["id"], "s": ["a", "b", "id"]})
    write_queries(tmp_path, {"q1.sql": UNQUALIFIED_SQL})

    workload = read_workload(tmp_path)

    assert unqualified_a(workload["q1"]) == A_IN_S


def test_read_workload_forwards_the_database_to_subdirectories(tmp_path: Path) -> None:
    write_queries(tmp_path, {"sub/q1.sql": UNQUALIFIED_SQL})

    workload = read_workload(tmp_path, recurse_subdirectories=True, database=db_with_a_in_r())

    assert unqualified_a(workload["sub/q1"]) == A_IN_R


# -- read_batch_workload ------------------------------------------------------------------------------------


def test_read_batch_workload_labels_queries_by_line(tmp_path: Path) -> None:
    batch_file = tmp_path / "batch.sql"
    batch_file.write_text("SELECT r.a FROM r\nSELECT s.b FROM s\n")

    workload = read_batch_workload(str(batch_file))

    assert dict(workload) == {1: Q1, 2: Q2}
    assert workload.name == "batch"


def test_read_batch_workload_binds_unqualified_columns_against_the_given_database(tmp_path: Path) -> None:
    batch_file = tmp_path / "batch.sql"
    batch_file.write_text(UNQUALIFIED_SQL + "\n")

    workload = read_batch_workload(str(batch_file), database=db_with_a_in_r())

    assert unqualified_a(workload[1]) == A_IN_R


def test_read_batch_workload_rejects_non_sql_lines(tmp_path: Path) -> None:
    batch_file = tmp_path / "batch.sql"
    batch_file.write_text("SELECT r.a FROM r\nthis is not SQL\n")

    with pytest.raises(Exception, match="syntax error"):
        read_batch_workload(str(batch_file))


# -- read_csv_workload --------------------------------------------------------------------------------------


def test_read_csv_workload_uses_labels_from_the_label_column(tmp_path: Path) -> None:
    csv_file = tmp_path / "queries.csv"
    csv_file.write_text('label,query,ignored\nfirst,"SELECT r.a FROM r",x\nsecond,"SELECT s.b FROM s",y\n')

    workload = read_csv_workload(csv_file, label_column="label")

    assert dict(workload) == {"first": Q1, "second": Q2}
    assert workload.name == "queries"


def test_read_csv_workload_uses_numeric_labels_without_a_label_column(tmp_path: Path) -> None:
    csv_file = tmp_path / "queries.csv"
    csv_file.write_text('sql\n"SELECT r.a FROM r"\n"SELECT s.b FROM s"\n')

    workload = read_csv_workload(csv_file, query_column="sql", label_column=None)

    assert dict(workload) == {1: Q1, 2: Q2}


def test_read_csv_workload_does_not_let_pd_args_override_the_query_parsing(tmp_path: Path) -> None:
    csv_file = tmp_path / "queries.csv"
    csv_file.write_text('query;comment\n"SELECT r.a FROM r";x\n')

    workload = read_csv_workload(
        csv_file, label_column=None, pd_args={"sep": ";", "converters": {}, "usecols": ["comment"]}
    )

    assert dict(workload) == {1: Q1}


def test_read_csv_workload_binds_unqualified_columns_against_the_given_database(tmp_path: Path) -> None:
    csv_file = tmp_path / "queries.csv"
    csv_file.write_text(f'query\n"{UNQUALIFIED_SQL}"\n')

    workload = read_csv_workload(csv_file, label_column=None, database=db_with_a_in_r())

    assert unqualified_a(workload[1]) == A_IN_R


# -- ready-made workloads -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "expected_name"),
    [
        ("job", "JOB"),
        ("imdb", "JOB"),
        ("job-light", "JOB-light"),
        ("joblight", "JOB-light"),
        ("job-complex", "JOB-complex"),
        ("jobcomplex", "JOB-complex"),
        ("stats", "Stats"),
        ("stack", "Stack"),
        ("ssb", "SSB"),
    ],
)
def test_fetch_workload_dispatches_to_the_ready_made_loader(workload_dir: Path, key: str, expected_name: str) -> None:
    write_queries(workload_dir, {"q1.sql": "SELECT r.a FROM r"})

    workload = fetch_workload(key)

    assert workload.name == expected_name
    assert dict(workload) == {"q1": Q1}


def test_fetch_workload_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="No pre-defined workload registered for key 'tpch'"):
        fetch_workload("tpch")


def test_ready_made_loaders_reject_an_empty_workload_directory(workload_dir: Path) -> None:
    with pytest.raises(ValueError, match="Could not load JOB workload"):
        workloads.job()


def test_stack_labels_queries_by_context(workload_dir: Path) -> None:
    write_queries(workload_dir, {"q1/q1-001.sql": "SELECT r.a FROM r", "q2/q2-001.sql": "SELECT s.b FROM s"})

    assert dict(workloads.stack()) == {"q1/q1-001": Q1, "q2/q2-001": Q2}


@pytest.mark.parametrize("loader", ["stack", "ssb"])
def test_loaders_bind_columns_if_a_database_is_given(workload_dir: Path, loader: str) -> None:
    """With an empty pool, an explicit database alone must switch on the default binding."""
    write_queries(workload_dir, {"q1.sql": UNQUALIFIED_SQL})

    workload = getattr(workloads, loader)(database=db_with_a_in_r())

    assert unqualified_a(workload["q1"]) == A_IN_R


@pytest.mark.parametrize("loader", ["stack", "ssb"])
def test_loaders_bind_columns_if_the_pool_is_not_empty(workload_dir: Path, fake_db: FakeDatabase, loader: str) -> None:
    fake_db.set_schema({"r": ["id"], "s": ["a", "b", "id"]})
    write_queries(workload_dir, {"q1.sql": UNQUALIFIED_SQL})

    workload = getattr(workloads, loader)()

    assert unqualified_a(workload["q1"]) == A_IN_S


@pytest.mark.parametrize("loader", ["stack", "ssb"])
def test_loaders_honour_an_explicit_bind_columns_false(workload_dir: Path, loader: str) -> None:
    write_queries(workload_dir, {"q1.sql": UNQUALIFIED_SQL})

    workload = getattr(workloads, loader)(bind_columns=False, database=db_with_a_in_r())

    assert unqualified_a(workload["q1"]) == UNBOUND_A


@pytest.mark.parametrize("key", ["stack", "ssb"])
def test_fetch_workload_forwards_the_database(workload_dir: Path, key: str) -> None:
    write_queries(workload_dir, {"q1.sql": UNQUALIFIED_SQL})

    workload = fetch_workload(key, database=db_with_a_in_r())

    assert unqualified_a(workload["q1"]) == A_IN_R


# -- _fetch_workload ----------------------------------------------------------------------------------------


@pytest.fixture
def postbound_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "pbhome"
    monkeypatch.setattr(workloads, "pbdir", home)
    return home


def test_fetch_workload_downloads_and_unpacks_the_archive_once(
    postbound_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads: list[str] = []

    def fake_download(url: str, target: Path) -> None:
        downloads.append(url)
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("stats/q-1.sql", "SELECT r.a FROM r")

    monkeypatch.setattr(workloads.urllib.request, "urlretrieve", fake_download)

    first = workloads._fetch_workload("Stats")
    second = workloads._fetch_workload("stats")

    assert first == second == postbound_home / "workloads" / "stats"
    assert [file.name for file in first.iterdir()] == ["q-1.sql"]
    assert downloads == [workloads._WorkloadSources["stats"]]
    assert not (postbound_home / "workloads" / "stats.zip").exists()


def test_fetch_workload_rejects_workloads_without_a_source(postbound_home: Path) -> None:
    with pytest.raises(ValueError, match="no known data source"):
        workloads._fetch_workload("ssb")


# -- public exposure ----------------------------------------------------------------------------------------


def test_workload_api_is_reachable_through_the_package() -> None:
    assert pb.Workload is Workload
    assert pb.workloads.read_workload is read_workload
    assert pb.workloads.fetch_workload is fetch_workload


# -- regression tests --------------------------------------------------------------------------------------


def test_union_operator_keeps_its_own_query_on_a_label_conflict() -> None:
    """Regression guard for 828cc52: when `Workload` became a read-only `Mapping`, `__or__` copied its own entries
    first and then applied ``update(other)``, so the right-hand operand overwrote conflicting labels. v0.21.6 (and the
    inline comment) retain the left-hand query.
    """
    own = Workload({"q1": OWN_QUERY})
    other = Workload({"q1": OTHER_QUERY, "q2": EXTRA_QUERY})

    merged = own | other

    assert dict(merged) == {"q1": OWN_QUERY, "q2": EXTRA_QUERY}


def test_read_workload_prefixes_labels_with_their_subdirectories(tmp_path: Path) -> None:
    """Regression guard for cdb1b5d: `read_workload` computed the subdirectory prefix but labelled every query by its
    bare file name. Queries with the same name in different subdirectories (Stack has a ``q1-001`` style name per
    context) silently overwrote each other, and labels did not match the documented ``q1/q1-001`` format.
    """
    write_queries(
        tmp_path,
        {"top.sql": "SELECT r.a FROM r", "a/q.sql": "SELECT s.b FROM s", "b/q.sql": "SELECT t.c FROM t"},
    )

    workload = read_workload(tmp_path, recurse_subdirectories=True)

    assert dict(workload) == {"top": Q1, "a/q": Q2, "b/q": Q10}


def test_read_workload_prefixes_nested_subdirectories_with_the_full_path(tmp_path: Path) -> None:
    """Regression guard for cdb1b5d: nested subdirectories must contribute every directory level to the label."""
    write_queries(tmp_path, {"a/b/q.sql": "SELECT r.a FROM r"})

    workload = read_workload(tmp_path, recurse_subdirectories=True, query_label_prefix="w")

    assert workload.labels() == ["w/a/b/q"]


@pytest.mark.parametrize("on_error", ["warn", "ignore"])
def test_read_workload_skips_unparsable_files_instead_of_reusing_another_query(tmp_path: Path, on_error) -> None:
    """Regression guard: with ``on_error="warn"/"ignore"`` the loop fell through to the label assignment after a parser
    error. The broken file was then labelled with the query parsed from the *previous* file, or the loader crashed with
    an `UnboundLocalError` if the broken file came first.
    """
    write_queries(tmp_path, {"a.sql": "SELECT r.a FROM r", "b.sql": "SELEC broken", "c.sql": "SELEC broken"})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workload = read_workload(tmp_path, on_error=on_error)

    assert dict(workload) == {"a": Q1}


def test_read_workload_does_not_bind_columns_if_binding_is_disabled(tmp_path: Path) -> None:
    """Regression guard: the parser used to bind against an explicitly passed schema even if ``bind_columns=False``.
    Since `read_workload` hands the schema of its `database` through unconditionally, this re-enabled binding that the
    caller had switched off. The root cause is guarded in ``test_parser.py``; this test covers the loader end to end.
    """
    write_queries(tmp_path, {"q1.sql": UNQUALIFIED_SQL})

    workload = read_workload(tmp_path, bind_columns=False, database=db_with_a_in_r())

    assert unqualified_a(workload["q1"]) == UNBOUND_A


def test_read_batch_workload_skips_empty_lines(tmp_path: Path) -> None:
    """Regression guard: lines are read including their trailing newline, so the ``if raw`` filter never dropped an
    empty line and the parser failed on the bare ``"\\n"``, contradicting the docstring.
    """
    batch_file = tmp_path / "batch.sql"
    batch_file.write_text("SELECT r.a FROM r\n\n   \nSELECT s.b FROM s\n")

    workload = read_batch_workload(str(batch_file))

    assert dict(workload) == {1: Q1, 2: Q2}


@pytest.mark.parametrize("loader", ["batch", "csv"])
@pytest.mark.usefixtures("no_column_binding")
def test_file_loaders_leave_the_binding_default_to_the_parser(
    tmp_path: Path, fake_db: FakeDatabase, loader: str
) -> None:
    """Regression guard: `read_batch_workload` and `read_csv_workload` used to call the parser without `bind_columns`,
    so the global `auto_bind_columns` setting applied. Adding an explicit ``bind_columns=True`` default silently
    overrode a disabled global setting.
    """
    fake_db.set_schema({"r": ["a", "id"], "s": ["b", "id"]})
    if loader == "batch":
        file = tmp_path / "batch.sql"
        file.write_text(UNQUALIFIED_SQL + "\n")
        workload = read_batch_workload(str(file))
    else:
        file = tmp_path / "queries.csv"
        file.write_text(f'query\n"{UNQUALIFIED_SQL}"\n')
        workload = read_csv_workload(file, label_column=None)

    assert unqualified_a(workload[1]) == UNBOUND_A
