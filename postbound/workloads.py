"""Provides abstractions to represent entire query workloads and utilities to read some pre-defined instances.

The main abstraction provided by this class is the `Workload`. A number of utility functions to read collections of queries
from different sources and input formats into workload objects exist as well. The pre-defined workloads include the
Join Order Benchmark [1]_ (including JOB-light [2]_ and JOB-Complex [3]_), Star Schema Benchmark [4]_, Stack Benchmark [5]_
and Stats Benchmark [6]_.

PostBOUND stores the workload queries in a dedicated directory, located relative to the user's home directory at
*$HOME/.postbound/*. If a workload is requested for the first time, it will be downloaded automatically. Therefore, the initial
usage of a novel workload may take slightly longer than usual.

References
----------

.. [1] Viktor Leis et al.: How Good Are Query Optimizers, Really? (Proc. VLDB Endow. 9, 3 (2015))
.. [2] Andreas Kipf et al.: Learned Cardinalities: Estimating Correlated Joins with Deep Learning. (CIDR'2019)
.. [3] Johannes Wehrstein.: JOB-Complex: A Challenging Benchmark for Traditional & Learned Query Optimization. (AIDB'2025)
.. [4] Patrick E. O'Neil et al.: The Star Schema Benchmark and Augmented Fact Table Indexing. (TPCTC'2009)
.. [5] Ryan Marcus et al.: Bao: Making Learned Query Optimization Practical. (SIGMOD'2021)
.. [6] Yuxing Han et al.: Cardinality Estimation in DBMS: A Comprehensive Benchmark Evaluation (Proc. VLDB Endow. 15, 4 (2022))
"""

from __future__ import annotations

import random
import shutil
import urllib.request
import warnings
import zipfile
from collections import UserDict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Optional, cast, overload

import natsort
import pandas as pd
import tqdm

from . import parser, util
from ._base import pbdir
from .db import DatabasePool
from .qal import SelectStatement, SqlQuery

_WorkloadSources = {
    "job": "https://zenodo.org/records/19205561/files/job.zip?download=1",
    "job-complex": "https://zenodo.org/records/19205561/files/job-complex.zip?download=1",
    "job-light": "https://zenodo.org/records/19205561/files/job-light.zip?download=1",
    # "ssb": "https://db4701.inf.tu-dresden.de:8443/public.php/dav/files/iXD5p3J5q6DwdbQ",
    "stack": "https://datashare.tu-dresden.de/public.php/dav/files/JSaFxzzGHGABbxY",
    "stats": "https://zenodo.org/records/19131189/files/queries.zip?download=1",
}


def _fetch_workload(name: str) -> Path:
    """Determines the local path of a workload, downloading it if necessary."""
    name = name.lower()
    workload_dir = pbdir / "workloads" / name
    if workload_dir.exists():
        return workload_dir
    workload_dir.mkdir(parents=True, exist_ok=True)

    archive_url = _WorkloadSources.get(name)
    if not archive_url:
        raise ValueError(
            f"Could not automatically load queries for workload '{name}': no known data source. "
            f"Please insert the queries manually at {workload_dir} and re-run the operation."
        )

    archive_file = workload_dir.parent / f"{name}.zip"
    urllib.request.urlretrieve(archive_url, archive_file)

    with zipfile.ZipFile(archive_file, "r") as zip:
        zip.extractall(workload_dir)

    temp_dir = workload_dir / name
    for file in temp_dir.iterdir():
        shutil.move(file, workload_dir)
    temp_dir.rmdir()

    archive_file.unlink()
    return workload_dir


class Workload[L: Hashable, Q: SqlQuery](UserDict[L, Q]):
    """A workload collects a number of queries (read: benchmark) and provides utilities to operate on them conveniently.

    In addition to the actual queries, each query is annotated by a label that can be used to retrieve the query more
    nicely. E.g. for queries in the Join Order Benchmark, access by their index is supported - such as ``job["1a"]``. Labels
    can be arbitrary types as long as they are hashable. Since the workload inherits from dict, the label can be used directly
    to fetch the associated query (and will raise ``KeyError`` instances for unknown labels).

    Each workload can be given a name, which is mainly intended for readability in ``__str__`` methods and does not serve
    a functional purpose. However, it may be good practice to use a normalized name that can be used in different contexts
    such as in file names, etc.

    When using methods that allow iteration over the queries, they will typically be returned in order according to the natural
    order of the query labels. However, since workloads can be shuffled randomly, this order can also be destroyed.

    A workload is implemented as an immutable data object. Therefore, it is not possible/not intended to change the contents
    of a workload object later on. All methods that mutate the contents instead provide new workload instances.

    Parameters
    ----------
    queries : dict[LabelType, SqlQuery]
        The queries that form the actual workload
    name : str, optional
        A name that can be used to identify or represent the workload, by default ``""``.
    root : Optional[Path], optional
        The root directory that contains the workload queries. This is mainly used to somehow identify the workload when no
        name is given or the workload contents do not match the expected queries. Defaults to ``None``.

    Notes
    -----
    Workloads support many of the Python builtin-methods thanks to inheriting from ``UserDict``. Namely, the *len*, *iter* and
    *in* methods work as expected on the labels. Furthermore, multiple workload objects can be added, subtracted and
    intersected using set semantics. Subtraction and intersection also work based on individual labels.
    """

    @staticmethod
    def read(
        root_dir: str | Path,
        *,
        query_file_pattern: str = "*.sql",
        name: str = "",
        label_prefix: str = "",
        file_encoding: str = "utf-8",
        bind_columns: bool = True,
        include_hints: bool = True,
        on_error: Literal["raise", "warn", "ignore"] = "raise",
        verbose: bool = False,
    ) -> Workload[str, SqlQuery]:
        """Reads all SQL queries from a specific directory into a workload object.

        .. deprecated:: 0.20.2
            Use the `read_workload` function instead, which provides more flexibility and features.
            We will unify the workload API in 0.21.0 to make sure all workload loading functions are methods of the
            `workloads` module.

        This method assumes that the queries are stored in individual files, one query per file. The query labels will be
        constructed based on the file name of the source files. For example, a query contained in file ``q-1-1.sql`` will
        receive label ``q-1-1`` (note that the trailing file extension is dropped). If the `label_prefix` is given, it will be
        inserted before the file name-based label.

        Parameters
        ----------
        root_dir : str
            Directory containing the individual query files
        query_file_pattern : str, optional
            File name pattern that is shared by all query files. Only files matching the pattern will be read and each matching
            file is assumed to be a valid workload query. This is resolved as a glob expression. Defaults to ``"*.sql"``
        name : str, optional
            An optional name that can be used to identify the workload. Empty by default.
        label_prefix : str, optional
            A prefix to add before each query label. Empty by default. Notice that the prefix will be prepended as-is, i.e. no
            separator character is inserted. If a separator is desired, it has to be part of the prefix.
        file_encoding : str, optional
            The encoding of the query files. All files must share the same encoding. Defaults to UTF-8 encoding.
        bind_columns : bool, optional
            Whether the parser should try to infer tables for all column references which do not use the explicit *alias.column*
            syntax. This requires an active database connection. If no such connection exists, this setting is ignored.
        include_hints : bool, optional
            Whether the parser should try to infer and extract hint blocks from the queries
        on_error : Literal["raise", "warn", "ignore"], optional
            How to react to parser errors. By default, the exception is propagated to the parent process.
        verbose : bool, optional
            Whether progress information should be printed.

        Returns
        -------
        Workload[str]
            A workload consisting of all query files contained in the root directory.

        See Also
        --------
        Path.glob
        parser.parse_query
        """
        queries: dict[str, SqlQuery] = {}
        root = Path(root_dir)

        if verbose:
            matching_files = list(root.glob(query_file_pattern))
            matching_files = tqdm.tqdm(matching_files, desc=root.name, unit="q")
        else:
            matching_files = root.glob(query_file_pattern)

        for query_file_path in matching_files:
            with open(query_file_path, "r", encoding=file_encoding) as query_file:
                raw_contents = query_file.readlines()
            query_contents = "\n".join([line for line in raw_contents])

            try:
                parsed_query = parser.parse_query(
                    query_contents,
                    include_hints=include_hints,
                    bind_columns=bind_columns,
                )
            except Exception as e:
                match on_error:
                    case "raise":
                        raise ValueError(f"Could not parse query from {query_file_path}", e)
                    case "warn":
                        warnings.warn(f"Could not parse query {query_file_path}: {e} ({type(e)})")
                    case "ignore":
                        pass

            query_label = query_file_path.stem
            queries[label_prefix + query_label] = parsed_query

        return Workload(queries, name=name, root=root)

    def __init__(
        self,
        queries: Mapping[L, Q],
        name: str = "",
        root: Optional[Path] = None,
    ) -> None:
        super().__init__(queries)
        self._name = name
        self._root = root

        self._sorted_labels = natsort.natsorted(list(self.keys()))
        self._sorted_queries: list[Q] = []
        self._update_query_order()

        self._label_mapping = util.dicts.invert(self.data)

    @property
    def name(self) -> str:
        """Provides the name of the workload.

        Returns
        -------
        str
            The name or an empty string if no name has been specified.
        """
        return self._name

    def queries(self) -> Sequence[Q]:
        """Provides all queries in the workload in natural order (according to their labels).

        If the natural order was manually destroyed, e.g. by shuffling, the shuffled order is used.

        Returns
        -------
        Sequence[SqlQuery]
            The queries
        """
        return list(self._sorted_queries)

    def labels(self) -> Sequence[L]:
        """Provides all query labels of the workload in natural order.

        If the natural order was manually destroyed, e.g. by shuffling, the shuffled order is used.

        Returns
        -------
        Sequence[LabelType]
            The labels
        """
        return list(self._sorted_labels)

    def entries(self) -> Sequence[tuple[L, Q]]:
        """Provides all (label, query) pairs in the workload, in natural order of the query labels.

        If the natural order was manually destroyed, e.g. by shuffling, the shuffled order is used.

        Returns
        -------
        Sequence[tuple[LabelType, SqlQuery]]
            The queries along with their labels
        """
        return list(zip(self._sorted_labels, self._sorted_queries))

    def head(self) -> Optional[tuple[L, Q]]:
        """Provides the first query in the workload.

        The first query is determined according to the natural order of the query labels by default. If that order was manually
        destroyed, e.g. by shuffling, the shuffled order is used.

        There is no policy to break ties in the order. An arbitrary query can be returned in this case.

        Returns
        -------
        Optional[tuple[LabelType, SqlQuery]]
            The first query, if there is at least one query in the workload. ``None`` otherwise.
        """
        if not self._sorted_labels:
            return None
        return self._sorted_labels[0], self._sorted_queries[0]

    def label_of(self, query: Q) -> L:
        """Provides the label of the given query.

        Parameters
        ----------
        query : SqlQuery
            The query to check

        Returns
        -------
        LabelType
            The corresponding label

        Raises
        ------
        KeyError
            If the query is not part of the workload
        """
        return self._label_mapping[query]

    def with_labels(self, labels: Iterable[L]) -> Workload[L, Q]:
        """Provides a new workload that contains only the queries with the specified labels.

        Parameters
        ----------
        labels : Iterable[LabelType]
            The labels to include in the new workload

        Returns
        -------
        Workload[LabelType]
            A workload that contains only the queries with the specified labels
        """
        labels = set(labels)
        selected_queries = {label: query for label, query in self.data.items() if label in labels}
        return Workload(selected_queries, name=self._name, root=self._root)

    def first(self, n: int) -> Workload[L, Q]:
        """Provides the first `n` queries of the workload, according to the natural order of the query labels.

        If there are less than `n` queries in the workload, all queries will be returned. Similar to other methods that rely
        on some sort of ordering of the queries, if the natural order has been manually broken due to shuffling, the shuffled
        order is used instead.

        Parameters
        ----------
        n : int
            The number of queries that should be returned

        Returns
        -------
        Workload[LabelType]
            A workload consisting of the first `n` queries of the current workload
        """
        first_n_labels = self._sorted_labels[:n]
        sub_workload = {label: self.data[label] for label in first_n_labels}
        return Workload(sub_workload, self._name, self._root)

    def last(self, n: int) -> Workload[L, Q]:
        """Provides the last `n` queries of the workload, according to the natural order of the query labels.

        If there are less than `n` queries in the workload, all queries will be returned. Similar to other methods that rely
        on some sort of ordering of the queries, if the natural order has been manually broken due to shuffling, the shuffled
        order is used instead.

        Parameters
        ----------
        n : int
            The number of queries that should be returned

        Returns
        -------
        Workload[LabelType]
            A workload consisting of the last `n` queries of the current workload
        """
        last_n_labels = self._sorted_labels[-n:]
        sub_workload = {label: self.data[label] for label in last_n_labels}
        return Workload(sub_workload, self._name, self._root)

    def pick_random(self, n: int) -> Workload[L, Q]:
        """Constructs a new workload, consisting of randomly selected queries from this workload.

        The new workload will once again be ordered according to the natural ordering of the labels.

        Parameters
        ----------
        n : int
            The number of queries to choose. If there are less queries in the workload, all will be selected.

        Returns
        -------
        Workload[LabelType]
            A workload consisting of `n` unique random queries from this workload
        """
        n = min(n, len(self._sorted_queries))
        selected_labels = random.sample(self._sorted_labels, n)
        sub_workload = {label: self.data[label] for label in selected_labels}
        return Workload(sub_workload, self._name, self._root)

    def with_prefix(self, label_prefix: L) -> Workload[L, Q]:
        """Filters the workload for all queries that have a lablel starting with a specific prefix.

        This method requires that all label instances provide a `startswith` method (as is the case for simple string
        labels). Most significantly, this means that integer-based indexing does not work with for the prefix-based filter.
        The *See Also* section provides some means to mitigate this problem.

        Parameters
        ----------
        label_prefix : LabelType
            The prefix to filter for

        Returns
        -------
        Workload[LabelType]
            All queries of this workload that have a label with a matching prefix. Queries will be sorted according to the
            natural order of their labels again.

        Raises
        ------
        ValueError
            If the prefix type does not provide a `startswith` method.

        See Also
        --------
        relabel - to change the labels into a type that provides `startswith`
        filter_by - to perform a custom prefix check for other types
        """
        if "startswith" not in dir(label_prefix):
            raise ValueError("label_prefix must have startswith() method")
        prefix_queries = {
            label: query
            for label, query in self.data.items()
            if label.startswith(label_prefix)  # type: ignore - this is guarded by the check above
        }
        return Workload(prefix_queries, name=self._name, root=self._root)

    def filter_by(self, predicate: Callable[[L, Q], bool]) -> Workload[L, Q]:
        """Provides all queries from the workload that match a specific predicate.

        Parameters
        ----------
        predicate : Callable[[LabelType, SqlQuery], bool]
            The filter condition. All queries that pass the check are included in the new workload. The filter predicate
            receives the label and the query for each query in the input

        Returns
        -------
        Workload[LabelType]
            All queries that passed the filter condition check. Queries will be sorted according to the natural order of their
            labels again.
        """
        matching_queries = {label: query for label, query in self.data.items() if predicate(label, query)}
        return Workload(matching_queries, name=self._name, root=self._root)

    def relabel[NewLabelType: Hashable](
        self, label_provider: Callable[[L, Q], NewLabelType]
    ) -> Workload[NewLabelType, Q]:
        """Constructs a new workload, leaving the queries intact but replacing the labels.

        The new workload will ordered according to the natural order of the new labels.

        Parameters
        ----------
        label_provider : Callable[[LabelType, SqlQuery], NewLabelType]
            Replacement method that maps all old labels to the new label values. This method has to provide unique labels. If
            that is not the case, conflicts will be resolved but in an arbitrary way. The replacement receives the old label
            as well as the query as input and produces the new label value.

        Returns
        -------
        Workload[NewLabelType]
            All queries of the current workload, but with new labels
        """
        relabeled_queries = {label_provider(current_label, query): query for current_label, query in self.data.items()}
        return Workload(relabeled_queries, self._name, self._root)

    def map[NQ: SqlQuery](self, transformation: Callable[[Q], NQ] | Callable, *args, **kwargs) -> Workload[L, NQ]:
        """Constructs a new workload, leaving the labels intact but replacing the queries.

        The new workload will ordered according to the natural order of the labels.

        Parameters
        ----------
        transformation : Callable[[LabelSqlQuery, ...], SqlQuery]
            Replacement method that maps all old queries to new query values. The replacement receives the old query as
            input and produces the new query value. Additional arguments can be passed to the transformation and will be
            forwarded.
        *args
            Additional positional arguments that should be forwarded to the transformation method. These are passed
            after the query argument.
        **kwargs
            Additional keyword arguments that should be forwarded to the transformation method.

        Returns
        -------
        Workload[LabelType]
            All queries of the current workload, but with new query values

        See Also
        --------
        transform : if the transformation also needs access to the query labels, or if labels should be updated as well
        """
        transformed_queries = {label: transformation(query, *args, **kwargs) for label, query in self.data.items()}
        return Workload(transformed_queries, self._name, self._root)

    def transform[NL: Hashable, NQ: SqlQuery](
        self,
        transformation: Callable[[L, Q], tuple[NL, NQ]] | Callable,
        *args,
        **kwargs,
    ) -> Workload[NL, NQ]:
        """Constructs a new workload, replacing both the labels and the queries.

        The new workload will ordered according to the natural order of the new labels.

        Parameters
        ----------
        transformation : Callable[[LabelType, SqlQuery, ...], tuple[LabelType, SqlQuery]]
            Replacement method that maps all old (label, query) pairs to new (label, query) pairs. If only queries should be
            updated according to their labels, the transformation is free to return the old label value. Additional arguments
            can be passed to the transformation and will be forwarded.
        *args
            Additional positional arguments that should be forwarded to the transformation method. These are passed after the
            label and query.
        **kwargs
            Additional keyword arguments that should be forwarded to the transformation method.

        Returns
        -------
        Workload[LabelType]
             All queries of the current workload, but with new labels and query values

        See Also
        --------
        map : if the transformation is purely query-based and does not need access to the labels
        """
        transformed_queries = [transformation(label, query, *args, **kwargs) for label, query in self.data.items()]
        return Workload(dict(transformed_queries), self._name, self._root)

    def shuffle(self) -> Workload[L, Q]:
        """Randomly changes the order of the queries in the workload.

        Returns
        -------
        Workload[LabelType]
            All queries of the current workload, but with the queries in a different order
        """
        shuffled_workload = Workload(self.data, self._name, self._root)
        shuffled_workload._sorted_labels = random.sample(self._sorted_labels, k=len(self))
        shuffled_workload._update_query_order()
        return shuffled_workload

    def ordered(self) -> Workload[L, Q]:
        """Enforces the natural ordering of the queries according to their labels.

        Returns
        -------
        Workload[LabelType]
            All queries of the current workload, but in their natural order.
        """
        return Workload(self.data, self._name, self._root)

    def _update_query_order(self) -> None:
        """Enforces that the order of the queries matches the order of the labels."""
        self._sorted_queries = [self.data[label] for label in self._sorted_labels]

    def __add__(self, other: Workload[L, Q]) -> Workload[L, Q]:
        if not isinstance(other, Workload):
            raise TypeError("Can only add workloads together")
        return Workload(
            other.data | self.data, name=self._name, root=self._root
        )  # retain own labels in case of conflict

    def __sub__(self, other: Workload[L, Q]) -> Workload[L, Q]:
        if not isinstance(other, Workload) and isinstance(other, Iterable):
            labels_to_remove = set(other)
            reduced_workload = {label: query for label, query in self.data.items() if label not in labels_to_remove}
            return Workload(reduced_workload, name=self._name, root=self._root)
        elif not isinstance(other, Workload):
            raise TypeError("Expected workload or labels to subtract")
        return Workload(
            util.dicts.difference(self.data, other.data),
            name=self._name,
            root=self._root,
        )

    def __and__(self, other: Workload[L, Q]) -> Workload[L, Q]:
        if not isinstance(other, Workload) and isinstance(other, Iterable):
            labels_to_include = set(other)
            reduced_workload = {label: query for label, query in self.data.items() if label in labels_to_include}
            return Workload(reduced_workload, name=self._name, root=self._root)
        elif not isinstance(other, Workload):
            raise TypeError("Expected workload or labels to compute union")
        return Workload(
            util.dicts.intersection(self.data, other.data),
            name=self._name,
            root=self._root,
        )

    def __or__(self, other: Mapping[L, Q]) -> Workload[L, Q]:
        merged = dict(self.data)
        merged.update(other)
        return Workload(merged, name=self._name, root=self._root)  # retain own labels in case of conflict

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self._name:
            return f"Workload: {self._name} ({len(self)} queries)"
        elif self._root:
            return f"Workload: {self._root.stem} ({len(self)} queries)"
        else:
            return f"Workload: {len(self)} queries"


def wrap_workload[T: SqlQuery](
    queries: Iterable[T], name: str = "", root: Path | str | None = None
) -> Workload[int, T]:
    """Wraps a number of queries in a workload with numerical labels."""
    return generate_workload(queries, name=name, workload_root=root)


def read_workload(
    path: str | Path,
    name: str = "",
    *,
    query_file_pattern: str = "*.sql",
    recurse_subdirectories: bool = False,
    query_label_prefix: str = "",
    file_encoding: str = "utf-8",
    bind_columns: bool = True,
    include_hints: bool = True,
    on_error: Literal["raise", "warn", "ignore"] = "raise",
    verbose: bool = False,
) -> Workload[str, SqlQuery]:
    """Loads a workload consisting of multiple different files, potentially scattered in multiple directories

    The main advantage of this method over using `Workload.read` directly is the support for recursive directory layouts: it
    can traverse subdirectories relative to the specified root and collect all workload files in a recursive manner. If
    subdirectories are used, their names will be used as prefixes to the query label, which is still inferred from the query
    file name.

    Parameters
    ----------
    path : str
        The root directory containing the workload files. Each query is expected to be stored in its own file.
    name : str, optional
        The name of the workload, by default ""
    query_file_pattern : str, optional
        A glob pattern that all query files have to match. All files that match the pattern are assumed to be valid query
        files. Defaults to ``"*.sql"``
    recurse_subdirectories : bool, optional
        Whether query files in subdirectories should be read as well. Defaults to ``False``, which emulates the behaviour of
        `Workload.read`
    query_label_prefix : str, optional
        A global prefix that should be added to all labels, no matter their placement in subdirectories. Defaults to an empty
        string.
    file_encoding : str, optional
        The encoding of the query files. All files must share a common encoding. Defaults to UTF-8
    include_hints : bool, optional
        Whether the parser should try to infer and extract hint blocks from the queries
    on_error : Literal["raise", "warn", "ignore"], optional
        How to react to parser errors. By default, the exception is propagated to the parent process.
    verbose : bool, optional
            Whether progress information should be printed.

    Returns
    -------
    Workload[str, SqlQuery]
        The workload
    """
    root = Path(path)
    queries: dict[str, SqlQuery] = {}

    if verbose:
        matching_files = list(root.glob(query_file_pattern))
        matching_files = tqdm.tqdm(matching_files, desc=root.name, unit="q")
    else:
        matching_files = root.glob(query_file_pattern)

    for query_file_path in matching_files:
        with open(query_file_path, "r", encoding=file_encoding) as query_file:
            raw_contents = query_file.readlines()
        query_contents = "\n".join([line for line in raw_contents])

        try:
            parsed_query = parser.parse_query(
                query_contents,
                include_hints=include_hints,
                bind_columns=bind_columns,
            )
        except Exception as e:
            match on_error:
                case "raise":
                    raise ValueError(f"Could not parse query from {query_file_path}", e)
                case "warn":
                    warnings.warn(f"Could not parse query {query_file_path}: {e} ({type(e)})")
                case "ignore":
                    pass

        query_label = query_file_path.stem
        queries[query_label] = parsed_query

    base_workload = Workload(queries, name=name, root=root)
    if not recurse_subdirectories:
        return base_workload

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        subdir_prefix = (
            (query_label_prefix + "/")
            if query_label_prefix and not query_label_prefix.endswith("/")
            else query_label_prefix
        )
        subdir_prefix += subdir.stem + "/"
        subdir_workload = read_workload(
            subdir,
            query_file_pattern=query_file_pattern,
            recurse_subdirectories=True,
            query_label_prefix=subdir_prefix,
            bind_columns=bind_columns,
            include_hints=include_hints,
            on_error=on_error,
            verbose=verbose,
        )
        queries |= subdir_workload.data
    return Workload(queries, name, root)


def read_batch_workload(filename: str, name: str = "", *, file_encoding: str = "utf-8") -> Workload[int, SqlQuery]:
    """Loads a workload consisting of multiple queries from a single file.

    The input file has to contain one valid SQL query per line. While empty lines are skipped, any non-SQL line will
    raise an Error.

    The workload will have numeric labels: the query in the first line will have label 1, the second one label 2 and
    so on.

    Parameters
    ----------
    filename : str
        The file to load. The extension does not matter, as long as it contains plain text and each query is placed on a single
        and separate line.
    name : str, optional
        The name of the workload. If omitted, this defaults to the file name.
    file_encoding : str, optional
        The encoding of the workload file. Defaults to UTF-8

    Returns
    -------
    Workload[int, SqlQuery]
        The workload
    """
    filepath = Path(filename)
    name = name if name else filepath.stem
    with open(filename, "r", encoding=file_encoding) as query_file:
        raw_queries = query_file.readlines()
        parsed_queries = [parser.parse_query(raw) for raw in raw_queries if raw]
        return generate_workload(parsed_queries, name=name, workload_root=filepath)


@overload
def read_csv_workload(
    filename: str | Path,
    name: str = "",
    *,
    query_column: str = "query",
    label_column: str,
    file_encoding: str = "utf-8",
    pd_args: Optional[dict] = None,
) -> Workload[str, SqlQuery]: ...


@overload
def read_csv_workload(
    filename: str | Path,
    name: str = "",
    *,
    query_column: str = "query",
    label_column: Literal[None],
    file_encoding: str = "utf-8",
    pd_args: Optional[dict] = None,
) -> Workload[int, SqlQuery]: ...


def read_csv_workload(
    filename: str | Path,
    name: str = "",
    *,
    query_column: str = "query",
    label_column: Optional[str] = None,
    file_encoding: str = "utf-8",
    pd_args: Optional[dict] = None,
) -> Workload[str, SqlQuery] | Workload[int, SqlQuery]:
    """Loads a workload consisting of queries from a CSV column.

    All queries are expected to be contained in the same column and each query is expected to be put onto its own row.

    The column containing the actual queries can be configured via the `query_column` parameter. Likewise, the
    CSV file can already provide query labels in the `label_column` column. If this parameter is omitted, labels will
    be inferred based on the row number.

    Parameters
    ----------
    filename : str | Path
        The name of the CSV file to read. The extension does not matter, as long as the file can be read by the pandas CSV
        parser. The parser can receive additional arguments via the `pd_args` parameter.
    name : str, optional
        The name of the workload. If omitted, this defaults to the file name.
    query_column : str, optional
        The CSV column that contains the workload queries. All rows of that column will be read, by default "query"
    label_column : Optional[str], optional
        The column containing the query labels. Each will receive a label from the `label_column` of the same row. If omitted,
        labels will be inferred based on the row number.
    file_encoding : str, optional
        The encoding of the CSV file. Defaults to UTF-8.
    pd_args : Optional[dict]
        Additional arguments to customize the behaviour of the `pandas.read_csv` method. They will be forwarded as-is. Consult
        the documentation of this method for more details on the allowed parameters and their functionality.

    Returns
    -------
    Workload[str, SqlQuery] | Workload[int, SqlQuery]
        The workload. It has string labels if `label_column` was provided, or numerical labels otherwise.

    See Also
    ---------
    pandas.read_csv
    """
    filepath = Path(filename)
    name = name or filepath.stem
    columns = [query_column, label_column] if label_column else [query_column]

    if pd_args:
        # Prepare the pd_args to not overwrite any of our custom parameters
        pd_args = dict(pd_args)
        pd_args.pop("usecols", None)
        pd_args.pop("converters", None)
        pd_args.pop("encoding", None)
    else:
        pd_args = {}

    workload_df = pd.read_csv(
        filepath,
        usecols=columns,
        converters={query_column: parser.parse_query},
        encoding=file_encoding,
        **pd_args,
    )  # type: ignore

    queries = workload_df[query_column].tolist()
    if label_column:
        labels = workload_df[label_column].tolist()
        label_provider = dict(zip(queries, labels))
    else:
        label_provider = None

    return generate_workload(queries, name=name, labels=label_provider, workload_root=filepath)


def generate_workload[L: Hashable, Q: SqlQuery](
    queries: Iterable[Q],
    *,
    name: str = "",
    labels: Optional[Mapping[Q, L]] = None,
    workload_root: Optional[Path | str] = None,
) -> Workload[L, Q]:
    """Wraps a number of queries in a workload object.

    The queries can receive optional labels, and will receive numerical labels according to their position in the `queries`
    iterable if no explicit labels are provided (counting from 1).

    The workload will be named according to the optional `name` parameter. If this fails, the name will be inferred from the
    optional `workload_root`. If this fails as well, an empty name will be used.

    Parameters
    ----------
    queries : Iterable[SqlQuery]
        The queries that should form the workload. This is only enumerated a single time, hence the iterable can "spent" its
        items.
    name : str, optional
        The name of the workload, by default ""
    labels : Optional[dict[SqlQuery, LabelType]], optional
        The labels of the workload queries. Defaults to ``None``, in which case numerical labels will be used. In the first
        case the label type is inferred from the dictionary values. In the second case, it will be `int`.
    workload_root : Optional[Path], optional
        The directory or file that originally contained the workload queries. Defaults to ``None`` if this is not known or not
        appropriate (e.g. for workloads that are read from a remote source)

    Returns
    -------
    Workload[LabelType]
        The workload
    """
    workload_root = Path(workload_root) if isinstance(workload_root, str) else workload_root
    name = name if name else (workload_root.stem if workload_root else "")

    if labels is not None:
        workload_contents = {labels[q]: q for q in queries}
    else:
        workload_contents = {i + 1: query for i, query in enumerate(queries)}

    return Workload(workload_contents, name, workload_root)  # type: ignore


def _assert_workload_loaded(workload: Workload, expected_dir: Path) -> None:
    """Ensures that workload queries have been read successfully. The expected directory is used for error messages."""
    if not workload:
        raise ValueError(
            f"Could not load {workload.name} workload. "
            f"Please check {expected_dir} and make sure that it contains valid query files. "
            f"If it does, please open an issue at https://github.com/Optimizer-Playground/PostBOUND to report the problem."
            f"If there are no queries in the directory, try re-running the workload access and open an issue if the problem "
            f"persists."
        )


def job(
    *,
    flavor: Literal["default", "light", "complex"] = "default",
    file_encoding: str = "utf-8",
) -> Workload[str, SelectStatement]:
    """Reads the Join Order Benchmark, with labels according to the original paper (e.g. *1a*, *21c*, etc.).


    Parameters
    ----------
    flavor : Literal["default", "light", "complex"], optional
        The flavor of the JOB benchmark to load. The default flavor is the original JOB benchmark. Use "light" or "complex" to
        load the respective variants.
    file_encoding : str, optional
        The encoding of the query files, by default UTF-8.

    Returns
    -------
    Workload[str]
        The workload

    See Also
    --------
    job_light
    job_complex

    References
    ----------

    .. Viktor Leis et al.: "How Good Are Query Optimizers, Really?" (Proc. VLDB Endow. 9, 3 (2015))
    """
    if flavor == "light":
        return job_light(file_encoding=file_encoding)
    elif flavor == "complex":
        return job_complex(file_encoding=file_encoding)

    workload_dir = _fetch_workload("JOB")
    # JOB only uses aliases column references, so no need for explicit binding
    job_workload = Workload.read(workload_dir, name="JOB", file_encoding=file_encoding, bind_columns=False)
    _assert_workload_loaded(job_workload, workload_dir)
    return cast(Workload[str, SelectStatement], job_workload)


def job_light(*, file_encoding: str = "utf-8") -> Workload[str, SelectStatement]:
    """Reads the JOB-light benchmark, with numeric query labels (q-1, q-2, q-3, ...).

    Parameters
    ----------
    file_encoding : str, optional
        The encoding of the query files, by default UTF-8.

    Returns
    -------
    Workload[str]
        The workload

    References
    ----------

    .. Andreas Kipf et al.: "Learned Cardinalities: Estimating Correlated Joins with Deep Learning" (CIDR'2019)
    """
    workload_dir = _fetch_workload("job-light")
    # JOB-light only uses aliases column references, so no need for explicit binding
    job_light_workload = Workload.read(workload_dir, name="JOB-light", file_encoding=file_encoding, bind_columns=False)
    _assert_workload_loaded(job_light_workload, workload_dir)
    return cast(Workload[str, SelectStatement], job_light_workload)


def job_complex(*, file_encoding: str = "utf-8") -> Workload[str, SelectStatement]:
    """Reads the JOB-complex benchmark, with numeric query labels (q-1, q-2, q-3, ...).

    Parameters
    ----------
    file_encoding : str, optional
        The encoding of the query files, by default "utf-8".

    Returns
    -------
    Workload[str]
        The workload

    References
    ----------

    .. Johannes Wehrstein et al.: "JOB-Complex: A Challenging Benchmark for Traditional & Learned Query Optimization"
       (AIDB'2025)
    """
    workload_dir = _fetch_workload("job-complex")
    # JOB-complex only uses aliases column references, so no need for explicit binding
    job_complex_workload = Workload.read(
        workload_dir,
        name="JOB-complex",
        file_encoding=file_encoding,
        bind_columns=False,
    )
    _assert_workload_loaded(job_complex_workload, workload_dir)
    return cast(Workload[str, SelectStatement], job_complex_workload)


def ssb(*, file_encoding: str = "utf-8", bind_columns: Optional[bool] = None) -> Workload[str, SelectStatement]:
    """Reads the Star Schema Benchmark, with labels according to the original data (e.g. *q1-1*, *q3-2*, etc.).

    Parameters
    ----------
    file_encoding : str, optional
        The encoding of the query files, by default UTF-8.
    bind_columns : Optional[bool], optional
        Whether all columns in the queries should be bound to their respective tables. Since SSB does not use qualified
        columns names, these must be inferred from the database schema. Thus, an active database connection is required to
        bind the columns. By default, binding is attempted if there is an active database connection.

    Returns
    -------
    Workload[str]
        The workload

    References
    ----------

    .. Patrick E. O'Neil et al.: "The Star Schema Benchmark and Augmented Fact Table Indexing." (TPCTC'2009)
    """
    bind_columns = bind_columns if bind_columns is not None else not DatabasePool.get_instance().empty()
    workload_dir = _fetch_workload("ssb")
    ssb_workload = Workload.read(workload_dir, name="SSB", file_encoding=file_encoding, bind_columns=bind_columns)
    _assert_workload_loaded(ssb_workload, workload_dir)
    return cast(Workload[str, SelectStatement], ssb_workload)


def stack(
    *,
    file_encoding: str = "utf-8",
    bind_columns: Optional[bool] = None,
) -> Workload[str, SelectStatement]:
    """Reads the Stack Benchmark, as shipped with the PostBOUND repository.

    Most queries use semi-numeric labels consisting of the context and the query number, e.g., *q1/q1-001*. However, some
    queries have completely random names, such as *q16/fc8f97968b9fce81df4011c8175eada15541abe0*. Still, all queries are
    grouped into one of 16 contexts.

    Parameters
    ----------
    file_encoding : str, optional
        The encoding of the query files, by default UTF-8.
    bind_columns : Optional[bool], optional
        Whether all columns in the queries should be bound to their respective tables. Since Stack does not use qualified
        columns names in some queries, these must be inferred from the database schema. Thus, an active database connection is
        required to bind the columns. By default, binding is attempted if there is an active database connection.

    Returns
    -------
    Workload[str]
        The workload.

    References
    ----------

    .. Ryan Marcus et al.: "Bao: Making Learned Query Optimization Practical." (SIGMOD'2021)
    """
    bind_columns = bind_columns if bind_columns is not None else not DatabasePool.get_instance().empty()
    workload_dir = _fetch_workload("stack")

    stack_workload = read_workload(
        workload_dir,
        "Stack",
        recurse_subdirectories=True,
        file_encoding=file_encoding,
        bind_columns=bind_columns,
    )
    _assert_workload_loaded(stack_workload, workload_dir)
    return cast(Workload[str, SelectStatement], stack_workload)


def stats(*, file_encoding: str = "utf-8") -> Workload[str, SelectStatement]:
    """Reads the Stats Benchmark, with semi-numeric query labels (e.g. *q-1*, *q-2*, etc.).

    Parameters
    ----------
    file_encoding : str, optional
        The encoding of the query files, by default UTF-8.

    Returns
    -------
    Workload[str]
        The workload

    References
    ----------

    .. Yuxing Han et al.: Cardinality Estimation in DBMS: A Comprehensive Benchmark Evaluation (Proc. VLDB Endow. 15, 4 (2022))
    """
    workload_dir = _fetch_workload("stats")
    stats_workload = Workload.read(workload_dir, name="Stats", file_encoding=file_encoding, bind_columns=False)
    _assert_workload_loaded(stats_workload, workload_dir)
    return cast(Workload[str, SelectStatement], stats_workload)


def fetch_workload(name: str, *, file_encoding: str = "utf-8") -> Workload[str, SelectStatement]:
    """Utility method to fetch a pre-defined workload by name."""
    match name:
        case "job" | "imdb":
            return job(flavor="default", file_encoding=file_encoding)
        case "job-light" | "joblight":
            return job(flavor="light", file_encoding=file_encoding)
        case "job-complex" | "jobcomplex":
            return job(flavor="complex", file_encoding=file_encoding)
        case "stats":
            return stats(file_encoding=file_encoding)
        case "stack":
            return stack(file_encoding=file_encoding)
        case "ssb":
            return ssb(file_encoding=file_encoding)
        case _:
            raise ValueError(
                f"No pre-defined workload registered for key '{name}'. "
                "Available workloads are: 'job', 'job-light', 'job-complex', "
                "'stats', 'stack', 'ssb'"
            )
