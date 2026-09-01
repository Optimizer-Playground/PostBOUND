from __future__ import annotations

import collections
import copy
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any, Literal, Optional

from . import util
from ._core import (
    Cardinality,
    ColumnReference,
    Cost,
    JoinOperator,
    PhysicalOperator,
    ScanOperator,
    TableReference,
)
from .qal import AbstractPredicate, ColumnExpression, SqlExpression
from .util import StateError, jsondict

JoinDirection = Literal["inner", "outer"]


class SortKey:
    """Sort keys describe how the tuples in a relation are sorted.

    Each sort key contains a set of columns that describe the equivalence class of the sort key, i.e. the column values in each
    row are all equal to one another. Therefore, the relation can be treated as being sorted by any of them.

    Most commonly, relations will only be sorted by a single column (which can be checked by calling *len()* on the sort key,
    or by checking the `equivalence_class` directly). In this case, the `column` property can be used to retrieve the
    corresponding expression that forms the column.

    To check, whether two sort keys are equivalent, the `is_compatible_with` method can be used. For more idiomatic access,
    ``column in sort_key`` is also supported.

    To create a new equivalence class, the `for_equivalence_class(columns)` method is available to create a new sort key from
    scratch. To combine two existing sort keys, `merge_with` can be used.

    Parameters
    ----------
    columns : Iterable[SqlExpression]
        The column(s) that is used to sort the tuples. This will usually contain plain column references (`ColumnExpression`),
        but can also use more complex expressions.
    ascending : bool
        Whether the sorting is ascending or descending. Defaults to ascending.
    """

    @staticmethod
    def of(column: SqlExpression | ColumnReference, *, ascending: bool = True) -> SortKey:
        """Creates a new sort key for a single column.

        Parameters
        ----------
        column : SqlExpression | ColumnReference
            The column that is used to sort the tuples. Can be a plain column reference, which will be wrapped by a
            `ColumnExpression` automatically.
        ascending : bool, optional
            Whether the sorting is ascending or descending. Defaults to ascending.

        Returns
        -------
        SortKey
            The sort key with an equivalence class for the single column.
        """
        if isinstance(column, ColumnReference):
            column = ColumnExpression(column)
        return SortKey([column], ascending=ascending)

    @staticmethod
    def for_equivalence_class(members: Iterable[SqlExpression | ColumnReference], *, ascending: bool = True) -> SortKey:
        """Creates a new sort key for an equivalence class of columns.

        This is just a more expressive alias for calling the constructor directly. This method assumes that the values for
        all columns in the equivalence class are equal to one another. The client is responsible for ensuring and checking
        that this is actually the case.

        Parameters
        ----------
        members : Iterable[SqlExpression  |  ColumnReference]
            The columns that describe the sorting of the relation. This can contain just a single item, in which case the
            method is pretty much the same as `of`. Any passed `ColumnReference` will be wrapped in a `ColumnExpression`.
        ascending : bool, optional
            Whether the sorting is ascending or descending. Defaults to ascending.

        Returns
        -------
        SortKey
            The sort key with an equivalence class for the columns.
        """
        members = [ColumnExpression(mem) if isinstance(mem, ColumnReference) else mem for mem in members]
        return SortKey(members, ascending=ascending)

    def __init__(self, columns: Iterable[SqlExpression], *, ascending: bool = True) -> None:
        self._members = frozenset(columns)
        if not self._members:
            raise ValueError("Sort key must contain at least one column")
        self._ascending = ascending

    __match_args___ = ("equivalence_class", "ascending")

    @property
    def column(self) -> SqlExpression:
        """For single-column sort keys, get this column."""
        if len(self._members) != 1:
            raise StateError("Sort key is not a single column reference")
        return next(iter(self._members))

    @property
    def equivalence_class(self) -> frozenset[SqlExpression]:
        """Get all columns that are part of the equivalence class. This will be 1 or more columns."""
        return self._members

    @property
    def ascending(self) -> bool:
        """Get the sort direction of this key."""
        return self._ascending

    def is_compatible_with(self, other: SortKey | ColumnReference) -> bool:
        """Checks, whether two keys are sorted the same way.

        For single column references, this essentially checks whether the column is part of the key's equivalence class.
        """
        if isinstance(other, ColumnReference):
            return other in self._members

        if self.ascending != other.ascending:
            return False
        return len(self._members & other._members) > 0

    def merge_with(self, other: SortKey) -> SortKey:
        """Merges the equivalence classes of two sort keys."""
        if self.ascending != other.ascending:
            raise ValueError("Cannot merge sort keys with different sort orders")
        return SortKey(self._members | other._members, ascending=self.ascending)

    def __json__(self) -> jsondict:
        return {"equivalence_class": self._members, "ascending": self._ascending}

    def __len__(self) -> int:
        return len(self._members)

    def __contains__(self, item: object) -> bool:
        return self.is_compatible_with(item) if isinstance(item, (ColumnReference, SortKey)) else False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._members == other._members and self._ascending == other._ascending

    def __hash__(self) -> int:
        return hash((self._members, self._ascending))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        suffix = "" if self.ascending else " DESC"
        if len(self._members) == 1:
            member = str(self.column)
        else:
            members = ", ".join(str(m) for m in self._members)
            member = f"{{{members}}}"
        return f"{member}{suffix}"


class PlanParams:
    """Plan parameters contain additional "structural" metadata about the operators in a query plan.

    This information is mostly concerned with how the operator should function, e.g. which table it should scan, or which index
    to use or how the tuples will be sorted.

    In addition to the pre-defined attributes, users can attach arbitrary metadata using a dict-like access into the
    parameters, e.g. ``params["custom"] = 42``.

    Parameters
    ----------
    base_table : Optional[TableReference], optional
        For scan nodes, this is the table being scanned. For all other nodes this is should be *None*.
    filter_predicate : Optional[AbstractPredicate], optional
        An arbitrary predicate to restrict the allowed tuples in the output of a relation. This should be mostly used for
        join nodes and scans.
    sort_keys : Optional[Sequence[SortKey]], optional
        How the tuples in a the output of a relation are sorted. Absence of a specific sort order can be indicated either
        through an empty list or by setting this parameter to *None*. In this case, tuples are assumed to be in some random
        order.
    parallel_workers : Optional[int], optional
        The number of parallel workers that should be used to execute the operator. The underlying processing model assumes
        that there exists some sort of main operator process which spawns additional worker processes. The worker processes
        will compute the output relation together with the main process. Hence, if some relation should be processed by two
        processes in parallel, the proper value for this parameter would be 1 (the main process and one additional worker).
        It is up to the actual execution engine to decide whether a lower number of workers has to be used.
    index : Optional[str], optional
        The name of the index that should be used to scan the table. This is only relevant for scan nodes and should be
        *None* for all other nodes.
    lookup_key : Optional[SqlExpression], optional
        The expression that is used to lookup tuples in some indexing structure. For scans, this could actually be the physical
        index. For intermediate operators such as hash tables or memoize nodes, this could be the expression that is used to
        build the table or to structure the memo.
    **kwargs
        Additional metadata that should be attached to the plan parameters.
    """

    def __init__(
        self,
        *,
        base_table: Optional[TableReference] = None,
        filter_predicate: Optional[AbstractPredicate] = None,
        sort_keys: Optional[Sequence[SortKey]] = None,
        parallel_workers: Optional[int] = None,
        index: Optional[str] = None,
        lookup_key: Optional[SqlExpression] = None,
        **kwargs,
    ) -> None:
        self._params: dict[str, Any] = {
            "base_table": base_table,
            "filter_predicate": filter_predicate,
            "sort_keys": tuple(sort_keys) if sort_keys else tuple(),
            "parallel_workers": parallel_workers if parallel_workers else 0,
            "index": index if index else "",
            "lookup_key": lookup_key if lookup_key else None,
            **kwargs,
        }

    @property
    def base_table(self) -> Optional[TableReference]:
        """Get the base table that is being scanned. For non-scan nodes, this is *None*."""
        return self._params["base_table"]

    @property
    def filter_predicate(self) -> Optional[AbstractPredicate]:
        """Get the filter predicate that is used to restrict the tuples in the output of a relation.

        For join nodes this would be the join condition and for scan nodes this would be the filter conditions from the
        WHERE clause. However, if the optimizer decides to delay the evaluation of some filter, or some filters need to be
        evaluated multiple times (e.g. recheck conditions in Postgres), this predicate can be more complex.
        """
        return self._params["filter_predicate"]

    @property
    def sort_keys(self) -> Sequence[SortKey]:
        """Get the sort keys describing the ordering of tuples in the output relation.

        Absence of a specific sort order is indicated by an empty sequence.
        """
        return self._params["sort_keys"]

    @property
    def parallel_workers(self) -> int:
        """Get the number of parallel workers that should be used to execute the operator.

        The underlying processing model assumes that there exists some sort of main operator process which spawns additional
        worker processes. The worker processes will compute the output relation together with the main process. Hence, if some
        relation should be processed by two processes in parallel, the proper value for this parameter would be 1 (the main
        process and one additional worker).

        It is up to the actual execution engine to decide whether a lower number of workers has to be used.

        Absence of parallelism is indicated by 0.
        """
        return self._params["parallel_workers"]

    @parallel_workers.setter
    def parallel_workers(self, workers: int) -> None:
        self._params["parallel_workers"] = workers

    @property
    def index(self) -> str:
        """Get the name of the index that should be used to scan the table.

        Absence of an index is indicated by an empty string.
        """
        return self._params["index"]

    @property
    def lookup_key(self) -> Optional[SqlExpression]:
        """Get the expression that is used to lookup tuples in some indexing structure.

        For scans, this could actually be the physical index. In this case, the lookup expression should be the one that is
        used to build the index, e.g., the primary key column. For intermediate operators such as hash tables or memoize nodes,
        this could be the expression that is used to build the table or to structure the memo.
        """
        return self._params["lookup_key"]

    def tables(self) -> set[TableReference]:
        """Provide all tables that are referenced at some point in the plan parameters.

        This includes only the well-defined properties available to all parameterizations, i.e. `base_table` and
        `filter_predicate`. If users decide to store additional metadata with table information in the parameters, these are
        not retained here.

        Returns
        -------
        set[TableReference]
            The tables
        """
        tables = set()
        if self.base_table:
            tables.add(self.base_table)
        if self.filter_predicate:
            tables |= self.filter_predicate.tables()
        if self.lookup_key:
            tables |= self.lookup_key.tables()
        return tables

    def columns(self) -> set[ColumnReference]:
        """Provides all columns that are referenced at some point in the plan parameters.

        This includes only the well-defined properties available to all parameterizations, i.e. just the `filter_predicate`. If
        users decide to store additional metadata with column information in the parameters, these are not retained here.

        Returns
        -------
        set[ColumnReference]
            The columns
        """
        return self.filter_predicate.columns() if self.filter_predicate else set()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves the value of a specific key from the parameters.

        This is similar to the *dict.get* method. An important distinction is that we never raise an error if there is no
        parameter with the given key. Instead, we return the default value, which is *None* by default.

        Parameters
        ----------
        key : str
            The parameter name
        default : Any, optional
            The default value to return if the parameter is not found. Defaults to *None*.

        Returns
        -------
        Any
            The parameter value if it exists, otherwise the default value.
        """
        value = self._params.get(key, default)
        if isinstance(value, float) and math.isnan(value):
            return default
        return value

    def items(self) -> Iterable[tuple[str, Any]]:
        """Provides all metadata that is currently stored in the parameters as key-value pairs, similar to *dict.items*"""
        return self._params.items()

    def update(self, items: Mapping[str, Any]) -> PlanParams:
        """Creates a new plan parameters object with updated values."""
        updated_items = dict(self._params)
        updated_items.update(items)
        return PlanParams(**updated_items)

    def clone(self, *, deep: bool = False) -> PlanParams:
        """Creates a copy of the current plan parameters.

        Parameters
        ----------
        deep : bool, optional
            Whether to create a deep copy of all parameters. Defaults to *False*.

        Returns
        -------
        PlanParams
            The copied parameters.
        """
        return self.__deepcopy__({}) if deep else self.__copy__()

    def __json__(self) -> jsondict:
        return self._params

    def __copy__(self) -> PlanParams:
        return PlanParams(**self._params)

    def __deepcopy__(self, memo: dict[int, object] = {}) -> PlanParams:
        params = copy.deepcopy(self._params, memo)
        return PlanParams(**params)

    def __contains__(self, key: object) -> bool:
        params = object.__getattribute__(self, "_params")
        return key in params

    def __getattribute__(self, name: str) -> Any:
        params = object.__getattribute__(self, "_params")
        if name == "_params":
            return params
        if name in params:
            return params[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value) -> None:
        if name == "_params":
            return object.__setattr__(self, name, value)
        params = object.__getattribute__(self, "_params")
        params[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._params[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._params[key] = value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self._params)


class PlanEstimates:
    """Plan estimates provide the optimizer's view on a specific (sub-)plan.

    This includes the estimated cardinality and cost of the plan. The cardinality is the number of tuples that are expected to
    be produced by the operator, while the cost is a measure of the resources that are consumed by the operator.
    Costs do not have a specific unit and it is the user's obligation to ensure that they are used in a sound way. Most
    importantly, this means that only costs from the same source should be compared since most database systems interpret costs
    in a different way.

    In addition to the pre-defined attributes, users can attach arbitrary metadata using a dict-like access into the
    parameters, e.g. ``estimates["custom"] = 42``.

    Parameters
    ----------
    cardinality : Cardinality, optional
        The estimated number of tuples that are produced by the operator. If no estimate is available, *NaN* can be used.
    cost : Cost, optional
        The approximate amount of abstract "work" that needs to be done to compute the result set of the operator. If no
        estimate is available, *NaN* can be used.
    **kwargs
        Additional metadata that should be attached to the plan estimates.

    Notes
    -----
    In case of parallel execution, all measures should be thought of "meaningful totals", i.e. the cardinality
    numbers are the total number of tuples produced by all workers. The execution time should denote the wall time, it
    took to execute the entire operator (which just happened to include parallel processing), **not** an average of the
    worker execution time or some other measure.
    """

    def __init__(
        self,
        *,
        cardinality: Cardinality | float | None = Cardinality.unknown(),
        cost: Cost = math.nan,
        **kwargs,
    ) -> None:
        cardinality = Cardinality.of(cardinality)
        self._params: dict[str, Any] = {"cardinality": cardinality, "cost": cost, **kwargs}

    @property
    def cardinality(self) -> Cardinality:
        """Get the estimated cardinality of the operator. Can be *NaN* if no estimate is available."""
        return self._params["cardinality"]

    @property
    def cost(self) -> Cost:
        """Get the estimated cost of the operator. Can be *NaN* if no estimate is available."""
        return self._params["cost"]

    @property
    def additional_estimates(self) -> dict[str, Any]:
        """Get all additional estimates that are not part of the well-defined attributes."""
        reserved_keys = ["cardinality", "cost"]
        return {k: v for k, v in self._params.items() if k not in reserved_keys}

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves the value of a specific key from the estimates.

        This is similar to the *dict.get* method. An important distinction is that we never raise an error if there is no
        parameter with the given key. Instead, we return the default value, which is *None* by default.

        Parameters
        ----------
        key : str
            The parameter name
        default : Any, optional
            The default value to return if the parameter is not found. Defaults to *None*.

        Returns
        -------
        Any
            The parameter value if it exists, otherwise the default value.
        """
        value = self._params.get(key, default)
        if isinstance(value, float) and math.isnan(value):
            return default
        return value

    def items(self) -> Iterable[tuple[str, Any]]:
        """Provides all estimates as key-value pairs, similar to the *dict.items* method."""
        return self._params.items()

    def update(self, items: Mapping[str, Any]) -> PlanEstimates:
        """Creates a new plan estimates object with updated values."""
        updated_items = dict(self._params)
        updated_items.update(items)
        return PlanEstimates(**updated_items)

    def clone(self, *, deep: bool = False) -> PlanEstimates:
        """Creates a copy of the current plan estimates.

        Parameters
        ----------
        deep : bool, optional
            Whether to create a deep copy of all estimates. Defaults to *False*.

        Returns
        -------
        PlanEstimates
            The copied estimates.
        """
        return self.__deepcopy__({}) if deep else self.__copy__()

    def __json__(self) -> jsondict:
        return self._params

    def __copy__(self) -> PlanEstimates:
        return PlanEstimates(**self._params)

    def __deepcopy__(self, memo: dict[int, object] = {}) -> PlanEstimates:
        params = copy.deepcopy(self._params, memo)
        return PlanEstimates(**params)

    def __contains__(self, key: object) -> bool:
        params = object.__getattribute__(self, "_params")
        return key in params

    def __getattribute__(self, name: str) -> Any:
        params = object.__getattribute__(self, "_params")
        if name == "_params":
            return params
        if name in params:
            return params[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value) -> None:
        if name == "_params":
            return object.__setattr__(self, name, value)
        params = object.__getattribute__(self, "_params")
        params[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._params[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._params[key] = value

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self._params)


class PlanMeasures:
    """Plan measures provide actual execution statistics of a specific (sub-)plan.

    Typically, this includes the actual cardinality of the result set as well as the execution time of the operator.
    Additionally, information about cache hits and misses for the shared buffer can be provided.

    Other than the pre-defined attributes, users can attach arbitrary metadata using a dict-like access into the parameters,
    e.g. ``measures["custom"] = 42``.

    Parameters
    ----------
    cardinality : Cardinality, optional
        The actual number of tuples that are produced by the operator. If no measurement is available, *NaN* can be used.
    execution_time : float, optional
        The total time (in seconds) that was spent to compute the result set of the operator. If no measurement is available,
        *NaN* can be used.
    cache_hits : Optional[int], optional
        The number of page reads that were satisfied by the shared buffer. If no measurement is available, *None* can be
        used.
    cache_misses : Optional[int], optional
        The number of page reads that had to be delegated to the disk and could not be satisfied by the shared buffer. If no
        measurement is available, *None* can be used.
    **kwargs
        Additional metadata that should be attached to the plan measures.

    Notes
    -----
    In case of parallel execution, all measures should be thought of "meaningful totals", i.e. the cardinality
    numbers are the total number of tuples produced by all workers. The execution time should denote the wall time, it
    took to execute the entire operator (which just happened to include parallel processing), **not** an average of the
    worker execution time or some other measure.
    """

    def __init__(
        self,
        *,
        cardinality: Cardinality | float | None = Cardinality.unknown(),
        execution_time: float = math.nan,
        cache_hits: Optional[int] = None,
        cache_misses: Optional[int] = None,
        **kwargs,
    ) -> None:
        cardinality = Cardinality.of(cardinality)
        self._params: dict[str, Any] = {
            "cardinality": cardinality,
            "execution_time": execution_time,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            **kwargs,
        }

    @property
    def cardinality(self) -> Cardinality:
        """Get the actual cardinality of the operator. Can be *NaN* if no measurement is available."""
        return self._params["cardinality"]

    @property
    def execution_time(self) -> float:
        """Get the actual execution time of the operator. Can be *NaN* if no measurement is available."""
        return self._params["execution_time"]

    @property
    def cache_hits(self) -> Optional[int]:
        """Get the number of page reads that were satisfied by the shared buffer.

        If no measurement is available, *None* is returned.
        """
        return self._params["cache_hits"]

    @property
    def cache_misses(self) -> Optional[int]:
        """Get the number of page reads that had to be delegated to the disk and could not be satisfied by the shared buffer.

        If no measurement is available, *None* is returned.
        """
        return self._params["cache_misses"]

    @property
    def additional_measures(self) -> dict[str, Any]:
        """Get all additional estimates that are not part of the well-defined attributes."""
        reserved_keys = ["cardinality", "execution_time", "cache_hits", "cache_misses"]
        return {k: v for k, v in self._params.items() if k not in reserved_keys}

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves the value of a specific key from the measures.

        This is similar to the *dict.get* method. An important distinction is that we never raise an error if there is no
        parameter with the given key. Instead, we return the default value, which is *None* by default.

        Parameters
        ----------
        key : str
            The parameter name
        default : Any, optional
            The default value to return if the parameter is not found. Defaults to *None*.

        Returns
        -------
        Any
            The parameter value if it exists, otherwise the default value.
        """
        value = self._params.get(key, default)
        if isinstance(value, float) and math.isnan(value):
            return default
        return value

    def items(self) -> Iterable[tuple[str, Any]]:
        """Provides all measures as key-value pairs, similar to the *dict.items* method."""
        return self._params.items()

    def update(self, items: Mapping[str, Any]) -> PlanMeasures:
        """Creates a new plan measures object with updated values."""
        updated_items = dict(self._params)
        updated_items.update(items)
        return PlanMeasures(**updated_items)

    def clone(self, *, deep: bool = False) -> PlanMeasures:
        """Creates a copy of the current plan measures.

        Parameters
        ----------
        deep : bool, optional
            Whether to create a deep copy of all measures. Defaults to *False*.

        Returns
        -------
        PlanMeasures
            The copied measures.
        """
        return self.__deepcopy__({}) if deep else self.__copy__()

    def __json__(self) -> jsondict:
        return self._params

    def __copy__(self) -> PlanMeasures:
        return PlanMeasures(**self._params)

    def __deepcopy__(self, memo: dict[int, object] = {}) -> PlanMeasures:
        params = copy.deepcopy(self._params, memo)
        return PlanMeasures(**params)

    def __contains__(self, key: object) -> bool:
        params = object.__getattribute__(self, "_params")
        return key in params

    def __getattribute__(self, name: str) -> Any:
        params = object.__getattribute__(self, "_params")
        if name == "_params":
            return params
        if name in params:
            return params[name]
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value) -> None:
        if name == "_params":
            return object.__setattr__(self, name, value)
        params = object.__getattribute__(self, "_params")
        params[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._params[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._params[key] = value

    def __bool__(self) -> bool:
        return any(not math.isnan(v) if isinstance(v, Number) else (v is not None) for v in self._params.values())

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self._params)


@dataclass(frozen=True)
class Subplan:
    """Subplans are used to model subqueries whose results are used while processing another operator in the main query.

    A typical example are correlated/dependent subqueries that are used in some predicate and need to be evaluated for each
    tuple of the outer relation (unless some algebraic optimization has been applied beforehand).

    Attributes
    ----------
    root : QueryPlan
        The root operator of the subplan
    target_name : str
        The name of the target table that the subplan should produce
    """

    root: QueryPlan
    target_name: str = ""

    def tables(self) -> set[TableReference]:
        """Provide all tables that are referenced at some point in the subplan.

        Returns
        -------
        set[TableReference]
            The tables. This set includes the target table that the subplan produces as a virtual table.
        """
        if not self.target_name:
            return self.root.tables()
        target_table = TableReference.create_virtual(self.target_name)
        return self.root.tables() | {target_table}

    def clone(self, *, deep: bool = False) -> Subplan:
        """Creates a copy of the current subplan.

        Parameters
        ----------
        deep : bool, optional
            Whether to create a deep copy of all contained plans. Defaults to *False*.

        Returns
        -------
        Subplan
            The copied subplan.
        """
        return self.__deepcopy__({}) if deep else self.__copy__()

    def __json__(self) -> jsondict:
        return {"root": self.root, "target_name": self.target_name}

    def __copy__(self) -> Subplan:
        return Subplan(self.root.clone(deep=False), self.target_name)

    def __deepcopy__(self, memo: dict[int, object] = {}) -> Subplan:
        return Subplan(self.root.clone(deep=True), self.target_name)


class QueryPlan:
    """Models the structure of a query execution plan (QEP).

    Query plans are constructed as a tree of operators. Each operator represents an entire query plan by itself. Hence, we
    use the *QueryPlan* to refer to the actual nodes in a hierarchical structure. Each node has a potentially large amount of
    metadata attached to it, e.g. regarding the table being scanned for scan nodes, the estimated cost of the operator or the
    actual cardinality of the result set. The different types of metadata are structured into three separate classes:

    - `PlanParams` contain all structural metadata about the operator, e.g. the table being scanned or the filter predicate.
    - `PlanEstimates` contain the optimizer's view on the operator, e.g. the estimated cardinality and cost.
    - `PlanMeasures` contain the actual execution statistics of the operator, e.g. the actual cardinality and execution time.

    Users are free to attach additional metadata to each of the containers to support there specific use-cases. However, these
    additional fields are typically not considered by the standard methods available on query plans. For example, if users
    store additional tables in the node, these are not considered in the `tables` method.

    Each query plan can contain an arbitrary number of child nodes. This is true even for scans, to accomodate bitmap scans
    that combine an arbitrary amount of index lookups with a final scan. If just a single child is present, it can be set more
    expressively using the `input_node` property.

    PostBOUND uses QEPs in two different ways: first, they can be used as the output of the optimization process (i.e. the
    optimization pipelines), being constructed by the different optimization stages. Second, they can also be extracted from
    an actual database system to encode the QEP that this system used to execute a specific query. This dichotomy leads to
    different granularities of query plans: actual database systems often have much more detailed QEPs. For example, Postgres
    represents a hash join as a hash join operator, whose inner child is a hash operator that constructs the hash table.
    The optimizer stages will typically not worry about such fine-grained details and simply demand a join to be executed as
    a hash join. To mitigate these issues, the query plans can be normalized by using the `canonical` method. This method
    removes all unnecessary details and only retains the join and scan operators.

    When constructing a query plan, the metadata can be provided in two ways: either as instances of the corresponding metadata
    objects, or explicitly as keyword arguments to enable a more convenient usage. Notice however, that these two ways cannot
    be mixed: either all metadata of a specific type is provided as wrapper instance, or all metadata is provided as keyword
    arguments. Mixing is only allowed across different metadata types, e.g. providing the estimates as a `PlanEstimates` object
    and the measurements as keyword arguments.

    In addition to the pre-defined metadata types, you can also add additional metadata as part of the *kwargs*. These will
    be added to the plan parameters (using the same mixing rules as the pre-defined types).
    Each query plan provides dict-like access to the plan parameters, estimates and measures, e.g. ``plan["custom"] = 42``,
    ``plan.get("custom", default)``, or ``"custom" in plan``.

    Query plans provide rather extensive support methods to check their shape (e.g. `is_linear()` or `is_bushy()`), to aid with
    traversal (e.g. `find_first_node()` or `find_all_nodes()`) or to extract specific information (e.g. `tables()` or
    `qerror()`).

    To convert between different optimization artifacts, a number of methods are available. For example, `to_query_plan` can
    be used to construct a query plan from a join order and a set of operators. Likewise, `explode_query_plan` converts the
    query plan back into join order, operators and parameters.

    Query plans support *len()*  (providing the plan depth without subplans) and *iter()* (providing all contained nodes
    including subplans).

    Parameters
    ----------
    node_type : str | PhysicalOperator
        The name of the operator. If this is supplied as a physical operator, the name is inferred from it.
    operator : Optional[PhysicalOperator], optional
        The actual operator that is used to compute the result set. This can be empty if there is no specific operator
        corresponding to the current node (e.g. for transient hash tables).
    children : Optional[QueryPlan | Iterable[QueryPlan]], optional
        The input nodes of the current operator. For nodes without an input (e.g. most scans), this can simply be *None* or
        an empty list. Nodes with exactly one input node (e.g. most aggregations) can supply their input either directly as
        a plan object, or as a singleton list. Nodes with two input nodes (e.g. joins) should supply them as an ordered
        iterable with the outer child first.
    plan_params : Optional[PlanParams], optional
        Structural metadata (e.g. parallel workers or accessed indexes) of the operator. If this is provided, no other
        plan parameters can be supplied as keyword arguments, including kwargs.
    subplan : Optional[Subplan], optional
        A subquery that has to be executed as part of this node. If this is provided, no other subplan components can be
        supplied as keyword arguments.
    estimates : Optional[PlanEstimates], optional
        The optimizer's view on the operator (e.g. estimated cardinality and cost). If this is provided, no other estimates
        can be supplied as keyword arguments.
    measures : Optional[PlanMeasures], optional
        The actual execution statistics of the operator (e.g. actual cardinality and execution time). If this is provided, no
        other measures can be supplied as keyword arguments.
    base_table : Optional[TableReference], optional
        The table that is being scanned. This is only relevant for scan nodes and should be *None* for all other nodes.
        If this argument is used, no other plan parameters can be supplied in the `plan_params` argument.
    filter_predicate : Optional[AbstractPredicate], optional
        An arbitrary predicate to restrict the allowed tuples in the output of a relation. This should be mostly used for
        join nodes and scans. If this argument is used, no other plan parameters can be supplied in the `plan_params` argument.
    parallel_workers : Optional[int], optional
        The number of parallel workers that should be used to execute the operator. If this argument is used, no other plan
        parameters can be supplied in the `plan_params` argument.
    index : Optional[str], optional
        The name of the index that should be used to scan the table. This is mostly relevant for scan nodes and should be
        *None* for all other nodes. If this argument is used, no other plan parameters can be supplied in the `plan_params`
        argument.
    lookup_key : Optional[SqlExpression], optional
        The expression that is used to lookup tuples in some indexing structure. For scans, this could actually be the
        physical index. For intermediate operators such as hash tables or memoize nodes, this could be the expression that is
        used to build the table or to structure the memo. If this argument is used, no other plan parameters can be supplied
        in the `plan_params` argument.
    sort_keys : Optional[Sequence[SortKey]], optional
        How the tuples in a the output of a relation are sorted. Absence of a specific sort order can be indicated either
        through an empty list or by setting this parameter to *None*. In this case, tuples are assumed to be in some random
        order. If this argument is used, no other plan parameters can be supplied in the `plan_params` argument.
    estimated_cardinality : Cardinality, optional
        The estimated number of tuples that are produced by the operator. If no estimate is available, *NaN* can be used.
        If this argument is used, no other estimates can be supplied in the `estimates` argument.
    estimated_cost : Cost, optional
        The approximate amount of abstract "work" that needs to be done to compute the result set of the operator. If no
        estimate is available, *NaN* can be used. If this argument is used, no other estimates can be supplied in the
        `estimates` argument.
    actual_cardinality : Cardinality, optional
        The actual number of tuples that are produced by the operator. If no measurement is available, *NaN* can be used.
        If this argument is used, no other measures can be supplied in the `measures` argument.
    execution_time : float, optional
        The total time (in seconds) that was spent to compute the result set of the operator. If no measurement is available,
        *NaN* can be used. If this argument is used, no other measures can be supplied in the `measures` argument.
    cache_hits : Optional[int], optional
        The number of page reads that were satisfied by the shared buffer. If no measurement is available, *None* can be
        used. If this argument is used, no other measures can be supplied in the `measures` argument.
    cache_misses : Optional[int], optional
        The number of page reads that had to be delegated to the disk and could not be satisfied by the shared buffer. If no
        measurement is available, *None* can be used. If this argument is used, no other measures can be supplied in the
        `measures` argument.
    subplan_root : Optional[QueryPlan], optional
        The root operator of the subplan. If this argument is used, no other subplan components can be supplied in the
        `subplan` argument.
    subplan_target_name : str, optional
        The name of the target table that the subplan should produce. If this argument is used, no other subplan components
        can be supplied in the `subplan` argument.
    **kwargs
        Additional metadata that should be attached to the plan parameters. If this is used, no other plan parameters can be
        supplied in the `plan_params` argument.

    See Also
    --------
    to_query_plan
    explode_query_plan
    OptimizerInterface.query_plan
    OptimizationPipeline.query_execution_plan
    """

    def __init__(
        self,
        node_type: str | PhysicalOperator,
        *,
        operator: Optional[PhysicalOperator] = None,
        children: Optional[QueryPlan | Iterable[QueryPlan]] = None,
        plan_params: Optional[PlanParams] = None,
        subplan: Optional[Subplan] = None,
        estimates: Optional[PlanEstimates] = None,
        measures: Optional[PlanMeasures] = None,
        base_table: Optional[TableReference] = None,
        filter_predicate: Optional[AbstractPredicate] = None,
        parallel_workers: Optional[int] = None,
        index: Optional[str] = None,
        sort_keys: Optional[Sequence[SortKey]] = None,
        lookup_key: Optional[SqlExpression] = None,
        estimated_cardinality: Cardinality | float | None = Cardinality.unknown(),
        estimated_cost: Cost = math.nan,
        actual_cardinality: Cardinality | float | None = Cardinality.unknown(),
        execution_time: float = math.nan,
        cache_hits: Optional[int] = None,
        cache_misses: Optional[int] = None,
        subplan_root: Optional[QueryPlan] = None,
        subplan_target_name: str = "",
        **kwargs,
    ) -> None:
        if not node_type:
            raise ValueError("Node type must be provided")

        custom_params = (
            base_table,
            filter_predicate,
            parallel_workers,
            index,
            sort_keys,
            lookup_key,
        )
        has_custom_params = any(v is not None for v in custom_params) or bool(kwargs)
        if plan_params is not None and has_custom_params:
            raise ValueError("PlanParams and individual parameters/kwargs cannot be provided at the same time")
        if plan_params is None:
            plan_params = PlanParams(
                base_table=base_table,
                filter_predicate=filter_predicate,
                sort_keys=sort_keys,
                lookup_key=lookup_key,
                parallel_workers=parallel_workers,
                index=index,
                **kwargs,
            )

        if estimates is not None and any(not math.isnan(v) for v in (estimated_cardinality, estimated_cost)):
            raise ValueError("PlanEstimates and individual estimates cannot be provided at the same time")
        if estimates is None:
            estimates = PlanEstimates(cardinality=estimated_cardinality, cost=estimated_cost)

        has_custom_measures = any(
            v is not None and not math.isnan(v) for v in (execution_time, cache_hits, cache_misses)
        )
        if measures is not None and has_custom_measures:
            raise ValueError("PlanMeasures and individual measures cannot be provided at the same time")
        if measures is None:
            measures = PlanMeasures(
                execution_time=execution_time,
                cardinality=actual_cardinality,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            )

        if subplan is not None and subplan_root is not None:
            raise ValueError("Subplan and individual subplan components cannot be provided at the same time")
        if subplan is None and subplan_root is not None:
            subplan = Subplan(subplan_root, subplan_target_name)

        match children:
            case None:
                children: list[QueryPlan] = []
            case QueryPlan():
                children = [children]
            case _:
                children = list(children)

        if isinstance(node_type, PhysicalOperator):
            operator = node_type
            node_type = operator.name

        self._node_type = node_type
        self._operator = operator

        self._input_node: QueryPlan | None
        if len(children) == 1:
            self._input_node = children[0]
        else:
            self._input_node = None

        self._children: tuple[QueryPlan, ...] = tuple(children) if children else ()
        self._plan_params = plan_params
        self._estimates = estimates
        self._measures = measures
        self._subplan = subplan

    @property
    def node_type(self) -> str:
        """Get the name of the operator."""
        return self._node_type

    @property
    def operator(self) -> Optional[PhysicalOperator]:
        """Get the actual operator that is used to compute the result set.

        For transient operators (e.g. hash tables), this can be *None*.
        """
        return self._operator

    @property
    def input_node(self) -> Optional[QueryPlan]:
        """Get the input node of the current operator.

        For nodes without an input (e.g. most scans), or nodes with multiple inputs (e.g. joins), this is *None*.
        """
        return self._input_node

    @property
    def children(self) -> Sequence[QueryPlan]:
        """Get the input nodes of the current operator.

        For nodes without an input (e.g. most scans), this is an empty list. For nodes with exactly one input (e.g. most
        aggregations), this is a singleton list. For nodes with two input nodes (e.g. joins), this is an ordered iterable
        with the outer child first.
        """
        return self._children

    @property
    def outer_child(self) -> Optional[QueryPlan]:
        """Get the outer input of the current operator.

        For nodes that do not have exactly two inputs, this is *None*.
        """
        if len(self._children) == 2:
            return self._children[0]
        return None

    @property
    def inner_child(self) -> Optional[QueryPlan]:
        """Get the inner input of the current operator.

        For nodes that do not have exactly two inputs, this is *None*.
        """
        if len(self._children) == 2:
            return self._children[1]
        return None

    @property
    def join_nodes(self) -> tuple[QueryPlan, QueryPlan]:
        """Get the outer and inner input nodes of the current join operator.

        If the node is not a (binary) join, an error is raised.
        """
        if not self.is_join() or len(self._children) != 2:
            raise ValueError(f"Not a binary join: {self}")
        return self._children

    @property
    def params(self) -> PlanParams:
        """Get the structural metadata of the operator."""
        return self._plan_params

    @property
    def base_table(self) -> Optional[TableReference]:
        """Get the table that is being scanned. For non-scan nodes, this will probably is *None*.

        This is just a shorthand for accessing the plan parameters manually.

        See Also
        --------
        PlanParams.base_table
        """
        return self._plan_params.base_table

    @property
    def filter_predicate(self) -> Optional[AbstractPredicate]:
        """Get the filter predicate that is used to restrict the tuples in the output of a relation.

        This is just a shorthand for accessing the plan parameters manually.

        See Also
        --------
        PlanParams.filter_predicate
        """
        return self._plan_params.filter_predicate

    @property
    def sort_keys(self) -> Sequence[SortKey]:
        """Get the sort keys describing the ordering of tuples in the output relation.

        Absence of a specific sort order is indicated by an empty sequence.

        This is just a shorthand for accessing the plan parameters manually.

        See Also
        --------
        PlanParams.sort_keys
        """
        return self._plan_params.sort_keys

    @property
    def lookup_key(self) -> Optional[SqlExpression]:
        """Get the expression that is used to lookup tuples in some indexing structure.

        This is just a shorthand for accessing the plan parameters manually.

        See Also
        --------
        PlanParams.lookup_key
        """
        return self._plan_params.lookup_key

    @property
    def parallel_workers(self) -> int:
        """Get the number of parallel workers that should be used to execute the operator.

        Absence of parallel execution is indicated by 0.

        This is just a shorthand for accessing the plan parameters manually.

        See Also
        --------
        PlanParams.parallel_workers
        """
        return self._plan_params.parallel_workers

    @property
    def estimates(self) -> PlanEstimates:
        """Get the optimizer's view on the operator."""
        return self._estimates

    @property
    def estimated_cardinality(self) -> Cardinality:
        """Get the cardinality estimate of the optimizer.

        This is just a shorthand for accessing the estimates manually.

        See Also
        --------
        PlanEstimates.cardinality
        """
        return self._estimates.cardinality

    @property
    def estimated_cost(self) -> Cost:
        """Get the cost estimate of the optimizer.

        This is just a shorthand for accessing the estimates manually.

        See Also
        --------
        PlanEstimates.cost
        """
        return self._estimates.cost

    @property
    def measures(self) -> PlanMeasures:
        """Get the actual execution statistics of the operator."""
        return self._measures

    @property
    def actual_cardinality(self) -> Cardinality:
        """Get the actual cardinality of the operator.

        This is just a shorthand for accessing the measures manually.

        See Also
        --------
        PlanMeasures.cardinality
        """
        return self._measures.cardinality

    @property
    def execution_time(self) -> float:
        """Get the actual execution time (in seconds) of the operator.

        The execution time is cumulative, i.e. it includes the time spent in all child operators as well.

        This is just a shorthand for accessing the measures manually.

        See Also
        --------
        PlanMeasures.execution_time
        """
        return self._measures.execution_time

    @property
    def operator_time(self) -> float:
        """Get the actual execution time (in seconds) spent in this operator only.

        The operator time excludes the time spent in child operators.

        Returns
        -------
        float
            The operator execution time in seconds. If no measurement is available, *NaN* is returned.
        """
        if not self.is_analyze():
            return math.nan
        child_time = sum(child.execution_time for child in self.children)
        return self.execution_time - child_time

    @property
    def subplan(self) -> Optional[Subplan]:
        """Get the subplan that has to be executed as part of this node."""
        return self._subplan

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a specific parameter from the plan.

        The lookup is performed in the following order:

        1. Plan parameters
        2. Plan estimates
        3. Plan measures

        If none of these containers contains the requested key, the default value is returned.
        """
        value = self._plan_params.get(key)
        if value is not None:
            return value
        value = self._estimates.get(key)
        if value is not None:
            return value
        value = self._measures.get(key)
        if value is not None:
            return value
        return default

    def is_join(self) -> bool:
        """Checks, whether the current node is a join operator."""
        return self._operator is not None and self._operator in JoinOperator

    def is_scan(self) -> bool:
        """Checks, whether the current node is a scan operator."""
        return self._operator is not None and self._operator in ScanOperator

    def is_auxiliary(self) -> bool:
        """Checks, whether the current node is an arbitrary intermediate operator (i.e. not a join nor a scan)."""
        return not self.is_join() and not self.is_scan()

    def is_analyze(self) -> bool:
        """Checks, whether the plan was executed in ANALYZE mode, i.e. whether runtime measurements are available."""
        return bool(self._measures)

    def is_ordered(self) -> bool:
        """Checks, whether the plan guarantees a specific order of the result tuples."""
        return bool(self._plan_params.sort_keys)

    def is_linear(self) -> bool:
        """Checks, whether the plan performs all joins in a linear order.

        This is the case if all join nodes compute their result by joining at least one base table (no matter whether it
        is the inner or outer child) with another relation (base relation or intermediate).

        As a special case, scan nodes are considered to be linear as well.
        """
        if self.is_scan():
            return True

        if not self.is_join() and self._input_node is None:
            raise ValueError(f"Cannot check linear order on node with {len(self._children)} children: {self}")
        elif self._input_node is not None:
            return self._input_node.is_linear()

        outer_child, inner_child = self.join_nodes
        outer_join = outer_child.find_first_node(QueryPlan.is_join)
        inner_join = inner_child.find_first_node(QueryPlan.is_join)
        return outer_join is None or inner_join is None

    def is_bushy(self) -> bool:
        """Checks, whether the plan performs joins in a bushy order.

        This is the case if at least one join node joins two intermediates that are themselves the result of a join.
        """
        if self.is_scan():
            return False

        if not self.is_join() and self._input_node is None:
            raise ValueError(f"Cannot check bushy order on node with {len(self._children)} children: {self}")
        elif self._input_node is not None:
            return self._input_node.is_bushy()

        outer_child, inner_child = self.join_nodes
        outer_join = outer_child.find_first_node(QueryPlan.is_join)
        inner_join = inner_child.find_first_node(QueryPlan.is_join)
        return outer_join is not None and inner_join is not None

    def is_left_deep(self) -> bool:
        """Checks, whether the plan performs all joins in a left-deep order.

        Left deep order means that the plan is linear and all joins are performed with the base table as the inner relation.
        As a special case, scan nodes are considered to be right-deep as well.
        """
        if self.is_scan():
            return True

        if not self.is_join() and self._input_node is None:
            raise ValueError(f"Cannot check left-deepness on node with {len(self._children)} children: {self}")
        elif self._input_node is not None:
            return self._input_node.is_left_deep()

        outer_child, inner_child = self.join_nodes
        inner_join = inner_child.find_first_node(QueryPlan.is_join)
        return inner_join is None and outer_child.is_left_deep()

    def is_right_deep(self) -> bool:
        """Checks, whether the plan performs all joins in a right-deep order.

        Right deep order means that the plan is linear and all joins are performed with the base table as the outer relation.
        As a special case, scan nodes are considered to be right-deep as well.
        """
        if self.is_scan():
            return True

        if not self.is_join() and self._input_node is None:
            raise ValueError(f"Cannot check right-deepness on node with {len(self._children)} children: {self}")
        elif self._input_node is not None:
            return self._input_node.is_bushy()

        outer_child, inner_child = self.join_nodes
        outer_join = outer_child.find_first_node(QueryPlan.is_join)
        return outer_join is None and inner_child.is_right_deep()

    def is_zigzag(self) -> bool:
        """Checks, whether the plan performs all joins in a zigzag order.

        Zig-zag order means that the plan is linear, but neither left-deep nor right-deep. Therefore, at least one join has
        to be performed with the base table as the outer relation and another join with the base table as the inner relation.
        As a special case, scan nodes are considered to be zig-zag as well.
        """
        if self.is_scan():
            return True
        return self.is_linear() and not self.is_left_deep() and not self.is_right_deep()

    def is_scan_branch(self) -> bool:
        """Checks, whether the current node directly leads to a scan node.

        For example, the plan *Hash(SeqScan(R))* is a scan branch, because the input of the hash node is a scan node.
        Likewise, the plan *Aggregate(Sort(R))* is a scan branch, because the input of the aggregate node is just a sort
        node which in turn contains a scan node. On the other hand, the plan *NestLoop(SeqScan(R), IdxScan(S))* is not a
        scan branch, because the nested-loop join contains two input nodes that are both scans.

        If a plan is a scan branch, `fetch_base_table()` can be used to directly retrieve the base table that is being scanned.
        """
        return bool(self.is_scan() or (self.input_node and self.input_node.is_scan_branch()))

    def is_base_join(self) -> bool:
        """Checks, whether the current node is a join node that joins two base tables.

        The base tables do not need to be direct children of the join, but both at least have to be scan branches, as in the
        case of *MergeJoin(Sort(SeqScan(R)), IdxScan(S))*.

        See Also
        --------
        is_scan_branch
        """
        if not self.is_join():
            return False
        return all(child.is_scan_branch() for child in self.children)

    def plan_depth(self) -> int:
        """Calculates the depth of the query plan.

        The depth of a query plan is the length of the longest path from the root to a leaf node. The leaf node is included in
        the calculation, i.e. the depth of the plan *SeqScan(R)* is 1.
        """
        return 1 + max((child.plan_depth() for child in self.children), default=0)

    def fetch_base_table(self) -> Optional[TableReference]:
        """Retrieves the base table that is being scanned by the plan.

        The base table is only specified for plans that directly lead to a scan node, as defined by `is_scan_branch()`.
        """
        if self.is_scan():
            return self.base_table
        elif self.is_join():
            return None

        if len(self.children) == 1:
            return self.children[0].fetch_base_table()
        return None

    def outermost_scan(self) -> Optional[QueryPlan]:
        """Retrieves the scan node that is furthest to the "left", i.e. on the outer-most position in the plan."""
        if self.is_scan():
            return self
        elif self.is_join():
            outer_child = self.outer_child
            assert outer_child
            return outer_child.outermost_scan()

        assert self.input_node is not None
        return self.input_node.outermost_scan()

    def tables(self) -> set[TableReference]:
        """Provides all tables that are accessed at some point in the plan.

        Notice that tables that are only accessed as part of user-specific metadata are not considered.
        """
        subplan_tabs: set[TableReference] = self._subplan.tables() if self._subplan else set()
        return self._plan_params.tables() | util.set_union(c.tables() for c in self._children) | subplan_tabs

    def columns(self) -> set[ColumnReference]:
        """Provides all columns that are accessed at some point in the plan.

        Notice that columns that are only accessed as part of user-specific metadata are not considered.
        """
        subplan_cols = self._subplan.root.columns() if self._subplan else set()
        return self._plan_params.columns() | util.set_union(c.columns() for c in self._children) | subplan_cols

    def iternodes(self) -> Iterable[QueryPlan]:
        """Provides all nodes that are contained in the plan in depth-first order, prioritizing outer child nodes."""
        return util.flatten(child.iternodes() for child in self._children) + [self]

    def lookup(self, tables: TableReference | Iterable[TableReference]) -> Optional[QueryPlan]:
        """Traverse the plan to find a specific intermediate node.

        If two nodes compute the same intermediate (i.e. provide the same tables), the node that is higher up in the plan is
        returned. If both appear on the same level, the outer child is preferred.

        Parameters
        ----------
        tables : TableReference | Iterable[TableReference]
            The tables that should be contained in the intermediate. If a single table is provided (either as-is or as a
            singleton iterable), the corresponding scan node will be returned. If multiple tables are provided, the highest
            node that provides all of them *exactly* is returned.

        Returns
        -------
        Optional[QueryPlan]
            The join tree node that contains the specified tables. If no such node exists, *None* is returned.
        """
        needle: set[TableReference] = set(util.enlist(tables))
        candidates = self.tables()

        if needle == candidates:
            return self
        if not needle.issubset(candidates):
            return None

        for child in self.children:
            result = child.lookup(needle)
            if result is not None:
                return result

        return None

    def find_first_node(
        self,
        predicate: Callable[[QueryPlan], bool],
        *args,
        direction: JoinDirection = "outer",
        **kwargs,
    ) -> Optional[QueryPlan]:
        """Recursively searches for the first node that matches a specific predicate.

        Parameters
        ----------
        predicate : Callable[[QueryPlan], bool]
            The predicate to check. The predicate is called on each node in the tree and should return a *True-ish* value if
            the node matches the desired search criteria.
        direction : JoinDirection, optional
            The traversal strategy to use. *Outer* (the default) indicates that the outer child should be traversed first if
            the check on the parent node fails. *Inner* indicates the opposite.
        args
            Additional positional arguments that are passed to the predicate *after* the current node.
        kwargs
            Additional keyword arguments that are passed to the predicate.

        Returns
        -------
        Optional[QueryPlan]
            The first node that matches the predicate. If no such node exists, *None* is returned.
        """
        if predicate(self, *args, **kwargs):
            return self
        if not self.children:
            return None

        if len(self.children) == 1:
            assert self._input_node is not None
            return self._input_node.find_first_node(predicate, *args, direction=direction, **kwargs)

        first_candidate, second_candidate = self.join_nodes
        if direction == "outer":
            first_candidate, second_candidate = second_candidate, first_candidate

        first_match = first_candidate.find_first_node(predicate, *args, direction=direction, **kwargs)
        if first_match:
            return first_match

        second_match = second_candidate.find_first_node(predicate, *args, direction=direction, **kwargs)
        if second_match:
            return second_match

        return (
            self._subplan.root.find_first_node(predicate, *args, direction=direction, **kwargs)
            if self._subplan
            else None
        )

    def find_all_nodes(self, predicate: Callable[[QueryPlan], bool], *args, **kwargs) -> Iterable[QueryPlan]:
        """Recursively searches for all nodes that match a specific predicate.

        The order in which the matching nodes appear is an implementation detail and should not be relied upon.

        Parameters
        ----------
        predicate : Callable[[QueryPlan], bool]
            The predicate to check. The predicate is called on each node in the tree and should return a *True-ish* value if
            the node matches the desired search criteria.
        args
            Additional positional arguments that are passed to the predicate *after* the current node.
        kwargs
            Additional keyword arguments that are passed to the predicate.

        Returns
        -------
        Iterable[QueryPlan]
            All nodes that match the predicate. If no such nodes exist, an empty iterable is returned.
        """
        matches: list[QueryPlan] = [self] if predicate(self, *args, **kwargs) else []
        for child in self._children:
            matches.extend(child.find_all_nodes(predicate, *args, **kwargs))
        if self._subplan:
            matches.extend(self._subplan.root.find_all_nodes(predicate, *args, **kwargs))
        return matches

    def cout(self, *, include_auxiliaries: bool = True) -> Cardinality:
        """Computes the *C-out* value of the operator.

        The *C-out* value is the sum of the cardinalities of the current operator and all its children.

        If the plan does not contain a measurement of the actual cardinality, the *C-out* value is undefined (indicated as
        *NaN*).

        Parameters
        ----------
        include_auxiliaries : bool, optional
            Whether to include auxiliary nodes in the computation (which is the default). If disabled, only the actual
            cardinality of join and scan nodes is considered.
        """
        if not self.is_analyze():
            return Cardinality.unknown()
        own_card = self.actual_cardinality if include_auxiliaries or not self.is_auxiliary() else Cardinality.zero()
        return own_card + sum(c.cout(include_auxiliaries=include_auxiliaries) for c in self.children)

    def qerror(self) -> float:
        """Computes the *Q-error* of the operator.

        If the plan does not contain an estimate of the cardinality, the *Q-error* value is undefined (indicated as *NaN*).

        Notes
        -----
        We use the a slight deviation from the standard definition:

        .. math ::
            qerror(e, a) = \\frac{max(e, a) + 1}{min(e, a) + 1}

        where *e* is the estimated cardinality of the node and *a* is the actual cardinality. Notice that we add 1 to both the
        numerator as well as the denominator to prevent infinity errors for nodes that do not process any rows (e.g. due to
        pruning).
        """
        if not self.is_analyze():
            return math.nan

        larger = max(self.estimated_cardinality, self.actual_cardinality) + 1
        smaller = min(self.estimated_cardinality, self.actual_cardinality) + 1
        return (larger / smaller).get()

    def parallelize(self, workers: int) -> QueryPlan:
        """Creates a parallel version of the query plan with the specified number of workers.

        The original plan is not changed.
        """
        plan_params = self._plan_params.clone()
        plan_params.parallel_workers = workers
        return QueryPlan(
            self.node_type,
            operator=self.operator,
            children=self.children,
            plan_params=plan_params,
            estimates=self._estimates,
            measures=self._measures,
            subplan=self.subplan,
        )

    def with_estimates(
        self,
        *,
        cardinality: Optional[Cardinality] = None,
        cost: Optional[Cost] = None,
        keep_measures: bool = False,
    ) -> QueryPlan:
        """Replaces the current estimates of the operator with new ones.

        Parameters
        ----------
        cardinality : Optional[Cardinality], optional
            The new estimated cardinality of the operator. If the estimate should be dropped *NaN* can be used. If the current
            cardinality should be kept, *None* can be passed (which is the default).
        cost : Optional[Cost], optional
            The new estimated cost of the operator. If the estimate should be dropped, *NaN* can be used. If the current cost
            should be kept, *None* can be passed (which is the default).
        keep_measures : bool, optional
            Whether to keep the actual measurements of the operator. If this is set to *False*, the actual cardinality and
            execution time are dropped. Measures are dropped by default because they usually depend on the estimates (which
            are now changed).
        """
        cardinality = self.estimated_cardinality if cardinality is None else cardinality
        cost = self.estimated_cost if cost is None else cost
        updated_estimates = PlanEstimates(cardinality=cardinality, cost=cost)
        updated_measures = self._measures if keep_measures else None
        return QueryPlan(
            self.node_type,
            operator=self.operator,
            children=self.children,
            plan_params=self.params,
            estimates=updated_estimates,
            measures=updated_measures,
            subplan=self.subplan,
        )

    def with_actual_card(
        self,
        *,
        cost_estimator: Optional[Callable[[QueryPlan, Cardinality], Cost]] = None,
        ignore_nan: bool = True,
    ) -> QueryPlan:
        """Replaces the current estimates of the operator with the actual measurements.

        The updated plan will not contain any measurements anymore and the costs will be set to *Nan* unless an explicit cost
        estimator is provided.

        Parameters
        ----------
        cost_estimator : Optional[Callable[[QueryPlan, Cardinality], Cost]], optional
            An optional cost function to compute new estimates based on the new estimates. If no cost estimator is provided,
            the cost is set to *NaN*. The estimator receives the old plan now along with the new cardinality estimate as input
            and should return the new cost estimate.
        ignore_nan : bool, optional
            Whether *NaN* cardinalities should also be swapped. By default, this is set to *True*, which only replaces the
            estimated cardinality if the actual cardinality is a meaningful value.

        Returns
        -------
        QueryPlan
            A new query plan with the actual cardinality as the estimated cardinality and the actual execution time as the
            estimated cost. The current plan is not changed.
        """
        if self.actual_cardinality:
            updated_cardinality = (
                self.estimated_cardinality
                if ignore_nan and self.actual_cardinality.isnan()
                else self.actual_cardinality
            )
            updated_cost = cost_estimator(self, updated_cardinality) if cost_estimator else math.nan
            updated_estimates = PlanEstimates(
                cardinality=updated_cardinality,
                cost=updated_cost,
                **(self._estimates.additional_estimates),
            )
            updated_measures = None
        else:
            updated_estimates = self._estimates
            updated_measures = None

        updated_children = [
            child.with_actual_card(cost_estimator=cost_estimator, ignore_nan=ignore_nan) for child in self.children
        ]

        if self.subplan:
            updated_subplan_root = self.subplan.root.with_actual_card(
                cost_estimator=cost_estimator, ignore_nan=ignore_nan
            )
            updated_subplan = Subplan(updated_subplan_root, self.subplan.target_name)
        else:
            updated_subplan = None

        return QueryPlan(
            self.node_type,
            operator=self.operator,
            children=updated_children,
            plan_params=self.params,
            estimates=updated_estimates,
            measures=updated_measures,
            subplan=updated_subplan,
        )

    def scale_cardinality(
        self,
        factor: float,
        *,
        kind: Literal["estimated", "actual", "both"] = "estimated",
        recursive: bool = False,
    ) -> QueryPlan:
        """Scales the cardinality estimate of the operator by a specific factor.

        Parameters
        ----------
        factor : float
            The factor by which the cardinality estimate should be scaled. For example, a factor of 0.5 reduces the estimate
            by half, while a factor of 2.0 doubles the estimate.
        kind : Literal["estimated", "actual", "both"], optional
            Specifies which cardinality estimates should be scaled. If set to *estimated* (the default), only the estimated
            cardinality is scaled. If set to *actual*, only the actual cardinality is scaled. If set to *both*, both estimates
            are scaled.
        recursive : bool, optional
            Whether to apply the scaling recursively to all child nodes. By default, this is set to *False*, which only
            scales the estimate of the current node.

        Returns
        -------
        QueryPlan
            A new query plan with the scaled cardinality estimate. The current plan is not changed.
        """

        updated_measures = self._measures
        updated_estimates = self._estimates
        if kind == "estimated" or kind == "both":
            updated_estimated = factor * self.estimated_cardinality
            updated_estimates = PlanEstimates(
                cardinality=updated_estimated,
                cost=self.estimated_cost,
                **(self._estimates.additional_estimates),
            )
        if kind == "actual" or kind == "both":
            updated_actual = factor * self.actual_cardinality
            updated_measures = PlanMeasures(
                cardinality=updated_actual,
                execution_time=self.execution_time,
                **(self._measures.additional_measures),
            )

        if recursive:
            updated_children = [child.scale_cardinality(factor, kind=kind, recursive=True) for child in self.children]
        else:
            updated_children = self.children

        if self.subplan:
            updated_subplan_root = self.subplan.root.scale_cardinality(factor, kind=kind, recursive=recursive)
            updated_subplan = Subplan(updated_subplan_root, self.subplan.target_name)
        else:
            updated_subplan = None

        return QueryPlan(
            self.node_type,
            operator=self.operator,
            children=updated_children,
            plan_params=self.params,
            estimates=updated_estimates,
            measures=updated_measures,
            subplan=updated_subplan,
        )

    def canonical(self) -> QueryPlan:
        """Creates a normalized version of the query plan.

        This normalized version will only contain scan and join nodes, without any auxiliary nodes. Estimates and measurements
        of these nodes are kept as they are.

        This method is mostly intended to remove system-specific elements of the QEP and provide a more stable representation.
        For example, Postgres uses a combination of Hash join node and Hash node to represent an actual hash join. Likewise,
        bitmap scans are represented as a bitmap heap scan with a number of bitmap index scans (and optional bitmap ANDs and
        ORs) as child nodes. With `canonical` all of these "implementation details" are removed and only the core of the query
        plan is kept.

        Notice that aggregations and groupings are also auxiliary nodes and will not be available after canonicalization.
        Therefore, the cost of the canonical query plan might be less than the cost of the original plan.
        """
        if self.subplan:
            updated_subplan_root = self.subplan.root.canonical()
            updated_subplan = Subplan(updated_subplan_root, self.subplan.target_name)
        else:
            updated_subplan = None

        if self.is_scan():
            # we remove all child nodes from scans to prevent any bitmap-scan shenanigans
            return QueryPlan(
                self.node_type,
                operator=self.operator,
                children=[],
                plan_params=self.params,  # params include the base table
                estimates=self.estimates,
                measures=self.measures,
                subplan=updated_subplan,
            )

        if not self.is_scan() and not self.is_join():
            # skip over auxiliary nodes
            assert self._input_node is not None
            return self._input_node.canonical()

        children = [child.canonical() for child in self.children]
        return QueryPlan(
            self.node_type,
            operator=self.operator,
            children=children,
            plan_params=self.params,
            estimates=self.estimates,
            measures=self.measures,
            subplan=updated_subplan,
        )

    def inspect(self, *, fields: Optional[Iterable[str]] = None) -> str:
        """Provides a human-readable representation of the query plan, inspired by Postgre's *EXPLAIN* output.

        By default, the output will contain fields akin to the *EXPLAIN* output of Postgres. For example, this includes the
        estimated cardinality and the operator cost, or for *ANALYZE* plans also the actual measurements.

        This can be customized by providing a list of fields that should be included in the output. The fields can either
        reference properties of the plan itself (e.g. ``estimated_cardinality``) or of a redirection to the metadata properties
        (e.g. ``params.index``). However, the current implementation only supports a single level of indirection, i.e. no
        ``params.custom_property.one_more_level``.
        """
        fields = [] if fields is None else list(fields)
        return _explainify(self, fields=fields)

    def explain(self) -> str:
        """Alias for `inspect`

        See Also
        --------
        inspect
        """
        return self.inspect()

    def plan_summary(self) -> dict[str, object]:
        """Provides a quick summary of important properties of the query plan, inspired by Panda's *describe* method."""
        all_nodes = list(self.iternodes())
        summary: dict[str, object] = {
            "operator": self.node_type,
            "intermediate": " ⋈ ".join(str(t) for t in self.tables()),
            "estimated_card": round(self.estimated_cardinality, 3),
            "actual_card": round(self.actual_cardinality, 3),
            "estimated_cost": round(self.estimated_cost, 3),
            "c_out": self.cout(),
            "max_qerror": round(max(node.qerror() for node in all_nodes), 3),
            "avg_qerror": round(sum(node.qerror() for node in all_nodes) / len(all_nodes), 3),
            "phys_ops": collections.Counter(child.node_type for child in all_nodes),
        }
        return summary

    def ast(self) -> str:
        """Provides the tree-structure of the plan in a human-readable format."""
        return _astify(self)

    def clone(self, *, deep: bool = False) -> QueryPlan:
        return self.__deepcopy__({}) if deep else self.__copy__()

    def __json__(self) -> jsondict:
        return {
            "node_type": self.node_type,
            "operator": self.operator,
            "children": self.children,
            "plan_params": self._plan_params,
            "estimates": self._estimates,
            "measures": self._measures,
            "subplan": self._subplan,
        }

    def __copy__(self) -> QueryPlan:
        return QueryPlan(
            self._node_type,
            operator=self._operator,
            children=self._children,
            plan_params=self._plan_params.clone(deep=False),
            estimates=self._estimates.clone(deep=False),
            measures=self._measures.clone(deep=False) if self._measures else None,
            subplan=self._subplan.clone(deep=False) if self._subplan else None,
        )

    def __deepcopy__(self, memo: dict[int, object] = {}) -> QueryPlan:
        return QueryPlan(
            self._node_type,
            operator=self._operator,
            children=[child.__deepcopy__(memo) for child in self._children],
            plan_params=self._plan_params.clone(deep=True),
            estimates=self._estimates.clone(deep=True),
            measures=self._measures.clone(deep=True) if self._measures else None,
            subplan=self._subplan.clone(deep=True) if self._subplan else None,
        )

    def __len__(self) -> int:
        return self.plan_depth()

    def __contains__(self, key: str | TableReference | Iterable[TableReference]) -> bool:
        if isinstance(key, TableReference):
            return key in self.tables()
        elif isinstance(key, Iterable):
            return set(key).issubset(self.tables())

        return bool(key in self._plan_params or key in self._estimates or (self._measures and key in self._measures))

    def __getitem__(self, key: str) -> Any:
        if key in self._plan_params:
            return self._plan_params[key]
        if key in self._estimates:
            return self._estimates[key]
        if self._measures and key in self._measures:
            return self._measures[key]
        raise KeyError(f"'{key}' not found")

    def __iter__(self) -> Iterator[QueryPlan]:
        yield self
        for child in self.children:
            yield from child
        if self.subplan:
            yield from self.subplan.root

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._node_type == other._node_type
            and self.base_table == other.base_table
            and self._children == other._children
        )

    def __hash__(self) -> int:
        return hash((self.node_type, self.base_table, self._children))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        normalized_node_type = self.node_type.replace(" ", "")
        if self.base_table:
            return f"{normalized_node_type}({self.base_table.identifier()})"
        child_texts = ", ".join(str(child) for child in self.children)
        return f"{normalized_node_type}({child_texts})"


_starting_indentation = 0


def _default_explain(plan: QueryPlan, *, padding: str) -> str:
    """Generates the Postgres-style *EXPLAIN* output for the current node."""
    components: list[str] = []
    metadata_indent = "      " if padding else "  "

    estimated_card = round(plan.estimated_cardinality, 3)
    estimated_cost = round(plan.estimated_cost, 3)
    components.append(
        f"{padding}{metadata_indent}Estimated Cardinality={estimated_card}, Estimated Cost={estimated_cost}"
    )

    if plan.is_analyze():
        actual_card = round(plan.actual_cardinality, 3)
        exec_time = round(plan.execution_time, 3)
        components.append(f"{padding}{metadata_indent}Actual Cardinality={actual_card}, Actual Time={exec_time}s")

    measures = plan.measures
    if measures.cache_hits is not None or measures.cache_misses is not None:
        cache_hits = measures.cache_hits if measures.cache_hits is not None else math.nan
        cache_misses = measures.cache_misses if measures.cache_misses is not None else math.nan
        components.append(f"{padding}{metadata_indent}Cache Hits={cache_hits}, Cache Misses={cache_misses}")

    params = plan.params
    if params.parallel_workers:
        components.append(f"{padding}{metadata_indent}Parallel Workers={params.parallel_workers}")
    if params.lookup_key:
        components.append(f"{padding}{metadata_indent}Lookup Key={params.lookup_key}")

    path_props: list[str] = []
    if params.index:
        path_props.append(f"Index={params.index}")
    if params.sort_keys:
        sort_keys = ", ".join(str(key) for key in params.sort_keys)
        path_props.append(f"Sort Keys={sort_keys}")
    if path_props:
        components.append(f"{padding}{metadata_indent}{', '.join(path_props)}")

    return "\n".join(components)


def _custom_explain(plan: QueryPlan, *, fields: list[str], padding: str) -> str:
    """Generates the user-specific *EXPLAIN* output for the current node."""
    attr_values: dict[str, str] = {}
    for attr in fields:
        if "." in attr:
            container_name, attr_name = attr.split(".")
            container = getattr(plan, container_name)
            value = getattr(container, attr_name)
        else:
            value = getattr(plan, attr)

        attr_values[attr] = str(round(value, 3)) if isinstance(value, Number) else str(value)

    attr_str = " ".join(f"{attr}={val}" for attr, val in attr_values.items())
    explain_data = f"{padding}   [{attr_str}]"
    return explain_data


def _explainify(plan: QueryPlan, *, fields: list[str], level: int = _starting_indentation) -> str:
    """Handler method to generate the *EXPLAIN* output for the current node and its children."""
    padding = "" if not level else "  " + "      " * (level - 1)
    prefix = f"{padding}->  " if padding else ""

    header = f"{plan.node_type}({plan.base_table})" if plan.is_scan() else plan.node_type
    explain_data = (
        _custom_explain(plan, fields=fields, padding=padding) if fields else _default_explain(plan, padding=padding)
    )
    child_explains = "\n".join(f"{_explainify(child, fields=fields, level=level + 1)}" for child in plan.children)
    subplan_explains = _explainify(plan.subplan.root, fields=fields, level=level + 1) if plan.subplan else ""
    if subplan_explains:
        child_explains = f"{child_explains}\n{subplan_explains}"

    if not child_explains:
        return f"{prefix}{header}\n{explain_data}"
    return f"{prefix}{header}\n{explain_data}\n{child_explains}"


def _astify(plan: QueryPlan, *, indentation: int = _starting_indentation) -> str:
    """Handler method to generate a tree-structure of the query plan."""
    padding = " " * indentation
    prefix = f"{padding}->  " if padding else ""
    if plan.is_scan():
        item_str = f"{prefix}{plan.node_type}({plan.base_table})"
    else:
        item_str = f"{prefix}{plan.node_type}"
    child_str = "\n".join(_astify(child, indentation=indentation + 2) for child in plan.children)
    return f"{item_str}\n{child_str}" if child_str else item_str
