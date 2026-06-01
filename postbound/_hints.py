from __future__ import annotations

import math
from collections.abc import Collection, Container, Iterable
from enum import Enum
from typing import Any, Generic, Literal, Optional, Self, TypeVar

from . import util
from ._base import T
from ._core import (
    Cardinality,
    IntermediateOperator,
    JoinOperator,
    PhysicalOperator,
    ScanOperator,
    TableReference,
)
from ._qep import JoinDirection, QueryPlan
from .util import StateError, jsondict

JoinTreeAnnotation = TypeVar("JoinTreeAnnotation")
"""The concrete annotation used to augment information stored in the join tree."""


class PhysicalOperatorAssignment:
    """The physical operator assignment stores the operators that should be used for specific joins or scans.

    The assignment can happen at different levels:

    - `global_settings` enable or disable specific operators for the entire query
    - `join_operators` and `scan_operators` are concerned with specific (joins of) base tables. These assignments overwrite the
      global settings, i.e. it is possible to assign a nested loop join to a specific set of tables, but disable NLJ globally.
      In this case, only the specified join will be executed as an NLJ and other algorithms are used for all other joins
    - `intermediate_operators` are used to pre-process the input for joins, e.g. by caching input tuples in a memo.

    The basic assumption here is that for all joins and scans that have no assignment, the database system should determine the
    best operators by itself. Likewise, the database system is free to insert intermediate operators wherever it sees fit.

    Although it is allowed to modify the different dictionaries directly, the high-level methods (e.g. `add` or
    `set_join_operator`) should be used instead. This ensures that all potential (future) invariants are maintained.

    The assignment enables ``__getitem__`` access and tries to determine the requested setting in an intelligent way, i.e.
    supplying a single base table will provide the associated scan operator, supplying an iterable of base tables the join
    operator and supplying an operator will return the global setting. If no item is found, *None* will be returned.
    ``__iter__`` and ``__contains__`` wrap scan and join operators and ``__bool__`` checks for any assignment
    (global or specific). Notice that intermediate operators are not considered in the container-like methods.

    Attributes
    ----------
    global_settings : dict[ScanOperators | JoinOperators | IntermediateOperator, bool]
        Contains the global settings. Each operator is mapped to whether it is enable for the entire query or not. If an
        operator is not present in the dictionary, the default setting of the database system is used.
    join_operators : dict[frozenset[TableReference], JoinOperatorAssignment]
        Contains the join operators that should be used for individual joins. All joins are identified by the base tables that
        they combine. If a join does not appear in this dictionary, the database system has to choose an appropriate operator
        (perhaps while considering the `global_settings`).
    scan_operators : dict[TableReference, ScanOperatorAssignment]
        Contains the scan operators that should be used for individual base table scans. Each scan is identified by the table
        that should be scanned. If a table does not appear in this dictionary, the database system has to choose an appropriate
        operator (perhaps while considering the `global_settings`).
    intermediate_operators : dict[frozenset[TableReference], IntermediateOperator]
        Contains the intermediate operators that are used to pre-process the input for joins. Keys are the intermediate tables
        that are processed by the operator, i.e. an entry ``intermediate_operators[{R, S}] = Materialize`` means that the
        result of the join between *R* and *S* should be materialized and *not* that the input to the join between *R* and *S*
        should be materialized. Notice that intermediate operators are not enforced in conjunction with the join operators. For
        example, a merge join assignment between *R* and *S* does not require the presence of sort operators for *R* and *S*.
        Such interactions must be handled by the database hinting backend.
    """

    def __init__(self) -> None:
        self.global_settings: dict[PhysicalOperator, bool] = {}
        self.join_operators: dict[
            frozenset[TableReference], JoinOperatorAssignment
        ] = {}
        self.intermediate_operators: dict[
            frozenset[TableReference], IntermediateOperator
        ] = {}
        self.scan_operators: dict[TableReference, ScanOperatorAssignment] = {}

    def get_globally_enabled_operators(
        self, include_by_default: bool = True
    ) -> frozenset[PhysicalOperator]:
        """Provides all operators that are enabled globally.

        This differs from just calling ``assignment.global_settings`` directly, since all operators are checked, not just the
        operators that appear in the global settings dictionary.

        Parameters
        ----------
        include_by_default : bool, optional
            The behaviour for operators that do not have a global setting set. If enabled, such operators are assumed to be
            enabled and are hence included in the set.

        Returns
        -------
        frozenset[PhysicalOperator]
            The enabled scan and join operators. If no global setting is available for an operator `include_by_default`
            determines the appropriate action.
        """
        enabled_scan_ops = [
            scan_op
            for scan_op in ScanOperator
            if self.global_settings.get(scan_op, include_by_default)
        ]
        enabled_join_ops = [
            join_op
            for join_op in JoinOperator
            if self.global_settings.get(join_op, include_by_default)
        ]
        enabled_intermediate_ops = [
            intermediate_op
            for intermediate_op in IntermediateOperator
            if self.global_settings.get(intermediate_op, include_by_default)
        ]
        return frozenset(enabled_scan_ops + enabled_join_ops + enabled_intermediate_ops)

    def set_operator_enabled_globally(
        self,
        operator: PhysicalOperator,
        enabled: bool,
        *,
        overwrite_fine_grained_selection: bool = False,
    ) -> Self:
        """Enables or disables an operator for all parts of a query.

        Parameters
        ----------
        operator : PhysicalOperator
            The operator to configure
        enabled : bool
            Whether the database system is allowed to choose the operator
        overwrite_fine_grained_selection : bool, optional
            How to deal with assignments of the same operator to individual nodes. If *True* all assignments that contradict
            the setting are removed. For example, consider a situation where nested-loop joins should be disabled globally, but
            a specific join has already been assigned to be executed with an NLJ. In this case, setting
            `overwrite_fine_grained_selection` removes the assignment for the specific join. This is off by default, to enable
            the per-node selection to overwrite global settings.

        Returns
        -------
        PhysicalOperatorAssignment
            The assignment is updated in place and returned for chaining.
        """
        self.global_settings[operator] = enabled

        if not overwrite_fine_grained_selection or enabled:
            return self

        # at this point we know that we should disable a scan or join operator that was potentially set for
        # individual joins or tables
        match operator:
            case ScanOperator():
                self.scan_operators = {
                    table: current_setting
                    for table, current_setting in self.scan_operators.items()
                    if current_setting != operator
                }
            case JoinOperator():
                self.join_operators = {
                    join: current_setting
                    for join, current_setting in self.join_operators.items()
                    if current_setting != operator
                }
            case IntermediateOperator():
                self.intermediate_operators = {
                    join: current_setting
                    for join, current_setting in self.intermediate_operators.items()
                    if current_setting != operator
                }
            case _:
                raise ValueError(f"Unknown operator type: {operator}")

        return self

    def set_join_operator(
        self,
        operator: JoinOperatorAssignment | JoinOperator,
        tables: Iterable[TableReference] | None = None,
    ) -> Self:
        """Enforces a specific join operator for the join that consists of the contained tables.

        This overwrites all previous assignments for the same join. Global settings are left unmodified since per-join settings
        overwrite them anyway.

        Parameters
        ----------
        join_operator : JoinOperatorAssignment | JoinOperator
            The join operator. Can be an entire assignment, or just a plain operator. If a plain operator is supplied, the
            actual tables to join must be provided in the `tables` parameter.
        tables : Iterable[TableReference], optional
            The tables to join. This parameter is only used if only a join operator without a proper assignment is supplied in
            the `join_operator` parameter. Otherwise it is ignored.

        Returns
        -------
        PhysicalOperatorAssignment
            The assignment is updated in place and returned for chaining.

        Notes
        -----
        You can also pass a `DirectionalJoinOperatorAssignment` to this method. In contrast to the normal assignment, this
        one also distinguishes between inner and outer relations of the join.
        """
        if isinstance(operator, JoinOperator):
            operator = JoinOperatorAssignment(operator, tables)

        self.join_operators[operator.join] = operator
        return self

    def set_scan_operator(
        self,
        operator: ScanOperatorAssignment | ScanOperator,
        table: TableReference | Iterable[TableReference] | None = None,
    ) -> Self:
        """Enforces a specific scan operator for the contained base table.

        This overwrites all previous assignments for the same table. Global settings are left unmodified since per-table
        settings overwrite them anyway.

        Parameters
        ----------
        scan_operator : ScanOperatorAssignment | ScanOperator
            The scan operator. Can be an entire assignment, or just a plain operator. If a plain operator is supplied, the
            actual table to scan must be provided in the `table` parameter.
        table : TableReference | Iterable[TableReference], optional
            The table to scan. This parameter is only used if only a scan operator without a proper assignment is supplied in
            the `scan_operator` parameter. Otherwise it is ignored.

        Returns
        -------
        PhysicalOperatorAssignment
            The assignment is updated in place and returned for chaining.
        """
        if isinstance(operator, ScanOperator):
            table = util.simplify(table)
            operator = ScanOperatorAssignment(operator, table)

        self.scan_operators[operator.table] = operator
        return self

    def set_intermediate_operator(
        self, operator: IntermediateOperator, tables: Iterable[TableReference]
    ) -> Self:
        """Enforces an intermediate operator to process specific tables.

        This overwrites all previous assignments for the same intermediate. Global settings are left unmodified since
        per-intermediate settings overwrite them anyway.

        Parameters
        ----------
        intermediate_operator : IntermediateOperator
            The intermediate operator
        tables : Iterable[TableReference]
            The tables to process. Notice that these tables are not the tables that are joined, but the input to the join.
            For example, consider a neste-loop join between *R* and *S* where the tuples from *S* should be materialized
            (perhaps because they stem from an expensive index access). In this case, the assignment should contain a
            nested-loop assignment for the intermediate *{R, S}* and an assignment for the materialize operator for *S*.

        Returns
        -------
        PhysicalOperatorAssignment
            The assignment is updated in place and returned for chaining.
        """
        self.intermediate_operators[frozenset(tables)] = operator
        return self

    def add(
        self,
        operator: ScanOperatorAssignment | JoinOperatorAssignment | PhysicalOperator,
        tables: TableReference | Iterable[TableReference] | None = None,
    ) -> Self:
        """Adds an arbitrary operator assignment to the current settings.

        In contrast to the `set_scan_operator` and `set_join_operator` methods, this method figures out the correct assignment
        type based on the input.

        Parameters
        ----------
        operator : ScanOperatorAssignment | JoinOperatorAssignment | PhysicalOperator
            The operator to use. If this is a complete assignment, it is used as such. Otherwise, the `tables` parameter must
            contain the tables that are affected by the operator.
        tables : TableReference | Iterable[TableReference] | None, optional
            The tables to join. This parameter is only used if a plain operator is supplied in the `operator` parameter.
            Otherwise it is ignored.

        Returns
        -------
        PhysicalOperatorAssignment
            The assignment is updated in place and returned for chaining.
        """
        match operator:
            case ScanOperator():
                self.set_scan_operator(operator, tables)
            case JoinOperator():
                tables = util.enlist(tables)
                self.set_join_operator(operator, tables)
            case ScanOperatorAssignment():
                self.set_scan_operator(operator)
            case JoinOperatorAssignment():
                self.set_join_operator(operator)
            case IntermediateOperator():
                tables = util.enlist(tables)
                self.set_intermediate_operator(operator, tables)
            case _:
                raise ValueError(f"Unknown operator assignment: {operator}")

        return self

    def set(self, operator: PhysicalOperator, enabled: bool) -> Self:
        """Shortcut method for `set_operator_enabled_globally`.

        Fine-grained selections are not overwritten.

        Returns
        -------
        PhysicalOperatorAssignment
            The assignment is updated in place and returned for chaining.
        """
        return self.set_operator_enabled_globally(operator, enabled)

    def merge_with(
        self, other_assignment: PhysicalOperatorAssignment
    ) -> PhysicalOperatorAssignment:
        """Combines the current assignment with additional operators.

        In case of assignments to the same operators, the settings from the other assignment take precedence. None of the input
        assignments are modified.

        Parameters
        ----------
        other_assignment : PhysicalOperatorAssignment
            The assignment to combine with the current assignment

        Returns
        -------
        PhysicalOperatorAssignment
            The combined assignment
        """
        merged_assignment = PhysicalOperatorAssignment()
        merged_assignment.global_settings = (
            self.global_settings | other_assignment.global_settings
        )
        merged_assignment.join_operators = (
            self.join_operators | other_assignment.join_operators
        )
        merged_assignment.scan_operators = (
            self.scan_operators | other_assignment.scan_operators
        )
        merged_assignment.intermediate_operators = (
            self.intermediate_operators | other_assignment.intermediate_operators
        )
        return merged_assignment

    def integrate_workers_from(
        self, params: PlanParameterization, *, fail_on_missing: bool = False
    ) -> PhysicalOperatorAssignment:
        """Adds parallel workers from plan parameters to all matching operators.

        Parameters
        ----------
        params : PlanParameterization
            Parameters that provide the number of workers for specific intermediates
        fail_on_missing : bool, optional
            Whether to raise an error if the plan parameters contain worker hints for an intermediate that does not have
            an operator assigned. The default is to just ignore such hints.

        Returns
        -------
        PhysicalOperatorAssignment
            The updated assignment. The original assignment is not modified.
        """
        assignment = self.clone()

        for intermediate, workers in params.parallel_workers.items():
            operator = assignment.get(intermediate)
            if not operator and fail_on_missing:
                raise ValueError(
                    f"Cannot integrate workers - no operator set for {list(intermediate)}"
                )
            elif not operator:
                continue

            match operator:
                case ScanOperatorAssignment(op, tab):
                    updated_assignment = ScanOperatorAssignment(op, tab, workers)
                case DirectionalJoinOperatorAssignment(op, outer, inner):
                    updated_assignment = DirectionalJoinOperatorAssignment(
                        op, inner, outer, parallel_workers=workers
                    )
                case JoinOperatorAssignment(op, join):
                    updated_assignment = JoinOperatorAssignment(
                        op, join, parallel_workers=workers
                    )
                case _:
                    raise RuntimeError(f"Unexpected operator type: {operator}")

            assignment.add(updated_assignment)

        return assignment

    def global_settings_only(self) -> PhysicalOperatorAssignment:
        """Provides an assignment that only contains the global settings.

        Changes to the global settings of the derived assignment are not reflected in this assignment and vice-versa.

        Returns
        -------
        PhysicalOperatorAssignment
            An assignment of the global settings
        """
        global_assignment = PhysicalOperatorAssignment()
        global_assignment.global_settings = dict(self.global_settings)
        return global_assignment

    def clone(self) -> PhysicalOperatorAssignment:
        """Provides a copy of the current settings.

        Changes to the copy are not reflected back on this assignment and vice-versa.

        Returns
        -------
        PhysicalOperatorAssignment
            The copy
        """
        cloned_assignment = PhysicalOperatorAssignment()
        cloned_assignment.global_settings = dict(self.global_settings)
        cloned_assignment.join_operators = dict(self.join_operators)
        cloned_assignment.scan_operators = dict(self.scan_operators)
        cloned_assignment.intermediate_operators = dict(self.intermediate_operators)
        return cloned_assignment

    def get(
        self,
        intermediate: TableReference | Iterable[TableReference],
        default: Optional[T] = None,
    ) -> Optional[ScanOperatorAssignment | JoinOperatorAssignment | T]:
        """Retrieves the operator assignment for a specific scan or join.

        This is similar to the *dict.get* method. An important distinction is that we never raise an error if there is no
        intermediate assigned to the operator. Instead, we return the default value, which is *None* by default.

        Notice that this method never provides intermediate operators!

        Parameters
        ----------
        intermediate : TableReference | Iterable[TableReference]
            The intermediate to retrieve the operator assignment for. For scans, either the scanned table can be given
            directly, or the table can be wrapped in a singleton iterable.
        default : Optional[T], optional
            The default value to return if no assignment is found. Defaults to *None*.

        Returns
        -------
        Optional[ScanOperatorAssignment | JoinOperatorAssignment | T]
            The assignment if it was found or the default value otherwise.
        """
        if isinstance(intermediate, TableReference):
            return self.scan_operators.get(intermediate, default)

        intermediate_set = frozenset(intermediate)
        return (
            self.scan_operators.get(intermediate)
            if len(intermediate_set) == 1
            else self.join_operators.get(intermediate_set, default)
        )

    def inspect(self) -> str:
        padding = "  "
        lines: list[str] = ["PhysicalOperatorAssignment"]

        if self.global_settings:
            lines.append(f"{padding}Global settings:")
            for op, enabled in self.global_settings.items():
                status = "enabled" if enabled else "disabled"
                lines.append(f"{padding * 2}+- {op.value}: {status}")

        if self.scan_operators:
            lines.append(f"{padding}Scan operators:")
            for scan in self.scan_operators.values():
                lines.append(
                    f"{padding * 2}+- {scan.table.identifier()}: {scan.operator.value}"
                )

        if self.join_operators:
            lines.append(f"{padding}Join operators:")
            for join in self.join_operators.values():
                intermediate = ", ".join(tab.identifier() for tab in join.join)
                lines.append(
                    f"{padding * 2}+- {{{intermediate}}}: {join.operator.value}"
                )

        if self.intermediate_operators:
            lines.append(f"{padding}Intermediate operators:")
            for intermediate, op in self.intermediate_operators.items():
                intermediate_str = ", ".join(tab.identifier() for tab in intermediate)
                lines.append(f"{padding * 2}+- {{{intermediate_str}}}: {op.value}")

        return "\n".join(lines)

    def __json__(self) -> jsondict:
        jsonized = {
            "global_settings": [],
            "scan_operators": [
                {"table": scan.table, "operator": scan.operator}
                for scan in self.scan_operators.values()
            ],
            "join_operators": [
                {"intermediate": join.join, "operator": join.operator}
                for join in self.join_operators.values()
            ],
            "intermediate_operators": [
                {"intermediate": intermediate, "operator": op}
                for intermediate, op in self.intermediate_operators.items()
            ],
        }

        global_settings: list[dict] = []
        for operator, enabled in self.global_settings.items():
            match operator:
                case ScanOperator():
                    global_settings.append(
                        {"operator": operator, "enabled": enabled, "kind": "scan"}
                    )
                case JoinOperator():
                    global_settings.append(
                        {"operator": operator, "enabled": enabled, "kind": "join"}
                    )
                case IntermediateOperator():
                    global_settings.append(
                        {
                            "operator": operator,
                            "enabled": enabled,
                            "kind": "intermediate",
                        }
                    )
        jsonized["global_settings"] = global_settings

        return jsonized

    def __bool__(self) -> bool:
        return (
            bool(self.global_settings)
            or bool(self.join_operators)
            or bool(self.scan_operators)
            or bool(self.intermediate_operators)
        )

    def __iter__(self) -> Iterable[ScanOperatorAssignment | JoinOperatorAssignment]:
        yield from self.scan_operators.values()
        yield from self.join_operators.values()

    def __contains__(self, item: TableReference | Iterable[TableReference]) -> bool:
        if isinstance(item, TableReference):
            return item in self.scan_operators

        items = frozenset(item)
        return (
            item in self.scan_operators
            if len(items) == 1
            else items in self.join_operators
        )

    def __getitem__(
        self,
        item: TableReference | Iterable[TableReference] | ScanOperator | JoinOperator,
    ) -> ScanOperatorAssignment | JoinOperatorAssignment | bool | None:
        if isinstance(item, ScanOperator) or isinstance(item, JoinOperator):
            return self.global_settings.get(item, None)
        elif isinstance(item, TableReference):
            return self.scan_operators.get(item, None)
        elif isinstance(item, Iterable):
            return self.join_operators.get(frozenset(item), None)
        else:
            return None

    def __hash__(self) -> int:
        return hash(
            (
                util.hash_dict(self.global_settings),
                util.hash_dict(self.scan_operators),
                util.hash_dict(self.join_operators),
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.global_settings == other.global_settings
            and self.scan_operators == other.scan_operators
            and self.join_operators == other.join_operators
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        global_str = ", ".join(
            f"{op.value}: {enabled}" for op, enabled in self.global_settings.items()
        )

        scans_str = ", ".join(
            f"{scan.table.identifier()}: {scan.operator.value}"
            for scan in self.scan_operators.values()
        )

        joins_keys = (
            (join, " ⨝ ".join(tab.identifier() for tab in join.join))
            for join in self.join_operators.values()
        )
        joins_str = ", ".join(
            f"{key}: {join.operator.value}" for join, key in joins_keys
        )

        intermediates_keys = (
            (intermediate, " ⨝ ".join(tab.identifier() for tab in intermediate))
            for intermediate in self.intermediate_operators.keys()
        )
        intermediates_str = ", ".join(
            f"{key}: {intermediate.value}" for intermediate, key in intermediates_keys
        )

        return f"global=[{global_str}] scans=[{scans_str}] joins=[{joins_str}] intermediates=[{intermediates_str}]"


ExecutionMode = Literal["sequential", "parallel"]
"""
The execution mode indicates whether a query should be executed using either only sequential operators or only parallel
ones.
"""


class PlanParameterization:
    """The plan parameterization stores metadata that is assigned to different parts of the plan.

    Currently, three types of parameters are supported:

    - `cardinalities` provide specific cardinality estimates for individual joins or tables. These can be used to overwrite
      the estimation of the native database system
    - `parallel_workers` indicate how many worker processes should be used to execute individual joins or table
      scans (assuming that the selected operator can be parallelized). Notice that this can also be indicated as part of
      the `PhysicalOperatorAssignment` which will take precedence over this setting.
    - `system_settings` can be used to enable or disable specific optimization or execution features of the target
      database. For example, they can be used to disable parallel execution or switch to another cardinality estimation
      method. Such settings should be used sparingly since they defeat the purpose of optimization algorithms that are
      independent of specific database systems. Using these settings can also modify properties of the connection and
      therefore affect later queries. It is the users's responsibility to reset such settings if necessary.

    In addition, the `execution_mode` can be used to control whether the optimizer should only consider sequential plans or
    parallel plans. Note that the `parallel_workers` take precedence over this setting. If the optimizer should decide
    whether a parallel execution is beneficial, this should be set to *None*.

    Although it is allowed to modify the different dictionaries directly, the more high-level methods should be used
    instead. This ensures that all potential (future) invariants are maintained.

    Attributes
    ----------
    cardinalities : dict[frozenset[TableReference], Cardinality]
        Contains the cardinalities for individual joins and scans. This is always the cardinality that is emitted by a
        specific operator. All joins are identified by the base tables that they combine. Keys of single tables correpond
        to scans. Each join should assume that all filter predicates that can be evaluated at this point have already been
        applied.
    parallel_workers : dict[frozenset[TableReference], int]
        Contains the number of parallel processes that should be used to execute a join or scan. All joins are identified
        by the base tables that they combine. Keys of single tables correpond to scans. "Processes" does not necessarily
        mean "system processes". The database system can also choose to use threads or other means of parallelization. This
        is not restricted by the join assignment.
    system_settings : dict[str, Any]
        Contains the settings for the target database system. The keys and values, as well as their usage depend entirely
        on the system. For example, in Postgres a setting like *enable_geqo = 'off'* can be used to disable the genetic
        optimizer.
    execution_mode : ExecutionMode | None
        Indicates whether the optimizer should only consider sequential plans, parallel plans, or leave the decision to the
        optimizer (*None*). The default is *None*.
    """

    def __init__(self) -> None:
        self.cardinalities: dict[frozenset[TableReference], Cardinality] = {}
        """
        Contains the cardinalities for individual joins and scans. This is always the cardinality that is emitted by a
        specific operator. All joins are identified by the base tables that they combine. Keys of single tables correpond
        to scans.
        Each join should assume that all filter predicates that can be evaluated at this point have already been applied.
        """

        self.parallel_workers: dict[frozenset[TableReference], int] = {}
        """
        Contains the number of parallel processes that should be used to execute a join or scan. All joins are identified
        by the base tables that they combine. Keys of single tables correpond to scans. "Processes" does not necessarily
        mean "system processes". The database system can also choose to use threads or other means of parallelization. This
        is not restricted by the join assignment.
        """

        self.system_settings: dict[str, Any] = {}
        """
        Contains the settings for the target database system. The keys and values, as well as their usage depend entirely
        on the system. For example, in Postgres a setting like *enable_geqo = 'off'* can be used to disable the genetic
        optimizer.
        """

        self.execution_mode: ExecutionMode | None = None
        """
        Indicates whether the optimizer should only consider sequential plans, parallel plans, or leave the decision to the
        optimizer (*None*). The default is *None*.
        """

    def add_cardinality(
        self, tables: Iterable[TableReference], cardinality: Cardinality | int | float
    ) -> Self:
        """Assigns a specific cardinality hint to a (join of) tables.

        Parameters
        ----------
        tables : Iterable[TableReference]
            The tables for which the hint is generated. This can be an iterable of a single table, which denotes a scan hint.
        cardinality : Cardinality | int | float
            The estimated or known cardinality.

        Returns
        -------
        PlanParameterization
            The parameters are updated in place and returned for chaining.
        """
        cardinality = Cardinality.of(cardinality)
        self.cardinalities[frozenset(tables)] = cardinality
        return self

    def set_workers(self, tables: Iterable[TableReference], num_workers: int) -> Self:
        """Assigns a specific number of parallel workers to a (join of) tables.

        How these workers are implemented depends on the database system. They could become actual system processes, threads,
        etc.

        Parameters
        ----------
        tables : Iterable[TableReference]
            The tables for which the hint is generated. This can be an iterable of a single table, which denotes a scan hint.
        num_workers : int
            The desired number of worker processes. This denotes the total number of processes, not an additional amount. For
            some database systems this is an important distinction since one operator node will always be created. This node
            is then responsible for spawning the workers, but can also take part in the actual calculation. To prevent one-off
            errors, we standardize this number to denote the total number of workers that take part in the calculation.

        Returns
        -------
        PlanParameterization
            The parameters are updated in place and returned for chaining.
        """
        self.parallel_workers[frozenset(tables)] = num_workers
        return self

    def set_system_settings(
        self, setting_name: str = "", setting_value: Any = None, **kwargs
    ) -> Self:
        """Stores a specific system setting.

        This may happen in one of two ways: giving the setting name and value as two different parameters, or combining their
        assignment in the keyword parameters. While the first is limited to a single parameter, the second can be used to
        assign an arbitrary number of settings. However, this is limited to setting names that form valid keyword names.

        Parameters
        ----------
        setting_name : str, optional
            The name of the setting when using the separate key/value assignment mode. Defaults to an empty string to enable
            the integrated keyword parameter mode.
        setting_value : Any, optional
            The setting's value when using the separate key/value assignment mode. Defaults to *None* to enable the
            integrated keyword parameter mode.
        kwargs
            The key/value pairs in the integrated keyword parameter mode.

        Returns
        -------
        PlanParameterization
            The parameters are updated in place and returned for chaining.

        Raises
        ------
        ValueError
            If both the `setting_name` as well as keyword arguments are given
        ValueError
            If neither the `setting_name` nor keyword arguments are given

        Examples
        --------
        Using the separate setting name and value syntax: ``set_system_settings("join_collapse_limit", 1)``
        Using the kwargs syntax: ``set_system_settings(join_collapse_limit=1, jit=False)``
        Both examples are specific to Postgres (see https://www.postgresql.org/docs/current/runtime-config-query.html).
        """
        if setting_name and kwargs:
            raise ValueError("Only setting or kwargs can be supplied")
        elif not setting_name and not kwargs:
            raise ValueError("setting_name or kwargs required!")

        if setting_name:
            self.system_settings[setting_name] = setting_value
        else:
            self.system_settings |= kwargs

        return self

    def merge_with(
        self, other_parameters: PlanParameterization
    ) -> PlanParameterization:
        """Combines the current parameters with additional hints.

        In case of assignments to the same hints, the values from the other parameters take precedence. None of the input
        parameterizations are modified.

        Parameters
        ----------
        other_parameters : PlanParameterization
            The parameterization to combine with the current parameterization

        Returns
        -------
        PlanParameterization
            The merged parameters. The original parameterizations are not modified.
        """
        merged_params = PlanParameterization()
        merged_params.cardinalities = (
            self.cardinalities | other_parameters.cardinalities
        )
        merged_params.parallel_workers = (
            self.parallel_workers | other_parameters.parallel_workers
        )
        merged_params.system_settings = (
            self.system_settings | other_parameters.system_settings
        )
        return merged_params

    def drop_workers(self) -> PlanParameterization:
        """Provides a copy of the current parameters without any parallel worker hints.

        Changes to the copy are not reflected back on this parameterization and vice-versa.

        Returns
        -------
        PlanParameterization
            The copy without any parallel worker hints
        """
        params = PlanParameterization()
        params.cardinalities = dict(self.cardinalities)
        params.system_settings = dict(self.system_settings)
        params.execution_mode = self.execution_mode
        return params

    def inspect(self) -> str:
        padding = "  "
        lines: list[str] = ["PlanParameterization"]
        if self.system_settings:
            lines.append(f"{padding}System settings:")
            for setting, value in self.system_settings.items():
                lines.append(f"{padding * 2}+- {setting}: {value}")
        if self.cardinalities:
            lines.append(f"{padding}Cardinalities:")
            for tables, cardinality in self.cardinalities.items():
                intermediate = ", ".join(tab.identifier() for tab in tables)
                lines.append(f"{padding * 2}+- {{{intermediate}}}: {cardinality}")
        if self.parallel_workers:
            lines.append(f"{padding}Parallel workers:")
            for tables, workers in self.parallel_workers.items():
                intermediate = ", ".join(tab.identifier() for tab in tables)
                lines.append(f"{padding * 2}+- {{{intermediate}}}: {workers}")
        return "\n".join(lines)

    def __json__(self) -> jsondict:
        return {
            "cardinalities": [
                {"intermediate": intermediate, "cardinality": card}
                for intermediate, card in self.cardinalities.items()
            ],
            "parallel_workers": [
                {"intermediate": intermediate, "workers": workers}
                for intermediate, workers in self.parallel_workers.items()
            ],
        }

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return (
            f"PlanParams(cards={self.cardinalities}, "
            f"system specific={self.system_settings}, par workers={self.parallel_workers})"
        )


class ScanOperatorAssignment:
    """Models the selection of a scan operator for a specific base table.

    Attributes
    -------
    operator : ScanOperators
        The selected operator
    table : TableReference
        The table that is scanned using the operator
    parallel_workers : float | int
        The number of parallel processes that should be used to execute the scan. Can be set to 1 to indicate sequential
        operation. Defaults to NaN to indicate that no choice has been made.
    """

    def __init__(
        self,
        operator: ScanOperator,
        table: TableReference,
        parallel_workers: float | int = math.nan,
    ) -> None:
        self._operator = operator
        self._table = table
        self._parallel_workers = parallel_workers
        self._hash_val = hash((self._operator, self._table, self._parallel_workers))

    __match_args__ = ("operator", "table", "parallel_workers")

    @property
    def operator(self) -> ScanOperator:
        """Get the assigned operator.

        Returns
        -------
        ScanOperators
            The operator
        """
        return self._operator

    @property
    def table(self) -> TableReference:
        """Get the table being scanned.

        Returns
        -------
        TableReference
            The table
        """
        return self._table

    @property
    def parallel_workers(self) -> int | float:
        """Get the number of parallel workers used for the scan.

        This number designates the total number of parallel processes. It can be 1 to indicate sequential operation, or even
        *NaN* if it is unknown.

        Returns
        -------
        int | float
            The number of workers
        """
        return self._parallel_workers

    def inspect(self) -> str:
        """Provides the scan as a natural string.

        Returns
        -------
        str
            A string representation of the assignment
        """
        return f"USING {self.operator}" if self.operator else ""

    def __json__(self) -> jsondict:
        return {
            "operator": self.operator.value,
            "table": self.table,
            "parallel_workers": self.parallel_workers,
        }

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.operator == other.operator
            and self.table == other.table
            and self.parallel_workers == other.parallel_workers
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.operator.value}({self.table})"


class JoinOperatorAssignment:
    """Models the selection of a join operator for a specific join of tables.

    Each join is identified by all base tables that are involved in the join. The assignment to intermediate results does not
    matter here. For example, a join between R ⨝ S and T is expressed as R, S, T even though the actual join combined an
    intermediate result with as base table.

    A more verbose model is provided by the `DirectionalJoinOperatorAssignment`. In addition to the joined tables, that model
    also distinguishes between inner and outer relation of the join.

    Parameters
    ----------
    operator : JoinOperators
        The selected operator
    join : Collection[TableReference]
        The base tables that are joined using the operator
    parallel_workers : float | int, optional
        The number of parallel processes that should be used to execute the join. Can be set to 1 to indicate sequential
        operation. Defaults to NaN to indicate that no choice has been made.

    Raises
    ------
    ValueError
        If `join` contains less than 2 tables
    """

    def __init__(
        self,
        operator: JoinOperator,
        join: Collection[TableReference],
        *,
        parallel_workers: float | int = math.nan,
    ) -> None:
        if len(join) < 2:
            raise ValueError("At least 2 join tables must be given")
        self._operator = operator
        self._join = frozenset(join)
        self._parallel_workers = parallel_workers

        self._hash_val = hash((self._operator, self._join, self._parallel_workers))

    __match_args__ = ("operator", "join", "parallel_workers")

    @property
    def operator(self) -> JoinOperator:
        """Get the operator that was selected for the join

        Returns
        -------
        JoinOperators
            The operator
        """
        return self._operator

    @property
    def join(self) -> frozenset[TableReference]:
        """Get the tables that are joined together.

        For joins of more than 2 base tables this usually means that the join combines an intermediate result with a base table
        or another intermediate result. These two cases are not distinguished by the assignment and have to be detected
        through other information, e.g. the join tree.

        The more verbose model of a `DirectionalJoinOperatorAssignment` also distinguishes between inner and outer relations.

        Returns
        -------
        frozenset[TableReference]
            The tables that are joined together
        """
        return self._join

    @property
    def intermediate(self) -> frozenset[TableReference]:
        """Alias for `join`"""
        return self._join

    @property
    def parallel_workers(self) -> float | int:
        """Get the number of parallel processes that should be used in the join.

        "Processes" does not necessarily mean "system processes". The database system can also choose to use threads or other
        means of parallelization. This is not restricted by the join assignment.

        Returns
        -------
        float | int
            The number processes to use. Can be 1 to indicate sequential processing or NaN to indicate that no choice has been
            made.
        """
        return self._parallel_workers

    def inspect(self) -> str:
        """Provides this assignment as a natural string.

        Returns
        -------
        str
            A string representation of the assignment.
        """
        return f"USING {self.operator}" if self.operator else ""

    def is_directional(self) -> bool:
        """Checks, whether this assignment contains directional information, i.e. regarding inner and outer relation.

        Returns
        -------
        bool
            Whether the assignment explicitly denotes which relation should be the inner relationship and which relation should
            be the outer relationship
        """
        return False

    def __json__(self) -> jsondict:
        return {
            "directional": self.is_directional(),
            "operator": self.operator.value,
            "join": self.join,
            "parallel_workers": self.parallel_workers,
        }

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._operator == other._operator
            and self._join == other._join
            and self._parallel_workers == other._parallel_workers
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        join_str = ", ".join(str(tab) for tab in self.join)
        return f"{self.operator.value}({join_str})"


class DirectionalJoinOperatorAssignment(JoinOperatorAssignment):
    """A more verbose model of join operators.

    The directional assignment does not only represent the relations that should be joined together, but also denotes which
    role they should play for the join. More specifically, the directional assignment provides the *inner* and *outer* relation
    of the join. The precise semantics of this distinction depends on the specific join operator and is also used
    inconsistently between different database systems. In PostBOUND we use the following definitions:

    - for nested-loop joins the outer relation corresponds to the outer loop and the inner relation is the inner loop. As a
      special case for index nested-loop joins ths inner relation is the one that is probed via an index
    - for hash joins the outer relation is the one that is aggregated in a hash table and the inner relation is the one that
      is probed against that table
    - for sort-merge joins the assignment does not matter

    Parameters
    ----------
    operator : JoinOperators
        The selected operator
    inner : Collection[TableReference]
        The tables that form the inner relation of the join
    outer : Collection[TableReference]
        The tables that form the outer relation of the join
    parallel_workers : float | int, optional
        The number of parallel processes that should be used to execute the join. Can be set to 1 to indicate sequential
        operation. Defaults to NaN to indicate that no choice has been made.

    Raises
    ------
    ValueError
        If either `inner` or `outer` is empty.
    """

    def __init__(
        self,
        operator: JoinOperator,
        inner: Collection[TableReference],
        outer: Collection[TableReference],
        *,
        parallel_workers: float | int = math.nan,
    ) -> None:
        if not inner or not outer:
            raise ValueError("Both inner and outer relations must be given")
        self._inner = frozenset(inner)
        self._outer = frozenset(outer)
        super().__init__(
            operator, self._inner | self._outer, parallel_workers=parallel_workers
        )

    __match_args__ = ("operator", "outer", "inner", "parallel_workers")

    @property
    def inner(self) -> frozenset[TableReference]:
        """Get the inner relation of the join.

        Returns
        -------
        frozenset[TableReference]
            The tables of the inner relation
        """
        return self._inner

    @property
    def outer(self) -> frozenset[TableReference]:
        """Get the outer relation of the join.

        Returns
        -------
        frozenset[TableReference]
            The tables of the outer relation
        """
        return self._outer

    def is_directional(self) -> bool:
        return True

    def __json__(self) -> jsondict:
        return {
            "directional": True,
            "operator": self.operator,
            "inner": self.inner,
            "outer": self.outer,
            "parallel_workers": self.parallel_workers,
        }

    __hash__ = JoinOperatorAssignment.__hash__

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._inner == other._inner
            and self._outer == other._outer
            and super().__eq__(other)
        )


class HintType(Enum):
    """Contains all hint types that are supported by PostBOUND.

    Notice that not all of these hints need to be represented in the `PlanParameterization`, since some of them concern other
    aspects such as the join order. Furthermore, not all database systems will support all operators. The availability of
    certain hints can be checked on the database system interface and should be handled as part of the optimization pre-checks.
    """

    LinearJoinOrder = "Join order"
    JoinDirection = "Join direction"
    BushyJoinOrder = "Bushy join order"
    Operator = "Physical operators"
    Parallelization = "Par. workers"
    Cardinality = "Cardinality"


class JoinTree(Container[TableReference], Generic[JoinTreeAnnotation]):
    """A join tree models the sequence in which joins should be performed in a query plan.

    A join tree is a composite structure that contains base tables at its leaves and joins as inner nodes. Each node can
    optionally be annotated with arbitrary metadata (`annotation` property). While a join tree does usually not contain any
    information regarding physical operators to execute its joins or scans, we do distinguish between inner and outer relations
    at the join level.

    Each join tree instance is immutable. To expand the join tree, either use the `join_with` member method or create a new
    join tree, for example using the `join` factory method. The metadata can be updated using the `update_annotation` method.

    Regular join trees
    -------------------

    Depending on the specific node, different attributes are available. For leaf nodes, this is just the `base_table`
    property. For joins, the `outer_child` and `inner_child` properties are available. The specific node type can be checked
    using the `is_scan` and `is_join` methods respectively. Notice that these methods are "binary": ``is_join() = False``
    implies ``is_scan() = True`` and vice versa.
    No matter the specific node type, the `children` property always provides iteration support for the input nodes of the
    current node (which in case of base tables is just an empty iterable). Likewise, the `annotation` property is always
    available, but its value is entirely up to the user.

    Empty join trees
    ----------------

    An empty join tree is a special case that can be created using the `empty` factory method or by calling the constructor
    without any arguments. Empty join trees should only be used when starting the construction of a join tree and never be
    returned as a result of the optimization process. Clients are not required to check for emptiness and empty join trees
    also violate some of the invariants of proper join trees. Consider them syntactic sugar to simplify the construction, but
    only use them sparingly. If you decide to work with empty join trees, use the `is_empty` method to check for emptiness.

    Parameters
    ----------
    base_table : TableReference, optional
        The base table being scanned. Accessing this property on join nodes raises an error.
    outer_child : JoinTree[AnnotationType] | None, optional
        The left child of the join. Accessing this property on base tables raises an error.
    inner_child : JoinTree[AnnotationType] | None, optional
        The right child of the join. Accessing this property on base tables raises
    annotation : AnnotationType | None, optional
        The annotation for the node. This can be used to store arbitrary data.
    """

    # Note for maintainers: if you add new methods that return a join tree, make sure to add similar methods with the same
    # signature to the LogicalJoinTree (and a return type of LogicalJoinTree) to keep the two classes in sync.
    # Likewise, some methods deliberately have the same signatures as the QueryPlan class to allow for easy duck-typed usage.
    # These methods should also be kept in sync.

    @staticmethod
    def scan(
        table: TableReference, *, annotation: Optional[JoinTreeAnnotation] = None
    ) -> JoinTree[JoinTreeAnnotation]:
        """Creates a new join tree with a single base table.

        Parameters
        ----------
        table : TableReference
            The base table to scan
        annotation : AnnotationType
            The annotation to attach to the base table node

        Returns
        -------
        JoinTree[AnnotationType]
            The new join tree
        """
        return JoinTree(base_table=table, annotation=annotation)

    @staticmethod
    def join(
        outer: JoinTree[JoinTreeAnnotation],
        inner: JoinTree[JoinTreeAnnotation],
        *,
        annotation: Optional[JoinTreeAnnotation] = None,
    ) -> JoinTree[JoinTreeAnnotation]:
        """Creates a new join tree by combining two existing join trees.

        Parameters
        ----------
        outer : JoinTree[AnnotationType]
            The outer join tree
        inner : JoinTree[AnnotationType]
            The inner join tree
        annotation : AnnotationType
            The annotation to attach to the intermediate join node

        Returns
        -------
        JoinTree[AnnotationType]
            The new join tree
        """
        return JoinTree(outer_child=outer, inner_child=inner, annotation=annotation)

    @staticmethod
    def empty() -> JoinTree[JoinTreeAnnotation]:
        """Creates an empty join tree.

        Returns
        -------
        JoinTree[AnnotationType]
            The empty join tree
        """
        return JoinTree()

    def __init__(
        self,
        *,
        base_table: TableReference | None = None,
        outer_child: JoinTree[JoinTreeAnnotation] | None = None,
        inner_child: JoinTree[JoinTreeAnnotation] | None = None,
        annotation: JoinTreeAnnotation | None = None,
    ) -> None:
        self._table = base_table
        self._outer = outer_child
        self._inner = inner_child
        self._annotation = annotation
        self._hash_val = hash((base_table, outer_child, inner_child))

    @property
    def base_table(self) -> TableReference:
        """Get the base table for join tree leaves.

        Accessing this property on a join node raises an error.
        """
        if not self._table:
            raise StateError("This join tree does not represent a base table.")
        return self._table

    @property
    def outer_child(self) -> JoinTree[JoinTreeAnnotation]:
        """Get the left child of the join node.

        Accessing this property on a base table raises an error.
        """
        if not self._outer:
            raise StateError("This join tree does not represent an intermediate node.")
        return self._outer

    @property
    def inner_child(self) -> JoinTree[JoinTreeAnnotation]:
        """Get the right child of the join node.

        Accessing this property on a base table raises an error.
        """
        if not self._inner:
            raise StateError("This join tree does not represent an intermediate node.")
        return self._inner

    @property
    def children(
        self,
    ) -> tuple[JoinTree[JoinTreeAnnotation], JoinTree[JoinTreeAnnotation]]:
        """Get the children of the current node.

        For base tables, this is an empty tuple. For join nodes, this is a tuple of the outer and inner child.
        """
        if self.is_empty():
            raise StateError("This join tree is empty.")
        if self.is_scan():
            return ()
        return self._outer, self._inner

    @property
    def annotation(self) -> JoinTreeAnnotation:
        """Get the annotation of the current node."""
        if self.is_empty():
            raise StateError("Join tree is empty.")
        return self._annotation

    def is_empty(self) -> bool:
        """Check, whether the current join tree is an empty one."""
        return self._table is None and (self._outer is None or self._inner is None)

    def is_join(self) -> bool:
        """Check, whether the current join tree node is an intermediate."""
        return self._table is None

    def is_scan(self) -> bool:
        """Check, whether the current join tree node is a leaf node."""
        return self._table is not None

    def is_linear(self) -> bool:
        """Checks, whether the join tree encodes a linear join sequence.

        In a linear join tree each join node is always a join between a base table and another join node or another base table.
        As a special case, this implies that join trees that only constist of a single node are also considered to be linear.

        The opposite of linear join trees are bushy join trees. There also exists a `is_base_join` method to check whether a
        join node joins two base tables directly.

        See Also
        --------
        is_bushy
        """
        if self.is_empty():
            raise StateError("An empty join tree does not have a shape.")
        if self.is_scan():
            return True
        return self._outer.is_scan() or self._inner.is_scan()

    def is_bushy(self) -> bool:
        """Checks, whether the join tree encodes a bushy join sequence.

        In a bushy join tree, at least one join node is a join between two other join nodes. This implies that the join tree is
        not linear.

        See Also
        --------
        is_linear
        """
        return not self.is_linear()

    def is_base_join(self) -> bool:
        """Checks, whether the current join node joins two base tables directly."""
        return self.is_join() and self._outer.is_scan() and self._inner.is_scan()

    def tables(self) -> set[TableReference]:
        """Provides all tables that are scanned in the join tree.

        Notice that this does not consider tables that might be stored in the annotation of the join tree nodes.
        """
        if self.is_empty():
            return set()
        if self.is_scan():
            return {self._table}
        return self._outer.tables() | self._inner.tables()

    def plan_depth(self) -> int:
        """Calculates the depth of the join tree.

        The depth of a join tree is the length of the longest path from the root to a leaf node. The depth of an empty join
        is defined to be 0, while the depth of a join tree with a single node is 1.
        """
        if self.is_empty():
            return 0
        if self.is_scan():
            return 1
        return 1 + max(self._outer.plan_depth(), self._inner.plan_depth())

    def lookup(
        self, table: TableReference | Iterable[TableReference]
    ) -> Optional[JoinTree[JoinTreeAnnotation]]:
        """Traverses the join tree to find a specific (intermediate) node.

        Parameters
        ----------
        table : TableReference | Iterable[TableReference]
            The tables that should be contained in the intermediate. If a single table is provided (either as-is or as a
            singleton iterable), the correponding leaf node will be returned. If multiple tables are provided, the join node
            that calculates the intermediate *exactly* is returned.

        Returns
        -------
        Optional[JoinTree[AnnotationType]]
            The join tree node that contains the specified tables. If no such node exists, *None* is returned.
        """
        needle: set[TableReference] = set(util.enlist(table))
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

    def update_annotation(
        self, new_annotation: JoinTreeAnnotation
    ) -> JoinTree[JoinTreeAnnotation]:
        """Creates a new join tree with the same structure, but a different annotation.

        The original join tree is not modified.
        """
        if self.is_empty():
            raise StateError("Cannot update annotation of an empty join tree.")
        return JoinTree(
            base_table=self._table,
            outer_child=self._outer,
            inner_child=self._inner,
            annotation=new_annotation,
        )

    def join_with(
        self,
        partner: JoinTree[JoinTreeAnnotation] | TableReference,
        *,
        annotation: Optional[JoinTreeAnnotation] = None,
        partner_annotation: JoinTreeAnnotation | None = None,
        partner_direction: JoinDirection = "inner",
    ) -> JoinTree[JoinTreeAnnotation]:
        """Creates a new join tree by combining the current join tree with another one.

        Both input join trees are not modified. If one of the join trees is empty, the other one is returned as-is. As a
        special case, joining two empty join trees results once again in an empty join tree.

        Parameters
        ----------
        partner : JoinTree[AnnotationType] | TableReference
            The join tree to join with the current tree. This can also be a base table, in which case it is treated as a scan
            node of the table. The scan can be further described with the `partner_annotation` parameter.
        annotation : Optional[AnnotationType], optional
            The annotation of the new join node.
        partner_annotation : AnnotationType | None, optional
            If the join partner is given as a plain table, this annotation is used to describe the corresponding scan node.
            Otherwise it is ignored.
        partner_direction : JoinDirection, optional
            Which role the partner node should play in the new join. Defaults to "inner", which means that the current node
            becomes the outer node of the new join and the partner becomes the inner child. If set to "outer", the roles are
            reversed.

        Returns
        -------
        JoinTree[AnnotationType]
            The resulting join tree
        """
        if isinstance(partner, JoinTree) and partner.is_empty():
            return self
        if self.is_empty():
            return self._init_empty_join_tree(partner, annotation=partner_annotation)

        if isinstance(partner, JoinTree) and partner_annotation is not None:
            partner = partner.update_annotation(partner_annotation)
        elif isinstance(partner, TableReference):
            partner = JoinTree.scan(partner, annotation=partner_annotation)

        outer, inner = (
            (self, partner) if partner_direction == "inner" else (partner, self)
        )
        return JoinTree.join(outer, inner, annotation=annotation)

    def inspect(self) -> str:
        """Provides a pretty-printed an human-readable representation of the join tree."""
        return _inspectify(self)

    def iternodes(self) -> Iterable[JoinTree[JoinTreeAnnotation]]:
        """Provides all nodes in the join tree, with outer nodes coming first."""
        if self.is_empty():
            return []
        if self.is_scan():
            return [self]
        return [self] + self._outer.iternodes() + self._inner.iternodes()

    def itertables(self) -> Iterable[TableReference]:
        """Provides all tables that are scanned in the join tree. Outer tables appear first."""
        if self.is_empty():
            return []
        if self.is_scan():
            return [self._table]
        return self._outer.itertables() + self._inner.itertables()

    def iterjoins(self) -> Iterable[JoinTree[JoinTreeAnnotation]]:
        """Provides all join nodes in the join tree, with outer nodes coming first."""
        if self.is_empty() or self.is_scan():
            return []
        return self._outer.iterjoins() + self._inner.iterjoins() + [self]

    def _init_empty_join_tree(
        self,
        partner: JoinTree[JoinTreeAnnotation] | TableReference,
        *,
        annotation: Optional[JoinTreeAnnotation] = None,
    ) -> JoinTree[JoinTreeAnnotation]:
        """Handler method to create a new join tree when the current tree is empty."""
        if isinstance(partner, TableReference):
            return JoinTree.scan(partner, annotation=annotation)

        if annotation is not None:
            partner = partner.update_annotation(annotation)
        return partner

    def __json__(self) -> jsondict:
        if self.is_scan():
            return {
                "type": "join_tree_generic",
                "table": self._table,
                "annotation": self._annotation,
            }
        return {
            "type": "join_tree_generic",
            "outer": self._outer,
            "inner": self._inner,
            "annotation": self._annotation,
        }

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __contains__(self, x: object) -> bool:
        return self.lookup(x)

    def __len__(self) -> int:
        return len(self.tables())

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._table == other._table
            and self._outer == other._outer
            and self._inner == other._inner
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self):
        if self.is_scan():
            return self._table.identifier()
        return f"({self._outer} ⋈ {self._inner})"


class LogicalJoinTree(JoinTree[Cardinality]):
    """A logical join tree is a special kind of join tree that has cardinality estimates attached to each node.

    Other than the annotation type, it behaves exactly like a regular `JoinTree`. The cardinality estimates can be directly
    accessed using the `cardinality` property.
    """

    @staticmethod
    def scan(
        table: TableReference, *, annotation: Optional[Cardinality] = None
    ) -> LogicalJoinTree:
        return LogicalJoinTree(table=table, annotation=annotation)

    @staticmethod
    def join(
        outer: LogicalJoinTree,
        inner: LogicalJoinTree,
        *,
        annotation: Optional[Cardinality] = None,
    ) -> LogicalJoinTree:
        return LogicalJoinTree(outer=outer, inner=inner, annotation=annotation)

    @staticmethod
    def empty() -> LogicalJoinTree:
        return LogicalJoinTree()

    def __init__(
        self,
        *,
        table: TableReference | None = None,
        outer: LogicalJoinTree | None = None,
        inner: LogicalJoinTree | None = None,
        annotation: Cardinality | None = None,
    ) -> None:
        super().__init__(
            base_table=table,
            outer_child=outer,
            inner_child=inner,
            annotation=annotation,
        )

    @property
    def cardinality(self) -> Cardinality:
        return self.annotation

    @property
    def outer_child(self) -> LogicalJoinTree:
        return super().outer_child

    @property
    def inner_child(self) -> LogicalJoinTree:
        return super().inner_child

    @property
    def children(self) -> tuple[LogicalJoinTree, LogicalJoinTree]:
        return super().children

    def lookup(
        self, table: TableReference | Iterable[TableReference]
    ) -> Optional[LogicalJoinTree]:
        return super().lookup(table)

    def update_annotation(self, new_annotation: Cardinality) -> LogicalJoinTree:
        return super().update_annotation(new_annotation)

    def join_with(
        self,
        partner: LogicalJoinTree | TableReference,
        *,
        annotation: Optional[Cardinality] = None,
        partner_annotation: Cardinality | None = None,
        partner_direction: JoinDirection = "inner",
    ) -> LogicalJoinTree:
        return super().join_with(
            partner,
            annotation=annotation,
            partner_annotation=partner_annotation,
            partner_direction=partner_direction,
        )

    def iternodes(self) -> Iterable[LogicalJoinTree]:
        return super().iternodes()

    def iterjoins(self) -> Iterable[LogicalJoinTree]:
        return super().iterjoins()

    def __json__(self) -> jsondict:
        if self.is_scan():
            return {
                "type": "join_tree_logical",
                "table": self._table,
                "annotation": self._annotation,
            }
        return {
            "type": "join_tree_logical",
            "outer": self._outer,
            "inner": self._inner,
            "annotation": self._annotation,
        }


def _inspectify(
    join_tree: JoinTree[JoinTreeAnnotation], *, indentation: int = 0
) -> str:
    """Handler method to generate a human-readable string representation of a join tree."""
    padding = " " * indentation
    prefix = "<- " if padding else ""

    if join_tree.is_scan():
        return f"{padding}{prefix}{join_tree.base_table} ({join_tree.annotation})"

    join_node = f"{padding}{prefix}⨝ ({join_tree.annotation})"
    child_inspections = [
        _inspectify(child, indentation=indentation + 2) for child in join_tree.children
    ]
    return f"{join_node}\n" + "\n".join(child_inspections)


def jointree_from_plan(
    plan: QueryPlan, *, card_source: Literal["estimates", "actual"] = "estimates"
) -> LogicalJoinTree:
    """Extracts the join tree encoded in a query plan.

    The cardinality estimates of the join tree can be inferred from either the estimated cardinalities or from the measured
    actual cardinalities of the query plan.
    """
    card = (
        plan.estimated_cardinality
        if card_source == "estimates"
        else plan.actual_cardinality
    )
    if plan.is_scan():
        return JoinTree.scan(plan.base_table, annotation=card)
    elif plan.is_join():
        outer = jointree_from_plan(plan.outer_child, card_source=card_source)
        inner = jointree_from_plan(plan.inner_child, card_source=card_source)
        return JoinTree.join(outer, inner, annotation=card)
    else:
        # auxiliary node handler
        return jointree_from_plan(plan.input_node, card_source=card_source)


def parameters_from_plan(
    query_plan: QueryPlan | LogicalJoinTree,
    *,
    target_cardinality: Literal["estimated", "actual"] = "estimated",
    fallback_estimated: bool = False,
) -> PlanParameterization:
    """Extracts the cardinality estimates from a join tree.

    The join tree can be either a logical representation, in which case the cardinalities are extracted directly. Or, it can be
    a full query plan, in which case the cardinalities are extracted from the estimates or actual measurements. The cardinality
    source depends on the `target_cardinality` setting.
    If actual cardinalities should be used, but some nodes do only have estimates, these can be used as a fallback if
    `fallback_estimated` is set.
    """
    params = PlanParameterization()

    if isinstance(query_plan, LogicalJoinTree):
        card = query_plan.annotation
        parallel_workers = None
    else:
        if target_cardinality == "estimated":
            card = query_plan.estimated_cardinality
        elif target_cardinality == "actual" and not fallback_estimated:
            card = query_plan.actual_cardinality
        else:  # we should use actuals, but are allowed to fall back to estimates if necessary
            card = (
                query_plan.actual_cardinality
                if query_plan.actual_cardinality.is_valid()
                else query_plan.estimated_cardinality
            )
        parallel_workers = query_plan.params.parallel_workers

    if not math.isnan(card):
        params.add_cardinality(query_plan.tables(), card)
    if parallel_workers:
        params.set_workers(query_plan.tables(), parallel_workers)

    for child in query_plan.children:
        child_params = parameters_from_plan(
            child,
            target_cardinality=target_cardinality,
            fallback_estimated=fallback_estimated,
        )
        params = params.merge_with(child_params)

    return params


def operators_from_plan(
    query_plan: QueryPlan, *, include_workers: bool = False
) -> PhysicalOperatorAssignment:
    """Extracts the operator assignment from a whole query plan.

    Notice that this method only adds parallel workers to the assignment if explicitly told to, since this is generally
    better handled by the parameterization.
    """
    assignment = PhysicalOperatorAssignment()
    if not query_plan.operator and query_plan.input_node:
        return operators_from_plan(query_plan.input_node)

    workers = query_plan.parallel_workers if include_workers else math.nan
    match query_plan.operator:
        case ScanOperator():
            operator = ScanOperatorAssignment(
                query_plan.operator,
                query_plan.base_table,
                workers,
            )
            assignment.add(operator)
        case JoinOperator():
            operator = JoinOperatorAssignment(
                query_plan.operator,
                query_plan.tables(),
                parallel_workers=workers,
            )
            assignment.add(operator)
        case _:
            assignment.add(query_plan.operator, query_plan.tables())

    for child in query_plan.children:
        child_assignment = operators_from_plan(child)
        assignment = assignment.merge_with(child_assignment)
    return assignment
