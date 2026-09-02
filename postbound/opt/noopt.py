"""Provides empty (dummy) strategies for the individual optimization stages."""

from __future__ import annotations

from typing import Optional

from .._hints import JoinTree, PhysicalOperatorAssignment, PlanParameterization
from .._stages import (
    JoinOrderOptimization,
    ParameterGeneration,
    PhysicalOperatorSelection,
)
from ..qal import SqlQuery


class EmptyJoinOrderOptimizer(JoinOrderOptimization):
    """Dummy implementation of the join order optimizer that does not actually optimize anything."""

    def __init__(self) -> None:
        super().__init__()

    def optimize_join_order(self, query: SqlQuery) -> Optional[JoinTree]:
        return None

    def describe(self) -> dict:
        return {"name": "no_ordering"}


class EmptyPhysicalOperatorSelection(PhysicalOperatorSelection):
    """Dummy implementation of operator optimization that does not actually optimize anything."""

    def select_physical_operators(self, query: SqlQuery, join_order: Optional[JoinTree]) -> PhysicalOperatorAssignment:
        return PhysicalOperatorAssignment()

    def describe(self) -> dict:
        return {"name": "no_selection"}


class EmptyParameterization(ParameterGeneration):
    """Dummy implementation of the plan parameterization that does not actually generate any parameters."""

    def generate_plan_parameters(
        self,
        query: SqlQuery,
        join_order: Optional[JoinTree],
        operator_assignment: Optional[PhysicalOperatorAssignment],
    ) -> PlanParameterization:
        return PlanParameterization()

    def describe(self) -> dict:
        return {"name": "no_parameterization"}
