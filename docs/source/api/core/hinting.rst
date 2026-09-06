Hinting Abstractions
====================

.. autoclass:: postbound.HintType
    :members:


Join Orders
-----------

.. autoclass:: postbound.JoinTree
    :members:

.. autoclass:: postbound.LogicalJoinTree
    :members:

.. autofunction:: postbound.jointree_from_plan


Physical Operators
------------------

.. autoclass:: postbound.ScanOperatorAssignment
    :members:

.. autoclass:: postbound.JoinOperatorAssignment
    :members:

.. autoclass:: postbound.DirectionalJoinOperatorAssignment
    :members:

.. autoclass:: postbound.PhysicalOperatorAssignment
    :members:

.. autofunction:: postbound.operators_from_plan


Plan Parameters
---------------

.. autoclass:: postbound.PlanParameterization
    :members:

.. autodata:: postbound.ExecutionMode

.. autofunction:: postbound.parameters_from_plan
