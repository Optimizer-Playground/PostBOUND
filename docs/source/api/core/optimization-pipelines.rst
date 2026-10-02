Optimization Pipelines
======================

.. autoclass:: postbound.OptimizationPipeline
    :members:

.. autoclass:: postbound.OptimizationStage
    :members:

Textbook Pipeline
------------------

.. autoclass:: postbound.TextBookOptimizationPipeline
    :members:

.. autoclass:: postbound.PlanEnumerator
    :members:

.. autoclass:: postbound.CostModel
    :members:

.. autoclass:: postbound.CardinalityEstimator
    :members:


Multi-stage Optimization Pipeline
----------------------------------

.. autoclass:: postbound.MultiStageOptimizationPipeline
    :members:

.. autoclass:: postbound.JoinOrdering
    :members:

.. autoclass:: postbound.OperatorSelection
    :members:

.. autoclass:: postbound.ParameterGeneration
    :members:

.. tip::

    The :class:`~postbound.CardinalityEstimator` can also be used as a :class:`~postbound.ParameterGeneration`.


Other Pipelines
---------------

.. autoclass:: postbound.IntegratedOptimizationPipeline
    :members:

.. autoclass:: postbound.CompleteOptimizationAlgorithm
    :members:

.. autoclass:: postbound.IncrementalOptimizationPipeline
    :members:

.. autoclass:: postbound.IncrementalOptimizationStep
    :members:

.. autofunction:: postbound.as_complete_algorithm


Support Functionality
---------------------

.. autoclass:: postbound.OptimizationSettings
    :members:
