Existing Optimizer Implementations
==================================

The core PostBOUND framework focuses on providing a flexible and extensible architecture to implement novel optimization
strategies. As part of the `Optimization-Techniques <https://github.com/Optimizer-Playground/Optimization-Techniques>`
companion-project these abstractions are used to implement influential research from the last couple of years.
For example, the project includes implementations of the MSCN cardinality estimator, the BAO learned optimizer, or the
SafeBound pessimistic estimator.

.. important::

    We are constantly looking for new contributions to the PostBOUND optimimzer library.
    Our goal is to provide a comprehensive collection of optimizers that can be used for research and benchmarking.
    If you have developed an optimizer prototype we would be happy to make it available.
    Just reach out to us at our `GitHub repository <https://github.com/Optimizer-Playground/Optimization-Techniques>`_
    or by emailing us at `rico.bergmann1@tu-dresden.de <mailto:rico.bergmann1@tu-dresden.de>`_.

In addition to the research prototypes, the core framework also provides a set of simple baseline optimizers. These
serve as fallbacks in the optimization pipelines, or provide functionality that is so frequently used that it is worth
having it in the core framework. Currently the following baseline optimizers are available:

For **cardinality estimation** PostBOUND provides:

- :class:`~postbound.opt.PerfectCardinalities` to compute the true cardinalities of an intermediate
- :class:`~postbound.opt.NativeCardinalityEstimator` to use the cardinality estimates from an actual database system
- :class:`~postbound.opt.OfflineCardinalities` to read pre-computed cardinalities from a file
- :class:`~postbound.opt.CardinalityCache` to avoid repeated computations for costly estimation algorithms

For **cost modeling** PostBOUND provides the :class:`~postbound.opt.NativeCostModel` which extracts the estimated cost
from an actual database system.

For full **plan enumeration** PostBOUND offers

- :class:`~postbound.opt.DynamicProgrammingEnumerator` as a rather simple implementation of the traditional DP algorithm.
  This class is used as a default in the :class:`~postbound.TextbookOptimizationPipeline` if required
- :class:`~postbound.opt.PostgresDynProg` as another dynamic programming-based plan enumerator. This algorithm closely
  mirrors the internal enumerator used by Postgres. It functions as the default enumerator in the
  :class:`~postbound.TextbookOptimizationPipeline` if Postgres is the target database system.

For **join ordering**, **operator selection**, and **plan parameterization** in the context of a
:class:`~postbound.MultiStageOptimizationPipeline` PostBOUND contains

- native strategies that extract join order, physical operators, and plan parameters (cardinalities + parallel workers)
  from an actual database system. These are defined in :class:`~postbound.opt.NativeJoinOrderOptimizer`,
  :class:`~postbound.opt.NativePhysicalOperatorSelection`, and :class:`~postbound.opt.NativePlanParameterization`
  respectively
- a whole :class:`~postbound.opt.NativeOptimizer` which extracts an entire query plan
- random strategies that select join order and physical operators at random. These are defined in
  :class:`~postbound.opt.RandomJoinOrderOptimizer` and :class:`~postbound.opt.RandomOperatorOptimizer`
- the random strategies can also be combined in the :class:`~postbound.opt.RandomPlanOptimizer` to obtain an entire
  query plan

In addition to these pre-defined optimization stages, the :mod:`~postbound.opt` module also contains a number of
utilities to load/store the output of different optimization stages (e.g., query plans, join orders, etc.).
Furthermore, the module provides enumerators to iterate over random join orders, query plans, etc., and enumerators to
exhaustively generate them. Check the module documentation for more details.
