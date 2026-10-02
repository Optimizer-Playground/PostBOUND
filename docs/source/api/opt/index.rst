Optimizer Package
=================

.. automodule:: postbound.opt


Cardinality Utilities
---------------------

.. autoclass:: postbound.opt.PerfectCardinalities
    :members:

.. autoclass:: postbound.opt.OfflineCardinalities
    :members:

.. autoclass:: postbound.opt.CardinalityCache
    :members:


Plan Utilities
--------------

.. autofunction:: postbound.opt.to_query_plan

.. autofunction:: postbound.opt.explode_query_plan

.. autofunction:: postbound.opt.update_plan

.. autofunction:: postbound.opt.read_query_plan_json

.. autofunction:: postbound.opt.read_jointree_json

.. autofunction:: postbound.opt.read_operator_json

.. autofunction:: postbound.opt.read_operator_assignment_json

.. autofunction:: postbound.opt.read_plan_params_json


Simple Optimizers
-----------------

In addition to general utilities, we also ship a number of simple or commonly-used optimizer implementations.
These should mostly be used as baselines or as part of a more complex optimizer.
The following optimizers are provided:

.. toctree::
    :maxdepth: 1

    dynprog
    randomized
    native
    enumeration
