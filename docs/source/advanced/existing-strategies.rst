Existing Optimizer Implementations
==================================

PostBOUND comes with some existing optimizer implementations from influential or interesting works of the last couple of
years.
These can be used to compare your novel idea against existing approaches.
In addition to the actual research strategies, there are also some "pseudo-strategies" such as using the native optimizer
of a database system or just randomly deciding.
All strategies are directly available form the :mod:`postbound.opt` package, e.g. as `pb.opt.dynprog` for the dynamic programming-based optimizer.
Internally, the algorithms are available as lazy imports. This prevents unnecessary dependencies from being installed with
PostBOUND. For example, many learned estimators require Pytorch for their implementation. Lazy imports ensure that you do not
need to install Pytorch if you do not want to use such an estimator.

Currently, the following optimizers are implemented:

.. important::

    We are constantly looking for new contributions to the PostBOUND optimimzer library.
    Our goal is to provide a comprehensive collection of optimizers that can be used for research and benchmarking.
    If you have developed an optimizer prototype we would be happy to include it in PostBOUND.
    Just reach out to us at our `GitHub repository <https://github.com/Optimizer-Playground/PostBOUND>`_ or by emailing us at
    `rico.bergmann1@tu-dresden.de <mailto:rico.bergmann1@tu-dresden.de>`_.

.. warning::

    The optimizers from related work (i.e. UES and TONIC) have been moved to a separate "optimizer playground" library and
    are no longer part of the main PostBOUND library. They are still listed below for reference.

+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------+--------------------------------------------------+
| Name            | Description                                                                                                                                                                 | Reference        | Package                                          |
+=================+=============================================================================================================================================================================+==================+==================================================+
| UES             | Upper-bound driven join order optimizer. Bounds are derived from base statistics, specifically most-common values.                                                          | [Hertzschuch21]_ | *moved out of PostBOUND*                         |
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------+--------------------------------------------------+
| TONIC           | Learned physical operator selection. Operators are selected based on past experience and optional pretraining. Learning utilizes a prefix tree instead of a neural network. | [Hertzschuch22]_ | *moved out of PostBOUND*                         |
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------+--------------------------------------------------+
| DP              | Dynamic programming-based join order optimizer, with an alternative algorithm that mimics the actual Postgres enumerator.                                                   |                  | :mod:`dynprog <postbound.opt.dynprog>`           |
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------+--------------------------------------------------+
| *native*.       | Native optimization uses the built-in query optimizer of a database system.                                                                                                 |                  | :mod:`native <postbound.opt.native>`             |
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------+--------------------------------------------------+
| *random*        | Random selection of join orders and physical operators.                                                                                                                     |                  | :mod:`randomized <postbound.opt.randomized>`     |
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------+--------------------------------------------------+
| *exhaustive*    | These algorithms do not actually select a plan, but rather enumerate all possible plans (or join orders, or operator assignments).                                          |                  | :mod:`enumeration <postbound.opt.enumeration>`   |
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------+--------------------------------------------------+
| *cardinalities* | A collection of cardinality estimation utilities.                                                                                                                           |                  | :class:`~postbound.opt.PreComputedCardinalities` |
|                 |                                                                                                                                                                             |                  | :class:`~postbound.opt.PreciseCardinalities`     |
|                 |                                                                                                                                                                             |                  | :class:`~postbound.opt.CardinalityDistortion`    |
+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------+--------------------------------------------------+

.. [Hertzschuch21]
    Axel Hertzschuch, Claudio Hartmann, Dirk Habich and Wolfgang Lehner:
    "*Simplicity Done Right for Join Ordering*"
    CIDR 2021 (Link: https://www.cidrdb.org/cidr2021/papers/cidr2021_paper01.pdf)

.. [Hertzschuch22]
    Axel Hertzschuch, Claudio Hartmann, Dirk Habich and Wolfgang Lehner:
    "*Turbo-Charging SPJ Query Plans with Learned Physical Join Operator Selections*"
    VLDB 2022 (DOI: https://doi.org/10.14778/3551793.3551825)
