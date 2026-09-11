"""PostBOUND - A research framework for query optimization in relational database systems.

PostBOUND allows to rapidly prototype novel ideas in query optimization and to benchmark them in a transparent and
reproducible manner. On a high level, the framework consists of the following components:

- **optimization pipelines** provide models for different optimizer architecures. Each pipeline provides *hooks*
  (called *optimization stages*) where users can plug in their own optimization strategies.
- **database backends** enable the evaluation of optimization pipelines on real-world database systems. Backends translate
  the optimization decisions into system-specific query hints that enforce the selected execution plan at runtime.
  PostBOUND has support for Postgres and DuckDB built in, but users are free to implement their own custom backends.
- **workloads and benchmarking utilities** allow to evaluate optimization pipelines on popular benchmarks in a reproducible
  manner.
- additional **infrastructure modules** handle the boilerplate part of query optimization, such as parsing and representing
  SQL queries, retrieving schema information, or obtaining database statistics.


Documentation
-------------

Visit https://postbound.readthedocs.io/ for the full documentation of PostBOUND, including guides, tutorials and API
references.

Package Structure
-----------------

- Core data structures and utilities are globally available (e.g. `OptimizationPipeline`, `Database`, or `SqlQuery`).
- SQL representation and parsing is handled by the `qal` and `parser` modules. As a convenience, `parse_query` is also
  available at the package root.
- Query modification is implemented in the `transform` module.
- The database interface is defined in the `db` module. `postgres` and `duckdb` provide concrete backends.
- Workloads and benchmarking utilities are available in `workloads` and `bench`.
- Simple optimization algorithms (e.g. dynamic-programming-based plan enumeration) are provided in the `opt` module.
"""

from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

try:
    __version__ = version("PostBOUND")
except PackageNotFoundError:
    __version__ = "<development>"

if find_spec("IPython") is not None:
    from ._ipython import setup_pretty_printers

    setup_pretty_printers()
