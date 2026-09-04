"""Contains the Postgres implementation of the Database interface.

In many ways the Postgres implementation can be thought of as the reference or blueprint implementation of the database
interface. This is due to two main reasons: first up, Postgres' capabilities follow a traditional architecture and its
features cover most of the general aspects of query optimization (i.e. supported operators, join orders and statistics).
Secondly, and on a more pragmatic note Potsgres was the first database system that was supported by PostBOUND and therefore
a lot of the original Postgres interfaces eventually evolved into the more abstract database-independent interfaces.
"""

from ._config import PostgresConfiguration, PostgresSetting
from ._ctl import is_running, start, stop
from ._explain import PostgresExplainNode, PostgresExplainPlan
from ._pg import (
    PostgresConfigInterface,
    PostgresHintService,
    PostgresInterface,
    PostgresJoinHints,
    PostgresOptimizer,
    PostgresPlanHints,
    PostgresScanHints,
    PostgresSchemaInterface,
    PostgresStatisticsInterface,
    connect,
)

__all__ = [
    "PostgresConfigInterface",
    "PostgresConfiguration",
    "PostgresExplainNode",
    "PostgresExplainPlan",
    "PostgresHintService",
    "PostgresInterface",
    "PostgresJoinHints",
    "PostgresOptimizer",
    "PostgresPlanHints",
    "PostgresScanHints",
    "PostgresSchemaInterface",
    "PostgresSetting",
    "PostgresStatisticsInterface",
    "connect",
    "is_running",
    "start",
    "stop",
]
