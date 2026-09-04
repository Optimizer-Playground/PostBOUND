"""PostBOUND provides a simple model for training data and related metadata.

At the core of this module is the `TrainingData` class. It encapsulates the actual training samples in a Pandas DataFrame.
Each dataset is associated with a `TrainingSpec` that describes the features it provides. In turn, this information is used
to match datasets to the requirements of training pipelines.

In addition to the data abstraction, the module also provides a `TrainingDataRepository` for managing multiple datasets.
The `TrainingMetrics` are used by the benchmarking tools to evaluate the training process.

Examples
--------
In the simplest case, a dataset can be loaded from a file. Its features are inferred directly from the column names.
>>> samples = TrainingData.from_df("query-samples.parquet")
>>> samples
TrainingData('query-samples.parquet', features=[query, cardinality, runtime_ms, estimated_cost])
>>> samples.provides("cardinality")
True
>>> samples.as_df(requested_spec=TrainingSpec("query", "cardinality"))
    query                cardinality
0   SELECT * FROM users  1000
>>> samples.as_df(requested_spec=TrainingSpec("query", "runtime_ms"))
    query                runtime_ms
0   SELECT * FROM users  42.42

Multiple datasets can be merged together and conformed to a specific spec:
>>> repo = TrainingDataRepository()
>>> repo.register(samples)
>>> repo.retrieve_all(TrainingSpec("query", "cardinality"))
[TrainingData('query-samples.parquet', features=[query, cardinality])]
"""

from ._train import (
    SpecViolations,
    TrainingData,
    TrainingDataRepository,
    TrainingFeature,
    TrainingMetrics,
    TrainingSpec,
)

__all__ = [
    "SpecViolations",
    "TrainingData",
    "TrainingDataRepository",
    "TrainingFeature",
    "TrainingMetrics",
    "TrainingSpec",
]
