from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import pandas as pd

from .. import util
from ..parser import parse_query
from ..util import jsondict

type TrainingMetrics = jsondict
"""Training metrics are a flexible structure to capture various performance indicators of a training process.

We do not restrict the kind of information that can be stored in the training metrics, since this is highly model-specific.
However, there are two requirements:

1. training metrics should always be a dictionary
2. all entries must either be core Python types, or they must provide their own JSON serialization method by implementing the
   *__json__* method.
"""

type TrainingFeature = Literal["query", "runtime_ms", "query_plan", "estimated_cost", "cardinality"] | str
"""The kinds of training data that PostBOUND supports out-of-the-box.

Currently, these have no actual functionality attached to them, but they serve as a common vocabulary for the training data
specification. If you require some of these features in your own training data, make sure to use the exact same string as the
feature name. That way, your training data can be easily shared/used with other optimizers that require the same features
(or a subset of them).

Notes
-----
If you require custom features, simply pass them as strings to the relevant methods. They are designed to work gracefully with
unknown features.

If you think that your feature is relevant to a wider range of optimizers, please open an issue or a pull request at
https://github.com/Optimizer-Playground/PostBOUND to have it added to the list of supported features. We are happy to extend
the list of supported features!
"""


class TrainingSpec:
    """The training data specification is the "shared language" for describing features of training datasets.

    The specification serves two distinct purposes:

    1. it is used by training datasets to indicate what kind of samples they provide
    2. it is used by training pipelines to indicate what kind of samples they require

    Consequently, the training spec is the "contract" that binds datasets and optimization stages together. Each spec can be
    used interchangeably in both contexts and we provide different methods to check whether a dataset satisfies the
    requirements of a spec, and whether two specs are compatible with each other (e.g. one requirements spec and on provider
    spec).

    The training spec can be iterated over, which yields the individual features in the order they were specified.

    Parameters
    ----------
    features: Iterable[TrainingFeature]
        The features that are provided by the training data. This can be any iterable of strings, but it is recommended to use
        the predefined feature names whenever possible (see Notes below)
    *args:
        Additional features that are provided by the training data. This is just a convenient way to specify multiple features
        without having to create an explicit list.

    Attributes
    ----------
    features: Sequence[TrainingFeature]
        The actual features. If the spec is interpreted as a requirements spec, the order of the features is the order in which
        they are expected to be provided by the training data. If the spec is interpreted as a provider spec, the order of the
        features is the order in which they appear in the training data samples.
    feature_set: frozenset[TrainingFeature]
        The features once again, but as a set. This is useful for quick membership checks.

    Notes
    -----
    While the type signatures of the different methods suggest that the spec only works with specific features, in practice,
    the spec is designed to work with any string as a feature name. This allows to support custom features that might be
    specific to a particular optimizer.
    """

    def __init__(self, features: TrainingFeature | Iterable[TrainingFeature], *args) -> None:
        features = [features] if isinstance(features, str) else list(features)
        self._features: list[TrainingFeature] = list(features)
        self._features.extend(args)
        self._feature_set = frozenset(self._features)

    @property
    def features(self) -> Sequence[TrainingFeature]:
        """Get the features as a sequence. The order of the features is preserved."""
        return self._features

    @property
    def feature_set(self) -> frozenset[TrainingFeature]:
        """Get the features as a set. This is useful for quick membership checks."""
        return self._feature_set

    def provides(self, feature: TrainingFeature | Iterable[TrainingFeature]) -> bool:
        """Checks, whether the spec provides the given feature(s).

        The order of the given features does not need to match the order of the features in the spec.

        Notes
        -----
        This is method implements the exact same logic as `requires`. It is only provided for semantic clarity to support the
        two views on the training spec (as a provider spec and as a requirements spec).
        """
        feature = {feature} if isinstance(feature, str) else set(feature)
        return feature.issubset(self._feature_set)

    def requires(self, feature: TrainingFeature | Iterable[TrainingFeature]) -> bool:
        """Checks, whether the spec requires the given feature(s).

        The order of the given features does not need to match the order of the features in the spec.

        Notes
        -----
        This is method implements the exact same logic as `provides`. It is only provided for semantic clarity to support the
        two views on the training spec (as a provider spec and as a requirements spec).
        """
        return self.provides(feature)

    def satisfies(self, other: TrainingSpec) -> SpecViolations:
        """Checks, whether this spec provides all features that are required by another spec.

        Parameters
        ----------
        other : TrainingSpec
            The features that need to be presented in this spec. The order of the features does not matter, only their presence.

        Returns
        -------
        SpecViolations
            All features that are required by the other spec, but not provided by this spec. `SpecViolations` implements
            the *__bool__* method and evaluates to *True* if there are no violations. Therefore, the result of this method
            can be used directly in an *if* statement, etc.

        Examples
        --------
        >>> requirements = TrainingSpec("query", "cardinality")
        >>> provided = TrainingSpec("query", "runtime_ms")
        >>> if not provided.satisfies(requirements):
        ...     print("Provided spec does not satisfy requirements")
        """
        missing = other._feature_set - self._feature_set
        return SpecViolations(missing)

    def __iter__(self):
        return iter(self.features)

    def __hash__(self) -> int:
        return hash(self._feature_set)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TrainingSpec) and self._feature_set == other._feature_set

    def __repr__(self) -> str:
        return f"TrainingSpec({self.features})"

    def __str__(self) -> str:
        features = ", ".join(self.features)
        return f"TrainingSpec({features})"


class SpecViolations:
    """Represents features that are required by a spec user, but not provided by a spec provider.

    This is a simple wrapper around a set of missing features. In addition, it implements the *__bool__* method to allow for
    easy violation checks: if there are no violations, the object evaluates to *True*, otherwise it evaluates to *False*.
    This is tuned to align with `TrainingSpec.satisfies` (which returns an instance of this class) to allow expressive
    integration in *if* statements, etc.

    Parameters
    ----------
    missing_features: Iterable[TrainingFeature]
        The features that are required by a spec user, but not provided by a spec provider.

    Attributes
    ----------
    missing_features: frozenset[TrainingFeature]
        The features that are required by a spec user, but not provided by a spec provider.

    See Also
    --------
    TrainingSpec.satisfies
    """

    def __init__(self, missing_features: Iterable[TrainingFeature]) -> None:
        self.missing_features = frozenset(missing_features)

    def contains_violations(self) -> bool:
        """Checks whether there are any violations."""
        return bool(self.missing_features)

    def __bool__(self) -> bool:
        return not bool(self.missing_features)

    def __repr__(self) -> str:
        return f"SpecViolations(missing_features={self.missing_features})"

    def __str__(self) -> str:
        if not self.missing_features:
            return "SpecViolations(no violations)"
        missing = ", ".join(self.missing_features)
        return f"SpecViolations(missing_features={missing})"


@dataclass
class TrainingData:
    """Represents a dataset of training samples, along with metadata about the features it provides.

    Each dataset contains its training samples as tabular data. The actual format should be considered as an implementation
    detail and only the high-level interface of the class should be used to interact with the training data.
    Samples are associated with a `TrainingSpec` that describes the features they provide.

    To get access to the actual training samples, use the `as_df` method. This allows to retrieve the samples as a data frame,
    optionally adapted to a specific `TrainingSpec` (e.g. containing only a subset of the features or with the features in a
    specific order).

    Creating Training Data
    ----------------------
    The easiest way to create a dataset is to load it from a file using the `from_df` method. The features of the dataset are
    inferred from the column names of the data frame. If the raw data contains a *query* column, the queries are automatically
    parsed into proper `SqlQuery` objects.

    Alternatively, you can create a dataset from an existing data frame using the constructor directly. In this case,
    you also need to provide a *feature map*. This is a mapping from the actual feature as known to PostBOUND (e.g. *query*,
    *runtime_ms*, etc.) to the column that contains the corresponding data. This allows to handle renamings. For example,
    the raw data frame might contain a column *sql_query* that contains the actual queries. Without the feature map, PostBOUND
    would not recognize that this column actually contains the *query* feature.

    Parameters
    ----------
    samples: pd.DataFrame
        The actual training samples as a data frame.
    source: Optional[Path]
        The source of the training data. This is used for metadata purposes and does not affect the actual training samples.
        If the training data is loaded from a file, it is recommended to provide the file path as the source.
    feature_map: dict[TrainingFeature, str]
        A mapping from the actual feature as known to PostBOUND (e.g. *query*, *runtime_ms*, etc.) to the column that contains
        the corresponding data.

    Example
    -------
    >>> job_samples = TrainingData.from_df("query-samples-job.parquet")
    >>> job_samples
    TrainingData('query-samples-job.parquet', features=[query, cardinality, runtime_ms, estimated_cost])
    >>> stats_samples = TrainingData.from_df("query-samples-stats.parquet")
    >>> stats_samples
    TrainingData('query-samples-stats.parquet', features=[query, cardinality, query_plan])
    >>> merged_samples = TrainingData.merge([job_samples, stats_samples], according_to=TrainingSpec("query", "cardinality"))
    >>> merged_samples
    TrainingData('intermediate', features=[query, cardinality])
    """

    @staticmethod
    def from_df(df: pd.DataFrame | Path | str, *, source: Path | str | None = None) -> TrainingData:
        """Reads training data from a data frame or a file.

        The features of the dataset are inferred from the column names of the data frame. If the raw data contains a *query*
        column, the queries are automatically parsed into proper `SqlQuery` objects.

        Parameters
        ----------
        df: pd.DataFrame | Path | str
            The training samples as a data frame, or a path to a file containing the training samples.
        source: Optional[Path | str]
            The source of the training data. This is used for metadata purposes and does not affect the actual training
            samples. If the training data is loaded from a file, it is recommended to provide the file path as the source.
            If the training data is already given as a file path, this parameter is ignored.
        """
        if isinstance(df, (str, Path)):
            source = Path(df)
            df = util.read_df(source)
        if isinstance(source, str):
            source = Path(source)

        feature_map: dict[TrainingFeature, str] = {feat: feat for feat in df.columns}
        query_col = feature_map.get("query")
        if query_col:
            df[query_col] = df[query_col].map(parse_query)

        return TrainingData(df, source=source, feature_map=feature_map)

    @staticmethod
    def merge(datasets: Iterable[TrainingData], *, according_to: TrainingSpec) -> TrainingData:
        """Combines multiple datasets into a single large dataset.

        All input datasets need to satisfy the same spec, which is given by the `according_to` parameter. The resulting dataset
        will contain the features required by the spec and nothing more.
        If any of the input datasets does not satisfy the spec, an error is raised.
        """
        datasets = list(datasets)
        merged, *rest = datasets
        for ds in rest:
            merged = merged.merge_with(ds)
        return merged.conform_to(according_to)

    def __init__(
        self,
        samples: pd.DataFrame,
        *,
        source: Path | None = None,
        feature_map: dict[TrainingFeature, str],
    ) -> None:
        self._source = source
        self._feature_map = feature_map
        self._samples = samples
        self._spec = TrainingSpec(self._feature_map.keys())

    @property
    def source(self) -> Path | None:
        """Get the data file that originally provided the training data, if available."""
        return self._source

    @property
    def feature_map(self) -> Mapping[TrainingFeature, str]:
        """Get the association between the features as known to PostBOUND and the actual column names."""
        return self._feature_map

    @property
    def samples(self) -> pd.DataFrame:
        """Get the actual training samples as a data frame.

        The columns of the data frame are the actual column names, not the feature names.
        """
        return self._samples

    @property
    def spec(self) -> TrainingSpec:
        """Get the training features that are provided by this dataset."""
        return self._spec

    def provides(self, feature: TrainingFeature) -> bool:
        """Checks, whether the dataset provides a specific feature."""
        return self._spec.provides(feature)

    def satisfies(self, spec: TrainingSpec) -> SpecViolations:
        """Checks, whether the dataset provides all features that are required by a given spec.

        Parameters
        ----------
        spec : TrainingSpec
            The features that need to be presented in this dataset. The order of the features does not matter, only their
            presence.

        Returns
        -------
        SpecViolations
            All features that are required by the spec, but not provided by this dataset. `SpecViolations` implements
            the *__bool__* method and evaluates to *True* if there are no violations. Therefore, the result of this method
            can be used directly in an *if* statement, etc.

        See Also
        --------
        TrainingSpec.satisfies
        """
        return self._spec.satisfies(spec)

    def conform_to(self, features: Iterable[TrainingFeature] | TrainingSpec) -> TrainingData:
        """Modifies the dataset to satisfy exactly the given spec.

        If the dataset does not satisfy the given spec, an error is raised.

        Parameters
        ----------
        features: Iterable[TrainingFeature] | TrainingSpec
            The features that need to be provided by the resulting dataset. An iterable of features will be interpreted as the
            features of a `TrainingSpec`. While the order of the features does not matter for the compatibility check, the
            resulting dataset will have the features in the exact order as given by the `features` parameter.

        Returns
        -------
        TrainingData
            A new dataset that provides exactly the features required by the given spec, exactly in the order required by the
            spec.
        """
        spec = features if isinstance(features, TrainingSpec) else TrainingSpec(features)
        if not self._spec.satisfies(spec):
            raise ValueError("Requested spec is not compatible with the training data")
        reduced_spec: dict[TrainingFeature, str] = {
            feature: col for feature, col in self._feature_map.items() if spec.requires(feature)
        }
        return TrainingData(self._samples, source=self._source, feature_map=reduced_spec)

    def as_df(self, requested_spec: TrainingSpec | None = None) -> pd.DataFrame:
        """Provides the training samples as a data frame.

        Columns of the data frame are automatically aligned with the features of the given spec. If no spec is given, the
        columns of the data frame are the original columns, but renamed according to the feature map.
        """
        renamings = {target: col for target, col in self._feature_map.items()}

        if requested_spec is None:
            return self._samples.rename(columns=renamings)

        if not self._spec.satisfies(requested_spec):
            raise ValueError("Requested spec is not compatible with the training data")

        columns = [self._feature_map[feature] for feature in requested_spec.features]
        return self._samples[columns].rename(columns=renamings, errors="ignore")

    def merge_with(
        self,
        other: TrainingData,
    ) -> TrainingData:
        """Combine the samples of this dataset with another dataset.

        The *other* dataset needs to satisfy the spec of this dataset, which will also be used for the resulting dataset.
        If it does not, an error is raised.
        """
        violations = other._spec.satisfies(self._spec)
        if violations.contains_violations():
            raise ValueError(
                "Cannot merge the given training data: "
                f"the other dataset is missing the following features: {violations.missing_features}"
            )

        other_samples = other.as_df(self._spec)
        merged_samples = pd.concat([self._samples, other_samples])
        return TrainingData(merged_samples, source=None, feature_map=self._feature_map)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def __getitem__(self, idx) -> list:
        sample = self._samples.iloc[idx]
        remapped = sample[self._feature_map.values()]
        return list(remapped)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        features = ", ".join(self._feature_map.keys())
        source = self._source if self._source else "intermediate"
        return f"TrainingData('{source}', features=[{features}])"


class TrainingDataRepository:
    """The training data repository contains multiple datasets and provides retrieval methods based on training specs.

    Individual datasets need to be added to the repository using the `register` method. Once they are registered, they can be
    retrieved using the `retrieve_*` methods.
    """

    def __init__(self) -> None:
        self._datasets: list[TrainingData] = []

    def register(self, samples: TrainingData) -> Self:
        """Adds a new dataset to the repository."""
        self._datasets.append(samples)
        return self

    def retrieve_first(self, spec: TrainingSpec) -> TrainingData | None:
        """Provides the first dataset in the repository that satisfies the given spec.

        If no dataset satisfies the given spec, *None* is returned.
        """
        dataset = next((ds for ds in self._datasets if ds.satisfies(spec)), None)
        if dataset is None:
            return None
        return dataset.conform_to(spec)

    def retrieve_all(self, spec: TrainingSpec) -> Sequence[TrainingData]:
        """Provides all datasets in the repository that satisfy the given spec."""
        return [ds for ds in self._datasets if ds.satisfies(spec)]

    def retrieve_merged(self, spec: TrainingSpec) -> TrainingData | None:
        """Provides all datasets in the repository that satisfy the given spec.

        The datasets are merged together into a single large dataset that contains all samples from the individual datasets.
        This dataset uses exactly the given spec. If no dataset satisfies the spec, *None* is returned.
        """
        matching = self.retrieve_all(spec)
        if not matching:
            return None
        elif len(matching) == 1:
            return matching[0]
        else:
            return TrainingData.merge(matching, according_to=spec)
