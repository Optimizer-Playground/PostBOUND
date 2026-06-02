from __future__ import annotations

import abc
import math
from collections.abc import Generator, Iterable
from typing import Optional, Self, Type

from . import util
from ._core import Cardinality, Cost, TableReference, TimeMs
from ._hints import JoinTree, PhysicalOperatorAssignment, PlanParameterization
from ._qep import QueryPlan
from .db import Database, DatabasePool, ResultSet
from .qal import SqlQuery
from .train import TrainingData, TrainingMetrics, TrainingSpec
from .util.jsonize import jsondict
from .validation import EmptyPreCheck, OptimizationPreCheck
from .workloads import Workload


def _missing_method_impls(
    child_cls: Type, base_cls: Type, *, methods: list[str]
) -> list[str]:
    """Scan the `child_cls` for methods that are implemented in the `base_cls` but not overridden in the `child_cls`.

    This check fails gracefully if none of the methods are implemented. This behavior is used to simulate our "poor man's
    protocol" and to indicate that the "child" simply does not implement that part of the base class's protocol.

    Returns
    -------
    list[str]
        The list of missing method implementations. If the list is empty, either all methods are implemented or the client
        does not implement this part of the protocol at all.
    """
    missing: list[str] = []
    for meth in methods:
        if getattr(child_cls, meth, None) != getattr(base_cls, meth, None):
            continue
        missing.append(meth)

    if len(missing) == len(methods):
        # all are missing - estimator does not implement this component
        return []
    return missing


class OptimizationStage:
    """Optimization stages are the core building blocks of optimization pipelines.

    Each stage implements a different, pipeline-specific step of the optimization process. When developing a new
    optimizer prototype, you generally identify which mental optimizer architecture is most suitable for your needs
    (i.e. which pipeline you need to use) and then implement the relevant stages for that pipeline and your specific idea.

    Optimization stages are generally "hooks" that extend the optimization process at specific points. If you do not care
    about a specific stage, you simply do not implement it. The pipeline will either skip the stage entirely or use a
    reasonable default implementation. However, there might be some stages that are required for a specific pipeline.
    Check the documentation of the pipeline you want to use for more details.

    Customizing Your Pipeline
    -------------------------
    When implementing a new optimization stage, the specific type of stage has at least one abstract method that you have
    to implement. This method is used to provide the core logic of the stage. For example, the `JoinOrderOptimization`
    stage requires you to implement the `optimize_join_order` method.

    In addition, you should also override the `describe` method. This method provides a JSON-serializable description of
    the specific optimization strategy along with important parameters. This information is used (among others) for
    benchmarking to document precisely how a pipeline was set up, with the end goal of being able to debug and reproduce
    results. For example, you could provide sample sizes, learning rates, etc. in the description. The default
    implementation only provides a name for the stage.

    Furthermore, you can implement the `pre_check` method to provide requirements that the input query or database system
    have to satisfy for the optimization stage to work properly. For example, if your hook only works for equi-join
    predicates, the pre-check can verify that the input query contains only such predicates. The benchmarking tools will
    make sure that only supported queries are passed to the stage.

    Finally, optimization stages provide a number of methods related to training. Specifically, each stage can specify that
    it needs to be trained on the database, the workload, or some sort of training samples in order to work properly.
    This is realized once again by a set of methods that you can implement to provide the actual training logic. The
    benchmarking tools will analyze the final optimization pipeline and make sure to initialize all of its stages with the
    appropriate kind of training data. The training methods generally come in pairs of two: one method to perform the
    actual training logic and another method to indicate whether the training process has already been completed. If you
    implement one of the training methods, you always have to implement the other one as well. Otherwise PostBOUND will
    raise an error when creating an instance of your stage. The training methods should return a `TrainingMetrics` object
    that contains information about the training process, e.g. how long it took, how many samples were used, etc. This
    information is included in the benchmark results for later analysis. PostBOUND does not make any assumptions about the
    kind of information that is included in the metrics object, so you should include whatever makes sense for your
    specific training process.

    The entire training process is completely optional. If you do not require any kind of training, you don't need to do
    anything.

    Data-driven Training
    --------------------
    This kind of training gives you access to the target database. You can execute arbitrary queries, fetch statistics,
    analyze the schema, etc. For example, this can be used to implement new kinds of statistics for cardinality estimation.
    To use data-driven training, implement the `fit_database` and `database_fit_completed` methods.

    Workload-based Training
    -----------------------
    This kind of training gives you access to the entire workload of queries that will be optimized. Since this is a severe
    leak of the test set, you should only use it to extract general information about the workload. For example, many
    research ideas need to know which joins are executed in the workload, or which columns are used for specific filter
    predicates. To use workload-based training, implement the `fit_workload` and `workload_fit_completed` methods.

    Sample-based Training
    ----------------------
    This kind of training implements traditional offline training based on a set of pre-computed training samples. Samples
    can contain arbitrary information. For example, a learned cardinality estimator might require pairs of SQL queries and
    their actual cardinalities as training samples. Since multiple optimization stages might require similar training data,
    PostBOUND implements a flexible system to describe the required training samples. Therefore, you need to implement
    three methods to use sample-based training: the usual `fit_samples` and `sample_fit_completed` methods, as well as a
    `sample_spec` method. This method provides a description of the kind of information that is required for training. The
    benchmarking tools will analyze the available training data and make sure to provide the appropriate samples to the
    stage.

    Online Training
    ---------------
    This kind of training allows you to learn from the actual execution of past queries, live during the benchmark.
    The benchmarking tools will provide the executed query, its execution time, and the raw result set. This can be used to
    implement a wide range of reinforcement learning-style approaches. To use online training, implement the
    `learn_from_feedback` and `uses_online_learning` methods.
    """

    def __new__(cls, *args, **kwargs) -> Self:
        missing = _missing_method_impls(
            cls, OptimizationStage, methods=["fit_database", "database_fit_completed"]
        )
        if missing:
            raise NotImplementedError(
                f"Optimization stage '{cls.__name__}' needs to implement all methods for "
                f"data-based training, but is currently lacking an implementation for {missing}"
            )

        missing = _missing_method_impls(
            cls, OptimizationStage, methods=["fit_workload", "workload_fit_completed"]
        )
        if missing:
            raise NotImplementedError(
                f"Optimization stage '{cls.__name__}' needs to implement all methods for "
                f"workload-based training, but is currently lacking an implementation for {missing}"
            )

        missing = _missing_method_impls(
            cls,
            OptimizationStage,
            methods=["fit_samples", "sample_spec", "sample_fit_completed"],
        )
        if missing:
            raise NotImplementedError(
                f"Optimization stage '{cls.__name__}' needs to implement all methods for "
                f"sample-based training, but is currently lacking an implementation for {missing}"
            )

        return super().__new__(cls)

    def __init__(self, name: str = "") -> None:
        self.name = name if name else type(self).__name__

    def fit_database(self, database: Database) -> TrainingMetrics:
        """Performs training based on the target database.

        This method is automatically called by the benchmarking tools before the actual optimization process starts. The
        training process can include arbitrary interactions with the database, e.g. executing queries, fetching statistics,
        analyzing the schema, etc.

        Notes
        -----
        If this method is implemented, `database_fit_completed` method has to be implemented as well. Otherwise, PostBOUND will
        raise an error when an instance of this stage is created.
        """
        raise NotImplementedError(
            f"OptimizationStage {self.name} does not learn from the database"
        )

    def database_fit_completed(self) -> bool:
        """Checks, if a data-driven optimization stage has already been trained on the target database.

        Notes
        -----
        If this method is implemented, `fit_database` method has to be implemented as well. Otherwise, PostBOUND will raise an
        error when an instance of this stage is created.
        """
        raise NotImplementedError(
            f"OptimizationStage {self.name} does not learn from the database"
        )

    @classmethod
    def requires_data_training(cls) -> bool:
        """Checks, if this optimization stage supports data-driven training.

        In contrast to `database_fit_completed`, this method does not check whether a specific instance of the optimization
        stage has already been trained on the target database, but rather whether the stage in general supports data-driven
        training.

        Notes
        -----
        This method uses reflection on the optimization stage and does not need to be implemented/overridden by the client.
        """
        return getattr(cls, "fit_database", None) != OptimizationStage.fit_database

    def fit_workload(self, queries: Workload, database: Database) -> TrainingMetrics:
        """Performs training based on the entire workload of queries.

        This method is automatically called by the benchmarking tools before the actual optimization process starts. The
        training process can include arbitrary interactions with the workload, e.g. analyzing which joins are executed, or
        which columns are used for specific filter predicates. Since this is a severe leak of the test set, it should only be
        used to extract general information about the workload.

        Notes
        -----
        If this method is implemented, `workload_fit_completed` method has to be implemented as well. Otherwise, PostBOUND will
        raise an error when an instance of this stage is created.
        """
        raise NotImplementedError(
            f"OptimizationStage {self.name} does not learn from workloads"
        )

    def workload_fit_completed(self) -> bool:
        """Checks, if a workload-driven optimization stage has already been trained on the target workload.

        Notes
        -----
        If this method is implemented, `fit_workload` method has to be implemented as well. Otherwise, PostBOUND will raise an
        error when an instance of this stage is created.
        """
        raise NotImplementedError(
            f"OptimizationStage {self.name} does not learn from workloads"
        )

    @classmethod
    def requires_workload_training(cls) -> bool:
        """Checks, if this optimization stage supports workload-driven training.

        In contrast to `workload_fit_completed`, this method does not check whether a specific instance of the optimization
        stage has already been trained, but rather whether the stage in general supports workload-driven training.

        Notes
        -----
        This method uses reflection on the optimization stage and does not need to be implemented/overridden by the client.
        """
        return getattr(cls, "fit_workload", None) != OptimizationStage.fit_workload

    def fit_samples(self, samples: TrainingData) -> TrainingMetrics:
        """Performs training based on a set of pre-computed training samples.

        This method is automatically called by the benchmarking tools before the actual optimization process starts. It is
        completely up to the implementation to decide what to do with the training samples.

        To make sure that the benchmarking tools provide the appropriate training samples, the `sample_spec` method is used.
        This method must describe the kind of data required for training.

        Notes
        -----
        If this method is implemented, `sample_spec` and `sample_fit_completed` methods have to be implemented as well.
        Otherwise, PostBOUND will raise an error when an instance of this stage is created.
        """
        raise NotImplementedError(
            f"OptimizationStage {self.name} does not require training samples"
        )

    def sample_spec(self) -> TrainingSpec:
        """Describes the structure of the training samples that are required to train this optimization stage.

        PostBOUND uses a tabular model for training data. The `TrainingSpec` describes what columns need to be present in
        a dataset. However, we currently cannot enforce any specific semantics for the columns. This needs to be handled by
        the user. This is a pragmatic choice to prevent us from implementing a full meta-model of data sets and to provide a
        rather lightweight interface to the user.

        Notes
        -----
        If this method is implemented, `fit_samples` and `sample_fit_completed` methods have to be implemented as well.
        Otherwise, PostBOUND will raise an error when an instance of this stage is created.
        """
        raise NotImplementedError(
            f"OptimizationStage {self.name} does not require training samples"
        )

    def sample_fit_completed(self) -> bool:
        """Checks, if a sample-driven optimization stage has already been trained.

        Notes
        -----
        If this method is implemented, `fit_samples` and `sample_spec` methods have to be implemented as well.
        Otherwise, PostBOUND will raise an error when an instance of this stage is created.
        """
        raise NotImplementedError(
            f"OptimizationStage {self.name} does not require training samples"
        )

    @classmethod
    def requires_sample_training(cls) -> bool:
        """Checks, if this optimization stage supports sample-based training.

        In contrast to `sample_fit_completed`, this method does not check whether a specific instance of the optimization
        stage has already been trained, but rather whether the stage in general supports data-driven training.

        Notes
        -----
        This method uses reflection on the optimization stage and does not need to be implemented/overridden by the client.
        """
        return getattr(cls, "fit_samples", None) != OptimizationStage.fit_samples

    def learn_from_feedback(
        self, query: SqlQuery, result_set: ResultSet, *, exec_time: TimeMs
    ) -> TrainingMetrics:
        """Performs online learning based on the execution of a past query.

        This method is automatically called by the benchmarking tools after each query is executed. Only valid runs are
        considered. If the query timed out or produced an error, this method will not be called.

        Parameters
        ----------
        query : SqlQuery
            The query that was executed, exactly as it was passed to the database system for execution.
        result_set : ResultSet
            The raw result set that was returned by the database system after executing the query. This is not processed in any
            way, so it is up to the implementation to extract any relevant information from it.
        exec_time : TimeMs
            The execution time of the query in milliseconds. This is measured directly by the benchmarking tools and will
            always be a valid number (i.e. not *NaN*, negative, nor infinite).


        Notes
        -----
        This method stands on its own and does not require any other methods to be implemented.
        """
        raise NotImplementedError(
            f"OptimizationStage {self.name} does not learn online"
        )

    @classmethod
    def uses_online_feedback(cls) -> bool:
        """Checks, if this optimization stage supports online learning.

        In contrast to `learn_from_feedback`, this method does not check whether a specific instance of the optimization
        stage has already been trained, but rather whether the stage in general supports online learning.

        Notes
        -----
        This method uses reflection on the optimization stage and does not need to be implemented/overridden by the client.
        """
        return (
            getattr(cls, "learn_from_feedback", None)
            != OptimizationStage.learn_from_feedback
        )

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return EmptyPreCheck()

    def describe(self) -> jsondict:
        """Provides a JSON-serializable representation of the specific strategy, as well as important parameters.

        Returns
        -------
        jsondict
            The description

        See Also
        --------
        OptimizationPipeline.describe
        """
        return {"name": self.name}

    def __json__(self) -> jsondict:
        return self.describe()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.name


class CompleteOptimizationAlgorithm(OptimizationStage, abc.ABC):
    """Constructs an entire query plan for an input query in one integrated optimization process.

    This stage closely models the behaviour of traditional optimization algorithms, e.g. based on dynamic programming.
    Implement the `optimize_query` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.

    Notes
    -----
    When implementing this class, make sure to call *super().__init__* to ensure that all of the
    internal data is set up properly.
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def optimize_query(self, query: SqlQuery) -> QueryPlan:
        """Constructs the optimized execution plan for an input query.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize

        Returns
        -------
        QueryPlan
            The optimized query plan
        """
        raise NotImplementedError


class JoinOrderOptimization(OptimizationStage, abc.ABC):
    """The join order optimization generates a complete join order for an input query.

    This is the first step in a multi-stage optimizer design.
    Implement the `optimize_join_order` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.

    Notes
    -----
    When implementing this class, make sure to call *super().__init__* to ensure that all of the
    internal data is set up properly.


    See Also
    --------
    postbound.MultiStageOptimizationPipeline
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def optimize_join_order(self, query: SqlQuery) -> Optional[JoinTree]:
        """Performs the actual join ordering process.

        The join tree can be further annotated with an initial operator assignment, if that is an inherent part of
        the specific optimization strategy. However, this is generally discouraged and the multi-stage pipeline will discard
        such operators to prepare for the subsequent physical operator selection.

        Other than the join order and operator assignment, the algorithm should add as much information to the join
        tree as possible, e.g. including join conditions and cardinality estimates that were calculated for the
        selected joins. This enables other parts of the optimization process to re-use that information.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize

        Returns
        -------
        Optional[LogicalJoinTree]
            The join order. If for some reason there is no valid join order for the given query (e.g. queries with just a
            single selected table), `None` can be returned. Otherwise, the selected join order has to be described using a
            `JoinTree`.
        """
        raise NotImplementedError


class JoinOrderOptimizationError(RuntimeError):
    """Error to indicate that something went wrong while optimizing the join order.

    Parameters
    ----------
    query : SqlQuery
        The query for which the optimization failed
    message : str, optional
        A message containing more details about the specific error. Defaults to an empty string.
    """

    def __init__(self, query: SqlQuery, message: str = "") -> None:
        super().__init__(
            f"Join order optimization failed for query {query}"
            if not message
            else message
        )
        self.query = query


class PhysicalOperatorSelection(OptimizationStage, abc.ABC):
    """The physical operator selection assigns scan and join operators to the tables of the input query.

    This is the second stage in the two-phase optimization process, and takes place after the join order has been
    determined.
    Implement the `select_physical_operators` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.

    Notes
    -----
    When implementing this class, make sure to call *super().__init__* to ensure that all of the
    internal data is set up properly.


    See Also
    --------
    postbound.MultiStageOptimizationPipeline
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def select_physical_operators(
        self, query: SqlQuery, join_order: Optional[JoinTree]
    ) -> PhysicalOperatorAssignment:
        """Performs the operator assignment.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize
        join_order : Optional[JoinTree]
            The selected join order of the query

        Returns
        -------
        PhysicalOperatorAssignment
            The operator assignment. If for some reason no operators can be assigned, an empty assignment can be returned

        Notes
        -----
        The operator selection should handle a `None` join order gracefully. This can happen if the query does not require
        any joins (e.g. processing of a single table.

        Depending on the specific optimization settings, it is also possible to raise an error if such a situation occurs and
        there is no reasonable way to deal with it.
        """
        raise NotImplementedError


class ParameterGeneration(OptimizationStage, abc.ABC):
    """The parameter generation assigns additional metadata to a query plan.

    Such parameters do not influence the previous choice of join order and physical operators directly, but affect their
    specific implementation. Therefore, this is an optional final step in a multi-stage optimization process.
    Implement the `generate_plan_parameters` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.

    Notes
    -----
    When implementing this class, make sure to call *super().__init__* to ensure that all of the
    internal data is set up properly.


    See Also
    --------
    postbound.MultiStageOptimizationPipeline
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def generate_plan_parameters(
        self,
        query: SqlQuery,
        join_order: Optional[JoinTree],
        operator_assignment: Optional[PhysicalOperatorAssignment],
    ) -> PlanParameterization:
        """Executes the actual parameterization.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize
        join_order : Optional[JoinTree]
            The selected join order for the query.
        operator_assignment : Optional[PhysicalOperatorAssignment]
            The selected operators for the query

        Returns
        -------
        PlanParameterization
            The parameterization. If for some reason no parameters can be determined, an empty parameterization can be returned

        Notes
        -----
        Since this is the final stage of the optimization process, a number of special cases have to be handled:

        - the previous phases might not have determined any join order or operator assignment
        - there might not have been a physical operator selection, but only a join ordering (which potentially included
          an initial selection of physical operators)
        - there might not have been a join order optimization, but only a selection of physical operators
        - both join order and physical operators might have been optimized (in which case only the actual operator
          assignment matters, not any assignment contained in the join order)
        """
        raise NotImplementedError


class CardinalityEstimator(ParameterGeneration, abc.ABC):
    """The cardinality estimator calculates how many tuples specific operators will produce.

    Implement the `calculate_estimate` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.
    In addition, use `initialize` and `cleanup` methods to implement any necessary setup and teardown logic for the
    current query.

    See Also
    --------
    TextBookOptimizationPipeline
    ParameterGeneration

    Notes
    -----
    If you only care about cardinality estimation, you should generally use this class in a
    `MultiStageOptimizationPipeline` instead of a `TextBookOptimizationPipeline`. This is because a multi-stage pipeline
    has a simple control flow from one stage to the next. This allows us to just generate cardinality estimates for all
    possible intermediates if no join order is given, or just for the intermediates defined in a specific join order
    otherwise. In contrast, the textbook pipeline is controlled by the plan enumerator which decides which plans to
    construct and by extension for which intermediates cardinality estimates are required. However, the framework
    implementation does not provide any way for the actual query optimizer of a database system to hook back into the
    framework to request such data. Therefore, we rely on emulating the behaviour of the actual plan enumerator of the
    target database system (unless a enumerator is explicitly provided). While our approximation for Postgres works quite
    well, it is not entirely accurate and other backends are much less supported.

    The default implementation of all methods related to the `ParameterGeneration` either request cardinality estimates for
    all possible intermediate results (in the `estimate_cardinalities` method), or for exactly those intermediates that are
    defined in a specific join order (in the `generate_plan_parameters` method that implements the protocol of the
    `ParameterGeneration` class). Therefore, developers working on their own cardinality estimation algorithm only need to
    implement the `calculate_estimate` method. All related processes are provided by the generator with reasonable default
    strategies.

    However, special care is required when considering cross products: depending on the setting intermediates can either
    allow cross products at all stages (by passing ``allow_cross_products=True`` during instantiation), or to disallow them
    entirely. Therefore, the `calculate_estimate` method should act accordingly. Implementations of this class should pass
    the appropriate parameter value to the super *__init__* method. If they support both scenarios, the parameter can also
    be exposed to the client. In either case, make sure to call *super().__init__* to ensure that all of the internal data
    is set up properly.
    """

    def __init__(self, *, allow_cross_products: bool = False) -> None:
        super().__init__()
        self.allow_cross_products = allow_cross_products
        self.target_db: Database = None  # type: ignore
        self.query: SqlQuery = None  # type: ignore

    @abc.abstractmethod
    def calculate_estimate(
        self, query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        """Determines the cardinality of a specific intermediate.

        Parameters
        ----------
        query : SqlQuery
            The query being optimized
        intermediate : TableReference | Iterable[TableReference]
            The intermediate for which the cardinality should be estimated. All filter predicates, etc. that are applicable
            to the intermediate can be assumed to be applied.

        Returns
        -------
        Cardinality
            The estimated cardinality of the specific intermediate
        """
        raise NotImplementedError

    def initialize(self, target_db: Database, query: SqlQuery) -> None:
        """Hook method that is called before the actual optimization process starts.

        This method can be overwritten to set up any necessary data structures, etc. and will be called before each query.
        The default implementation stores the target database and query as attributes for later use.

        Parameters
        ----------
        target_db : Database
            The database for which the optimized queries should be generated.
        query : SqlQuery
            The query to be optimized
        """
        self.target_db = target_db
        self.query = query

    def cleanup(self) -> None:
        """Hook method that is called after the optimization process has finished.

        This method can be overwritten to remove any temporary state that was specific to the last query being optimized
        and should not be shared with later queries.

        The default implementation removes the references to the target database and query.
        """
        self.target_db = None  # type: ignore
        self.query = None  # type: ignore

    def generate_intermediates(
        self, query: SqlQuery
    ) -> Generator[frozenset[TableReference], None, None]:
        """Provides all intermediate results of a query.

        The inclusion of cross-products between arbitrary tables can be configured via the `allow_cross_products` attribute.

        Parameters
        ----------
        query : SqlQuery
            The query for which to generate the intermediates

        Yields
        ------
        Generator[frozenset[TableReference], None, None]
            The intermediates

        Warnings
        --------
        The default implementation of this method does not work for queries that naturally contain cross products. If such a
        query is passed, no intermediates with tables from different partitions of the join graph are yielded.
        """
        preds = query.predicates()
        for candidate_join in util.powerset(query.tables()):
            if not candidate_join:
                # skip empty set (which is an artefact of the powerset method)
                continue
            if not self.allow_cross_products and not preds.joins_tables(candidate_join):
                continue
            yield frozenset(candidate_join)

    def estimate_cardinalities(self, query: SqlQuery) -> PlanParameterization:
        """Produces all cardinality estimates for a specific query.

        The default implementation of this method delegates the actual estimation to the `calculate_estimate` method. It is
        called for each intermediate produced by `generate_intermediates`.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize

        Returns
        ------
        PlanParameterization
            A parameterization containing cardinality hints for all intermediates. Other attributes of the parameterization are
            not modified.
        """
        parameterization = PlanParameterization()
        for join in self.generate_intermediates(query):
            estimate = self.calculate_estimate(query, join)
            if not math.isnan(estimate):
                parameterization.add_cardinality(join, estimate)
        return parameterization

    def generate_plan_parameters(
        self,
        query: SqlQuery,
        join_order: Optional[JoinTree],
        operator_assignment: Optional[PhysicalOperatorAssignment],
    ) -> PlanParameterization:
        if join_order is None:
            return self.estimate_cardinalities(query)

        parameterization = PlanParameterization()
        for intermediate in join_order.iternodes():
            estimate = self.calculate_estimate(query, intermediate.tables())
            if not math.isnan(estimate):
                parameterization.add_cardinality(intermediate.tables(), estimate)

        return parameterization


class CostModel(OptimizationStage, abc.ABC):
    """The cost model estimates how expensive computing a certain query plan is.

    Implement the `estimate_cost` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.
    In addition, use `initialize` and `cleanup` methods to implement any necessary setup and teardown logic for the
    current query.

    Notes
    -----
    When implementing this class, make sure to call *super().__init__* to ensure that all of the
    internal data is set up properly.


    See Also
    --------
    postbound.TextBookOptimizationPipeline
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def estimate_cost(self, query: SqlQuery, plan: QueryPlan) -> Cost:
        """Computes the cost estimate for a specific plan.

        The following conventions are used for the estimation: the root node of the plan will not have any cost set. However,
        all input nodes will have already been estimated by earlier calls to the cost model. Hence, while estimating the cost
        of the root node, all earlier costs will be available as inputs. It is further assumed that all nodes already have
        associated cardinality estimates.
        This method explicitly does not make any assumption regarding the relationship between query and plan. Specifically,
        it does not assume that the plan is capable of computing the entire result set nor a correct result set. Instead,
        the plan might just be a partial plan that computes a subset of the query (e.g. a join of some of the tables).
        It is the implementation's responsibility to figure out the appropriate course of action.

        It is not the responsibility of the cost model to set the estimate on the plan, this is the task of the enumerator
        (which can decide whether the plan should be considered any further).

        Parameters
        ----------
        query : SqlQuery
            The query being optimized
        plan : QueryPlan
            The plan to estimate.

        Returns
        -------
        Cost
            The estimated cost
        """
        raise NotImplementedError

    def initialize(self, target_db: Database, query: SqlQuery) -> None:
        """Hook method that is called before the actual optimization process starts.

        This method can be overwritten to set up any necessary data structures, etc. and will be called before each query.

        Parameters
        ----------
        target_db : Database
            The database for which the optimized queries should be generated.
        query : SqlQuery
            The query to be optimized
        """
        pass

    def cleanup(self) -> None:
        """Hook method that is called after the optimization process has finished.

        This method can be overwritten to remove any temporary state that was specific to the last query being optimized
        and should not be shared with later queries.
        """
        pass

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that input query or database system have to satisfy for the optimizer to work properly.

        Returns
        -------
        OptimizationPreCheck
            The check instance. Can be an empty check if no specific requirements exist.
        """
        return EmptyPreCheck()


class PlanEnumerator(OptimizationStage, abc.ABC):
    """The plan enumerator traverses the space of different candidate plans and ultimately selects the optimal one.

    Implement the `generate_execution_plan` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.

    Notes
    -----
    When implementing this class, make sure to call *super().__init__* to ensure that all of the
    internal data is set up properly.


    See Also
    --------
    postbound.TextBookOptimizationPipeline
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def generate_execution_plan(
        self,
        query: SqlQuery,
        *,
        cost_model: CostModel,
        cardinality_estimator: CardinalityEstimator,
    ) -> QueryPlan:
        """Computes the optimal plan to execute the given query.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize
        cost_model : CostModel
            The cost model to compare different candidate plans
        cardinality_estimator : CardinalityEstimator
            The cardinality estimator to calculate the sizes of intermediate results

        Returns
        -------
        QueryPlan
            The query plan

        Notes
        -----
        The precise generation "style" (e.g. top-down vs. bottom-up, complete plans vs. plan fragments, etc.) is completely up
        to the specific algorithm. Therefore, it is really hard to provide a more expressive interface for the enumerator
        beyond just generating a plan. Generally the enumerator should query the cost model to compare different candidates.
        The top-most operator of each candidate will usually not have a cost estimate set at the beginning and it is the
        enumerator's responsibility to set the estimate correctly. The `jointree.update_cost_estimate` function can be used to
        help with this.
        """
        raise NotImplementedError


class IncrementalOptimizationStep(OptimizationStage, abc.ABC):
    """Incremental optimization allows to chain different smaller optimization strategies.

    Each step receives the query plan of its predecessor and can change its decisions in arbitrary ways. For example, this
    scheme can be used to gradually correct mistakes or risky decisions of individual optimizers.

    Implement the `optimize_query` method to provide the actual optimization logic.
    The `describe` and `pre_check` methods should be overridden to provide metadata about the specific algorithm for
    benchmarking and to ensure that the input query and database system are compatible with the algorithm.

    Notes
    -----
    When implementing this class, make sure to call *super().__init__* to ensure that all of the
    internal data is set up properly.
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def optimize_query(self, query: SqlQuery, current_plan: QueryPlan) -> QueryPlan:
        """Determines the next query plan.

        If no further optimization steps are configured in the pipeline, this is also the final query plan.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize
        current_plan : QueryPlan
            The execution plan that has so far been built by predecessor strategies. If this step is the first step in the
            optimization pipeline, this might also be a plan from the target database system

        Returns
        -------
        QueryPlan
            The optimized plan
        """
        raise NotImplementedError


class _CompleteAlgorithmEmulator(CompleteOptimizationAlgorithm):
    """Utility to use implementations of staged optimization strategies when a complete algorithm is expected.

    The emulation is enabled by supplying ``None`` values at all places where the stage expects input from previous stages.
    The output of the actual stage is used to obtain a query plan which in turn is used to generate the required optimizer
    information.

    Parameters
    ----------
    database : Optional[Database], optional
        The database for which the queries should be executed. This is required to obtain complete query plans for the input
        queries. If omitted, the database is inferred from the database pool.
    join_order_optimizer : Optional[JoinOrderOptimization], optional
        The join order optimizer if any.
    operator_selection : Optional[PhysicalOperatorSelection], optional
        The physical operator selector if any.
    plan_parameterization : Optional[ParameterGeneration], optional
        The plan parameterization (e.g. cardinality estimator) if any.

    Raises
    ------
    ValueError
        If all stages are ``None``.

    """

    def __init__(
        self,
        database: Optional[Database] = None,
        *,
        join_order_optimizer: Optional[JoinOrderOptimization] = None,
        operator_selection: Optional[PhysicalOperatorSelection] = None,
        plan_parameterization: Optional[ParameterGeneration] = None,
    ) -> None:
        super().__init__()
        self.database = (
            database
            if database is not None
            else DatabasePool.get_instance().current_database()
        )
        if all(
            stage is None
            for stage in (
                join_order_optimizer,
                operator_selection,
                plan_parameterization,
            )
        ):
            raise ValueError("Exactly one stage has to be given")
        self._join_order_optimizer = join_order_optimizer
        self._operator_selection = operator_selection
        self._plan_parameterization = plan_parameterization

    def stage(
        self,
    ) -> JoinOrderOptimization | PhysicalOperatorSelection | ParameterGeneration:
        """Provides the actually specified stage.

        Returns
        -------
        JoinOrderOptimization | PhysicalOperatorSelection | ParameterGeneration
            The optimization stage.
        """
        if self._join_order_optimizer is not None:
            return self._join_order_optimizer

        if self._operator_selection is not None:
            return self._operator_selection

        assert self._plan_parameterization is not None
        return self._plan_parameterization

    def optimize_query(self, query: SqlQuery) -> QueryPlan:
        join_order = (
            self._join_order_optimizer.optimize_join_order(query)
            if self._join_order_optimizer is not None
            else None
        )
        physical_operators = (
            self._operator_selection.select_physical_operators(query, None)
            if self._operator_selection is not None
            else None
        )
        plan_params = (
            self._plan_parameterization.generate_plan_parameters(query, None, None)
            if self._plan_parameterization is not None
            else None
        )
        hinted_query = self.database.hinting().generate_hints(
            query,
            join_order=join_order,
            physical_operators=physical_operators,
            plan_parameters=plan_params,
        )
        return self.database.optimizer().query_plan(hinted_query)

    def describe(self) -> jsondict:
        return self.stage().describe()

    def pre_check(self) -> OptimizationPreCheck:
        return self.stage().pre_check()


def as_complete_algorithm(
    stage: JoinOrderOptimization | PhysicalOperatorSelection | ParameterGeneration,
    *,
    database: Optional[Database] = None,
) -> CompleteOptimizationAlgorithm:
    """Enables using a partial optimization stage in situations where a complete optimizer is expected.

    This emulation is achieved by using the partial stage to obtain a partial query plan. The target database system is then
    tasked with filling the gaps to construct a complete execution plan.

    Basically this method is syntactic sugar in situations where a `MultiStageOptimizationPipeline` would be filled with only a
    single stage. Using `as_complete_algorithm`, the construction of an entire pipeline can be omitted. Furthermore it can seem
    more natural to "convert" the stage into a complete algorithm in this case.

    Parameters
    ----------
    stage : JoinOrderOptimization | PhysicalOperatorSelection | ParameterGeneration
        The stage that should become a complete optimization algorithm
    database : Optional[Database], optional
        The target database to execute the optimized queries in. This is required to fill the gaps of the partial query plans.
        If the database is omitted, it will be inferred based on the database pool.

    Returns
    -------
    CompleteOptimizationAlgorithm
        A emulated optimization algorithm for the optimization stage
    """
    join_order_optimizer = stage if isinstance(stage, JoinOrderOptimization) else None
    operator_selection = stage if isinstance(stage, PhysicalOperatorSelection) else None
    parameter_generation = stage if isinstance(stage, ParameterGeneration) else None
    return _CompleteAlgorithmEmulator(
        database,
        join_order_optimizer=join_order_optimizer,
        operator_selection=operator_selection,
        plan_parameterization=parameter_generation,
    )
