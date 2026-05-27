"""Utilities to optimize and execute queries and workloads in a reproducible and transparent manner."""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import natsort
import numpy as np
import pandas as pd

from . import transform, util
from ._pipelines import (
    IntegratedOptimizationPipeline,
    MultiStageOptimizationPipeline,
    OptimizationPipeline,
    TextBookOptimizationPipeline,
)
from ._stages import (
    CompleteOptimizationAlgorithm,
    CostModel,
    JoinOrderOptimization,
    OptimizationStage,
    ParameterGeneration,
    PhysicalOperatorSelection,
    PlanEnumerator,
)
from .db import (
    Database,
    DatabasePool,
    PrewarmingSupport,
    ResultSet,
    StopwatchSupport,
    TimeoutSupport,
    simplify_result_set,
)
from .qal import SqlQuery
from .train import TrainingData, TrainingDataRepository
from .util.jsonize import Jsonizable
from .workloads import Workload, generate_workload

PredefLogger = Literal["tqdm"]
"""Pre-defined loggers that can be used to track progress during workload execution."""

ErrorHandling = Literal["raise", "log", "ignore"]
"""How to handle errors during optimization or execution:

- *raise*: Raise the exception immediately
- *log*: Include the failed query in the resulting data frame, just like successful queries. The *status* column will indicate
   the specific error and the *failure reason* column will contain the exception message.
- *ignore*: Silently ignore the error and do not include the failed query in the resulting data frame
"""

ExecStatus = Literal["ok", "timeout", "optimization-error", "execution-error"]
"""Describes the result of a query execution:

- *ok*: The query was executed successfully
- *timeout*: The query was cancelled due to a timeout
- *optimization-error*: The query could not be optimized by PostBOUND
- *execution-error*: The query could not be executed by the database system

For errors, the actual reason is contained in the `failure_reason` column of the resulting data frame.
"""


@dataclass
class ExecutionResult:
    """Captures all relevant components of a query optimization and execution result."""

    query: SqlQuery
    """The query that was executed. If the query was optimized and transformed, these modifications are included."""

    status: ExecStatus = "ok"
    """Whether the query was executed successfully or not."""

    query_result: ResultSet | None = None
    """The result set of the query or *None* if the query failed."""

    optimization_time: float = np.nan
    """The time in seconds it took to optimized the query by PostBOUND.

    This does not account for optimization by the actual database system and depends heavily on the quality of the
    implementation of the optimization strategies.

    For queries that were not optimized within PostBOUND, this value is *NaN*.
    """

    execution_time: float = np.nan
    """The time in seconds it took to execute the (potentially optimized) query by the actual database system.

    This execution time includes the entire end-to-end processing, i.e. starting with supplying the query to the
    database until the last byte of the result set was transferred back to PostBOUND. Therefore, this duration also
    includes the optimization time by the database system, as well as the entire time for data transfer.

    A value of *Inf* indicates that the query did not complete successfully and was cancelled due to a timeout. *NaN* encodes
    a failure during optimization or execution. See `status` for more details.
    """

    @staticmethod
    def passed(
        query: SqlQuery,
        *,
        query_result: ResultSet,
        execution_time: float,
        optimization_time: float = np.nan,
    ) -> ExecutionResult:
        """Constructs an `ExecutionResult` for a successfully executed query.

        The optimization time can be omitted if the query was not optimized in PostBOUND.
        """
        return ExecutionResult(
            query=query,
            status="ok",
            query_result=query_result,
            execution_time=execution_time,
            optimization_time=optimization_time,
        )

    @staticmethod
    def execution_error(
        query: SqlQuery, *, optimization_time: float = np.nan
    ) -> ExecutionResult:
        """Constructs an `ExecutionResult` for a query that failed during execution."""
        return ExecutionResult(
            query=query,
            status="execution-error",
            query_result=None,
            execution_time=np.nan,
            optimization_time=optimization_time,
        )

    @staticmethod
    def optimization_error(query: SqlQuery) -> ExecutionResult:
        return ExecutionResult(
            query=query,
            status="optimization-error",
            query_result=None,
            execution_time=np.nan,
            optimization_time=np.nan,
        )


class QueryPreparation:
    """This service handles transformations of input queries that are executed before running the query.

    These transformations mostly ensure that all queries in a workload provide the same type of result even in face
    of input queries that are structured slightly differently. For example, the preparation service can transform
    all the queries to be executed as *EXPLAIN* or *COUNT(\\*)* queries. Furthermore, the preparation service can
    store SQL statements that have to be executed before running the query. For example, a statement that disables
    parallel execution could be supplied here.

    Parameters
    ----------
    projection : Literal["none", "star", "count_star", "\\*", "count(\\*)"], optional
        Modify the *SELECT* clause of all queries to provide a uniform projection. Allowed values are *none* (no modification),
        *star* or *\\** for *SELECT \\** and *count_star* or *count(\\*)* for *SELECT COUNT(\\*)*. Defaults to *none*.

    output : Literal["default", "explain", "analyze", "explain_analyze"], optional
        What kind of results to gather. *default* executes the queries as they are, *explain* transforms all queries to
        *EXPLAIN* queries, *analyze* or *explain_analyze* transforms all queries to *EXPLAIN ANALYZE* queries.

    explain : bool, optional
        Whether to force all queries to be executed as *EXPLAIN* queries, by default *False*

        .. deprecated:: 0.21.0
            This option is deprecated in favor of the more versatile `output` option.

    count_star : bool, optional
        Whether to force all queries to be executed as *COUNT(\\*)* queries, overwriting their default projection. Defaults to
        *False*

        .. deprecated:: 0.21.0
            This option is deprecated in favor of the more versatile `projection` option.

    analyze : bool, optional
        Whether to force all queries to be executed as ``EXPLAIN ANALYZE`` queries. Setting this option implies `explain`,
        which therefore does not need to set manually. Defaults to *False*

        .. deprecated:: 0.21.0
            This option is deprecated in favor of the more versatile `output` option.

    prewarm : bool, optional
        For database systems that support prewarming, this inflates the buffer pool with pages from the prepared query.

    preparatory_statements : Optional[list[str]], optional
        Statements that are executed as-is on the database connection before running the query, by default *None*

    See Also
    --------
    db.PrewarmingSupport : Technical details on how prewarming is implemented in PostBOUND
    """

    def __init__(
        self,
        *,
        projection: Literal["none", "star", "count_star", "*", "count(*)"] = "none",
        output: Literal["default", "explain", "analyze", "explain_analyze"] = "default",
        explain: Optional[bool] = None,
        count_star: Optional[bool] = None,
        analyze: Optional[bool] = None,
        prewarm: bool = False,
        preparatory_statements: Optional[list[str]] = None,
    ) -> None:
        if explain is not None:
            warnings.warn(
                "The 'explain' parameter is deprecated in favor of the more versatile 'output' parameter. "
                "Please use 'output=\"explain\"' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.output = "explain" if explain else output
        else:
            self.output = output

        if analyze is not None:
            warnings.warn(
                "The 'analyze' parameter is deprecated in favor of the more versatile 'output' parameter. "
                "Please use 'output=\"analyze\"' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.output = "analyze" if analyze else self.output

        if count_star is not None:
            warnings.warn(
                "The 'count_star' parameter is deprecated in favor of the more versatile 'projection' parameter. "
                "Please use 'projection=\"count_star\"' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.projection = "count_star" if count_star else projection
        else:
            self.projection = projection

        if self.output == "explain" and prewarm:
            warnings.warn(
                "Prewarming is not compatible with EXPLAIN queries. Ignoring prewarm setting.",
                stacklevel=2,
            )
            self.prewarm = False
        else:
            self.prewarm = prewarm

        self.preparatory_stmts = preparatory_statements or []

    def prepare_query(
        self, query: SqlQuery, *, on: Optional[Database] = None
    ) -> SqlQuery:
        """Applies the selected transformations to the given input query and executes the preparatory statements

        Parameters
        ----------
        query : SqlQuery
            The query to prepare
        on : Optional[Database]
            The database to execute the preparatory statements on. If no database is given, only query transformations are
            applied.

        Returns
        -------
        SqlQuery
            The prepared query
        """
        match self.projection:
            case "none":
                pass
            case "star" | "*":
                query = transform.as_star_query(query)
            case "count_star" | "count(*)":
                query = transform.as_count_star_query(query)
            case _:
                raise ValueError(f"Unsupported projection type: {self.projection}")

        match self.output:
            case "default":
                pass
            case "explain":
                query = transform.as_explain(query)
            case "analyze" | "explain_analyze":
                query = transform.as_explain_analyze(query)

        if on is None:
            return query

        if self.prewarm:
            if not isinstance(on, PrewarmingSupport):
                warnings.warn(
                    "Ignoring prewarm setting since the database does not support prewarming",
                    stacklevel=2,
                )
            else:
                on.prewarm_tables(query.tables())

        for stmt in self.preparatory_stmts:
            on.execute_query(stmt, cache_enabled=False)

        return query

    def __json__(self) -> util.jsondict:
        return {
            "output": self.output,
            "projection": self.projection,
            "prewarm": self.prewarm,
            "preparatory_statements": self.preparatory_stmts,
        }

    def __repr__(self) -> str:
        return f"QueryPreparation(output={self.output}, projection={self.projection}, prewarm={self.prewarm}, preparatory_statements={self.preparatory_stmts})"

    def __str__(self) -> str:
        return repr(self)


def _wrap_workload(
    queries: Iterable[SqlQuery] | Workload,
) -> Workload:
    """Transforms an iterable of queries into a proper workload object to enable execution by the runner methods."""
    return queries if isinstance(queries, Workload) else generate_workload(queries)


def _wrap_optimization_stage(stage: OptimizationStage) -> OptimizationPipeline:
    """Create an appropriate optimization pipeline for a specific optimization algorithm."""
    target_db = DatabasePool.get_instance().current_database()
    match stage:
        case CompleteOptimizationAlgorithm():
            pipeline = IntegratedOptimizationPipeline(target_db)
            pipeline.setup_optimization_algorithm(stage).build()
        case (
            JoinOrderOptimization()
            | PhysicalOperatorSelection()
            | ParameterGeneration()
        ):
            pipeline = MultiStageOptimizationPipeline(target_db)
            pipeline.use(stage).build()
        case PlanEnumerator() | CostModel():
            # We don't check for CardinalityEstimator here b/c every cardest is also a ParameterGeneration instance
            # and if we don't have a plan enumerator or a cost model (as is the case here), the parameter generation is the
            # much more well-suited stage to wrap
            pipeline = TextBookOptimizationPipeline(target_db)
            pipeline.use(stage).build()
        case _:
            raise TypeError(f"Unsupported optimization stage: {stage}")
    return pipeline


@dataclass
class _BenchmarkConfig:
    workload: Workload
    target_db: Database
    optimizer: OptimizationPipeline | None
    output: Path | None
    output_args: dict
    workload_repetitions: int
    per_query_repetitions: int
    shuffled: bool
    query_prep: QueryPreparation | None
    timeout: float | None
    pre_exec_callback: Callable[[SqlQuery], None | dict] | None
    post_exec_callback: Callable[[ExecutionResult], None | dict] | None
    log: _LoggerImpl
    error_action: ErrorHandling
    start_time: datetime | None = None


def _init_benchmark_log(experiment: str, *, cfg: _BenchmarkConfig) -> None:
    if not cfg.output:
        return
    out_dir = cfg.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    start_timestamp = datetime.now()
    cfg.start_time = start_timestamp

    log_data = {
        "experiment": experiment,
        "start_time": start_timestamp.isoformat(),
        "end_time": None,
        "status": "running",
        "workload": {"name": cfg.workload.name, "queries": {}},
        "target_db": cfg.target_db.describe(),
        "optimizer": cfg.optimizer.describe() if cfg.optimizer else None,
        "result_file": cfg.output.expanduser().resolve(),
        "workload_repetitions": cfg.workload_repetitions,
        "per_query_repetitions": cfg.per_query_repetitions,
        "shuffled": cfg.shuffled,
        "query_preparation": cfg.query_prep,
        "query_timeout": cfg.timeout,
    }
    for label, query in cfg.workload.entries():
        log_data["workload"]["queries"][label] = {
            "query": str(query),
        }

    log_file = out_dir / "experiment_log.json"
    if not log_file.exists():
        with open(log_file, "w", encoding="utf-8") as f:
            util.to_json_dump([log_data], f)
        return

    with open(log_file, "r+", encoding="utf-8") as f:
        existing_logs = json.load(f)
        complete_log = existing_logs + [log_data]
        f.seek(0)
        util.to_json_dump(complete_log, f, indent=2)


def _finalize_benchmark_log(experiment: str, *, cfg: _BenchmarkConfig) -> None:
    if not cfg.output:
        return
    out_dir = cfg.output.parent
    log_file = out_dir / "experiment_log.json"
    if cfg.start_time is None:
        raise ValueError(
            f"Malformed experiment log '{log_file}': start time is missing!"
        )

    with open(log_file, "r+", encoding="utf-8") as f:
        existing_logs = json.load(f)
        for log_entry in existing_logs:
            same_experiment = log_entry["experiment"] == experiment
            same_iteration = log_entry["start_time"] == cfg.start_time.isoformat()
            if not same_experiment or not same_iteration:
                continue
            log_entry["end_time"] = datetime.now().isoformat()
            log_entry["status"] = "completed"
        f.seek(0)
        json.dump(existing_logs, f, indent=2)


def _failed_benchmark_log(
    experiment: str, *, cfg: _BenchmarkConfig, error: Exception
) -> None:
    if not cfg.output:
        return
    out_dir = cfg.output.parent
    log_file = out_dir / "experiment_log.json"
    if cfg.start_time is None:
        raise ValueError(
            f"Malformed experiment log '{log_file}': start time is missing!"
        )

    with open(log_file, "r+", encoding="utf-8") as f:
        existing_logs = json.load(f)
        for log_entry in existing_logs:
            same_experiment = log_entry["experiment"] == experiment
            same_iteration = log_entry["start_time"] == cfg.start_time.isoformat()
            if not same_experiment or not same_iteration:
                continue
            log_entry["end_time"] = datetime.now().isoformat()
            log_entry["status"] = "failed"
            log_entry["error"] = {"type": type(error).__name__, "message": str(error)}
        f.seek(0)
        json.dump(existing_logs, f, indent=2)


ExecutionTarget = Database | OptimizationPipeline | OptimizationStage
"""Specifies what to do with the workload queries:

- providing a `Database` executes the queries as-is on the database
- passing an `OptimizationPipeline` optimizes the queries using the pipeline before executing them on the target database
  of the pipeline
- passing an `OptimizationStage` generates an appropriate optimization pipeline for the stage and then proceeds as above.
  Notice that in this mode, the target database is assumed to be the current database of the `DatabasePool`.

"""


@dataclass
class _SuccessfullOptimization:
    optimized_query: SqlQuery
    optimization_time: float


@dataclass
class _FailedOptimization:
    error: Exception


@dataclass
class _NoOptimization:
    pass


_InternalOptResult = _SuccessfullOptimization | _FailedOptimization | _NoOptimization


def _optimize_query(
    query: SqlQuery, *, pipeline: Optional[OptimizationPipeline] = None
) -> _InternalOptResult:
    """Tries to run a query through the optimization pipeline while gracefully handling errors."""
    if pipeline is None:
        return _NoOptimization()

    try:
        opt_start = time.perf_counter_ns()
        optimized_query = pipeline.optimize_query(query)
        opt_end = time.perf_counter_ns()
        optimization_time = (opt_end - opt_start) / 10**9  # convert to seconds
        return _SuccessfullOptimization(
            optimized_query=optimized_query, optimization_time=optimization_time
        )
    except Exception as e:
        return _FailedOptimization(error=e)


@dataclass
class _SuccessfullExecution:
    query_result: Any
    exec_time: float


@dataclass
class _TimeoutExecution:
    timeout: float


@dataclass
class _FailedExecution:
    error: Exception


_InternalExecResult = _SuccessfullExecution | _TimeoutExecution | _FailedExecution


def _execute_query(
    query: SqlQuery,
    *,
    on: Database,
    timeout: Optional[float] = None,
    query_prep: Optional[QueryPreparation] = None,
) -> _InternalExecResult:
    """Prepares and executes a query on an actual database system while gracefully handling timeouts and errors.

    This is a simple handler that does not care about the larger control flow of the benchmarking process. It simply executes
    stuff and lets the more high-level control loops deal with the rest (e.g. proper error handling, etc).
    """
    if timeout and not isinstance(on, TimeoutSupport):
        raise ValueError(f"Database system {on} does not provide timeout support")

    if query_prep:
        query = query_prep.prepare_query(query, on=on)

    try:
        if timeout:
            exec_start = time.perf_counter_ns()
            result_set = on.execute_with_timeout(query, timeout=timeout)  # type: ignore[arg-type]
            exec_end = time.perf_counter_ns()
            if result_set is None:
                return _TimeoutExecution(timeout=timeout)
        else:
            exec_start = time.perf_counter_ns()
            result_set = on.execute_query(query, cache_enabled=False, raw=True)
            exec_end = time.perf_counter_ns()
        exec_time = (exec_end - exec_start) / 10**9  # convert to seconds

        if isinstance(on, StopwatchSupport):
            exec_time = on.last_query_runtime()

    except Exception as e:
        return _FailedExecution(error=e)

    return _SuccessfullExecution(query_result=result_set, exec_time=exec_time)


class _NoOpLogger:
    def next_workload_iter(self) -> None:
        pass

    def next_query(self, label: str) -> None:
        pass

    def next_query_rep(self) -> None:
        pass


class _CustomLogger:
    def __init__(
        self, logger: Callable[[str], None], *, workload_reps: int = 1
    ) -> None:
        self._logger = logger
        self._workload_reps = workload_reps
        self._workload_iter: int = 0

    def next_workload_iter(self) -> None:
        self._workload_iter += 1

    def next_query(self, label: str) -> None:
        log_msg = f"Now benchmarking query {label} (repetition {self._workload_iter}/{self._workload_reps})"
        self._logger(log_msg)

    def next_query_rep(self) -> None:
        pass


class _TqdmLogger:
    def __init__(
        self, *, workload_reps: int = 1, query_reps: int = 1, total_queries: int
    ) -> None:
        from tqdm import tqdm

        self._rep_progress = tqdm(total=workload_reps, desc="Workload Rep.", unit="rep")
        self._query_progress = tqdm(total=total_queries, desc="Query", unit="q")
        self._query_rep = tqdm(total=query_reps, desc="Query Rep.", unit="rep")
        self._initial: bool = True

    def next_workload_iter(self) -> None:
        if self._initial:
            self._initial = False
            return
        self._rep_progress.update(1)
        self._query_progress.reset()

    def next_query(self, label: str) -> None:
        self._query_progress.update(1)
        self._query_rep.set_description(f"Query {label}")
        self._query_rep.reset()

    def next_query_rep(self) -> None:
        self._query_rep.update(1)


_LoggerImpl = _NoOpLogger | _CustomLogger | _TqdmLogger


class _ResultSample:
    """A result sample corresponds to all executions of a single query within one workload repetition.

    It captures all relevant data (measurements or other artifacts) related to the optimization and execution of the query.
    Each sample should be generated by the `_ExecutionResults` "manager" which keeps track of most of the global state of the
    benchmarking process.
    """

    def __init__(
        self,
        *,
        query: SqlQuery,
        label: str,
        initial_exec_idx: int,
        current_workload_rep: int,
        query_prep: QueryPreparation | None,
        max_query_reps: int,
    ) -> None:
        # static data
        self.label = label
        self.query = query
        self.current_workload_rep = current_workload_rep
        self.query_prep = query_prep

        # collected data
        self.timestamps: list[datetime | None] = []
        self.status: list[ExecStatus] = []
        self.optimization_time: float = np.nan
        self.optimization_pipeline: OptimizationPipeline | None = None
        self.optimized_query: SqlQuery | None = None
        self.failure_reasons: list[str] = []
        self.result_sets: list[ResultSet | None] = []
        self.exec_times: list[float] = []
        self.db_configs: list[util.jsondict] = []

        self._pre_exec_data: list[dict] = []
        self._post_exec_data: list[dict] = []

        # internal fields
        self._max_query_reps = max_query_reps
        self._optimization_failure: Exception | None = None
        self._initial_idx = initial_exec_idx

    def optimization_results(
        self,
        *,
        optimized_query: SqlQuery,
        pipeline: OptimizationPipeline | None,
        optimization_time: float,
    ) -> None:
        self.optimization_time = optimization_time
        self.optimization_pipeline = pipeline
        self.optimized_query = optimized_query

    def optimization_failure(
        self, reason: Exception, *, pipeline: OptimizationPipeline | None
    ) -> None:
        self._optimization_failure = reason
        self.optimization_pipeline = pipeline
        self.timestamps += [None] * self._max_query_reps
        self.status += ["optimization-error"] * self._max_query_reps  # type: ignore[list-item]]
        self.failure_reasons += [str(reason)] * self._max_query_reps
        self.result_sets += [None] * self._max_query_reps
        self.exec_times += [np.nan] * self._max_query_reps
        self.db_configs += [{}] * self._max_query_reps

    def start_execution(self) -> None:
        self.timestamps.append(datetime.now())

    def add_exec_sample(
        self, result_set: ResultSet, *, exec_time: float, db_config: util.jsondict
    ) -> None:
        self.status.append("ok")
        self.failure_reasons.append("")
        self.result_sets.append(result_set)
        self.exec_times.append(exec_time)
        self.db_configs.append(db_config)

    def add_exec_timeout(self, timeout: float, *, db_config: util.jsondict) -> None:
        self.status.append("timeout")
        self.failure_reasons.append("")
        self.result_sets.append(None)
        self.exec_times.append(timeout)
        self.db_configs.append(db_config)

    def add_exec_failure(self, reason: Exception, *, db_config: util.jsondict) -> None:
        self.status.append("execution-error")
        self.failure_reasons.append(str(reason))
        self.result_sets.append(None)
        self.exec_times.append(np.nan)
        self.db_configs.append(db_config)

    def add_pre_exec_data(self, data: dict | None) -> None:
        if data is None:
            return
        self._pre_exec_data.append(data)

    def add_post_exec_data(self, data: dict | None) -> None:
        if data is None:
            return
        self._post_exec_data.append(data)

    def failed_optimization(self) -> Optional[Exception]:
        return self._optimization_failure

    def num_executions(self) -> int:
        return len(self.result_sets)

    def last_successful(self) -> bool:
        return bool(self.status and self.status[-1] in ("ok", "timeout"))

    def last_result(self) -> Optional[ExecutionResult]:
        if self._optimization_failure or not self.result_sets:
            return None

        result_set = simplify_result_set(self.result_sets[-1])
        return ExecutionResult(
            query=self.query,
            status=self.status[-1],
            query_result=result_set,
            optimization_time=self.optimization_time,
            execution_time=self.exec_times[-1],
        )

    def to_df(self, *, only_last: bool = False) -> pd.DataFrame:
        n_samples = len(self)
        if not n_samples:
            return pd.DataFrame()

        if only_last:
            result_set = simplify_result_set(self.result_sets[-1])
            rows = {
                "exec_index": [self._initial_idx + n_samples - 1],
                "label": [self.label],
                "timestamp": [self.timestamps[-1]],
                "workload_repetition": [self.current_workload_rep],
                "query_repetition": [n_samples],
                "query": [self.query],
                "status": [self.status[-1]],
                "query_result": [result_set],
                "exec_time": [self.exec_times[-1]],
                "failure_reason": [self.failure_reasons[-1]],
                "db_config": [self.db_configs[-1]],
                "query_preparation": [self.query_prep],
                "optimization_time": [self.optimization_time],
                "optimization_pipeline": [self.optimization_pipeline],
                "optimized_query": [self.optimized_query],
            }

            if self._pre_exec_data:
                pre_exec = self._pre_exec_data[-1]
                for key, val in pre_exec.items():
                    rows[key] = [val]
            if self._post_exec_data:
                post_exec = self._post_exec_data[-1]
                for key, val in post_exec.items():
                    rows[key] = [val]

        else:
            rows = {
                "exec_index": [self._initial_idx + i for i in range(n_samples)],
                "label": [self.label] * n_samples,
                "timestamp": self.timestamps,
                "workload_repetition": [self.current_workload_rep] * n_samples,
                "query_repetition": [i + 1 for i in range(n_samples)],
                "query": [self.query] * n_samples,
                "status": self.status,
                "query_result": [simplify_result_set(rs) for rs in self.result_sets],
                "exec_time": self.exec_times,
                "failure_reason": self.failure_reasons,
                "db_config": self.db_configs,
                "query_preparation": [self.query_prep] * n_samples,
                "optimization_time": [self.optimization_time] * n_samples,
                "optimization_pipeline": [self.optimization_pipeline] * n_samples,
                "optimized_query": [self.optimized_query] * n_samples,
            }

            if self._pre_exec_data:
                for key in self._pre_exec_data[0].keys():
                    rows[key] = []
                for pre_exec in self._pre_exec_data:
                    for key, val in pre_exec.items():
                        rows[key].append(val)
            if self._post_exec_data:
                for key in self._post_exec_data[0].keys():
                    rows[key] = []
                for post_exec in self._post_exec_data:
                    for key, val in post_exec.items():
                        rows[key].append(val)

        return pd.DataFrame(rows)

    def to_execution_results(self) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for i, status in enumerate(self.status):
            if status in ["optimization-error", "execution-error"]:
                continue

            sample = ExecutionResult(
                query=self.query,
                status=status,
                query_result=self.result_sets[i],
                optimization_time=self.optimization_time,
                execution_time=self.exec_times[i],
            )
            results.append(sample)
        return results

    def write_progressive(self, file: Path | None, output_args: dict) -> None:
        if file is None:
            return

        df = self.to_df(only_last=True)
        df = prepare_export(df)

        match file.suffix:
            case ".csv":
                self._write_csv(df, file, writer_args=output_args)
            case ".parquet":
                self._write_parquet(df, file, writer_args=output_args)
            case ".hdf" | ".hdf5" | ".h5":
                self._write_hdf(df, file, writer_args=output_args)
            case ".json":
                self._write_json(df, file, writer_args=output_args)

    def _write_csv(self, df: pd.DataFrame, file: Path, *, writer_args: dict) -> None:
        if file.is_file():
            df.to_csv(file, mode="a", header=False, index=False, **writer_args)
            return

        # We are the first sample to write to the output file. Create it with headers
        file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file, index=False, **writer_args)

    def _write_parquet(
        self, df: pd.DataFrame, file: Path, *, writer_args: dict
    ) -> None:
        if file.is_file():
            existing_results = pd.read_parquet(file)
            df = pd.concat([existing_results, df], ignore_index=True)
        else:
            file.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(file, index=False, **writer_args)

    def _write_hdf(self, df: pd.DataFrame, file: Path, *, writer_args: dict) -> None:
        if "key" not in writer_args:
            raise ValueError(
                "'key' must be given when writing to HDF. "
                "Please specify the key in the 'output_args' of execute_workload()!"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if file.is_file():
                df.to_hdf(file, mode="a", index=False, **writer_args)
                return

            file.parent.mkdir(parents=True, exist_ok=True)
            df.to_hdf(file, mode="w", index=False, **writer_args)

    def _write_json(self, df: pd.DataFrame, file: Path, *, writer_args: dict) -> None:
        if file.is_file():
            existing_results = pd.read_json(file)
            df = pd.concat([existing_results, df], ignore_index=True)
        else:
            file.parent.mkdir(parents=True, exist_ok=True)

        df.to_json(file, index=False, **writer_args)

    def __len__(self) -> int:
        return self.num_executions()

    def __str__(self) -> str:
        reps = self.num_executions()
        return f"{self.label} ({reps} / {self._max_query_reps})"


class _ExecutionResults:
    """
    Execution results act as the central container for all result samples that are produced during the benchmarking process.

    It is responsible for creating the result samples for new queries. The benchmarking control loops must keep the results
    up to date in the sense that they must notify it about new workload repetitions or the start of new queries.
    """

    def __init__(self, *, query_reps: int, query_prep: QueryPreparation | None) -> None:
        self.query_reps = query_reps
        self.query_prep = query_prep

        self._execution_counter = 1
        self._workload_rep = 0
        self._samples: list[_ResultSample] = []

    def next_workload_repetition(self) -> None:
        self._workload_rep += 1

    def next_query(self, query: SqlQuery, *, label: str) -> _ResultSample:
        if self._samples:
            last_sample = self._samples[-1]
            self._execution_counter += last_sample.num_executions()

        sample = _ResultSample(
            query=query,
            label=label,
            initial_exec_idx=self._execution_counter,
            current_workload_rep=self._workload_rep,
            query_prep=self.query_prep,
            max_query_reps=self.query_reps,
        )
        self._samples.append(sample)
        return sample

    def scratch_last_sample(self) -> None:
        """Removes the latest sample from the results."""
        if not self._samples:
            raise ValueError("No samples to scratch")
        self._samples.pop()

    def to_df(self) -> pd.DataFrame:
        """Provides all results as a single data frame. This is the final output of the benchmarking process."""
        samples = [sample.to_df() for sample in self._samples]
        return pd.concat(samples, ignore_index=True)


def _exec_ctl_loop(
    query: SqlQuery,
    *,
    sample: _ResultSample,
    cfg: _BenchmarkConfig,
) -> None:
    """
    The execution control loop handles the execution of a single query and records its results as specified by the benchmarking
    config. This includes logging, error handling, callbacks and progressive output.

    Note that per-query repetitions need to be handled by the main workload control loop.
    """
    cfg.log.next_query_rep()

    if cfg.pre_exec_callback is not None:
        data = cfg.pre_exec_callback(query)
        sample.add_pre_exec_data(data)

    db_config = cfg.target_db.describe()
    sample.start_execution()
    match _execute_query(
        query, on=cfg.target_db, timeout=cfg.timeout, query_prep=cfg.query_prep
    ):
        case _SuccessfullExecution(result_set, exec_time):
            sample.add_exec_sample(result_set, exec_time=exec_time, db_config=db_config)
        case _TimeoutExecution(exec_time):
            sample.add_exec_timeout(exec_time, db_config=db_config)
        case _FailedExecution(err) if cfg.error_action == "log":
            sample.add_exec_failure(err, db_config=db_config)
        case _FailedExecution(err) if cfg.error_action == "raise":
            raise err
        case _FailedExecution(_) if cfg.error_action == "ignore":
            pass
        case _FailedExecution(_):
            raise ValueError(f"Unknown error action: {cfg.error_action}")
        case _ as other:
            raise RuntimeError(f"Unhandled execution result: {other}")

    if cfg.post_exec_callback and sample.last_successful():
        data = cfg.post_exec_callback(sample.last_result())  # type: ignore
        sample.add_post_exec_data(data)

    # we must write after the post_exec_callback b/c the callback might add additional data to the
    # result data frame
    if cfg.output and sample.last_successful():
        sample.write_progressive(cfg.output, cfg.output_args)


def _workload_ctl_loop(
    queries: Workload, *, results: _ExecutionResults, cfg: _BenchmarkConfig
) -> None:
    """
    The workload control loop handles the query optimization for each query in the workload and sets up the _exec_ctl_loop for
    the actual execution.
    """
    for label, query in queries.entries():
        cfg.log.next_query(label)
        sample = results.next_query(query, label=label)

        match _optimize_query(query, pipeline=cfg.optimizer):
            case _SuccessfullOptimization(optimized, opt_time):
                query = optimized
                sample.optimization_results(
                    optimized_query=optimized,
                    pipeline=cfg.optimizer,
                    optimization_time=opt_time,
                )
            case _NoOptimization():
                pass
            case _FailedOptimization(err):
                sample.optimization_failure(reason=err, pipeline=cfg.optimizer)

        if sample.failed_optimization():
            match cfg.error_action:
                case "log" if cfg.output:
                    sample.write_progressive(cfg.output, cfg.output_args)
                case "log" if not cfg.output:
                    # we handle the output to CSV as part of the normal result export, no need to do anything here
                    pass
                case "raise":
                    err = sample.failed_optimization()
                    assert err is not None
                    raise err
                case "ignore":
                    results.scratch_last_sample()
            continue

        for _ in range(cfg.per_query_repetitions):
            _exec_ctl_loop(
                query,
                sample=sample,
                cfg=cfg,
            )

        if (pipeline := cfg.optimizer) is None:
            continue
        if not pipeline.requires_online_training():
            continue

        for exec_result in sample.to_execution_results():
            assert exec_result.query_result is not None
            exec_time = 1000 * exec_result.execution_time
            pipeline.learn_from_feedback(
                query, exec_result.query_result, exec_time=exec_time
            )


def execute_workload(
    queries: Iterable[SqlQuery] | Workload,
    on: ExecutionTarget,
    *,
    name: str = "<unnamed>",
    workload_repetitions: int = 1,
    per_query_repetitions: int = 1,
    shuffled: bool = False,
    query_preparation: Optional[QueryPreparation | dict[str, Any]] = None,
    training_data: Optional[TrainingData | TrainingDataRepository] = None,
    timeout: Optional[float] = None,
    exec_callback: Optional[Callable[[ExecutionResult], None]] = None,
    pre_exec_callback: Optional[Callable[[SqlQuery], None | dict]] = None,
    post_exec_callback: Optional[Callable[[ExecutionResult], None | dict]] = None,
    repetition_callback: Optional[Callable[[int], None]] = None,
    progressive_output: Optional[str | Path] = None,
    output_args: Optional[dict] = None,
    logger: Optional[Callable[[str], None] | PredefLogger] = None,
    error_action: ErrorHandling = "log",
) -> pd.DataFrame:
    """Simple benchmarking interface.

    This function runs a query workload on a database system and measures the execution time of each query. All workload
    queries can be optimized through an `OptimizationPipeline`.

    Parameters
    ----------
    queries : Iterable[SqlQuery] | Workload
        The queries to be executed.
    on : ExecutionTarget
        This is a catch-all parameter to specify the database system to execute the queries on, as well as the (optional)
        pipeline to optimize the queries. If a pipeline is provided, all queries are first passed through the pipeline before
        executing them on the pipeline's target database. It is even possible to provide a single optimization stage, in which
        case the stage is first expanded into a full optimization pipeline.
    name : str, optional
        A human-readable name that describes the current experiment. This is only used in log files, etc. to identify the
        the experiment.
    workload_repetitions : int, optional
        The number of times the entire workload should be repeated. By default, the workload is only executed once.
    per_query_repetitions : int, optional
        The number of times each query should be repeated within each workload repetition. The per-query repetitions happen
        sequentially one after another before transitioning to the next query. By default each query is only executed once.
    shuffled : bool, optional
        Whether to randomize the execution order of each query within the workload. Shuffling is applied before each workload
        repetition. Per query repetitions are *not* influenced by this setting.
    query_preparation : Optional[QueryPreparation  |  dict], optional
        Preparation steps that should be performed before running the query. The preparation result will be used in place of
        the original query for all repetitions. If a dictionary is passed, all keys are assumed to be valid parameters to the
        `QueryPreparation` constructor.
    training_data : Optional[TrainingData | TrainingDataRepository], optional
        If any of the stages in the optimization pipeline requires training on data samples, this data has to be passed
        here. This only applies if the stages have not been trained already. Data-driven and workload-driven stages will be
        trained automatically without any explicit action needed by the user.
    timeout : Optional[float], optional
        The maximum time in seconds that the query is allowed to run. If the query exceeds this time, the execution is
        cancelled and the execution time is set to *Inf*. If this parameter is omitted, no timeout is enforced. Notice that
        timeouts require the database to implement `TimeoutSupport`.
    exec_callback : Optional[Callable[[ExecutionResult], None]], optional
        An optional callback that is executed after each query execution.

        .. deprecated:: v0.21.1
            This callback is deprecated in favor of post_exec_callback. Both are functionally equivalent. This is really
            just a renaming to not cause confusion regarding the precise moment when the callback is executed.
    pre_exec_callback : Optional[Callable[[SqlQuery], None | dict]], optional
        An optional callback that is executed right before each query execution. This query includes all hinted optimization
        decisions (if any).

        The callback can either return *None*, or an arbitrary (JSON-serializable) dictionary. If it returns a dictionary, its keys
        are included as additional columns in the resulting data frame. The following rules must be followed:

        1. If the callback returns a dictionary, it must always do so.
        2. The callback must always return a dictionary of the same shape (i.e. same keys). If some keys should sometimes not contain
           values for whatever reason, appropriate placeholder/default values must be used. Selectively omitting a key (or an entire
           dictionary) results in an error.
        3. All values in the dictionary must be JSON-serializable
    post_exec_callback : Optional[Callable[[ExecutionResult], None | dict]], optional
        An optional callback that is executed right after each query execution.

        The callback can either return *None*, or an arbitrary (JSON-serializable) dictionary. If it returns a dictionary, its keys
        are included as additional columns in the resulting data frame. The following rules must be followed:

        1. If the callback returns a dictionary, it must always do so.
        2. The callback must always return a dictionary of the same shape (i.e. same keys). If some keys should sometimes not contain
           values for whatever reason, appropriate placeholder/default values must be used. Selectively omitting a key (or an entire
           dictionary) results in an error.
        3. All values in the dictionary must be JSON-serializable
    progressive_output : Optional[str  |  Path], optional
        If provided, results will be written to this file as soon as they are obtained. If the file already exists, it
        will be appended. Supported file formats are CSV, JSON, Parquet, and HDF. When writing to HDF, the output key must
        be specified in the `output_args`.
    output_args: Optional[dict], optional
        Additional arguments to pass to the Pandas writer function for the progressive output file.
    logger : Optional[Callable[[str], None]  |  PredefLogger], optional
        Configures how progress should be logged. Depending on the specific argument, a number of different strategies are
        available:

        - passing *None* (the default) disables logging
        - passing a callable invokes the function before every query execution. It receives information about the current
          execution as argument
        - referencing a pre-defined logger invokes. Currently, only *tqdm*  is supported. It uses the corresponding library to
          print a progress bar

    Returns
    -------
    pd.DataFrame
        The execution results for the input workload. The data frame will be structured as follows:

        - the data frame will contain one row per query repetition
        - *exec_index* contains an absolute index indicating when the query was executed
        - *timestamp* is the time when the query execution started
        - *label* is an identifier of the current query, usually inferred from the `Workload` object
        - *workload_repetition* indicates the current workload repetition
        - *query_repetition* indicates the current per-query repetition (in contrast to repetitions of the entire workload)
        - *query* contains the input query being executed. If the query was optimized or prepared, these modifications are
          **not** included here
        - *status* indicates whether the query was executed successfully, or whether an error occurred.
          Possible values are "ok", "timeout", "optimization-error", and "execution-error"
        - *result_set*  is the actual result of the query. Scalar results are represented as-is. In case of an error this will
          be *None*
        - *exec_time* contains the time it took to execute the query (in seconds). This includes the entire time from
          sending the query to the database until the last byte of the result set has been transferred back to PostBOUND.
          In case of an error this will be *NaN* and for timeouts this will be timeout itself.
        - *failure_reason* contains a description of the error that occurred during optimization or execution
        - *db_config* describes the database (and its state) on which the query was executed. The state is obtained just before
          query execution started and after the optimization and query preparation steps have been applied
        - *query_preparation* contains the settings that were used to prepare the query after optimization but before execution
        - *optimization_time* contains the time it took to optimize the query using PostBOUND (in seconds). If the query was
          not optimized, this will be *NaN*
        - *optimization_pipeline* contains the optimization pipeline that was used to optimize the query. If the query was not
          optimized, this will be *None*
        - *optimized_query* contains the optimized query that was actually executed on the database. If the query was not
          optimized, this will be *None*
        - additional columns added by the pre-exec or post-exec callbacks


    Other Parameters
    ----------------
    exec_callback : Optional[Callable[[ExecutionResult], None]], optional
        A post-process action that should be executed after each repetition of the query has been completed.
    repetition_callback : Optional[Callable[[int], None]], optional
        An optional post-process action that is executed after each workload repetition. The current repetition number is
        provided as the only argument. Repetitions start at 1.
    error_action : ErrorHandling, optional
        Configures how errors during optimization or execution are handled. By default, failing queries are still contained
        in the result data frame, but some columns might not contain meaningful values. Check the *status* column of the
        data frame to see what happened.

    Notes
    -----
    If the database system does provide accurate timing information through the `StopwatchSupport` interface, these
    measurements will be preferred over the wall-clock timing that is obtained in the benchmarking process.
    """
    queries = _wrap_workload(queries)
    if isinstance(on, OptimizationStage):
        on = _wrap_optimization_stage(on)
    target_db = on if isinstance(on, Database) else on.target_database()
    optimizer = on if isinstance(on, OptimizationPipeline) else None

    if optimizer is not None and optimizer.requires_data_training():
        optimizer.train_on_database(target_db)
    if optimizer is not None and optimizer.requires_workload_training():
        optimizer.train_on_workload(queries, target_db)
    if optimizer is not None and optimizer.requires_sample_training():
        missing_training_data = training_data is None
        fit_completed = optimizer.sample_fit_completed()
        if missing_training_data and not fit_completed:
            raise ValueError(
                "Some optimization stages still need training data. "
                "Make sure to supply applicable samples via the training_data parameter!"
            )
        elif not missing_training_data:
            optimizer.train_on_samples(training_data)
        else:
            # We have not received any training data, but the optimizer is also already trained.
            # Everything is in order.
            pass

    query_preparation = (
        QueryPreparation(**query_preparation)  # type: ignore
        if isinstance(query_preparation, dict)
        else query_preparation
    )
    progressive_output = Path(progressive_output) if progressive_output else None

    if exec_callback is not None:
        warnings.warn(
            "exec_callback is deprecated. "
            "Use the functionally equivalent post_exec_callback instead.",
            category=DeprecationWarning,
        )
        if post_exec_callback is not None:
            raise ValueError(
                "Cannot specify both exec_callback and post_exec_callback. "
                "Use post_exec_callback and remove exec_callback from your code. "
                "Both are functionally equivalent."
            )
        post_exec_callback = exec_callback

    log: _LoggerImpl
    if logger == "tqdm":
        log = _TqdmLogger(
            workload_reps=workload_repetitions,
            query_reps=per_query_repetitions,
            total_queries=len(queries),
        )
    elif logger is not None:
        log = _CustomLogger(logger, workload_reps=workload_repetitions)
    else:
        log = _NoOpLogger()

    cfg = _BenchmarkConfig(
        workload=queries,
        target_db=target_db,
        optimizer=optimizer,
        output=progressive_output,
        output_args=output_args or {},
        workload_repetitions=workload_repetitions,
        per_query_repetitions=per_query_repetitions,
        shuffled=shuffled,
        timeout=timeout,
        query_prep=query_preparation,
        pre_exec_callback=pre_exec_callback,
        post_exec_callback=post_exec_callback,
        log=log,
        error_action=error_action,
    )
    results = _ExecutionResults(
        query_reps=per_query_repetitions, query_prep=query_preparation
    )

    _init_benchmark_log(name, cfg=cfg)

    # The overall control flow looks rougly like this:
    #   ++ workload repetitions [handled here]
    #        ++ workload control loop [handled by _workload_ctl_loop]
    #           query optimization
    #           per-query repetitions
    #             ++ high-level query execution and result generation [handled by _exec_ctl_loop]
    #                  ++ low-level query execution [handled by _execute_query]

    try:
        for i in range(workload_repetitions):
            log.next_workload_iter()
            results.next_workload_repetition()
            if shuffled:
                queries = queries.shuffle()

            _workload_ctl_loop(
                queries,
                results=results,
                cfg=cfg,
            )

            if repetition_callback:
                repetition_callback(i + 1)
    except Exception as e:
        _failed_benchmark_log(name, cfg=cfg, error=e)
        raise e

    log.next_workload_iter()  # to finalize progress bars
    _finalize_benchmark_log(name, cfg=cfg)
    return results.to_df()


def prepare_export(results_df: pd.DataFrame) -> pd.DataFrame:
    """Modifies a benchmark result dataframe such that it can be written to CSV files without problems.

    This mostly involves converting Python objects to JSON counterparts that allow a reconstruction of equivalent data.

    More specifically, the function handles two main aspects:

    1. making sure that the query result can be written to CSV, and
    2. making sure that the description of the optimization pipeline can be written to CSV.

    In both cases, the column values will be transformed to JSON-objects if necessary.

    Parameters
    ----------
    results_df : pd.DataFrame
        The result dataframe created by one of the benchmark functions

    Returns
    -------
    pd.DataFrame
        The prepared dataframe

    See Also
    --------
    postbound.experiments.runner : Functions to obtain benchmark results
    """
    if not len(results_df):
        return results_df

    prepared_df = results_df.copy()

    example_result = prepared_df.iloc[0]
    for col in example_result.index:
        if not isinstance(example_result[col], (list, tuple, dict, Jsonizable)):
            continue

        prepared_df[col] = prepared_df[col].apply(util.to_json)

    return prepared_df


def sort_results(
    results_df: pd.DataFrame,
    by_column: str | tuple[str] = ("label", "exec_index"),  # type: ignore
) -> pd.DataFrame:
    """Provides a better sorting of the benchmark results in a data frame.

    By default, the entries in the result data frame will be sorted either sequentially, or by a lexicographic ordering on the
    label column. This function uses a natural ordering over the column labels.

    In contrast to lexicographic sorting, natural sorting handles numeric labels in a better way: labels like
    1a, 10a and 100a are sorted in this order instead of in reverse.

    Parameters
    ----------
    results_df : pd.DataFrame
        Data frame containing the results to sort
    by_column : str | tuple[str], optional
        The columns by which to order, by default `(COL_LABEL, COL_EXEC_IDX)`. A lexicographic ordering will
        be applied to all of them.

    Returns
    -------
    pd.DataFrame
        A reordered data frame. The original data frame is not modified
    """
    return results_df.sort_values(
        by=by_column, key=lambda series: np.argsort(natsort.index_natsorted(series))
    )
