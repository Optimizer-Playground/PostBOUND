"""Tests for the Postgres timeout-query state machine, driven entirely offline.

`_TimeoutQueryExecutor.execute_query` itself is not unit-testable: it spawns a real
`multiprocessing.Process` and opens two extra psycopg connections (a watchdog and, inside the worker, a
fresh `PostgresDatabase`). But the *decision logic* -- what to do for each event the worker reports -- lives
in a handful of small functions that only touch two things: a `status_pipe` (``.recv()`` / ``.poll()``) and
an ``executor`` (``._abort_backend()``). Faking both turns every transition, including the error paths a live
timeout test almost never reaches, into a sub-millisecond assertion.

Given the project's history of bugs in the timeout machinery, this is the single highest-value place to add
coverage: it is cheap enough to run on every commit and exercises exactly the control flow that has broken
before.
"""

from __future__ import annotations

import pytest

from postbound.postgres._pg import (
    _BackendConnectedEvent,
    _QueryFinishedEvent,
    _QueryReadyEvent,
    _ResultEvent,
    _timeout_worker_abort,
    _timeout_worker_await_result,
    _timeout_worker_ctl,
    _timeout_worker_prep,
    _timeout_worker_run_query,
    _timeout_worker_timeout,
    _WorkerErrorEvent,
)
from postbound.util import StateError


class ScriptedPipe:
    """A fake `multiprocessing.connection.Connection` driving the state machine from a fixed script.

    Parameters
    ----------
    events : list
        Values returned by successive `recv()` calls, in order.
    poll_result : bool, optional
        What `poll()` reports. *True* (the default) means "an event is ready"; *False* simulates the query
        still running when the timeout elapses.
    """

    def __init__(self, events: list, *, poll_result: bool = True) -> None:
        self._events = iter(events)
        self.poll_result = poll_result
        self.poll_calls: list[float] = []

    def recv(self):
        try:
            return next(self._events)
        except StopIteration:
            raise AssertionError("ScriptedPipe.recv() called more times than the script provides events for") from None

    def poll(self, timeout: float | None = None) -> bool:
        self.poll_calls.append(timeout)
        return self.poll_result


class RecordingExecutor:
    """A fake `_TimeoutQueryExecutor` that only records `_abort_backend` calls."""

    def __init__(self) -> None:
        self.aborted_pids: list[int] = []

    def _abort_backend(self, pid: int) -> None:
        self.aborted_pids.append(pid)


BACKEND_PID = 4242


# -- _ResultEvent factories ----------------------------------------------------------------------------


def test_result_event_ok() -> None:
    event = _ResultEvent.ok([(1,)], 0.5)

    assert event.status == "success"
    assert event.result_set == [(1,)]
    assert event.exec_time == 0.5
    assert event.error is None


def test_result_event_timeout() -> None:
    event = _ResultEvent.timeout(2.0)

    assert event.status == "timeout"
    assert event.result_set is None
    assert event.exec_time == 2.0


def test_result_event_failed() -> None:
    error = ValueError("boom")

    event = _ResultEvent.failed(error)

    assert event.status == "failure"
    assert event.result_set is None
    assert event.error is error
    import math

    assert math.isnan(event.exec_time)


# -- _timeout_worker_ctl (entry point) ------------------------------------------------------------------


def test_ctl_transitions_to_prep_on_backend_connected() -> None:
    """A connected backend, followed immediately by the query becoming ready, reaches "run"."""
    pipe = ScriptedPipe(
        [_BackendConnectedEvent(BACKEND_PID), _QueryReadyEvent(), _QueryFinishedEvent(), _ResultEvent.ok([], 0.1)]
    )
    executor = RecordingExecutor()

    result = _timeout_worker_ctl(pipe, timeout=5.0, executor=executor)

    assert result.status == "success"
    assert executor.aborted_pids == []


def test_ctl_returns_failure_on_worker_error() -> None:
    error = RuntimeError("connection refused")
    pipe = ScriptedPipe([_WorkerErrorEvent(error)])
    executor = RecordingExecutor()

    result = _timeout_worker_ctl(pipe, timeout=5.0, executor=executor)

    assert result.status == "failure"
    assert result.error is error
    # No backend PID was ever reported, so there is nothing to cancel.
    assert executor.aborted_pids == []


def test_ctl_raises_on_unexpected_event() -> None:
    pipe = ScriptedPipe([_QueryReadyEvent()])  # not a valid first event

    with pytest.raises(StateError, match="Unexpected event"):
        _timeout_worker_ctl(pipe, timeout=5.0, executor=RecordingExecutor())


# -- _timeout_worker_prep --------------------------------------------------------------------------------


def test_prep_transitions_to_run_query_on_query_ready() -> None:
    pipe = ScriptedPipe([_QueryReadyEvent(), _QueryFinishedEvent(), _ResultEvent.ok([(1,)], 0.2)])

    result = _timeout_worker_prep(pipe, timeout=5.0, backend_pid=BACKEND_PID, executor=RecordingExecutor())

    assert result.status == "success"
    assert result.result_set == [(1,)]


def test_prep_aborts_backend_on_worker_error() -> None:
    """An error during query preparation must still cancel the already-connected backend."""
    error = RuntimeError("failed to apply preparatory statement")
    pipe = ScriptedPipe([_WorkerErrorEvent(error)])
    executor = RecordingExecutor()

    result = _timeout_worker_prep(pipe, timeout=5.0, backend_pid=BACKEND_PID, executor=executor)

    assert result.status == "failure"
    assert result.error is error
    assert executor.aborted_pids == [BACKEND_PID]


def test_prep_raises_on_unexpected_event() -> None:
    pipe = ScriptedPipe([_ResultEvent.ok([], 0.0)])  # not valid during preparation

    with pytest.raises(StateError, match="Unexpected event"):
        _timeout_worker_prep(pipe, timeout=5.0, backend_pid=BACKEND_PID, executor=RecordingExecutor())


# -- _timeout_worker_run_query ---------------------------------------------------------------------------


def test_run_query_transitions_to_await_result_when_finished_in_time() -> None:
    pipe = ScriptedPipe([_QueryFinishedEvent(), _ResultEvent.ok([(7,)], 1.5)], poll_result=True)

    result = _timeout_worker_run_query(pipe, timeout=5.0, backend_pid=BACKEND_PID, executor=RecordingExecutor())

    assert result.status == "success"
    assert result.result_set == [(7,)]
    # The timeout given to the caller must be the one passed to poll().
    assert pipe.poll_calls == [5.0]


def test_run_query_times_out_and_aborts_backend() -> None:
    """The critical timeout path: poll() reporting nothing ready must cancel the backend."""
    pipe = ScriptedPipe([], poll_result=False)
    executor = RecordingExecutor()

    result = _timeout_worker_run_query(pipe, timeout=0.5, backend_pid=BACKEND_PID, executor=executor)

    assert result.status == "timeout"
    assert result.exec_time == 0.5
    assert result.result_set is None
    assert executor.aborted_pids == [BACKEND_PID]


def test_run_query_aborts_backend_on_worker_error() -> None:
    error = RuntimeError("backend crashed mid-query")
    pipe = ScriptedPipe([_WorkerErrorEvent(error)], poll_result=True)
    executor = RecordingExecutor()

    result = _timeout_worker_run_query(pipe, timeout=5.0, backend_pid=BACKEND_PID, executor=executor)

    assert result.status == "failure"
    assert result.error is error
    assert executor.aborted_pids == [BACKEND_PID]


def test_run_query_raises_on_unexpected_event() -> None:
    pipe = ScriptedPipe([_BackendConnectedEvent(BACKEND_PID)], poll_result=True)  # not valid here

    with pytest.raises(StateError, match="Unexpected event"):
        _timeout_worker_run_query(pipe, timeout=5.0, backend_pid=BACKEND_PID, executor=RecordingExecutor())


# -- _timeout_worker_await_result ------------------------------------------------------------------------


def test_await_result_returns_the_result_event_directly() -> None:
    final = _ResultEvent.ok([(1,), (2,)], 3.0)
    pipe = ScriptedPipe([final])

    result = _timeout_worker_await_result(pipe, backend_pid=BACKEND_PID, executor=RecordingExecutor())

    assert result is final


def test_await_result_aborts_backend_on_worker_error() -> None:
    error = RuntimeError("failed while fetching results")
    pipe = ScriptedPipe([_WorkerErrorEvent(error)])
    executor = RecordingExecutor()

    result = _timeout_worker_await_result(pipe, backend_pid=BACKEND_PID, executor=executor)

    assert result.status == "failure"
    assert executor.aborted_pids == [BACKEND_PID]


def test_await_result_raises_on_unexpected_event() -> None:
    pipe = ScriptedPipe([_QueryFinishedEvent()])  # not valid here -- already past this state

    with pytest.raises(StateError, match="Unexpected event"):
        _timeout_worker_await_result(pipe, backend_pid=BACKEND_PID, executor=RecordingExecutor())


# -- direct unit tests for the leaf transitions -----------------------------------------------------------


def test_timeout_worker_timeout_aborts_and_returns_timeout_event() -> None:
    executor = RecordingExecutor()

    result = _timeout_worker_timeout(3.0, backend_pid=BACKEND_PID, executor=executor)

    assert result.status == "timeout"
    assert result.exec_time == 3.0
    assert executor.aborted_pids == [BACKEND_PID]


def test_timeout_worker_abort_aborts_and_returns_failure_event() -> None:
    error = ValueError("some worker failure")
    executor = RecordingExecutor()

    result = _timeout_worker_abort(error, BACKEND_PID, executor=executor)

    assert result.status == "failure"
    assert result.error is error
    assert executor.aborted_pids == [BACKEND_PID]


# -- full protocol walk-throughs --------------------------------------------------------------------------


def test_full_success_protocol() -> None:
    """Walks the entire documented 7-step protocol for a query that finishes normally."""
    pipe = ScriptedPipe(
        [
            _BackendConnectedEvent(BACKEND_PID),
            _QueryReadyEvent(),
            _QueryFinishedEvent(),
            _ResultEvent.ok([(42,)], 0.8),
        ]
    )
    executor = RecordingExecutor()

    result = _timeout_worker_ctl(pipe, timeout=10.0, executor=executor)

    assert result.status == "success"
    assert result.result_set == [(42,)]
    assert result.exec_time == 0.8
    assert executor.aborted_pids == []


def test_full_timeout_protocol() -> None:
    """Walks the protocol for a query that is still running when the timeout elapses."""
    pipe = ScriptedPipe(
        [_BackendConnectedEvent(BACKEND_PID), _QueryReadyEvent()],
        poll_result=False,
    )
    executor = RecordingExecutor()

    result = _timeout_worker_ctl(pipe, timeout=0.1, executor=executor)

    assert result.status == "timeout"
    assert result.exec_time == 0.1
    assert executor.aborted_pids == [BACKEND_PID]


def test_full_connection_failure_protocol() -> None:
    """A failure right at connection establishment must never attempt to cancel a backend."""
    error = RuntimeError("could not connect to server")
    pipe = ScriptedPipe([_WorkerErrorEvent(error)])
    executor = RecordingExecutor()

    result = _timeout_worker_ctl(pipe, timeout=10.0, executor=executor)

    assert result.status == "failure"
    assert result.error is error
    assert executor.aborted_pids == []
