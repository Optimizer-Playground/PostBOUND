"""Contains utilities to conveniently log different information."""

from __future__ import annotations

import atexit
import functools
import inspect
import pprint
import sys
from collections.abc import Callable
from datetime import datetime
from typing import IO

Logger = Callable[..., None]
"""Type alias for our loggers.

Each logger accepts an arbitrary amount of arguments and is inteded to function as a replacement for the `print` function.
"""


def timestamp() -> str:
    """Provides the current time as a nice and normalized string."""
    return datetime.now().strftime("%y-%m-%d %H:%M:%S")


def make_logger(
    enabled: bool = True,
    *,
    file: IO[str] = sys.stderr,
    pretty: bool = False,
    prefix: str | Callable[[], str] = "",
) -> Logger:
    """Creates a new logging utility.

    The generated method can be used like a regular `print`, but with defaults that are better suited for logging purposes.

    If `enabled` is `False`, calling the logging function will not actually print anything and simply return. This
    is especially useful to implement logging-hooks in longer functions without permanently re-checking whether logging
    is enabled or not.

    By default, all logging output will be written to stderr, but this can be customized by supplying a different
    `file`.

    If `pretty` is enabled, structured objects such as dictionaries will be pretty-printed instead of being written
    on a single line. Note that pprint is used for all of the logging data everytime in that case.

    Parameters
    ----------
    enabled : bool, optional
        Whether logging is enabled, by default *True*
    file : IO[str], optional
        Destination to write the log entries to, by default ``sys.stderr``
    pretty : bool, optional
        Whether complex objects should be pretty-printed using the ``pprint`` module, by default *False*
    prefix : str | Callable[[], str], optional
        A common prefix that should be added before each log entry. Can be either a hard-coded string, or a callable that
        dynamically produces a string for each logging action separately (e.g. timestamp).

    Returns
    -------
    Callable
        _description_
    """

    def _log(*args, **kwargs) -> None:
        if prefix and isinstance(prefix, str):
            args = [prefix, *list(args)]
        elif prefix and not isinstance(prefix, str):
            args = [prefix(), *list(args)]
        print(*args, file=file, **kwargs)

    def _dummy_log(*args, **kwargs) -> None:
        pass

    if pretty and enabled:
        return functools.partial(pprint.pprint, stream=file)

    return _log if enabled else _dummy_log


def standard_logger(enabled: bool = True) -> Logger:
    """A standardized logger for consistent output throughout the codebase."""

    def _standard_prefix() -> str:
        ts = timestamp()
        callstack = inspect.stack()[2:]
        callee = "<main>"
        while callstack:
            current_frame = callstack.pop(0)
            module = inspect.getmodule(current_frame.frame)
            if not module:
                continue

            mod_path = module.__name__.split(".")
            mod_name = next((mod for mod in mod_path[::-1] if not mod.startswith("_")), None)
            if mod_name is None:
                continue

            callee = mod_name
            break

        return f"{ts} ({callee}) -"

    return make_logger(enabled, file=sys.stderr, prefix=_standard_prefix)


def print_stderr(*args, **kwargs) -> None:
    """A normal `print` that writes to stderr instead of stdout."""
    kwargs.pop("file", None)
    print(*args, file=sys.stderr, **kwargs)


def print_if(should_print: bool, *args, use_stderr: bool = False, **kwargs) -> None:
    """A normal `print` that only prints something if `should_print` evaluates true-ish. Can optionally print to stderr."""
    if should_print:
        out_device = kwargs.pop("file", sys.stderr if use_stderr else sys.stdout)
        print(*args, file=out_device, **kwargs)


class _TeeLogger:
    def __init__(self, target_file: str, output_mode: str = "a") -> None:
        self._original_stdout = sys.stdout
        self._log_out = open(target_file, output_mode)  # noqa: SIM115 -- closed by the atexit hook below
        atexit.register(lambda: self._log_out.close())

    def write(self, message: str) -> None:
        self._original_stdout.write(message)
        self._log_out.write(message)


def tee_stdout(target_file: str, output_mode: str = "a") -> None:
    sys.stdout = _TeeLogger(target_file, output_mode)
