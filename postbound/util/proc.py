"""Provides utilities to interact with outside processes."""

from __future__ import annotations

import os
import pathlib
import subprocess
from collections.abc import Iterable
from typing import Any


class ProcResult(str):
    """Wrapper for the result of an external process.

    In contrast to `CompletedProcess` provided by the `subprocess` module, this class is designed for more convenient usage.
    More specifically, it can be used directly as a substitute for the *stdout* of the process (hence the subclassing of
    `str`). Furthermore, bool checks ensure that the process exited with a zero exit code.

    All output is provided in dedicated attributes.

    Parameters
    ----------
    out_data : str
        The stdout of the process.
    err_data : str
        The stderr of the process.
    exit_code : int
        The exit code of the process.
    """

    def __init__(self, out_data: str, err_data: str, exit_code: int) -> None:
        self.out_data = out_data
        self.err_data = err_data
        self.exit_code = exit_code

    def __new__(cls, out_data: str, err_data: str, exit_code: int):
        return str.__new__(cls, out_data)

    def echo(self) -> None:
        """Provides the contents of stdout and stderr in a format for debugging by humans."""
        print("stdout:")
        print(self.out_data)
        print("stderr:")
        print(self.err_data)

    def raise_if_error(self) -> None:
        """Raises an exception if the process exited with a non-zero exit code."""
        if self.exit_code != 0:
            raise RuntimeError(f"Process exited with code {self.exit_code}: '{self.err_data}'")

    def __bool__(self) -> bool:
        return self.exit_code == 0

    def __repr__(self) -> str:
        return f"ProcResult(exit_code={self.exit_code}, stdout={self.out_data!r}, stderr={self.err_data!r})"

    def __str__(self) -> str:
        return self.out_data


def run_cmd(
    cmd: str | Iterable[Any],
    *args,
    work_dir: str | pathlib.Path | None = None,
    **kwargs,
) -> ProcResult:
    """Executes an arbitrary external command.

    The command can be executed in an different working directory. After execution the working directory is restored.

    This function delegates to `subprocess.run`. Therefore, most arguments accepted by this function follow the same rules
    as the `run` function.

    Parameters
    ----------
    cmd : str | Iterable[Any]
        The program to execute. Can be either a single invocation, or a list of the program name and its arguments.
    work_dir : Optional[str  |  pathlib.Path], optional
        The working directory where the process should be executed. If `None`, the current working directory is used.
        Otherwise, the current working directory is changed to the desired directory for the duration of the process execution
        and restored afterwards.
    *args
        Additional arguments to be passed to the command.
    **kwargs
        Additional arguments to customize the subprocess invocation.

    Returns
    -------
    ProcResult
        The result of the process execution. If the command can be executed but fails, the `exit_code` will be non-zero. On the
        other hand, if the command cannot be executed at all (e.g. because it is not found or the user does not have the
        required permissions), an error is raised.
    """
    work_dir = os.getcwd() if work_dir is None else str(work_dir)
    current_dir = os.getcwd()

    if isinstance(cmd, Iterable) and not isinstance(cmd, str):
        cmd, args = str(cmd[0]), cmd[1:] + list(args)
    invocation = [cmd] + [str(arg) for arg in args]

    os.chdir(work_dir)
    res = subprocess.run(invocation, capture_output=True, text=True, **kwargs)
    os.chdir(current_dir)

    return ProcResult(res.stdout, res.stderr, res.returncode)
