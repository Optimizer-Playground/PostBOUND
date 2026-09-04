"""Provides utilities to access (operating system) related information."""

from __future__ import annotations

import os
import sys
import warnings

from . import proc


def open_files(pid: int | None = None) -> list[str]:
    """Provides all files (e.g. text files and shared objects) opened by the given process/PID.

    Parameters
    ----------
    pid : Optional[int], optional
        The PID of the process to query. Defaults to the current process.

    Returns
    -------
    list[str]
        All opened files
    """
    if os.name != "posix":
        warnings.warn("Can only check for open files on POSIX systems.", stacklevel=2)
        return []

    pid = os.getpid() if pid is None else pid
    ext = ".dylib" if sys.platform == "darwin" else ".so"

    # lsof -p produces some "weird" (or rather impractical) output from time to time (and depending on the lsof version)
    # we do the following:
    # lsof -Fn -p gives the names of all opened files for a specific PID
    # But: it prefixes those names with a "n" to distinguish from other files (e.g. sockets)
    # Hence, we grep for ^n to only get real files
    # Afterwards, we remove the n prefix with cut
    # Still, some files are weird because lsof adds a suffix like (path dev=...) to the output. As of right now, I don't know
    # how to interpret this output nor how to get rid of it. The second cut removes this suffix.
    # Lastly, the final grep filters for shared objects. Notice that we don't grep for '.so$' in order to keep files like
    # loibc.so.6
    res = proc.run_cmd(
        f"lsof -Fn -p {pid} | grep '^n' | cut -c2- | cut -d' ' -f1 | grep '{ext}'",
        shell=True,
    )

    return res.splitlines()
