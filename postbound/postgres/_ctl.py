from __future__ import annotations

import os
import subprocess
from pathlib import Path


def start(pgdata: str | Path = "", *, logfile: str | Path = "") -> None:
    """Starts a local Postgres server.

    This function assumes that *pg_ctl* is available on the system PATH and either the server's data directory is specified
    explicitly, or set via the *PGDATA* environment variable.
    """
    if subprocess.call("which pg_ctl") != 0:
        raise ValueError("Cannot start Postgres server: pg_ctl is not on PATH")

    pgdata = pgdata or os.environ.get("PGDATA", "")
    pgdata = Path(pgdata).expanduser()
    if not pgdata:
        raise ValueError(
            "Cannot start Postgres server: Must either supply pgdata argument or set PGDATA environment variable"
        )

    args = ["pg_ctl", "-D", pgdata]
    if logfile:
        args.extend(["-l", logfile])
    args.append("start")

    subprocess.run(args, check=True)


def stop(pgdata: str | Path = "", *, raise_on_error: bool = False) -> None:
    """Stops a running (local) Postgres server.

    This function assumes that *pg_ctl* is available on the system PATH and either the server's data directory is specified
    explicitly, or set via the *PGDATA* environment variable.

    If the server cannot be stopped due to whatever reason, an error can be raised by setting the corresponding parameter.
    Otherwise, it is silently ignored.
    """
    if subprocess.call("which pg_ctl") != 0:
        raise ValueError("Cannot stop Postgres server: pg_ctl is not on PATH")

    pgdata = pgdata or os.environ.get("PGDATA", "")
    pgdata = Path(pgdata).expanduser()
    if not pgdata:
        raise ValueError(
            "Cannot stop Postgres server: Must either supply pgdata argument or set PGDATA environment variable"
        )

    subprocess.run(["pg_ctl", "-D", pgdata, "stop"], check=raise_on_error)


def is_running(pgdata: str | Path = "") -> bool:
    """Checks, whether a local Postgres server is currently running.

    This function assumes that *pg_ctl* is available on the system PATH. A data directory can be supplied to check whether
    a server is running for the specific database. If *pgdata* is not supplied, the *PGDATA* environment variable is used as
    a fallback.
    """
    if subprocess.call("which pg_ctl") != 0:
        raise ValueError("Cannot start Postgres server: pg_ctl is not on PATH")

    cmd = ["pg_ctl"]
    pgdata = pgdata or os.environ.get("PGDATA", "")
    if pgdata:
        cmd.extend(["-D", str(pgdata)])
    cmd.append("status")

    res = subprocess.run(cmd)
    return res.returncode == 0
