"""Runs the test tier appropriate for the git hook that invoked it.

This is the entry point of the ``tests`` hook in ``.pre-commit-config.yaml``. Changes that land on the protected
branch are "dangerous" and get the embedded tier on top of the offline one; everything else only gets tier 0:

==================  ====================================  =================
Stage               Condition                             Tier
==================  ====================================  =================
pre-commit          HEAD is the protected branch          1
pre-merge-commit    HEAD is the protected branch          1
pre-push            the push updates the protected branch 1, otherwise 0
==================  ====================================  =================

Commits on other branches run no tests at all, matching the previous behaviour.

Usage (pre-commit passes the stage via ``PRE_COMMIT_HOOK_STAGE``)::

    uv run python -m tools.hook-tests
"""

from __future__ import annotations

import os
import subprocess
import sys

PROTECTED_BRANCH = "main"


def _git(*args: str) -> str | None:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _on_protected_branch() -> bool:
    return _git("symbolic-ref", "--short", "-q", "HEAD") == PROTECTED_BRANCH


def _push_updates_protected_branch() -> bool:
    if os.environ.get("PRE_COMMIT_REMOTE_BRANCH") == f"refs/heads/{PROTECTED_BRANCH}":
        return True

    # pre-commit only exposes the *first* ref of a multi-ref push (e.g. `git push origin feature main`), so we also
    # treat the push as dangerous whenever the local protected branch is ahead of its remote counterpart. This may
    # run tier 1 for a push that does not actually include the protected branch, which is cheap and errs on the safe
    # side.
    remote = os.environ.get("PRE_COMMIT_REMOTE_NAME")
    if not remote:
        return False
    remote_ref = f"refs/remotes/{remote}/{PROTECTED_BRANCH}"
    local_ref = f"refs/heads/{PROTECTED_BRANCH}"
    if (
        _git("rev-parse", "--verify", "-q", remote_ref) is None
        or _git("rev-parse", "--verify", "-q", local_ref) is None
    ):
        return False
    ahead = _git("rev-list", "--count", f"{remote_ref}..{local_ref}")
    return ahead is not None and int(ahead) > 0


def _select_tier(stage: str) -> str | None:
    match stage:
        case "pre-commit" | "pre-merge-commit":
            return "1" if _on_protected_branch() else None
        case "pre-push":
            return "1" if _push_updates_protected_branch() else "0"
        case _:
            return "0"


def main() -> int:
    stage = os.environ.get("PRE_COMMIT_HOOK_STAGE", "")
    tier = _select_tier(stage)
    if tier is None:
        return 0

    print(f"Running test tier {tier} ({stage or 'manual'})", flush=True)
    return subprocess.run([sys.executable, "-m", "pytest", "--tier", tier, "-q"]).returncode


if __name__ == "__main__":
    sys.exit(main())
