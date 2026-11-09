"""Start the backend for the browser E2E run on a fresh, throwaway data folder.

Playwright runs ``webServer.command`` through the platform shell: ``cmd.exe`` on Windows, where the old
POSIX one-liner (``rm -rf``, ``mkdir -p``, ``VAR=value cmd``) failed with "'HOME' is not recognized". The
environment now comes from the config's ``env`` option, and this script does the folder reset in Python so
the same command works on Windows, macOS and Linux.
"""

from __future__ import annotations

import os
import runpy
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    home = os.environ.get("CHAOSENGINE_E2E_HOME")
    if not home:
        raise SystemExit("CHAOSENGINE_E2E_HOME is not set; run this through the Playwright config.")
    home_path = Path(home).resolve()
    # Only ever wipe the run's own folder inside e2e/.run, never a real home.
    if home_path.parent.name != ".run" or home_path.parent.parent != Path(__file__).resolve().parent:
        raise SystemExit(f"Refusing to reset {home_path}: not the E2E run folder.")
    shutil.rmtree(home_path, ignore_errors=True)
    home_path.mkdir(parents=True)
    sys.path.insert(0, str(REPO_ROOT))
    sys.argv = ["backend_service.app"]
    runpy.run_module("backend_service.app", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
