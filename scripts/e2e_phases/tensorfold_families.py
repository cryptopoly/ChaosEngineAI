"""Phase 1 helper: the TensorFold family table comes from the installed binary (FU-168).

Called from the ``TensorFold 1.2.7 families`` check in ``scripts/e2e_test_suite.py``.
Reads ``families`` from ``GET /api/setup/tensorfold-native-status`` and runs the
same binary's ``capabilities --json`` itself: the report must list exactly the
binary's families as served here, and none of them may lack a curated row.
Skips when no 1.x binary is installed. Needs the backend on this machine.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Callable


def family_report_check(cli_json: Callable[..., tuple[int, Any, Any]]) -> tuple[str, str, dict[str, Any]]:
    rc, state, _ = cli_json("call", "GET", "/api/setup/tensorfold-native-status", timeout=60.0)
    if rc != 0 or not isinstance(state, dict):
        return "fail", "tensorfold-native-status failed", {}
    binary = state.get("binary")
    if not state.get("installed") or not binary:
        return "skip", "TensorFold 1.x binary not installed", {}
    try:
        probe = subprocess.run([binary, "capabilities", "--json"], capture_output=True, text=True, timeout=60, check=False)
        families = set((json.loads(probe.stdout).get("families") or {}).keys())
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError) as exc:
        return "fail", f"{binary} capabilities --json failed: {exc}", {}
    report = state.get("families")
    if not isinstance(report, list):
        return "fail", "status route has no 'families' report", {}
    served = {entry.get("modelType") for entry in report if entry.get("servedHere")}
    if served != families:
        return "fail", (
            f"report differs from the binary: missing {sorted(families - served)}, extra {sorted(served - families)}"
        ), {}
    uncurated = sorted(entry.get("modelType") for entry in report if entry.get("servedHere") and not entry.get("curated"))
    if uncurated:
        return "fail", f"families with no curated row: {', '.join(uncurated)}", {}
    return "pass", "", {"families": len(families), "offered": sum(1 for e in report if e.get("offered"))}
