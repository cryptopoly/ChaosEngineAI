"""Phase 11 check (FU-139): off Apple Silicon the music runtime reports why, and install is refused.

Called from ``phase_11`` in ``scripts/e2e_test_suite.py`` as ``check(call, runtime)`` where ``call(method, path,
payload)`` returns ``(status, json)`` against ``/api/music``. On Windows and Linux (and Intel Macs) it asserts the
single reason sentence, its code, no ``cuda`` in any catalog row's platforms, and that ``POST /install`` starts
nothing (the job answers ``error`` with the same reason). On an Apple Silicon Mac it asserts the supported path:
no reason, no code. It never starts an install on a supported Mac.
"""

from __future__ import annotations

import platform
from typing import Any, Callable

REASON_START = "Music generation runs on Apple Silicon Macs in this version."
CODE = "music.unsupportedPlatform"


def check(call: Callable[..., tuple[int, Any]], runtime: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    apple = platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")
    if apple:
        if runtime.get("unavailableReason") or runtime.get("unavailableCode"):
            return "fail", f"Apple Silicon reports unavailable: {runtime.get('unavailableReason')}", {}
        return "pass", "", {"lane": runtime.get("lane"), "path": "apple-silicon"}

    reason = runtime.get("unavailableReason") or ""
    if not reason.startswith(REASON_START) or runtime.get("unavailableCode") != CODE:
        return "fail", f"runtime reason/code wrong: {reason!r} / {runtime.get('unavailableCode')!r}", {}
    sound = runtime.get("sound") or {}
    if sound.get("unavailableReason") != reason:
        return "fail", f"the sounds lane says something else: {sound.get('unavailableReason')!r}", {}
    code, body = call("GET", "/catalog")
    rows = [v for f in (body or {}).get("families", []) for v in f.get("variants", [])]
    if code != 200 or not rows or any("cuda" in (v.get("platforms") or []) for v in rows):
        return "fail", f"catalog HTTP {code}; a row still lists cuda", {}
    code, job = call("POST", "/install")
    if code != 200 or job.get("phase") != "error" or job.get("running") or job.get("error") != reason:
        return "fail", f"install was not refused: HTTP {code} {job}", {}
    return "pass", "", {"path": "unsupported", "rows": len(rows)}
