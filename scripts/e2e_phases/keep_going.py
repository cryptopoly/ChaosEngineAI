"""Keep going (FU-191) run API against a live backend: start is idempotent and armed until the brief, Stop ends
the run with a reason and drops the grant, Resume refuses a run that never had a brief, the cap and grant hours
are bounded, and deleting the chat drops the run. No model is needed and nothing renders.

Called from phase 7 of ``scripts/e2e_test_suite.py`` as ``check(suite)``; returns ``(status, reason, detail)``.
"""

from __future__ import annotations

import json
from typing import Any


def check(suite: Any):
    def api(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        code, raw = suite._voice_http(method, path, body=body, content_type="application/json" if body else None, timeout=15.0)
        try:
            return code, json.loads(raw.decode("utf-8")) if raw.strip() else None
        except ValueError:
            return code, None

    code, created = api("POST", "/api/chat/sessions", {"title": "e2e keep going"})
    sid = ((created or {}).get("session") or {}).get("id") if isinstance(created, dict) else None
    if code != 200 or not sid:
        return "fail", f"could not make a chat: HTTP {code}", {}
    base = f"/api/chat/keepgoing/{sid}"
    deleted = False
    try:
        code, data = api("GET", base)
        if code == 404:
            return "skip", "Keep going is not in this build", {}
        if code != 200 or (data or {}).get("run") is not None:
            return "fail", f"a new chat should have no run: HTTP {code} {data}", {}
        code, data = api("POST", f"{base}/start", {"cap": 5, "grant": "auto", "hours": 2})
        run = (data or {}).get("run") or {}
        if code != 200 or run.get("state") != "armed" or run.get("cap") != 5 or (run.get("grant") or {}).get("mode") != "auto":
            return "fail", f"start: HTTP {code} {run}", {}
        code, again = api("POST", f"{base}/start", {"cap": 50, "grant": "manual"})
        if ((again or {}).get("run") or {}).get("runId") != run.get("runId"):
            return "fail", "a second start made a second run", {}
        for bad in ({"cap": 101}, {"hours": 13}, {"grant": "always"}):
            code, _ = api("POST", f"{base}/start", bad)
            if code != 422:
                return "fail", f"start accepted {bad} (HTTP {code})", {}
        code, data = api("POST", f"{base}/stop")
        stopped = (data or {}).get("run") or {}
        if stopped.get("state") != "stopped" or stopped.get("stopCode") != "user_stopped" or stopped.get("grant") is not None:
            return "fail", f"stop: {stopped}", {}
        code, _ = api("POST", f"{base}/resume", {"grant": "manual"})
        if code != 409:
            return "fail", f"resume of a run with no brief answered HTTP {code}", {}
        api("DELETE", f"/api/chat/sessions/{sid}")
        deleted = True
        code, data = api("GET", base)
        if code != 200 or (data or {}).get("run") is not None:
            return "fail", "the run stayed after its chat was deleted", {}
        return "pass", "", {"runId": run.get("runId"), "stopReason": stopped.get("stopReason")}
    finally:
        if not deleted:
            api("DELETE", f"/api/chat/sessions/{sid}")
