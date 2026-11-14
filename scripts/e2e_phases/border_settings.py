"""Settings round trip for the stronger-borders switch: ``strongControlBorders`` false, then true, through ``/api/settings``.

Called from phase 7 of ``scripts/e2e_test_suite.py`` as ``check(suite)`` and returns the ``(status, reason, detail)``
tuple the suite's ``_check`` helper expects. The value the user had is put back at the end.
"""

from __future__ import annotations

import json
from typing import Any


def check(suite: Any):
    def api(method: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        code, raw = suite._voice_http(method, "/api/settings", body=body, content_type="application/json" if body else None, timeout=15.0)
        try:
            return code, json.loads(raw.decode("utf-8"))
        except ValueError:
            return code, None

    code, data = api("GET")
    settings = (data or {}).get("settings") if isinstance(data, dict) else None
    if code != 200 or not isinstance(settings, dict) or "strongControlBorders" not in settings:
        return "fail", f"GET /api/settings HTTP {code}: strongControlBorders missing", {}
    original = bool(settings["strongControlBorders"])
    seen: list[bool] = []
    try:
        for value in (False, True):
            code, data = api("PATCH", {"strongControlBorders": value})
            got = ((data or {}).get("settings") or {}).get("strongControlBorders") if isinstance(data, dict) else None
            code2, data2 = api("GET")
            again = ((data2 or {}).get("settings") or {}).get("strongControlBorders") if isinstance(data2, dict) else None
            if code != 200 or got is not value or again is not value:
                return "fail", f"set {value}: HTTP {code}, patch read {got}, get read {again}", {}
            seen.append(value)
    finally:
        api("PATCH", {"strongControlBorders": original})
    return "pass", "", {"sequence": seen, "restoredTo": original}
