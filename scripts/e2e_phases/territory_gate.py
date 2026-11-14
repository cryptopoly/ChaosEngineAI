"""D-76 / FU-166 checks: territory-limited rows (MiniMax H3) are absent from the catalog, and download, preload and
generate are refused, while the user has not confirmed the territory statement. No weights are needed and nothing
downloads: when this host has confirmed the statement the refusal calls are skipped (they would start a 35 GB
download), and the listing check then expects the rows instead. ``h`` is the suite module; each function returns
``(status, reason, detail)`` for the suite's ``_check``.
"""

from __future__ import annotations

import json
from typing import Any

H3_KEY = "minimax-h3"
H3_REPO = "leejet/MiniMax-H3-GGUF"
H3_ID = f"{H3_REPO}~fl2va-pruned-q4km"


def _call(h: Any, method: str, path: str, payload: Any = None) -> tuple[int, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    code, data = h._voice_http(method, path, body=body, content_type="application/json" if body else None, timeout=60.0)
    try:
        return code, json.loads(data.decode("utf-8")) if data.strip() else None
    except ValueError:
        return code, data


def check(h: Any, kind: str) -> tuple[str, str, dict[str, Any]]:
    code, status = _call(h, "GET", "/api/licence-gate")
    if code == 404:
        return "skip", "licence gate not present in this build", {}
    if code != 200 or not isinstance(status, dict):
        return "fail", f"/api/licence-gate: HTTP {code}", {}
    confirmed = {m["key"] for m in status.get("models") or [] if m.get("confirmed")}
    route = "/api/images/catalog" if kind == "image" else "/api/video/catalog"
    code, catalog = _call(h, "GET", route)
    if code != 200 or not isinstance(catalog, dict):
        return "fail", f"{route}: HTTP {code}", {}
    gated = [v for f in catalog.get("families") or [] for v in f.get("variants") or [] if v.get("territoryGate")]
    leaked = [v["id"] for v in gated if v.get("territoryGate") not in confirmed]
    if leaked:
        return "fail", f"territory-limited rows listed without a confirmation: {leaked}", {}
    if H3_KEY in confirmed:
        return "skip", "this host confirmed the MiniMax H3 statement; refusal calls skipped (they would download)", {
            "listed": [v["id"] for v in gated]}
    refusals = (
        [("POST", "/api/video/download", {"repo": H3_REPO, "modelId": H3_ID}),
         ("POST", "/api/video/preload", {"modelId": H3_ID}),
         ("POST", "/api/video/generate", {"modelId": H3_ID, "prompt": "e2e territory gate"})]
        if kind == "video"
        else [("POST", "/api/images/download", {"repo": H3_REPO})]
    )
    for method, path, payload in refusals:
        code, body = _call(h, method, path, payload)
        if code != 403:
            return "fail", f"{path} answered HTTP {code}, expected 403: {str(body)[:160]}", {}
    return "pass", "", {"released": bool(status.get("models")), "refused": len(refusals)}
