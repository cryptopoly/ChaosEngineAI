"""FU-165 checks: catalog rows say what their time was measured at (``estimateRef``), and the Storyboard estimate
grows with the project's size. No model runs. Each function returns ``(status, reason, detail)`` for the suite's
``_check``; ``h`` is the suite module.
"""

from __future__ import annotations

import json
from typing import Any


def catalog_refs(h: Any, kind: str) -> tuple[str, str, dict[str, Any]]:
    """Every ``estimateRef`` in the live ``kind`` catalog has four positive integers; at least one row has one."""
    rc, payload, err = h._cli_json(f"{kind}-catalog", timeout=20.0)
    if rc != 0 or not isinstance(payload, dict):
        return "fail", f"{kind}-catalog failed: {err[:160]}", {}
    with_ref, problems = 0, []
    for family in payload.get("families") or []:
        for variant in family.get("variants") or []:
            ref = variant.get("estimateRef")
            if ref is None:
                continue
            with_ref += 1
            bad = [k for k in ("width", "height", "frames", "steps")
                   if not isinstance((ref or {}).get(k), int) or isinstance(ref.get(k), bool) or ref[k] <= 0]
            if bad or not variant.get("estimatedGenerationSeconds"):
                problems.append(f"{variant.get('id')}: bad estimateRef {ref!r}")
    if problems:
        return "fail", "; ".join(problems[:5]), {"count": len(problems)}
    if not with_ref:
        return "fail", f"no {kind} row carries estimateRef", {}
    return "pass", "", {"rowsWithRef": with_ref}


def storyboard_size_scales(h: Any) -> tuple[str, str, dict[str, Any]]:
    """Doubling a project's size (720p to 1440p) raises the batch estimate's ``seconds``."""

    def call(method: str, path: str, payload: Any = None) -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        code, data = h._voice_http(method, f"/api/storyboard{path}", body=body,
                                   content_type="application/json" if body else None, timeout=60.0)
        try:
            return code, json.loads(data.decode("utf-8")) if data.strip() else None
        except ValueError:
            return code, data

    code, body = call("POST", "/projects", {"name": "E2E time estimate"})
    if code == 404:
        return "skip", "storyboard not present in this build", {}
    if code != 200:
        return "fail", f"create project: HTTP {code}", {}
    pid, rev = body["project"]["id"], body["project"]["revision"]
    try:
        def op(name: str, args: dict[str, Any]) -> None:
            nonlocal rev
            code, out = call("POST", f"/projects/{pid}/ops", {"revision": rev, "op": name, "args": args})
            if code != 200:
                raise RuntimeError(f"{name}: HTTP {code} {str(out)[:160]}")
            rev = out["project"]["revision"]

        op("add_shot", {"title": "sea", "stillPrompt": "a lighthouse at dusk"})
        op("edit_project", {"defaults": {"resolution": "720p"}})
        _, small = call("POST", f"/projects/{pid}/estimate", {"what": "both"})
        op("edit_project", {"defaults": {"resolution": "1440p"}})
        _, large = call("POST", f"/projects/{pid}/estimate", {"what": "both"})
        a, b = (small or {}).get("seconds"), (large or {}).get("seconds")
        if a is None or b is None:
            return "skip", f"no estimate without installed models: {(small or {}).get('missing')}", {}
        detail = {"seconds720p": a, "seconds1440p": b, "image": (small or {}).get("imageModel"),
                  "video": (small or {}).get("videoModel")}
        refs = set()
        for kind in ("image", "video"):
            _, payload, _ = h._cli_json(f"{kind}-catalog", timeout=20.0)
            refs |= {v.get("name") for f in (payload or {}).get("families") or [] for v in f.get("variants") or []
                     if v.get("estimateRef")}
        if detail["image"] not in refs and detail["video"] not in refs:
            return "skip", "the models this project would use carry no estimateRef (flat catalog time)", detail
        if b <= a:
            return "fail", f"1440p estimate {b} s is not above 720p {a} s", detail
        return "pass", "", detail
    finally:
        h._voice_http("DELETE", f"/api/storyboard/projects/{pid}?revision={rev}")
