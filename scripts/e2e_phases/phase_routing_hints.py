"""E2E phase 214: the model's own location decides which model it is, not the repo hints a request carries (L-16).

Checks, against a running backend, with the smallest local MLX text model in the Hugging Face cache:
1. loaded with a ``canonicalRepo`` naming a TensorFold-only catalogue row, it loads on MLX as itself
   (``canonicalRepo`` reported back is the cache folder's own repo) and answers a prompt;
2. loaded with a ``modelRef`` and ``modelName`` naming that row as well, it still loads as itself;
3. a chat message carrying the same wrong hints keeps the loaded model's repo and answers.
Before the fix the first check answered 500 "Qwen3.8 Flash Next runs only on TensorFold" (D-13).
Nothing is written to disk; the model is unloaded at the end.
"""

from __future__ import annotations

import json
from typing import Any

WRONG_REPO = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"  # a TensorFold-only catalogue row
WRONG_NAME = "Qwen3.8 Flash Next MLX 4-bit (TensorFold)"


def phase_routing_hints(cap: Any):
    import e2e_test_suite as suite  # scripts/ is on sys.path when the suite runs

    phase = suite.PhaseResult(phase=214, name="Routing from the model's path, not repo hints")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 214", "skip", reason="backend not reachable"))
        return phase
    http = suite._voice_http
    state: dict[str, Any] = {}

    def _json(method: str, path: str, payload: dict | None = None, timeout: float = 120.0):
        body = json.dumps(payload).encode() if payload is not None else None
        code, raw = http(method, path, body=body, content_type="application/json" if body else None, timeout=timeout)
        try:
            return code, json.loads(raw.decode() or "null")
        except ValueError:
            return code, raw

    def _detail(body: Any) -> str:
        detail = body.get("detail") if isinstance(body, dict) else body
        if isinstance(detail, dict):
            detail = detail.get("message")
        return str(detail)

    def _model() -> dict[str, Any] | None:
        if "item" not in state:
            code, body = _json("GET", "/api/workspace", timeout=60.0)
            library = (body or {}).get("library") if code == 200 and isinstance(body, dict) else []
            rows = [
                i for i in library or []
                if i.get("modelType") == "text" and not i.get("broken") and i.get("format") == "MLX"
                and i.get("sourceKind") == "HF cache" and i.get("name") != WRONG_REPO
            ]
            rows.sort(key=lambda i: float(i.get("sizeGb") or 0))
            state["item"] = rows[0] if rows else None
        return state["item"]

    def _unload() -> None:
        _json("POST", "/api/models/unload", {})

    def _load(model_ref: str, model_name: str):
        item = _model()
        _unload()  # each check is a fresh load, not the same-model shortcut
        payload = {"modelRef": model_ref, "modelName": model_name, "canonicalRepo": WRONG_REPO, "source": "library",
                   "backend": "auto", "path": item["path"], "contextTokens": 2048}
        code, body = _json("POST", "/api/models/load", payload, timeout=600.0)
        if code != 200:
            return "fail", f"load HTTP {code}: {_detail(body)[:200]}", {"model": item["name"]}
        runtime = (body or {}).get("runtime") or {}
        loaded = runtime.get("loadedModel") or {}
        detail = {"model": item["name"], "engine": runtime.get("engine"), "ref": loaded.get("ref"),
                  "canonicalRepo": loaded.get("canonicalRepo")}
        if loaded.get("canonicalRepo") != item["name"]:
            return "fail", f"canonicalRepo is {loaded.get('canonicalRepo')!r}, not the folder's {item['name']!r}", detail
        if loaded.get("ref") != item["name"]:
            return "fail", f"loaded as {loaded.get('ref')!r}, not {item['name']!r}", detail
        if runtime.get("engine") != "mlx":
            return "fail", f"engine {runtime.get('engine')!r}", detail
        return "pass", "", detail

    def _chat():
        item = _model()
        code, gen = _json("POST", "/api/chat/generate", {
            "prompt": "Name the capital of France in one word.", "maxTokens": 24, "temperature": 0.0,
            "modelRef": WRONG_REPO, "modelName": WRONG_NAME, "canonicalRepo": WRONG_REPO, "source": "library",
            "backend": "auto", "path": item["path"], "thinkingMode": "off",
        }, timeout=300.0)
        if code != 200 or not isinstance(gen, dict):
            return "fail", f"generate HTTP {code}: {_detail(gen)[:200]}", {}
        session = gen.get("session") or {}
        try:
            text = str(session["messages"][-1]["text"])
        except (KeyError, IndexError, TypeError):
            return "fail", "no reply in the session", {}
        detail = {"reply": text[:80], "canonicalRepo": session.get("canonicalRepo")}
        if session.get("canonicalRepo") != item["name"]:
            return "fail", f"session canonicalRepo is {session.get('canonicalRepo')!r}", detail
        if not text.strip():
            return "fail", "empty reply", detail
        return "pass", "", detail

    def _guarded(fn):
        def check():
            if _model() is None:
                return "skip", "no MLX text model in the Hugging Face cache", {}
            return fn()
        return check

    try:
        item = _model()
        for name, fn in (
            ("Wrong canonicalRepo: loads as the model at the path, on MLX",
             _guarded(lambda: _load(item["name"], item["name"]))),
            ("Wrong canonicalRepo, modelRef and modelName: still loads as itself",
             _guarded(lambda: _load(WRONG_REPO, WRONG_NAME))),
            ("Chat message with the same wrong hints keeps the repo and answers", _guarded(_chat)),
        ):
            phase.checks.append(suite._check(name, fn))
    finally:
        _unload()
    statuses = {c.status for c in phase.checks}
    phase.status = "fail" if "fail" in statuses else ("pass" if "pass" in statuses else "skip")
    return phase
