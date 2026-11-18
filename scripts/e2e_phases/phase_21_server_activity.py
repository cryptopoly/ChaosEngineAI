"""Phase 21: Server Activity (D-88). The structured model events appear for a real load, answer and unload.

Loads the smallest local Qwen3 0.6B or Qwen2.5 0.5B MLX model through the CLI (load, one prompt, unload), then reads
``/api/server/events`` since the newest id seen before the load and checks the model's life is there: requested,
loading, loaded (with seconds and memory), a generation with tokens per second, and an unload by the user, the load
events sharing one request id. Also reads the first line of the event stream and of the old log stream (kept for
the CLI and the agent bridge). Skips the model part with the reason when no small MLX model is installed.

Called from ``scripts/e2e_test_suite.py`` as ``phase_21_server_activity(cap)``.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

SMALL_MODELS = ("Qwen3-0.6B", "Qwen2.5-0.5B")


def _get(suite: Any, path: str) -> tuple[int, Any]:
    code, raw = suite._voice_http("GET", path, timeout=30.0)
    try:
        return code, json.loads(raw.decode("utf-8")) if raw.strip() else None
    except ValueError:
        return code, raw


def _first_sse_line(suite: Any, path: str, timeout: float = 10.0) -> str:
    """The first ``data:`` line of a server-sent stream (empty when none arrives)."""
    base = f"http://{suite._HOST}:{suite._PORT}"
    token = None
    try:
        with urllib.request.urlopen(f"{base}/api/auth/session", timeout=5.0) as resp:
            token = (json.loads(resp.read().decode("utf-8")) or {}).get("apiToken")
    except Exception:  # noqa: BLE001
        pass
    request = urllib.request.Request(f"{base}{path}")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        for _ in range(400):
            line = resp.readline().decode("utf-8", errors="replace")
            if not line:
                break
            if line.startswith("data: "):
                return line[6:].strip()
    return ""


def phase_21_server_activity(cap: Any):
    import e2e_test_suite as suite  # scripts/ is on sys.path when the suite runs

    phase = suite.PhaseResult(phase=21, name="Server Activity (model events)")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 21", "skip", reason="backend not reachable"))
        return phase
    state: dict[str, Any] = {}

    def events_route():
        code, body = _get(suite, "/api/server/events?limit=1")
        if code == 404:
            return "skip", "events route not in this build", {}
        if code != 200 or not isinstance(body, dict) or not {"events", "capacity", "dropped"} <= set(body):
            return "fail", f"HTTP {code}: {str(body)[:160]}", {}
        state["since"] = body["events"][-1]["id"] if body["events"] else 0
        state["ok"] = True
        return "pass", "", {"capacity": body["capacity"], "dropped": body["dropped"]}

    def load_answer_unload():
        if not state.get("ok"):
            return "skip", "events route not present", {}
        pick = None
        for needle in SMALL_MODELS:
            pick = suite._pick_model_by_ref_prefix(cap.local_mlx_models, needle)
            if pick:
                break
        if not pick or not cap.mlx_usable:
            return "skip", f"no small MLX model ({', '.join(SMALL_MODELS)}) or MLX not usable", {}
        ref, path = pick
        status, reason, detail = suite._load_unload_prompt(ref, path=path, max_tokens=24, context=2048, load_timeout=600.0)
        if status != "pass":
            return status, f"load/prompt/unload: {reason}", detail
        state["ref"] = ref
        return "pass", "", {"model": ref, "tokS": detail.get("tokS")}

    def events_for_the_run():
        if "ref" not in state:
            return "skip", "no model run", {}
        code, body = _get(suite, f"/api/server/events?since={state['since']}&limit=500")
        if code != 200 or not isinstance(body, dict):
            return "fail", f"HTTP {code}", {}
        events = body["events"]
        kinds = [e["kind"] for e in events]
        missing = [k for k in ("requested", "loading", "loaded", "generation", "unloaded") if k not in kinds]
        if missing:
            return "fail", f"missing {missing}; saw {kinds}", {}
        loaded = [e for e in events if e["kind"] == "loaded"][-1]
        load_ids = {e["requestId"] for e in events if e["kind"] in ("requested", "loading", "loaded")}
        generation = [e for e in events if e["kind"] == "generation"][-1]
        unloaded = [e for e in events if e["kind"] == "unloaded"][-1]
        problems = []
        if not loaded["data"].get("seconds"):
            problems.append("loaded has no seconds")
        if loaded["data"].get("memoryBytes") is None:
            problems.append("loaded has no memory")
        if len(load_ids) != 1:
            problems.append(f"load events carry {len(load_ids)} request ids")
        if not (generation["data"].get("tokS") or 0) > 0:
            problems.append("generation has no tok/s")
        if unloaded["data"].get("reason") != "user":
            problems.append(f"unload reason {unloaded['data'].get('reason')}")
        if problems:
            return "fail", "; ".join(problems), {"kinds": kinds}
        return "pass", "", {
            "kinds": kinds,
            "loadSeconds": loaded["data"]["seconds"],
            "memoryBytes": loaded["data"]["memoryBytes"],
            "tokS": generation["data"]["tokS"],
            "engine": loaded.get("engine"),
        }

    def event_stream():
        if not state.get("ok"):
            return "skip", "events route not present", {}
        line = _first_sse_line(suite, "/api/server/events/stream")
        if not line:
            return "skip", "no event yet to stream" if not state.get("ref") else "stream sent nothing", {}
        event = json.loads(line)
        return ("pass", "", {"kind": event.get("kind")}) if "kind" in event else ("fail", line[:160], {})

    def log_stream_kept():
        line = _first_sse_line(suite, "/api/server/logs/stream")
        if not line:
            return "skip", "no log lines yet", {}
        entry = json.loads(line)
        if not {"ts", "source", "level", "message"} <= set(entry):
            return "fail", line[:160], {}
        return "pass", "", {}

    phase.checks.append(suite._check("events route shape", events_route))
    phase.checks.append(suite._check("load, answer and unload a small MLX model", load_answer_unload))
    phase.checks.append(suite._check("events for that run (requested to unloaded)", events_for_the_run))
    phase.checks.append(suite._check("event stream sends events", event_stream))
    phase.checks.append(suite._check("old log stream still answers", log_stream_kept))
    if any(c.status == "fail" for c in phase.checks):
        phase.status = "fail"
    elif all(c.status == "skip" for c in phase.checks):
        phase.status = "skip"
    return phase
