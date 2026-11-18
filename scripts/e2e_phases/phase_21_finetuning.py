"""Phase 21 (Fine-Tuning part): the adapter list and the honest "not built" answer of the training route.

Called from ``scripts/e2e_test_suite.py`` as ``run(cap, suite)``. Needs no model, because the app has no training
backend: ``POST /api/finetuning/start`` must answer 501 and start nothing, and the status must stay ``idle``. The day
training is built, the 501 check fails on purpose: replace it with a real run (smallest MLX model, a few iterations,
progress, cancel, the adapter listed, the adapter applied in Chat). The manual run that proves mlx-lm can do this on
the Mac is in ``docs/progress/review-b.md``.

The adapter checks put two tiny folders (an ``adapter_config.json`` and a 4 KB weights file each) in a folder with a
unique name under the backend's data directory, which the scan reads first, and delete that folder at the end. They
need the backend on this machine and skip when its data directory is not a local folder.

Registered under key 212 so it cannot collide with another review stream's phase 21 at merge time.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

PHASE_KEY = 212


def bind(suite: Any):
    """The suite's ``PHASES`` entry: a named function of ``cap``, so the run log prints its name."""

    def phase_21_finetuning(cap: Any):
        return run(cap, suite)

    return phase_21_finetuning


def run(cap: Any, suite: Any):
    phase = suite.PhaseResult(phase=PHASE_KEY, name="Fine-Tuning (adapter list, training not built)")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 21 fine-tuning", "skip", reason="backend not reachable"))
        return phase

    def api(method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        code, raw = suite._voice_http(method, path, body=body, content_type="application/json" if body else None, timeout=timeout)
        try:
            return code, json.loads(raw.decode("utf-8")) if raw.strip() else None
        except ValueError:
            return code, raw

    tag = f"e2e-adapters-{uuid.uuid4().hex[:8]}"
    state: dict[str, Any] = {}

    def adapters_shape():
        code, data = api("GET", "/api/adapters")
        if code == 404:
            return "skip", "fine-tuning routes not present in this build", {}
        if code != 200 or not isinstance(data, dict) or not isinstance(data.get("adapters"), list):
            return "fail", f"HTTP {code}: {str(data)[:160]}", {}
        if data.get("count") != len(data["adapters"]):
            return "fail", f"count {data.get('count')} for {len(data['adapters'])} rows", {}
        state["ok"] = True
        return "pass", "", {"adapters": data["count"]}

    def seeded_adapters_listed():
        if not state.get("ok"):
            return "skip", "fine-tuning routes not present", {}
        code, body = api("GET", "/api/settings")
        raw = ((body or {}).get("settings") or {}).get("dataDirectory") if code == 200 and isinstance(body, dict) else None
        root = Path(raw).expanduser() if raw else None
        if root is None or not root.is_dir():
            return "skip", "the backend's data directory is not a folder on this machine", {}
        base = root / tag
        state["folder"] = base
        configs = {
            "peft-style": {"base_model_name_or_path": "example-org/base-model", "r": 8, "lora_alpha": 16},
            "mlx-lm-style": {"model": "/models/example-mlx-model", "fine_tune_type": "lora", "lora_parameters": {"rank": 8}},
        }
        for name, config in configs.items():
            (base / name).mkdir(parents=True)
            (base / name / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
            (base / name / "adapters.safetensors").write_bytes(b"\0" * 4096)
        code, data = api("GET", "/api/adapters")
        if code != 200:
            return "fail", f"HTTP {code}", {}
        mine = {row.get("name"): row for row in data["adapters"] if tag in str(row.get("path", ""))}
        if set(mine) != set(configs):
            return "fail", f"listed {sorted(mine)}, expected {sorted(configs)}", {}
        # The fields the tab reads; it crashed on a row without them.
        for name, row in mine.items():
            missing = [key for key in ("name", "path", "base_model", "size_bytes", "created") if key not in row]
            if missing:
                return "fail", f"{name} lacks {missing}", {}
            if not isinstance(row["size_bytes"], int) or row["size_bytes"] < 4096:
                return "fail", f"{name} size_bytes is {row['size_bytes']!r}", {}
        bases = {name: row["base_model"] for name, row in mine.items()}
        if bases != {"peft-style": "example-org/base-model", "mlx-lm-style": "/models/example-mlx-model"}:
            return "fail", f"base models read as {bases}", {}
        return "pass", "", {"bases": bases}

    def status_idle():
        if not state.get("ok"):
            return "skip", "fine-tuning routes not present", {}
        code, data = api("GET", "/api/finetuning/status")
        if code != 200 or not isinstance(data, dict):
            return "fail", f"HTTP {code}", {}
        if data.get("status") != "idle" or data.get("run_id"):
            return "fail", f"status {data.get('status')!r}, run_id {data.get('run_id')!r}: nothing can train, so nothing can be running", {}
        return "pass", "", {"status": data["status"]}

    def bad_body_refused():
        if not state.get("ok"):
            return "skip", "fine-tuning routes not present", {}
        answers = {
            "empty model path": api("POST", "/api/finetuning/start", {"modelPath": "", "datasetPath": "x.jsonl"})[0],
            "rank 0": api("POST", "/api/finetuning/start", {"modelPath": "m", "datasetPath": "x.jsonl", "loraRank": 0})[0],
            "negative learning rate": api("POST", "/api/finetuning/start", {"modelPath": "m", "datasetPath": "x.jsonl", "learningRate": -1})[0],
        }
        if any(code != 422 for code in answers.values()):
            return "fail", f"expected 422 for each, got {answers}", {}
        return "pass", "", answers

    def start_says_not_built():
        if not state.get("ok"):
            return "skip", "fine-tuning routes not present", {}
        out = Path(state["folder"]) / "would-be-output" if state.get("folder") else None
        payload = {"modelPath": "/nonexistent/model", "datasetPath": "/nonexistent/train.jsonl"}
        if out is not None:
            payload["outputPath"] = str(out)
        code, data = api("POST", "/api/finetuning/start", payload)
        if code == 200:
            return "fail", ("the route accepted a run. If training is now built, replace this check with a real run; "
                            "if not, it is claiming a job that never starts"), {"reply": str(data)[:200]}
        if code != 501:
            return "fail", f"HTTP {code}, expected 501", {}
        detail = (data or {}).get("detail") if isinstance(data, dict) else None
        message = detail.get("message") if isinstance(detail, dict) else str(detail)
        if "not available" not in (message or ""):
            return "fail", f"the 501 does not say why: {message!r}", {}
        after = api("GET", "/api/finetuning/status")[1]
        if (after or {}).get("status") != "idle":
            return "fail", f"status moved to {(after or {}).get('status')!r} after a refused start", {}
        if out is not None and out.exists():
            return "fail", "the refused start created the output folder", {}
        return "pass", "", {"message": message}

    try:
        for label, fn in [
            ("GET /api/adapters answers with rows and a matching count", adapters_shape),
            ("adapter folders are listed with the fields the tab reads (PEFT and mlx-lm configs)", seeded_adapters_listed),
            ("training status is idle", status_idle),
            ("a bad start body is refused with 422", bad_body_refused),
            ("a valid start is refused with 501, starts nothing and leaves the status idle", start_says_not_built),
        ]:
            phase.checks.append(suite._check(label, fn))
    finally:
        folder = state.get("folder")
        # Only ever the folder this run made: its name carries the run's own tag.
        if folder is not None and Path(folder).name == tag:
            shutil.rmtree(folder, ignore_errors=True)
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase
