"""E2E phase 211: Benchmarks and Conversion to MLX (overnight review, stream A).

Checks, against a running backend:

Benchmarks
1. request guards: an unknown mode and a too-small ``maxTokens`` are refused (422), a model that is not on
   disk is a plain 400;
2. a throughput run on the smallest local MLX text model and on the smallest local GGUF text model gives a
   speed above zero, generated tokens, and lands first in the saved history;
3. a perplexity run (WikiText-2, 8 samples of 128 tokens) gives a finite perplexity above 1;
4. a task-accuracy run (MMLU, 10 questions, zero-shot) answers every question and gives an accuracy
   between 0 and 1;
5. the saved history survives a fresh ``GET /api/workspace``, one run can be deleted, and every run this
   phase made is removed again.

Conversion
6. request guards: no source, a dtype outside the list, 9 quantisation bits, a local path that does not
   exist and a GGUF-only repo as the base are each refused with their own message, and none starts a run;
7. real conversion of the smallest local Transformers checkpoint (10 GB at most, never downloaded) to
   4-bit MLX in a temporary folder: the folder holds ``config.json`` and safetensors, the log names the
   bits per weight, and the size is reported;
8. converting into the same folder again is refused ("already exists") and leaves the folder alone;
9. the converted model loads and answers a prompt.
The temporary folder is removed at the end. Nothing is downloaded by the conversion checks; the two
dataset checks (3, 4) need the Hub once and skip when it cannot be reached.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

_MAX_CONVERT_SOURCE_GB = 10.0
_OFFLINE_MARKERS = ("couldn't reach", "could not reach", "offline", "connection", "max retries", "timed out",
                    "name resolution", "temporary failure")


def phase_21_benchmarks_conversion(cap: Any):
    import e2e_test_suite as suite  # scripts/ is on sys.path when the suite runs

    phase = suite.PhaseResult(phase=211, name="Benchmarks and Conversion to MLX")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 211", "skip", reason="backend not reachable"))
        return phase
    http = suite._voice_http
    state: dict[str, Any] = {"runs": []}

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

    def _library() -> list[dict[str, Any]]:
        if "library" not in state:
            code, body = _json("GET", "/api/workspace", timeout=60.0)
            state["workspace"] = body if code == 200 and isinstance(body, dict) else {}
            state["library"] = [
                item for item in state["workspace"].get("library") or []
                if item.get("modelType") == "text" and not item.get("broken")
            ]
        return state["library"]

    def _smallest(fmt: str) -> dict[str, Any] | None:
        rows = sorted((i for i in _library() if i.get("format") == fmt), key=lambda i: float(i.get("sizeGb") or 0))
        return rows[0] if rows else None

    def _run(item: dict[str, Any], **extra: Any):
        payload = {
            "modelRef": item["name"], "modelName": item["name"], "source": "library",
            "backend": item.get("backend") or "auto", "path": item["path"],
            "contextTokens": 4096, "maxTokens": 96, **extra,
        }
        code, body = _json("POST", "/api/benchmarks/run", payload, timeout=1800.0)
        if code == 200 and isinstance(body, dict) and isinstance(body.get("result"), dict):
            state["runs"].append(body["result"]["id"])
        return code, body

    def _unload() -> None:
        _json("POST", "/api/models/unload", {})
        _json("POST", "/api/models/processes/stop-idle", {})

    # ---- Benchmarks -------------------------------------------------------------------------------

    def _bench_guards():
        code, _ = _json("POST", "/api/benchmarks/run", {"mode": "nonsense"})
        if code != 422:
            return "fail", f"unknown mode: expected 422, got {code}", {}
        code, _ = _json("POST", "/api/benchmarks/run", {"maxTokens": 1})
        if code != 422:
            return "fail", f"maxTokens 1: expected 422, got {code}", {}
        code, body = _json("POST", "/api/benchmarks/run", {
            "modelRef": "nobody/not-a-model-e2e", "path": "/nonexistent/e2e-model", "source": "library", "backend": "mlx",
        }, timeout=300.0)
        if code == 200:
            if isinstance(body, dict) and isinstance(body.get("result"), dict):
                state["runs"].append(body["result"]["id"])
            return "fail", "a model that is not on disk produced a benchmark result", {}
        if code != 400 or "Traceback" in _detail(body):
            return "fail", f"missing model: expected a plain 400, got {code}: {_detail(body)[:160]}", {}
        return "pass", "", {"missingModel": _detail(body)[:120]}

    def _throughput(fmt: str, backend_ok: bool, why: str):
        def check():
            if not backend_ok:
                return "skip", why, {}
            item = _smallest(fmt)
            if item is None:
                return "skip", f"no local {fmt} text model", {}
            code, body = _run(item, mode="throughput")
            if code != 200:
                return "fail", f"HTTP {code}: {_detail(body)[:200]}", {"model": item["name"]}
            result = body["result"]
            problems = []
            if not (result.get("tokS") or 0) > 0:
                problems.append(f"tokS {result.get('tokS')}")
            if not (result.get("completionTokens") or 0) > 0:
                problems.append("no completion tokens")
            if (body.get("benchmarks") or [{}])[0].get("id") != result["id"]:
                problems.append("run is not first in the history")
            if result.get("mode") != "throughput":
                problems.append(f"mode {result.get('mode')}")
            state[f"speed_{fmt}"] = item
            detail = {"model": item["name"], "tokS": result.get("tokS"), "completionTokens": result.get("completionTokens"),
                      "loadSeconds": result.get("loadSeconds"), "engine": result.get("engineLabel")}
            return ("fail", ", ".join(problems), detail) if problems else ("pass", "", detail)
        return check

    def _dataset_skip(body: Any) -> str | None:
        text = _detail(body).lower()
        return "dataset could not be fetched (offline)" if any(marker in text for marker in _OFFLINE_MARKERS) else None

    def _perplexity():
        item = state.get("speed_MLX")
        if item is None:
            return "skip", "no MLX model was benchmarked", {}
        code, body = _run(item, mode="perplexity", perplexityNumSamples=8, perplexitySeqLength=128)
        if code != 200:
            skip = _dataset_skip(body)
            return ("skip", skip, {}) if skip else ("fail", f"HTTP {code}: {_detail(body)[:200]}", {})
        result = body["result"]
        ppl = result.get("perplexity")
        if not isinstance(ppl, (int, float)) or not math.isfinite(ppl) or ppl <= 1:
            return "fail", f"perplexity {ppl}", {}
        if result.get("perplexityNumSamples") != 8:
            return "fail", f"samples {result.get('perplexityNumSamples')}", {}
        return "pass", "", {"perplexity": ppl, "stdError": result.get("perplexityStdError"),
                            "evalSeconds": result.get("evalSeconds")}

    def _task_accuracy():
        item = state.get("speed_MLX")
        if item is None:
            return "skip", "no MLX model was benchmarked", {}
        code, body = _run(item, mode="task_accuracy", taskName="mmlu", taskLimit=10, taskNumShots=0)
        if code != 200:
            skip = _dataset_skip(body)
            return ("skip", skip, {}) if skip else ("fail", f"HTTP {code}: {_detail(body)[:200]}", {})
        result = body["result"]
        accuracy = result.get("taskAccuracy")
        if result.get("taskTotal") != 10 or not isinstance(accuracy, (int, float)) or not 0 <= accuracy <= 1:
            return "fail", f"total {result.get('taskTotal')}, accuracy {accuracy}", {}
        return "pass", "", {"accuracy": accuracy, "correct": result.get("taskCorrect")}

    def _history():
        if not state["runs"]:
            return "skip", "no benchmark run was made", {}
        code, body = _json("GET", "/api/workspace", timeout=60.0)
        saved = [run.get("id") for run in (body or {}).get("benchmarks") or []] if code == 200 else []
        missing = [run_id for run_id in state["runs"] if run_id not in saved]
        if missing:
            return "fail", f"runs not in the saved history: {missing}", {}
        code, _ = _json("DELETE", "/api/benchmarks/bench-not-there-e2e")
        if code != 404:
            return "fail", f"delete of an unknown run: expected 404, got {code}", {}
        left = list(state["runs"])
        for run_id in list(state["runs"]):
            code, body = _json("DELETE", f"/api/benchmarks/{run_id}")
            if code == 200:
                state["runs"].remove(run_id)
                left = [run.get("id") for run in body.get("benchmarks") or []]
                if run_id in left:
                    return "fail", f"{run_id} still listed after its delete", {}
            else:
                return "fail", f"delete {run_id}: HTTP {code}", {}
        return "pass", "", {"deleted": len(saved) - len(left)}

    # ---- Conversion -------------------------------------------------------------------------------

    def _convert(payload: dict[str, Any], timeout: float = 120.0):
        return _json("POST", "/api/models/convert", payload, timeout=timeout)

    def _convert_guards():
        cases = (
            ("no source", {}, 400, "not a valid target"),
            ("dtype outside the list", {"modelRef": "org/model", "dtype": "floaty"}, 422, ""),
            ("9 bits", {"modelRef": "org/model", "qBits": 9}, 422, ""),
            ("missing local path", {"modelRef": "org/model", "path": "/nonexistent/e2e-source"}, 400, "Source path not found"),
            ("GGUF-only repo as the base", {"modelRef": "org/model-GGUF", "hfRepo": "org/model-GGUF"}, 400, "GGUF-only"),
        )
        _library()
        system = (state.get("workspace") or {}).get("system") or {}
        problems = []
        for name, payload, want, needle in cases:
            code, body = _convert(payload)
            if code == 400 and not system.get("converterAvailable") and want == 400:
                continue  # without the converter every 400 is the "unavailable" message
            if code != want or (needle and needle not in _detail(body)):
                problems.append(f"{name}: HTTP {code} {_detail(body)[:90]}")
        return ("fail", "; ".join(problems), {}) if problems else ("pass", "", {})

    def _convert_real():
        _library()
        system = (state.get("workspace") or {}).get("system") or {}
        if not system.get("converterAvailable"):
            return "skip", "MLX converter not available on this machine", {}
        candidates = sorted(
            (i for i in _library() if i.get("format") == "Transformers"
             and 0 < float(i.get("sizeGb") or 0) <= _MAX_CONVERT_SOURCE_GB),
            key=lambda i: float(i.get("sizeGb") or 0),
        )
        if not candidates:
            return "skip", f"no local Transformers checkpoint of {_MAX_CONVERT_SOURCE_GB:g} GB or less (none is downloaded)", {}
        workdir = Path(tempfile.mkdtemp(prefix="chaosengine-e2e-convert-"))
        state["workdir"] = workdir
        tried = []
        for item in candidates[:4]:
            output = workdir / "converted-4bit-mlx"
            code, body = _convert({
                "modelRef": item["name"], "path": item["path"], "hfRepo": item["name"], "outputPath": str(output),
                "quantize": True, "qBits": 4, "qGroupSize": 64, "dtype": "float16",
            }, timeout=3600.0)
            if code != 200:
                tried.append(f"{item['name']}: {_detail(body)[:110]}")
                if "does not support architecture" in _detail(body):
                    continue  # this mlx-lm cannot convert it; try the next checkpoint
                return "fail", "; ".join(tried), {}
            conversion = body.get("conversion") or {}
            problems = []
            if not (output / "config.json").is_file():
                problems.append("no config.json in the output")
            if not list(output.glob("*.safetensors")):
                problems.append("no safetensors in the output")
            if "bits per weight" not in str(conversion.get("log")):
                problems.append("log does not name the bits per weight")
            if not (conversion.get("outputSizeGb") or 0) > 0:
                problems.append(f"outputSizeGb {conversion.get('outputSizeGb')}")
            if conversion.get("qBits") != 4:
                problems.append(f"qBits {conversion.get('qBits')}")
            state["converted"] = {"path": str(output), "source": item}
            detail = {"source": item["name"], "sourceSizeGb": conversion.get("sourceSizeGb"),
                      "outputSizeGb": conversion.get("outputSizeGb")}
            return ("fail", ", ".join(problems), detail) if problems else ("pass", "", detail)
        return "skip", "no local checkpoint has an architecture this mlx-lm converts: " + "; ".join(tried), {}

    def _convert_again():
        converted = state.get("converted")
        if not converted:
            return "skip", "nothing was converted", {}
        output = Path(converted["path"])
        before = sorted(p.name for p in output.iterdir())
        item = converted["source"]
        code, body = _convert({"modelRef": item["name"], "path": item["path"], "hfRepo": item["name"],
                               "outputPath": str(output), "quantize": True, "qBits": 4}, timeout=3600.0)
        if code != 400 or "already exists" not in _detail(body):
            return "fail", f"expected 400 'already exists', got {code}: {_detail(body)[:160]}", {}
        if not output.is_dir() or sorted(p.name for p in output.iterdir()) != before:
            return "fail", "the existing output folder was changed by the refused run", {}
        return "pass", "", {}

    def _converted_chats():
        converted = state.get("converted")
        if not converted:
            return "skip", "nothing was converted", {}
        return suite._load_unload_prompt("e2e-converted-4bit-mlx", path=converted["path"], backend="mlx",
                                         prompt="What is the capital of France? One short sentence.", max_tokens=64)

    try:
        for name, fn in (
            ("benchmark request guards", _bench_guards),
            ("throughput on the smallest MLX model", _throughput("MLX", cap.mlx_usable, "MLX not usable on this machine")),
            ("throughput on the smallest GGUF model", _throughput("GGUF", cap.gguf_available, "llama.cpp not available")),
            ("perplexity on WikiText-2", _perplexity),
            ("task accuracy on MMLU", _task_accuracy),
            ("history saved, one run deleted, phase runs removed", _history),
            ("conversion request guards", _convert_guards),
            ("Transformers checkpoint converted to 4-bit MLX", _convert_real),
            ("existing output folder refused and untouched", _convert_again),
            ("converted model loads and answers", _converted_chats),
        ):
            phase.checks.append(suite._check(name, fn))
    finally:
        _unload()
        for run_id in state["runs"]:
            _json("DELETE", f"/api/benchmarks/{run_id}")
        workdir = state.get("workdir")
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)
    statuses = {c.status for c in phase.checks}
    phase.status = "fail" if "fail" in statuses else ("pass" if "pass" in statuses else "skip")
    return phase
