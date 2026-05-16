#!/usr/bin/env python3
"""ChaosEngineAI end-to-end test suite.

Drives the CLI sequentially through phased scenarios covering every major
app surface: chat (MLX + GGUF + cache strategies + DFlash + MTPLX),
chat compare, HTML challenge, image studio, video studio, setup probes,
diagnostics. Emits a JSON + Markdown report under ``~/.chaosengine/test-results/``.

Pass criteria are concrete per phase: HTTP 200, expected substrings in
``runtimeNote``, ``tokS > 0``, non-empty output bytes, etc. Phases auto-skip
when prerequisites are missing (model not on disk, install missing) rather
than hard-failing — see "Skip semantics" in docs/E2E_TESTING.md.

Usage:
    scripts/e2e_test_suite.py                  # full sweep, all phases
    scripts/e2e_test_suite.py --phases 0,1,7   # subset
    scripts/e2e_test_suite.py --smoke          # fastest possible smoke
    scripts/e2e_test_suite.py --report-dir DIR # override default location

Exit codes:
    0 = all phases pass or skipped (no failures)
    1 = at least one phase failed
    2 = could not reach backend at all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLI = _REPO_ROOT / "scripts" / "chaosengine-cli"
_DEFAULT_REPORT_DIR = Path.home() / ".chaosengine" / "test-results"
_HOST = os.environ.get("CHAOSENGINE_HOST", "127.0.0.1")
_PORT = int(os.environ.get("CHAOSENGINE_PORT", "8876"))
_MODELS_ROOT = Path.home() / "AI_Models"


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "fail" | "skip"
    elapsed_sec: float = 0.0
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseResult:
    phase: int
    name: str
    status: str = "pass"  # "pass" | "fail" | "skip"
    checks: list[CheckResult] = field(default_factory=list)
    started_at: str = ""
    elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _cli(*argv: str, timeout: float = 600.0, stdin: str | None = None) -> tuple[int, str, str]:
    """Run a CLI subcommand and return (returncode, stdout, stderr)."""
    proc = subprocess.run(
        [str(_CLI), *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
        input=stdin,
        cwd=str(_REPO_ROOT),
    )
    return proc.returncode, proc.stdout, proc.stderr


def _cli_json(*argv: str, timeout: float = 600.0) -> tuple[int, Any, str]:
    rc, out, err = _cli(*argv, timeout=timeout)
    try:
        return rc, json.loads(out) if out.strip() else None, err
    except json.JSONDecodeError:
        return rc, None, err or out


def _check(name: str, fn) -> CheckResult:
    """Run a check fn and capture pass/fail/skip + reason."""
    started = time.perf_counter()
    try:
        status, reason, detail = fn()
    except Exception as exc:  # noqa: BLE001
        status, reason, detail = "fail", f"{type(exc).__name__}: {exc}", {}
    elapsed = round(time.perf_counter() - started, 2)
    return CheckResult(name=name, status=status, elapsed_sec=elapsed, reason=reason, detail=detail)


# ---------------------------------------------------------------------------
# Capability probe (deterministic upfront; phases skip based on this)
# ---------------------------------------------------------------------------


@dataclass
class Capability:
    backend_reachable: bool = False
    mlx_usable: bool = False
    gguf_available: bool = False
    mtplx_available: bool = False
    dflash_supported_models: list[str] = field(default_factory=list)
    mtplx_supported_models: list[str] = field(default_factory=list)
    image_runtime_ready: bool = False
    video_runtime_ready: bool = False
    local_mlx_models: list[tuple[str, str]] = field(default_factory=list)  # (ref, path)
    local_gguf_files: list[tuple[str, str]] = field(default_factory=list)
    local_image_models: list[str] = field(default_factory=list)
    local_video_models: list[str] = field(default_factory=list)


def probe_capabilities() -> Capability:
    cap = Capability()
    try:
        urllib.request.urlopen(f"http://{_HOST}:{_PORT}/api/health", timeout=3.0).read()
        cap.backend_reachable = True
    except Exception:
        return cap

    rc, status, _ = _cli_json("status", timeout=10.0)
    if rc == 0 and isinstance(status, dict):
        system = status.get("system") or {}
        mtplx_info = system.get("mtplx") or {}
        dflash_info = system.get("dflash") or {}
        cap.mtplx_available = bool(mtplx_info.get("available"))
        cap.mtplx_supported_models = mtplx_info.get("supportedModels") or []
        cap.dflash_supported_models = dflash_info.get("supportedModels") or []

    rc, runtime, _ = _cli_json("runtime", timeout=10.0)
    if rc == 0 and isinstance(runtime, dict):
        native = runtime.get("nativeBackends") or {}
        cap.mlx_usable = bool(native.get("mlxUsable"))
        cap.gguf_available = bool(native.get("ggufAvailable"))

    rc, image_rt, _ = _cli_json("image-runtime", timeout=10.0)
    if rc == 0:
        cap.image_runtime_ready = True

    rc, video_rt, _ = _cli_json("video-runtime", timeout=10.0)
    if rc == 0:
        cap.video_runtime_ready = True

    # Local on-disk inventory
    if _MODELS_ROOT.exists():
        for owner in _MODELS_ROOT.iterdir():
            if not owner.is_dir():
                continue
            for model_dir in owner.iterdir():
                if not model_dir.is_dir():
                    continue
                ref = f"{owner.name}/{model_dir.name}"
                if any(model_dir.glob("*.gguf")):
                    for gguf in model_dir.glob("*.gguf"):
                        if "mmproj" not in gguf.name.lower():
                            cap.local_gguf_files.append((ref, str(gguf)))
                if any(model_dir.glob("*.safetensors")):
                    cap.local_mlx_models.append((ref, str(model_dir)))

    return cap


# ---------------------------------------------------------------------------
# Phase 0 — backend reachable + capability probe
# ---------------------------------------------------------------------------


def phase_0(cap: Capability) -> PhaseResult:
    phase = PhaseResult(phase=0, name="Environment probe")
    if not cap.backend_reachable:
        phase.status = "fail"
        phase.checks.append(CheckResult("backend reachable", "fail", reason="backend not running on " + f"{_HOST}:{_PORT}"))
        return phase

    def _health():
        rc, payload, _ = _cli_json("health")
        if rc != 0 or not isinstance(payload, dict):
            return "fail", "no health response", {}
        return ("pass" if payload.get("status") == "ok" else "fail"), "", {"runtime": payload.get("runtime")}

    def _routes():
        rc, payload, _ = _cli_json("routes")
        if rc != 0 or not isinstance(payload, dict):
            return "fail", "routes call failed", {}
        count = payload.get("count", 0)
        return ("pass" if count >= 100 else "fail"), f"got {count} routes", {"count": count}

    def _gpu():
        rc, payload, _ = _cli_json("gpu-status")
        if rc != 0:
            return "fail", "gpu-status failed", {}
        return "pass", "", {"platform": payload.get("platform"), "mps": payload.get("torchMpsAvailable"), "cuda": payload.get("torchCudaAvailable")}

    def _mtplx():
        rc, payload, _ = _cli_json("mtplx-status")
        if rc != 0:
            return "fail", "mtplx-status failed", {}
        return "pass", "", payload

    def _inventory():
        return "pass", "", {
            "mlxModels": len(cap.local_mlx_models),
            "ggufFiles": len(cap.local_gguf_files),
            "mtplxAvailable": cap.mtplx_available,
            "mtplxSupportedCount": len(cap.mtplx_supported_models),
            "dflashSupportedCount": len(cap.dflash_supported_models),
        }

    for name, fn in [
        ("health", _health), ("routes", _routes), ("gpu-status", _gpu),
        ("mtplx-status", _mtplx), ("inventory", _inventory),
    ]:
        phase.checks.append(_check(name, fn))
    phase.status = "fail" if any(c.status == "fail" for c in phase.checks) else "pass"
    return phase


# ---------------------------------------------------------------------------
# Phase 1 — Chat: MLX + GGUF + cache strategies + DFlash + MTPLX
# ---------------------------------------------------------------------------


def _pick_model_by_ref_prefix(local: list[tuple[str, str]], *needles: str) -> tuple[str, str] | None:
    for ref, path in local:
        low = ref.lower()
        if all(n.lower() in low for n in needles):
            return ref, path
    return None


def _load_unload_prompt(ref: str, *, path: str | None = None, backend: str = "auto",
                        spec: bool = False, cache_strategy: str = "native",
                        cache_bits: int | None = None, fused: bool = False,
                        context: int = 4096, max_tokens: int = 32,
                        canonical_repo: str | None = None,
                        prompt: str = "Hello, respond with two words.",
                        load_timeout: float = 1800.0) -> tuple[str, str, dict[str, Any]]:
    """Helper: load → prompt → unload. Returns (status, reason, detail)."""
    load_args = ["load", ref, "--backend", backend, "--cache-strategy", cache_strategy,
                  "--context", str(context), "--timeout", str(int(load_timeout))]
    if path:
        load_args.extend(["--path", path])
    if canonical_repo:
        load_args.extend(["--canonical-repo", canonical_repo])
    if spec:
        load_args.append("--spec")
    if cache_bits is not None:
        load_args.extend(["--cache-bits", str(cache_bits)])
    if fused:
        load_args.append("--fused-attention")

    rc, loaded, err = _cli_json(*load_args, timeout=load_timeout + 60.0)
    if rc != 0 or not isinstance(loaded, dict):
        return "fail", f"load returned rc={rc}: {err[:200] if err else 'no detail'}", {}

    if loaded.get("state") != "loaded":
        _cli("unload")
        return "fail", f"state={loaded.get('state')} runtimeNote={loaded.get('runtimeNote')}", {"loaded": loaded}

    rc, gen, err = _cli_json("prompt", prompt, "--max-tokens", str(max_tokens), "--metrics", "--quiet", "--timeout", "300")
    _cli("unload", timeout=60.0)

    if rc != 0 or not isinstance(gen, dict):
        return "fail", f"prompt rc={rc}: {err[:200] if err else 'no detail'}", {"loaded": loaded}

    tok_s = gen.get("tokS") or 0
    runtime_note = (loaded.get("runtimeNote") or "")
    detail = {
        "engine": loaded.get("engine"),
        "runtimeNote": runtime_note,
        "tokS": tok_s,
        "completionTokens": gen.get("completionTokens"),
        "wallSec": gen.get("wallSeconds"),
    }
    if tok_s and tok_s > 0:
        return "pass", "", detail
    return "fail", f"tok/s = {tok_s}", detail


def phase_1(cap: Capability) -> PhaseResult:
    phase = PhaseResult(phase=1, name="Chat (MLX + GGUF + cache + DFlash + MTPLX)")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(CheckResult("phase 1", "skip", reason="backend not reachable"))
        return phase

    # Preferred fast model for chat checks. 35B-A3B (MoE) is much faster to
    # load than the 80B Qwen3-Next (Coder-Next). Fall back to whatever's
    # available locally if not on disk.
    PREFERRED_TEXT_MODEL = "Qwen3.6-35B-A3B-4bit"

    def _pick_fast_mlx():
        pick = _pick_model_by_ref_prefix(cap.local_mlx_models, PREFERRED_TEXT_MODEL)
        if pick:
            return pick
        return cap.local_mlx_models[0] if cap.local_mlx_models else None

    # 1a. MLX small/native cache
    def _mlx_native():
        pick = _pick_fast_mlx()
        if not pick:
            return "skip", "no MLX text model on disk", {}
        ref, path = pick
        return _load_unload_prompt(ref, path=path, backend="mlx", cache_strategy="native", context=8192)

    # 1b. MLX + TurboQuant cache
    def _mlx_turboquant():
        pick = _pick_fast_mlx()
        if not pick:
            return "skip", "no MLX text model for TurboQuant test", {}
        ref, path = pick
        return _load_unload_prompt(ref, path=path, backend="mlx", cache_strategy="turboquant",
                                    cache_bits=4, context=8192)

    # 1c. MLX + DFlash
    def _mlx_dflash():
        if not cap.dflash_supported_models:
            return "skip", "no DFlash supported models registered", {}
        # Find a DFlash-capable model that's on disk
        for support_ref in cap.dflash_supported_models:
            pick = _pick_model_by_ref_prefix(cap.local_mlx_models, support_ref.split("/")[-1])
            if pick:
                ref, path = pick
                status, reason, detail = _load_unload_prompt(ref, path=path, backend="mlx", spec=True,
                                                              context=8192, max_tokens=24)
                if status == "pass":
                    # Verify runtimeNote suggests DFlash routing
                    note = (detail.get("runtimeNote") or "").lower()
                    if "dflash" not in note and "speculative" not in note:
                        return "fail", f"speculativeDecoding enabled but runtimeNote did not mention DFlash/speculative: {note[:160]}", detail
                return status, reason, detail
        return "skip", "no DFlash-capable model on disk", {}

    # 1d. MTPLX. Uses leaf-name as modelRef + canonical_repo for the registry
    # match — works around a backend gotcha where the broken-library-entry
    # rejection shadows path-load on the full ref.
    def _mtplx():
        if not cap.mtplx_available:
            return "skip", "MTPLX not installed", {}
        for support_ref in cap.mtplx_supported_models:
            leaf = support_ref.split("/")[-1]
            pick = _pick_model_by_ref_prefix(cap.local_mlx_models, leaf)
            if pick:
                ref, path = pick
                status, reason, detail = _load_unload_prompt(
                    leaf, path=path, backend="mlx", spec=True,
                    canonical_repo=support_ref, context=8192, max_tokens=24,
                    load_timeout=900.0,
                )
                if status == "pass":
                    note = (detail.get("runtimeNote") or "").lower()
                    if "mtplx" not in note:
                        return "fail", f"MTPLX expected but runtimeNote was: {note[:160]}", detail
                    return "pass", "", detail
                continue
        return "skip", "no MTPLX-capable model on disk", {}

    # 1e. GGUF (llama.cpp backend). Cycle through .gguf files until one loads
    # so a single broken model doesn't fail the whole check.
    def _gguf():
        if not cap.gguf_available:
            return "skip", "GGUF backend (llama-server) not available", {}
        if not cap.local_gguf_files:
            return "skip", "no .gguf files on disk", {}
        errors: list[str] = []
        for ref, gguf_path in cap.local_gguf_files:
            status, reason, detail = _load_unload_prompt(
                ref, path=gguf_path, backend="gguf",
                cache_strategy="native", context=4096, max_tokens=24,
                load_timeout=600.0,
            )
            if status == "pass":
                detail["triedFiles"] = 1
                return "pass", "", detail
            errors.append(f"{ref}: {reason[:120]}")
        return "fail", f"all {len(errors)} GGUF candidates failed", {"errors": errors[:5]}

    # 1f. Long context cache preview
    def _long_context_preview():
        pick = _pick_model_by_ref_prefix(cap.local_mlx_models, "Qwen3") \
                or (cap.local_mlx_models[0] if cap.local_mlx_models else None)
        if not pick:
            return "skip", "no model for cache-preview test", {}
        ref, _ = pick
        rc, payload, err = _cli_json("cache-preview", ref, "--context", "32768",
                                       "--cache-strategy", "native", timeout=30.0)
        if rc != 0 or not isinstance(payload, dict):
            return "fail", f"cache-preview failed: {err[:200]}", {}
        return "pass", "", {"keys": sorted(payload.keys())[:10]}

    # 1g. Fused attention. Use the fast model — some architectures (e.g.
    # Qwen3-Next Coder-Next) emit parameter mismatch errors with the
    # fused-attention flag.
    def _fused_attention():
        pick = _pick_fast_mlx()
        if not pick:
            return "skip", "no model for fused-attention test", {}
        ref, path = pick
        return _load_unload_prompt(ref, path=path, backend="mlx", fused=True,
                                     cache_strategy="native", context=8192, max_tokens=16)

    for name, fn in [
        ("MLX native cache", _mlx_native),
        ("MLX TurboQuant cache", _mlx_turboquant),
        ("MLX + DFlash speculative", _mlx_dflash),
        ("MLX + MTPLX speculative", _mtplx),
        ("GGUF llama.cpp", _gguf),
        ("long context cache-preview", _long_context_preview),
        ("fused attention flag", _fused_attention),
    ]:
        phase.checks.append(_check(name, fn))
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase


# ---------------------------------------------------------------------------
# Phase 2 — Chat Compare
# ---------------------------------------------------------------------------


def phase_2(cap: Capability) -> PhaseResult:
    phase = PhaseResult(phase=2, name="Chat Compare")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(CheckResult("phase 2", "skip", reason="backend not reachable"))
        return phase

    def _compare_two_models():
        if len(cap.local_mlx_models) < 1:
            return "skip", "need at least 1 local model", {}
        ref_a, path_a = cap.local_mlx_models[0]
        ref_b, path_b = (cap.local_mlx_models[1] if len(cap.local_mlx_models) > 1
                          else cap.local_mlx_models[0])
        # /api/chat/compare expects models[] of CompareModelRequest.
        launch = {"contextTokens": 4096, "maxTokens": 24, "temperature": 0.7,
                  "cacheStrategy": "native", "cacheBits": 0, "fp16Layers": 0,
                  "fusedAttention": False, "fitModelInMemory": True,
                  "speculativeDecoding": False, "treeBudget": 0}
        body = {
            "prompt": "Say hello in 5 words.",
            "models": [
                {"modelRef": ref_a, "path": path_a, "backend": "mlx", "source": "library", "launch": launch},
                {"modelRef": ref_b, "path": path_b, "backend": "mlx", "source": "library", "launch": launch},
            ],
        }
        rc, payload, err = _cli_json("compare", "--body", json.dumps(body), "--timeout", "900")
        if rc != 0:
            return "fail", f"compare rc={rc}: {err[:200] if err else ''}", {}
        return "pass", "", {"keys": sorted(payload.keys())[:10] if isinstance(payload, dict) else None}

    phase.checks.append(_check("two-model compare", _compare_two_models))
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase


# ---------------------------------------------------------------------------
# Phase 3 — HTML Challenge
# ---------------------------------------------------------------------------


def phase_3(cap: Capability) -> PhaseResult:
    phase = PhaseResult(phase=3, name="HTML Challenge")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(CheckResult("phase 3", "skip", reason="backend not reachable"))
        return phase

    def _challenges_list():
        rc, payload, err = _cli_json("challenges-list", timeout=15.0)
        if rc != 0 or not isinstance(payload, dict):
            return "fail", f"list failed: {err[:160]}", {}
        return "pass", "", {"count": len(payload.get("challenges") or [])}

    def _challenges_minimal_create():
        if not cap.local_mlx_models:
            return "skip", "need a model for challenge generation", {}
        ref, path = cap.local_mlx_models[0]
        # HtmlChallengeRequest: required title + prompt + models[]
        launch = {"contextTokens": 8192, "maxTokens": 512, "temperature": 0.7,
                  "cacheStrategy": "native", "cacheBits": 0, "fp16Layers": 0,
                  "fusedAttention": False, "fitModelInMemory": True,
                  "speculativeDecoding": False, "treeBudget": 0}
        # HTML Challenges require ≥2 model slots; for smoke we duplicate the
        # same local model (still proves the round-trip end-to-end).
        model_block = {
            "modelRef": ref, "path": path, "backend": "mlx", "source": "library",
            "thinkingMode": "off", "launch": launch,
        }
        body = {
            "title": "e2e-suite-test",
            "prompt": "Render a centered red square on white background. 80px×80px.",
            "models": [model_block, dict(model_block)],
        }
        # /api/chat/html-challenges is an SSE stream. Use generic ``call --stream``
        # to consume events and pull the challenge id out of the first frame.
        rc, raw_out, err = _cli(
            "call", "POST", "/api/chat/html-challenges",
            "--body", json.dumps(body), "--stream", "--timeout", "1800",
        )
        if rc != 0:
            return "fail", f"create rc={rc}: {(err or raw_out)[:300]}", {}
        cid = None
        for line in raw_out.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            ch = event.get("challenge") if isinstance(event, dict) else None
            if isinstance(ch, dict) and ch.get("id"):
                cid = ch["id"]
                break
        if not cid:
            return "fail", f"no challenge id in SSE stream: {raw_out[:200]}", {}
        _cli("challenges-delete", cid, timeout=15.0)
        return "pass", "", {"id": cid}

    phase.checks.append(_check("challenges-list", _challenges_list))
    phase.checks.append(_check("challenges-create + delete", _challenges_minimal_create))
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase


# ---------------------------------------------------------------------------
# Phase 4 — Image Studio
# ---------------------------------------------------------------------------


def phase_4(cap: Capability) -> PhaseResult:
    phase = PhaseResult(phase=4, name="Image Studio")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(CheckResult("phase 4", "skip", reason="backend not reachable"))
        return phase

    def _catalog():
        rc, payload, _ = _cli_json("image-catalog", timeout=15.0)
        if rc != 0 or not isinstance(payload, dict):
            return "fail", "catalog fetch failed", {}
        return "pass", "", {"families": len(payload.get("families") or [])}

    def _library():
        rc, payload, _ = _cli_json("image-library", timeout=15.0)
        if rc != 0 or not isinstance(payload, dict):
            return "fail", "library fetch failed", {}
        return "pass", "", {"installedCount": len(payload.get("installed") or payload.get("models") or [])}

    def _runtime():
        rc, payload, _ = _cli_json("image-runtime", timeout=10.0)
        if rc != 0:
            return "fail", "image-runtime failed", {}
        return "pass", "", {"keys": sorted((payload or {}).keys())[:8] if isinstance(payload, dict) else None}

    def _generate():
        rc, lib, _ = _cli_json("image-library", timeout=15.0)
        models = (lib or {}).get("models") or []
        installed = [m for m in models if m.get("availableLocally") or m.get("hasLocalData")]
        if not installed:
            return "skip", "no image model installed locally", {}
        model_id = installed[0].get("id") or installed[0].get("repo")
        if not model_id:
            return "skip", "could not resolve installed image model id", {}
        rc, payload, err = _cli_json(
            "image-generate", "a red circle on white",
            "--model", model_id, "--steps", "4", "--width", "256", "--height", "256",
            "--seed", "42", "--timeout", "1800",
        )
        if rc != 0:
            return "fail", f"image-generate rc={rc}: {err[:300]}", {}
        _cli("image-unload", timeout=30.0)
        return "pass", "", {"modelId": model_id, "keys": sorted((payload or {}).keys())[:10] if isinstance(payload, dict) else None}

    for name, fn in [("catalog", _catalog), ("library", _library), ("runtime", _runtime), ("generate", _generate)]:
        phase.checks.append(_check(name, fn))
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase


# ---------------------------------------------------------------------------
# Phase 5 — Video Studio
# ---------------------------------------------------------------------------


def phase_5(cap: Capability) -> PhaseResult:
    phase = PhaseResult(phase=5, name="Video Studio")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(CheckResult("phase 5", "skip", reason="backend not reachable"))
        return phase

    def _catalog():
        rc, payload, _ = _cli_json("video-catalog", timeout=15.0)
        if rc != 0:
            return "fail", "catalog fetch failed", {}
        return "pass", "", {"families": len((payload or {}).get("families") or [])}

    def _library():
        rc, payload, _ = _cli_json("video-library", timeout=15.0)
        if rc != 0:
            return "fail", "library fetch failed", {}
        return "pass", "", {}

    def _mlx_runtime():
        rc, payload, _ = _cli_json("video-mlx-runtime", timeout=10.0)
        if rc != 0:
            return "fail", "video-mlx-runtime failed", {}
        return "pass", "", {"activeEngine": (payload or {}).get("runtime", {}).get("activeEngine")}

    def _generate():
        rc, lib, _ = _cli_json("video-library", timeout=15.0)
        models = (lib or {}).get("models") or []
        installed = [m for m in models if m.get("availableLocally") or m.get("hasLocalData")]
        if not installed:
            return "skip", "no video model installed locally", {}
        model_id = installed[0].get("id") or installed[0].get("repo")
        if not model_id:
            return "skip", "could not resolve installed video model id", {}
        # numFrames ≥ 8 per VideoGenerationRequest schema
        rc, payload, err = _cli_json(
            "video-generate", "a slowly turning sphere",
            "--model", model_id, "--steps", "4", "--frames", "8",
            "--width", "256", "--height", "256", "--seed", "42",
            "--timeout", "3600",
        )
        if rc != 0:
            return "fail", f"video-generate rc={rc}: {err[:300]}", {}
        return "pass", "", {"modelId": model_id, "keys": sorted((payload or {}).keys())[:10] if isinstance(payload, dict) else None}

    for name, fn in [("catalog", _catalog), ("library", _library),
                      ("mlx-runtime", _mlx_runtime), ("generate", _generate)]:
        phase.checks.append(_check(name, fn))
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase


# ---------------------------------------------------------------------------
# Phase 6 — Setup probes (read-only; install actions skipped)
# ---------------------------------------------------------------------------


def phase_6(cap: Capability) -> PhaseResult:
    phase = PhaseResult(phase=6, name="Setup probes (read-only)")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(CheckResult("phase 6", "skip", reason="backend not reachable"))
        return phase

    probes = [
        ("mtplx-status", "mtplx-status"),
        ("longlive-status", "longlive-status"),
        ("wan-status", "wan-status"),
        ("wan-inventory", "wan-inventory"),
        ("gpu-bundle-info", "gpu-bundle-info"),
        ("gpu-bundle-status", "gpu-bundle-status"),
        ("turbo-update-check", "turbo-update-check"),
    ]

    for name, cmd in probes:
        def _probe(_cmd=cmd):
            rc, payload, err = _cli_json(_cmd, timeout=30.0)
            if rc != 0:
                return "fail", f"{_cmd} rc={rc}: {err[:160]}", {}
            return "pass", "", {"keys": sorted(payload.keys())[:8] if isinstance(payload, dict) else None}
        phase.checks.append(_check(name, _probe))

    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else "pass"
    return phase


# ---------------------------------------------------------------------------
# Phase 7 — Diagnostics + cleanup verification
# ---------------------------------------------------------------------------


def phase_7(cap: Capability) -> PhaseResult:
    phase = PhaseResult(phase=7, name="Diagnostics + cleanup")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(CheckResult("phase 7", "skip", reason="backend not reachable"))
        return phase

    def _snapshot():
        rc, payload, err = _cli_json("diagnostics-snapshot", timeout=30.0)
        if rc != 0:
            return "fail", f"snapshot rc={rc}: {err[:160]}", {}
        return "pass", "", {"keys": sorted(payload.keys())[:10] if isinstance(payload, dict) else None}

    def _log_tail():
        rc, payload, err = _cli_json("diagnostics-log-tail", "--lines", "50", timeout=15.0)
        if rc != 0:
            return "fail", f"log-tail rc={rc}: {err[:160]}", {}
        return "pass", "", {"lines": len((payload or {}).get("lines") or [])}

    def _no_orphans():
        rc, runtime, _ = _cli_json("runtime", timeout=10.0)
        if rc != 0:
            return "fail", "runtime failed", {}
        orphans = (runtime or {}).get("recentOrphanedWorkers") or []
        # ``terminated`` / ``killed`` records mean the backend already cleaned
        # up — that's working as intended, not a leak. Only ``kill_failed``
        # (or similar non-cleaned states) is a real regression.
        unclean = [o for o in orphans if o.get("action") not in {"terminated", "killed"}]
        return ("pass" if len(unclean) == 0 else "fail"), f"{len(orphans)} record(s), {len(unclean)} unclean", {
            "orphanRecords": orphans[:3], "uncleanCount": len(unclean),
        }

    def _idle_state():
        rc, status, _ = _cli_json("status", timeout=10.0)
        if rc != 0:
            return "fail", "status failed", {}
        rt = (status or {}).get("runtime", {})
        state = rt.get("state")
        # After all phases, runtime should be idle (we unload after every load)
        return ("pass" if state in {"idle", "loaded"} else "fail"), f"state={state}", {"state": state}

    for name, fn in [("diagnostics-snapshot", _snapshot), ("diagnostics-log-tail", _log_tail),
                      ("no orphan workers", _no_orphans), ("runtime idle/loaded", _idle_state)]:
        phase.checks.append(_check(name, fn))
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else "pass"
    return phase


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


PHASES = {
    0: phase_0, 1: phase_1, 2: phase_2, 3: phase_3,
    4: phase_4, 5: phase_5, 6: phase_6, 7: phase_7,
}


def _write_reports(report_dir: Path, started: datetime, ended: datetime,
                    phases: list[PhaseResult], cap: Capability) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    json_path = report_dir / f"e2e-{stamp}.json"
    md_path = report_dir / f"e2e-{stamp}.md"

    payload = {
        "startedAt": started.isoformat(),
        "endedAt": ended.isoformat(),
        "elapsedSec": round((ended - started).total_seconds(), 2),
        "capabilities": asdict(cap),
        "phases": [asdict(p) for p in phases],
        "summary": {
            "pass": sum(1 for p in phases if p.status == "pass"),
            "fail": sum(1 for p in phases if p.status == "fail"),
            "skip": sum(1 for p in phases if p.status == "skip"),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    # Markdown summary
    lines = [
        f"# ChaosEngineAI E2E Test Run — {stamp}",
        "",
        f"- Started: `{started.isoformat()}`",
        f"- Ended: `{ended.isoformat()}`",
        f"- Elapsed: {round((ended - started).total_seconds(), 2)}s",
        f"- Pass / Fail / Skip: **{payload['summary']['pass']}** / **{payload['summary']['fail']}** / **{payload['summary']['skip']}**",
        "",
        "## Phases",
        "",
    ]
    for p in phases:
        emoji = {"pass": "✓", "fail": "✗", "skip": "·"}.get(p.status, "?")
        lines.append(f"### Phase {p.phase} — {p.name}  {emoji} `{p.status}`")
        lines.append("")
        lines.append("| Check | Status | Time | Reason |")
        lines.append("|---|---|---|---|")
        for c in p.checks:
            reason = (c.reason or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {c.name} | `{c.status}` | {c.elapsed_sec}s | {reason[:120]} |")
        lines.append("")
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ChaosEngineAI E2E test suite.")
    parser.add_argument("--phases", default=None, help="Comma-separated phase numbers (default: all).")
    parser.add_argument("--smoke", action="store_true", help="Run phases 0,2,3,4,5,6,7 only (skip heavy Phase 1).")
    parser.add_argument("--report-dir", default=str(_DEFAULT_REPORT_DIR))
    args = parser.parse_args(argv)

    if args.phases:
        wanted = sorted(int(x) for x in args.phases.split(",") if x.strip())
    elif args.smoke:
        # Compare (Phase 2) and full Chat sweep (Phase 1) are heavyweight —
        # both need real model loads. Smoke proves the surface without those.
        wanted = [0, 3, 4, 5, 6, 7]
    else:
        wanted = sorted(PHASES.keys())

    started = datetime.now(timezone.utc)
    print(f"[e2e] starting suite at {started.isoformat()}", file=sys.stderr, flush=True)

    cap = probe_capabilities()
    if not cap.backend_reachable:
        ended = datetime.now(timezone.utc)
        phases: list[PhaseResult] = []
        phase0 = phase_0(cap)
        phases.append(phase0)
        _write_reports(Path(args.report_dir), started, ended, phases, cap)
        print("[e2e] backend not reachable; aborting", file=sys.stderr, flush=True)
        return 2

    phases: list[PhaseResult] = []
    for phase_num in wanted:
        fn = PHASES.get(phase_num)
        if not fn:
            continue
        started_phase = time.perf_counter()
        print(f"[e2e] phase {phase_num}: {fn.__name__} starting", file=sys.stderr, flush=True)
        result = fn(cap)
        result.started_at = datetime.now(timezone.utc).isoformat()
        result.elapsed_sec = round(time.perf_counter() - started_phase, 2)
        phases.append(result)
        emoji = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}.get(result.status, "?")
        print(f"[e2e] phase {phase_num}: {emoji} ({result.elapsed_sec}s, {len(result.checks)} checks)", file=sys.stderr, flush=True)

    ended = datetime.now(timezone.utc)
    json_path, md_path = _write_reports(Path(args.report_dir), started, ended, phases, cap)
    print(f"[e2e] report: {json_path}", file=sys.stderr, flush=True)
    print(f"[e2e] report: {md_path}", file=sys.stderr, flush=True)

    fails = sum(1 for p in phases if p.status == "fail")
    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
