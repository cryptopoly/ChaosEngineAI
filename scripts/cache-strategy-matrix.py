#!/usr/bin/env python3
"""ChaosEngineAI cache-strategy + speculative-decoding matrix runner.

Sweeps the supported (strategy × spec-dec × model) grid through a running
backend and writes a CSV + Markdown summary to ``~/.chaosengine/test-results/``.

Skips cells where:
- the strategy is not available on this platform (per ``/api/cache/strategies``)
- the spec-dec method is not supported for the given backend (DFlash/DDTree
  require MLX or vLLM, not GGUF)
- the model is not in the local library

Verifies the **FU-030 legacy alias coercion** by including
``cacheStrategy=chaosengine`` and ``cacheStrategy=rotorquant`` rows; the
backend must coerce both to ``turboquant`` and the runtime note + load
report must reflect that.

Usage:
    .venv/bin/python scripts/cache-strategy-matrix.py [--port 8876]
                                                      [--quick]
                                                      [--out PATH]

``--quick`` drops the larger models so the matrix completes in ~5 minutes
(useful for smoke runs in CI). The full run takes ~20 minutes wall-time on
M-series Macs.

Backend prerequisite: the FastAPI sidecar must be running on the chosen
port (default 8876). The script does not start it for you.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PORT = 8876
DEFAULT_OUT_DIR = Path.home() / ".chaosengine" / "test-results"
DEFAULT_PROMPT = "Explain in three sentences why deterministic seeding matters."
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0  # deterministic — required for output-hash compares
DEFAULT_SEED = 42

# ── Matrix definition ────────────────────────────────────────────────

@dataclass(frozen=True)
class MatrixCell:
    """One scheduled inference run."""

    label: str
    model_ref: str
    backend: str          # ``mlx`` | ``gguf`` | ``vllm``
    strategy: str         # ``native`` | ``turboquant`` | ``triattention`` | legacy aliases
    bits: int             # 0 for native, otherwise per-strategy bit count
    spec_decode: str      # ``none`` | ``dflash`` | ``ddtree`` | ``mtplx`` | ``gguf-mtp``
    tree_budget: int = 0  # only meaningful when spec_decode == ``ddtree``
    quick: bool = True    # included in the ``--quick`` smoke set


# Smallest-on-disk MLX target so the matrix exercises every code path
# without burning hours of wall-time. Heavier sweeps (35B-A3B etc.) flip
# ``quick=False`` and are gated by the absence of the ``--quick`` flag.
#
# Smoke targets are current-gen Qwen3 (replacing Qwen2.5 in 2026-05) so the
# plumbing layer exercises a model architecture that's still under active
# upstream development. Feature-quality cells (DFlash, MTPLX, MTP-GGUF)
# target the smallest *capability-supported* model rather than the smallest
# on disk — see ``DRAFT_MODEL_MAP`` (dflash/__init__.py) and ``MTP_MODEL_MAP``
# (backend_service/inference/_mtp.py) for the gating tables.
SMALL_MLX = "mlx-community/Qwen3-0.6B-4bit"
MID_MLX_DFLASH_CAPABLE = "mlx-community/Qwen3-4B-bf16"
# FU-073: was ``mlx-community/Qwen3.5-4B-bf16`` — a VL conversion that
# carries no MTP heads and isn't in ``MTP_MODEL_MAP`` / ``_MTP_ALIASES``,
# so the MTPLX cell could never actually exercise MTP. The canonical
# ``Qwen/Qwen3.5-4B`` is a direct ``MTP_MODEL_MAP`` key (``mtp.*`` tensors
# present in its safetensors index) and a catalog variant, so MTPLX
# resolves heads and the cell can run once the repo is on disk.
MID_MLX_MTPLX_CAPABLE = "Qwen/Qwen3.5-4B"
SMALL_GGUF = "lmstudio-community/Qwen3-0.6B-GGUF"
LARGE_GGUF_MTP = "ggml-org/Qwen3.6-27B-MTP-GGUF"

# vLLM targets are raw HF safetensors repos (vLLM doesn't consume MLX or GGUF
# weights). Capability-gated to ``vllmAvailable`` so these all skip cleanly on
# macOS. All marked ``quick=False`` — the CUDA box runs the full sweep.
VLLM_SMALL = "Qwen/Qwen3-0.6B"
VLLM_MID = "Qwen/Qwen3.5-4B"

MATRIX: list[MatrixCell] = [
    # MLX × strategies — every text strategy on the smallest model
    MatrixCell("native MLX (smoke)",            SMALL_MLX, "mlx",  "native",       0, "none"),
    MatrixCell("turboquant MLX 3-bit",          SMALL_MLX, "mlx",  "turboquant",   3, "none"),
    MatrixCell("triattention MLX",              SMALL_MLX, "mlx",  "triattention", 3, "none"),

    # FU-030 legacy alias coercion — both must run as turboquant + report it
    MatrixCell("legacy id chaosengine -> turboquant", SMALL_MLX, "mlx", "chaosengine", 4, "none"),
    MatrixCell("legacy id rotorquant  -> turboquant", SMALL_MLX, "mlx", "rotorquant",  3, "none"),

    # Speculative decoding — DFlash + DDTree require MLX backend + a
    # DRAFT_MODEL_MAP-supported target. The 4B Qwen3 path covers both.
    MatrixCell("dflash spec-dec (Qwen3-4B)", MID_MLX_DFLASH_CAPABLE, "mlx", "native", 0, "dflash", quick=False),
    MatrixCell("ddtree spec-dec budget=16",  MID_MLX_DFLASH_CAPABLE, "mlx", "native", 0, "ddtree", tree_budget=16, quick=False),

    # MTPLX speculative decoding — MTP_MODEL_MAP-supported target via the
    # standalone MTPLX subprocess engine. Qwen3.5-4B is the smallest entry.
    MatrixCell("mtplx spec-dec (Qwen3.5-4B)", MID_MLX_MTPLX_CAPABLE, "mlx", "native", 0, "mtplx", quick=False),

    # GGUF lane — native is enough to verify the standard binary path.
    # TurboQuant on GGUF needs llama-server-turbo; runner skips when the
    # binary is missing rather than hard-failing.
    MatrixCell("native GGUF (smoke)",     SMALL_GGUF, "gguf", "native",     0, "none"),
    MatrixCell("turboquant GGUF 3-bit",   SMALL_GGUF, "gguf", "turboquant", 3, "none", quick=False),

    # GGUF MTP speculative decoding (FU-047) — llama.cpp ``--spec-type
    # draft-mtp`` against a model with baked-in MTP heads. Requires
    # llama-server master ≥ 2026-05-16. 29 GB Q8_0 target so quick=False.
    MatrixCell("gguf MTP (Qwen3.6-27B)", LARGE_GGUF_MTP, "gguf", "native", 0, "gguf-mtp", quick=False),

    # vLLM lane — CUDA-only. All cells capability-gated; macOS skips cleanly.
    # Raw HF safetensors targets; vLLM doesn't consume MLX or GGUF weights.
    MatrixCell("vllm native (Qwen3-0.6B)",       VLLM_SMALL, "vllm", "native",       0, "none",   quick=False),
    MatrixCell("vllm turboquant (Qwen3-0.6B)",   VLLM_SMALL, "vllm", "turboquant",   3, "none",   quick=False),
    MatrixCell("vllm triattention (Qwen3-0.6B)", VLLM_SMALL, "vllm", "triattention", 3, "none",   quick=False),
    MatrixCell("vllm dflash (Qwen3.5-4B)",       VLLM_MID,   "vllm", "native",       0, "dflash", quick=False),
]


# ── HTTP helpers ─────────────────────────────────────────────────────

def _api(method: str, path: str, *, port: int, body: dict | None = None, timeout: float = 60) -> dict:
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode()
        except Exception:
            pass
        raise RuntimeError(f"API {method} {path} -> {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Cannot reach ChaosEngineAI at port {port}. Is the backend running? ({exc.reason})"
        ) from exc


def _stream_inference(path: str, *, port: int, body: dict, timeout: float = 300) -> tuple[str, dict]:
    """POST to an SSE endpoint. Returns ``(full_text, done_payload)``."""
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")

    full_text = ""
    done_payload: dict = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buffer = ""
        while True:
            raw = resp.read(4096)
            if not raw:
                break
            buffer += raw.decode("utf-8", errors="replace")
            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                for line in event_str.strip().splitlines():
                    if not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:])
                    if "token" in payload:
                        full_text += payload["token"]
                    if "reasoning" in payload:
                        # Reasoning models (Qwen3, DeepSeek-R1) emit the
                        # ``<think>...</think>`` block on a separate channel;
                        # roll it into ``full_text`` so output-hash compares
                        # cover both reasoning + answer tokens.
                        full_text += payload["reasoning"]
                    if "error" in payload:
                        raise RuntimeError(f"Inference error: {payload['error']}")
                    if payload.get("done"):
                        done_payload = payload
    return full_text, done_payload


# ── Capability probes (decide which cells to skip) ───────────────────

@dataclass
class BackendCapabilities:
    available_strategies: set[str]
    dflash_available: bool
    ddtree_available: bool
    mtplx_available: bool
    gguf_mtp_available: bool
    vllm_available: bool
    has_turbo_binary: bool
    library_refs: set[str]
    # FU-056 Phase 9: vLLM-via-WSL bridge availability. On Windows boxes
    # native vLLM never works (no Windows wheels), but the WSL bridge
    # gives the same engine class through a subprocess. The matrix
    # runner treats either as "vllm cells can run" so a Windows + RTX
    # box isn't permanently locked out of the vLLM lane.
    wsl_vllm_available: bool = False


def probe_backend(port: int) -> BackendCapabilities:
    workspace = _api("GET", "/api/workspace", port=port)
    health = _api("GET", "/api/health", port=port)
    system = workspace.get("system", {})
    strategies = system.get("availableCacheStrategies") or []
    available = {s["id"] for s in strategies if s.get("available")}
    dflash = system.get("dflash") or {}
    native_backends = health.get("nativeBackends") or {}
    library = workspace.get("library") or []
    refs: set[str] = set()
    for item in library:
        name = item.get("name") or ""
        if name:
            refs.add(name)
        for variant in item.get("variants", []) or []:
            repo = variant.get("repo") or ""
            if repo:
                refs.add(repo)
    return BackendCapabilities(
        available_strategies=available,
        dflash_available=bool(dflash.get("available")),
        ddtree_available=bool(dflash.get("ddtreeAvailable")),
        mtplx_available=bool(native_backends.get("mtplxAvailable")),
        gguf_mtp_available=bool(native_backends.get("ggufMtpAvailable")),
        # ``vllmAvailable`` (native) OR ``wslVllmAvailable`` (Windows
        # bridge) — either route can serve the vllm cells. The runner
        # doesn't care which path the backend chose; it cares whether
        # a vllm load will succeed at all.
        vllm_available=(
            bool(native_backends.get("vllmAvailable"))
            or bool(native_backends.get("wslVllmAvailable"))
        ),
        wsl_vllm_available=bool(native_backends.get("wslVllmAvailable")),
        has_turbo_binary=bool(system.get("llamaServerTurboPath")),
        library_refs=refs,
    )


def skip_reason(cell: MatrixCell, caps: BackendCapabilities, *, quick: bool) -> str | None:
    if quick and not cell.quick:
        return "deferred to full run (drop --quick)"

    if cell.backend == "vllm" and not caps.vllm_available:
        # ``vllm_available`` already considers the WSL bridge (FU-056
        # Phase 8) — if neither route serves vLLM, the skip reason
        # depends on the platform so the user gets the right next step.
        # The runner doesn't know the OS, so name both paths.
        return "vLLM not available (install via Diagnostics → WSL2 vLLM bridge on Windows, or pip install vllm on Linux+CUDA)"

    canonical = {"chaosengine": "turboquant", "rotorquant": "turboquant"}.get(
        cell.strategy, cell.strategy,
    )
    if canonical not in caps.available_strategies and canonical != "native":
        return f"strategy '{canonical}' unavailable in this runtime"

    if cell.backend == "gguf" and canonical == "turboquant" and not caps.has_turbo_binary:
        return "llama-server-turbo binary missing"

    if cell.spec_decode in ("dflash", "ddtree"):
        if cell.backend == "gguf":
            return "speculative decoding requires MLX/vLLM, not GGUF"
        if not caps.dflash_available:
            return "DFlash runtime not installed"
        if cell.spec_decode == "ddtree" and not caps.ddtree_available:
            return "DDTree runtime not available"

    if cell.spec_decode == "mtplx":
        if cell.backend != "mlx":
            return "MTPLX speculative decoding requires MLX backend"
        if not caps.mtplx_available:
            return "MTPLX runtime not installed"

    if cell.spec_decode == "gguf-mtp":
        if cell.backend != "gguf":
            return "GGUF MTP requires GGUF backend"
        if not caps.gguf_mtp_available:
            return "llama-server lacks --spec-type draft-mtp (FU-047) — upgrade llama.cpp"

    if cell.model_ref not in caps.library_refs:
        return f"model not in library ({cell.model_ref})"

    return None


# ── Cell execution ───────────────────────────────────────────────────

# Substrings the backend uses when a model's weights aren't actually on
# disk. ``library_refs`` is built from the *catalog* (every variant repo),
# so a catalogued-but-undownloaded model (or an interrupted pull that left
# an empty ``refs/main``-only HF cache dir) passes the ``skip_reason``
# library check and only fails at load time. That's a missing download, not
# a product failure — same false-positive class as FU-053 — so we classify
# it as a skip rather than a fail.
_WEIGHTS_MISSING_MARKERS = (
    "weights found in HF cache entry",
    "No .gguf, .safetensors, or pytorch weights",
)


def classify_load_skip(error_message: str) -> str | None:
    """Return a skip reason if a load error means the weights aren't on
    disk, else None (a genuine load failure to surface)."""
    for marker in _WEIGHTS_MISSING_MARKERS:
        if marker in error_message:
            return "weights not downloaded"
    return None


@dataclass
class CellResult:
    label: str
    model_ref: str
    backend: str
    strategy: str
    bits: int
    spec_decode: str
    tree_budget: int
    skipped: bool = False
    skip_reason: str = ""
    ok: bool = False
    error: str = ""
    tokens_per_sec: float = 0.0
    dflash_acceptance: float | None = None
    output_sha: str = ""
    output_chars: int = 0
    actual_strategy: str = ""
    runtime_note: str = ""
    duration_seconds: float = 0.0


def run_cell(cell: MatrixCell, *, port: int) -> CellResult:
    result = CellResult(
        label=cell.label,
        model_ref=cell.model_ref,
        backend=cell.backend,
        strategy=cell.strategy,
        bits=cell.bits,
        spec_decode=cell.spec_decode,
        tree_budget=cell.tree_budget,
    )

    body = {
        "modelRef": cell.model_ref,
        "modelName": cell.model_ref.split("/")[-1],
        "canonicalRepo": cell.model_ref,
        "source": "library",
        "backend": cell.backend,
        "cacheStrategy": cell.strategy,
        "cacheBits": cell.bits,
        "fp16Layers": 0,
        "fusedAttention": False,
        "fitModelInMemory": True,
        "contextTokens": 4096,
        "speculativeDecoding": cell.spec_decode != "none",
        "treeBudget": cell.tree_budget,
        "thinkingMode": "off",
    }

    started = time.monotonic()
    try:
        try:
            load_resp = _api("POST", "/api/models/load", port=port, body=body, timeout=180)
        except (RuntimeError, ConnectionError, urllib.error.URLError) as load_exc:
            skip = classify_load_skip(str(load_exc))
            if skip is None:
                raise
            result.skipped = True
            result.skip_reason = f"{skip} ({cell.model_ref})"
            result.duration_seconds = round(time.monotonic() - started, 2)
            return result
        loaded = ((load_resp.get("runtime") or {}).get("loadedModel")) or load_resp.get("loadedModel") or {}
        result.actual_strategy = loaded.get("cacheStrategy", "")
        result.runtime_note = loaded.get("runtimeNote") or ""

        gen_body = {
            "prompt": DEFAULT_PROMPT,
            "maxTokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
            "seed": DEFAULT_SEED,
            "thinkingMode": "off",
        }
        text, done = _stream_inference("/api/chat/generate/stream", port=port, body=gen_body, timeout=240)
        result.duration_seconds = round(time.monotonic() - started, 2)
        # tok/s lives in the streamed done event under
        # ``assistant.metrics.tokS`` (see state/metrics.py
        # stream_assistant_metrics_payload), not a top-level
        # ``tokensPerSecond`` field — reading the wrong key reported
        # 0.0 tok/s for every cell. ``dflashAcceptanceRate`` (when the
        # MLX spec-dec path actually engaged) also lives there.
        _metrics = (done.get("assistant") or {}).get("metrics") or {}
        result.tokens_per_sec = float(_metrics.get("tokS") or 0.0)
        result.dflash_acceptance = (
            float(_metrics["dflashAcceptanceRate"])
            if _metrics.get("dflashAcceptanceRate") is not None
            else None
        )
        result.output_chars = len(text)
        result.output_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        result.ok = bool(text.strip())
        if not result.ok:
            result.error = "empty output"
    except (RuntimeError, ConnectionError, urllib.error.URLError) as exc:
        result.error = str(exc)[:200]
        result.duration_seconds = round(time.monotonic() - started, 2)
    return result


# ── Reporting ────────────────────────────────────────────────────────

def write_csv(out_dir: Path, results: list[CellResult]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    csv_path = out_dir / f"cache-strategy-matrix-{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "label", "model_ref", "backend", "strategy", "bits", "spec_decode",
            "tree_budget", "skipped", "skip_reason", "ok", "error",
            "tokens_per_sec", "duration_seconds", "actual_strategy",
            "output_sha", "output_chars", "runtime_note",
        ])
        for r in results:
            writer.writerow([
                r.label, r.model_ref, r.backend, r.strategy, r.bits, r.spec_decode,
                r.tree_budget, r.skipped, r.skip_reason, r.ok, r.error,
                f"{r.tokens_per_sec:.2f}", f"{r.duration_seconds:.2f}",
                r.actual_strategy, r.output_sha, r.output_chars, r.runtime_note,
            ])
    return csv_path


def write_markdown(out_dir: Path, results: list[CellResult]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    md_path = out_dir / f"cache-strategy-matrix-{timestamp}.md"

    ran = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]
    passed = [r for r in ran if r.ok]
    failed = [r for r in ran if not r.ok]

    lines = [
        f"# Cache strategy matrix run ({timestamp})",
        "",
        f"- Total cells: **{len(results)}**",
        f"- Ran: **{len(ran)}** ({len(passed)} pass / {len(failed)} fail)",
        f"- Skipped: **{len(skipped)}**",
        "",
        "## Results",
        "",
        "| Label | Strategy | Spec-dec | Outcome | tok/s | SHA-12 | Note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.skipped:
            outcome = f"SKIP — {r.skip_reason}"
        elif r.ok:
            outcome = "PASS"
        else:
            outcome = f"FAIL — {r.error}"
        lines.append(
            f"| {r.label} | {r.strategy}({r.bits}b) | {r.spec_decode} | {outcome} | "
            f"{r.tokens_per_sec:.1f} | {r.output_sha or '—'} | {r.runtime_note[:80]} |"
        )

    # FU-030 coercion section: legacy ids must report ``actual_strategy``
    # of ``turboquant`` even though the request asked for chaosengine /
    # rotorquant. Surface it explicitly so regressions are obvious.
    legacy = [r for r in ran if r.strategy in ("chaosengine", "rotorquant")]
    if legacy:
        lines += [
            "",
            "## FU-030 legacy alias coercion",
            "",
            "| Requested | Loaded | Coercion correct? |",
            "|---|---|---|",
        ]
        for r in legacy:
            ok_mark = "yes" if r.actual_strategy == "turboquant" else "**NO**"
            lines.append(f"| {r.strategy} | {r.actual_strategy or '—'} | {ok_mark} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def print_summary(results: list[CellResult]) -> int:
    ran = [r for r in results if not r.skipped]
    passed = [r for r in ran if r.ok]
    failed = [r for r in ran if not r.ok]
    skipped = [r for r in results if r.skipped]
    coercion_failures = [
        r for r in ran
        if r.strategy in ("chaosengine", "rotorquant")
        and r.actual_strategy != "turboquant"
    ]
    print()
    print(f"  Cells:    {len(results)}")
    print(f"  Ran:      {len(ran)}  ({len(passed)} pass / {len(failed)} fail)")
    print(f"  Skipped:  {len(skipped)}")
    if failed:
        print()
        print("  Failures:")
        for r in failed:
            print(f"    - {r.label}: {r.error}")
    if coercion_failures:
        print()
        print("  FU-030 coercion regression:")
        for r in coercion_failures:
            print(f"    - {r.label}: requested={r.strategy} loaded={r.actual_strategy or 'n/a'}")
        return 2
    return 0 if not failed else 1


# ── Entry point ──────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--quick", action="store_true",
                        help="run only the smoke subset (~5 min wall-time)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help="results directory (default ~/.chaosengine/test-results)")
    args = parser.parse_args()

    print(f"Probing backend at http://127.0.0.1:{args.port}/api/workspace ...")
    try:
        caps = probe_backend(args.port)
    except ConnectionError as exc:
        # The matrix runner is meant to exercise the installed app's
        # runtime, the same way ``e2e_test_suite.py`` does. A failure to
        # reach the backend almost always means "the app isn't open" —
        # surface that clearly instead of just echoing the ConnectionError.
        print(f"  ! {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "Open the ChaosEngineAI app and re-run this command — the matrix "
            "is designed to exercise the production embedded runtime + extras.",
            file=sys.stderr,
        )
        print(
            f"(advanced: `npm run tauri:dev` or `python -m backend_service.app "
            f"--port {args.port}` works for dev runs, but won't match the user-install path)",
            file=sys.stderr,
        )
        return 3
    print(f"  available strategies: {sorted(caps.available_strategies)}")
    print(f"  dflash={caps.dflash_available} ddtree={caps.ddtree_available} turbo-binary={caps.has_turbo_binary}")
    print(f"  library models: {len(caps.library_refs)}")

    results: list[CellResult] = []
    for i, cell in enumerate(MATRIX, 1):
        print(f"\n[{i}/{len(MATRIX)}] {cell.label}")
        skip = skip_reason(cell, caps, quick=args.quick)
        if skip:
            print(f"  skip: {skip}")
            results.append(CellResult(
                label=cell.label,
                model_ref=cell.model_ref,
                backend=cell.backend,
                strategy=cell.strategy,
                bits=cell.bits,
                spec_decode=cell.spec_decode,
                tree_budget=cell.tree_budget,
                skipped=True,
                skip_reason=skip,
            ))
            continue
        result = run_cell(cell, port=args.port)
        if result.ok:
            accept = (
                f"  accept={result.dflash_acceptance:.0f}%"
                if result.dflash_acceptance is not None else ""
            )
            print(f"  pass  {result.tokens_per_sec:.1f} tok/s  sha={result.output_sha}{accept}  ({result.duration_seconds:.1f}s)")
        else:
            print(f"  FAIL  {result.error}")
        results.append(result)

    csv_path = write_csv(args.out, results)
    md_path = write_markdown(args.out, results)
    print()
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {md_path}")
    return print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
