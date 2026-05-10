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
DEFAULT_MAX_TOKENS = 96
DEFAULT_TEMPERATURE = 0.0  # deterministic — required for output-hash compares
DEFAULT_SEED = 42

# ── Matrix definition ────────────────────────────────────────────────

@dataclass(frozen=True)
class MatrixCell:
    """One scheduled inference run."""

    label: str
    model_ref: str
    backend: str          # ``mlx`` | ``gguf``
    strategy: str         # ``native`` | ``turboquant`` | ``triattention`` | legacy aliases
    bits: int             # 0 for native, otherwise per-strategy bit count
    spec_decode: str      # ``none`` | ``dflash`` | ``ddtree``
    tree_budget: int = 0  # only meaningful when spec_decode == ``ddtree``
    quick: bool = True    # included in the ``--quick`` smoke set


# Smallest-on-disk MLX target so the matrix exercises every code path
# without burning hours of wall-time. Heavier sweeps (35B-A3B etc.) flip
# ``quick=False`` and are gated by the absence of the ``--quick`` flag.
SMALL_MLX = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MID_MLX_DFLASH_CAPABLE = "mlx-community/Qwen3-4B-bf16"
SMALL_GGUF = "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF"

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

    # GGUF lane — native is enough to verify the standard binary path.
    # TurboQuant on GGUF needs llama-server-turbo; runner skips when the
    # binary is missing rather than hard-failing.
    MatrixCell("native GGUF (smoke)",     SMALL_GGUF, "gguf", "native",     0, "none"),
    MatrixCell("turboquant GGUF 3-bit",   SMALL_GGUF, "gguf", "turboquant", 3, "none", quick=False),
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
    has_turbo_binary: bool
    library_refs: set[str]


def probe_backend(port: int) -> BackendCapabilities:
    workspace = _api("GET", "/api/workspace", port=port)
    system = workspace.get("system", {})
    strategies = system.get("availableCacheStrategies") or []
    available = {s["id"] for s in strategies if s.get("available")}
    dflash = system.get("dflash") or {}
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
        has_turbo_binary=bool(system.get("llamaServerTurboPath")),
        library_refs=refs,
    )


def skip_reason(cell: MatrixCell, caps: BackendCapabilities, *, quick: bool) -> str | None:
    if quick and not cell.quick:
        return "deferred to full run (drop --quick)"

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

    if cell.model_ref not in caps.library_refs:
        return f"model not in library ({cell.model_ref})"

    return None


# ── Cell execution ───────────────────────────────────────────────────

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
        load_resp = _api("POST", "/api/models/load", port=port, body=body, timeout=180)
        result.actual_strategy = (load_resp.get("loadedModel") or {}).get("cacheStrategy", "")
        result.runtime_note = (load_resp.get("loadedModel") or {}).get("runtimeNote") or ""

        gen_body = {
            "prompt": DEFAULT_PROMPT,
            "maxTokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
            "seed": DEFAULT_SEED,
            "thinkingMode": "off",
        }
        text, done = _stream_inference("/api/generate/stream", port=port, body=gen_body, timeout=240)
        result.duration_seconds = round(time.monotonic() - started, 2)
        result.tokens_per_sec = float(done.get("tokensPerSecond") or 0.0)
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
        print(f"  ! {exc}", file=sys.stderr)
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
            print(f"  pass  {result.tokens_per_sec:.1f} tok/s  sha={result.output_sha}  ({result.duration_seconds:.1f}s)")
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
