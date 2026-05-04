#!/usr/bin/env python3
"""ChaosEngineAI inference test runner.

Connects to a running ChaosEngineAI backend, discovers available models,
lets you configure runtime parameters interactively, runs real inference,
and saves full results (configuration + stats) to JSON.

Usage:
    python scripts/inference-test-runner.py [--port 8876] [--prompt "..."]

Requirements:
    The ChaosEngineAI backend must be running.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import urllib.error
import urllib.request

# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_PORT = 8876
DEFAULT_PROMPT = "Explain how KV-cache compression works in 3 sentences."
RESULTS_DIR = Path.home() / ".chaosengine" / "test-results"


# ── HTTP helpers ─────────────────────────────────────────────────────

def _api(method: str, path: str, *, port: int, body: dict | None = None, timeout: float = 120) -> dict:
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
        raise RuntimeError(f"API {method} {path} → {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Cannot reach ChaosEngineAI at port {port}. "
            f"Is the backend running? ({exc.reason})"
        ) from exc


def _stream_api(path: str, *, port: int, body: dict, timeout: float = 300) -> tuple[str, str, dict]:
    """POST to an SSE endpoint. Returns (full_text, full_reasoning, done_payload)."""
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")

    full_text = ""
    full_reasoning = ""
    done_payload: dict = {}
    token_count = 0
    reasoning_count = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # Read in chunks rather than byte-by-byte for reliability
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
                        token_count += 1
                        if token_count % 10 == 0:
                            print(".", end="", flush=True)
                    if "reasoning" in payload:
                        full_reasoning += payload["reasoning"]
                        reasoning_count += 1
                        if reasoning_count % 20 == 0:
                            print("t", end="", flush=True)
                    if payload.get("reasoningDone"):
                        print(" [thinking done] ", end="", flush=True)
                    if "error" in payload:
                        raise RuntimeError(f"Inference error: {payload['error']}")
                    if payload.get("done"):
                        done_payload = payload

    if token_count >= 10 or reasoning_count >= 20:
        print()  # Newline after progress indicators

    return full_text, full_reasoning, done_payload


# ── Interactive menus ────────────────────────────────────────────────

def _pick_one(label: str, options: list[str], *, allow_skip: bool = False) -> int | None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    for i, opt in enumerate(options, 1):
        print(f"  {i:3d}. {opt}")
    if allow_skip:
        print(f"    0. (skip)")
    while True:
        raw = input("\n  Choice: ").strip()
        if not raw:
            continue
        try:
            choice = int(raw)
        except ValueError:
            print("  Enter a number.")
            continue
        if allow_skip and choice == 0:
            return None
        if 1 <= choice <= len(options):
            return choice - 1
        print(f"  Pick 1–{len(options)}.")


def _pick_float(label: str, default: float, lo: float, hi: float) -> float:
    while True:
        raw = input(f"  {label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = float(raw)
        except ValueError:
            print(f"  Enter a number ({lo}–{hi}).")
            continue
        if lo <= val <= hi:
            return val
        print(f"  Must be {lo}–{hi}.")


def _pick_int(label: str, default: int, lo: int, hi: int) -> int:
    while True:
        raw = input(f"  {label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
        except ValueError:
            print(f"  Enter an integer ({lo}–{hi}).")
            continue
        if lo <= val <= hi:
            return val
        print(f"  Must be {lo}–{hi}.")


def _pick_bool(label: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {label} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


# ── Model discovery ──────────────────────────────────────────────────

def _format_model_line(m: dict) -> str:
    parts = [m["name"]]
    if m.get("quantization"):
        parts.append(f"[{m['quantization']}]")
    parts.append(f"({m['format']})")
    if m.get("sizeGb"):
        parts.append(f"{m['sizeGb']:.1f} GB")
    if m.get("backend"):
        parts.append(f"via {m['backend']}")
    return " ".join(parts)


def _model_ref(m: dict) -> str:
    """Derive the model ref the backend expects for loading."""
    path = m.get("path", "")
    name = m.get("name", "")
    # HF cache models have a repo-style name
    if "/" in name:
        return name
    return path or name


def discover_models(port: int) -> list[dict]:
    print("Fetching workspace from backend...")
    workspace = _api("GET", "/api/workspace", port=port)

    library = workspace.get("library", [])
    # Filter to text models that aren't broken
    models = [
        m for m in library
        if not m.get("broken")
        and m.get("modelType", "text") == "text"
    ]
    if not models:
        print("\n  No models found! Check your model directories in ChaosEngineAI settings.")
        sys.exit(1)

    return models


# ── Configuration ────────────────────────────────────────────────────

def configure_runtime(
    workspace: dict,
    selected_model: dict,
) -> dict:
    """Interactive configuration of all runtime parameters."""
    system = workspace.get("system", {})
    strategies = system.get("availableCacheStrategies", [])
    dflash = system.get("dflash", {})
    launch_prefs = workspace.get("settings", {}).get("launchPreferences", {})

    print(f"\n{'═' * 60}")
    print("  RUNTIME CONFIGURATION")
    print(f"{'═' * 60}")

    model_format = selected_model.get("format", "unknown")
    model_backend = selected_model.get("backend", "auto")

    # ── Backend selection ──
    backends = ["auto"]
    if system.get("mlxUsable"):
        backends.append("mlx")
    if system.get("ggufAvailable"):
        backends.append("llama.cpp")
    if system.get("vllmAvailable"):
        backends.append("vllm")

    if len(backends) > 1:
        idx = _pick_one("Inference backend", backends)
        backend = backends[idx]
        # Normalize backend name for API
        if backend == "llama.cpp":
            backend = "gguf"
    else:
        backend = "auto"
        print(f"\n  Backend: auto (only option available)")

    # ── Cache strategy ──
    available_strategies = [s for s in strategies if s.get("available")]
    if available_strategies:
        strat_labels = []
        for s in available_strategies:
            label = s["name"]
            if s.get("bitRange"):
                label += f" ({s['bitRange'][0]}–{s['bitRange'][1]} bit)"
            if s.get("requiredLlamaBinary") == "turbo":
                label += " [turbo binary]"
            strat_labels.append(label)
        idx = _pick_one("Cache strategy", strat_labels)
        cache_strategy = available_strategies[idx]
    else:
        cache_strategy = {"id": "native", "name": "Native f16", "bitRange": None, "defaultBits": None}
        print(f"\n  Cache strategy: native (only option)")

    strategy_id = cache_strategy["id"]

    # ── Cache bits ──
    cache_bits = 0
    if cache_strategy.get("bitRange"):
        lo, hi = cache_strategy["bitRange"]
        default_bits = cache_strategy.get("defaultBits") or lo
        cache_bits = _pick_int(
            f"Cache quantisation bits ({lo}–{hi})",
            default_bits, lo, hi,
        )

    # ── FP16 layers ──
    fp16_layers = 0
    if cache_strategy.get("supportsFp16Layers"):
        fp16_layers = _pick_int("FP16 layers (full-precision head/tail)", 4, 0, 16)

    # ── Context tokens ──
    default_ctx = launch_prefs.get("contextTokens", 8192)
    max_ctx_hint = selected_model.get("maxContext") or 131072
    context_tokens = _pick_int(
        f"Context tokens (256–{max_ctx_hint})",
        default_ctx, 256, max_ctx_hint,
    )

    # ── Max output tokens ──
    max_tokens = _pick_int("Max output tokens", 4096, 64, 32768)

    # ── Temperature ──
    temperature = _pick_float("Temperature", 0.7, 0.0, 2.0)

    # ── Fused attention ──
    fused_attention = _pick_bool("Fused attention", False)

    # ── Fit model in memory ──
    fit_model_in_memory = _pick_bool("Fit model in memory", True)

    # ── Thinking mode ──
    thinking_mode = "off"
    thinking_choice = _pick_one("Thinking mode", ["off — no reasoning tokens", "auto — model decides when to think"])
    if thinking_choice == 1:
        thinking_mode = "auto"

    # ── DFlash speculative decoding ──
    speculative_decoding = False
    tree_budget = 0
    if dflash.get("available"):
        speculative_decoding = _pick_bool("Enable DFlash speculative decoding", False)
        if speculative_decoding:
            tree_budget = _pick_int("DFlash tree budget", 16, 1, 64)
            # DFlash forces native cache
            if strategy_id != "native":
                print(f"  Note: DFlash overrides cache strategy to native (was {strategy_id})")
                strategy_id = "native"
                cache_bits = 0
                fp16_layers = 0

    return {
        "backend": backend,
        "cacheStrategy": strategy_id,
        "cacheBits": cache_bits,
        "fp16Layers": fp16_layers,
        "contextTokens": context_tokens,
        "maxTokens": max_tokens,
        "temperature": temperature,
        "fusedAttention": fused_attention,
        "fitModelInMemory": fit_model_in_memory,
        "speculativeDecoding": speculative_decoding,
        "treeBudget": tree_budget,
        "thinkingMode": thinking_mode,
    }


# ── Prompt selection ─────────────────────────────────────────────────

BUILTIN_PROMPTS = [
    ("Quick sanity check",
     "What is 2+2? Answer in one word."),
    ("Short explanation",
     "Explain how KV-cache compression works in 3 sentences."),
    ("Reasoning task",
     "A farmer has 17 sheep. All but 9 run away. How many does he have left? Think step by step."),
    ("Code generation",
     "Write a Python function that checks if a string is a valid IPv4 address. Include type hints."),
    ("Long-form output",
     "Write a detailed comparison of transformer attention mechanisms: multi-head attention, "
     "grouped-query attention, and multi-query attention. Cover memory usage, throughput, and quality tradeoffs."),
    ("Creative writing",
     "Write a short story (200 words) about a machine learning model that becomes self-aware "
     "during a benchmark run."),
    ("Multi-turn context",
     "I'm building a FastAPI application. How should I structure the project? "
     "After you answer, I'll ask follow-up questions."),
]


def select_prompt(cli_prompt: str | None) -> str:
    if cli_prompt:
        print(f"\n  Using CLI prompt: {cli_prompt[:80]}{'...' if len(cli_prompt) > 80 else ''}")
        return cli_prompt

    labels = [f"{name} — {prompt[:60]}..." for name, prompt in BUILTIN_PROMPTS]
    labels.append("Custom prompt")
    idx = _pick_one("Test prompt", labels)

    if idx < len(BUILTIN_PROMPTS):
        return BUILTIN_PROMPTS[idx][1]

    custom = input("\n  Enter your prompt: ").strip()
    if not custom:
        print("  Empty prompt, using default.")
        return BUILTIN_PROMPTS[1][1]
    return custom


# ── Run inference ────────────────────────────────────────────────────

def run_inference(
    port: int,
    model: dict,
    config: dict,
    prompt: str,
    run_id: str,
) -> dict:
    """Load the model, run inference, collect results."""
    model_ref = _model_ref(model)
    model_path = model.get("path")

    # ── Step 1: Load the model ──
    print(f"\n{'═' * 60}")
    print(f"  LOADING MODEL")
    print(f"{'═' * 60}")
    print(f"  Model:    {model['name']}")
    print(f"  Ref:      {model_ref}")
    print(f"  Backend:  {config['backend']}")
    print(f"  Cache:    {config['cacheStrategy']} @ {config['cacheBits']} bit")
    if config["speculativeDecoding"]:
        print(f"  DFlash:   enabled (tree budget {config['treeBudget']})")
    print(f"  Context:  {config['contextTokens']} tokens")
    print()

    load_start = time.perf_counter()
    try:
        load_resp = _api("POST", "/api/models/load", port=port, body={
            "modelRef": model_ref,
            "modelName": model["name"],
            "source": model.get("sourceKind", "catalog"),
            "backend": config["backend"],
            "path": model_path,
            "cacheStrategy": config["cacheStrategy"],
            "cacheBits": config["cacheBits"],
            "fp16Layers": config["fp16Layers"],
            "fusedAttention": config["fusedAttention"],
            "fitModelInMemory": config["fitModelInMemory"],
            "contextTokens": config["contextTokens"],
            "speculativeDecoding": config["speculativeDecoding"],
            "treeBudget": config["treeBudget"],
            # FU-002: forward kvBudget so TriAttention MLX strategy
            # picks up the configured budget at apply time.
            "kvBudget": config.get("kvBudget", 2048),
        }, timeout=300)
    except RuntimeError as exc:
        return {
            "runId": run_id,
            "status": "load_failed",
            "error": str(exc),
            "model": model,
            "config": config,
            "prompt": prompt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    load_elapsed = time.perf_counter() - load_start

    runtime = load_resp.get("runtime", {})
    loaded_model = runtime.get("loadedModel", {})
    print(f"  Loaded in {load_elapsed:.1f}s")
    print(f"  Engine:   {loaded_model.get('engine', '?')}")
    if loaded_model.get("runtimeNote"):
        print(f"  Note:     {loaded_model['runtimeNote']}")
    if loaded_model.get("dflashDraftModel"):
        print(f"  Draft:    {loaded_model['dflashDraftModel']}")

    # ── Step 2: Run streaming inference ──
    print(f"\n{'═' * 60}")
    print(f"  GENERATING")
    print(f"{'═' * 60}")
    print(f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"  Max tokens: {config['maxTokens']}, temp: {config['temperature']}")
    print()

    session_id = f"test-{run_id}"
    thinking_mode = config.get("thinkingMode", "off")
    gen_start = time.perf_counter()
    try:
        full_text, full_reasoning, done_payload = _stream_api(
            "/api/chat/generate/stream",
            port=port,
            body={
                "sessionId": session_id,
                "title": f"Test run {run_id[:8]}",
                "prompt": prompt,
                "modelRef": model_ref,
                "modelName": model["name"],
                "source": model.get("sourceKind", "catalog"),
                "path": model_path,
                "backend": config["backend"],
                "thinkingMode": thinking_mode,
                "temperature": config["temperature"],
                "maxTokens": config["maxTokens"],
                "cacheStrategy": config["cacheStrategy"],
                "cacheBits": config["cacheBits"],
                "fp16Layers": config["fp16Layers"],
                "fusedAttention": config["fusedAttention"],
                "fitModelInMemory": config["fitModelInMemory"],
                "contextTokens": config["contextTokens"],
                "speculativeDecoding": config["speculativeDecoding"],
                "treeBudget": config["treeBudget"],
                "kvBudget": config.get("kvBudget", 2048),
                # Bug 1 / multimodal images: base64 blobs forwarded
                # straight through; backend dispatches via
                # is_multimodal_family + mlx_vlm.generate.
                "images": config.get("images") or [],
            },
            timeout=300,
        )
    except RuntimeError as exc:
        return {
            "runId": run_id,
            "status": "generation_failed",
            "error": str(exc),
            "model": model,
            "config": config,
            "prompt": prompt,
            "loadTimeSeconds": round(load_elapsed, 2),
            "loadedModel": loaded_model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    gen_elapsed = time.perf_counter() - gen_start

    # ── Step 3: Collect results ──
    assistant = done_payload.get("assistant", {})
    metrics = assistant.get("metrics", {})
    session = done_payload.get("session", {})
    runtime_status = done_payload.get("runtime", {})

    # Use streamed reasoning if the done payload didn't include it
    saved_reasoning = assistant.get("reasoning") or full_reasoning or None

    # Print summary
    tok_s = metrics.get("tokS", 0)
    prompt_tokens = metrics.get("promptTokens", 0)
    completion_tokens = metrics.get("completionTokens", 0)
    dflash_rate = metrics.get("dflashAcceptanceRate")

    print(f"\n{'─' * 60}")
    print(f"  RESULTS")
    print(f"{'─' * 60}")
    print(f"  Tokens:     {prompt_tokens} prompt + {completion_tokens} completion")
    print(f"  Speed:      {tok_s} tok/s")
    print(f"  Gen time:   {gen_elapsed:.2f}s")
    print(f"  Load time:  {load_elapsed:.2f}s")
    if dflash_rate is not None:
        # dflashAcceptanceRate is avg accepted tokens per step, not a 0-1 ratio
        print(f"  DFlash:     {dflash_rate:.1f} avg accepted tokens/step")
    if saved_reasoning:
        print(f"  Reasoning:  {len(saved_reasoning)} chars")
    print(f"\n  Output preview:")
    preview = full_text[:500].strip()
    if preview:
        for line in preview.splitlines():
            print(f"    {line}")
        if len(full_text) > 500:
            print(f"    ... ({len(full_text)} chars total)")
    else:
        print(f"    (no visible text — all output was reasoning/thinking)")
        if saved_reasoning:
            print(f"\n  Reasoning preview:")
            r_preview = saved_reasoning[:500].strip()
            for line in r_preview.splitlines():
                print(f"    {line}")
            if len(saved_reasoning) > 500:
                print(f"    ... ({len(saved_reasoning)} chars total)")

    result = {
        "runId": run_id,
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": model["name"],
            "ref": model_ref,
            "path": model.get("path"),
            "format": model.get("format"),
            "quantization": model.get("quantization"),
            "backend": model.get("backend"),
            "sizeGb": model.get("sizeGb"),
        },
        "config": config,
        "prompt": prompt,
        "loadedModel": loaded_model,
        "loadTimeSeconds": round(load_elapsed, 2),
        "generationTimeSeconds": round(gen_elapsed, 2),
        "output": {
            "text": full_text,
            "reasoning": saved_reasoning,
        },
        "metrics": metrics,
        "runtimeStatus": runtime_status,
    }
    return result


# ── Save results ─────────────────────────────────────────────────────

def save_result(result: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = result.get("model", {}).get("name", "unknown")
    model_slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in model_slug)[:40]
    filename = f"{ts}_{model_slug}_{result['runId'][:8]}.json"
    path = results_dir / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return path


# ── Batch mode ───────────────────────────────────────────────────────

def run_batch(port: int, batch_file: Path) -> None:
    """Run multiple tests from a JSON batch file.

    Batch file format:
    [
        {
            "modelRef": "mlx-community/Qwen3-8B-4bit",
            "modelName": "Qwen3 8B 4bit",
            "path": "/path/to/model",
            "backend": "auto",
            "cacheStrategy": "native",
            "cacheBits": 0,
            "fp16Layers": 0,
            "contextTokens": 8192,
            "maxTokens": 4096,
            "temperature": 0.7,
            "fusedAttention": false,
            "fitModelInMemory": true,
            "speculativeDecoding": false,
            "treeBudget": 0,
            "prompt": "Hello!"
        }
    ]
    """
    with open(batch_file, "r", encoding="utf-8") as f:
        tests = json.load(f)

    if not isinstance(tests, list):
        print("Batch file must be a JSON array.")
        sys.exit(1)

    print(f"\n  Running {len(tests)} batch test(s)...")
    results = []
    for i, test in enumerate(tests, 1):
        run_id = uuid.uuid4().hex
        print(f"\n{'━' * 60}")
        print(f"  TEST {i}/{len(tests)}")
        print(f"{'━' * 60}")

        model = {
            "name": test.get("modelName", test.get("modelRef", "unknown")),
            "path": test.get("path"),
            "format": test.get("format", "unknown"),
            "backend": test.get("backend", "auto"),
        }
        config = {
            "backend": test.get("backend", "auto"),
            "cacheStrategy": test.get("cacheStrategy", "native"),
            "cacheBits": test.get("cacheBits", 0),
            "fp16Layers": test.get("fp16Layers", 0),
            "contextTokens": test.get("contextTokens", 8192),
            "maxTokens": test.get("maxTokens", 4096),
            "temperature": test.get("temperature", 0.7),
            "fusedAttention": test.get("fusedAttention", False),
            "fitModelInMemory": test.get("fitModelInMemory", True),
            "speculativeDecoding": test.get("speculativeDecoding", False),
            "treeBudget": test.get("treeBudget", 0),
            "thinkingMode": test.get("thinkingMode", "off"),
            # FU-002: TriAttention MLX kv_budget. Backend defaults
            # to 2048 server-side; only consulted when
            # cacheStrategy == "triattention".
            "kvBudget": test.get("kvBudget", 2048),
            # Bug 1 / multimodal images: base64-encoded image blobs
            # forwarded to the chat /stream endpoint. Empty list →
            # text-only request.
            "images": test.get("images", []),
        }
        prompt = test.get("prompt", DEFAULT_PROMPT)
        result = run_inference(port, model, config, prompt, run_id)
        results.append(result)

    # Save all results in one file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"batch-{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    passed = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - passed
    print(f"\n{'━' * 60}")
    print(f"  BATCH COMPLETE: {passed} passed, {failed} failed")
    print(f"  Results: {path}")
    print(f"{'━' * 60}")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChaosEngineAI inference test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/inference-test-runner.py\n"
            "  python scripts/inference-test-runner.py --port 9090\n"
            '  python scripts/inference-test-runner.py --prompt "Write a haiku"\n'
            "  python scripts/inference-test-runner.py --batch tests/batch.json\n"
            "  python scripts/inference-test-runner.py --results-dir ./my-results\n"
        ),
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Backend port (default: {DEFAULT_PORT})")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to use (skips prompt menu)")
    parser.add_argument("--batch", type=str, default=None, help="Path to batch test file (JSON array)")
    parser.add_argument("--results-dir", type=str, default=None, help=f"Results directory (default: {RESULTS_DIR})")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    print(f"\n{'━' * 60}")
    print("  ChaosEngineAI Inference Test Runner")
    print(f"{'━' * 60}")

    # Check backend is reachable
    try:
        health = _api("GET", "/api/health", port=args.port, timeout=5)
        print(f"  Backend:  v{health.get('appVersion', '?')} on port {args.port}")
        print(f"  Engine:   {health.get('engine', '?')}")
        loaded = health.get("loadedModel")
        if loaded:
            print(f"  Loaded:   {loaded.get('name', '?')}")
        else:
            print(f"  Loaded:   (none)")
    except ConnectionError as exc:
        print(f"\n  ERROR: {exc}")
        print(f"\n  Start the backend first:")
        print(f"    cd {Path(__file__).resolve().parents[1]}")
        print(f"    .venv/bin/python -m backend_service")
        sys.exit(1)

    # Batch mode
    if args.batch:
        run_batch(args.port, Path(args.batch))
        return

    # Interactive mode
    workspace = _api("GET", "/api/workspace", port=args.port)
    models = discover_models(args.port)

    # ── Select model ──
    model_labels = [_format_model_line(m) for m in models]
    idx = _pick_one(f"Select model ({len(models)} available)", model_labels)
    selected_model = models[idx]

    # ── Configure runtime ──
    config = configure_runtime(workspace, selected_model)

    # ── Select prompt ──
    prompt = select_prompt(args.prompt)

    # ── Confirm and run ──
    run_id = uuid.uuid4().hex
    print(f"\n{'═' * 60}")
    print(f"  READY TO RUN")
    print(f"{'═' * 60}")
    print(f"  Model:    {selected_model['name']}")
    print(f"  Cache:    {config['cacheStrategy']}", end="")
    if config["cacheBits"]:
        print(f" @ {config['cacheBits']} bit", end="")
    print()
    print(f"  Context:  {config['contextTokens']}")
    print(f"  Thinking: {config.get('thinkingMode', 'off')}")
    print(f"  DFlash:   {'on' if config['speculativeDecoding'] else 'off'}")
    print(f"  Prompt:   {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    print()

    confirm = input("  Run inference? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("  Aborted.")
        return

    result = run_inference(args.port, selected_model, config, prompt, run_id)

    # Save
    path = save_result(result, results_dir)
    print(f"\n  Results saved to: {path}")

    # Ask to run another
    while True:
        again = input("\n  Run another test? [y/N]: ").strip().lower()
        if again not in ("y", "yes"):
            break

        # Offer to keep same model or pick new one
        keep_model = _pick_bool("Keep same model", True)
        if not keep_model:
            idx = _pick_one(f"Select model ({len(models)} available)", model_labels)
            selected_model = models[idx]
            config = configure_runtime(workspace, selected_model)

        keep_config = _pick_bool("Keep same runtime config", True)
        if not keep_config:
            config = configure_runtime(workspace, selected_model)

        prompt = select_prompt(None)
        run_id = uuid.uuid4().hex
        result = run_inference(args.port, selected_model, config, prompt, run_id)
        path = save_result(result, results_dir)
        print(f"\n  Results saved to: {path}")

    print("\n  Done. All results in:", results_dir)


if __name__ == "__main__":
    main()
