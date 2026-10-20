#!/usr/bin/env python3
"""Measure how much slower a compressed KV cache decodes than Native.

The launch-modal estimate (``backend_service/helpers/cache.py``
``_strategy_speed_map``) is a table of speed ratios per strategy and bit
width. This script produces measured ratios to put in it: for one model it
loads Native and then the chosen strategy at each bit width, runs a
fixed-length reply after prompts of several sizes, and divides decode tok/s
(prefill excluded) by Native's at the same prompt size.

Usage (backend must already be running, e.g. ``chaosengine-cli serve``):

    .venv/bin/python scripts/calibrate-cache-speed.py \\
        --model mlx-community/Qwen3-4B-bf16 --bits 2,3,4 \\
        --contexts 512,4096,16384

It prints a table, a ready-to-paste ``turboquant`` line for
``_strategy_speed_map``, and writes the raw numbers to
``~/.chaosengine/test-results/cache-speed-<timestamp>.json``. Run it on the
hardware the estimate is meant to describe; the model should be one the
backend can load with every strategy (a small MLX model is plenty).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Rough tokens per filler sentence; only used to size the prompt.
_TOKENS_PER_FILLER = 12
_REPLY_TOKENS = 128
_HEADROOM_TOKENS = 256


def _load_matrix_helpers():
    """Reuse the matrix runner's HTTP helpers (its file name has a hyphen)."""
    path = Path(__file__).with_name("cache-strategy-matrix.py")
    spec = importlib.util.spec_from_file_location("cache_strategy_matrix", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_prompt(target_tokens: int) -> str:
    """A deterministic prompt of roughly ``target_tokens`` tokens."""
    count = max(1, target_tokens // _TOKENS_PER_FILLER)
    lines = [f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(count)]
    return "\n".join(lines) + "\n\nIn one sentence, what repeats in the list above?"


def speed_ratios(
    native: dict[int, float],
    compressed: dict[int, dict[int, float]],
) -> dict[int, dict[int, float]]:
    """``{bits: {context: tok/s ÷ native tok/s at that context}}``.

    Contexts missing from Native, or with a non-positive Native speed, are
    skipped rather than dividing by zero.
    """
    ratios: dict[int, dict[int, float]] = {}
    for bits, by_context in compressed.items():
        row = {
            context: round(tok_s / native[context], 3)
            for context, tok_s in by_context.items()
            if native.get(context, 0) > 0 and tok_s > 0
        }
        if row:
            ratios[bits] = row
    return ratios


def suggest_map(ratios: dict[int, dict[int, float]]) -> dict[int, float]:
    """Median ratio across contexts for each bit width (what the table stores)."""
    return {bits: round(statistics.median(row.values()), 2) for bits, row in sorted(ratios.items())}


def format_map_line(strategy: str, suggested: dict[int, float]) -> str:
    body = ", ".join(f"{bits}: {value}" for bits, value in suggested.items())
    return f'"{strategy}": {{{body}}},'


def _measure(helpers, port: int, contexts: list[int], reply_tokens: int) -> dict[int, float]:
    """Decode tok/s for each prompt size on the model that is currently loaded."""
    results: dict[int, float] = {}
    for context in contexts:
        body = {
            "prompt": build_prompt(context),
            "maxTokens": reply_tokens,
            "temperature": 0.0,
            "seed": 42,
            "thinkingMode": "off",
        }
        _text, done = helpers._stream_inference(
            "/api/chat/generate/stream", port=port, body=body, timeout=900,
        )
        metrics = (done.get("assistant") or {}).get("metrics") or {}
        results[context] = float(metrics.get("tokS") or 0.0)
        print(f"    prompt ~{context:>6} tokens -> {results[context]:7.1f} tok/s", flush=True)
    return results


def _load(helpers, port: int, args, strategy: str, bits: int, context_tokens: int) -> None:
    helpers._api(
        "POST", "/api/models/load", port=port, timeout=600,
        body={
            "modelRef": args.model,
            "modelName": args.model.split("/")[-1],
            "canonicalRepo": args.model,
            "source": "library",
            "backend": args.backend,
            "cacheStrategy": strategy,
            "cacheBits": bits,
            "fp16Layers": args.fp16_layers if strategy != "native" else 0,
            "fusedAttention": False,
            "fitModelInMemory": True,
            "contextTokens": context_tokens,
            "speculativeDecoding": False,
            "treeBudget": 0,
            "thinkingMode": "off",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Library model ref, e.g. mlx-community/Qwen3-4B-bf16")
    parser.add_argument("--backend", default="mlx", help="mlx (default) or gguf")
    parser.add_argument("--strategy", default="turboquant")
    parser.add_argument("--bits", default="2,3,4", help="Comma-separated bit widths to test")
    parser.add_argument("--contexts", default="512,4096,16384", help="Comma-separated prompt sizes (tokens)")
    parser.add_argument("--tokens", type=int, default=_REPLY_TOKENS, help="Reply length for every run")
    parser.add_argument("--fp16-layers", type=int, default=0, help="FP16 layers kept uncompressed")
    parser.add_argument("--port", type=int, default=8876)
    args = parser.parse_args()

    bits_list = [int(part) for part in args.bits.split(",") if part.strip()]
    contexts = [int(part) for part in args.contexts.split(",") if part.strip()]
    context_tokens = max(contexts) + args.tokens + _HEADROOM_TOKENS
    helpers = _load_matrix_helpers()

    started = time.monotonic()
    print(f"Native f16 ({args.model})", flush=True)
    _load(helpers, args.port, args, "native", 0, context_tokens)
    native = _measure(helpers, args.port, contexts, args.tokens)

    compressed: dict[int, dict[int, float]] = {}
    for bits in bits_list:
        print(f"{args.strategy} {bits}-bit", flush=True)
        _load(helpers, args.port, args, args.strategy, bits, context_tokens)
        compressed[bits] = _measure(helpers, args.port, contexts, args.tokens)

    ratios = speed_ratios(native, compressed)
    suggested = suggest_map(ratios)

    print("\nSpeed ratio vs Native (decode tok/s, prefill excluded)")
    print("bits  " + "  ".join(f"{context:>7}" for context in contexts) + "   median")
    for bits, row in sorted(ratios.items()):
        cells = "  ".join(f"{row.get(context, float('nan')):7.2f}" for context in contexts)
        print(f"{bits:>4}  {cells}   {suggested[bits]:6.2f}")
    print("\nPaste into _strategy_speed_map (backend_service/helpers/cache.py):")
    print("    " + format_map_line(args.strategy, suggested))

    out_dir = Path.home() / ".chaosengine" / "test-results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cache-speed-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps({
        "model": args.model,
        "backend": args.backend,
        "strategy": args.strategy,
        "fp16Layers": args.fp16_layers,
        "replyTokens": args.tokens,
        "elapsedSeconds": round(time.monotonic() - started, 1),
        "nativeTokS": native,
        "compressedTokS": compressed,
        "ratios": ratios,
        "suggested": suggested,
    }, indent=2))
    print(f"\nRaw numbers: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
