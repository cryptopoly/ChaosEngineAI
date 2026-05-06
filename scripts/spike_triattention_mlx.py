"""FU-002 spike: validate triattention.mlx on a small Qwen.

Loads mlx-community/Qwen2.5-0.5B-Instruct-4bit via mlx_lm, applies
``apply_triattention_mlx(model, kv_budget=2048)``, runs a short generation,
and reports wall-time + first-256-char output. Compare to baseline (same
model without TriAttention) to gauge whether the integration is shippable.

Run: ``./.venv/bin/python scripts/spike_triattention_mlx.py``
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback


def _format_section(title: str) -> str:
    return f"\n=== {title} ===\n"


def _run(model_id: str, *, with_triattention: bool, kv_budget: int, max_tokens: int, prompt: str) -> dict:
    from mlx_lm import load, generate

    print(_format_section(f"loading {model_id} (with_triattention={with_triattention})"))
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    print(f"load wall-time: {time.perf_counter() - t0:.2f}s")

    if with_triattention:
        from triattention.mlx import apply_triattention_mlx
        print(f"applying apply_triattention_mlx(kv_budget={kv_budget})")
        t1 = time.perf_counter()
        try:
            apply_triattention_mlx(model, kv_budget=kv_budget)
            print(f"apply wall-time: {time.perf_counter() - t1:.2f}s")
        except Exception as exc:
            print(f"apply_triattention_mlx FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return {"failed": True, "stage": "apply", "error": str(exc)}

    print(_format_section(f"generate (max_tokens={max_tokens})"))
    t2 = time.perf_counter()
    try:
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    except Exception as exc:
        print(f"generate FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return {"failed": True, "stage": "generate", "error": str(exc)}
    elapsed = time.perf_counter() - t2

    print(f"gen wall-time: {elapsed:.2f}s ({max_tokens / max(elapsed, 0.001):.1f} tok/s)")
    print(f"output (first 256 chars):\n{out[:256]!r}")

    return {
        "failed": False,
        "elapsed": elapsed,
        "output": out,
        "tokens_per_sec": max_tokens / max(elapsed, 0.001),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="HF model id loadable by mlx_lm.load",
    )
    parser.add_argument("--kv-budget", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt",
        default="Write one sentence about why caching helps inference:",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the no-TriAttention baseline run (saves time).",
    )
    args = parser.parse_args(argv)

    print(_format_section("environment check"))
    try:
        import triattention  # noqa: F401
        from triattention.mlx import apply_triattention_mlx  # noqa: F401
        print("triattention.mlx import: OK")
    except ImportError as exc:
        print(f"triattention.mlx NOT importable: {exc}")
        return 2

    try:
        import mlx_lm  # noqa: F401
        print(f"mlx_lm import: OK (version {getattr(mlx_lm, '__version__', 'unknown')})")
    except ImportError as exc:
        print(f"mlx_lm NOT importable: {exc}")
        return 2

    if not args.skip_baseline:
        print(_format_section("BASELINE (no triattention)"))
        baseline = _run(
            args.model,
            with_triattention=False,
            kv_budget=args.kv_budget,
            max_tokens=args.max_tokens,
            prompt=args.prompt,
        )
    else:
        baseline = None

    print(_format_section("WITH TRIATTENTION"))
    triatt = _run(
        args.model,
        with_triattention=True,
        kv_budget=args.kv_budget,
        max_tokens=args.max_tokens,
        prompt=args.prompt,
    )

    print(_format_section("verdict"))
    if triatt.get("failed"):
        print(f"FAIL — TriAttention {triatt.get('stage')} stage raised. FU-002 stays parked.")
        return 1

    if not triatt.get("output", "").strip():
        print("FAIL — generation returned empty string with TriAttention applied.")
        return 1

    if baseline and not baseline.get("failed"):
        speedup = baseline["elapsed"] / max(triatt["elapsed"], 0.001)
        print(f"baseline: {baseline['elapsed']:.2f}s")
        print(f"triatt:   {triatt['elapsed']:.2f}s")
        print(f"speedup:  {speedup:.2f}x  ({'helpful' if speedup > 1.05 else 'neutral or slower'})")

    print("PASS — apply_triattention_mlx works on this model. FU-002 unblocked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
