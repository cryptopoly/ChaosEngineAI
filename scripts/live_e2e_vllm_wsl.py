"""Live end-to-end test for the vLLM WSL bridge (FU-056 Phase 8).

Spawns ``VllmWslEngine`` against a tiny model (facebook/opt-125m,
~250 MB), waits for the server to come up, generates a single
completion, prints the result, and tears down. Not part of the
regular test suite — runs once to validate the bridge end to end
with real vLLM + real CUDA + real WSL.

Usage (run from the repo root, Windows + WSL with vllm-venv ready):
    .venv\\Scripts\\python.exe scripts\\live_e2e_vllm_wsl.py

Exit code 0 → bridge works, 1 → see stderr for the failure mode.
"""

from __future__ import annotations

import sys
import time
import traceback

from backend_service.inference.capabilities import _probe_native_backends
from backend_service.inference.vllm_wsl_engine import VllmWslEngine


def main() -> int:
    print("=" * 60)
    print("LIVE E2E: VllmWslEngine")
    print("=" * 60)

    # 1) Capabilities probe — bail loudly if WSL bridge isn't ready.
    print("\n[1/5] Probing capabilities...")
    caps = _probe_native_backends()
    print(f"  wsl2Available:      {caps.wsl2Available}")
    print(f"  wslDistroName:      {caps.wslDistroName}")
    print(f"  wslCudaAvailable:   {caps.wslCudaAvailable}")
    print(f"  wslVllmAvailable:   {caps.wslVllmAvailable}")
    print(f"  wslVllmVersion:     {caps.wslVllmVersion}")
    if not (caps.wsl2Available and caps.wslCudaAvailable and caps.wslVllmAvailable):
        print("\nBridge not ready — bail.", file=sys.stderr)
        return 1

    # 2) Construct the engine.
    print("\n[2/5] Constructing VllmWslEngine...")
    engine = VllmWslEngine(caps)

    # 3) Load a tiny chat-tuned model. ``Qwen/Qwen2.5-0.5B-Instruct``
    #    is 0.5B params, ~1 GB on disk, vLLM-compatible AND ships a
    #    chat template (OPT-125m doesn't — caught live on take 4).
    #    Downloads + loads in 1-3 min from cold cache.
    test_model = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n[3/5] Loading {test_model} through the WSL bridge...")
    print("      (vLLM cold-start: 30-90 s for graph build + CUDA warmup)")
    start = time.perf_counter()
    try:
        info = engine.load_model(
            model_ref=test_model,
            model_name="Qwen2.5-0.5B-Instruct",
            canonical_repo=test_model,
            source="catalog",
            backend="vllm",
            path=None,
            runtime_target=None,
            cache_strategy="native",
            cache_bits=0,
            fp16_layers=0,
            fused_attention=False,
            fit_model_in_memory=True,
            context_tokens=2048,
        )
    except Exception:  # noqa: BLE001 — print the full trace for triage
        print("\nLOAD FAILED:", file=sys.stderr)
        traceback.print_exc()
        return 1
    load_elapsed = time.perf_counter() - start
    print(f"  Loaded in {load_elapsed:.1f}s")
    print(f"  engine:        {info.engine}")
    print(f"  ref:           {info.ref}")
    print(f"  runtimeTarget: {info.runtimeTarget}")
    print(f"  runtimeNote:   {info.runtimeNote}")
    print(f"  pid:           {engine.process_pid()}")
    print(f"  port:          {engine.port}")

    # 4) Generate a small completion.
    print("\n[4/5] Generating: 'The capital of France is'")
    try:
        result = engine.generate(
            prompt="The capital of France is",
            history=[],
            system_prompt=None,
            max_tokens=20,
            temperature=0.0,
        )
    except Exception:  # noqa: BLE001
        print("\nGENERATE FAILED:", file=sys.stderr)
        traceback.print_exc()
        try:
            engine.unload_model()
        except Exception:  # noqa: BLE001
            pass
        return 1

    print(f"  text:             {result.text!r}")
    print(f"  finishReason:     {result.finishReason}")
    print(f"  promptTokens:     {result.promptTokens}")
    print(f"  completionTokens: {result.completionTokens}")
    print(f"  tokS:             {result.tokS}")
    print(f"  responseSeconds:  {result.responseSeconds}")

    # 5) Clean up.
    print("\n[5/5] Unloading + terminating WSL subprocess...")
    engine.unload_model()
    print("  Done.")

    print("\n" + "=" * 60)
    print("LIVE E2E: SUCCESS")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
