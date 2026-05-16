# Inference engines

Every LLM inference path in ChaosEngineAI implements the same
`BaseInferenceEngine` interface defined in
[`backend_service/inference/base.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/base.py).
The `RuntimeController` picks one per model load and the routing layer
proxies through whichever engine is active.

## Engines

### MLX worker — `MLXWorkerEngine`

The default on Apple Silicon. Loads `mlx-lm` models inside a separate
Python subprocess (`backend_service/mlx_worker.py`) and speaks JSON-RPC
over a pipe to the FastAPI parent.

Why subprocess: a stuck `mlx.generate()` would hang the FastAPI thread
otherwise. Memory hogs stay isolated; killing the worker reclaims the
GPU memory.

The worker is itself sliced into 11 sibling modules
(`mlx_worker_{request,prompt,io,diagnostics,multimodal,cache,eval,loader,
lifecycle,speculative,generate}.py`) — each one owns a coherent slice of
worker responsibilities so the main `mlx_worker.py` stays at ~318 LOC.

Supports: most `mlx-lm`-compatible models, vision (`mlx-vlm`), DFlash +
DDTree speculative decoding, TurboQuant MLX cache, TriAttention MLX
direct path, multimodal inputs.

### llama.cpp — `LlamaCppEngine`

The default on Linux + CUDA and Windows. Wraps `llama-server` (or
`llama-server-turbo` for TurboQuant) as a subprocess.

The engine picks which binary to use via
`_select_llama_binary(strategy)` — the active cache strategy's
`required_llama_binary()` returns `"standard"` or `"turbo"`, and the
binary path is resolved against:

1. `CHAOSENGINE_LLAMA_SERVER` / `CHAOSENGINE_LLAMA_SERVER_TURBO` env vars.
2. `~/.chaosengine/bin/` (managed by `scripts/build-llama-turbo.sh`).
3. The bundled runtime under the workspace.
4. `PATH` (Homebrew on macOS, system installs on Linux).

Cache types are pre-validated against the binary's `--help` output to
catch mismatched flags before the server boots. The full
`--cache-type-k turbo{2,3,4}` set is only valid on the turbo fork; the
standard binary only accepts `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl,
q5_0, q5_1`.

Supports: every GGUF model in the curated catalog, vision (mmproj),
DFlash via the upstream `--draft-model` flag.

### MTPLX — `MtplxEngine`

Apple Silicon only, gated on `mtplxAvailable` + the model carrying MTP
heads. Spawns `mtplx start --model <path> --port N` from the isolated
venv at `~/.chaosengine/mtplx-venv/` and proxies `/v1/chat/completions`
through it.

Why a separate venv: MTPLX ships its own forked `mlx` runtime that
can't co-exist with upstream MLX in the same site-packages.

Fallback contract: `MtplxEngine.load_model` raises on any startup
failure. The `RuntimeController` catches that and falls back to
`MLXWorkerEngine`. The fallback path records the reason in
`runtimeNote`.

See [MTPLX deep dive](../features/mtplx.md).

### vLLM — `VLLMEngine`

Linux + CUDA only, gated on `vllmAvailable`. Loads vLLM's `LLM` class
in-process (no subprocess). Used for high-throughput / concurrent
serving where llama.cpp's single-threaded scheduler is a bottleneck.

The cache strategy's `apply_vllm_patches()` hook fires before the
`LLM` instance is constructed — that's how TriAttention gets its
scheduler hooks installed.

Supports: HF safetensors checkpoints, TriAttention cache compression,
DFlash via the `dflash` package.

### Remote OpenAI — `RemoteOpenAIEngine`

Forwards every chat / completion request to a configured remote
OpenAI-compatible provider. Configured per-model from the Settings
tab → Remote providers. Used as a fallback when local inference isn't
desired or the model isn't available locally.

### Mock — `MockInferenceEngine`

Returns canned responses without spawning anything. Used by the test
suite's `FakeRuntime`.

## Common contract

Every engine implements:

- `load_model(model_ref, **kwargs) → LoadedModelInfo`
- `unload_model() → None`
- `generate(payload) → GenerationResult` (non-streaming)
- `generate_stream(payload) → Iterator[StreamChunk]` (streaming)
- `process_pid() → int | None` (for hardware telemetry attribution)
- `update_profile(**kwargs) → None` (per-turn overrides)

Generation results carry:

- `text` — the completion.
- `metrics` — tok/s, TTFT, eval-token count.
- `runtimeNote` — engine + binary + cache strategy + speculative decoding
  state. Critical for routing diagnostics — the UI surfaces this verbatim
  on the per-turn host strip.

## Subprocess lifecycle

The `RuntimeController` tracks every subprocess it spawns. On a clean
load → unload → load cycle, the previous worker is shut down before the
new one starts (this is the FU-008 / v0.8.0 memory-leak fix — without
the `JsonRpcProcess.close()` ceiling, force-killing a worker holding
~47 GB of MLX weights could timeout, the exception got swallowed by the
route layer's broad `except`, and the next load spawned a second worker
alongside the dying one).

If a subprocess dies unexpectedly (OOM, segfault), the parent surfaces
the death cleanly:

- `runtimeNote` records the failure.
- The runtime state flips to `idle` or `error`.
- The diagnostics snapshot's `recentOrphanedWorkers` is populated.
- The user can retry from a clean state.

The E2E suite's Phase 7 asserts `recentOrphanedWorkers == []` after the
full sweep — see [E2E testing](../testing/e2e-testing.md).

## See also

- [Routing](routing.md) — `_select_engine` decision tree.
- [Cache strategies](../features/cache-strategies.md).
- [Runtime paths](runtime-paths.md).
