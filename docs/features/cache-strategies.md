# Cache strategies

ChaosEngineAI uses a pluggable cache strategy system. Each strategy implements
a common interface
([`cache_compression.CacheStrategy`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/cache_compression/__init__.py))
so the MLX worker, the `llama.cpp` engine, and the diffusion runtimes can use
it without knowing the details.

## Strategy registry

The current registry (after FU-030 cleanup):

| ID | Display | Domain | Backing | Platforms |
|---|---|---|---|---|
| `native` | Native f16 | text | Built-in | All |
| `triattention` | TriAttention | text | `triattention` + `vllm` | Linux + CUDA (via vLLM) |
| `turboquant` | TurboQuant | text | `turboquant-mlx-full` (MLX) or `llama-server-turbo` (GGUF fork) | Apple Silicon (MLX) + Linux/Windows + CUDA / Metal (llama.cpp fork) |
| `teacache` | TeaCache | image / video | Vendored forward patches | DiT: FLUX, HunyuanVideo, LTX-Video, CogVideoX, Mochi |
| `fbcache` | First Block Cache | image / video | `diffusers ≥0.36` | DiT: FLUX, SD3.5, Wan, HunyuanVideo, LTX-Video, CogVideoX, Mochi |
| `taylorseer` | TaylorSeer Cache | image / video | `diffusers ≥0.38` | DiT (per-pipeline) |
| `magcache` | MagCache | image / video | `diffusers ≥0.38` | FLUX only without calibration |
| `pab` | Pyramid Attention Broadcast | image / video | `diffusers ≥0.38` | DiT (per-pipeline) |
| `fastercache` | FasterCache | image / video | `diffusers ≥0.38` | DiT (per-pipeline) |

UNet pipelines (SD 1.5, SDXL) don't accept the diffusion caches — the
strategy raises `NotImplementedError` and the backend surfaces that as a
`runtimeNote` so the UI can grey out the unsupported choice.

## Legacy aliases

FU-030 removed the `chaosengine` and `rotorquant` strategy slots. Persisted
configs that still reference them coerce silently to `turboquant`:

```python
# cache_compression/__init__.py
_LEGACY_STRATEGY_ALIASES = {
    "chaosengine": "turboquant",
    "rotorquant": "turboquant",
}
```

The coercion happens at `registry.resolve_legacy_id()` and at the public
`registry.get()` entry point so callers don't need to special-case it. The
frontend mirrors the alias map in
[`src/components/runtimeSupport.ts`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/src/components/runtimeSupport.ts).

## Native f16

The built-in path. No compression — `--cache-type-k f16 --cache-type-v f16`
for `llama-server`, default MLX cache for the MLX worker. Maximum quality,
no install needed, supports every model. This is the only cache strategy
that's compatible with DFlash / DDTree speculative decoding today.

## TurboQuant

Hadamard / Walsh-Hadamard rotation-based KV cache compression. Two
backends share the strategy id:

- **MLX path** — `turboquant-mlx-full ≥0.3.0` (pinned in `pyproject.toml`).
  Native MLX cache replacement; auto-falls back to `QuantizedKVCache` when
  the full TurboQuant path is unavailable. v0.3.0 (2026-05-03) brought
  asymmetric K/V bits, layer-adaptive precision, the `--no-quant` eval
  flag, NumPy 2.0 + transformers 5.x compatibility.
- **llama.cpp path** — the `llama-server-turbo` fork at
  [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
  (`feature/turboquant-kv-cache` branch). Adds `--cache-type-k turbo{2,3,4}`
  cache types alongside the standard ones. Build it locally with
  `scripts/build-llama-turbo.sh`; the binary lands at
  `~/.chaosengine/bin/llama-server-turbo`. The `LlamaCppEngine._select_llama_binary`
  method routes to the turbo binary when the strategy reports
  `required_llama_binary() == "turbo"`.

Bit range: 1 - 4. Default: 3. Layer-adaptive fp16-layer count is
configurable from the launch modal.

## TriAttention

KV cache compression integrated into vLLM's scheduler. **Linux + CUDA
only** — the upstream package wires into vLLM's request batching, which
isn't packaged for macOS.

The strategy's `apply_vllm_patches()` hook is called by `VLLMEngine.load_model()`
before constructing the `LLM` instance. There's also a TriAttention-MLX
direct path (`apply_triattention_mlx`) used inside `WorkerState._apply_cache_profile`
when `cacheStrategy == "triattention"` on Apple Silicon — this gives a
norm-only norm-scored compressor with a configurable `kvBudget` (default
2048 tokens), but it's not the full Triattention experience.

Bit range: 1 - 4. Default: 3.

## Diffusion caches

All five diffusion-side strategies share the same contract:

- `applies_to() == {"image", "video"}` — they don't surface in the LLM
  cache picker.
- They patch `pipeline.transformer.enable_cache(<Config>)` or apply a
  forward-level hook before the denoise loop.
- They expose a single primary knob — TeaCache's `rel_l1_thresh` (default
  0.4), FBCache's threshold (default 0.12), TaylorSeer / PAB / FasterCache's
  cache interval or skip range.

FBCache + SageAttention stack multiplicatively on CUDA — community Wan 2.1
720P reports cumulative 54% wall-time reduction.

## Cache preview

The launch modal previews how much memory each strategy will use for the
current context length. The `GET /api/cache/preview` endpoint takes the
model architecture parameters (layer count, head count, hidden size, KV
heads) and returns baseline + optimised bytes per strategy. The MLX worker
+ llama.cpp engine consult the same helper at load time.

```bash
./scripts/chaosengine-cli cache-preview \
    --layers 64 --heads 64 --hidden 8192 --context 32768 --kv-heads 8 \
    --strategy turboquant --bits 3
```

## See also

- [DFlash](dflash.md) — speculative decoding currently requires native
  cache.
- [MTPLX](mtplx.md) — same restriction.
- [Fused attention](fused-attention.md) — orthogonal performance knob.
