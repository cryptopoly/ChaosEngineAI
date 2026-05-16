# Engine routing

When a load request lands, the `RuntimeController._select_engine`
function in
[`backend_service/inference/controller.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/controller.py)
picks which engine handles it. The decision is deterministic — same
inputs, same engine, every time.

## Inputs

```python
def _select_engine(
    self,
    *,
    backend: str,                # "auto" | "mlx" | "gguf" | "vllm" | "remote"
    runtime_target: str | None,  # resolved on-disk path
    path: str | None,            # caller-supplied path hint
    model_ref: str = "",         # canonical HF repo id
    canonical_repo: str | None = None,
    speculative_decoding: bool = False,
) -> BaseInferenceEngine:
```

## Decision tree

The function follows this order:

1. **Explicit remote / cloud.** If `backend in {"remote", "openai",
   "cloud"}` → `RemoteOpenAIEngine`.

2. **Explicit MLX.** If `backend == "mlx"`:
   - If `speculative_decoding` and MTPLX is available and the model
     has MTP heads (`has_mtp_heads(canonical_repo or model_ref)`) →
     `MtplxEngine`.
   - Otherwise → `MLXWorkerEngine`.
   - If MLX isn't usable on this machine → raise with a clear error
     telling the user to fall back to GGUF.

3. **Explicit GGUF.** If `backend in {"gguf", "llama.cpp",
   "llama-cpp"}` and GGUF is available → `LlamaCppEngine`.

4. **Explicit vLLM.** If `backend == "vllm"` and vLLM is installed →
   `VLLMEngine`.

5. **Auto-detect.** If the target looks like a GGUF path
   (`_looks_like_gguf(target)`) and GGUF is available →
   `LlamaCppEngine`. Otherwise:
   - If MLX is usable → `MLXWorkerEngine`.
   - Else if GGUF is available → `LlamaCppEngine`.
   - Else → raise (no backend available).

## Speculative-decoding routing

The MLX branch handles the MTPLX vs DFlash priority:

| `speculativeDecoding` | MTPLX available | Has MTP heads | DFlash available | Engine | Speculative mode |
|---|---|---|---|---|---|
| false | — | — | — | `MLXWorkerEngine` | off |
| true | yes | yes | — | `MtplxEngine` | mtplx |
| true | yes | no | yes | `MLXWorkerEngine` | dflash |
| true | yes | no | no | `MLXWorkerEngine` | off + runtimeNote |
| true | no | — | yes | `MLXWorkerEngine` | dflash |
| true | no | — | no | `MLXWorkerEngine` | off + runtimeNote |

DFlash isn't a separate engine — it runs inside the MLX worker
(`mlx_worker_speculative.py`) when `dflashAvailable` + the target
resolves to a draft via `model_resolution.resolve_dflash_target_ref`.

The vLLM engine has its own DFlash branch via the upstream `dflash`
package. MTPLX has no vLLM equivalent today.

## Fallback ladder

If the chosen engine raises during `load_model()`, the controller
catches the failure and tries the next reasonable engine. The full
ladder:

1. **Chosen engine.** Selected by `_select_engine` per the table above.
2. **MLXWorkerEngine.** Hit when MTPLX falls back (see below) — also
   the implicit fallback when the chosen engine wasn't strictly
   required (e.g. the user picked "auto").
3. **LlamaCppEngine.** Hit when MLX isn't usable but a GGUF binary is
   resolvable.

MTPLX → MLX is the documented contract: `MtplxEngine.load_model` raises
`RuntimeError` on any startup failure, and the `RuntimeController`
catches that and immediately tries `MLXWorkerEngine`. The same model
ref loads with one ladder rung difference.

For the llama.cpp engine, the cache-strategy fallback is **two-level**
after FU-030 (was three): requested strategy → native (the deprecated
`chaosengine` / `rotorquant` slots were removed). If the requested
strategy needs the turbo binary and that binary isn't installed, the
engine falls back to native and logs a runtimeNote explaining why.

## Why runtimeNote matters

Every routing decision the controller makes ends up in `runtimeNote`.
The frontend renders this on every chat / compare turn so the user can
see what actually happened, not what they asked for. Common notes you'll
see:

- `"mlx + dflash"` — DFlash fired.
- `"mtplx"` — MTPLX engine.
- `"mlx (mtplx fallback: venv missing)"` — MTPLX requested but venv not
  installed.
- `"llama-server-turbo + turboquant3"` — GGUF turbo path active.
- `"llama-server (cache fallback: turbo not installed)"` — turbo
  binary missing, fell back to native.
- `"remote openai-compatible"` — request hit a remote provider.

The E2E suite's Phase 1 asserts on `runtimeNote` to prove routing
correctness — see [E2E testing](../testing/e2e-testing.md).

## See also

- [Inference engines](inference-engines.md).
- [MTPLX](../features/mtplx.md).
- [DFlash](../features/dflash.md).
- [Cache strategies](../features/cache-strategies.md).
