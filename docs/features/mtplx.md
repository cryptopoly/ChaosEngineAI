# MTPLX — Native Multi-Token Prediction

**MTPLX** is the speculative-decoding path for models that were trained with
Multi-Token Prediction (MTP) heads. Unlike DFlash (which pairs a separately-
trained small draft model with the target), MTPLX uses the target model's
own trained MTP heads to propose multiple tokens per forward pass — yielding
1.8 - 2.2× speedup with zero quality loss on supported models.

MTPLX is **Apple Silicon only** today. It's powered by the
[`mtplx`](https://github.com/youssofal/mtplx) package (Apache 2.0), which
ships its own forked `mlx` runtime.

## Supported model registry

The canonical list lives in
[`backend_service/inference/_mtp.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/_mtp.py)
(`MTP_MODEL_MAP` + `_MTP_ALIASES`):

| Family | Canonical repo |
|---|---|
| **Youssofal MTPLX-Optimized** | `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`, `-Speed-FP16`, `-Quality` |
| **Qwen3.5** | `Qwen/Qwen3.5-{4B,7B,9B,14B,27B,35B-A3B,122B-A10B}` |
| **Qwen3.6** | `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B` |
| **Qwen3-Coder-Next** | `Qwen/Qwen3-Coder-Next` |
| **DeepSeek V3 / R1** | `deepseek-ai/DeepSeek-V3`, `DeepSeek-V3-0324`, `DeepSeek-R1` |

`_MTP_ALIASES` extends the registry to community quants:
`mlx-community/Qwen3.5-*-{4bit,8bit}`, `mlx-community/Qwen3.6-27B-{4bit,8bit,bf16}`,
`lmstudio-community/Qwen3-Coder-Next-MLX-4bit`, etc. Aliases auto-resolve to
the canonical repo for the draft-n lookup.

If your model isn't in the registry, MTPLX isn't offered — the backend's
`_select_engine` only routes through MTPLX when `has_mtp_heads(repo)`
returns true.

## How routing works

Inside `RuntimeController._select_engine`:

```python
if hint == "mlx":
    if (
        speculative_decoding
        and self.capabilities.mtplxAvailable
        and has_mtp_heads(canonical_repo or model_ref)
    ):
        return MtplxEngine(self.capabilities)
    return MLXWorkerEngine(self.capabilities)
```

Three conditions all have to hold for MTPLX to fire:

1. Speculative decoding is enabled (launch modal toggle or
   `--spec` on the CLI).
2. `mtplx` capability probe succeeded — the isolated venv at
   `~/.chaosengine/mtplx-venv/` exists and `mtplx --version` runs cleanly.
3. The model ref (or its alias) is in `MTP_MODEL_MAP`.

If any condition fails the backend falls back to `MLXWorkerEngine` and
records the reason in `runtimeNote` so you can see why MTPLX didn't fire.

## Install

The fastest path is **Setup → Install MTPLX** in the desktop app. The
installer:

1. Verifies you're on native arm64 Python 3.10+ (the mtplx fork won't
   build under Rosetta).
2. Creates `~/.chaosengine/mtplx-venv/` (removing any prior install).
3. Upgrades `pip` inside the venv.
4. Installs the `mtplx` package; its forked `mlx` is pulled in as a
   transitive dependency.
5. Writes a `~/.chaosengine/bin/mtplx.version` marker for the diagnostics
   tab to read.

Headless install:

```bash
./scripts/install-mtplx.sh
```

Or via the CLI:

```bash
./scripts/chaosengine-cli mtplx-install
./scripts/chaosengine-cli mtplx-status | jq '.installed, .version'
```

## Why an isolated venv

`mtplx` ships with a **fork of `mlx`** that can't co-exist with the
upstream `mlx` the rest of the backend uses for normal inference. The
isolated venv keeps mtplx's runtime out of the main `.venv/`'s site-
packages.

This is why MTPLX runs as a subprocess (`mtplx start --model <path>
--port N`) rather than an in-process module — the backend talks to it
over HTTP, the same pattern `LlamaCppEngine` uses for `llama-server`. The
adapter lives at
[`backend_service/inference/mtplx_engine.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/mtplx_engine.py).

## Fallback contract

`MtplxEngine.load_model` raises `RuntimeError` on any startup failure
(MTPLX venv missing, port already bound, model not loadable). The
`RuntimeController` catches that and falls back to the standard
`MLXWorkerEngine`. The fallback path records the reason in `runtimeNote`
so the UI surfaces it clearly.

## Routing diagnostics

Every chat / compare turn that runs through MTPLX includes:

- `runtimeNote` containing the literal token `"mtplx"`.
- `engine: "mtplx"` in the `/api/runtime` snapshot.
- The MTPLX subprocess PID is tracked by the runaway-guard so it's killed
  cleanly on backend shutdown.

If you toggled speculative decoding on but `runtimeNote` shows only
`"mlx"` and the model is in the registry, check that the MTPLX venv
exists and `mtplx-status` reports `installed: true`.

## Known limits

- **Apple Silicon only.** No Linux / Windows / CUDA build of the mtplx
  fork at the time of writing. On those platforms, use DFlash.
- **GGUF path is separate.** llama.cpp has a parallel MTP implementation
  (`--spec-type mtp` in PR #22673, draft as of 2026-05-10); see
  [FU-028 in CLAUDE.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/CLAUDE.md)
  for the upstream status.
- **Native cache only.** MTPLX doesn't currently support TurboQuant or
  TriAttention cache compression in the same run.

See also: [DFlash](dflash.md), [Cache strategies](cache-strategies.md).
