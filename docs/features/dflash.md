# DFlash + DDTree speculative decoding

**DFlash** is a draft-model based speculative decoder: a small fast model
proposes a block of tokens, the target model verifies the block in a single
forward pass, accepted tokens are committed instantly, and rejected tokens
fall back to standard autoregressive generation.

**DDTree** (Diffusion Draft Tree) extends DFlash with tree-structured
candidate exploration. Instead of verifying a single linear draft path,
DDTree builds a tree of top-k candidates using a max-probability heap and
verifies the whole tree in one forward pass with a tree-structured
attention mask. The longest verified path is accepted — higher acceptance
rates than linear DFlash at the cost of additional memory for the
attention mask.

Both produce 1.8 - 5× speedup with **zero quality loss** (the verifier
guarantees output is identical to standard generation).

## Backends

- **`dflash-mlx`** — Apple Silicon native, MLX-based draft + target. Pinned
  to commit `fada1eb` (HEAD as of 2026-05-10) which adds the Gemma 4 backend,
  v0.1.5 serving surface, live server metrics, prefix-cache survival test
  gate, branchless Metal kernels, and fused draft KV projections.
- **`dflash`** — Linux + CUDA, vLLM-based.

Both report through the same capability probe (`dflashAvailable`); install
the right one for your platform.

## Supported model registry

The canonical list lives in
[`dflash/__init__.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/dflash/__init__.py)
(`DRAFT_MODEL_MAP` + `_ALIASES`).

| Family | Canonical repo | Draft checkpoint |
|---|---|---|
| Qwen3 | `Qwen/Qwen3-{4B,8B}` | `z-lab/Qwen3-{4B,8B}-DFlash-b16` |
| Qwen3-Coder | `Qwen/Qwen3-Coder-{4B,8B,30B-A3B,Next}` | `z-lab/Qwen3-Coder-{*}-DFlash` |
| Qwen3.5 | `Qwen/Qwen3.5-{4B,7B,9B,14B,27B,35B-A3B,122B-A10B}` | `z-lab/Qwen3.5-{*}-DFlash` |
| Qwen3.6 | `Qwen/Qwen3.6-35B-A3B` | `z-lab/Qwen3.6-35B-A3B-DFlash` |
| Gemma 4 | `google/gemma-4-{31B-it,26B-A4B-it}` | `z-lab/gemma-4-{*}-DFlash` |
| LLaMA | `meta-llama/Llama-3.1-8B-Instruct` | `z-lab/Llama-3.1-8B-Instruct-DFlash` |
| gpt-oss | `gpt-oss/gpt-oss-{20B,120B}` | `z-lab/gpt-oss-{*}-DFlash` |
| MiniMax | `MiniMaxAI/MiniMax-M{2.5,2.7}` | `z-lab/MiniMax-M{*}-DFlash` |
| Kimi | `moonshotai/Kimi-K{2.5,2.6}` | `z-lab/Kimi-K{*}-DFlash` |

The lookup helpers apply fuzzy matching to handle quantised / community
variants — `mlx-community/Qwen3.5-14B-{4bit,8bit,bf16}` and
`lmstudio-community/Qwen3-Coder-Next-MLX-4bit` all resolve to the same
underlying draft.

If your model isn't in the registry, the launch modal hides the DFlash
toggle entirely (see FU-034 — don't surface options the user can't make
work).

## Configuring DDTree

When DFlash is enabled, the launch modal exposes a **Tree budget** slider
(0 - 64):

- **0** — linear DFlash. One candidate path, single verify per block.
- **1 - 8** — light tree exploration. Modest acceptance-rate boost, low
  memory overhead.
- **16 - 32** — heavier tree, higher acceptance, more memory.
- **64** — full upper bound. Significant memory; only worth it when
  acceptance rates at lower budgets feel constrained.

DDTree falls back to linear DFlash on any failure, and DFlash falls back
to standard generation.

## How routing works

`RuntimeController._select_engine` doesn't pick DFlash directly — DFlash
runs inside the MLX worker (`backend_service/mlx_worker_speculative.py`)
or the vLLM engine, gated on:

1. `speculative_decoding=true` on the load request.
2. DFlash backend probe succeeded (`dflashAvailable`).
3. Target model resolves to a draft in `DRAFT_MODEL_MAP` (via
   `model_resolution.resolve_dflash_target_ref` for fuzzy matching).
4. MTPLX did **not** fire — MTPLX takes priority when both apply (the
   `_select_engine` code routes the MLX hint to MTPLX before DFlash).

Failed conditions show up in `runtimeNote` so the UI can tell the user
exactly why DFlash didn't kick in.

## CLI

```bash
# Load with speculative decoding
./scripts/chaosengine-cli load Qwen/Qwen3.5-14B --spec

# Probe whether DFlash is available + which draft was resolved
./scripts/chaosengine-cli status | jq '.capabilities.dflashAvailable, .runtime.runtimeNote'
```

## Install

DFlash is **not bundled** — install it manually for your platform:

```bash
# Apple Silicon
.venv/bin/pip install dflash-mlx

# Linux + CUDA
.venv/bin/pip install dflash
```

Or use the Setup tab's "Install DFlash" action when it appears. The
backend's pin is enforced in two places:
[`pyproject.toml`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml)
and
[`scripts/stage-runtime.mjs`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/stage-runtime.mjs).
The pre-build check fails if these drift.

## See also

- [MTPLX](mtplx.md) — native MTP heads, takes priority over DFlash for
  Qwen3.5 / 3.6 / DeepSeek / Coder-Next.
- [Cache strategies](cache-strategies.md) — DFlash currently only works
  with native f16 cache.
