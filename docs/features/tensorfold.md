# TensorFold — Exact Speculative Decoding

**TensorFold** is an optional inference engine for Apple Silicon. It runs a
model family with hand-written kernels and drafts several tokens per step
(from the model's own MTP head and, for some families, a small draft model).
Every draft is **verified against the target model**, so a drafted reply is
byte-identical to one-token-at-a-time decoding: the speedup costs no quality.

TensorFold is the [`ashhart/TensorFold`](https://github.com/ashhart/TensorFold)
project (MIT, alpha). ChaosEngineAI runs it as a local subprocess and talks to
it over its OpenAI-compatible HTTP API, the same way it runs MTPLX and
`llama-server`.

## What it adds

TensorFold does two jobs, and they route differently:

- **An accelerator for models we already run.** For checkpoints TensorFold is
  tested with (for example Qwen3.8 27B and Gemma 4), turning on speculative
  decoding and having TensorFold installed routes the load through it instead of
  stock MLX. Leave the toggle off and the same model loads on stock MLX, so the
  checkbox is also your A/B switch.
- **A runtime for models stock MLX cannot load.** Some families (Qwen3.8 Flash
  Next, GLM-5.3-Flash, DeepSeek-V4-Flash, Ternary Bonsai 2) use architectures
  that `mlx-lm` has no implementation of. They always run on TensorFold; if it
  is not installed the load fails with an install prompt instead of a stack
  trace.

## Supported model registry

The canonical list lives in
[`backend_service/inference/_tensorfold.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/_tensorfold.py)
(`TENSORFOLD_FAMILIES`). It records what TensorFold's own `families/*` modules
declare, at the pinned version.

| Family | Tested checkpoint | Routing | Memory |
|---|---|---|---|
| Qwen3.8 dense | `Vontra/Qwen3.8-27B-MLX-4bit` | accelerator | — |
| Nemotron 3.5 Lightning | `Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | accelerator | — |
| Gemma 4 | `mlx-community/gemma-4-26b-a4b-it-4bit` | accelerator | — |
| Qwen3.8 Flash Next | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` | TensorFold only | 128 GB |
| GLM-5.3-Flash | `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` | TensorFold only | 256 GB |
| DeepSeek-V4-Flash | `mlx-community/DeepSeek-V4-Flash-4bit` | TensorFold only | 256 GB |
| Ternary Bonsai 2 | `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` | TensorFold only | 32 GB |

Matching is by **exact repo id** (or the `model_type` in the snapshot's
`config.json` for the TensorFold-only families), not the fuzzy name matching
DFlash and MTPLX use. TensorFold reads specific weight layouts, so a different
community conversion of the same model will not work even when the name is
close.

Qwen3.6-35B-A3B is not listed: TensorFold ships only a CUDA engine for it.

## How routing works

Inside `RuntimeController._select_tensorfold` (called from `_select_engine`):

1. **Explicit backend.** `backend == "tensorfold"` always selects it (and
   refuses GGUF files or a missing install with a clear message).
2. **TensorFold-only family.** The model's family needs TensorFold →
   `TensorFoldEngine` if installed, otherwise an error that says to install it
   and how much memory the checkpoint needs.
3. **Tested checkpoint.** Speculative decoding requested **and** TensorFold
   installed → `TensorFoldEngine`. It outranks MTPLX and DFlash when several
   could serve the model.
4. **Anything else** falls through to the usual MLX / MTPLX / llama.cpp rules.

TensorFold keeps its own KV caches (it rolls them back when a draft is
rejected), so the launch modal locks the cache strategy to Native while it runs
and shows "TensorFold manages its own KV cache".

Loading a TensorFold model unloads everything else first: it sizes its memory
budget from physical RAM and its models run to 100+ GB.

## Install

The fastest path is the **TensorFold** control in the launch settings, or
**Setup → Install TensorFold**. The installer:

1. Verifies native arm64 Python 3.11–3.13 (the `mlx` wheels it pins have no
   3.14 build yet, and Rosetta Python will not work).
2. Creates `~/.chaosengine/tensorfold-venv/`.
3. Installs TensorFold at the pinned commit; it pulls `mlx` and `mlx-lm` at the
   versions it requires into that venv.
4. Checks that `tensorfold`, `mlx` and `mlx-lm` import and that the
   `tensorfold` command runs, then writes `~/.chaosengine/bin/tensorfold.version`
   (its presence is what marks the install usable).

Headless install:

```bash
./scripts/install-tensorfold.sh
# or, against a running backend:
./scripts/chaosengine-cli tensorfold-install --wait
./scripts/chaosengine-cli tensorfold-status | jq '.installed, .version'
```

Set `TENSORFOLD_REF=<commit|tag>` to try a newer build, or `TENSORFOLD_SOURCE`
to install from a local checkout or an internal mirror.

## Why an isolated venv

TensorFold pins `mlx >=0.32.2,<0.32.4` and `mlx-lm >=0.31.3,<0.32`, which can
differ from what the app runs. The engine starts the server from the venv's own
interpreter with a scrubbed environment: the packaged desktop app exports
`PYTHONHOME`, `PYTHONPATH` and `DYLD_*` for its embedded runtime, which would
otherwise make the venv import the app's MLX.

## How the engine behaves

`TensorFoldEngine`
([`tensorfold_engine.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/inference/tensorfold_engine.py))
spawns `tensorfold serve <model dir>` on a local port and proxies
`/v1/chat/completions`.

- **Local directory, offline.** The server is started on the model's snapshot
  directory with `HF_HUB_OFFLINE=1`. Downloads stay in Discover and the Library,
  with their progress and the relocated-cache setting. The one thing the engine
  fetches itself is the small optional draft model.
- **In-memory prefix snapshots.** `--snapshot-dir none`: the default would write
  multi-GB snapshots under `~/.cache` on every model swap.
- **Reasoning and tool calls** stream through as separate channels, so a
  thinking model shows its reasoning and a tool call is executed rather than
  printed.
- **Metrics.** Throughput, time to first token and the average accepted draft
  tokens per step are surfaced in the routing badge ("Exact drafts",
  "N avg accepted").
- **Reply-budget refusals.** When the context leaves too little room for the
  requested reply, TensorFold refuses; the engine turns that into a message that
  says how many tokens fit.

## Fallback contract

`TensorFoldEngine.load_model` raises `RuntimeError` on any startup failure
(missing venv, port in use, weights incomplete, a model the kernels reject).

- For checkpoints stock MLX can also load, the controller falls back to
  `MLXWorkerEngine` and records `TensorFold startup failed (...); using
  standard MLX.` in `runtimeNote`.
- For TensorFold-only families there is nothing to fall back to, so the error
  reaches the user.

## Known limits

- **Apple Silicon only.** The Mac engine is the only one wired up here.
- **Alpha software.** The pin is an exact commit; the registry and the
  installer move together when it is bumped.
- **No vision or structured output yet.** TensorFold's `--vision` and grammar
  extras are not installed, so image input and JSON-schema output are not
  offered on this lane.
- **Native cache only**, as above.
- **Big models need big Macs.** The TensorFold-only families list their minimum
  memory, and the launch modal warns when the Mac has less.

See also: [MTPLX](mtplx.md), [DFlash](dflash.md),
[Cache strategies](cache-strategies.md).
