# Image Studio

Image Studio is the prompt-to-image workspace. It runs Stable Diffusion–class
models locally via Hugging Face Diffusers (on MPS or CUDA) or via the
stable-diffusion.cpp engine (cross-platform, lightweight) and exposes the
output through the Image Gallery.

## Supported models

| Model | Provider | Speed (M-series) | Default resolution |
|---|---|---|---|
| **FLUX.1 Schnell** | Black Forest Labs | ~4 s / 1024² | 1024 × 1024 |
| **FLUX.1 Dev** | Black Forest Labs | ~7 s / 1024² | 1024 × 1024 |
| **Stable Diffusion 3.5 Medium** | Stability AI | ~6 s / 1024² | 1024 × 1024 |
| **SD 3.5 Large Turbo** | Stability AI | ~2 s / 1024² | 1024 × 1024 |
| SDXL family (incl. SDXL-Turbo, SDXL-Lightning) | Various | ~2 – 8 s | 1024 × 1024 |
| SD 1.5 / 2.x family | Stability AI / community | ~2 s | 512 × 512 |
| Qwen-Image, Z-Image (+ Turbo) | Tongyi / community | ~3 – 8 s | 1024 × 1024 |

Discover and download new models from the **Image Discover** tab; installed
models surface in the **Image Models** library with a one-click **Generate**
shortcut into Image Studio.

## Composing a generation

The right rail collects every input the runtime needs:

- **Prompt** + optional **negative prompt**.
- **Aspect ratio / quality preset.** Square (1024²), portrait (768×1344),
  landscape (1344×768), wide (1536×864), or custom.
- **Steps**, **CFG**, **sampler**, **seed**.
- **CFG decay** (flow-match models only) — linear ramp from initial CFG to
  a 1.5 floor over the schedule.
- **Distill LoRA / transformer.** Auto-populated for Hyper-SD, FLUX
  Turbo-Alpha; you can also point at a Nunchaku SVDQuant transformer for
  CUDA paths.
- **Cache strategy.** FBCache (default, threshold 0.12), TeaCache,
  TaylorSeer, MagCache, PAB, FasterCache. Strategy availability filters
  to whatever the loaded model's pipeline supports.

The **Enhance** pill next to the Prompt label runs the active prompt
through a local Qwen2.5-0.5B-Instruct-4bit rewriter that anchors the
phrasing to the loaded DiT's training distribution (per-family system
prompts for FLUX / Wan / LTX / HunyuanVideo / SDXL / SD3). On non-Apple-
Silicon platforms it falls back to a deterministic template suffix.

## Live progress

While the pipeline runs:

- A **step counter** updates in real time from the diffusers
  `callback_on_step_end` hook.
- **Live thumbnails** decode every few steps through a swapped-in TAESD /
  TAEHV tiny VAE — see what's being generated, not just a progress bar.
  Image previews are capped at ~192 px to keep the polled
  `/api/images/progress` endpoint cheap.
- **Phase log** lists each major step: VAE load, transformer load, LoRA
  fuse, denoise, decode.

Cancel mid-run with the **Cancel** button — the pipeline aborts at the
next callback boundary.

## Completion

When the run finishes the output card surfaces the full metadata: model,
prompt, seed, steps, resolution, sampler, cache strategy, distill state,
CFG decay state, and wall time. From here:

- **Open** the file directly.
- **Reveal on disk** in your file manager.
- **Clone settings** — copies the entire payload back into Image Studio
  for a re-run, optionally with a fresh seed.
- **Save** — moves the artifact into the gallery's permanent store; cold
  outputs are pruned after a configurable retention period.

## Image Gallery

Every artifact lands in the gallery automatically. Filter by model, runtime,
frame size, or full-text search across prompts. Each card carries the
full generation manifest so you can reproduce a result exactly.

## CLI

```bash
# Browse the catalog
./scripts/chaosengine-cli image-catalog | jq '.entries[].repo'

# Download a model
./scripts/chaosengine-cli image-download "black-forest-labs/FLUX.1-schnell"

# Generate
./scripts/chaosengine-cli image-generate "a neon city skyline at night" \
    --model FLUX.1-schnell --steps 4 --width 1024 --height 1024 --seed 42

# Watch progress
./scripts/chaosengine-cli image-progress

# List outputs
./scripts/chaosengine-cli image-outputs | jq '.outputs[] | {id, prompt, model}'
```

See [CLI recipes](../cli/recipes.md) for batch-generation patterns.
