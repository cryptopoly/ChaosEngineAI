# Upstream research — 2026-10-01 (image + video models)

Scan of the Hugging Face Hub, PyPI and upstream READMEs for the newest open
image and video models, and what each needs to run in ChaosEngineAI. Target
box: Apple Silicon, 64 GB. Tracked in the follow-ups table (FU-082, FU-096).

## Image

### Qwen-Image-2.1 (`Qwen/Qwen-Image-2.1`, 2026-09-14)

- 7B single-stream DiT (32 layers) with a Qwen3-VL-8B text encoder and a new
  64-channel VAE. 33 GB in bf16 (14.2 GB transformer, 17.5 GB encoder, 1.4 GB
  VAE). The first Qwen-Image was 20B and 57 GB; it is killed by the OS while
  loading on a 64 GB Mac.
- One model for text-to-image, instruction editing (up to 10 reference images,
  masks) and native transparent (RGBA) output. 2K output, 40 steps, no guidance.
- **Licence: Qwen Research Licence — non-commercial use only.** The first
  Qwen-Image was Apache 2.0.

How it runs:

| Lane | State | Notes |
|---|---|---|
| stable-diffusion.cpp | **Shipped** (FU-096) | GGUF 4.2 GB (Q4_K) + Qwen3-VL-8B Q4_K_M + VAE. Metal: ~13 s/step (525 s for 1024² / 40 steps). CUDA path is the same code, not run yet. |
| mflux 0.20 | **Shipped** (FU-096) | MLX-native, own venv. Upstream: ~1.5 s/step on an M5 Max. Measured on an M4 Max: 375 s for 1024² / 40 steps (~7.5 s/step), peak ~29 GB. |
| diffusers | Blocked | `QwenImage21Pipeline` exists only on diffusers git main; PyPI latest is 0.40.0 (2026-08-20). Needs `transformers>=5.17`. |

### Other current image models

- FLUX.2 klein 4B (Apache 2.0, in the catalog; 21 s for 1024² on MPS) and 9B
  (non-commercial).
- Z-Image / Z-Image-Turbo (Apache 2.0, 6B), ERNIE-Image (Apache 2.0, 8B).
- Krea 2 Turbo (2026-06, 12B, gated), Ideogram 4 (typography),
  Ming-Image-0.1-Design (2026-09, MIT, posters and RGBA), Lens (3.8B).

## Video

- **MiniMax-H3** (`MiniMaxAI/MiniMax-H3`, 2026-07-28): video with stereo audio,
  4–15 s, 768p with a 2K regenerate stage, 11 dialogue languages. 498 GB repo,
  Qwen3-VL-32B encoder, community licence. Its prompt-rewriting stage is
  hosted-only. Not practical on 64 GB; no runtime path here.
- **LTX-2.5** (`Lightricks/LTX-2.5`, 2026-07-23): 22B, gated, Gemma-4-12B text
  encoder, LTX-2.x community licence. A diffusers repo exists
  (`Lightricks/LTX-2.5-Diffusers`). No MLX conversion yet and mlx-video
  supports LTX-2 / 2.3 only (last upstream commit 2026-05-13).
- **Wan**: nothing new and open since Wan 2.2. Wan 2.5 / 2.6 / 2.7 / 3.0 have
  no real repo (placeholders only). Newest official: Wan2.2-Animate-2,
  Wan-Dancer-14B.
- **LongCat-Video** (Meituan, MIT, 13.6B): `mlx-community/LongCat-Video-{q4,q8,bf16}`
  since 2026-06; the q4 DiT is ~9 GB. Apple-native but needs its own MLX
  package.
- LTX-2.3 distilled stays the best working option on Apple Silicon.

## Findings from the same pass

- Wan 2.2 TI2V 5B renders over-saturated, burnt colours on MPS for both the
  GGUF and bf16 variants, at 832×480 and at 1280×704. A fp32 VAE changes
  nothing. Cause unknown; prefer LTX for video acceptance testing.
- The older sd.cpp FLUX / SD3 image variants pass no companion files
  (`--vae`, `--clip_l`, `--t5xxl`), so they likely fail; the new `sdcppAux`
  catalog field is the fix (FU-096 e).
