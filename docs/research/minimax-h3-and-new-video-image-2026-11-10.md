# MiniMax H3 and newer image and video models (scan of 2026-11-10)

Prompted by the owner: "any reason why we're not supporting MiniMax H3?" and "scan HF for other image and video models at this quality level that fit my hardware". Sources: the Hugging Face API and model cards, the MiniMax H3 licence text, the stable-diffusion.cpp `docs/` folder and releases. Nothing was downloaded or run. Sizes are from the repo file listings.

This replaces the MiniMax-H3 note in FU-082 and FU-125 ("498 GB, not practical on 64 GB, no runnable path"). That was true on 2026-10-01 and is out of date.

## Headline

1. **The size objection is gone.** The community now ships pruned, quantized and few-step builds. A pruned Q4_K_M diffusion transformer is 11.4 GB. stable-diffusion.cpp (a runtime we already ship) runs MiniMax H3 and MiniMax-H3-Image, with a doc page for each.
2. **The licence is the real blocker.** The MiniMax H3 Community License (and the H3-Image one, same wording) grants rights only in the "Applicable Territory", defined as worldwide **excluding the European Union, the United Kingdom, the Republic of Korea and the United States of America**. Use, display and distribution outside it, and use of the Outputs outside it, are not authorised (§II, §III, §V.4, Acceptable Use item 1). D-48 already skipped H3 for this reason. If the owner is in the UK or EU, the owner cannot lawfully run it for testing either. This is a reading of the text, not legal advice.
3. **There is also a second, official image model.** `MiniMaxAI/MiniMax-H3-Image` (2026-11-03) is a distilled 6-step text-to-image and editing model that makes about 16 megapixel images. Same territory exclusion.

## MiniMax H3 (video with stereo audio)

| Fact | Value |
|---|---|
| Official repo | `MiniMaxAI/MiniMax-H3`, 2026-07-28, no longer gated, 498.5 GB (two full variants) |
| Variants | FL2VA (text, first frame, last frame) and Ref2VA (up to 9 images, 3 videos, 3 audio clips) |
| Output | 4 to 15 s, 24 fps, 32 kHz stereo audio, 768p short side by default, 2K through a separate regenerate stage |
| Transformer | 33B dense, about 13B of it in AdaLN branches that are not needed for inference-only use (about 20B to load) |
| Text encoder | Qwen3-VL-32B (Apache 2.0), hidden states from layer 50 |
| Prompt rewriting | H3-Context-IR is hosted-only; a community Qwen3.8-27B rewriter exists (`lightx2v/MiniMax-H3-Prompt-Rewriter-Qwen3.8-27B-Ref2VA`) |
| Licence | MiniMax H3 Community License, territory exclusion above, 20 M USD revenue cap, "MiniMax H3" shown in commercial UIs, no use of Outputs to improve other models, safeguards duty for services offered to third parties |

Builds that matter (all derived works under the same licence):

| Repo | What | Size |
|---|---|---|
| `leejet/MiniMax-H3-GGUF` (author of stable-diffusion.cpp) | Q4_K_M full DiT 18.8 GB, **pruned DiT Q4_K_M 11.4 GB**, pruned Q2 6.7 GB, text encoder Q4_K_M 18.2 GB or Q2 13.1 GB | 98 GB repo |
| `unsloth/MiniMax-H3-GGUF` | pruned Q8_0 21.4 GB, Q6_K 16.6 GB, Q5_0 13.9 GB | 218 GB repo |
| `Comfy-Org/MiniMax-H3` | bf16 DiT 66.3 GB, pruned bf16 40.2 GB, int8 34 GB, text encoder bf16 51.5 GB / int8 27 GB, VAEs | 535 GB repo |
| `lightx2v/Minimax-h3-Turbo` | 4-step and 8-step distillation LoRAs (2 GB each), fp8 4-step 34 GB | 72 GB repo |
| `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` | pruned nvfp4 12.5 GB, pruned int4/int8 mix 15 to 16 GB, int8 21 GB (Blackwell GPUs) | 128 GB repo |
| `blackestk/MiniMax-H3-Ref2VA-MLX-pruned-Q4` | MLX 4-bit Ref2VA transformer, 12.6 GB (one person's port, unverified) | 12.6 GB |
| `VideoRebirthAI/hyperflow-h3-flash`, `TurboDiffusion/TurboH3-FL2VA-I2VA` | few-step (4 to 8) H3 derivatives | 88 to 188 GB repos |

The Hugging Face diffusers library has a MiniMax-H3 pipeline (open issues and PRs about VAE decode memory and multi-GPU sequence parallel), so a CUDA path through diffusers is likely. We did not test it.

### Fit by machine (weights only; activations at 768p and 15 s are not measured)

| Machine | Route | Verdict |
|---|---|---|
| RTX 4090 24 GB | stable-diffusion.cpp, pruned Q4_K_M DiT (11.4 GB) plus Q4 text encoder (18.2 GB) with `--offload-to-cpu`; Turbo LoRA to cut steps | Should run; needs enough system RAM for the encoder (assume 32 GB or more); speed unknown |
| M4 Max 64 GB | stable-diffusion.cpp on Metal, same files (about 36 GB of weights plus VAEs), or the MLX port | Plausible for short clips; unmeasured, activations may not fit at 768p x 15 s |
| Mac Studio 512 GB | bf16 pruned DiT (40 GB) or full DiT (66 GB) plus bf16 encoder (51.5 GB) | Fits with room to spare; compute-bound, speed unknown |
| DGX Spark 128 GB | pruned bf16 DiT 40 GB plus int8 encoder 27 GB, or nvfp4 12.5 GB (Blackwell) | Fits; the 273 GB/s memory is the limit, so slow |
| Multi-GPU box or cluster | diffusers sequence parallel (Ulysses) PRs are open | Not ready to count on |

## MiniMax-H3-Image

- Official `MiniMaxAI/MiniMax-H3-Image` (2026-11-03), 204 GB repo. H3 architecture without the audio layers, same Qwen3-VL-32B encoder, about 16 MP output (4096x4096 up to 6272x2688), distilled to 6 steps with no CFG, text-to-image plus editing with up to 16 reference images.
- stable-diffusion.cpp has `docs/minimax_h3_image.md`; the encoder is shared with the video model, so one 18 GB encoder download serves both.
- Community quants: `Abiray/MiniMax-H3-Image-GGUF` (53 GB repo of several quants), `nativ-community/MiniMax-H3-Image-4bit` (MLX, 54.5 GB repo). I did not read the per-quant sizes or measure quality.
- Same territory exclusion as the video model.

## Other models found, ranked for the owner's hardware (24 GB 4090 and 64 GB M4 Max)

I did not run any of these. Quality claims below are not verified; only licence, size and runtime are from the repos.

| Model | Licence | Size and runtime | Note |
|---|---|---|---|
| LTX-2.5 (already in the catalog through `dgrauet/ltx-2.5-mlx-q4`) | LTX-2 community licence | already wired; a 2-step distill LoRA `lvladikov/LTX-2.5-Stage1-Distill-2step-LoRA` appeared 2026-11-09 | Still the safest default on a 64 GB Mac |
| Vimoo-1.0-Lite (`Vimoo-AI/Vimoo-1.0-Lite`) | Apache-2.0 | about 9.5B (Vimoo-1.0 is 11.7B), released 2026-09-30, 720p 24 fps 5 s clips; official MLX weights `Vimoo-AI/Vimoo-1.0-Lite-MLX` 22.4 GB; no independent review, speed or quality data found | **Dropped by the owner 2026-11-11** (nothing measured; revisit if reviews appear) |
| Cosmos3-Edge (`nvidia/Cosmos3-Edge`) | OpenMDW 1.1 | 4B physical-AI world model, released 2026-07-20, video output capped at 480p; third-party MLX 8-bit port 5.0 GB | **Dropped by the owner 2026-11-11** (wrong fit for Video Studio; its text reasoner for camera-footage questions is a separate, unbuilt idea) |
| Motif-Video-2B (`Motif-Technologies/Motif-Video-2B`) | Apache-2.0 | 2B | Small; likely fast, likely below H3 quality |
| LongCat-Video (Meituan) | MIT | 13.6B, `mlx-community/LongCat-Video-q4` DiT about 9 GB; needs its own MLX package (FU-082) | Still unwired |
| Image families that stable-diffusion.cpp already documents but our catalog lacks | various | `krea2`, `ideogram4`, `ming_image`, `moss_image`, `qwen_image_2.1`, `hidream_o1_image`, `longcat_image`, `ernie_image`, `minit2i`, `sefi_image`, `sensenova_u1`, `boogu_image`, `lens`, `mage_flow`; video: `hunyuan_video`, `lingbot_video`, `wan` | Cheapest additions, since the runtime exists; each needs a licence and size check |

Our image catalog today: flux-2, flux-3, flux-dev, flux-fast, iris-3b, qwen-image, sd35-medium, sd35-turbo, sdxl-balanced, z-image.

## Options

1. **Do not add H3 (video or image) to the catalog.** Reason: the territory exclusion covers the US, EU and UK, which is most likely users, and the owner cannot test it. Keep this note as the record.
2. **Add H3 behind a territory gate.** Hidden unless the user ticks that they are outside the excluded territories, with the licence text shown and a "Powered by MiniMax H3" label. Needs a legal read of §III, §V.2 and §V.5 (the safeguards duty applies to products that let third parties generate) before building. Untestable by the owner.
3. **Hosted API lane.** Not checked: MiniMax's API terms may carry the same territory limits, and it breaks the local-first design.

Recommendation: option 1 now. Vimoo-1.0-Lite and Cosmos3-Edge were dropped by the owner (2026-11-11). If effort is spent later, add the stable-diffusion.cpp image families that are already documented (Krea 2, Ideogram 4, Qwen-Image 2.1, HiDream-O1) after a licence and size check each.

## Not checked

- Whether our pinned stable-diffusion.cpp build includes the H3 code (docs exist on master; check the pin before any wiring).
- Memory and speed of any of this on the M4 Max or the 4090.
- Output quality of anything listed; "at H3 quality" is not established for the other models.
- MiniMax's hosted API terms.
