# Generation times on an Apple M4 Max: ChaosEngineAI against the field

Internal research, 2026-11-07. This doc names competing apps on purpose, because the comparison is the point
(exception to the CLAUDE.md editorial rule, which still applies to code, UI and other docs).

**Test machine (ours).** `system_profiler`: MacBook Pro `Mac16,6`, Apple M4 Max, 16-core CPU (12P + 4E),
**40-core GPU**, 64 GB unified memory. The 40-core part has about 546 GB/s memory bandwidth (the 32-core part
has about 410 GB/s). The hub says the same in `docs/hub/overview.md` ("M4 Max with 64 GB").
A 128 GB M4 Max has the same GPU, so the speed is the same. The difference is that big models (Qwen-Image,
ACE-Step XL, the bf16 LTX rows) fit without memory pressure.

**Labels.** MEASURED = a published or repo run (source given). ESTIMATED = scaled by us (method given).
"Wall" means the time from submit to file, warm worker, unless the row says cold.

---

## (a) Executive summary

| Modality | Verdict | One-line why |
|---|---|---|
| **Image** | **In line** with the fastest Mac runtimes | We run mflux (the reference MLX implementation) directly. Per-step times match scaled community numbers. The Mac-native app (Draw Things, Metal FlashAttention) may be up to about 20 to 50 % faster per step on some models (its own claim, not verified on an M4 Max). An RTX 4090 is 5 to 15x faster for the same model, and that gap is the hardware. |
| **Video, LTX (default path)** | **In line**, maybe slightly ahead | ltx-2-mlx q4 distilled: 4 s at 768x512 in about 128 to 138 s. The only M4 Max ComfyUI-MLX figure is 88 s for 2 s at 704x480. An RTX 5090 does a 720p 4 s clip in about 25 s, roughly 10x faster. |
| **Video, Wan 2.2 and the legacy bf16 LTX rows** | **Slower** | Wan runs through diffusers on MPS (293 s for 33 frames at 832x480, 20 steps). MLX ports and distilled Wan builds are several times faster. The bf16 mlx-video LTX rows were measured at 1130 s for 33 frames on 64 GB, about 10x slower than published MLX fp16 timings on a 128 GB M4 Max (probably memory pressure). |
| **Music** | **Faster** than the published Mac numbers, and close to an RTX 3090 | ACE-Step 1.5 turbo on MLX DiT + MLX VAE with the planner LM off: a 60 s song in 5.1 to 5.9 s warm (about 10x real time). The only other Mac figure is about 3.4x real time on an M1 Max with the LM on and a PyTorch DiT. Upstream says under 10 s per full song on an RTX 3090. |
| **Voice (TTS)** | **In line or faster on speed, behind on choice** | Kokoro warm: 0.17 s for 6 s of speech (about 35x real time). A published M4 Max mlx-audio run gives 22x, and a 4060 Ti gives 35 to 50x. We ship only Kokoro, while competitors ship voice cloning (Qwen3-TTS, Chatterbox, F5), which runs at roughly 1.5 to 2x real time on Macs. |

**Five numbers worth remembering**

1. **LTX-2.5 q4, 4 s 768x512 clip with sound: 128 s on our M4 Max** (MEASURED, `docs/hub/progress/runs-2026-10-29.md`). For context, an RTX 5090 does a 720p 4 s clip in about 25 s.
2. **Qwen-Image-2.1 bf16, 1024 px, 40 steps: 258 to 271 s (about 6.6 s/step), 46 to 50 GB peak** (MEASURED, `docs/hub/progress/settings-rec.md`). An RTX 4090 takes 18.7 s (SGLang cookbook), about 14x faster.
3. **ACE-Step 1.5 turbo, 60 s song: 5.9 s warm, 25 s extra on a cold start** (MEASURED, `backend_service/catalog/music_models.py`).
4. **Iris-3B Turbo, 1024 px, 4 steps: 12.6 s warm** (MEASURED). On the same chip class, FLUX.2 klein 4B takes 10.9 s at 1024 px and 4 steps through Core ML (HF card, M4 Max 128 GB), and FLUX.1 schnell takes about 19 s (2 steps, cold) in mflux.
5. **Kokoro: 0.17 s for 6 s of speech** (MEASURED, FU-117), against 22x real time in the vllm-mlx M4 Max benchmark.

---

## (b) Tables per modality

Column "Ours" = ChaosEngineAI on the machine above. "M4 Max field" = other runtimes and apps on an M4 Max (or scaled).
"CUDA" = context only. It is a different class of hardware, not a target.

### Image (1024x1024 unless stated)

| Model | Steps | Ours (M4 Max 40c, 64 GB) | M4 Max field | CUDA reference |
|---|---|---|---|---|
| Iris-3B Turbo (mflux 0.24) | 4 | **12.6 s warm, 14.6 s first run; 8.4 s at 768 px; 12 to 13 GB** (MEASURED, settings-rec) | No other runtime ships Iris. Closest class: FLUX.2 klein 4B **10.9 s** fp16 at 4 steps, Core ML, M4 Max 128 GB (MEASURED, [HF card](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI)). FLUX.2 klein 4B in mflux on an M1 Max: 21.4 s pure, 31.7 s wall (MEASURED, [lilting](https://lilting.ch/en/articles/flux2-klein-4b-mflux-iris-m1-max)); **about 9 to 11 s on an M4 Max** (ESTIMATED: about 2 to 2.4x for 25 % more cores, a newer GPU and 37 % more bandwidth) | n/a |
| FLUX 3 Image Klein (mflux) | 8 (Studio default) | **40 s; 15 GB** (MEASURED) | No published FLUX 3 timings. FLUX.2 klein 9B: 85 s on an M4 Max per [lilting](https://lilting.ch/en/articles/flux2-klein-apple-silicon) (unverified: no steps or app given). Draw Things on an M5 Max: klein 4B about 10 s, 9B about 15 s ([note.com](https://note.com/tamilab/n/n89032ac3645d?hl=en); M5 has neural accelerators, so it is not comparable) | 4090: FLUX.2 klein 9B 12 to 18 s (lilting, secondary) |
| FLUX 3 Image Klein | 16 (maker default) | **70 to 78 s** (MEASURED) | as above | |
| FLUX 3 Image Klein, edit with 2 references, 1024x576 | 8 | **81 to 91 s per still** (MEASURED, lighthouse `run.log`) | n/a | |
| FLUX 3 Image Klein, 3072 px | 8 | **110 s, 41 GB** (MEASURED, hub status) | n/a | |
| Qwen-Image-2.1 Turbo (mflux) | 8 | **72 to 75 s; 46 to 50 GB** (MEASURED) | n/a | |
| Qwen-Image-2.1 bf16 (mflux) | 40 | **258 to 271 s (6.6 s/step); 46 to 50 GB** (MEASURED) | mflux bf16 on an M5 Max: 78 s, 1.5 s/step, 46 GB ([modelfit](https://modelfit.io/blog/qwen-image-2-1-mac-apple-silicon/); M5 neural accelerators). ComfyUI + GGUF on an M4 Pro 48 GB: about 13 s/step (same source), so **about 6 to 7 s/step on an M4 Max** (ESTIMATED: 2x GPU cores, 2x bandwidth) | **4090: 18.7 s**; RTX PRO 6000: 8.0 s; DGX Spark: 35.4 s (SGLang cookbook via modelfit, 40 steps, MEASURED) |
| FLUX.1 schnell (not timed by us) | 2 | catalog only | **mflux: about 19 s**, including cold start, M4 Max 128 GB (MEASURED, [mflux README](https://pypi.org/project/mflux/0.6.2/)) | |
| FLUX.1 dev (not timed by us) | 20 | catalog only (diffusers, GGUF, sd.cpp, mflux rows) | Draw Things, M4 Max 40c: **about 170 s** (Reddit r/drawthingsapp, March 2025, anecdotal). ComfyUI fp8, M4 Max 48 GB: 45 to 70 s ([bitdoze](https://www.bitdoze.com/ai-images-mac/), unverified). Draw Things, M4 Pro: about 50 s, Q6_K ([heyuan110](https://www.heyuan110.com/posts/ai/2026-02-15-mac-mini-local-image-generation/)). The sources disagree by 3x | 4090: 11 to 18 s; 5090: 5.5 to 9 s (several secondary sources, fp8 vs bf16) |
| Z-Image Turbo (not timed by us) | 8 to 9 | catalog only | ComfyUI on an M4 Pro: 60 to 80 s (heyuan110). Draw Things says it is "up to 54 % faster than other implementations" ([Draw Things](https://releases.drawthings.ai/p/quantify-z-image-turbo-efficiency)) | 4090: about 2.3 s (secondary) |

### Video

| Model / engine | Size, frames, steps | Ours | M4 Max field | CUDA reference |
|---|---|---|---|---|
| LTX-2.3 distilled q4 (ltx-2-mlx) | 768x512, 33 f (1.4 s), 8+3 passes | **55 to 57 s, 14 GB** (MEASURED, settings-rec) | ComfyUI LTX-2 MLX node, q4 distilled, 704x480, 49 f, M4 Max 64 GB: **88 s** (MEASURED by its author, [comfy.icu](https://comfy.icu/node/LTX2MLXModelLoader)); its two-stage modes take about 16 min | |
| LTX-2.3 q4 | 768x512, 97 f (4 s) | **135 to 138 s, 15 GB** (MEASURED) | ESTIMATED from the 88 s above, scaled by frames: about 150 to 175 s, so we are level or a little faster | 5090: about 25 s for 720p, 4 s (NVIDIA guide via [runaihome](https://runaihome.com/blog/ltx-2-5-local-ai-video-hardware-guide-2026/)) |
| LTX-2.3 q4 | 1280x704, 121 f (5 s) | **363 s, 18 GB** (MEASURED, runs-2026-10-29) | MLX q4, video only, 576x1024, 121 f, M4 Max 128 GB: **154 s denoise only** (stage 1 51 s + stage 2 101 s), against 264 s for PyTorch MPS bf16 (MEASURED, [HF card](https://huggingface.co/gajesh/ltx-2.3-mlx-q4)). Scaled to our 1.53x pixel count plus encoder, decode and audio: about 300 to 350 s (ESTIMATED). In line | 3090 int8 distilled, 5 s: about 2.5 min (community, via runaihome) |
| LTX-2.3 q4 | 768x512, 257 f (10.7 s) | **352 s, 18 GB** (MEASURED) | n/a | 5090: an 8 s 720p clip spills out of VRAM and takes about 3 min |
| LTX-2.5 distilled q4 (ltx-2-mlx) | 768x512, 97 f | **128 s, 17 GB** (MEASURED, runs-2026-10-29) | ltx-2-mlx README: the only timing is an M1 Max (2.5 q8, 576x1024x241, I2V), 701 s, or 563 s with fp16 compute ([repo](https://github.com/dgrauet/ltx-2-mlx)) | 5090: 960x544 x 121 f in 16.9 s (SGLang LTX-2.5 cookbook, via search) |
| LTX-2.5 q4, image to video | 768x448, 97 f | **120 to 145 s per clip** (MEASURED, lighthouse `run.log`, 8 clips) | n/a | |
| LTX-2.3 distilled bf16 (mlx-video, legacy row) | 768x512, 33 f | **1130 s (34 s/frame), 40 GB** (MEASURED 2026-10-21, catalog comment) | MLX fp16, M4 Max **128 GB**, 576x1024 x 121 f: 152 s denoise (MEASURED, [HF card](https://huggingface.co/gajesh/ltx-2.3-mlx-fp16)). **Our legacy row is about 10x slower**, probably because a 40 GB footprint on a 64 GB Mac pages memory | |
| Wan 2.2 TI2V 5B (diffusers, MPS) | 832x480, 33 f, 20 steps | **293 s** (393 s at the old CFG of 7.5); 9 f = 182 s (MEASURED, FU-122) | FastVideo FastMetal, distilled 5B, INT8, 3 steps, 1280x704 x 81 f, M4 Max 36 GB: **151 s** end to end, or 47 s in fast mode ([haoailab](https://haoailab.com/blogs/fastmetal/)). mlx-video, undistilled, 40 steps, 1280x704 x 41 f, M3 Max 64 GB: 17 min 43 s (via [modelfit](https://modelfit.io/video/wan-2-2-5b/)). Scaled to our size, steps and an M4 Max: about 120 to 180 s (ESTIMATED: pixel-frames 2.8x, steps 2x, M4 Max about 1.25x faster than M3 Max). **We are probably 1.5 to 2x slower than mlx-video** | 4090: a 5 s 720p clip in about 9 min at 50 steps (Alibaba's claim; one test died at VAE decode on 24 GB) |
| Wan 2.1 1.3B / 14B (diffusers) | not timed by us | catalog only | FastMetal distilled, M4 Max: 1.3B 480p x 81 f = 110 s; 14B 480p x 81 f = 602 s (haoailab) | |

### Music

| Model | Length | Ours (warm unless stated) | M4 Max field | CUDA reference |
|---|---|---|---|---|
| ACE-Step 1.5 turbo 2B (MLX DiT + MLX VAE, 8 steps, LM off) | 30 s | **2.8 to 3.6 s**; cold 11 to 14.6 s | No M4 Max figure published by anyone else | Upstream `BENCHMARK.md` example: 30 s in 3.21 s with LM off (hardware not stated) |
| same | 60 s (song with lyrics) | **5.1 to 5.9 s, 22.9 GB peak** | M1 Max 64 GB, **LM 0.6B on (MLX) + PyTorch MPS DiT**: 100 s song in about 29 s per song (MEASURED, [dev.to](https://dev.to/bigkijimon/100-free-fully-local-ai-music-generation-on-an-m1-mac-ace-step-15-two-100-second-vocal-songs-3h1b)). Scaled to an M4 Max: about 12 to 15 s per 100 s (ESTIMATED: about 2x GPU). Ours, LM off, would be about 9 to 10 s | **3090: under 10 s per full song; A100: under 2 s** (upstream README) |
| same | 180 / 300 / 600 s | **18.1 / 33.5 / 82.4 s** | n/a | |
| ACE-Step 1.5 XL turbo 4B (fp32 on MPS) | 30 / 60 s | **5.4 / 12.4 s, 47 GB peak** | n/a | XL card: 20 GB or more without offload |
| Cold start (first track) | | **about +25 s** (imports 6 s, load 4 to 6 s, MLX compile about 9 s) | n/a | |

### Voice

| Model | Task | Ours | M4 Max field | CUDA reference |
|---|---|---|---|---|
| Kokoro 82M (mlx-audio lane) | TTS | **0.17 s for 6 s of speech, about 35x real time** (MEASURED, FU-117). Lighthouse voiceover, 9.5 s of audio: 5 s wall, including the queue and worker (run.log) | vllm-mlx (mlx-audio), M4 Max 128 GB: **22x RT bf16, 20x 4-bit, load 0.4 to 0.8 s** (MEASURED, [vllm-mlx](https://vllm-mlx.is-a.dev/benchmarks/audio/)) | Kokoro-FastAPI on a 4060 Ti: 35 to 50x RT, about 300 ms to first audio (README) |
| Kokoro (kokoro-onnx, portable lane, same Mac) | TTS | **0.47 s** for the same line (MEASURED, FU-117) | n/a | |
| Qwen3-TTS 0.6B (not shipped) | TTS / cloning | n/a | Swift MLX port on an M2 Max: RTF 0.55, so about 1.8x faster than real time ([soniqo](https://soniqo.audio/guides/speak)). mlx-audio: RTF about 0.7, chip not stated. **About RTF 0.4 to 0.5 on an M4 Max** (ESTIMATED: 37 % more bandwidth than an M2 Max for a small autoregressive model) | |
| Chatterbox (not shipped) | TTS / cloning | n/a | chatterbox-mlx on a base M4 32 GB: "RTF 0.65 to 0.74", 62.7 s for 94 words. The RTF convention is ambiguous ([PyPI](https://pypi.org/project/chatterbox-mlx/)) | Chatterbox Turbo about 6x RT on consumer GPUs (blog) |
| F5-TTS (not shipped) | TTS / cloning | n/a | f5-tts-mlx: about 4 s per sample on an M3 Max (README, length not stated) | |
| Parakeet TDT 0.6B v3 (STT, context) | STT | **0.09 s for a 6 s clip** (MEASURED, FU-117) | vllm-mlx on an M1 Max: about 54x RT | |

---

## (c) What dominates our time, and where we lose

**Image (mflux).**
- **Denoising steps dominate.** Iris and FLUX 3 Klein spend about 90 % of the wall time in the transformer, at about 3 s/step (Iris, 4 steps) and about 5 s/step (FLUX 3 Klein). Load adds about 2 s on the first run (14.6 s against 12.6 s). The text encoder and VAE are small.
- **Picking fewer steps is our biggest measured win.** FLUX 3 Klein starts at 8 steps instead of 16 (40 s against 70 s, same look on 3 prompts).
- **Qwen-Image-2.1 is limited by memory more than compute.** It peaks at 46 to 50 GB on a 64 GB Mac. The bf16 row is the only one measured; the 8-bit row is not timed. On Apple GPUs, int8 weights save memory but do not speed up compute (the Core ML klein card shows fp16 10.9 s against int8 11.6 s).
- **Where we may lose:**
  - Draw Things uses its own Metal FlashAttention kernels. It claims about 20 % over ComfyUI and up to 54 % over "other implementations" for Z-Image. Nobody has published a head-to-head on an M4 Max, and its M4 Max FLUX.1 dev anecdote (170 s) is slower than ComfyUI reports, so the claim is unproven for our models.
  - Core ML (Apple's `coreai-models`) runs FLUX.2 klein 4B at 4 steps in 10.9 s, a little faster than our Iris class at the same step count.
  - M5 Macs are 3 to 4.6x faster than M4 on image work in both mflux and Draw Things, because of the neural accelerators. A new Mac will beat any software gain.
- **Not timed by us:** FLUX.1, SD3.5, SDXL and Z-Image through diffusers, and the sd.cpp rows. These are the places where an MPS-diffusers path could be slower than mflux or Draw Things.

**Video.**
- **LTX via ltx-2-mlx (default) is good.** The time is in the two-stage distilled sampler. On the HF card's M4 Max run, stage 2 (3 steps at full resolution) is about 66 % of denoising. Text encoding (Gemma), VAE decode and the audio branch come on top.
  - The q4 transformer keeps the footprint at 14 to 18 GB, so a 64 GB Mac never pages memory.
  - Cold model plus encoder load is about 24 to 41 s (ComfyUI node author's figure) and lands on the first clip.
- **Where we lose:**
  1. **The legacy mlx-video bf16 rows** (1130 s for 33 frames, 40 GB) are about 10x slower than published MLX fp16 runs on a 128 GB M4 Max. Hide them on 64 GB Macs, or point them at the q4 pack.
  2. **Wan 2.x through diffusers on MPS:** 293 s for 2 s at 480p and 20 steps. MLX ports are probably 1.5 to 2x faster. Distilled 3-step builds (FastMetal) do a 5 s 720p clip in 151 s, or 47 s in fast mode.
  3. **Against CUDA:** a 5090 is about 10x faster on LTX at the same length. Expect that gap; no software change closes it.

**Music.**
- **The time splits roughly evenly between DiT diffusion (8 steps) and VAE decode.** The spike measured VAE at 0.06 s per audio second. Two settings mattered: VAE chunk 256 plus a 2 GB MLX cache limit took a 60 s track from 9.6 s and 66 GB to 5.2 s and 27 GB.
- **The cold start (about 25 s) is the biggest share of a one-off track.** It is made of Python imports, weight load and MLX compile. Idle stop frees 15.6 GB, but the next track pays the cost again.
- **We lose nothing on speed against other Mac users**, who mostly run the upstream Gradio or API server with the planner LM on (+2 to 3 s per track, +22.7 s load) and sometimes fall back to PyTorch. The XL model runs in fp32 on MPS (47 GB). bf16 would halve memory and might speed it up (not tried).

**Voice.**
- **Generation is trivial.** Worker start and model load dominate the first call. The 9.5 s voiceover took 5 s wall through the queue, while warm generation is about 0.3 s.
- **We lose on features, not speed.** Competitors ship cloning TTS (Qwen3-TTS, Chatterbox, F5) at about 1.5 to 2.5x real time on Macs. That is roughly 15x slower than Kokoro, so adding them changes the user's wait from "instant" to "about half the clip length".

---

## (d) Caveats

- **Our numbers come from one machine,** a MacBook Pro (`Mac16,6`) with 64 GB, often with other jobs running. The music spike notes ±20 %. Long video runs on a laptop can throttle, while a Mac Studio M4 Max holds clocks better. Nothing here was re-run for this doc (as instructed). All our numbers come from the repo's progress pages, the FU rows, catalog comments and demo logs, dated 2026-10-21 to 2026-11-07.
- **Our timings are mostly wall time through the app** (HTTP, queue, saving the file). Published Mac numbers are often "denoise only" (the gajesh LTX cards) or include a cold start (the mflux README). Rows that mix these are marked.
- **Very few M4 Max numbers exist anywhere.** Several guides say so outright. Many published "M4 Max" figures are unverified blog tables with no steps, precision or app. The FLUX.1 dev spread (45 to 170 s) shows how unreliable they are. Treat "in line" as ±30 %.
- **Scaling rules used:** compute-bound work scales with GPU cores times clock, and generation-over-generation gains (M1 to M4 about 2 to 2.4x). Small autoregressive models (TTS) scale with bandwidth (M2 Max 400 GB/s to M4 Max 546 GB/s, 1.37x). M5 numbers are not scaled down; they are context only, because the neural accelerators change the ratio per model.
- **The CUDA numbers are secondary sources** with mixed precision (fp8 / bf16 / int8) and mixed settings. They are context, not a target.
- **Model versions in this repo** (FLUX 3 Image Klein, Iris-3B, Qwen-Image-2.1, LTX-2.5, mflux 0.24) are newer than most published benchmarks. Several comparisons use the nearest older model of the same size class.

### Could not source

- A controlled FLUX.1 dev or schnell timing on an M4 Max in Draw Things or ComfyUI (only anecdotes and unverified blog tables).
- Any third-party timing for FLUX 3 Image Klein, Iris-3B or Qwen-Image-2.1 on an M4 Max.
- Undistilled Wan 2.2 5B on an M4 Max in any runtime (only an M3 Max mlx-video figure and the distilled FastMetal builds).
- ACE-Step 1.5 on an M4 Max by anyone else, and a 4090 or 5090 timing for ACE-Step 1.5 (upstream gives only the 3090 and A100).
- Qwen3-TTS, Chatterbox or F5 on an M4 Max specifically.
- LTX-2.5 at 768x512 on a 4090 or 5090 (only 720p and 960x544 figures).

### Follow-ups this suggests (not done)

1. Time FLUX.2 klein 4B (mflux) and Z-Image Turbo (diffusers) on this Mac. Those two let us compare directly with the Core ML 10.9 s figure and the Draw Things Z-Image claim.
2. Retime or hide the bf16 mlx-video LTX rows on 64 GB Macs (1130 s against about 130 s on the q4 path).
3. Price an MLX Wan path (mlx-video or a distilled 3-step build) against diffusers MPS, expected 1.5 to 3x faster.
4. Try ACE-Step XL in bf16 on MPS (47 GB peak today).

## Sources

- Repo: `docs/hub/progress/settings-rec.md`, `docs/hub/progress/runs-2026-10-29.md`, `docs/hub/status.md`, `FOLLOW_UPS.md` (FU-117, FU-122), `docs/plans/music-spike.md`, `docs/features/music.md`, `backend_service/catalog/{image,video,music}_models.py`, `~/ChaosEngineAI-demos/lighthouse-keeper/run.log`, `docs/examples/lighthouse-keeper-demo/`.
- mflux README speed table: https://pypi.org/project/mflux/0.6.2/
- FLUX.2 klein 4B Core AI (M4 Max): https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI
- FLUX.2 klein 4B mflux / iris.c (M1 Max): https://lilting.ch/en/articles/flux2-klein-4b-mflux-iris-m1-max
- FLUX.2 klein 9B Mac table (unverified): https://lilting.ch/en/articles/flux2-klein-apple-silicon
- Draw Things on M5 Max (FLUX.2): https://note.com/tamilab/n/n89032ac3645d?hl=en
- Draw Things Z-Image claims: https://releases.drawthings.ai/p/quantify-z-image-turbo-efficiency ; MFA v2.5 (M5 vs M4): https://releases.drawthings.ai/p/metal-flashattention-v25-w-neural
- Mac mini M4 Pro, Draw Things vs ComfyUI: https://www.heyuan110.com/posts/ai/2026-02-15-mac-mini-local-image-generation/
- Flux on Mac guide (unverified M4 Max ranges): https://www.bitdoze.com/ai-images-mac/
- Qwen-Image-2.1 on Mac + SGLang CUDA table: https://modelfit.io/blog/qwen-image-2-1-mac-apple-silicon/
- FLUX.1 dev 4090 / 5090: https://dev.to/jovan_chan_9500711396d4e6/comfyui-nvfp4-in-2026-3x-faster-image-generation-on-rtx-50-series-and-the-right-format-for-rtx-2i8b , https://insiderllm.com/guides/rtx-5090-local-ai-worth-it/
- LTX-2.3 MLX q4 / fp16 (M4 Max 128 GB): https://huggingface.co/gajesh/ltx-2.3-mlx-q4 , https://huggingface.co/gajesh/ltx-2.3-mlx-fp16
- ComfyUI LTX-2 MLX node (M4 Max 64 GB): https://comfy.icu/node/LTX2MLXModelLoader
- ltx-2-mlx: https://github.com/dgrauet/ltx-2-mlx
- LTX-2.5 hardware guide (5090 and 3090 figures): https://runaihome.com/blog/ltx-2-5-local-ai-video-hardware-guide-2026/
- FastMetal Wan (M4 Max): https://haoailab.com/blogs/fastmetal/ ; Wan 2.2 5B on Mac: https://modelfit.io/video/wan-2-2-5b/
- Wan 2.2 on a 4090: https://computingforgeeks.com/run-wan-video-generation-locally/ , https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- ACE-Step 1.5: https://github.com/ace-step/ACE-Step-1.5 ; M1 Max run: https://dev.to/bigkijimon/100-free-fully-local-ai-music-generation-on-an-m1-mac-ace-step-15-two-100-second-vocal-songs-3h1b
- vllm-mlx audio benchmarks (Kokoro, Whisper on M4 Max): https://vllm-mlx.is-a.dev/benchmarks/audio/
- Qwen3-TTS Swift MLX (M2 Max): https://soniqo.audio/guides/speak ; chatterbox-mlx: https://pypi.org/project/chatterbox-mlx/ ; f5-tts-mlx: https://github.com/lucasnewman/f5-tts-mlx
- Kokoro-FastAPI (4060 Ti): https://github.com/opencoca/AI-Voice-Kokoro-FastAPI (fork README; other forks quote 35 to 100x)
