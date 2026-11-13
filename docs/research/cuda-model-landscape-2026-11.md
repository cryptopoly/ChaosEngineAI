# CUDA model landscape (Windows and Linux), November 2026

Internal research, 2026-11-09. Written after the v0.10.0 test on the RTX 4090 (24 GB VRAM, 64 GB RAM, Windows 11).
Sources: Hugging Face model cards and the HF API (`/api/models/<repo>`), GitHub repos and release APIs, PyPI JSON, diffusers release notes and docs.
**UNVERIFIED** means the claim rests on a secondary source or on our own arithmetic. Nothing here was run on the 4090 yet.
Builds on `docs/research/new-video-models.md` (2026-11-07, Mac-focused).

**Checked on the 4090 PC against the installed diffusers 0.42.1 (source, not docs):** `LTX2Pipeline`, `LTX2ConditionPipeline` (keyframe `conditions`, audio VAE), `LTX2ImageToVideoPipeline`, `WanImageToVideoPipeline` (takes `image` and `last_image`), `QwenImage21Pipeline`, `QwenImageEditPlusPipeline`, `Krea2Pipeline`, `Flux2Pipeline`, `Flux2KleinPipeline`, `ModularPipeline`, `HunyuanVideo15Pipeline` and a `kandinsky6` package all exist. Nothing was loaded or run yet.

## Summary

The CUDA lane is behind the Mac mainly because of wiring, not because the models are missing.
The pinned **diffusers 0.42.1 is the newest release** (27 Oct 2026). It already contains the pipelines for LTX-2.3/2.5 (with image-to-video, keyframes and audio), Wan 2.2 image-to-video with start **and** end frames, Kandinsky 6, FLUX.3 Image Klein, Qwen-Image 2.1, Krea 2 and Ideogram 4. So the biggest gaps close without a dependency bump.
The storyboard can run on Windows by wiring **Wan 2.2 I2V A14B** (`last_image`, Apache 2.0, small effort), then **LTX-2.3/2.5 via diffusers** (same model as the Mac, medium to large effort because 22B plus a 12B text encoder needs quantisation and offload on 24 GB).
For images, add **FLUX.3 Image Klein** and **Qwen-Image 2.1** through diffusers with INT4/FP8 transformers.
For voice, **Qwen3-TTS has an official PyTorch package** (`qwen-tts`, Apache 2.0), and **Parakeet TDT v3 runs on Windows through ONNX** (`onnx-asr`).
For music, **ACE-Step 1.5 supports CUDA and Windows officially**; it only needs the CUDA lane wired and measured.
For chat, llama.cpp publishes **Windows CUDA 12.4 and 13.x zips plus cudart zips** for every tag; the NVIDIA EULA allows redistributing cudart with conditions, but downloading it on demand is simpler. vLLM still has no native Windows support.

---

## 1. Video

diffusers 0.42.1 pipeline arrival: `ltx2` (LTX2Pipeline, LTX2ImageToVideoPipeline, LTX2ConditionPipeline, latent upsample) in 0.37.0; LTX-2.5 via `LTX2Pipeline` in 0.40.0; LTX-2.5 DFR keyframe pipelines in 0.41.0; `Kandinsky6TI2VAPipeline` in 0.42.0; Wan 2.2 in 0.35.0. The 0.42.1 tree also has `hunyuan_video1_5`, `helios`, `motif_video`, `skyreels_v2`, `sana_video`. No LongCat-Video pipeline.

| Model | Author | Repo | Released | Licence | CUDA runtime path | 24 GB fit | Key capabilities | Windows caveats |
|---|---|---|---|---|---|---|---|---|
| LTX-2.5 | Lightricks | `Lightricks/LTX-2.5-Diffusers` (single-file: `Lightricks/LTX-2.5`) | 2026-07 | LTX-2.x Community: paid at >= US$10M yearly revenue; **gated** (click-through); keep watermark/disclosure | diffusers >= 0.40 `LTX2ImageToVideoPipeline`, `LTX2ConditionPipeline` (keyframes `index=0` and `-1`), `LTX2DFRPipeline` (>= 0.41) | bf16 no (about 44 GB transformer + Gemma 12B encoder). Needs quantised transformer + offload; **UNVERIFIED** on 4090. Quants: official int8/nvfp4 single files; `unsloth/LTX-2.5-GGUF`, `unsloth/LTX-2.5-FP8` | t2v, i2v, start + end frame, audio (24 kHz) | DFR decoder wants NATTEN, Linux-only; use plain VAE with tiling on Windows. GGUF load in diffusers for LTX2 **UNVERIFIED** |
| LTX-2.3 | Lightricks | `diffusers/LTX-2.3-Diffusers` (not gated); originals `Lightricks/LTX-2.3` | 2026-03 (diffusers port 2026-05) | LTX-2 Community (same US$10M threshold) | same classes; distilled 8 steps, CFG 1 | same as 2.5. Quants: `Lightricks/LTX-2.3-fp8`, `unsloth/LTX-2.3-GGUF`, `QuantStack/LTX-2.3-GGUF` | t2v, i2v, start + end, audio | as above. Model cards saying "diffusers coming soon" are out of date |
| LTX-2 official package | Lightricks | GitHub `Lightricks/LTX-2` (ltx-core, ltx-pipelines) | 2026 | as above | `uv sync`; `--quantization fp8-cast --offload cpu` | low-VRAM mode documented, no figure | as above | natten extra skipped on Windows |
| Wan 2.2 I2V A14B | Alibaba Wan | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 2025-08 | Apache 2.0 | diffusers >= 0.35 `WanImageToVideoPipeline(image=, last_image=)` (two experts, `boundary_ratio`) | Yes with offload or GGUF. bf16 experts about 56 GB, tight in 64 GB RAM (**UNVERIFIED**). Quants: `QuantStack/Wan2.2-I2V-A14B-GGUF`; 4-step LoRAs `lightx2v/Wan2.2-Distill-Loras` | i2v, **start + end frame** | flash-attn optional |
| Wan 2.2 TI2V 5B | Alibaba Wan | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 2025-08 | Apache 2.0 | `WanImageToVideoPipeline` / `WanPipeline` | **Yes**: card says 24 GB (RTX 4090); 720p 5 s under 9 min | t2v, i2v start frame | none known |
| Wan 2.2 Animate / Animate-2 | Alibaba Wan | `Wan-AI/Wan2.2-Animate-14B-Diffusers`, `Wan-AI/Wan2.2-Animate-2-14B-Diffusers` | 2025 / 2026-08 | Apache 2.0 | `WanAnimatePipeline` | **UNVERIFIED** | character image + pose video | not storyboard keyframes |
| Kandinsky 6.0 Lite (3B) | Kandinsky Lab | `kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers` (also Pro, 29B) | 2026-09/10 | MIT | diffusers >= 0.42 `Kandinsky6TI2VAPipeline(image=)`, 10 steps | with `enable_model_cpu_offload()`; speed **UNVERIFIED** (about 437 s on a 4090, third-party) | t2v, i2v start frame, audio 44 kHz | SR example uses flex attention + `torch.compile` (needs triton on Windows) |
| Helios (14B, autoregressive) | PKU / ByteDance | `BestWishYsh/Helios-Distilled` | 2026-02 | Apache 2.0 | `HeliosPipeline`, `HeliosPyramidPipeline` (>= 0.37) | docs: optimised config about 6 GB VRAM; repo 173 GB | i2v start frame, minute-long video | **UNVERIFIED** |
| Motif-Video 2B | Motif Technologies | `Motif-Technologies/Motif-Video-2B` | 2026-04 | Apache 2.0 | `MotifVideoImage2VideoPipeline` (>= 0.39) | Yes with CPU offload (docs) | i2v start frame | none known |
| HunyuanVideo 1.5 | Tencent | `tencent/HunyuanVideo-1.5` | 2025 | **Territory excludes EU, UK, South Korea** | `hunyuan_video1_5` | — | t2v, i2v | **Cannot ship** |
| LongCat-Video | Meituan | `meituan-longcat/LongCat-Video` | 2025 | MIT | own torchrun code, no diffusers | — | i2v, continuation | high effort |
| SkyReels V2 | Skywork | see prior doc | 2025 | Skywork custom | `skyreels_v2` in 0.42.1 | — | i2v, long video | licence not reviewed |
| Wan 2.5 / 2.6 / 3.x | Alibaba | none | — | closed, no weights | — | — | — | — |
| MiniMax H3 | MiniMax | see prior doc | 2026 | **excludes EU, UK, Korea, US** | — | — | first/last frame, audio | **Cannot ship** |

### Gaps vs our catalog (video)
- No image-to-video on CUDA at all. The `startImagePath`/`endImagePath` gate only allows `ltx2_mlx` repos (`routes/video.py::_checked_input_images`, `mcp_bridge/models.py`).
- No LTX-2.x on CUDA, although diffusers 0.42.1 has it. Our CUDA LTX rows are LTX-Video 0.9 (2024).
- Wan 2.2 has only T2V A14B; no I2V A14B or TI2V 5B row.
- No video with audio on CUDA (LTX-2.x and Kandinsky 6 both have it).
- HunyuanVideo, Mochi 1 and CogVideoX rows are 2024-era and could be retired later.

---

## 2. Image

diffusers releases since 0.38: 0.38.0 NucleusMoE-Image, Ernie-Image, FLUX.2 Klein Inpaint, FLUX.2 small decoder (`black-forest-labs/FLUX.2-small-decoder`); 0.39.0 `Ideogram4Pipeline`, `Krea2Pipeline`, PRX Pixel, Anima, `JoyImageEditPipeline`; 0.40.0 JoyAI-Image-Edit-Plus, Cosmos 3 modular; 0.41.0 `QwenImage21Pipeline`; 0.42.0 FLUX.3 Image Klein (via `ModularPipeline`), `BriaFibo2Pipeline`. Our pin 0.42.1 is the newest release.

| Model | Author | Repo | Released | Licence | CUDA runtime path | 24 GB fit | Key capabilities | Windows caveats |
|---|---|---|---|---|---|---|---|---|
| FLUX.3 Image Klein (+ base, + 9B prompt upsampler `-pup`) | Black Forest Labs | `black-forest-labs/flux-3-image-klein`, `-base` | 2026-10-13 | FLUX Kommunity v1.0: **gated**, commercial only under US$5M revenue; use bans | diffusers >= 0.42 `ModularPipeline.from_pretrained`; distilled 16 steps no CFG | bf16 peak 31 GB: offload. INT4: `lite-infer/flux-3-image-klein-nunchaku-lite-int4-r32` (8.7 GiB peak), `tonera/flux-3-image-klein-nunchaku`; GGUF `leejet/flux-3-image-klein-GGUF`, `unsloth/flux-3-image-klein-GGUF`; FP8 `unsloth/flux-3-image-klein-FP8`; in-repo fp8r, nvfp4 (Blackwell only) | t2i, edit, up to 10 refs, box layout | Nunchaku Windows support **UNVERIFIED**. Upsampler about 7 GiB, ship it off |
| Qwen-Image 2.1 / Turbo | Qwen | `Qwen/Qwen-Image-2.1`, `Qwen/Qwen-Image-2.1-Turbo` | 2026-09-14 / 10-09 | **Qwen Research License** (HF tag "other"); commercial use unclear, read it | diffusers >= 0.41 `QwenImage21Pipeline` | FP8 or INT4 fits; bf16 needs offload (**UNVERIFIED**). `unsloth/Qwen-Image-2.1-FP8`, `tonera/Qwen-Image-2.1-Nunchaku`, `lite-infer/qwen-image-2.1-nunchaku-lite-int4-r32`, `leejet/Qwen-Image-2.1-GGUF`; LoRA `lightx2v/Qwen-Image-2.1-Lightning` | t2i, edit, up to 10 refs, RGBA | none known |
| Krea 2 Turbo / Raw | Krea | `krea/Krea-2-Turbo`, `krea/Krea-2-Raw` | 2026-06 | Krea 2 Community, **gated**; commercial terms **UNVERIFIED** (PDF) | diffusers >= 0.39 `Krea2Pipeline`; 8 steps | 12B: INT4 fits (`lite-infer/krea-2-turbo-nunchaku-lite-int4-r32`) | t2i | none known |
| Ideogram 4 | Ideogram | `ideogram-ai/ideogram-4-nf4-diffusers` | 2026-06-03 | **Non-commercial, gated** | diffusers >= 0.39 `Ideogram4Pipeline` | NF4 likely fits (**UNVERIFIED**); `leejet/ideogram-4-GGUF` | t2i, best text rendering, JSON prompts | — |
| MOSS-Image SFT | OpenMOSS | `OpenMOSS-Team/MOSS-Image-SFT` | 2026-10-11 | **Apache 2.0** | diffusers `custom_pipeline` + `trust_remote_code` | about 5B: fits; `lite-infer/moss-image-sft-nunchaku-lite-int4-r128` | t2i | remote code: vendor and pin it |
| HiDream-O1-Image | HiDream | `HiDream-ai/HiDream-O1-Image` (+ `-Dev`) | 2026-05-08 | MIT | **no diffusers**: custom `inference.py`, flash-attn recommended | 8B pixel-space; fit **UNVERIFIED** | t2i, edit, multi-ref | flash-attn on Windows is hard |
| Bria Fibo 2.0 | Bria | `briaai/Fibo-2.0-dev`, `briaai/Fibo-2.0-edit-turbo` | 2026-10-18 | **CC BY-NC 4.0, gated** | diffusers >= 0.42 `BriaFibo2Pipeline` | 8B, offload (**UNVERIFIED**) | t2i, edit | reject (NC) |
| Iris-3B | Sperid Labs | `speridlabs/iris-3b` | 2026-10 | Apache 2.0 | **no diffusers**: reference repo, PyTorch >= 2.7.1 | fits | t2i pixel-space, depth, 4x upscale | reference code only |
| HunyuanImage 3.0 | Tencent | `tencent/HunyuanImage-3.0` | 2025-09 / 2026-01 | Tencent licence (territory exclusion likely, **UNVERIFIED**) | transformers only | 80B MoE: **no** | — | skip |
| Z-Image Edit / Omni | Tongyi-MAI | not released (org has only Z-Image, Z-Image-Turbo) | — | — | — | — | — | — |
| Stability, Lumina, FLUX.2 | — | nothing new open; FLUX.2 small decoder is the only addition | — | — | — | — | — | — |

### Gaps vs our catalog (image)
- FLUX.3 Image Klein is Mac-only (mflux); diffusers 0.42 and GGUF/INT4 paths exist for CUDA.
- Qwen-Image 2.1 / Turbo on CUDA only through sd.cpp GGUF; the diffusers path (`QwenImage21Pipeline`) is unused. It also replaces the separate Edit-2511 row.
- No Krea 2, Ideogram 4 or MOSS-Image rows.
- Existing FLUX.2 rows do not use the faster FLUX.2 small decoder.
- No INT4 "nunchaku-lite" rows (they load with stock diffusers + `kernels`, no nunchaku wheel).

---

## 3. Voice

### TTS

| Model | Author | Repo | Released | Licence | CUDA runtime path | 24 GB fit | Key capabilities | Windows caveats |
|---|---|---|---|---|---|---|---|---|
| Qwen3-TTS 0.6B / 1.7B (Base, CustomVoice, 1.7B VoiceDesign) | Qwen | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, `-1.7B-Base`, `-CustomVoice`, `-1.7B-VoiceDesign`, `Qwen/Qwen3-TTS-Tokenizer-12Hz` | 2026-01-22 | Apache 2.0 | PyPI `qwen-tts` 0.1.1 (pins transformers==4.57.3, accelerate==1.12.0) | yes (small) | 10 languages, cloning from about 3 s, 9 presets + instructions, voice design | flash-attn optional; pins may clash with the voice venv |
| Chatterbox Nano | Resemble AI | `ResembleAI/chatterbox-nano` | 2026-04-14 | MIT | `chatterbox-tts` 0.1.7 (`nano=True`; pins torch==2.6.0) | yes | en, cloning, tags; watermark | — |
| Chatterbox Multilingual locale fine-tunes | Resemble AI | `ResembleAI/Chatterbox-Multilingual-{hi,zh-cmn,pt-br,pt-pt,es-es,es-mx-latam}` | 2026-04-22 | MIT (assumed from family, **UNVERIFIED**) | `chatterbox-tts` | yes | per-locale cloning | — |
| Chatterbox Flash | Resemble AI | `ResembleAI/chatterbox-flash` | 2026-05-28 | MIT | `chatterbox-flash` (FlashInfer, CUDA graphs) | yes | en, cloning, RTF about 0.1 | FlashInfer on Windows **UNVERIFIED** |
| VoxCPM2 (wired) | OpenBMB | `openbmb/VoxCPM2` | 2026-04-03 | Apache 2.0 | `voxcpm` | about 8 GB | 30 languages, cloning, voice design | — |
| Dia2 1B / 2B | Nari Labs | `nari-labs/Dia2-1B`, `nari-labs/Dia2-2B` | 2025-11-15 | Apache 2.0 | `dia2` | yes | en two-speaker dialogue, up to about 2 min | CUDA 12.8+ driver |
| Fun-CosyVoice3 0.5B | FunAudioLLM | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | 2025-12 | Apache 2.0 | git repo + requirements | yes | 9 languages + dialects, cloning, instructions | `ttsfrd` wheel Linux-only, SoX: real Windows risk |
| MOSS-TTS v1.5 / Nano-100M | OpenMOSS | `OpenMOSS-Team/MOSS-TTS-v1.5` | 2026 | Apache 2.0 (Nano **UNVERIFIED**) | transformers 5.0 + remote code | 8B heavy; Nano tiny | 31 languages, cloning | torch 2.9.1+cu128 pin |
| Kokoro (wired, ONNX) | hexgrad | `hexgrad/Kokoro-82M` | 2025 | Apache 2.0 | kokoro / ONNX | yes | presets, no cloning | — |
| Orpheus 3B (wired) | Canopy Labs | `unsloth/orpheus-3b-0.1-ft-GGUF` | 2025 | tag Apache 2.0, but a Llama-3.2-3B derivative: Llama licence likely applies (**UNVERIFIED**) | llama.cpp + SNAC | 6-8 GB (**UNVERIFIED**) | en, emotion tags | flag licence |
| IndexTTS-2 | bilibili | `IndexTeam/IndexTTS-2` | 2025 | **custom** (permission above 100M MAU / RMB 1B; no use to train other models) | index-tts repo | yes | zh/en cloning, emotion, duration | flag |
| Rejected (non-commercial / research) | — | `SWivid/F5-TTS` (CC-BY-NC), `bosonai/higgs-tts-3-4b` (research), `mistralai/Voxtral-4B-TTS-2603` (CC-BY-NC), `microsoft/VibeVoice-Realtime-0.5B` (research only) | — | — | — | — | — | — |
| Custom / unclear | — | `nineninesix/kani-tts-2-en` (LFM 1.0), `neuphonic/neutts-2e` (gated, other), `stepfun-ai/Step-Audio-EditX` (weights licence unstated), `kyutai/tts-1.6b-en_fr` (**UNVERIFIED** CC-BY-4.0) | — | — | — | — | — | — |

No open Qwen3.5 / Omni TTS found. MiniMax has published no open TTS.

### STT

| Model | Author | Repo | Released | Licence | CUDA runtime path | 24 GB fit | Key capabilities | Windows caveats |
|---|---|---|---|---|---|---|---|---|
| faster-whisper (wired) | SYSTRAN | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` etc. | 1.2.1, 2025-10-31 | MIT | ctranslate2 | yes | 99 languages | none |
| Parakeet TDT 0.6B v3 | NVIDIA | `nvidia/parakeet-tdt-0.6b-v3`; ONNX `istupakov/parakeet-tdt-0.6b-v3-onnx` | 2025-08 | CC-BY-4.0 (attribution) | `onnx-asr` 0.12.0 (MIT) `[gpu]` on onnxruntime-gpu | yes | 25 European languages, fast | onnx-asr supports Windows; NeMo does not |
| Canary 1B v2 / 180M flash | NVIDIA | `istupakov/canary-1b-v2-onnx`, `istupakov/canary-180m-flash-onnx` | 2025 | CC-BY-4.0 (1b-v2 **UNVERIFIED**) | `onnx-asr` | yes | ASR + translation | as above |
| Qwen3-ASR 0.6B / 1.7B | Qwen | `Qwen/Qwen3-ASR-0.6B`, `Qwen/Qwen3-ASR-1.7B` | 2026-01-28 | Apache 2.0 | PyPI `qwen-asr` (transformers backend) | yes | 30 languages + 22 dialects; timestamps via `Qwen3-ForcedAligner-0.6B` | transformers path on Windows **UNVERIFIED**; vLLM path Linux-only |
| Nemotron unified ASR 0.6B | NVIDIA | `nvidia/nemotron-labs-unified-asr-0.6b` | 2026-08-21 | OpenMDW-1.1 (permissive) | NeMo 3.1 only | yes | 25 languages, offline + streaming, beats Parakeet v3 | NeMo Linux-only; no ONNX yet (**UNVERIFIED**) |
| Nemotron Speech Streaming 0.6B | NVIDIA | `nvidia/nemotron-speech-streaming-en-0.6b` | 2025-12-17 | NVIDIA Open Model Licence (custom, commercial OK) | NeMo | yes | en streaming | Linux-only |
| Voxtral Mini 4B Realtime | Mistral | `mistralai/Voxtral-Mini-4B-Realtime-2602` | 2026-01 | Apache 2.0 | vLLM (**UNVERIFIED**) | yes | 13 languages, streaming | vLLM Linux-only |
| Nemotron 3.5 Transcribe 30B-A4B | NVIDIA | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` | 2026-10 | OpenMDW-1.1 | vLLM / transformers / GGML | NVFP4 fit **UNVERIFIED** | 40 languages, diarization | heavy |

### Gaps vs our catalog (voice)
- No Qwen3-TTS on CUDA. Every Qwen3-TTS row is `lanes=['mlx']`; the official `qwen-tts` package covers Base, CustomVoice and VoiceDesign on CUDA.
- No voice design on CUDA except VoxCPM2.
- STT on CUDA is faster-whisper only; the Mac has Parakeet v3 and Nemotron ASR.
- Orpheus licence needs a second look (Llama base).

---

## 4. Music and sound effects

| Model | Author | Repo | Released | Licence | CUDA runtime path | 24 GB fit | Key capabilities | Windows caveats |
|---|---|---|---|---|---|---|---|---|
| ACE-Step 1.5 (2B, XL turbo/sft + 0.6/1.7/4B LM) | ACE Studio / StepFun | `ACE-Step/Ace-Step1.5`, `ACE-Step/acestep-v15-xl-turbo` | XL 2026-04-02 | MIT | official PyTorch package (our `music_env.py`, `ACE_REF`) | yes: 2B <= 6 GB (LM off); XL + 1.7B LM fits 24 GB without offload (card) | text to music with vocals, BPM/key | official Windows package; use the `pt` LM backend (vllm backend is Linux) |
| Stable Audio Open 1.0 / Small (wired) | Stability AI | `stabilityai/stable-audio-open-1.0`, `-small` | 2024/2025 | Stability Community (gated; revenue cap from memory **UNVERIFIED**) | stable-audio-tools | yes | SFX, loops | already working |
| HeartMuLa 3B | HeartMuLa | `HeartMuLa/HeartMuLa-oss-3B`, `HeartMuLa/HeartMuLa-RL-oss-3B-20260123` | 2026-01/02 | Apache 2.0 (relicensed 2026-01-20) | `heartlib` from source, Python 3.10 | **UNVERIFIED** (has lazy load) | lyrics + tags to song, up to 4 min | no Windows notes |
| DiffRhythm 1.2 / DiffRhythm2 | ASLP lab | `ASLP-lab/DiffRhythm-1_2-full` | 2025 | Apache 2.0 (VAE repo "other") | repo | about 8 GB (**UNVERIFIED**) | fast full songs with vocals | — |
| MiniMax-Music3 | MiniMax | `MiniMaxAI/MiniMax-Music3` | 2026-08-07 | **custom**: show "MiniMax-Music3" in UI; permission above US$20M revenue | diffusers snippet on CUDA | about 22 GB with offload; 8 GB with layer streaming | songs with vocals up to 5 min; best quality | needs owner licence decision |
| ThinkSound | FunAudioLLM | `FunAudioLLM/ThinkSound` | 2025-07 | HF tag Apache 2.0 (GitHub terms not checked) | repo | **UNVERIFIED** | video to SFX | — |
| Rejected | — | `m-a-p/YuE2-3B` (CC-BY-NC), `hkchengrex/MMAudio` (CC-BY-NC weights), `HKUSTAudio/AudioX-Turbo` (CC-BY-NC), `tencent/HunyuanVideo-Foley` (Tencent custom) | — | — | — | — | — | — |
| Unclear | — | `lglg666/SongGeneration-v2-large` (licence unknown), SongBloom (repo not found) | — | — | — | — | — | — |

### Gaps vs our catalog (music)
- ACE-Step rows list `cuda` in `platforms`, but the CUDA lane is not wired or measured (`peakGb: None` for the 2B).
- No second song model with an open licence (HeartMuLa, DiffRhythm).
- No video-to-SFX model on any lane.

---

## 5. Chat (llama.cpp and vLLM)

### llama.cpp Windows CUDA assets (GitHub releases API)

| Tag | Windows CUDA zips | cudart zips (no tag in the name) | Other Windows |
|---|---|---|---|
| b11981 (2026-10-29, pinned) | `llama-b11981-bin-win-cuda-12.4-x64.zip` (280 MB), `llama-b11981-bin-win-cuda-13.4-x64.zip` (165 MB) | `cudart-llama-bin-win-cuda-12.4-x64.zip` (391 MB), `cudart-llama-bin-win-cuda-13.4-x64.zip` (424 MB) | `win-vulkan-x64` (37 MB), `win-rocm-10.0-x64` (265 MB), `win-sycl-x64`, `win-openvino` |
| b12229, b12230, b12233 (2026-11-09) | `...-win-cuda-12.4-x64.zip` (289 MB), `...-win-cuda-13.5-x64.zip` (171 MB) | `cudart-llama-bin-win-cuda-12.4-x64.zip`, `cudart-llama-bin-win-cuda-13.5-x64.zip` (426 MB) | same |

- Linux: `llama-b11981-bin-ubuntu-cuda-12.8-x64.tar.gz` and `...-ubuntu-cuda-13.4-x64.tar.gz`, plus `ubuntu-rocm-10.0`.
- AMD is named `rocm`, not `hip`.
- cudart: the NVIDIA CUDA EULA (Attachment A) lists `cudart`, `cublas` and `cublasLt` as redistributable inside an application that adds material functionality, not standalone, with licence terms that protect NVIDIA. Our reading, not legal advice. Downloading the cudart zip on first use avoids the question and about 400 MB in the installer.
- Vulkan vs CUDA on a 4090: no controlled published benchmark (**UNVERIFIED**). Reports say prompt processing is slower on Vulkan and generation is close. Measure with `llama-bench -p 512 -n 128 -ngl 99 -fa 1` on both zips.
- MTP hidden-state fix (#30141) merged 2026-11-03, after b11981.

### vLLM
- Docs: Linux only, "does not support Windows natively"; WSL2 is the documented route. Latest PyPI version 0.33.0. Unofficial Windows forks exist; not shippable.

### Chat models for 24 GB not in our catalog (verified on HF)

| Model | Author | GGUF repo | Released | Licence | Size | Notes |
|---|---|---|---|---|---|---|
| gpt-oss-20b | OpenAI | `ggml-org/gpt-oss-20b-GGUF` | 2025-08 | Apache 2.0 | MXFP4 12.1 GB | repo includes an EAGLE3 drafter (Q8_0, 0.92 GB) |
| Clef | Cloudflare | `ggml-org/Clef-GGUF` | 2026-10 | Apache 2.0 | Q4_K_M 19.5 GB | vision (mmproj) |
| Clef-Flash | Cloudflare | `ggml-org/Clef-Flash-GGUF` | 2026-10 | Apache 2.0 | Q4_K_M 6.7 GB | vision |
| Qwen3-Coder-30B-A3B-Instruct | Qwen | `ggml-org/Qwen3-Coder-30B-A3B-Instruct-GGUF` | 2025-07 | Apache 2.0 | Q4_K_M 18.0 GB | coding |
| Qwen3.8-27B (Unsloth quants + MTP file) | Qwen / Unsloth | `unsloth/Qwen3.8-27B-GGUF` | 2026-08 | Apache 2.0 | UD-Q4_K_M 16.5 GB | `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`; loading a separate MTP file **UNVERIFIED** |

Already in catalog: Qwen3.8-27B, Qwen3.6-27B / 35B-A3B (with MTP), Gemma 4 26B-A4B / 31B, Nemotron 3.5 Lightning, LFM2.5-24B, Granite Switch, Devstral Small, and others. Qwen3.8-Flash-Next (UD-Q4_K_XL 111 GB) is in the catalog but does not fit 24 GB.

### Speculative decoding in llama.cpp (CUDA)
- `--spec-type draft-mtp` (PR #22673, 2026-05) and `draft-eagle3` (PR #18039, 2026-06) are merged. `draft-dflash` / `draft-dspark` exist on master; presence in b11981 **UNVERIFIED**.
- Working pairs: Qwen3.6 with built-in MTP GGUFs (in catalog); gpt-oss-20b with its EAGLE3 GGUF. Gemma 4 E2B as a draft for Gemma 4 31B is **UNVERIFIED**. Qwen3.8 has no small sibling, so use MTP.

### Gaps vs our catalog (chat)
- Windows ships Vulkan only; no CUDA build option even though upstream publishes one per tag.
- No CUDA 13.x / 12.4 choice by driver version.
- vLLM rows are meaningless on native Windows; they should be hidden or marked WSL-only.
- Missing gpt-oss-20b + EAGLE3 pair and Clef / Clef-Flash.

---

## Recommended additions (ranked)

### Video
1. **Wan 2.2 I2V A14B with start and end frame** (`WanImageToVideoPipeline(image, last_image)`, Apache 2.0) + lightx2v 4-step LoRAs + QuantStack GGUF tier. Unblocks the storyboard on Windows. Gate change in `routes/video.py` and `mcp_bridge/models.py` to a per-variant `acceptsImageInput` / `acceptsEndImage` flag. **S**
2. **LTX-2.3 then LTX-2.5 via diffusers** (`diffusers/LTX-2.3-Diffusers`, `Lightricks/LTX-2.5-Diffusers`): same model as the Mac, keyframes + audio. Needs a 24 GB recipe (FP8/layerwise-cast transformer, quantised Gemma encoder, offload, VAE tiling), audio mux, gated-licence flow for 2.5, no DFR on Windows. **M-L**
3. **Wan 2.2 TI2V 5B**: fits 24 GB natively per its card; fast light tier, start frame only. **S**
4. **Kandinsky 6.0 Lite distill** (MIT, start frame + 44 kHz audio). **M**
5. **Motif-Video 2B I2V** (Apache 2.0, simple) or **Helios-Distilled** (long video). **S / M**

### Image
1. **FLUX.3 Image Klein** (distilled + base) via diffusers `ModularPipeline` with the nunchaku-lite INT4 transformer (about 9 GiB), or sd.cpp with `leejet/flux-3-image-klein-GGUF`. Upsampler off by default. Gated, US$5M cap. **M**
2. **Qwen-Image 2.1 + Turbo via `QwenImage21Pipeline`** with FP8 or INT4 transformer and the Lightning LoRA; one model for t2i, edit and RGBA. Label the Research licence. **S-M**
3. **Krea 2 Turbo** (`Krea2Pipeline`, 8 steps, INT4 available). Read the licence PDF first. **S**
4. **MOSS-Image SFT** (the only strong 2026 Apache 2.0 model); vendor and pin the remote pipeline code. **M**
5. **FLUX.2 small decoder** for existing FLUX.2 rows (speed win). Ideogram 4 only with a clear non-commercial label. **S**

### Voice
1. **Qwen3-TTS on CUDA** via `qwen-tts` (Base, CustomVoice, VoiceDesign; Apache 2.0); matches the Mac rows. Check the transformers 4.57.3 pin against the voice venv. **M**
2. **Parakeet TDT 0.6B v3 STT** via `onnx-asr` + `istupakov/parakeet-tdt-0.6b-v3-onnx` on onnxruntime-gpu (CC-BY-4.0, attribution). Works on Windows; matches the Mac row. Canary 1B v2 ONNX comes nearly free after. **S-M**
3. **Chatterbox Nano + Multilingual locale fine-tunes** through the existing `chatterbox-tts` path (MIT). **S**
4. **Qwen3-ASR 0.6B / 1.7B** via `qwen-asr` transformers backend (Apache 2.0, 52 languages and dialects). **M**
5. **Dia2 1B / 2B** for two-speaker dialogue (Apache 2.0). **M**

### Music and SFX
1. **ACE-Step 1.5 on CUDA**: reuse the music venv and `ACE_REF`, `pt` LM backend on Windows, measure peak VRAM on the 4090 and fill `estimates.cuda`. **S-M**
2. **HeartMuLa-oss-3B** (Apache 2.0, lyrics to song). **M**
3. **DiffRhythm 1.2 / DiffRhythm2** (Apache 2.0, fast). **M**
4. **MiniMax-Music3**: best quality, but needs an owner licence decision (UI attribution, US$20M cap). **M**
5. **ThinkSound** for video-to-SFX after checking the GitHub licence. **M**

### Chat
1. **Windows CUDA llama.cpp option**: download `llama-<tag>-bin-win-cuda-12.4-x64.zip` + `cudart-llama-bin-win-cuda-12.4-x64.zip` on demand (13.x for newer drivers), keep Vulkan bundled as the default. Benchmark both on the 4090. **M**
2. **Bump the pin past b11981** (for example b12233) for the MTP hidden-state fix; confirm assets exist and update `stage-runtime.mjs` patterns for 13.5. **S**
3. **gpt-oss-20b + EAGLE3 drafter** (`ggml-org/gpt-oss-20b-GGUF`). **S**
4. **Clef / Clef-Flash** (Apache 2.0, vision) and mark Qwen3.8-Flash-Next as not fitting 24 GB. **S**
5. **vLLM**: do not ship on native Windows; WSL2-only if ever. Cheap ROCm note: the `win-rocm-10.0-x64` zip makes an AMD chat option a pattern change only. **S (ROCm) / L (vLLM)**
