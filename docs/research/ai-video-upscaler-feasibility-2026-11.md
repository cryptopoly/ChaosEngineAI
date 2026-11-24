# AI video upscaler: feasibility (2026-11-24)

Research only: nothing was run, downloaded or installed. Facts come from model cards and repos read on 2026-11-24. "Unverified" means the source did not state it; "estimate" means a guess that section 5 replaces with measurements.

## 1. Verdict

Feasible, and worth doing. A fast tier (Real-ESRGAN family, xinntao) is a small job: BSD-3 code and weights, tiny models, runs in the PyTorch the GPU runtime already ships (torch 2.11 CUDA in the dev venv; MPS on a Mac), and is clearly better than the lanczos stretch for compressed or noisy footage. A quality tier (one-step diffusion: FlashVSR+ from Shanghai AI Lab, or SeedVR2 from ByteDance, both Apache-2.0 code and weights) gives much more real detail but is slow, 0.2 to 1 frame per second at 4K on a 4090 by the authors' own numbers, so it suits a three-minute film overnight, not a one-hour family tape. Expect very good results on generated 720p clips and good ones, with some invented detail, on old tapes; faces in family footage are the main quality risk. Size: fast tier plus Storyboard export plus a standalone tool is a medium job (about two weeks of agent work with tests); the quality tier is a large job, mostly Windows install risk and long-job plumbing.

## 2. Candidate models

| Model (author) | Kind | Scale | Code licence | Weights licence | Weights source | Runtimes | Old or compressed footage | Temporal consistency | Speed | VRAM |
|---|---|---|---|---|---|---|---|---|---|---|
| Real-ESRGAN `realesr-general-x4v3` (xinntao) | Single-image GAN, compact (SRVGGNet, about 1.2 M parameters) | 4x; has a denoise-strength blend | BSD-3-Clause | Not stated separately; shipped under the repo's BSD-3 | GitHub release assets, no official Hugging Face repo (pin by SHA-256) | PyTorch CUDA/MPS, ONNX, ncnn/Vulkan; loads through `spandrel` (chaiNNer-org, MIT) | Good on blocking and noise; smooths fine texture | None built in; mild shimmer on noisy input | 24 to 42 ms per 512x512 frame on an RTX 5060 Ti (third-party card); 4090 estimate 20 to 40 fps at 480p input | Under 2 GB |
| Real-ESRGAN `RealESRGAN_x4plus` / `x2plus` (xinntao) | Single-image GAN, large (RRDB) | 4x, 2x | BSD-3-Clause | As above | GitHub release assets | Same | Strongest of the GAN rows on heavy compression; can look waxy | None; visible flicker in textures | Estimate 1 to 3 fps at 480p to 720p input on a 4090 | 4 to 10 GB, tiles below that |
| `realesr-animevideov3` (xinntao); Real-CUGAN (bilibili) | Single-image GAN for animation | 2x to 4x | BSD-3; MIT | BSD-3 (repo); MIT | GitHub releases | PyTorch, ncnn/Vulkan | Animation only | Fair | Fast | Under 2 GB |
| RealBasicVSR, BasicVSR++ (Chan et al.; MMagic, OpenMMLab) | Video-recurrent GAN | 4x | Apache-2.0 | Not stated | Cloud drive, not Hugging Face | PyTorch with mmcv (2022 pins) | Built for real-world degradation | Good: uses neighbouring frames | Slow | High, unverified |
| SwinIR real-SR (Liang et al.), HAT `Real_HAT_GAN_SRx4` (XPixelGroup) | Single-image transformer GAN | 4x | Apache-2.0 | Not stated | GitHub releases; cloud drive (HAT) | PyTorch, `spandrel` | Unverified against RRDB on tape noise | None | Slower than RRDB (heavier per pixel) | Needs tiles |
| SeedVR2 3B and 7B (ByteDance-Seed) | Diffusion, one step | Any target size | Apache-2.0 | Apache-2.0 (tag on both cards) | `ByteDance-Seed/SeedVR2-3B`, `ByteDance-Seed/SeedVR2-7B`; GGUF at `cmeka/SeedVR2-GGUF` | PyTorch CUDA (official code wants flash-attn and apex); a community Apache-2.0 CLI by numz and AInVFX adds Windows, MPS, block swap, tiled VAE, GGUF and fp8; `mflux` 0.24 (already in the app) runs it on MLX for images only | Card warns: weak on heavy degradation and large motion, over-sharpens light degradation such as 720p AI video | Good within a batch of 4n+1 frames; seams between batches need overlap | 4090: 0.18 fps at 4K (FlashVSR+ authors), about 0.45 fps to 1080p (community, 2025). One H100 80 GB handles 100 frames at 720x1280 (official) | 24 GB for fp16 3B; 8 to 16 GB with GGUF and block swap (community claim) |
| FlashVSR v1.1 (OpenImagingLab) | Diffusion, one step, streaming, on Wan 2.1 | 4x recommended | Apache-2.0 | Apache-2.0 | `JunhaoZhuang/FlashVSR-v1.1` | PyTorch CUDA; needs a compiled Block-Sparse-Attention kernel; RTX 40/50 support "unknown" on the card | Unverified | Good: streaming by design | 17 fps at 768x1408 on an A100 (paper) | Unverified |
| **FlashVSR+** (Shanghai AI Lab and others, 2026-10) | Diffusion, one step, streaming, tuned from Wan 2.1 T2V 1.3B | Sizes divisible by 128; demos 768p to 4K | Apache-2.0 | Apache-2.0 | `ruchangyao/FlashVSR_plus` (Base and Turbo) | PyTorch CUDA only, Linux listed; attention backends: Triton (recommended for a 4090), compiled block-sparse, or plain PyTorch SDPA with no compiled part | Unverified; paper claims better text | Good; `--temporal-compression` for large motion | 4090 at 4K: Base 0.50 fps, Turbo 0.98 fps (authors) | 4K untiled on 24 GB (authors) |
| STAR (NJU-PCALab) | Diffusion, multi-step | 4x | Unverified | I2VGen-XL variants MIT; CogVideoX-5B variant under the CogVideoX licence (**restricted, flag**) | `SherryX/STAR` (no licence tag) | PyTorch CUDA | Has a heavy-degradation checkpoint | Good | Slow | 39 GB for 72 frames at 240p input; 24 GB minimum |
| Upscale-A-Video (S-Lab, NTU) | Diffusion, multi-step | 4x | NTU S-Lab 1.0: **non-commercial** | Same, cloud drive | Not on Hugging Face | PyTorch CUDA | Good | Good | Slow | High |
| VEnhancer (Vchitect) | Diffusion, multi-step, also raises frame rate | 1x to 8x, 2K cap | Unverified | Unverified, no tag | `jwhejwhe/VEnhancer` | PyTorch CUDA | Made for AI video, not tapes | Good | Slow (15 steps at best) | "At least A100 80G" |
| DOVE (Chen et al.); DUO-VSR (CVPR 2026) | Diffusion, one step | 4x | DOVE Apache-2.0 | DOVE sits on CogVideoX1.5-5B (**restricted base, flag**); no DUO-VSR release found | Cloud drive (DOVE) | PyTorch CUDA | Unverified | Unverified | Unverified | Unverified |
| RIFE 4.25 / 4.26 (hzwer, Practical-RIFE) | Frame interpolation, optional add-on | 2x to 4x frame rate | MIT | MIT (stated) | Cloud drive (pin by SHA-256) | PyTorch, ncnn/Vulkan | n/a | n/a | Fast | Under 4 GB |

Pre-steps need no model: ffmpeg has `idet` (detect interlacing), `bwdif` (deinterlace), `nlmeans`, `atadenoise` and `bm3d` (denoise) and `deblock`. Caution: `hqdn3d`, `nnedi` and `pp` are GPL-only filters. The dev venv's ffmpeg is a GPL build that has them; the app's pinned Windows and Linux build is LGPL (`backend_service/ffmpeg_fetch/pins.py`) and should not. Check `ffmpeg -filters` on the pinned build.

Left out on purpose: the GPU vendor's closed video super-resolution SDK (its own licence, not permissive).

## 3. Recommendation

**Fast tier, every platform, first release.** `realesr-general-x4v3` as the default ("Fast") and `RealESRGAN_x4plus` as "Fast, stronger clean-up", with `realesr-animevideov3` for animation. Run them in PyTorch through `spandrel`: CUDA on Windows and Linux, MPS on the Mac, CPU as a slow fallback. No new runtime, about 70 MB of weights in total, one small MIT package. ONNX or ncnn/Vulkan would only add AMD and Intel GPUs; later. The models are 4x; for 1440p or 2160p targets run 4x and scale down with lanczos, which also hides GAN over-sharpening.

**Quality tier, big GPUs, second release.** Two candidates, to be settled by a spike:

- **FlashVSR+ Turbo/Base** as the first choice on NVIDIA: five times faster than SeedVR2 at 4K on the same 4090 by its authors' figures, streaming so long videos keep flat memory, and its demo case (768p to 4K) is exactly the Storyboard case. Risks: six weeks old, Linux-only install notes, and the fast attention path on a 4090 is a Triton operator. Lanes to try in order: native Windows with the SDPA fallback, native Windows with Triton, WSL2 (the app already has a WSL lane, `routes/setup/vllm_wsl.py`).
- **SeedVR2 3B** as the fallback on NVIDIA and the only option on the Mac (FlashVSR+ is CUDA-only): proven on Windows and MPS through the community CLI, GGUF rows for 8 to 16 GB cards, 7B fp16 for 48 to 80 GB. On the Mac, `mflux` already ships SeedVR2 but for single images; frame-by-frame use would flicker, so the Mac lane needs either batch support there or the PyTorch MPS path. Unproven until run on the Mac.

RealBasicVSR was built for old footage, but its 2022 mmcv stack and cloud-drive weights make it a poor fit; revisit only if both diffusion rows fail on tapes. STAR (CogVideoX variant), Upscale-A-Video and DOVE carry restricted licences: do not ship.

**Fit by hardware (starting gates, to be measured):** 12 GB and below: fast tier, tiled when needed; quality tier hidden or marked slow. 16 to 24 GB: quality tier to 1080p/1440p, 4K with FlashVSR+ on 24 GB. 48 to 80 GB: SeedVR2 7B and larger batches, shown as untested until the borrowed-GPU day.

**How it fits the app.**

- *Worker.* A new `backend_service/upscale_worker.py` on `warm_worker_base.serve` (same JSON-lines protocol, cancel and step events as `lipsync_worker.py`), started from `backend_service/upscale_runtime.py`. Never in the FastAPI process. Fast tier runs in the backend's own Python; the quality tier gets its own venv, since FlashVSR+ pins its own requirements.
- *Frame pipeline.* ffmpeg decodes to raw frames on a pipe (deinterlace and denoise filters applied here), the worker upscales in chunks, ffmpeg encodes each chunk to a segment file, then concatenates and re-muxes the original audio (`-c:a copy`, AAC when the container refuses), as `lipsync_runtime.mux_command` and `video_runtime/audio_mux.py` do today. Tiles with overlap for big frames; chunk overlap of a few frames for the diffusion rows.
- *Resume.* One manifest per job listing finished segments, in the style of `RenderJob.manifest` in `storyboard/render_adapter.py`. A restart skips finished segments. This is essential: an hour of tape is 90,000 frames.
- *Queue, progress, cancel.* Take the heavy slot in `helpers/job_queue.py`; progress through `backend_service/progress.py`; time estimates in `helpers/time_estimate.py`; an `upscale_video` agent tool beside `mcp_bridge/lipsync_tool.py`.
- *Add-on installer.* Weights and the quality venv under `~/.chaosengine/addons/video-upscale/`, modelled on `lipsync_score_addon.py` (pinned packages, SHA-256 for non-Hugging Face files) and `music_env.py` (own venv); routes like `routes/setup/lipsync_score.py`.
- *Catalog and fit.* New `backend_service/catalog/upscale_models.py` with repo, pinned revision, licence text and VRAM need; gating through `helpers/fit.py` (`_media_footprint`, `variant_fit`) and `helpers/memory_gate.py`.
- *Storyboard export.* Keep `upscale` as the target size and add a method field (`lanczos`, `fast`, `quality`) in `storyboard/models.py` (line 308), `formats.py` (`UPSCALES`, `upscale_size`), `render_plan.py` (`upscale_to`, line 338), `timeline.py` (`upscale_target`) and the enum in `mcp_bridge/storyboard_tools.py` (line 69). Upscale each shot's clip before assembly, cached by clip hash, and assemble at the final size, instead of filtering the finished film in `assemble_graphs.upscale_filter` (line 162; called from `assemble.py` line 594). Captions, logos and cards are then drawn sharp at the final size (a diffusion model smears burnt-in text), and a re-made shot does not send the whole film through the upscaler again.
- *Frontend.* A "Video Upscale" tab in the Tools group (`src/constants/tabs.ts`, `sidebarGroups.ts`) with its feature folder under `src/features/`, an "Upscale" action on Video Gallery items, and the method choice in `src/features/storyboard/ExportTab.tsx`.
- *i18n and gates.* New keys in `src/locales/en/studio.json` and `setup.json` with null placeholders in the other nine locales; a new phase in `scripts/e2e_test_suite.py`; rows in `THIRD_PARTY_NOTICES.md` and the upstream table.

## 4. Cost, risks, first-release cut

| Part | Size |
|---|---|
| Fast-tier worker, frame pipeline, tiles, audio re-mux | M |
| Resume manifest, queue, progress, cancel | S |
| Catalog rows, fit gates, add-on installer and Setup routes | S |
| Standalone tool UI, Gallery action, i18n | M |
| Storyboard per-shot upscale with cache, export option, agent tool | M |
| Deinterlace, denoise and SD colour (BT.601 to 709) pre-steps with auto-detect | S |
| Quality tier: venv installer, FlashVSR+ or SeedVR2 engine, Windows proof | L |
| Mac quality lane | M, unknown until tried |
| RIFE interpolation add-on | S to M |

Risks: (1) Licence: Real-ESRGAN and SwinIR weights have no licence statement beyond the repo's. (2) Windows install: FlashVSR+ lists Linux only and SeedVR2's official code wants flash-attn and apex wheels; either may force WSL2. (3) VRAM: decode can exceed 24 GB at 4K without tiles. (4) Flicker: single-image models shimmer on noisy tape; chunk seams on diffusion rows. (5) Speed: the quality tier needs days for long tapes. (6) Invented detail: diffusion models can change faces and text, so family footage needs a faithful default. (7) The community SeedVR2 CLI's repo name contains another app's name, which the editorial rule forbids writing.

Cut for a first release: the quality tier, RIFE, ONNX/ncnn lanes, animation row, batch folders, 10-bit and HDR, double-rate deinterlacing. Ship fast tier, pre-steps, resume, standalone tool and Storyboard option.

## 5. Testing plan

**Clips** (put under a scratch folder, never in the repo):

1. `hd_ref.mp4`: 5 s of a known 1080p clip. Round trip source: `ffmpeg -i hd_ref.mp4 -vf scale=640:360:flags=bicubic -c:v h264_nvenc -b:v 500k -c:a copy lr_360p.mp4`
2. A real 30 s tape capture from the owner (576i, noisy, with sound); a weaker stand-in is `-vf "scale=720:576,noise=alls=18:allf=t" -b:v 1200k`.
3. One 720p clip generated in the app (Video Gallery), ideally with a face and some text.

**On the owner's RTX 4090** (when the GPU is free; scratch venv and downloads need approval, as D-133 did):

- Quality: restore `lr_360p.mp4` to 1080p with each row and with lanczos, then `ffmpeg -i out.mp4 -i hd_ref.mp4 -lavfi "[0:v][1:v]psnr=stats_file=psnr.log" -f null -` and the same with `ssim`. LPIPS with the `lpips` package (BSD-2) in the scratch venv. GAN and diffusion rows often lose to lanczos on PSNR while looking better, so LPIPS and eyes decide.
- Flicker: mean absolute value of (output frame difference minus the lanczos-upscaled input frame difference) over the clip, in numpy; lower is steadier. Report against lanczos as the floor.
- Speed and memory: frames per second end to end (decode, model, encode) and model-only; peak VRAM from `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > vram.log`.
- Function: audio in sync and bit-identical when copied; cancel mid-run; kill the worker and resume; interlaced clip detected and deinterlaced; a 10-minute clip for memory drift; a Storyboard export at 1440p and 2160p with captions.
- Looked at by a person: faces in clip 2, text in clip 3.

Expected on the 4090 (**estimates**): fast compact row 20 to 40 fps from 480p, 8 to 15 fps from 720p to 2160p; RRDB row 1 to 3 fps; FlashVSR+ Turbo about 1 fps to 4K and 3 to 4 fps to 1080p; SeedVR2 3B 0.2 fps to 4K and 0.4 to 0.5 fps to 1080p. A 3-minute film (4,320 frames) to 4K: about 5 to 10 minutes fast, 1.2 hours FlashVSR+ Turbo, 6 hours SeedVR2. A 60-minute tape to 1080p: about 1 hour fast, 7 to 8 hours FlashVSR+, over 50 hours SeedVR2.

**Needs a Mac:** fast tier on MPS (speed, memory, same picture as CUDA within tolerance); SeedVR2 3B on MPS or MLX (does it run on video, flicker, time per frame, unified memory peak); ffmpeg filter list on the Mac's ffmpeg; hardware encode at 2160p.

**Borrowed big-GPU day (48 to 80 GB, Linux).** Same three clips, plus a 100-frame cut of each.

```
# SeedVR2 3B (official code); repeat with the 7B config
python -c "from huggingface_hub import snapshot_download as d; d('ByteDance-Seed/SeedVR2-3B', local_dir='ckpts')"
torchrun --nproc-per-node=1 projects/inference_seedvr2_3b.py --video_path in/ --output_dir out_1080/ --seed 42 --res_h 1080 --res_w 1920 --sp_size 1
torchrun --nproc-per-node=1 projects/inference_seedvr2_3b.py --video_path in/ --output_dir out_2160/ --seed 42 --res_h 2160 --res_w 3840 --sp_size 1

# FlashVSR+ (weights from ruchangyao/FlashVSR_plus into ckpts/)
python -m inference.run_stream --input in/clip.mp4 --mode turbo --output out/clip_turbo_4k.mp4 --height 2176 --width 3840 --decode-every 28
python -m inference.run_stream --input in/clip.mp4 --mode base --output out/clip_base_1080.mp4 --height 1024 --width 1920
```

Run `nvidia-smi` logging as above beside each. The 7B script name and whether 4K fits one 80 GB card are unverified (the SeedVR README says 1080p and 2K took four H100s with sequence parallelism). Fill in:

| Model | GPU | Clip | Output size | Frames | Seconds | fps | Peak VRAM GB | LPIPS | Flicker | Faces and text OK (y/n) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SeedVR2 3B | | | 1920x1080 | | | | | | | | |
| SeedVR2 3B | | | 3840x2160 | | | | | | | | |
| SeedVR2 7B | | | 3840x2160 | | | | | | | | |
| FlashVSR+ Base | | | 1920x1024 | | | | | | | | |
| FlashVSR+ Turbo | | | 3840x2176 | | | | | | | | |

## 6. Open questions and decisions

Open questions:

- Does FlashVSR+ run on native Windows with the SDPA or Triton backend, and at what speed and memory? Decides whether the quality tier needs WSL2.
- How do the diffusion rows treat real tape faces? SeedVR2's own card warns about heavy degradation.
- Is the Mac quality lane practical at all (SeedVR2 on MPS or MLX for video)?
- Does the pinned LGPL ffmpeg have every pre-step filter, including `zscale` for SD colour?
- Should restricted-licence rows (STAR CogVideoX variant, Upscale-A-Video, DOVE) ever be offered as user-initiated downloads? Recommendation: no, the Apache rows cover the need.

Decisions needed from the owner:

1. **First-release scope.** Recommend: ship the fast tier (standalone tool, Storyboard option, pre-steps, resume) first, and run the quality tier as a spike in parallel. Alternative: hold everything until the quality tier is proven.
2. **Quality engine and the naming rule.** Recommend: spike FlashVSR+ on native Windows first; fall back to SeedVR2 3B. For SeedVR2, either write our own worker on ByteDance's Apache code (more work, clean naming) or depend on the community CLI and allow its repo name in `THIRD_PARTY_NOTICES.md` as a vendored dependency. Recommend our own worker, borrowing only ideas.
3. **A GPU window and scratch downloads for the spike.** Recommend: yes, a scratch folder only (Real-ESRGAN weights about 70 MB; FlashVSR+ and SeedVR2 3B weights, sizes not stated on the cards, likely several GB each; one scratch venv), nothing in the app or constraints until the numbers are in.
