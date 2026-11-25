# AI video upscaler: quality-tier spike (2026-11-25)

Follow-up to [ai-video-upscaler-feasibility-2026-11.md](ai-video-upscaler-feasibility-2026-11.md), section 5 and the open question "does FlashVSR+ run on native Windows". Run on the owner's RTX 4090 24 GB, Windows 11, driver 621.24, under D0140 (scratch downloads and a GPU window). No product code, dependency or package file was changed. Everything (code, weights, venvs, clips, outputs, frames, logs, scripts) is in `E:\tmp\upscale-spike\`.

## 1. Verdict

**A quality tier is viable, with FlashVSR+ Turbo on native Windows. WSL2 is not needed.** It runs through the project's own Triton attention operator on `triton-windows` (PyPI, MIT, published from the `triton-lang/triton-windows` repository), with no compiled kernel, no flash-attn, no apex and no Block-Sparse-Attention. Measured end to end on the 4090: **5.7 fps from 576p to 1280x1024, 3.8 fps from 540p to 1920x1024, 2.1 fps from 720p to 2560x1408**, at 8.8, 11.2 and 17.2 GB of VRAM. That is 5 to 10 times faster than the feasibility doc assumed. The picture is clearly better than the fast tier: it removes tape noise, blocking and leftover combing, and adds believable texture that holds steady from frame to frame. **4K does not fit**: on Windows the 3840x2176 run spilled 10 to 12.5 GB into shared system memory and had not finished 49 frames after 16 minutes. SeedVR2 was not run. FlashVSR+ settled the question, and reading SeedVR2's official code showed a port as large as writing our own worker (section 5).

Recommendation: FlashVSR+ **Turbo** as the quality engine, native Windows and Linux, CUDA only, output up to 1440p on 24 GB cards. Size **L**. Ship it in **the release after v0.10.0** and keep v0.10.0 to the fast tier (reasons in section 6).

## 2. Setup and install effort

| Item | FlashVSR+ (measured) | SeedVR2 3B (read only, not run) |
|---|---|---|
| Code | `OpenImagingLab/FlashVSR_Plus` @ `b515ffc` (2026-11-17), downloaded as a commit archive | `ByteDance-Seed/SeedVR` @ `e4de8c2` (2026-01-27) |
| Weights | `ruchangyao/FlashVSR_plus` @ `46b7952`: Turbo or Base `dit.safetensors` 5.68 GB + `lq_proj_in.safetensors` 1.15 GB each, shared `TCDecoder.ckpt` 0.19 GB (Wan VAE 0.51 GB is optional and was not used). One variant is about 7.0 GB. | `ByteDance-Seed/SeedVR2-3B` @ `37255ff`: `seedvr2_ema_3b.pth` 13.6 GB (a pickle, no safetensors) + `ema_vae.pth` 1.0 GB. The repo also hosts Linux `apex` wheels. |
| Python and torch | Python 3.12 venv; torch 2.7.1+cu128 from the PyTorch index (the repo asks for `torch>=2.5,<2.8`); einops 0.8.1, imageio 2.37.0, imageio-ffmpeg 0.6.0, numpy 2.2.6, opencv-python-headless 4.12.0.88, Pillow 11.3.0, safetensors 0.6.2, tqdm 4.67.1 | Pins torch 2.3.0, diffusers 0.29.1, transformers 4.38.2, torchvision 0.18.0 |
| Attention | `--attn-kernel triton_block_sparse` with `triton-windows==3.3.1.post21` (wheel only) and the repo's own `operators/flashvsr_attn` (pure Python plus a Triton kernel; it imports only torch and triton, so the README's flash-attn step is not needed). Block-Sparse-Attention was not built: no MSVC on the path, and the CUDA 13.5 toolkit does not match torch's cu128. | `from flash_attn import flash_attn_varlen_func` at import, with no fallback. `apex` fused norms are optional. |
| SDPA fallback (`--hardware-profile generic`) | **Not usable at real sizes.** It turns the sparse block mask into a dense mask and refuses above 2e9 entries: 1280x1024 failed at once. 768x640 worked: 49 frames in 8.5 s, 13.6 GB. | n/a |
| Safety reading | Every `torch.load` in the code passes `weights_only=True`; the DiT weights are safetensors. Nothing runs at import or install (`pyproject` has no dependencies and no build hooks). It calls ffmpeg from imageio-ffmpeg through list-form `subprocess.run` to copy audio. | `torch.load(..., mmap=True)` without `weights_only` on the 13.6 GB pickle; it would need a patch. NCCL `torch.distributed` initialisation; the whole clip goes through in one pass (no chunking). |
| Fragility | The repo is 5 weeks old. `triton-windows` is tied to the torch minor version (3.3 goes with torch 2.7). The first run compiles Triton kernels, a few seconds that are then cached. The FlashVSR+ CLI muxes audio with `-shortest`, so a clip whose audio is shorter than its video loses frames (193 in, 189 out on `ai720c`). Our own mux avoids this. Sizes must be multiples of 128; the CLI centre-crops. | High: a flash-attn build on Windows, 2024 pins, distributed init. The community Windows port exists for this reason. |

Scripts used: `E:\tmp\upscale-spike\scripts\` (`run_flashvsr.py` driver run with `python -I` from outside the code folder, `run_fast.py` runs the shipped fast tier through `backend_service.video_upscale.pipeline.run_job`, `measure.py`, `frames.py`, `gpu_run.sh` holds the shared GPU lock and samples `nvidia-smi` every 500 ms).

## 3. Clips

| Clip | Source | Frames | Purpose |
|---|---|---|---|
| `tape576p` | 8 s of an app Storyboard export (`Sunrise-Road-Trip-Music-Video`, AI-generated, under `E:\tmp\e2e-final-home`), turned into a tape stand-in: 4:3 crop, 720x576, blur, 50p to 25i `interlace`, temporal noise 14, desaturated, MPEG-2 at 4 Mb/s (`tape576i.mpg`), then the app's own pre-step `bwdif=mode=send_frame` to 25p | 200 | (a) old interlaced SD |
| `pattern576p` | `E:\tmp\interlace-test\unflagged.avi` (synthetic test pattern with a timecode, 3 s, interlaced), deinterlaced with `bwdif` | 75 | text and saturated colour |
| `ai540c` | `E:\tmp\upscale-test\hd_ref.mp4` (AI-generated 720p music-video cut with a face; the fast-tier test reference), bicubic down to 960x540 at 1 Mb/s, cropped to 960x512, 24 fps CFR | 192 | (b) compressed AI footage, 540p to 1080p, face |
| `ai720c` | The same Storyboard export at its native 720p, cropped to 1280x704, 24 fps CFR | 193 | (b) AI 720p to 1440p |

The interlace-test clips are only 2 to 3 s of synthetic pattern, so the natural tape stand-in was built from app footage. A real tape capture is still missing (section 7). Crops make the aspect ratio match FlashVSR+'s multiple-of-128 sizes, so every engine gets the same pixels.

## 4. Measurements (RTX 4090, native Windows)

"Seconds" is the inference call end to end (decode, model, encode; model load of about 6 s excluded). VRAM is the `nvidia-smi` peak minus the idle desktop (about 1.1 GB), with torch's own reserved peak in brackets. LPIPS was not measured: there is no ground truth at these sizes. Flicker columns are explained below the table. Lower is steadier.

| Model | GPU | Clip | Output size | Frames | Seconds | fps | Peak VRAM GB | LPIPS | Flicker low / pump | Faces and text OK (y/n) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fast `realesr-general-x4v3` (shipped) | 4090 Win | tape576p | 1280x1024 | 200 | 7.3 | 27.4 | 0.3 (0.31) | not measured | 1.71 / 0.07 | n/a | Blotchy water, vertical streak artefacts that change every frame |
| Fast `RealESRGAN_x4plus` (shipped) | 4090 Win | tape576p | 1280x1024 | 200 | 59.9 | 3.3 | ~6.4 (2.76) | not measured | 2.24 / 0.12 | n/a | Same artefacts, a bit sharper |
| **FlashVSR+ Turbo** | 4090 Win, Triton | tape576p | 1280x1024 | 200 | 35.3 | **5.7** | **8.8** (8.22) | not measured | 3.40 / 0.43 | n/a | Clean, noise and combing gone, water texture invented but steady; brighter, cooler colour (+3 RGB) |
| FlashVSR+ Turbo `--cf` | 4090 Win, Triton | tape576p | 1280x1024 | 200 | 41.6 | 4.8 | 8.7 (8.22) | not measured | 3.05 / **0.10** | n/a | Colour correction removes the shift (+0.1 RGB) and the brightness pumping |
| FlashVSR+ Base | 4090 Win, Triton | tape576p | 1280x1024 | 200 | 36.7 | 5.4 | 12.9 (12.24) | not measured | 3.71 / 0.38 | n/a | Slightly sharper than Turbo, larger colour shift (+6 RGB) |
| FlashVSR+ Turbo | 4090 Win, SDPA | tape576p | 768x640 | 49 | 8.5 | 5.8 | 14.3 (13.56) | not measured | not measured | n/a | SDPA refuses 1280x1024 (dense-mask guard) |
| FlashVSR+ Turbo | 4090 Win, Triton | pattern576p | 1280x1024 | 75 | 13.7 | 5.5 | 8.8 | not measured | 1.45 / 0.24 | **text n** | Changing timecode digits smeared and wrong ("01.200" read "01.000"); saturated bars dulled |
| Fast general | 4090 Win | pattern576p | 1280x1024 | 75 | 2.9 | 25.7 | 0.3 | not measured | 0.31 / 0.01 | text y | |
| Fast general | 4090 Win | ai540c | 1920x1024 | 192 | 10.2 | 18.9 | 0.4 (0.36) | not measured | 0.99 / 0.02 | face y (waxy) | |
| Fast x4plus | 4090 Win | ai540c | 1920x1024 | 192 | 70.2 | 2.7 | ~6.4 (3.42) | not measured | 0.89 / 0.02 | face y (waxy) | |
| **FlashVSR+ Turbo** | 4090 Win, Triton | ai540c | 1920x1024 | 192 | 50.1 | **3.8** | **11.2** (10.49) | not measured | 2.12 / 0.29 | face y, see note | Real hair strands and skin detail; adds crow's-feet lines the source barely shows; stable over frames; clean scene cut |
| FlashVSR+ Turbo `--temporal-compression 2` | 4090 Win, Triton | ai540c | 1920x1024 | 192 | 82.7 | 2.3 | 11.0 (10.49) | not measured | 1.88 / 0.30 | face y | 1.65x slower for slightly steadier motion |
| FlashVSR+ Base | 4090 Win, Triton | ai540c | 1920x1024 | 192 | 56.9 | 3.4 | 17.2 (16.41) | not measured | 2.39 / 0.29 | face y, see note | Marginally sharper than Turbo |
| Fast general | 4090 Win | ai720c | 2560x1408 | 193 | 17.3 | 11.2 | 0.7 (0.65) | not measured | 1.19 / 0.03 | n/a | |
| Fast x4plus | 4090 Win | ai720c | 2560x1408 | 193 | 134.2 | 1.4 | ~6.4 (5.89) | not measured | 1.10 / 0.02 | n/a | |
| **FlashVSR+ Turbo** | 4090 Win, Triton | ai720c | 2560x1408 | 193 | 91.0 | **2.1** | **17.2** (16.43) | not measured | 2.30 / 0.32 | n/a | Denim weave and foliage detail; same VRAM with `--decode-every 28` (17.4 GB) |
| FlashVSR+ Base | 4090 Win, Triton | ai720c | 2560x1408 | 193 | 156.0 | 1.2 | 14.2 (13.43) | not measured | 2.62 / 0.39 | n/a | |
| FlashVSR+ Turbo | 4090 Win, Triton | ai720c | 3840x2176 | 193 | killed after 17 min | none | 23.9 + 10 GB shared | not measured | none | n/a | `auto` input mode: spills into system memory, no progress |
| FlashVSR+ Turbo `--decode-every 28 --cache-offload-mode always` | 4090 Win, Triton | ai720c | 3840x2176 | 49 | killed after 16 min | none | 23.9 + 12.5 GB shared | not measured | none | n/a | Still spills; 4K needs spatial tiles or a Linux check |
| SeedVR2 3B | none | none | 1920x1080 | none | not measured | not measured | not measured | not measured | not measured | not measured | Not run (section 5) |
| SeedVR2 3B / 7B | none | none | 3840x2160 | none | not measured | not measured | not measured | not measured | not measured | not measured | Not run |

**Flicker columns.** "Flicker low" is the mean absolute difference between the output's frame-to-frame change and a lanczos upscale's frame-to-frame change, on luma averaged in 8x8 blocks (0 to 255 scale). "Pump" is the mean frame-to-frame change of the output's overall brightness offset against lanczos. Both rise when a model adds detail that lanczos lacks, so FlashVSR+ scores worse than the fast tier on these numbers. The eye says the opposite: on five consecutive crops (`frames\strip_tape_fv_vs_fast.png`) the FlashVSR+ water ripples hold steady, while the fast tier's streaks and blotches change every frame. The largest brightness jumps are at hard cuts in the source (ai540c frames 165 to 166, tape 122 to 124), not every chunk; no periodic chunk flicker was found at periods 4, 8, 16, 28 or 32 (`scripts\pump.py`). The real temporal problem was colour pumping, and `--cf` fixes it (0.43 to 0.10).

**Detail (mean absolute Laplacian, lanczos in brackets):** tape: fast general 2.69 (3.67, it smooths away the noise and detail), FlashVSR+ Turbo 3.69; ai540c: fast general 3.59 (2.14), FlashVSR+ Turbo 3.81, Base 3.97; ai720c: fast general 3.19 (2.05), Turbo 3.17, Base 3.68. The numbers are close; the difference you see is in kind: Real-ESRGAN sharpens edges and paints flat areas, while FlashVSR+ produces texture (hair, denim, ripples).

**Frames to look at** (`E:\tmp\upscale-spike\frames\`): `tape_sheet.png`, `tape_lanczos_f100.png` against `tape_fv_turbo_..._f100.png` (the clearest case), `ai540_sheet.png` (face), `ai720_sheet.png`, `strip_tape_fv_vs_fast.png` (temporal), `pattern_text_fast_vs_fv.png` (text), `cut_*.png` (scene cut). All outputs are in `out\`.

**Time per job at the measured rates (Turbo, `--cf`, 4090):** a 3-minute 24 fps film (4,320 frames) 720p to 1440p takes about 35 minutes, or 19 minutes 540p to 1080p; a 60-minute PAL tape (90,000 frames after `bwdif`) to 1280x1024 takes about 5.2 hours. The fast tier does the same tape in about 1 hour.

## 5. SeedVR2: why it was not run

The plan made it the fallback "if FlashVSR+ fails"; FlashVSR+ did not fail. Reading the official code (`projects/inference_seedvr2_3b.py`, `models/dit_v2/attention.py`, `projects/video_diffusion_sr/infer.py`) showed what running it on Windows would take: a flash-attn varlen shim on SDPA, swapping the NCCL distributed init for a single process, patching both `torch.load` calls to `weights_only=True` for a 13.6 GB pickle, replacing `torchvision.io.read_video` and the 2024 pins, and writing our own chunking with overlap, because the script sends the whole clip through the DiT and VAE in one pass, which does not fit 24 GB at 1080p for 200 frames. That is the integration work itself, not a spike, and it would have cost another 14.6 GB of downloads. Licences as printed: the SeedVR2-3B card says `license: apache-2.0` and "SeedVR and SeedVR2 are licensed under the Apache 2.0."; the GitHub repo carries the Apache License 2.0; `cmeka/SeedVR2-GGUF` says `license: apache-2.0` with a community base model. SeedVR2 stays the only candidate for a Mac quality lane, which this spike did not touch.

## 6. Recommendation

- **Engine:** FlashVSR+ **Turbo** as the only quality row at first. Base is barely sharper, 1.1 to 1.7 times slower and up to 6 GB hungrier, so it is not worth a second 7 GB download yet. Turn `--cf` (colour correction) on by default. Offer `--temporal-compression 2` as a "fast motion" option later, not at first.
- **Platform:** native Windows and Linux, NVIDIA only. No WSL2. Its own add-on venv: Python 3.12, torch 2.7.1+cu128, `triton-windows==3.3.1.post21` on Windows (official `triton` on Linux comes with torch), plus the eight pinned packages above. Download about 3.3 GB of torch, about 0.1 GB of other wheels and 7.0 GB of weights.
- **Fit gates (measured peak, Turbo, plus the CUDA context):** output up to 1280x1024: 10 GB; up to 1920x1088: 12 GB; up to 2560x1408: 18 GB. 4K: not offered (it spills on 24 GB under Windows; it would need our own spatial tiles with overlap, roughly 4 times the time, or a Linux re-test). Cards under 12 GB: fast tier only.
- **Integration size: L.** (1) The add-on installer with its own venv and the Triton pin bound to torch. (2) A worker that drives `build_inference_pipeline` with tensors (`InferenceRequest(lq_video=...)`), not its CLI. The shipped `Upscaler` contract (a frames-in, frames-out callable) does not fit a streaming model with a KV cache, so the quality runner works per segment, with a few frames of warm-up overlap at each segment start and the multiple-of-128 pad and crop handled inside, then lanczos to the exact target as the fast tier does. (3) Our own mux (not the CLI's `-shortest`). (4) Resume at segment granularity. (5) Catalog row, fit gate, Setup route, E2E phase. (6) The Storyboard per-shot path is unchanged.
- **Release:** the **second release**, not v0.10.0. v0.10.0 is already at changelog stage; the quality tier is a new 10 GB add-on with a fresh, 5-week-old upstream and a torch-bound Triton fork, and it needs its own E2E phase and a real tape check. The fast tier covers v0.10.0.

## 7. Risks and what was not tested

- **Invented detail.** It adds wrinkles around the eyes on the ai540 face and makes up water texture. That is acceptable for AI footage; for family tapes, show a before-and-after preview and keep the fast tier as the faithful choice.
- **Text.** Burnt-in text that changes (a timecode) is mangled, and static text was not checked separately. The plan already draws Storyboard captions after upscaling; warn on the standalone tool.
- **Colour.** +2 to +6 RGB brighter and cooler without `--cf`; fixed with it.
- **Frame count and sync.** The CLI drops frames when the audio is shorter than the video (`-shortest`). Our pipeline must count frames and mux itself.
- **4K** not possible on Windows 24 GB as shipped. The authors' "4K untiled on a 24 GB 4090" claim was not reproduced (Windows WDDM and about 1.1 GB of desktop use; Linux not tried).
- **Fragility.** Pin the upstream by commit. `triton-windows` must move with torch. First-run Triton compile is in seconds and should need no compiler on the user's machine (the package says it bundles what it needs; not checked), but this was only seen on a machine that has the CUDA toolkit installed; **a clean machine without the CUDA toolkit was not tested.**
- **Not tested:** WSL2 (not needed); Linux speed with official Triton and Block-Sparse-Attention; a real 576i tape capture with faces (the stand-in is degraded AI footage); LPIPS or PSNR against ground truth; clips longer than 8 s (memory drift); cancel and resume; RTX 30 or 50 series and 12 to 16 GB cards; SeedVR2 entirely; any Mac lane.

## 8. Side finding in the shipped fast tier

`backend_service/video_upscale/pipeline.py` `_mux` ends with `os.replace(tmp, output)` from the work folder (`~/.chaosengine/cache/video-upscale/...`) to the output path. When the output is on another drive (the `videoOutputsDirectory` setting can point there), Windows refuses with `WinError 17` ("cannot move the file to a different disk drive") after the whole job has run. Reproduced in this spike with a work folder on C: and output on E:. A `shutil.move` fallback (or a temp file next to the output) fixes it. Not changed here.

## 9. Downloads

About 17.8 GB in total: FlashVSR+ weights 14.35 GB (both variants and the optional VAE; Turbo alone needs 7.0 GB), the torch 2.7.1+cu128 wheel 3.27 GB, other wheels about 0.15 GB (triton-windows included), code archives 11.6 MB (FlashVSR+ 8.9 MB, SeedVR 2.7 MB). SeedVR2 weights were not downloaded. Nothing went into the shared Hugging Face cache or the app's venv.
