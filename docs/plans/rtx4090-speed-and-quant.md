# RTX 4090: speed-up and quantisation pass

Branch `test/rtx4090`. Record every item PASS / FAIL / SKIP with numbers in `docs/progress/rtx4090-progress.md` (new section per workstream). Commit and push after each workstream; push only `test/rtx4090`. One test suite at a time. Stop and ask before any download over 20 GB, and whenever a model needs a licence or sign-in. No attribution lines in commits. Do not name third-party desktop apps in code or docs.

Hardware must not be assumed to be this card: every default or gate this pass adds has to read the GPU (free memory, compute capability, driver) and work on 12 GB cards up to 5090, DGX Spark and bigger.

## How to measure (all workstreams)

Backend: `CHAOSENGINE_PORT=8876 CHAOSENGINE_VOICE_UNMEASURED=1 .venv/Scripts/python -m backend_service.app`. Drive it over HTTP like `scratchpad/ltx2_live.py` (peak VRAM sampled with `nvidia-smi`, wall time, frames saved). For each row record: generation seconds, peak VRAM, speedup vs the no-cache run, and a quality note (same seed, look at first and late frames, plus a pixel difference score against the baseline: PSNR or mean absolute difference). A speed-up that changes the picture visibly is a FAIL for that setting, with the threshold at which it stopped.

Fixed test set, same seed each time:
- Image: FLUX.2 Klein 4B, 1024x640, its default steps; one more DiT image model if installed (Z-Image or Qwen-Image).
- Video: Wan 2.2 TI2V 5B 832x480 49 frames 20 steps (baseline 42.5 s / 12.4 GB), and LTX-2.3 4-bit 768x512 97 frames (baseline 70 s / 22.3 GB, text mode).

## W1. Diffusion cache strategies on CUDA (the main ask)

Strategies in `cache_compression/`: firstblockcache, teacache, taylorseer, magcache, pab, fastercache.

1. List which strategies the engine reports as available for each Studio on Windows (`applies_to`, `is_available`), and which the Studio actually offers. Record the table.
2. For each strategy x model in the fixed set, run baseline and cached with the same seed. Sweep the strength knob (FBCache threshold 0.08 / 0.12 / 0.20; the others at their documented low / default / high).
3. Check the runtimeNote: when a strategy is silently "not applied" (non-DiT, wrong signature) that is a FAIL of the claim, not a pass. LTX-2 (audio-video transformer) and Wan TI2V (`expand_timesteps`) are the unknowns: confirm the hook attaches, or record the exact error.
4. Distilled models (LTX-2.3 distilled runs 8 steps, FLUX.2 Klein 4 steps): caching has little to skip. Measure it anyway; if it is slower or worse, the Studio should hide the option for those rows (change `fixedSampler` rows to not offer it, with a test).
5. Interaction checks: cache + img2video start picture, cache + start/end pictures, cache + model CPU offload (the 4-bit LTX-2.3 path), cache + Storyboard clip.
6. Fixes found go in with tests. Add an E2E check in `scripts/e2e_test_suite.py` phase 4 / 5 (one cheap cached generation, skip on the host memory gate).
7. Output: a recommended default per model family (strategy + strength, or off), applied in the model defaults, with the measured speed-up in the catalog note.

## W2. Other speed-ups for diffusion models

Measure each against the same baselines; keep what helps without changing the picture, hide or default off what does not.
- `torch.compile` on the transformer (regional compile), including the first-run compile cost.
- Attention backend: SDPA default vs flash / sage attention where a wheel exists for Windows; record the install effort.
- VAE tiling / slicing and decode time (LTX-2.3 and Wan decode are a visible share of the 70 s).
- Fewer steps with the right scheduler for non-distilled rows; the existing model-tuned defaults table is the place to change.
- TeaCache-style skipping is W1; do not repeat it here.

## W3. Quantisation for image and video

- Inventory what is already supported (NF4 via bitsandbytes for LTX-2.3). Add 8-bit and fp8 paths where diffusers supports them: torchao int8 / fp8 weight-only on the transformer, `layerwise casting` (fp8 storage, bf16 compute) which needs no extra package.
- Test on FLUX.2 Klein and Wan TI2V first (small downloads, already installed), then the LTX-2.3 dev row if a bf16 fp8 variant is small enough. For each: VRAM saved, time vs bf16, quality difference vs the bf16 baseline.
- Nunchaku / SVDQuant (4-bit FLUX family): check whether a Windows CUDA wheel exists for this torch; if it does, one install and one test; if not, SKIP with the reason.
- GGUF image / video weights (diffusers `GGUFQuantizationConfig`): one test on a small row if a pack is in the library or under 10 GB; otherwise SKIP.
- NVFP4 / fp8 hardware paths are for Blackwell (5090, Spark): the 4090 cannot run NVFP4 natively. Do not fake it; write the capability gate (compute capability >= 10) and a test for it, mark the live run SKIP (no hardware) and list it as owner-test.
- Output: a table of quant options x model with VRAM / time / quality, and a catalog change only for rows that pass.

## W4. Text-model cache strategies and speculative decoding

- TurboQuant KV cache: context sweep (4K, 16K, 32K, 64K where the card allows) on the CUDA llama.cpp build: tokens/s, VRAM, and a needle-in-haystack answer check, vs the native f16 cache and vs q8_0 / q4_0 native KV types.
- DFlash drafter: acceptance rate and tok/s at the same context sweep, on one more model that has a pinned drafter besides Nemotron 3.5 Bolt, if one is already in the library.
- TriAttention: check whether it can run off the Mac at all (what the install and `is_available` do on Windows). Run it if it can; otherwise record why not.
- DDTree, MTPLX, TensorFold: confirm each is Mac-only by reading the code path (not by assuming), and file a follow-up row in `FOLLOW_UPS.md` for a CUDA lane only if the algorithm is portable. DDTree is a tree-based speculative decoder: say whether a llama.cpp or torch route could exist.
- Re-run `scripts/cache-strategy-matrix.py` (quick, then full) after any change and record which cells now run.

## W5. Leftover PC items

1. Per-step watchdog for video and image generation: a step that exceeds a limit (scaled from the measured step time, with a floor) cancels the run and unloads, with a clear message. The cancel flag today only works between steps, so also add a hard stop path (worker kill) for the one-hour case. Test with a fake pipeline whose step sleeps.
2. Rebuild the installer with all post-installer fixes (`npm run tauri build` path used earlier); install on top and rerun the section 7 smoke (backend starts, GPU runtime installs, torch is the CUDA build). Ask before replacing the installed app.
3. Storyboard "renders appear in the project" failure from section 5: find out why the model chose `generate_image` instead of `storyboard_project`, fix the tool description or routing, retest.
4. Windows test portability: the 38 Python and 4 vitest failures. Take only the trivially Windows-only ones (path separators, `signal.SIGKILL`, `C:\C:\` spawn path); leave Mac-only ones to the Mac session. Report the count before and after.
5. Paste-image check: ask the user to do the hand test.
6. Storyboard with LTX-2.3 as the project default model (stills + end frames to sound clips): one 3-shot run, timing and VRAM.

## Order and gates

W1, then W2 (shares the harness), then W3, W4, W5. After each workstream: relevant pytest files, `npx tsc --noEmit`, `npm run i18n:validate` when UI text changed, then commit and push. Finish with a single "Report back" block in the progress doc: what is faster, what is smaller, what was refused and why, defaults changed, owner tests still needed (Blackwell, Spark, 16 GB and smaller cards).
