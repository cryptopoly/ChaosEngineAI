---
title: "Progress: Wan 2.2 image-to-video (D-48)"
status: done, in review
---
# Progress: Wan 2.2 image-to-video (D-48)

Updated: 2026-11-07 12:55. Owner: Claude agent. Branch: `feat/wan-i2v` (from `feat/home-studio`).

## Goal

Measure whether Wan 2.2 image-to-video (start + end frame) is usable on the owner's M4 Max 64 GB against LTX,
then wire Wan start / end images into Studio, the storyboard and the agent bridge.

## Done

- Bench and verdict: `docs/research/wan-vs-ltx-bench.md`. LTX-2.5 q4 at 768 x 448 x 33:
  54-58 s, 18.6 GB. Wan 2.2 I2V A14B Q4 distill: does not fit at 768 x 448 (above 57 GB, 8 min of steps);
  at 480 x 272: 98-114 s, 53-55 GB. LTX stays the storyboard default on 64 GB Macs.
- Weights on disk (31.2 GB): `jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF` @ `338fb8e` (two Q4_K_M
  files) and `Wan-AI/Wan2.2-I2V-A14B-Diffusers` @ `596658f` without transformer weights. `refs/main` was written
  by hand for the base repo (the bench download pinned a revision).
- Wiring: catalog row `Wan-AI/Wan2.2-I2V-A14B-Diffusers-distill-gguf-q4km` (`ggufFile` + `ggufFile2`),
  `WanImageToVideoPipeline`, both GGUF experts on the device on MPS, `image=` / `last_image=`, MPS cache freed
  before VAE passes, shared rule `helpers/video_image_input.py` (route, Studio flags, storyboard, agent bridge),
  Studio Generate waits for a start image, measured settings for `apple-64`, E2E phase 9 case.
- Live check through the app (port 8894): start + end clip at 480 x 272 x 33 in 103 s.

## Not done

- CUDA run (no machine). MLX Wan ports not timed. Attention split not shipped.
- Test clip left in the owner's video gallery: `~/.chaosengine/videos/outputs/2026-11-07/vid-61fd18a75777.mp4`.

## How to resume

1. `cd` to the worktree, `git checkout feat/wan-i2v`, `git status`.
2. Tests: `.venv/bin/python -m pytest tests/test_wan_i2v.py tests/test_video_runtime.py tests/test_media_input.py -q`
   and `npx vitest run src/features/video`.
3. Bench: `scripts/bench-wan-i2v.py` (see the doc's Reproduce section). Before any real run:
   `ps aux | grep -E "music_worker|mflux|ltx|voice_worker|wan"` and `vm_stat`; one heavy job at a time.

## Decisions to propose

- D-48 follow-up: keep LTX as the storyboard and Studio default on 64 GB Macs; offer Wan as an opt-in
  image-to-video model with its measured 480 x 272 cost, and as the stronger choice on CUDA. Reason: memory, not
  speed (Wan needs 53-55 GB at a third of LTX's picture size).
- Optional: time an MLX Wan port (start image only) before ruling out Wan on Macs for good.
