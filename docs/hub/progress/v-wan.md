---
title: "Progress: Wan 2.2 image-to-video (D-48)"
status: in progress
---
# Progress: Wan 2.2 image-to-video (D-48)

Updated: 2026-11-07 10:40. Owner: Claude agent. Branch: `feat/wan-i2v` (from `feat/home-studio`).

## Goal

Measure whether Wan 2.2 image-to-video (start + end frame) is usable on the owner's M4 Max 64 GB against LTX-2.3 q4,
then wire Wan start / end images into Studio, the storyboard and the agent bridge.

## Done

- Candidate choice: Wan 2.2 I2V A14B, lightx2v 4-step distill, GGUF Q4_K_M experts
  (`jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF` @ `338fb8e`) with text encoder and VAE from
  `Wan-AI/Wan2.2-I2V-A14B-Diffusers` @ `596658f`. About 31 GB. The only Wan build that takes an end frame
  (the 5B ignores `last_image`; mlx-video takes a start image only).

- Baseline: LTX-2.5 q4 (the catalog default, already on disk), not LTX-2.3 (not on disk; no download).
- Wiring, ahead of the verdict because it is needed either way (CUDA at least): new catalog row
  `Wan-AI/Wan2.2-I2V-A14B-Diffusers-distill-gguf-q4km` (`ggufFile` + `ggufFile2`), `WanImageToVideoPipeline`
  in the registry, both GGUF experts loaded and kept on the device on MPS, `image=` / `last_image=` from the
  start / end paths (centre crop to the clip shape), shared rule `helpers/video_image_input.py` used by the
  video route, Studio flags, storyboard clips and the agent bridge; Studio Generate waits for a start image.
  Commits 9efae17, 2edcc81; E2E phase 9 case.

## Not done

- Benchmark runs, contact sheets, `docs/research/wan-vs-ltx-bench.md`.
- Measured time into the catalog row and `recommended_settings.py` once timed.

## How to resume

1. `cd` to the worktree, `git checkout feat/wan-i2v`, `git status`.
2. Bench script lives in the agent scratchpad; copy kept at `scripts/bench_wan_i2v.py` once committed.
3. Before any real run: `ps aux | grep -E "music_worker|mflux|ltx|voice_worker|wan"` and `vm_stat`; one heavy job at a time.

## Decisions to propose

- None yet.
