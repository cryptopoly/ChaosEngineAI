# v0.10.0 S-R3: stable-diffusion.cpp pin bump 5dcfa79 to 532d2c1

Branch `v0100/bump-sdcpp2`, worktree `/Users/dan/ChaosEngineAI-v0100-bump-sdcpp2`, cut from `origin/release/v0.10.0` at `f32a88a8`. Same method as [v0100-bump-sdcpp.md](v0100-bump-sdcpp.md).

## State

- [x] `check-upstreams.py` rerun 2026-11-16: newest `master` is `532d2c15c430a785a712b2d94281f04e733008a4` (2026-11-15, master-1106, 2 commits ahead of `5dcfa79`). After the bump it reports `ok`.
- [x] Commits read (below).
- [x] Old binary kept as `~/.chaosengine/bin/sd.bak-20261116` (plus `sd.version.bak-20261116`); reports `master-1104-5dcfa79`.
- [x] `./scripts/build-sdcpp.sh` at `532d2c1` (Metal): built and installed, `sd --version` reports `master-1106-532d2c1`.
- [x] Qwen-Image 2.1 image on both builds: pass, bit-identical (below).
- [x] Tests: `pytest tests/test_sdcpp_image.py tests/test_sdcpp_video.py tests/test_sdcpp_progress.py tests/test_licence_gate.py tests/test_model_fit.py tests/test_check_upstreams.py`: 172 passed. `tests/test_hub_tracker.py`: 5 passed.
- [x] Pin (`build-sdcpp.sh`, `update-sdcpp.sh`), comments, CLAUDE.md row, `check-upstreams.py` note, THIRD_PARTY_NOTICES. Hub, FOLLOW_UPS (FU-166 still says `5dcfa79`) and plans not touched by rule.

## What changed upstream

- `01c463a` docs: adds a C# binding to `docs/ecosystem.md`. Nothing for us.
- `532d2c1` refactor(api) #2365: the C API moves to opaque options, tensors and results (91 files). For the CLI:
  - Flag set unchanged: the option-name strings in `examples/common/common.cpp` are identical at both commits, and `sd --help` old vs new differs only in help text.
  - Defaults move into the library: an omitted `--seed` is now random (was 42), an omitted `--video-frames` is 6 (was 1), `--threads`, `--conditioning-cache-size` and `--skip-layers` use library defaults. The app always passes `--seed` (image and both video builders) and `--video-frames` (both video builders) and never passes the other three, so no code change.
  - Progress output unchanged (`src/core/util.cpp` only renames types); `parse_step` read all 40 steps off the new log.
  - `docs/models/minimax_h3.md`: two lines of C API wording; the H3 command line and flags are unchanged, so `_build_h3_cli_args` stands. No change to `qwen_image_2.1.md`, `flux.md`, `wan.md` or any other model doc.
  - Still logs `--fa is deprecated` and the `auto-fit` plan (all three models on Metal), as at `5dcfa79`.

## Result: Qwen-Image 2.1 sd.cpp, old vs new build

Row `leejet/Qwen-Image-2.1-GGUF-q4k`, argument list from `SdCppImageEngine._build_cli_args` (model files resolved offline from the HF cache, nothing downloaded): `--diffusion-model <Q4_K gguf> -p ... -W 1024 -H 1024 --steps 40 --cfg-scale 1 --seed 42 -o ... --vae <vae> --llm <Qwen3-VL-8B Q4_K_M> --sampling-method euler --fa`, same fox prompt as the previous bump. New build first, then old, back to back. The old build's output from 2026-11-14 no longer exists, so the old binary was rerun.

| Build | Wall | Sampling | Decode | Peak memory footprint | Output |
| --- | --- | --- | --- | --- | --- |
| `5dcfa79` (old, `.bak`) | 895 s | 871 s (about 21.8 s/step) | 18.9 s | 11.10 GB | fox, sign reads "ChaosEngine" |
| `532d2c1` (new) | 720 s | 701 s (about 17.5 s/step) | 14.6 s | 11.09 GB | same picture |

Pixel comparison: bit-identical (mean absolute difference 0, max 0). Looked at the picture: sane photo, fox in snow at sunrise, sign reads "ChaosEngine".

Timings are indicative only: another agent's `llama-server` and a busy Chrome GPU process were running during both runs (per-step rate swung 13 to 31 s/step), and both runs were slower than the clean 510 s at `5dcfa79` on 2026-11-14. With identical output and an API-only refactor, no real speed change is expected; the 20 % gap here is load noise.

## Not tested

- CUDA, Windows and Linux builds of `532d2c1`; `--offload-to-cpu` and `--auto-fit` off macOS.
- FLUX.1 schnell sd.cpp row and any video run (no Wan app path; H3 never run, by rule).
- The app's own Images tab end to end (same argument list from the CLI, not through the backend).
- MiniMax H3, by rule.

## Noted, not changed

- `backend_service/catalog/video_minimax_h3.py` docstring still says "the pinned sd.cpp build predates its support (upstream 32d516c)" for H3-Image; that has been untrue since the `5dcfa79` bump. Pre-existing, left alone.
