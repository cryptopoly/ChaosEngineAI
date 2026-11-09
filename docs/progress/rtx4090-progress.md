# RTX 4090 test progress — v0.10.0

Windows native test run of `release/v0.10.0` on branch `test/rtx4090`. One line of evidence per item, marked PASS / FAIL / SKIP. Plan: [rtx4090-test.md](../plans/rtx4090-test.md).

## Section 1 flags — re-test after merging release/v0.10.0

Merged `release/v0.10.0` into `test/rtx4090` (merge commit; my section-0/1 work preserved) and restarted the backend from the merged code (`/api/health` 200). Re-ran `pytest tests/test_ffmpeg_fetch.py tests/test_mflux_worker.py -q`.

| Flag | After merge | After my fix | Evidence |
|---|---|---|---|
| Warm-worker crash restart (Errno 22) | **FAIL** | **PASS** | The Mac fix (swallow the pipe-close OSError, commit `8fee49d7`) was necessary but not sufficient. On Windows a worker that calls `os._exit` still reads `Popen.poll() == None` (proven with a repro), so `_ensure_started` reused the dead process and the next `stdin.write` raised `OSError 22`. Fixed in [warm_worker_client.py](../../backend_service/warm_worker_client.py): a `_crashed` flag set by the reader thread when a worker's output stream closes (the reliable death signal) now gates the reuse check, so the next request restarts the worker. `tests/test_mflux_worker.py` 21/21 pass; no regression in `test_ltx2_worker`, `test_music_worker`, `test_voice_io`. |
| ffmpeg archive path-traversal (zip-slip) | **FAIL** | **PASS** | **App code was already correct on Windows** — proven: an archive whose member keeps a literal backslash (`orig_filename == 'x\\bin\\ffmpeg'`) is rejected by `install.extract_members` with "unsafe path" (the Mac fix checking `orig_filename`, commit `4a40fc20`, holds). The two tests failed only because their `make_zip` used `zipfile.writestr`, and on Windows `ZipInfo` rewrites `\`→`/` at **write** time, so the malicious member was never stored. Fixed the test builder in [test_ffmpeg_fetch.py](../../tests/test_ffmpeg_fetch.py) to store names verbatim (reproduce an off-Windows-built archive), so the tests now genuinely exercise the Windows rejection. No app change. |

Both flags now pass on Windows (`tests/test_ffmpeg_fetch.py tests/test_mflux_worker.py -q` → 56 passed, exit 0).


## Machine

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| Driver | 591.86 |
| VRAM | 24564 MiB (24 GB) |
| RAM | 63.8 GB |
| Windows build | 10.0.26300 (26300) |
| Python (venv) | 3.12.10 (3.11 not installed; 3.12 is in the test matrix) |
| Free disk C: | 81.5 GB at start |
| CUDA (torch) | torch 2.11.0+cu128, torchvision 0.26.0+cu128, `cuda.is_available()` True, GPU matmul OK |

## 0. Setup

| Item | Result | Evidence |
|---|---|---|
| Fetch + check out `release/v0.10.0`, branch `test/rtx4090` | PASS | `git checkout -b test/rtx4090` from `release/v0.10.0` (up to date with origin); on `test/rtx4090`. |
| Record GPU, driver, VRAM, RAM, Windows build, Python, free disk | PASS | See Machine table above. |
| Create venv + install per README (CUDA extras, `-c release-constraints.txt`) | PASS | `pip install -e .[desktop,images,dev,ffmpeg] -c release-constraints.txt` (exit 0). Deviations: (1) Python 3.11 not on box → used 3.12.10. (2) PyPI torch is CPU-only on Windows; constraints pin torch==2.11.0, which has no cu124 wheel (cu124 tops at 2.6.0) → reinstalled `torch==2.11.0 torchvision==0.26.0` from the cu128 index (`--no-deps`), matching the Setup route's cu124→cu126→cu128 fallback. vLLM/mlx/triattention extras skipped (no Windows wheels). |
| Backend starts, `/api/health` answers | PASS | `python -m backend_service.app` → `http://127.0.0.1:8876/api/health` returns `{"status":"ok","appVersion":"0.10.0",...,"ggufAvailable":true,"mlxAvailable":false}`. llama-server resolved; mlx false as expected on Windows. |

## 1. Automated checks (Windows native)

| Item | Result | Evidence |
|---|---|---|
| `pytest tests/ -q` | FAIL (28 failed, rest pass) | 28 of ~3600 fail; exit 1. **None is a confirmed app-logic bug** — all are test-portability, missing-dep, CUDA-driver, platform-gating, or 2 flag items (see breakdown below). |
| `npm test` (vitest) | FAIL (4 failed, 1487 passed) | 2 files fail; all test-only. See breakdown below. |
| `npx tsc --noEmit` | PASS | Exit 0, no type errors. |
| `npm run i18n:validate` | PASS | Exit 0, `[i18n-validate] OK` (mode = warn-only). en 100%, 9 shipping locales 88.4%, tlh 93.5% (experimental). The 88.4% sits below the 95% target (CLAUDE.md §6) but the gate is warn-only and keys fall back to en — pre-existing, not a Windows issue. Confirms the 3 `validateGate` vitest fails are harness-only. |
| `e2e_test_suite.py --smoke` | FAIL (phases 4, 5) | Needs a running backend on 8876 (does not spawn its own). Phases 0,3,6,7,8 PASS; 4,5 FAIL on Apple-only routing checks (see below). Report: `~/.chaosengine/test-results/e2e-20261109-150924.{json,md}`. |
| `e2e_test_suite.py` (full) | FAIL (phases 4, 5, 17) | 14 phases PASS, phase 2 SKIP, phases 4/5/17 FAIL. **Real CUDA image + video `generate` checks PASSED** (image 58.6s, video 42.7s). Report: `~/.chaosengine/test-results/e2e-20261109-151244.{json,md}`. |

### e2e full — phase results

PASS (14): 0 (backend/catalog, 17), 1 (automated-checks surface, 18), 3 (challenges, 2), 6 (12), 7 (4), 8 (16), 9 (agent bridge round-trip, 8), 10 (storyboard, 7), 11 (music, 5), 14 (4), 15 (7), 16 (6), 18 (ffmpeg video tools, 4), 19 (skills, 6). SKIP: 2 (Chat Compare — needs models).

FAIL (3), none a confirmed app bug:
- **Phase 4 — Image Studio.** Only fail: `new image families (Qwen-Image / Z-Image / FLUX.2)` → `Qwen/Qwen-Image-2.1-mflux missing or not routed to mflux`. mflux is Apple-only; the check hard-`fail`s on Windows instead of `skip` (its sibling `generate via mflux` correctly skips with "mflux needs Apple Silicon"). **The real `generate` check PASSED (71.6s smoke / diffusers on CUDA).** Suite portability: this check should skip off Apple.
- **Phase 5 — Video Studio.** Only fail: `LTX-2.5 on ltx-2-mlx (routing + live)` → `dgrauet/ltx-2.5-mlx-q4 is missing from the video catalog or not routed to ltx-2-mlx`. ltx-2-mlx is Apple-only (MLX). **The real `generate` check PASSED (43.9s on CUDA).** Same suite-portability issue — should skip off Apple.
- **Phase 17 — Agent bridge v2.** Only fail: `resources and prompts list and read; bad ids refused` → actual prompts `['short_film_from_brief', 'poster_set', 'soundtrack_for_project', 'start_here']`. The check's expected prompt set differs from the current MCP prompt registry. **Not Windows-specific — would fail on the Mac too**; e2e-check vs registry drift. Flag for the Mac session.

## Section 1 verdict

The app runs natively on Windows + RTX 4090: backend boots, CUDA torch is live on the GPU, `tsc` and `i18n:validate` are clean, and the e2e suite's **real image and video generation ran on CUDA** (image ~72 s, video ~43 s). The failing checks (28 pytest, 4 vitest, 3 e2e phases) are **not confirmed app-logic bugs** — they split into test-portability bugs (hardcoded `/`, `PosixPath`, `signal.SIGKILL`, a doubled-drive `C:\C:\` spawn path), optional audio deps not installed in this from-source venv (`soundfile`, `librosa`), CUDA-13-driver stale tests (real driver 591 → cu130-first), Apple/MLX platform-gating in tests and in the e2e suite (mflux + ltx-2-mlx checks `fail` instead of `skip` off Apple), and one registry-drift e2e check. Two items need a real look.

### No fixes applied here (owner asked to stop after section 1 and summarise; more code lands later). Nothing on this branch beyond this progress file.

### For the Mac session

**Investigate (possible real bugs, not test-only):**
- `install.extract_members` zip-slip guard: one unsafe archive member is not rejected on Windows (`test_ffmpeg_fetch::Extract::test_path_traversal_is_rejected`). The ffmpeg BtbN extractor is a Windows/Linux-facing feature (D-49). **Security.**
- Warm mflux/LTX worker crash handling on Windows surfaces `OSError: [Errno 22]` instead of a clean `WarmWorkerError` (`test_mflux_worker::ProtocolTests`).

**Suite portability (small, deliberate):**
- e2e phases 4/5: make the Apple-only `new image families` (mflux) and `LTX-2.5 ltx-2-mlx` checks `skip` off Apple (match their sibling skips) so Windows runs are green.
- pytest G1: tests that hardcode `/`, build `PosixPath(...)`, or send `signal.SIGKILL` need `os.sep`/`Path`/`skipif(win)`; vitest `validateGate` builds a `C:\C:\…` spawn path; `VideoStudioTab` hardcodes `Apple Silicon (detected)`.
- The storyboard resumable-render feature is **unverified on Windows** (its test can't send SIGKILL) — needs a Windows-native kill path to actually exercise it.

**Not Windows-specific (would fail on the Mac/CI too):**
- 5 `test_setup_routes` CUDA-index tests assume cu124-first; on a driver-580+ box the app correctly returns cu130-first. The install-path tests should mock `nvidia_driver_major`.
- e2e phase 17 MCP prompt-set mismatch — update the check to the current prompt registry.

**Environment (would pass with a fuller install):**
- Install `soundfile`, `librosa` (+ the voice runtime) to clear the 4 G2 voice/audio pytest fails; these exercise code the base `desktop,images,dev,ffmpeg` venv doesn't pull in.

### pytest — 28 failures, classified

Progress bar reached 100%; the final count bar did not print under `-q`, but 28 `FAILED` lines were emitted and exit was 1. Classification by cause:

**G1 — Windows-only TEST bugs (test code hardcodes `/`, `PosixPath`, or `signal.SIGKILL`; app logic unaffected):** 13
- `test_measure_voice_models::PlanTests::test_each_wave2_row_is_measured_its_usual_way` — `\c\conds.safetensors` vs expected `/c/conds.safetensors`.
- `test_media_input::LtxCommandTests::test_start_and_end_images_anchor_first_and_last_frame` — `\o.mp4` vs `/o.mp4`.
- `test_media_input::ReferenceImageTests::{test_flux3_uses_the_edit_command_with_all_images, test_init_image_command, test_no_reference_keeps_the_normal_command}` — command-path assertions fail on `\` vs `/`.
- `test_music_sound_1_0::EngineArgumentTests::test_each_model_reads_its_own_folder` — `\data\...\model_config.json` vs `/data/...`.
- `test_voice_cuda_lane::RoutingTests::test_a_gpu_box_runs_the_cuda_row_of_each_family` — `\snap\...gguf` vs `/snap/...`. **CUDA routing logic itself works** (plan built, engine=torch device=cuda); only the test's hardcoded `/` path fails.
- `test_voice_model_files::{RuntimePartsTests::test_four_parts_with_sizes, RuntimeRemovalTests::test_remove_keeps_downloaded_models_by_default, test_remove_with_the_runtime_models, test_the_home_folder_is_never_deleted}` — `NotImplementedError: cannot instantiate 'PosixPath' on your system` (test constructs `PosixPath` directly).
- `test_voice_clone::SafetyTests::test_apply_watermark_refuses_without_the_component_and_calls_the_worker` — `\g.pth` vs `/g.pth`.
- `test_storyboard_v2::RenderV2Tests::test_a_killed_render_is_interrupted_and_resumes_from_its_finished_shots` — `AttributeError: module 'signal' has no attribute 'SIGKILL'` (test sends `signal.SIGKILL`, absent on Windows). The resume feature itself is **unverified on Windows**, not proven broken.

**G2 — Missing optional audio deps in this from-source venv (base-install only; installed `desktop,images,dev,ffmpeg`, not voice/audio extras). Would pass with the voice venv / `soundfile`+`librosa`:** 4
- `test_music_sound_1_0::EngineArgumentTests::test_the_engine_holds_a_clip_to_the_loaded_models_length` — `ModuleNotFoundError: No module named 'soundfile'`.
- `test_voice_watermark::PlumbingTests::{test_detect_reports_probability_and_message, test_watermark_keeps_length_and_rate_and_adds_the_signal}` — `ModuleNotFoundError: No module named 'librosa'`.
- `test_voice_io::InstalledMarkerTests::test_capabilities_report_when_the_pack_passed_its_check` — fails inside `voice_env.tts_data_installed()` (voice-runtime probe; voice stack not installed in this venv).

**G3 — Real NVIDIA CUDA-13 driver env (driver 591 → CUDA 13). `cuda_torch_indexes()` correctly returns cu130-first on driver ≥580; these install-path tests do not mock the driver, so they assume the default cu124-first order. Would fail on ANY driver-580+ box, not Windows-specific; app behaviour is correct:** 5
- `test_setup_routes::SetupRouteTests::{test_cuda_torch_order_follows_the_driver_and_cpu, test_install_cuda_torch_falls_through_to_later_indexes, test_install_cuda_torch_reports_failure_after_all_attempts, test_install_cuda_torch_stops_at_first_success, test_install_cuda_torch_targets_extras_directory}`.

**G4 — Platform-gating divergences (test assumes Apple/MLX or Mac unified-memory; app behaves correctly on Windows+CUDA):** 4
- `test_size_limits::{ImageRouteGateTests::test_an_oversize_request_is_refused_on_a_small_machine, VideoGateRouteHelperTests::test_a_big_machine_runs_it}` — the memory gate on a CUDA host uses **GPU VRAM** (FU-133), but these tests mock system RAM, so the 503/200 does not match.
- `test_media_input::CatalogFlagsTests::test_video_payloads_flag_ltx2_only` — `KeyError: 'dgrauet/ltx-2.5-mlx-q4'` (MLX-only LTX-2.5 row not present in the Windows catalog).
- `test_voice_speak::VoiceListTests::test_one_list_with_builtin_preset_design_and_clone_voices` — `54 != 49`: the 5 Japanese Kokoro voices are excluded only on MLX; on Windows (no MLX) all 54 are listed.

**G5 — Possible REAL Windows gaps (NOT test-only; flag for the Mac session):** 2
- `test_ffmpeg_fetch::Extract::test_path_traversal_is_rejected` — `FetchError not raised`: one unsafe archive member (`/abs/bin/ffmpeg`, `x/../../bin/ffmpeg`, or `x\bin\ffmpeg`) is **not rejected on Windows** by `install.extract_members`. The ffmpeg BtbN extractor is a Windows-facing feature (D-49). **Security (zip-slip) — investigate.**
- `test_mflux_worker::ProtocolTests::test_a_crash_is_reported_and_the_next_request_restarts` — a worker crash surfaces `OSError: [Errno 22] Invalid argument` instead of the clean `WarmWorkerError` the test expects. The warm mflux/LTX worker ships on Windows; crash handling may not be Windows-clean. **Investigate.**

### vitest — 4 failures (2 files), all test-only

- **G1 (Windows-only test bug) ×3** — `src/i18n/__tests__/validateGate.test.ts::{passes when only the experimental locale is below 95 %, still fails when a shipping locale is below 95 %, still reports an ICU error in the experimental locale}`. The test spawns `scripts/i18n-validate.mjs` with a malformed Windows path (`Cannot find module 'C:\C:\Users\Dan\ChaosEngineAI\scripts\i18n-validate.mjs'` — doubled drive), so the subprocess never runs and stdout is empty. The validator itself is fine (`npm run i18n:validate` passes — see below). Test-harness path construction bug.
- **G4 (platform-gating) ×1** — `src/features/video/__tests__/VideoStudioTab.test.tsx::keeps the device and capacity facts as the memory item's tooltip`. Test expects the tooltip to contain `"Apple Silicon (detected)"`; on Windows+CUDA the UI correctly renders `"GPU (detected) · …"`. Mac-hardcoded expectation; app behaviour correct.
