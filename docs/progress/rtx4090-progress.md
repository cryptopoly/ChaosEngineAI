# RTX 4090 test progress — v0.10.0

Windows native test run of `release/v0.10.0` on branch `test/rtx4090`. One line of evidence per item, marked PASS / FAIL / SKIP. Plan: [rtx4090-test.md](../plans/rtx4090-test.md).

## Section 1 flags — re-test after merging release/v0.10.0

Merged `release/v0.10.0` into `test/rtx4090` (merge commit; my section-0/1 work preserved) and restarted the backend from the merged code (`/api/health` 200). Re-ran `pytest tests/test_ffmpeg_fetch.py tests/test_mflux_worker.py -q`.

| Flag | After merge | After my fix | Evidence |
|---|---|---|---|
| Warm-worker crash restart (Errno 22) | **FAIL** | **PASS** | The Mac fix (swallow the pipe-close OSError, commit `8fee49d7`) was necessary but not sufficient. On Windows a worker that calls `os._exit` still reads `Popen.poll() == None` (proven with a repro), so `_ensure_started` reused the dead process and the next `stdin.write` raised `OSError 22`. Fixed in [warm_worker_client.py](../../backend_service/warm_worker_client.py): a `_crashed` flag set by the reader thread when a worker's output stream closes (the reliable death signal) now gates the reuse check, so the next request restarts the worker. `tests/test_mflux_worker.py` 21/21 pass; no regression in `test_ltx2_worker`, `test_music_worker`, `test_voice_io`. |
| ffmpeg archive path-traversal (zip-slip) | **FAIL** | **PASS** | **App code was already correct on Windows** — proven: an archive whose member keeps a literal backslash (`orig_filename == 'x\\bin\\ffmpeg'`) is rejected by `install.extract_members` with "unsafe path" (the Mac fix checking `orig_filename`, commit `4a40fc20`, holds). The two tests failed only because their `make_zip` used `zipfile.writestr`, and on Windows `ZipInfo` rewrites `\`→`/` at **write** time, so the malicious member was never stored. Fixed the test builder in [test_ffmpeg_fetch.py](../../tests/test_ffmpeg_fetch.py) to store names verbatim (reproduce an off-Windows-built archive), so the tests now genuinely exercise the Windows rejection. No app change. |

Both flags now pass on Windows (`tests/test_ffmpeg_fetch.py tests/test_mflux_worker.py -q` → 56 passed, exit 0).

## Voice runtime install + the 4 voice/audio pytest failures

Installed the voice runtime through the app's own endpoint (not pip): `POST /api/voice/install {"extras":["kokoro"]}` → job ran to `phase: done, 100%, "voice pack ok"`, no error, ~5 min. Lane = **portable** (Windows: `av`, `numpy`, `faster_whisper`, `kokoro_onnx`, `onnxruntime`, `sherpa_onnx` + CUDA libs for faster-whisper, into the isolated `~/.chaosengine/voice-venv`). `GET /api/voice/runtime` → `installed: true`. No licence/sign-in needed for the base runtime.

**The 4 section-1 voice/audio failures, re-run:**

| Test | After voice runtime install | Note |
|---|---|---|
| `test_voice_io::InstalledMarkerTests::test_capabilities_report_when_the_pack_passed_its_check` | **PASS** | Already cleared by the release/v0.10.0 merge (not the runtime install). |
| `test_music_sound_1_0::EngineArgumentTests::test_the_engine_holds_a_clip_to_the_loaded_models_length` | **not cleared by the runtime install; PASS after `soundfile` in the main venv** | `ModuleNotFoundError: No module named 'soundfile'` at `music_worker_stableaudio.py:222`. |
| `test_voice_watermark::PlumbingTests::test_detect_reports_probability_and_message` | **not cleared by the runtime install; PASS after `librosa` in the main venv** | `ModuleNotFoundError: No module named 'librosa'` at `voice_worker_watermark.py:74`. |
| `test_voice_watermark::PlumbingTests::test_watermark_keeps_length_and_rate_and_adds_the_signal` | **not cleared by the runtime install; PASS after `librosa` in the main venv** | same. |

**Finding:** the voice runtime install does **not** clear the `soundfile`/`librosa` failures, because these tests import those libs **in-process in the main backend venv** (they call `music_worker_stableaudio` / `voice_worker_watermark` directly), while the runtime install only provisions the isolated `voice-venv`. Both libs are pinned in `release-constraints.txt` (`soundfile==0.13.1`, `librosa==0.11.0`) but are pulled by no extra in `.[desktop,images,dev,ffmpeg]`; `pip install soundfile librosa -c release-constraints.txt` into the main venv clears all 3 (3/3 pass). Not app bugs — test-environment provisioning. (Added them to the main venv for the rest of the run.)

**Other voice/audio failures (not among the 4; Windows test-portability, Mac session per the owner's scope note — not fixed here):** `test_voice_model_files` `RuntimeParts` + `RuntimeRemoval`×3 + the merge's new D-75 `AddonRemoval`×4 all raise `NotImplementedError: cannot instantiate 'PosixPath'` (tests build `PosixPath` literals); `test_music_sound_1_0::test_each_model_reads_its_own_folder`, `test_voice_clone::...apply_watermark` (`\g.pth`) are `\`-vs-`/`; `test_voice_cuda_lane`, `test_measure_voice_models` are `\`-vs-`/`; `test_voice_speak::...one_list` is MLX-gating (54 vs 49 Kokoro voices).


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

## 2. Speech (Voice Studio)

Driven through the backend API (`POST /api/voice/speak`) on the running backend; gen time, audio duration and peak GPU VRAM measured per run. Idle GPU baseline ≈ 965 MiB.

| Item | Result | Evidence |
|---|---|---|
| Kokoro: speak a paragraph | PASS | gen 1.67 s for 10.07 s of audio (RTF 0.17), valid 2-chunk WAV, model `kokoro`. Kokoro runs on `kokoro-onnx` (CPU on Windows): VRAM delta 0. |
| Speed + pitch sliders | PASS | speed 1.5 → 6.41 s (from 10.07 s at 1.0); pitch +3 semitones applied (post-processing). |
| Text over 400 characters (chunking) | PASS | ~500-char paragraph → 3 chunks (vs 2 for the shorter one), 27.56 s audio. |
| Qwen3-TTS CustomVoice 1.7B (neutral vs style) | SKIP (not on Windows) | Every Qwen3-TTS row is `lanes=['mlx']`; v0.10.0 has **no CUDA Qwen3-TTS**. The Speech models table says so ("Not for this machine — Apple Silicon Macs only for now"). |
| Qwen3-TTS VoiceDesign 1.7B | SKIP (not on Windows) | same |
| Qwen3-TTS Base: clone from a short sample | SKIP (not on Windows) | same; cloning on this PC goes through the section 2.1 CUDA rows (needs a voice sample from you, own voice only). |
| Qwen3-TTS 0.6B vs 1.7B timing (FU-148 on CUDA) | SKIP (not on Windows) | same — FU-148's "faster on CUDA?" cannot be answered until a CUDA Qwen3-TTS lane exists. |
| Speech-model download Cancel | PASS | Started 3 downloads, `POST /api/voice/clone/models/download/cancel` → each `cancelled`; 6 s later **0** `snapshot_download` processes left. |
| Cancel mid-run (Speak) | _pending_ | Kokoro finishes in ~1.5 s; will test on a slower CUDA model (2.1). |
| Speech to text (default model) — after fix | PASS | Whisper Large v3 Turbo (faster-whisper) on CUDA through `POST /api/voice/transcribe`: **2.66 s cold (incl. load), 0.31 s warm** for 10.07 s of Kokoro speech; exact transcript, language `en`, word timings per segment. Needed the cuBLAS fix below. |
| Speech to text (default model) — first attempt | FAIL → fixed — **two bugs found** | Clicked "Download 1.6 GB" on Whisper Large v3 Turbo; within seconds the card said **DOWNLOADED** while `model.bin` was still `.incomplete` (250 MB of ~1.5 GB). Transcribe then fails: `RuntimeError: Unable to open file 'model.bin'`. Cause: `voice_runtime.list_stt_models` sets `installed` from "snapshot folder exists" (`voice_runtime.py:128`), true as soon as the small config files land. Cross-platform, so left for the Mac session. Re-test once the download completes. |
| Kokoro add-on: untick | PASS | Unticked in Voice runtime > Add-ons: panel now reads "8.0 GB … the Kokoro quick voices are not installed"; Kokoro row "NOT INSTALLED — Tick it under Voice runtime > Add-ons"; `~/.chaosengine/voice/kokoro` removed (the `kokoro_onnx` package stays in the base pack, correct); Studio > Voices: "The Kokoro quick voices are not installed. Tick them under …". Re-tick pending (download). |
| Voice runtime panel: parts, sizes, "In memory now", Unload | PASS (one stale-view bug) | Parts read "8.0 GB on disk: speech packages 8.0 GB, turn detection and speaker labels 0.04 GB" (Kokoro off), matching the measured venv. With Whisper loaded, `GET /api/voice/runtime` reports `inMemory: Whisper Large v3 Turbo` and a `workerPid`, but the panel kept saying "nothing" until a full page reload (the voice context is not refetched after a model loads; cross-platform). After reload: "In memory now: Whisper Large v3 Turbo [Unload]"; Unload → VRAM **4.79 → 2.26 GB**, worker process gone, panel "nothing". |
| Kokoro add-on: tick again (re-download) | PASS | `POST /api/voice/install {"extras":["kokoro"]}` merged with the GPU add-on (kept, not reinstalled): 70 s, `kokoro-v1.0.onnx` + `voices-v1.0.bin` back in `~/.chaosengine/voice/kokoro`, both add-ons report installed. |
| Voices: the Fallback card (live) | PASS | Chat read-aloud = Orpheus `tara`, fallback = Kokoro Heart: read-aloud spoke as `orpheus:tara`. Deleted Orpheus → read-aloud spoke as `af_heart` with the note "Tara could not speak here (Orpheus 3B (GGUF) is not downloaded. Download it in Voice > Models), so the fallback voice Heart (Kokoro) reads instead." Fallback set to none → **409** "The chosen voice could not speak and no fallback voice is set. Pick one on Voice Studio > Voices." (UI of the card itself: Playwright `21-kokoro-optional`, PASS.) |
| Delete a speech model: confirm names it, its size and what another model keeps; folder gone | PASS (size bug fixed) | Delete plan for Orpheus: remove `unsloth/orpheus-3b-0.1-ft-GGUF` + `onnx-community/snac_24khz-ONNX`, keep none; after delete both folders are gone and the worker was stopped. The plan said **7.14 GB** for 3.6 GB on disk — see the size fix below. |
| Delete is refused while Speak is running | PASS | During a long VoxCPM2 job, `POST /api/voice/clone/models/delete {voxcpm2-cuda}` → 409 "Voice is working on a job (speech or a transcript). Try again when it has finished."; model kept. |
| Cancel mid-run (Speak) | PASS | `POST /api/voice/speak/cancel` 8 s into a long VoxCPM2 job → `{"cancelled": true}`; the job ended at once with 499 "Speech was cancelled." |
| Delete a dictation model | PASS | Downloaded Whisper Base (0.14 GB) → listed installed → `POST /api/voice/stt/delete {whisper-base}` → deleted, folder gone, row back to not installed. (The "···" menu itself: Playwright `11-voice-pack`, passes when run alone.) |
| Chat voice call with a microphone | **BLOCKED (needs owner/hardware)** | Needs a real microphone; cannot be driven headless. |
| Voices: Speak, Chat and Call set to different voices | PASS | `POST /api/voice/settings {"speakVoice":"am_michael","ttsVoice":"bf_emma","callVoice":"bm_george"}` → 200, read back unchanged; read-aloud with the chat voice spoke as `bf_emma`, a call line as `bm_george`; the UI passes these (`ReadAloud.tsx`, `callVoiceOf`). Restored to Heart afterwards. A clone id the voice list does not know is refused (400 "Unknown voice …"). |
| Remove voice runtime (Troubleshooting), with and without the Kokoro tick, then reinstall | **NOT RUN** | It deletes the 8 GB voice venv (portable lane + speech-gpu with torch cu128) and re-downloads it at ~2 MB/s, about an hour per pass; left for a session where the download speed allows, or for you to run. |

**New Windows bug fixed — Speak offered a Mac-only download.** With Kokoro off, Voice Studio > Speak defaults to the voice "Ryan" and shows "Download 3.08 GB" for Qwen3-TTS 1.7B CustomVoice. On Windows the backend accepted that download and pulled `mlx-community/...-8bit` weights the CUDA worker cannot load (`start_download` only checked `available_here()`, never whether the row's lanes run here). Fixed in [voice_clone.py](../../backend_service/voice_clone.py): `start_download` now refuses a row that does not `runs_here`, with "X runs on Apple Silicon Macs only, so it cannot be used on this computer." (400). Tests: two new cases in `tests/test_voice_clone.py` (MLX row refused on a CUDA host, CUDA row refused on a Mac, no download started); the download tests that target MLX rows now pin the lane so they don't depend on the host. **Still open (frontend, Mac session):** Speak should not pick a non-runnable default voice on Windows, and the "Download" link should not show for it.

**New Windows bug fixed — dictation crashed on every NVIDIA PC.** Once `model.bin` was complete, transcription failed with `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`. The DLLs are in the voice venv (`nvidia/cublas/bin`, `nvidia/cudnn/bin`), and `voice_worker._add_cuda_dll_dirs` already registers them with `os.add_dll_directory`, but ctranslate2 loads cuBLAS lazily with a plain `LoadLibrary`, which searches **PATH** only. Reproduced outside the app (fails with `add_dll_directory` alone, works with the folders on PATH), then fixed in [voice_worker.py](../../backend_service/voice_worker.py): the same folders are also put on `PATH` once. Tests: `CudaDllPathTests` in `tests/test_voice_io.py` (folders on PATH and registered, no duplicates on a second call, nothing changes off Windows).

**Bug for the Mac session — "stop the voice worker" cannot stop a stuck worker.** After the cuBLAS crash the worker wedged on the next request. `POST /api/voice/worker/stop` calls `VoiceWorker.stop()`, which waits for the request lock that the stuck request holds for its whole timeout, so Stop hung (40 s, no answer) until I ended the worker process by hand; it then answered in 0.1 s. `VoiceWorker.interrupt()` exists for exactly this ("does not wait for the request lock") but the route never calls it. Suggested fix: `interrupt_all()` before `stop_all()` in the route. Cross-platform, so not fixed here.

**Network:** Hugging Face downloads run at ~2 MB/s on this PC (Wi-Fi with a NordLynx/NordVPN tunnel up), which paces everything left in 2.1, 3 and 4.

**Windows voice architecture note:** every torch-based TTS model (Qwen3-TTS, Chatterbox, VoxCPM, Orpheus) needs the **speech-gpu** add-on — `backend_service/voice_worker_torch.py` runs "in the portable voice venv with the speech GPU add-on", and torch is **not** in the base portable voice-venv (only `kokoro-onnx` + faster-whisper are). So on Windows, section 2's Qwen3-TTS checks require installing speech-gpu (torch 2.9.1 cu128 into `~/.chaosengine/voice-venv`), which also unlocks the section 2.1 wave-2 models. Installing it now. No model is gated (all Apache-2.0/MIT), so no licence/sign-in is needed for the downloads.

UI-only checks from the plan (Voice runtime panel Kokoro untick/retick, the Voices Fallback card, the Voice popover, the Voice Models download/cancel/delete table with Task Manager, Delete confirms) need the running frontend/desktop app and are grouped for a browser-pane pass after the generation checks.

## 4. Image and video sizes

| Item | Result | Evidence |
|---|---|---|
| One image model at its largest listed size | PASS | FLUX.2-klein-4B (diffusers on CUDA) at **2048×2048** (its 4 MP `maxPixels`), 24 steps: HTTP 200 in **74.6 s** (load 30.6 s, denoise 39.8 s, decode 1.4 s), **peak VRAM 19.1 GB** of 24. Output looked at: clean, detailed lighthouse, no artefacts. |
| Video model at 720p, 1080p, largest "Tight" (FU-145) | FAIL (hint did not match) | Wan 2.1 T2V 1.3B (the only installed CUDA video model). Video Studio: 480p and 720p selectable, **1080p and up disabled** ("Above this model's size limit", Wan caps at 1280 px). **720p** (1280×704, 33 frames, 30 steps): HTTP 200 but **578.5 s** (load 37.8 s, denoise 536.7 s ≈ 17.9 s/step) and **peak VRAM 24.05 GB — the whole card**; the Studio strip had said **"About 1 min, Peak 16.7 GB of 24 GB"**. Filling the card points to the Windows driver spilling into shared system memory, which explains the crawl. **480p** (896×512): 95.4 s warm (3.1 s/step) vs "About 1 min". New-project form for the same model says **480p Tight, 720p Will not fit, "about 74 GB GB at peak for a 4-second clip"** — the two estimators disagree by 4× and neither matched what happened; also a duplicated unit ("GB GB"). The 720p frame looked at: clean and coherent. |
| Frame rates 24 / 30 / 48 / 60 in a Storyboard project | PASS | New-project form with Wan 2.1 1.3B: every rate shows "Wan 2.1 T2V 1.3B makes 16 fps clips. They are converted to N fps when the film is joined: frames are repeated or dropped, not interpolated." — correct native rate. |
| Export a project at 1080p and 2160p with Upscale | PASS (with imported clips) | Project "RTX 4090 export check" (`~/.chaosengine/storyboards/10a8faeb46d7f53d`), Wan 2.1 1.3B, two 2 s shots. Making clips is refused on Windows (below), so the two Wan clips from the size test were brought in with the `import_take` op. Exports (probed with ffmpeg): 480p project → **854×480**; project set to 1080p → **1920×1080**; 1080p + Upscale 2160p → **3840×2160**; all 4.08 s, H.264 + AAC, 2–6 s each. Warning "loudness not normalised: the mix is silent" (no audio lanes) is expected. |
| Storyboard: make clips on Windows | **FAIL (Mac session / parity)** | `POST …/batches {"what":"clip"}` → 400 "Wan 2.1 T2V 1.3B cannot start a clip from an image, so it would ignore this shot's still or the previous shot. Pick LTX-2.3 or LTX-2.5…", even for shots with no still: `clip_image_refusal(project)` refuses any non-LTX-2 model for the whole batch, and new shots default to starting from a generated still. With no LTX-2 on Windows, **a Windows user cannot make any Storyboard clip**, only import them. Fix belongs with the CUDA image-to-video work (Wan 2.2 I2V / LTX-2 on diffusers) and letting text-only shots through. |
| Video Models: "Slow on this computer" + time one Wan and one LTX clip | PASS (labels) / SKIP (LTX timing) | `document.body.innerText` on Video Models: **0** "Slow on this computer" across all 20 rows. LTX-2 rows are MLX-only and correctly absent on Windows; the only LTX here is LTX-Video (diffusers), not installed (its row lists 236 GB) — not downloaded at ~2 MB/s. Wan timings above. |

**After a video run the GPU stays full:** following the 720p run, idle VRAM read **24.0 GB** with nothing generating — the diffusers pipeline stays resident and PyTorch's cache keeps the card reserved, so chat or voice on the GPU would be starved until the model is unloaded. `POST /api/video/unload` freed it (back to 1.2 GB). Worth an `empty_cache()` after each run, or freeing on idle.

**Storyboard on Windows:** `storyboard/generators.make_clip` only passes a still to LTX-2 (MLX); any other video model refuses a shot that starts from a still or the previous shot (`_no_image_input`, D-40). The Windows/CUDA catalog has no image-to-video model (Wan 2.2 I2V was removed, D-48), so the Storyboard's still → clip flow cannot run on this PC; only text-only clips do. This affects section 5.

## 3. Music and sound

| Item | Result | Evidence |
|---|---|---|
| ACE-Step 1.5: install into the music venv, one 30 s track | SKIP (not on Windows) | `POST /api/music/install` → "Music runs on Apple Silicon (MLX) in this version. The Windows and Linux (CUDA) lane is not built yet." (FU-139). |
| Stable Audio Open: one sound effect | SKIP (not on Windows) | `POST /api/music/sound/install` → "Sound effects run on Apple Silicon in this version. The Windows and Linux lane is not built yet." Licence/token is fine: the app's Hugging Face token passes `auth_check` for both `stabilityai/stable-audio-open-1.0` and `-small`. |
| Stable Audio Open 1.0 at 2 s and 30 s (timing, no timeout) | SKIP (not on Windows) | same |
| Music and Audio Library with 20+ items scrolls to the last card | _see Playwright run_ | Cannot make 20 real items here (no music engine); covered by `e2e/tests/21-library-scroll.spec.ts` in the Windows Playwright run below. |

**Music and Audio Models page on Windows** is mostly honest ("Music needs an Apple Silicon Mac for now. … The Windows and Linux (CUDA) lane is not built yet."), but contradicts itself (Mac session):
- the header still says "Music models run on this computer. Download one, then open the Studio to make a track.";
- both ACE-Step rows carry a **CUDA** platform chip (catalog `platforms: ["apple-silicon","cuda"]`) and XL is rated **"Tight"** while turbo says "Not for this machine";
- not-installed rows show a **Delete model** action;
- Stable Audio rows show "NEEDS LICENCE / Accept licence" although the token already has access.

## Windows harness and backend fixes found while testing

**New Windows bug fixed — the Playwright suite could not start on Windows.** `e2e/playwright.config.ts` launched the backend with a POSIX one-liner (`rm -rf … && mkdir -p … && HOME=… cmd`); Playwright runs `webServer.command` through `cmd.exe` on Windows → `'HOME' is not recognized as an internal or external command`. Now the variables go through the config's `env` option and a new `e2e/start-backend.py` resets the throwaway home (refuses any folder outside `e2e/.run`) and starts the backend; same command on every OS. Also added `e2e/warmup.ts` (`globalSetup`): one page load before the tests, because the first Vite compile on Windows took over 30 s and failed whichever test ran first. `01-smoke.spec.ts`: 3/3 pass on Windows after both.

**New Windows bug fixed — GPU metrics froze the backend on NVIDIA PCs.** `GET /api/metrics/gpu` was `async def` and called the blocking `get_gpu_metrics()` on the event loop, and on an NVIDIA host that snapshot first spawned a Python child that imports torch (2.6 s per call here) before trying `nvidia-smi`. The UI polls it, so every poll stalled every other request: in the Playwright run the "I agree" save on the terms screen never finished (trace: `PATCH /api/settings` and the catalog calls pending behind three `metrics/gpu` calls). Fixed: the route is a plain `def` (thread pool), and `GPUMonitor._snapshot_nvidia` tries `nvidia-smi` first (fast, and it reports utilisation, temperature and power, which the torch path left empty), keeping the torch probe as the fallback when `nvidia-smi` is missing. Tests in `tests/test_gpu.py` (order both ways, route is not a coroutine). The terms test went from FAIL to PASS on the next Windows run.

## Playwright browser E2E on Windows (`cd e2e; npm ci; npx playwright install chromium --only-shell; npx playwright test --workers=1`)

After the two harness fixes above: **124 passed, 8 failed** in 19.0 min (Chromium 158, headless). None of the 8 is an app bug on Windows:

| Spec | Why it fails here |
|---|---|
| `04-music-sounds-licence` ×2 | Expect "The music pack is not installed yet." / "Download Stable Audio Open Small…"; the real Windows backend says music and sounds are Apple-only. Host-dependent spec. |
| `06-busy-advice` ×3 | The mocked library model is `format: "mlx"`; on Windows it shows "Cannot run on this machine" and **Chat in a new thread [disabled]**, so `/api/busy-advice` is never asked (30 s timeout). Host-dependent; a GGUF mock would make it run on every host. |
| `14-voice-speak` (Voices groups…) | Expects the "Add Mandarin Chinese voices" button, an MLX-lane extra not offered on the portable lane. Host-dependent. |
| `21-slow-row-labels` "on an Apple Silicon Mac" | Fakes `platform: darwin` in the workspace response, but the real Windows backend's video catalog has no `prince-canuma/LTX-2.3-distilled` row to label. Host-dependent. |
| `11-voice-pack` (dictation Delete) | Flaky under the full run (only a `route.fetch: Test ended` error is reported); **passes when run alone** (13.6 s). |

**Owner-named checks covered here (all PASS on Windows, mocked data):** `21-library-scroll` (Music and Audio library and Voice library: the 24th card scrolls into view, desktop and 375 px); `22-sound-steps` (Steps changes the estimate and is sent with the request; fits at desktop and 375 px); `21-kokoro-optional` (Kokoro tick box on by default; unticking removes it; Voices Fallback card picks any allowed voice or none; without Kokoro no fallback is in effect and the Quick group says how to add it; Chat Voice popover has a Fallback line); `21-slow-row-labels` "nothing new shows on an NVIDIA CUDA machine" and "no NVIDIA GPU: every diffusion row expected to be slow"; `13-sounds-scroll`.

## 2.1 Wave-2 speech models on the GPU (backend started with `CHAOSENGINE_VOICE_UNMEASURED=1`)

Driver 591.86; voice venv torch 2.9.1+cu128 (speech-GPU add-on); NumPy 2.5.4; chatterbox-tts 0.1.7, voxcpm 2.0.3.

| Item | Result | Evidence |
|---|---|---|
| "More speech models on the NVIDIA GPU" add-on appears and installs | PASS | Offered on this NVIDIA PC; `POST /api/voice/install {"extras":["speech-gpu"]}` ran to done, log ends `speech gpu ok` then `voice pack ok`. Voice venv **7.44 GB** after (torch/nvidia/transformers/chatterbox/voxcpm ≈ 6.25 GB of it); the base pack alone was ~2.3 GB per the README. The four CUDA rows appear only with `CHAOSENGINE_VOICE_UNMEASURED=1`. |
| Imports / missing packages | PASS | `import torch, chatterbox.tts_turbo, chatterbox.mtl_tts, voxcpm` and `torch.cuda.is_available()` all OK; no package missing. |
| Only the listed files arrive | PASS | Chatterbox-Turbo: the 11 listed files, 3.8 GB; Chatterbox Multilingual V3: the 6 listed files, 3.0 GB (the 13.9 GB repo was not pulled). VoxCPM2 and Orpheus: _pending download_. |
| Turbo: which decoder does `from_local` read? | PASS → **fixed** | `chatterbox/tts_turbo.py` builds `S3Gen(meanflow=True)` and loads `s3gen_meanflow.safetensors` only; `s3gen.safetensors` (1.0 GB) is never read. Dropped from `voice_models_cuda.py`; size 4.04 → 2.99 GB. |
| Turbo voice with `[laugh]` | PASS | Cold 49.6 s (load) / **warm 3.21 s for 9.27 s of speech** (≈3.5 s per 10 s; the Mac measured 2.1 s), ~3.5 GB VRAM. Files: `%TEMP%\voice-checks\turbo-warm.wav` (listen for the laugh). |
| Chatterbox Multilingual V3, English and German, Expression 0.3 / 0.8 | **FAIL (needs your OK)** | `TypeError: ChatterboxMultilingualTTS.from_local() got an unexpected keyword argument 't3_model'`. The worker asks for V3 (`t3_model="v3"`) and the catalog downloads only `t3_mtl23ls_v3.safetensors`, but chatterbox-tts **0.1.7 (newest on PyPI) only knows v2**; V3 exists only on upstream GitHub `master` (commit `65b1843719`, 2026-06-10; head `ab55923c`). Fix: pin `chatterbox-tts` to that commit as a source archive in `voice_env.SPEECH_GPU_NO_DEPS`. My install of the archive was blocked by the permission check (external code) — needs your OK. |
| Japanese on Chatterbox (no pykakasi) | _blocked with V3_ | |
| Clone with each cloning row: tick, AI label | PASS (Turbo) | Your sample `TEST v2.m4a` (8.17 s). Without the tick → 400 "Confirm that this is your own voice…"; with basis own + tick → saved, reference text auto-transcribed on CUDA ("I'm Dan Eve and this is my voice…"). Cloned Turbo speech: **24.5 s cold / 4.7 s warm** for 6.4 s, ~3.9 GB VRAM, note "Labelled as AI-made and watermarked." VoxCPM2 clone: _pending download_. |
| AudioSeal after speed 1.3 and pitch +2 | PASS | `GET /api/voice/clone/watermark/{id}` → `{"probability": 1.0, "message": "CE5A", "matches": true}`. |
| PerTh on a Chatterbox clip after speed 1.3 and pitch +2; AI tag says `"builtinWatermark": "perth"` | **FAIL → fixed** | In the voice venv, `perth.PerthImplicitWatermarker().get_watermark(...)`: plain cloned clip **1.0**, after speed 1.3 + pitch +2 **0.0**, Kokoro control 0.0. The WAV's AI tag (`LIST/INFO ICMT`) still claimed `"builtinWatermark":"perth"` on the shifted file. Fixed: `voice_clone.mark_cloned(..., altered=)` records PerTh only when speed is 1.0 and pitch 0 (Speak and Chat read-aloud); AudioSeal and `machine_generated` stay on every clip. Test added. |

| Orpheus `tara` with `<laugh>` | PASS (slow on this setup) | Only the listed `orpheus-3b-0.1-ft-Q8_0.gguf` + the SNAC ONNX companion arrived. 8.34 s of speech in **37.6 s** (slower than real time) with **no VRAM use**: this from-source backend resolves the stale CPU-only `~/.chaosengine/bin/llama-server.exe` (build 9189); a packaged app would use the bundled Vulkan b11981. Re-measured on the CUDA build below. File: `%TEMP%\voice-checks\orpheus-tara.wav`. |
| Orpheus: llama-server starts on 127.0.0.1 and stops with the voice worker | **FAIL → fixed** | It started (`llama-server.exe --model …orpheus-3b-0.1-ft-Q8_0.gguf` on a local port), but after `POST /api/voice/worker/stop` the worker was gone and **llama-server kept running**. `voice_client._kill` closed stdin and called `terminate()` at once — TerminateProcess on Windows — so the worker's `atexit` `stop_server` never ran. Fixed: the worker gets 5 s to exit on its own after stdin closes, and a forced stop ends its child processes first (psutil). Tests: `VoiceClientTests.test_stop_lets_the_worker_clean_up_its_children`, `…ignores_stdin_still_loses_its_children` (real child processes). |

| VoxCPM2 (GPU): voice design | PASS | Only listed files arrived (4.96 GB). Design "A warm, deep male narrator…": cold 24.6 s (load) for 5.9 s; warm **5.8 s for 6.1 s of speech** (≈9.5 s per 10 s; the Mac measured 5.2 s). Peak VRAM +6.7 GB. Files: `%TEMP%\voice-checks\vox-design-*.wav`. |
| VoxCPM2 (GPU): clone with your sample | PASS | Exact mode 13.0 s for 6.6 s; steered ("excited and happy") 7.9 s for 6.2 s with the note "VoxCPM2 followed the style; the steered mode trades a little likeness for it." Both "Labelled as AI-made and watermarked." The m4a sample decoded through the new PyAV path. |

**Two more Windows/CUDA cloning bugs fixed on the way** (cloning had never run on the CUDA lane):
- m4a/AAC samples failed with audioread's `NoBackendError`: chatterbox-tts and voxcpm read the sample with librosa/soundfile, which cannot decode AAC on Windows (the Mac has CoreAudio). `voice_worker_torch.readable_audio` decodes non-WAV/FLAC samples once with PyAV to a 24 kHz WAV, as the MLX path already does.
- every cloned Turbo voice then failed with `expected scalar type Float but found Double`: chatterbox-tts 0.1.7's `norm_loudness` multiplies by pyloudnorm's `numpy.float64`, and under NumPy 2's promotion rules the sample becomes float64. `keep_float32_loudness` casts it back after load.

## 5. Projects workspace

| Item | Result | Evidence |
|---|---|---|
| Flower-shop brief in Chat with studio tools | **PASS on the second try; first try crashed the backend** | Nemotron 3.5 Bolt (llama.cpp CUDA), Studio on, "Make two posters for a flower shop": a Plan to approve card per poster (FLUX.2 Klein 4B, 1024², "about 6 s, about 16 GB", "Nemotron stays loaded: about 41 GB free"), Approve → **18.8 s** and **9.1 s** (first one includes the load), reply lists both files. The first try died during the second render: Windows logged "Your system is low on virtual memory" at 21:19:42 and the backend process vanished (no traceback, no crash event); job-3 stayed `running` on disk, the chat turn disappeared from the thread and was never saved, and the two llama-servers were left orphaned. See the memory note below. |
| ...renders appear in the project | **FAIL** | No project gets the renders: they appear in the chat and in Images outputs only. Nemotron used `generate_image`, not `storyboard_project`. With the thread still on Qwen3-0.6B (an earlier mistake of mine), the model skipped the plan card, called `storyboard_project create` twice (two empty projects, "Flower Shop Poster 1/2", still listed), and then claimed "both posters have completed their renderings" while nothing rendered. A tiny model inventing results is expected; an empty project from a tool call that only takes a name may deserve a guard. |
| Cast: character with four pictures; run a shot with it | **PASS / FAIL** | `POST /api/storyboard/uploads` ×4 (PNG, up to 6.4 MB) → `add_reference` character "Rosa" with **4 pictures**, shown on the Cast tab. Shot with the character, still batch on FLUX.2 Klein 4B: **refused in 3 s**: "FLUX.2 Klein 4B cannot take a reference image. Use FLUX.1, FLUX.2 Klein, Qwen Image 2.1 or FLUX 3 Klein on the mflux (MLX) engine." (`helpers/media_input.py:70`). Reference pictures are MLX-only, so Cast does nothing on Windows/Linux, and the message names the model it is refusing. diffusers 0.42.1 here has `Flux2KleinPipeline` with image input (see the CUDA landscape note), so this is a parity gap, not a hard limit. |
| Mixer: voice, effect, music lanes; export MP4 | PASS (with library WAVs) | Sound effects and music can't be generated on Windows (section 3), so three Voice library WAVs were added as voice, sfx (−6 dB) and music (−12 dB, duck + loop) lanes on the export-check project; settings kept, each tagged `clonedVoice` from its AI tag. Export: **1 s**, `1920x1080 h264 24 fps + AAC 48 kHz stereo, 4.10 s`, mean −18.3 dB / peak −1.5 dB. The render status came back with no warnings; the Export tab's preflight panel itself was not opened. |
| Paste an image with Ctrl+V into a reference box | **SKIP (needs a hand check)** | A picture was put on the Windows clipboard and Ctrl+V sent to the Cast tab from the test browser: nothing arrived (no upload request). The test browser's synthetic key does not carry the OS clipboard, so this proves nothing either way. |
| Open the project after a restart | PASS | After the backend restart and a page reload, the export-check project made before the restart opened with its 2 shots, settings and exports. (Dev build, not the packaged app: that is section 7.) |

**Memory on Windows (the crash above).** Windows does not overcommit: every allocation counts against the commit limit, which here is 64 GB RAM + a 4 GB page file = 67.8 GB. With two llama-servers warm, the browser pane and the GPU driver, about 31 GB is committed before an image model loads. Measured on the second try: loading FLUX.2 Klein 4B in the backend took commit from **31 → 48 GB**, the second render peaked at **75.0 GB**, which only fitted because Windows had grown the page file after the first crash. The plan card's "about 41 GB free" is free RAM, not free commit. On the Mac and Linux this does not fail. Not fixed here (it is the design of keeping image pipelines in the backend process, CLAUDE.md "process isolation"); suggested: run image jobs in a worker like video, and on Windows have the fit check read `\Memory\Commit Limit − Committed Bytes`. Owner: a bigger page file avoids it today.

## 6. Text models on CUDA

Model: `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB` (Q4_K_M, 6.2 GB) downloaded through the app; its pinned DFlash drafter (`…-DFlash-BF16.gguf`, 0.74 GB) arrived with it. llama.cpp b11981 Windows builds fetched from the pinned GitHub release (with your OK) into `~/.chaosengine/test-llama-b11981/{cuda,vulkan}`: `--list-devices` shows **CUDA0: NVIDIA GeForce RTX 4090** and **Vulkan0: NVIDIA GeForce RTX 4090**. Each server was launched with the app's own flags (`--parallel 1 --ctx-size 8192 --jinja --flash-attn on`, f16 cache, `--reasoning off`, `--fit on`; drafter: `--model-draft … --spec-type draft-dflash --spec-draft-n-max 15`); one warm-up, then 3 greedy runs of 400 tokens of Python, llama.cpp's own timings.

| Build | Decode tok/s | + DFlash drafter | Prefill (~5K-token prompt) | Load | VRAM |
|---|---|---|---|---|---|
| CUDA 13.4, b11981 | **169.7** (169.3–170.0) | **331.9** (2.0×, 43 % of drafts accepted) | **13,463 tok/s** | 2.5 s | +3.8 GB (+5.6 with drafter) |
| Vulkan, b11981 (what the Windows bundle ships) | **169.6** | 92.8 → 271.8 → 311.8 (still warming up), 53 % accepted | 10,999 tok/s | 10.6 s (3.0 s second start) | +3.5 GB |
| CPU build 9189 in `~/.chaosengine/bin` (what this from-source backend resolves today) | **21.6** | — | 177 tok/s (short prompt) | 3.5 s | none |

Verdict: on this 4B-class model Vulkan decodes as fast as CUDA; CUDA wins on prefill (+22 %), model load (4× faster cold) and steadier speculative decoding. The real problem on this PC is the stale CPU-only `llama-server.exe` (build 9189) in `~/.chaosengine/bin`, which a from-source backend picks up first: chat and Orpheus run **8× slower** on it, with no GPU use. Packaged builds use the bundled Vulkan binary and are not affected. Same outputs on every build (greedy).

| Item | Result | Evidence |
|---|---|---|
| llama.cpp with CUDA: load a GGUF, chat, tokens/s | PASS (outside the app) | Table above. In-app CUDA lane does not exist on Windows (`llama-cuda-status` → `supported:false`; installer is Linux-only). |
| vLLM lane | SKIP | No native Windows vLLM (WSL2 only, per D-66 that is a later pass). |
| Through the app on the CUDA build | PASS | Backend started with `CHAOSENGINE_LLAMA_SERVER=…/test-llama-b11981/cuda/llama-server.exe`; `POST /api/models/load` Nemotron with speculative decoding: load 3.6 s, note "Speculative decoding active (draft-dflash with drafter Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf)"; two 400-token replies at **329.2 and 333.7 tok/s**. Switching models keeps the previous server in the warm pool (`warmModels`), by design. |
| `cache-strategy-matrix.py --quick` and full | PASS (1 cell runs) / **FAIL → fixed** | `lmstudio-community/Qwen3-0.6B-GGUF` added for the smoke cell (not a catalog row, so all four quants came down, 2.35 GB). Quick: native GGUF smoke **pass, 513 tok/s**. The full run first **crashed on Windows**: `UnicodeEncodeError: 'charmap' codec can't encode character '→'` printing the vLLM skip reason to a cp1252 console. Fixed: the script switches stdout/stderr to UTF-8 (`_utf8_console`, test in `test_cache_strategy_matrix_runner.py`); full run then completes: 1 pass (GGUF smoke 246 tok/s), 15 skips (MLX/DFlash/DDTree/MTPLX/TensorFold are Mac-only, vLLM is WSL-only, `ggml-org/Qwen3.6-27B-MTP-GGUF` not downloaded, TurboQuant see below). Report: `~/.chaosengine/test-results/cache-strategy-matrix-20261109-210404Z.md`. |
| TurboQuant GGUF lane on Windows | **FAIL (Mac session)** | Both TurboQuant cells skip with "strategy 'turboquant' unavailable in this runtime", although `~/.chaosengine/bin/llama-server-turbo.exe` is installed: `cache_compression/turboquant.py` `is_available()` only checks for the Mac-only `turboquant-mlx-full` package, so on Windows and Linux the strategy is unavailable everywhere, including the llama.cpp lane that only needs the turbo binary. Changing what "available" means touches both engines, so not patched here. |

**Orpheus on the CUDA build** (same backend): `tara` with `<laugh>` cold 7.26 s / **warm 4.57 s for 8.3 s of speech** (≈5.5 s per 10 s; the Mac measured 7.2 s), +4.3 GB VRAM — 8× faster than on the stale CPU build. **Live check of the shutdown fix:** after `POST /api/voice/worker/stop` the Orpheus llama-server is gone; only the warm-pool chat server remains.


**New Windows bug fixed — every Hugging Face model showed twice its size on Windows.** `helpers/discovery._path_size_bytes` deduplicates files by `(st_dev, st_ino)`, but on Windows `os.scandir`'s cached stat gives 0 for those on plain files while a followed symlink gets the real ids, so a blob and the snapshot link to it were both counted. Measured: Whisper Large v3 Turbo 3.24 GB counted for 1.62 GB of blobs; the Orpheus delete confirm said 7.14 GB for 3.6 GB. This size feeds the Disk usage page, delete confirmations and "freed GB". Fixed by fetching the real ids with `os.stat()` when scandir has none; after the fix Whisper 1.62, VoxCPM2 4.96, Turbo 4.04 GB (exact). Test in `tests/test_discovery.py` simulates the Windows scandir on any OS.

**Disk clean-up (approved by you):** deleted CogVideoX-2b (12.8 GB) and the three Mac-only Qwen3-TTS partial downloads (1.4 GB) through `POST /api/models/download/delete`, and the folders `~/.chaosengine/longlive` (25.4 GB) and `~/.chaosengine/prebuilt-llama` (2.7 GB). Free space 108.3 → **150.8 GB**.

## 7. Packaged installer

No v0.10.0 installer is published (newest Windows asset: v0.9.9), so it was built here the way `release.yml` does: a fresh Python 3.12 venv with the CI package list under `release-constraints.txt`, `npx tauri build --bundles nsis` (release staging, no updater signing). Installed on the current profile (your choice; a clean Windows account is yours to create), over an existing v0.9.9 install.

| Item | Result | Evidence |
|---|---|---|
| Build the Windows installer | PASS after a fix | First build failed in staging: `tar (child): Cannot connect to C: resolve failed`. Git Bash's GNU tar is first on PATH there and reads `C:\…\runtime.tar.gz` given to `-f` as host `C`; CI runs the Windows bsdtar so never hit it. Fixed: the archive name is passed relative to `cwd` (`ff00a1b9`, test in `test_stage_runtime_scripts.py`). Then `ChaosEngineAI_0.10.0_x64-setup.exe`, **136.5 MB**; manifest `llamaServer: bin/llama-server.exe` (Vulkan b11981), `warnings: []`; the archive holds `install-mflux.sh`, `install-ltx2-mlx.sh`, `install-mtplx.sh`, `install-tensorfold.sh`. |
| Install, first launch, runtime extraction, sidecar | PASS | Silent install (`/S`) 10 s, registry shows 0.10.0. Launch → `/api/health` ok after **25 s** (runtime extracted to `%TEMP%\chaosengine-embedded-runtime\windows-x86_64-8ce9411e`, 397 MB), sidecar uses the bundled Vulkan llama-server. Two runtimes from earlier installs (499 MB + 346 MB, Nov 4) are still in that folder: old extractions are never cleaned up. |
| Speech from the installed app | PASS | Kokoro `af_heart`: HTTP 200, valid WAV, 6.9 s including worker start. |
| One project export from the installed app | PASS | RTX 4090 export check: 3 s, `1920x1080 h264 24 fps + AAC stereo`, 4.10 s. |
| One image from the installed app | **PASS after six fixes** | Final run on driver **621.14** (CUDA 13.5): Install GPU runtime → all 17 packages, `torch 2.14.1+cu130`, CUDA verified, 466 s; FLUX.2 Klein 4B 1024², 4 steps: **HTTP 200, `Diffusers local engine (cuda)`, 14.8 s** (51.7 s with the first load); a clean watercolor lighthouse. Getting there took six Windows bugs, each fixed with a test (detail below). |
| Uninstall leaves no running processes | PASS | `uninstall.exe /S` with the app and its backend running: exit 0 in 2 s; 15 s later no ChaosEngineAI, backend python or llama-server process, port 8876 free, uninstall entry gone. Left behind on purpose: `%LOCALAPPDATA%\ChaosEngineAI\extras` (the GPU runtime, kept across updates) and the runtimes unpacked into `%TEMP%\chaosengine-embedded-runtime` (one per build; never cleaned, see above). |

The GPU runtime's packages after torch are not held to `release-constraints.txt` (newest at install time), (the CUDA index list used to stop at 13.0; see the cu134 note below).

**Installed app: the GPU runtime on Windows, six bugs fixed** (all in `test/rtx4090`, each with a test; the dev build never hit them because it runs from the dev venv):
1. `ff00a1b9` — local release build: Git Bash's GNU tar read `C:\…untime.tar.gz` as a remote host. The archive name is now relative to `cwd`.
2. `4df43569` — CPU torch from PyPI replaced the CUDA torch, and the repair pass missed it because it read the stale `+cu130` dist-info. It now reads `torch/version.py`.
3. `47ffa182` — after an install the backend **did not start** ("cannot import name '_common' from partially initialized module 'psutil'"): pip could not replace the loaded `_psutil_windows.pyd` and left `psutil/` with 3 files. The start-up overlay guard now moves aside a bundled package's overlay copy whose RECORD files are missing.
4. `6fd76791` — the guard ran after `from fastapi import …`, so a broken overlay `pydantic_core` crashed first (the "Backend failed to start" screen you saw); a repeat move of the same version raised "already exists" and stopped the whole tidy-up; and `pydantic` 2.14 stayed behind without its core. Guard now runs before any third-party import (test checks `app.py`), each move gets its own folder and fails alone, and `pydantic` + `pydantic-core` are lock-step.
5. `d869a4ee` — every later GPU-runtime install failed on six packages with `WinError 5` because the backend had loaded the overlay's `yaml\_yaml.pyd` and `psutil\_psutil_windows.pyd` (the overlay is first on `sys.path` by design). Same-version overlay copies of bundled packages with a native extension are now moved aside at start, so the bundle's copy runs and nothing in the overlay is locked.
6. `eabb6bf7` — diffusers still failed: "tokenizers>=0.23.1 is required … but found tokenizers==0.22.2". pip `--target` leaves the older version's `.dist-info` next to the new one and `importlib.metadata` read the old record. The guard now keeps the record pip wrote last for packages the bundle does not ship (records only, files untouched). First start moved 20+ such records left by a May install.

**Newest CUDA torch for the driver (`06953d2b`).** The GPU runtime stopped at the CUDA 13.0 index. PyTorch also publishes **cu132 and cu134** (torch 2.15.0, Windows, Linux x86-64 and ARM64, Python 3.12 and 3.13); nothing newer (no cu135: that index returns 403), although driver 621.14 reports CUDA 13.5. `cuda_torch_indexes` now reads the CUDA version from the `nvidia-smi` header (`CUDA UMD Version: 13.5` on new drivers, `CUDA Version: 13.1` on older ones) and starts at cu134 for 13.4+, cu132 for 13.2+, else as before (cu130 for driver 580+, cu128 for 570+, the old list below that). Two tests (order, header parser). Timing, FLUX.2 Klein 4B bf16 on the RTX 4090, same render, torch 2.14.1+cu130 vs 2.15.0+cu134 (release venv, two runs each): **1024², 4 steps: 1.47 s vs 1.36 s (about 7 % faster, identical across runs)**; load 9.5-11.5 s vs 11.5-11.9 s; 2048² (peak 26.3 GB, spills past the 24 GB card): 37-72 s vs 28-37 s, noisy because of the spill. No import or kernel problems on 13.4.

After uninstall the app leaves 14 GB in `extras` (the GPU runtime, kept on purpose) and 4.0 GB in ten unpacked runtimes under `%TEMP%\chaosengine-embedded-runtime`.

Also seen, not fixed: when a GPU-runtime install runs after torch was loaded (any image attempt), `_purge_stale_torch_from_extras` deletes torch's Python files before it hits the locked DLLs, leaving torch unusable until "Restart backend, then Install GPU runtime again" (the app's message says exactly that, and it worked). The root cause of most of the above is that the GPU runtime installs packages the app already ships (pydantic, psutil, pyyaml, huggingface-hub) into the overlay; pinning those to the bundled versions during the install would stop the churn.


## Report back (sections 0 to 7)

### What failed (details and logs in the sections above)

Fixed on this branch, each with a test (all Windows or CUDA-lane only):
1. Warm worker never restarted after a crash (`Popen.poll()` stays None after `os._exit`); ffmpeg-fetch test on Windows zips.
2. Speak offered and downloaded Mac-only Qwen3-TTS weights on Windows (`voice_clone.start_download` guard).
3. Dictation crashed on every NVIDIA PC: ctranslate2 finds cuBLAS on PATH only.
4. Cloning on CUDA: m4a samples could not be read; cloned Turbo voices crashed (float64 loudness under NumPy 2); the PerTh tag claimed a watermark that speed/pitch had erased.
5. Orpheus's llama-server outlived "stop voice worker" (TerminateProcess skips `atexit`).
6. GPU metrics blocked the whole backend on NVIDIA PCs (torch probe on the event loop).
7. Every Hugging Face model counted twice its size (scandir has no inode on Windows).
8. Cache-strategy matrix crashed on a cp1252 console; the Playwright suite could not start on Windows.
9. Chatterbox-Turbo row listed a 1.0 GB file it never reads.

Not fixed here (for you or the Mac session):
- **Backend crash from the Windows commit limit** during two chat renders (section 5): FLUX.2 Klein in the backend process took commit to 75 GB of a 67.8 GB limit. Today's cheap fix: a larger page file. Proper fix: image jobs in a worker process, and a fit check that reads free commit on Windows.
- **No Storyboard clips and no Cast stills on Windows/Linux**: clip making refuses every non-LTX-2 model; reference pictures are MLX-only, and the refusal names FLUX.2 Klein as both refused and suggested.
- **No music, sound effects or Qwen3-TTS on Windows** (MLX-only lanes); the Music page contradicts itself (CUDA chips, "Tight" rating, Delete on rows not installed, licence prompt the token already passes).
- **Chatterbox Multilingual V3** needs chatterbox-tts from upstream master (PyPI 0.1.7 has only v2); waiting on your OK to pin the commit.
- **TurboQuant unavailable off the Mac** (`is_available` checks only the MLX package).
- **No in-app CUDA llama.cpp on Windows**, and a stale CPU `llama-server.exe` in `~/.chaosengine/bin` is picked first by from-source backends (8× slower).
- Cross-platform, seen here: STT "installed" while `model.bin` is still downloading; voice runtime panel not refreshed after a load; "stop voice worker" waits behind a stuck request; studio chat with a small model makes empty projects and claims renders; a crashed backend leaves job files `running`, loses the chat turn and orphans its llama-servers.
- Not run: remove/reinstall voice runtime (about an hour per pass at 2 MB/s); Ctrl+V paste (needs a hand check); voice call (needs a microphone); LTX timing (not on Windows).

### Slower than expected
- Wan 2.1 1.3B at 720p: **578 s** and the full 24 GB, against "About 1 min, 16.7 GB" (FU-145); the new-project form says 74 GB instead. 480p: 95 s.
- After any video run the card stays full (24 GB) until Unload.
- Orpheus and chat on the stale CPU llama-server: 37.6 s for 8.3 s of speech, 21.6 tok/s (CUDA build: 4.57 s, 170 tok/s).
- Hugging Face downloads at ~2 MB/s over the VPN on this PC.

### What the Mac values got wrong for this PC
- Chatterbox-Turbo: Mac 2.1 s per 10 s of speech; here 3.5 s. VoxCPM2 design: Mac 5.2 s, here 9.5 s per 10 s.
- Orpheus: Mac 7.2 s per 10 s; here 5.5 s on the CUDA build (faster), 45 s on the stale CPU build.
- Video estimates (above) are 4–10× off on CUDA, in both directions between the two estimators.
- FLUX.2 Klein plan card: "about 6 s" per poster; measured 18.8 s first (load) and 9.1 s warm; "41 GB free" ignores the Windows commit limit.

### One-line verdict per section
- **0 Setup:** PASS. Torch needed the cu128 wheel (cu124 stops at 2.6).
- **1 Automated checks:** PASS after fixes. Both release flags pass. Remaining failures are host-dependent tests, handed to the Mac session.
- **2 Speech:** PASS for Kokoro, Whisper on CUDA, Chatterbox-Turbo, VoxCPM2 and Orpheus, including cloning, fallback, delete and cancel. Six Windows bugs fixed. Qwen3-TTS is Mac-only; Multilingual V3 waits on your OK.
- **3 Music and sound:** SKIP. Mac-only in v0.10.0; the UI checks pass in Playwright with mocks.
- **4 Image and video:** Images PASS (2048² in 75 s). Video FAIL on estimates (FU-145). Storyboard export PASS, but clips cannot be made on Windows.
- **5 Projects:** Mixed. Plan, approve and render work. The backend crashed once on the Windows commit limit. Renders don't land in a project, Cast stills are refused, and the Mixer export passes.
- **6 Text on CUDA:** PASS. 170 tok/s, 332 with DFlash, 330 through the app on the CUDA build. Vulkan matches CUDA on decode. The vLLM and TurboQuant lanes are not available on Windows.
- **7 Packaged installer:** PASS after fixes. It builds (136.5 MB), installs in 10 s, the backend is up in 16–25 s, and speech, export, FLUX on CUDA and uninstall all work. The GPU-runtime install on Windows needed six fixes before an installed app could make an image; built locally, since no v0.10.0 installer is published.
