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
| Chatterbox Multilingual V3, English and German, Expression 0.3 / 0.8 | **PASS after the pin (your OK)** | `chatterbox-tts` is now pinned to upstream commit `ab55923c` (source archive, `--no-deps --force-reinstall`; installed through the app's voice-runtime route). English and German at Expression 0.3 and 0.8: **2.4-2.7 s warm for 3.6-4.4 s of speech** (19-30 s cold); Whisper transcribes both languages word for word (one German run heard the compound noun as two words). Files: `%TEMP%\voice-checks\mtl-{en,de}-{0.3,0.8}.wav`. |
| Japanese on Chatterbox (no pykakasi) | **FAIL → fixed** | Without `pykakasi` (GPL-3.0) Chatterbox fed raw kanji to its tokenizer: the clip transcribed as unrelated syllables (三重は木の殻からする頃…). New `backend_service/voice_ja.py` answers `pykakasi.kakasi().convert()` with `fugashi` (MIT) + `unidic-lite` (BSD), registered only when the real package is absent; the packages are in the add-on list. Same sentence now: 東大使は朝日が水平線から登る頃長い階段を登っていった (expected 灯台守は…昇るころ…上っていった): one rare-word homophone, the rest right. 8 tests. |
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
| TurboQuant GGUF lane on Windows | **FAIL → fixed** | Both TurboQuant cells skipped with "unavailable in this runtime" although `llama-server-turbo.exe` was installed: `is_available()` checked only the Mac-only MLX package. It now also counts the turbo binary (and says to install it off the Mac). Re-run on the installed 0.10.0 app: strategy list shows **Ready**; full matrix: `turboquant GGUF 3-bit` **pass, 89.5 tok/s** (Qwen3-0.6B; native 249 tok/s on the same model). 3 tests. |

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
1. `ff00a1b9` — local release build: Git Bash's GNU tar read `C:\…
untime.tar.gz` as a remote host. The archive name is now relative to `cwd`.
2. `4df43569` — CPU torch from PyPI replaced the CUDA torch, and the repair pass missed it because it read the stale `+cu130` dist-info. It now reads `torch/version.py`.
3. `47ffa182` — after an install the backend **did not start** ("cannot import name '_common' from partially initialized module 'psutil'"): pip could not replace the loaded `_psutil_windows.pyd` and left `psutil/` with 3 files. The start-up overlay guard now moves aside a bundled package's overlay copy whose RECORD files are missing.
4. `6fd76791` — the guard ran after `from fastapi import …`, so a broken overlay `pydantic_core` crashed first (the "Backend failed to start" screen you saw); a repeat move of the same version raised "already exists" and stopped the whole tidy-up; and `pydantic` 2.14 stayed behind without its core. Guard now runs before any third-party import (test checks `app.py`), each move gets its own folder and fails alone, and `pydantic` + `pydantic-core` are lock-step.
5. `d869a4ee` — every later GPU-runtime install failed on six packages with `WinError 5` because the backend had loaded the overlay's `yaml\_yaml.pyd` and `psutil\_psutil_windows.pyd` (the overlay is first on `sys.path` by design). Same-version overlay copies of bundled packages with a native extension are now moved aside at start, so the bundle's copy runs and nothing in the overlay is locked.
6. `eabb6bf7` — diffusers still failed: "tokenizers>=0.23.1 is required … but found tokenizers==0.22.2". pip `--target` leaves the older version's `.dist-info` next to the new one and `importlib.metadata` read the old record. The guard now keeps the record pip wrote last for packages the bundle does not ship (records only, files untouched). First start moved 20+ such records left by a May install.

**Newest CUDA torch for the driver (`06953d2b`).** The GPU runtime stopped at the CUDA 13.0 index. PyTorch also publishes **cu132 and cu134** (torch 2.15.0, Windows, Linux x86-64 and ARM64, Python 3.12 and 3.13); nothing newer (no cu135: that index returns 403), although driver 621.14 reports CUDA 13.5. `cuda_torch_indexes` now reads the CUDA version from the `nvidia-smi` header (`CUDA UMD Version: 13.5` on new drivers, `CUDA Version: 13.1` on older ones) and starts at cu134 for 13.4+, cu132 for 13.2+, else as before (cu130 for driver 580+, cu128 for 570+, the old list below that). Two tests (order, header parser). Timing, FLUX.2 Klein 4B bf16 on the RTX 4090, same render, torch 2.14.1+cu130 vs 2.15.0+cu134 (release venv, two runs each): **1024², 4 steps: 1.47 s vs 1.36 s (about 7 % faster, identical across runs)**; load 9.5-11.5 s vs 11.5-11.9 s; 2048² (peak 26.3 GB, spills past the 24 GB card): 37-72 s vs 28-37 s, noisy because of the spill. No import or kernel problems on 13.4.

After uninstall the app leaves 14 GB in `extras` (the GPU runtime, kept on purpose) and 4.0 GB in ten unpacked runtimes under `%TEMP%\chaosengine-embedded-runtime`.

Also seen, not fixed: when a GPU-runtime install runs after torch was loaded (any image attempt), `_purge_stale_torch_from_extras` deletes torch's Python files before it hits the locked DLLs, leaving torch unusable until "Restart backend, then Install GPU runtime again" (the app's message says exactly that, and it worked). The root cause of most of the above is that the GPU runtime installs packages the app already ships (pydantic, psutil, pyyaml, huggingface-hub) into the overlay; pinning those to the bundled versions during the install would stop the churn.


## Follow-up work after your go-ahead (2026-11-10)

You said OK to the chatterbox pin and to carry on with the remaining Windows/CUDA fixes. Everything is on `test/rtx4090`, one commit per change, each with a test that fails without it.

| Change | Commit | Evidence |
|---|---|---|
| Chatterbox Multilingual V3 on CUDA | `18ad7db1` | Pinned to `resemble-ai/chatterbox@ab55923c` (MIT). pip kept the installed PyPI 0.1.7 because the archive reports the same version, so the add-on step now uses `--force-reinstall`. English/German/Japanese results in section 2.1. |
| Japanese readings without GPL | `18ad7db1` | `voice_ja.py` + `fugashi==1.5.2`, `unidic-lite==1.0.8` (licences in `THIRD_PARTY_NOTICES.md`). The worker imports it as a sibling module (it runs as a script). |
| Newest CUDA torch for the driver | `06953d2b` | cu134 / cu132 / cu130 by the `nvidia-smi` CUDA version; FLUX 1024²: 1.36 s (cu134) vs 1.47 s (cu130). |
| TurboQuant available off the Mac | `5adc858c` | Matrix cell passes at 89.5 tok/s on the installed app. |
| Old unpacked runtimes removed | `19f039b7` | The shell prunes `\chaosengine-embedded-runtime\<platform>-*` folders of earlier builds after it extracts or reuses the current one, in a background thread, skipping any folder whose interpreter or llama-server cannot be opened for writing (a running app). Installed build: **10 folders / 3.74 GB → 1** at launch. 3 Rust tests. |
| Startup probe no longer waits 2 s per poll | `f02c5b27` | Windows takes about 2 s to refuse a connection to a closed local port, so each startup poll before the sidecar was up cost 2 s; `connect_timeout(250 ms)`. The Rust test `wait_for_port_gives_up_at_once_when_the_caller_says_stop` failed at 2.06 s on this PC before and passes now (whole Rust suite 0.28 s, 8 of 8). |
| VRAM handed back after image and video runs | `30631da2` | `empty_cache()` after every run (also failed ones), pipeline stays loaded. Installed app, FLUX 2048² (70 s): GPU memory 2.8 GB idle → 3.3 GB after (it was 24 GB after one Wan 720p clip before). Tests for both engines. |
| GPU-runtime install refuses before deleting when torch is loaded | `5df46fc8` | Checks the DLLs for a write handle first; says "Restart Backend, then install again" and changes nothing, instead of deleting torch's Python files and leaving the locked DLLs behind. |
| Test portability (3 assertions) | `18ad7db1` | Orpheus and SNAC path assertions and the watermark path assertion built with `Path`, so they pass on Windows. |

**Decided against, with reasons**
- *Pinning the packages the app already ships during the GPU-runtime install:* a pin the new torch stack cannot satisfy turns a working install into `ResolutionImpossible`; the start-up overlay guard (section 7) now removes the conflicting overlay copies instead.
- *A Windows commit-limit gate for image jobs:* the data does not say which allocation tipped the limit (commit rose 31 → 48 GB on load, then the page file grew mid-run), so a gate would be a guess. Larger page file stays the advice.
- *Storyboard clips from text-only shots on CUDA:* the refusal is the deliberate D-40 rule (a clip must not silently ignore its still). The real fix is a "text only" shot option in the UI plus image-to-video on CUDA (Wan 2.2 I2V / LTX-2 on diffusers, see `docs/research/cuda-model-landscape-2026-11.md`).
- *Stale CPU `llama-server.exe` in `~/.chaosengine/bin`:* only a from-source backend picks it; packaged builds bundle Vulkan. Worth a "this llama-server has no GPU" warning, not done.

**Full pytest on this PC after all of the above:** 38 failures, all of which also fail on the commit before today's work (compared test by test in a clean worktree), none new. The bulk are Windows test portability (19 × `cannot instantiate 'PosixPath'`, the Kokoro and voice-model-file tests), the five CUDA-index tests that call the real `nvidia-smi`, and the Mac-only media-input and music-sound tests; all for the Mac session. Rust: 8 of 8.


## Image-to-video and Storyboard clips on the NVIDIA PC (2026-11-10, FU-159)

Section 4 found that a Windows user could not make any Storyboard clip (the video route and Storyboard allowed a start picture only for the Mac's LTX-2 engine). Fixed with **Wan 2.2 TI2V 5B** (Apache 2.0, 5B, text and start-picture video in one model, fits 24 GB) and a **text-only shot** option. Commits `7a1e6152`, `99560201`, `9902407d`, `e13d14b6`.

| Check | Result | Evidence |
|---|---|---|
| Text-to-video, 832x480, 49 frames, 20 steps | PASS | 42.5 s generation (83 s with the first load), peak VRAM 12.4 GB, idle 1.6 GB after unload. The old Wan 2.1 1.3B took 95 s for 33 frames at 480p. |
| Picture-to-video from a FLUX.2 Klein still | PASS | 49.9 s, peak 13.8 GB. Frame 0 is the still (red lighthouse at dusk); at 1.5 s the waves have broken and birds have appeared, scene unchanged. Frames in `%TEMP%\wan-ti2v-check\`. |
| 5-second clip at the catalog's recommended 1280x704, 121 frames, 20 steps | PASS | **273 s** generation (304 s total), peak **18.5 GB**. Wan 2.1 1.3B at 720p took 578 s and filled all 24 GB (section 4). The catalog's "about 540 s" guess is now 275 s. |
| Storyboard project, FLUX.2 Klein + Wan 2.2 TI2V 5B, 480p | PASS | Three shots: one that starts from its generated still, one text-only, one continuing from the previous shot's last frame. Still + 3 clips in **286 s**, then export 854x480 H.264, 5.63 s. |
| Text-only shots on the old Wan 2.1 1.3B (no image input at all) | PASS | Default shots: 400 "cannot start a clip from an image… Wan 2.2 TI2V 5B on an NVIDIA PC". After `startFrom: text`: two 2 s clips in 170 s. |
| Real diffusers call path, tiny random pipeline on CPU | PASS | `WanImageToVideoPipeline` built with `expand_timesteps=True` from `pruna-test/wan-t2v-tiny-random`; the app's `build_pipeline_kwargs` + `invoke_pipeline` returned 9 frames. |
| E2E suite | PASS | New checks in phase 5 (flags, refusals, live run when installed: 14 s for 17 frames) and phase 10 (text-only shot). |
| Frontend | PASS | `tsc` clean; vitest: 3 new picker tests pass; the 4 failures are the same ones as in section 1. `npm run i18n:validate` OK (new string in all 11 locales). |

**What it took (three things the first live run found that the unit tests could not):**
- The memory gate refused: "needs about 34 GB of free memory, only 29.8 GB available". The row's disk size (34.2 GB) is double what loads, because the repo stores the transformer in fp32 and the app loads bf16. A row can now declare `loadMemoryGb` (24).
- 22 GB of pipeline does not fit a 24 GB card with activations, and full placement falls back to sequential offload (every layer, every step). Placement now uses model-level CPU offload when the whole pipeline is too big but its largest component fits (`_prefers_model_cpu_offload`; the old path stays for a component bigger than the card).
- The runtime table said 11 GB for the 5B (transformer only); it is 22 GB with the text encoder.

**Design points.** `image_input_support(repo)` is now the single answer to "does this model take a start / end picture" (LTX-2 MLX: both; TI2V-5B: start only, diffusers' expanded-timestep path ignores a last frame), used by the catalog flags (`acceptsImageInput`, new `acceptsEndImage`), the video route, the agent bridge, Storyboard and the Studio's picker (end picker only when `acceptsEndImage`). The row is `cudaOnly`, so a Mac does not see it (MPS corrupts Wan 2.2, FU-102/125); `platform_filter` gained that filter. Tests: 13 new in `test_video_image_input.py`, 4 in `test_storyboard_store.py`, 4 in `test_platform_filter.py`, 1 in `test_memory_gate.py`, 3 frontend picker tests and 2 E2E checks.

## LTX-2.3 with sound on the NVIDIA PC (2026-11-10)

`OzzyGT/LTX-2.3-Distilled-bnb-nf4` (4-bit, 31 GB) through the dev backend, 768x512, 97 frames at 24 fps, FLUX.2 Klein stills for the start and end pictures. Every clip carries a 48 kHz stereo AAC track.

| Mode | Generation | Peak VRAM | Sound (mean / max) |
|---|---|---|---|
| Text only | 69.6 s | 22.3 GB | -20.3 / -3.7 dB |
| Start picture | 70.8 s | 22.3 GB | -23.2 / -2.8 dB |
| Start + end pictures | 74.9 s | 22.3 GB | -12.1 / -2.4 dB |

First frame of the start clip matches the still; the late frame of the start+end clip has moved to the end still's night scene. The first attempt sat on step 1 of 8 for over an hour: the 4-bit weights are loaded onto the card by bitsandbytes, so the free-memory budget read 1 GB and ruled out model offload. `_cuda_memory_gb` now counts what this process already holds (`356eae6d`); the same pack then ran at 70 s. Not run live: the two bf16 rows (100 GB downloads); they are covered by the tiny-pipeline tests and the placement matrix.

**Not done.** The newest open video models are still Mac-only or unwired on CUDA: LTX-2.3/2.5 (March / July 2026, audio, diffusers 0.42.1 has the pipelines), Kandinsky 6 (MIT), Motif-Video; Wan 2.2 I2V A14B for an end frame needs offload or GGUF. See `docs/research/cuda-model-landscape-2026-11.md`.

## TensorFold 1.x on the RTX 4090 (2026-11-11, FU-111)

Standalone evaluation of the native engine (release 1.2.6, `tensorfold-native serve`), outside the app. Nothing in the app changed.

**Verdict: it works on a 4090 (sm_89), but only as a source build. There is no Windows release binary.**

- **Release assets (v1.2.6):** `linux-x86_64`, `linux-aarch64` and `macos-arm64` archives (47 / 47 / 6 MB, each with a `.sha256` and a `.sha256.sig`). No Windows asset. `docs/WINDOWS.md` says "supported from source only".
- **The Linux archive carries no sm_89 image.** Its fatbins hold sm_80, 86, 120 and 121. The runbook: "`sm_89` builds, but the release set holds no image qualified on it". Not run here: WSL2 has no distro on this PC. A Linux 4090 user would build `-Dgpu=sm_89` too (or try the sm_86 image; untested).
- **Driver floor:** R580 or newer (CUDA 13.0 `libcuda`); the server refuses an older driver at startup. This PC: 621.14, accepted.
- **Qualified set (README, "Pick a model"):** Qwen3.5-2B (one Mac or one GB10), Qwen3.8-27B + DFlash2 (one Mac or one GB10), Nemotron 3.5 Lightning (GB10, RTX 3090, RTX PRO 6000), Bolt (RTX PRO 6000, 24 GB checkpoint), Gemma 4 31B, Qwen3.6-35B-A3B, LFM2.5-24B-A2B (GB10). The README states "no NVFP4/FP8 model run on sm_80 to sm_89 is recorded", so the 4090 is not model-qualified by the project, only by the Windows guide's PR note (2B ~340 tok/s, 27B 40 plain and 80 to 160 with the drafter).
- **Docs drift:** `docs/WINDOWS.md` still says `zig build native install -Dsm=89 ...`; on `main` that step no longer exists (`error: no such step: native`). The runbook's `zig build -Dgpu=sm_89 -Dnvcc=...` is the working form.

### Build (Windows, this PC)

- Tools: Zig 0.17.0 portable zip (100,268,137 bytes, sha256 `b5663f69...c428` matched ziglang.org's index), nvcc 13.2, MSVC Build Tools 2022 (`vcvars64.bat`).
- Command: `zig build -Dgpu=sm_89 -Dnvcc="...\CUDA\v13.2\bin\nvcc.exe"` from a vcvars64 prompt. About 8 minutes cold; output is one `tensorfold-native.exe` of 49,212,416 bytes (no `lib\` beside it; the kernel images are embedded). Without `tfgrammar.dll`, structured output and required tool calls answer HTTP 400 and everything else serves.
- `tensorfold-native capabilities --json` reports `"chip": "nvidia-sm89"`, `"backends": ["cuda"]` and lists `qwen3_5` (mlx-q4g64, exl3), nemotron_h, gemma4, lfm2_moe, qwen4 and more. Licence: Apache 2.0.

### Qwen3.5-2B MLX 4-bit (1.60 GiB pull at 4.4 MiB/s, revision `93760be4`)

`serve ... --name local-model --context 8192 --no-thinking --parallel 1`: loaded in 1.5 s; startup line "CUDA sm_89 device 0 (NVIDIA GeForce RTX 4090): model 1.58 GiB; 1 stream at once, 0.17 GiB each at a 8192-token window".

| Run | Result |
|---|---|
| 400 tokens, greedy, one request | **457 to 461 tok/s** (server log 484 tok/s decode, ttft 0.019 s) |
| 64-token reply | 0.41 s total |
| 8 requests at once (server `--parallel 1`) | 454 tok/s aggregate, so serial; the README's 1,500 across eight needs `--parallel 8` (not run) |
| VRAM, whole card with the model loaded | 3.1 GB (the card was shared; the other session's use came and went) |

Both figures beat the Windows guide's 340 tok/s. No refusal, no kernel error, replies coherent.

### Qwen3.8-27B MLX 4-bit + DFlash2 (14.98 + 3.58 GiB pulled, revisions: DFlash2 `50307d4c`)

`serve TensorFold/Qwen3.8-27B-MLX-4bit --name q27 --context 8192 --no-thinking --parallel 1 --temperature 0 --drafter z-lab/Qwen3.8-27B-DFlash2`. Fits: startup line "model 16.06 GiB (target + drafter); 1 stream at once, 1.46 GiB each at 8192 tokens, 5.4 GiB left"; 18.1 GB on the card in total. Target alone loads in 130 s, with the drafter 73 s (file cache warm). **The first request stalled 259 s** (the driver's first use of each kernel, as the README warns), so an app needs a long first-request timeout or a warm-up request.

| Greedy, one request, card idle (rerun after the other session finished; memory flat at 18.8 GB, three rounds) | tok/s |
|---|---|
| Plain (`"draft": false`), prose, 300 tokens | **46.8 to 48.9** |
| Plain, code, 400 tokens | **49.2** |
| DFlash2, prose | **147 to 151** |
| DFlash2, code | **300 to 307** |

The first run, made while the other session was still using the card, read 10 to 26 plain and 118 to 161 drafted; the figures above replace it. The second server start had no first-request stall (kernel cache warm, loaded in 44 s, first reply in 0.11 s); the 259 s stall seen on the first ever start is a one-off per driver cache, so an app still needs a long first-request timeout or warm-up. The README's 4090 numbers hold (40 plain, 80 to 160 drafted) and are beaten on this card: 47 to 49 plain, 147 to 151 on prose, about 300 on code (greedy, code is very predictable, so treat it as a ceiling). The 700 tok/s headline is a 5090 number. Replies stay identical to plain decoding per the README (not byte-compared here).

### Not run

- **llama.cpp on the same 27B:** no GGUF of Qwen3.8-27B here and none pulled (another 16 GB); the existing reference is Bolt 170 tok/s plain, 332 with DFlash (a different model, so not a like-for-like). **Nemotron Bolt** (24 GB checkpoint; no headroom on a 24 GB card).
- **WSL2 / Linux archive path**, `--parallel 8`, `tfgrammar.dll` build, and a run through the app.

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
- _(fixed, see "Follow-up work" below)_ Chatterbox Multilingual V3 and Japanese; TurboQuant off the Mac.
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


## Speed-up and quantisation pass, W1: diffusion cache strategies on CUDA (2026-11-11)

Plan: `docs/plans/rtx4090-speed-and-quant.md`. Harness: HTTP to the dev backend on 8876, same seed (7) for every run, peak VRAM from `nvidia-smi` every second, PSNR against the uncached output (frames 0, middle and last for video). Baselines are warm generation seconds, first runs after a load thrown away: FLUX.2 Klein 4B 1024x640 4 steps **4.2 s** (noise 4.2 to 5.0 s, 9.6 GB); Wan 2.2 TI2V 5B 832x480 49 frames 20 steps **35.5 s** (12.4 GB; the 42.5 s measured earlier included a cold start); LTX-2.3 4-bit 768x512 97 frames **59.6 s** (the 70 s earlier was a first run; 22.3 GB).

### 1. What the engine reports, and what the Studio offers (Windows, CUDA)

| Strategy | `is_available` | `applies_to` | Studio offers it |
|---|---|---|---|
| fbcache, taylorseer, pab, fastercache, magcache | yes (diffusers 0.42.1) | image, video | fbcache, taylorseer, pab only |
| teacache | yes, vendored patches for Flux, Hunyuan, LTX-Video, CogVideoX, Mochi | image, video | FLUX family on the image side; filtered on the video side |
| native, triattention, turboquant | text only (the last two not installed here) | text | n/a |

### 2. What the sweep found (all fixed, with tests)

| Finding | Fix |
|---|---|
| **The video route dropped `cacheStrategy` and `cacheRelL1Thresh`.** `VideoGenerationRequest` had no such fields, so no video the Studio sent ever ran cached. | Fields added and passed to `VideoGenerationConfig` (`test_generate_forwards_the_cache_strategy_to_the_runtime`). |
| **`cacheRelL1Thresh` capped at 1**, but TaylorSeer, PAB and FasterCache read it as a whole-step interval (2 and up): HTTP 422 for every interval. | Cap 16. |
| **Hooks stayed on the loaded pipeline.** The variant key ignores the cache choice, so an "off" run after a cached one stayed cached, and a second `enable_cache` raises "already enabled". FBCache used `apply_first_block_cache` directly, which `disable_cache` cannot undo. | `reset_diffusion_cache` runs before every request; FBCache goes through `enable_cache`. |
| **The video runtime threw the "not applied" note away** (the image runtime kept it), so a refused cache looked like a speed-up that did not happen. | Note kept on the clip. |
| **"Applied" but doing nothing: TaylorSeer on Wan.** diffusers' default module patterns match nothing in Wan's blocks (0 hooked modules, same speed, identical pixels). | Counts hooked modules, falls back to block-level patterns, refuses if still none (test on a tiny Wan transformer). |
| **FBCache on FLUX.2 crashed the run** (`unsupported operand type(s) for -: 'Tensor' and 'list'`: dual-stream head block, single-stream tail block) and fell back to a placeholder picture. **On LTX-2.3 it was HTTP 500** (`Model class LTX2VideoTransformerBlock not registered`). | Preflight refuses both; any hook-attach error becomes a "not applied" note. |
| **FasterCache crashes inside step 1** on FLUX.2 and Wan (`tuple index out of range`) after the weights are on the card. | A cached run that fails is retried once without the cache, re-seeded (the retry matches the uncached output, PSNR 99), and the note says so. |
| **TaylorSeer, PAB and FasterCache stall LTX-2.3 4-bit:** more than 15, 8 and 8 minutes against 60 s, card full at 24 GB, GPU at 99 %, no output (stopped by hand). Per-block tensors on top of a 22.3 GB peak spill into system memory. This is the "one step took an hour" case. | Per-block caches are refused when the pipeline is CPU-offloaded and this process leaves under 6 GB of the card free. Headroom is read from the card, so a 5090 or a Spark is unaffected. Live: LTX refuses at once (headroom 2.8 GB); Wan TI2V (also offloaded, headroom 11 GB) still applies it. |

### 3. Results (warm, same seed, PSNR against the uncached output)

**FLUX.2 Klein 4B, 1024x640, 4 steps** (baseline 4.2 s):

| Setting | Gen s | Speed-up | Peak VRAM | PSNR | Verdict |
|---|---|---|---|---|---|
| fbcache 0.08 / 0.12 / 0.20 | n/a | n/a | 9.6 GB | n/a | **FAIL, fixed:** crashed (above); now refused with a note |
| teacache | n/a | n/a | 9.6 GB | n/a | SKIP: no patch for `Flux2Transformer2DModel` (noted) |
| taylorseer 2 / 3 / 4 | 8.6 / 6.3 / 5.6 | 0.49 / 0.67 / 0.75x | 10.3 GB | 24.3 / 22.8 / 21.2 | **FAIL:** slower, different picture |
| magcache | 4.3 | 0.98x | 9.6 GB | 24.2 | **FAIL:** no gain, picture changes |
| pab 2 / 3 | 4.9 | 0.86x (noise) | 9.8 GB | 34.9 | **FAIL:** no gain (identical for both: 4 steps leave nothing to skip) |
| fastercache 2 / 3 | n/a | n/a | 9.4 to 16.5 GB | n/a | **FAIL, fixed:** `tuple index out of range`; now retried stock |

Four steps leave nothing to cache. The Studio now offers only "Off" for FLUX.2, FLUX schnell, "turbo" and other step-distilled rows (`isCacheUselessImageRepo`, vitest).

**Wan 2.2 TI2V 5B, 832x480, 49 frames, 20 steps** (baseline 35.5 s, 12.4 GB). Every setting keeps the scene (pink dusk, lighthouse, rocks) but changes fine detail (gulls, glow, sky texture); 20 dB or below is a visibly different clip.

| Setting | Gen s | Speed-up | Peak VRAM | PSNR (first / mid / last) | Verdict |
|---|---|---|---|---|---|
| fbcache 0.08 | 35.5 | 1.00x | 12.7 GB | 99 / 99 / 99 | PASS, useless: never skips |
| fbcache 0.12 | 28.7 | 1.23x | 12.8 GB | 19.8 / 20.3 / 21.2 | details change |
| fbcache 0.20 | 24.3 | 1.46x | 12.8 GB | 17.0 / 17.5 / 17.9 | **FAIL:** visibly different |
| teacache | 35.5 | 1.00x | 12.4 GB | identical | SKIP: no patch for Wan (noted) |
| taylorseer 2 / 3 / 4 (after the fix) | 27.2 / 24.1 / 23.0 | 1.31 / 1.47 / 1.55x | 16.1 GB (+3.7) | 17.5 to 19.1 | **FAIL on quality:** sky artefacts, different birds |
| magcache | n/a | n/a | n/a | n/a | SKIP: calibrated for FLUX only (noted) |
| pab 2 / 3 / 5 | 34.0 / 33.6 / 32.9 | 1.04 / 1.06 / 1.08x | 13.1 to 13.5 GB | 19 to 26 | **FAIL:** no gain, picture changes |
| fastercache 2 / 3 / 4 | 38.2 to 39.0 | 0.91 to 0.93x | 22.4 GB | 10 before the re-seed fix, 99 after | **FAIL, fixed:** crashes in step 1; the retry runs stock |

**Wan 2.2 TI2V 5B from a start picture** (FLUX.2 Klein still; baseline 36.2 s): fbcache 0.12 **1.24x**, 13.3 GB, PSNR 37.7 / 21.3 / 18.3; taylorseer 3 **1.45x**, 16.5 GB, PSNR 38.1 / 19.2 / 18.0. Frame 0 still matches the picture; later frames drift as in text-to-video.

**LTX-2.3 distilled 4-bit, 768x512, 97 frames** (baseline 59.6 s, 8 steps, model CPU offload): fbcache refused (block class not registered by diffusers; was HTTP 500); teacache and magcache not applied (no patch / FLUX only); taylorseer, pab and fastercache stalled before the headroom guard and are refused after it. No setting speeds it up. The Studio already hides the cache controls for distilled LTX (`fixedSampler`).

### 4. Interactions

| Check | Result |
|---|---|
| Cache + start picture (Wan TI2V) | PASS: 1.24x (fbcache 0.12), 1.45x (taylorseer 3), frame 0 matches the still |
| Cache + start and end pictures | SKIP: the only model that takes an end picture (LTX-2.3) has no working cache; TI2V refuses an end picture |
| Cache + model CPU offload (LTX-2.3 4-bit) | FAIL found, fixed: the stall above |
| Cache + Storyboard clip | not reachable: Storyboard builds its `VideoGenerationRequest` without cache fields, so clips run stock by design |

### 5. Recommended defaults (applied)

| Family | Default | Why |
|---|---|---|
| FLUX.2 Klein, FLUX schnell, turbo rows | off; the Studio offers only Off | 4 to 8 steps, nothing to skip; every cache was slower, equal, crashed or changed the picture |
| Wan 2.2 TI2V 5B | **off** (`recommended_settings` row carrying the measured reason) | the speed-ups (1.2 to 1.5x) all change the gulls and the glow; fbcache 0.12 and taylorseer 3 stay opt-in |
| LTX-2.3 distilled 4-bit | off, controls hidden | nothing applies; two caches stalled it |

Not changed: the FBCache 0.08 recommendation on the Wan 2.1 and LTX-Video rows (not measured here; the Wan 2.2 data says 0.08 may never skip).

**Tests:** `tests/test_cache_strategies.py` (+14), `tests/test_video_routes.py` (+1), `tests/test_recommendations.py` (+1), vitest `imageCacheStrategies.test.ts` (2), `tsc` clean. E2E: new check "Video diffusion cache strategy reaches the runtime" in phase 5 of `scripts/e2e_test_suite.py`. `tests/test_video_routes.py::test_catalog_slow_on_labels_reach_the_payload` fails here for a reason unrelated to this change (KeyError on the Mac-only `prince-canuma/LTX-2.3-distilled` row).


## Speed-up and quantisation pass, W2: other diffusion speed-ups (2026-11-11)

Same harness and baselines as W1 (Wan 2.2 TI2V 5B 832x480, 49 frames, 20 steps: **35.5 s**, 12.4 GB; LTX-2.3 4-bit 768x512, 97 frames: **59.6 s**, 22.3 GB). The measuring knobs live in `backend_service/helpers/speedups.py` (`CHAOSENGINE_SPEEDUPS=vae_tiling,vae_slicing,compile,time_decode`, off unless set, covered by `tests/test_speedups.py`).

| Item | Result | Numbers |
|---|---|---|
| `torch.compile` on the repeated transformer blocks (Wan) | **measured, off by default** | First attempt failed with "Failed to find C compiler": `cl.exe` is installed (VS Build Tools 2022) but not on PATH. Run from a `vcvars64` prompt with `triton-windows 3.7.1` (scratch folder, not the app) it works: first run **136.8 s** (about 106 s of compilation), then **30.9 s and 29.0 s** against 35.5 s (**1.22x**). Peak VRAM 23.3 GB on the two compiled runs against 12.4 GB, and fine detail changes (PSNR 21 dB against the stock clip). Not worth a default: the gain pays back the compile after about 16 clips and the 23 GB peak leaves no room on smaller cards. The option checks for Triton and a compiler (`cl` on PATH or `CC`) and reports why instead of failing a step. |
| Attention backend: SDPA vs sage vs flash | **SKIP** | `set_attention_backend("sage")` needs `sageattention>=2.1.1`; PyPI stops at 1.0.6 (Triton v1), so the app's existing "SageAttention unavailable" path is what runs. `flash-attn>=2.6.3` has no Windows wheel on PyPI. SDPA stays. Building SageAttention 2 needs MSVC plus the CUDA toolkit. |
| VAE tiling / slicing (Wan, 832x480) | **no gain, no harm** | 34.8 s / 35.3 s against 35.3 s; peak 12.6 GB in all three. VAE decode is 7.5 s of the 35 s (21 %). |
| VAE tiling (LTX-2.3 4-bit) | **no gain** | 58.4 s against 58.0 s; peak 18.1 GB. Decode is 3.7 s of 58 s (6 %). |
| Fewer steps on Wan TI2V (`unipc`) | measured, default **kept at 20** | 8 / 12 / 16 / 20 steps: 21.1 / 25.4 / 29.9 / 34.7 s (about 1.2 s per step on top of 14 s fixed: text encoder, decode, offload swaps). Fewer steps give a different clip rather than a blurrier one: composition stays at 16, a different (still clean) lighthouse at 12 and 8; PSNR against the 20-step clip 12 to 16 dB. 16 steps (1.16x) or 12 (1.37x) suit drafts. |
| Scheduler at 12 steps (Wan) | no better | `dpm++` 26.4 s, `flow-euler` 26.0 s, `unipc` 25.4 s. Plain `euler` is refused for Wan (flow prediction; the app already offers `flow-euler` for that). |
| Distilled rows (Klein 4 steps, LTX-2.3 8 steps) | n/a | nothing to cut; their model cards set the steps. |
| TeaCache-style skipping | see W1 | |

**Defaults changed:** none; nothing in W2 beat the stock path on this card. **Owner tests:** `torch.compile` on Linux CUDA and on cards under 24 GB (peak), SageAttention 2 on Linux CUDA; VAE tiling on 12 to 16 GB cards, where decode peak matters more than it does on 24 GB.


## Speed-up and quantisation pass, W3: quantisation for image and video (2026-11-11)

Standalone runs (one fresh process per option, warm second generation, same seed 7, PSNR against the bf16 output), outside the app so each option is isolated: scripts in the session scratch folder. Optional packages (`torchao 0.17`, `gguf 0.19`, `nunchaku 1.1.0`) went into scratch folders, not the app's overlay.

**Already in the app:** NF4 via bitsandbytes (FLUX.1 only, `try_load_nf4_flux_transformer`, plus LTX-2.3), torchao int8 weight-only (FLUX.1, Mac only), GGUF single-file transformers, Nunchaku SVDQuant (FLUX / Qwen-Image / SD3 / SANA, catalog rows), and FP8 layerwise casting (`fp8LayerwiseCasting`, gated to compute capability 8.9 and up).

**FLUX.2 Klein 4B, 1024x640, 4 steps, whole pipeline on the card** (bf16: transformer 7.75 GB, 9.2 s, peak 17.6 GB):

| Option | Transformer | Time | Peak VRAM | PSNR | Looks | Verdict |
|---|---|---|---|---|---|---|
| FP8 layerwise (E4M3 storage, bf16 compute) | 3.89 GB | 10.4 s (+13 %) | 13.8 GB | 28.4 | same picture | **PASS** (already shipped; confirmed live) |
| NF4 (bitsandbytes) | 1.94 GB | 9.7 s (+5 %) | 12.1 GB | 22.1 | same picture | **PASS** for the model; the app only wires NF4 for FLUX.1 |
| 8-bit (bitsandbytes int8) | 3.88 GB | 10.9 s | 13.8 GB | 10.7 | **pure noise** | **FAIL** (upstream: int8 on FLUX.2 gives noise) |
| torchao int8 weight-only | n/a (logical size) | 10.2 s (+11 %) | 13.8 GB | **33.1** | same picture | **PASS, best quality**; not wired on CUDA (FU-160) |
| torchao fp8 weight-only | n/a | 12.9 s (+40 %) | 13.8 GB | 27.4 | one stray object | PASS, slow |

**Wan 2.2 TI2V 5B, 832x480, 49 frames, 20 steps, model CPU offload** (bf16: transformer 10.02 GB, 32.6 s; peak 11.5 GB is the text encoder, so weight savings do not show in the peak). PSNR is against the bf16 clip; 20 dB or less is a visibly different clip.

| Option | Transformer | Time | PSNR | Looks | Verdict |
|---|---|---|---|---|---|
| FP8 layerwise | 5.11 GB | 32.7 s | 19.1 | scene and lighthouse kept, birds differ | **PASS** |
| NF4 (bitsandbytes) | 2.54 GB | 31.6 s | 13.7 | the lighthouse breaks into a light beam | **FAIL on quality** (5B is too small for 4-bit) |
| GGUF Q4_K_M (diffusers) | 3.45 GB | **154 s (4.7x slower)** | 17.0 | | **FAIL on speed** (dequantises every layer every step) |
| torchao int8 / fp8 weight-only | n/a | 34.0 s / 35.1 s | 19.5 to 20.4 / 17.0 to 18.9 | | neutral: no speed or peak gain with offload |

Finding while testing: the generic `DiffusionPipeline.from_pretrained(..., transformer=obj)` silently ignores a quantised transformer for Wan 2.2 (`pipe.transformer is obj` is False, bf16 weights are loaded); the specific `WanPipeline.from_pretrained` honours it. The app calls the specific class (`pipeline_cls.from_pretrained`), so it is not affected; worth knowing before anyone switches to `AutoPipeline`.

| Item | Result |
|---|---|
| Nunchaku / SVDQuant on this torch | **SKIP, does not load.** A Windows wheel for torch 2.11 and cp312 exists (`nunchaku-ai/nunchaku`, `nunchaku-1.1.0+torch2.11`, 112 MB); installed alone into a scratch folder, `import nunchaku` fails with "DLL load failed while importing _C: The specified procedure could not be found" against this PC's `torch 2.11.0+cu128`. The wheel's binary does not match that CUDA build (the app's own GPU runtime installs a cu130 torch). The 6.75 GB SVDQuant pack and the 10 GB of FLUX.1-schnell text encoders were downloaded for it (HF cache, 17 GB, Apache 2.0) and are still there. **Owner test:** with the cu130 GPU runtime. The app's pin is 1.2.1; the newest torch 2.11 Windows wheel is 1.1.0. The PyPI `nunchaku` (0.16.1, 17 KB) is an unrelated stub, as the app already notes. |
| GGUF image / video | measured above (Wan TI2V Q4_K_M, 3.4 GB, Apache 2.0, from `QuantStack/Wan2.2-TI2V-5B-GGUF`): works, but 4.7x slower. GGUF stays a memory-only option for models that do not fit. |
| LTX-2.3 bf16 with fp8 | SKIP: the bf16 rows are 100 GB downloads. The 4-bit row already runs (section "LTX-2.3 with sound"). |
| NVFP4 / hardware fp8 | **Gate written, live run SKIP (no hardware).** `backend_service/helpers/gpu_formats.py`: `nvfp4_supported(cc)` is true from compute capability 10.0 (B200 10.x, RTX 50-series 12.0, DGX Spark 12.1) and false for this card (8.9); `fp8_supported` from 8.9. Read from the card, never assumed. `tests/test_gpu_formats.py`. **Owner test:** an RTX 5090 or a Spark. |

**Catalog change:** none. Every row that passed is either already offered (FP8 layerwise, NF4 for FLUX.1) or needs a new load path and a toggle (torchao int8 on CUDA, FU-160); the two Wan options that save memory fail on quality or speed.


## Speed-up and quantisation pass, W4: text-model caches and speculative decoding (2026-11-11)

llama.cpp server on the 4090, `-ngl 99 -fa on`, one process per KV type, a seeded prose filler with the needle sentence ("The secret passphrase for the vault is crimson-otter-7731.") at mid-depth and the question at the end, temperature 0, the context window set just above the prompt. Two binaries: the TurboQuant fork `llama-server-turbo.exe` (CUDA, `feature/turboquant-kv-cache`, adds `turbo2/3/4` cache types) for the KV sweep, and the standard b11981 CUDA build for the drafter runs (the fork's server rejected the DFlash drafter path). A first sweep was discarded: a stray earlier copy of the script shared the port and the GPU. Numbers below are from clean reruns with the card at 0.5 GB idle.

### TurboQuant and the native KV types

**Nemotron 3.5 Bolt E4B Q4_K_M** (hybrid model, small KV: 1.04 GB at f16 for 66K tokens). Needle found in every cell. VRAM is the whole process; decode tok/s at 4K / 16K / 32K / 64K:

| KV type | KV buffer | VRAM | Decode tok/s (4K / 16K / 32K / 64K) | Needle |
|---|---|---|---|---|
| f16 (native) | 1040 MiB | 5.5 GB | 141 / 152 / 143 / 125 | 4/4 |
| q8_0 | 552 MiB | 5.1 to 5.3 GB | 132 / 138 / 116 / 90 | 4/4 |
| q4_0 | 292 MiB | 4.8 to 5.0 GB | 130 / 138 / 117 / 93 | 4/4 |
| turbo4 | 276 MiB | 5.1 to 5.3 GB | 115 / 119 / 101 / 80 | 4/4 |
| turbo3 | 203 MiB | 5.0 to 5.2 GB | 126 / 132 / 112 / 89 | 4/4 |

Prefill 8.3 to 12.2K tok/s for all types (within 10 %). On this model the KV is 1 GB of 5.5, so no type saves more than 0.7 GB, and every quantised type is slower than f16 (11 to 36 % at 64K). **TurboQuant buys nothing here.**

**Qwen3-0.6B Q8_0** (dense, KV-heavy: 3.7 GB f16 at 33K tokens), contexts up to 32K (its trained window is 40K):

| KV type | KV buffer | VRAM | Decode tok/s (4K / 16K / 32K) | Needle (4K / 16K / 32K) |
|---|---|---|---|---|
| f16 | 3696 MiB | 5.3 GB | 395 / 269 / 180 | pass / pass / pass |
| q8_0 | 1964 MiB | 3.7 GB | 363 / 254 / 174 | pass / pass / pass |
| q4_0 | 1040 MiB | 2.8 GB | 360 / 259 / 183 | **fail** x3 |
| turbo4 | 982 MiB | 2.7 GB | 362 / 260 / 179 | **fail** x3 |
| turbo3 | 722 MiB | 2.5 GB | 386 / 304 / 225 | **fail** x3 |

On a KV-heavy model the saving is real (f16 to turbo3: 5.3 to 2.5 GB; turbo3 decodes 25 % faster at 32K because it moves less KV), but 4-bit keys and values (q4_0, turbo4, turbo3 alike) lose the needle on a 0.6B model, even at 4K. q8_0 is the safe setting: half the KV, the answer intact, speed unchanged. TurboQuant's recovery of 4-bit quality (the published claim) did not show on this small model: turbo4 failed where q4_0 failed. Not tested: a 7B or larger dense model (none in the library; the 27B Qwen3.8 GGUF folder is empty). **Owner test:** a 7B to 14B dense model at 32K and 64K, f16 against q8_0 against turbo4.

### DFlash drafter (Nemotron 3.5 Bolt, standard b11981 CUDA build, 128 generated tokens)

| Context | Plain decode | With the DFlash drafter | Drafted / accepted (first request) |
|---|---|---|---|
| 4K | 130 tok/s | 108 (0.83x, first request) | 28 / 6 |
| 16K | 148 | 189 (**1.28x**) | 28 / 6 |
| 32K | 136 | 178 (**1.31x**) | 28 / 6 |
| 64K | 119 | 165 (**1.38x**) | 28 / 6 |

The drafter's gain grows with context because the target's own decode slows while the 0.74 GB drafter's cost stays flat; its VRAM cost is 1.7 GB (5.6 to 7.3 GB). The continuation of a repetitive filler is a hard case for drafting (21 % of drafted tokens accepted); section 6's 332 tok/s was on ordinary prose. **SKIP: a second model with a pinned drafter** (none is in the library; the other pinned drafters are 8 to 27B downloads).

### TriAttention, DDTree, MTPLX, TensorFold (read from the code, then checked)

| Item | Result |
|---|---|
| TriAttention off the Mac | **SKIP, cannot run.** The pinned commit builds and imports on Windows (a pure-Python 0.2.0 wheel into a scratch folder), but `TriAttentionStrategy.is_available` needs the MLX entry point or vLLM, and vLLM has no Windows build; the app's WSL2 route (`vllm_wsl_engine.py`) needs a distro, and this PC has none (`wsl -l` reports no distributions). Badge "Install" with the Linux/CUDA hint. **Owner test:** Linux CUDA or a WSL2 distro. |
| DDTree | **Mac-only as built; the algorithm is portable.** `ddtree.py` builds the tree with NumPy, but verification (`ddtree_attention.py`: `tree_forward`, `compact_kv_cache`, `prefill_with_capture`) imports `mlx` and `mlx_lm`. A llama.cpp route does not exist (the server's `--spec-type` has linear drafters only: `draft-dflash`, `draft-mtp`, ngram; tree verification lives in the example binary, not the server). A torch route could exist: a transformers target with a 4D tree attention mask, plus the same DFlash drafter. Follow-up FU-161. |
| MTPLX | **Mac-only** (forked MLX in its own venv; `routes/setup/mtplx.py` reports `supported` only for `darwin` + `arm64`). The CUDA counterpart already exists without it: llama.cpp's `--spec-type draft-mtp` for MTP-head GGUFs, wired in `llama_cpp_engine.py`. No follow-up needed. |
| TensorFold | **Mac-only in the app** (`scripts/install-tensorfold.sh` requires arm64 Python; Python 0.6.6 pin). The 1.x native engine builds from source on this PC (2B at 460 tok/s); the other session's section "TensorFold 1.x on the RTX 4090" and FU-111 cover it. |
| `cache-strategy-matrix.py` | **Not re-run:** nothing in this pass touched the text-model cache or engine code, so the cells are as recorded in section 6 (one cell runs on this PC). |

**Defaults changed:** none. The sweep supports what the app already does (native f16 default, q8_0 as the safe quantised choice); it argues against offering 4-bit KV types on models under a few billion parameters.


## Speed-up and quantisation pass, W5: leftover PC items (2026-11-11)

| # | Item | Result |
|---|---|---|
| 1 | Per-step watchdog (image and video) | **DONE, live-verified with tight limits.** `backend_service/helpers/step_watchdog.py`, wired around the denoising call of both runtimes (`image_runtime`, `video_runtime`). A side thread reads the progress tracker. The limit is 8x the median step time seen so far, never under 120 s; before two steps have finished it is 600 s (the first step carries warm-up). Loading, encoding and decoding are not timed. When a step passes the limit the cancel flag is raised with a message ("The video stopped: step 2 of 8 ran for 300 s without finishing... usually means the GPU ran out of memory and spilled into system memory"); if the step then returns, the step-end callback raises and the route answers **504 with that message** (a user's Cancel stays 409 "cancelled"). If the run is still on the same step 120 s later nothing in-process can end a call stuck in a GPU kernel, so the hard stop writes `~/.chaosengine/last_stall.json` and exits the backend with code 75 (the desktop shell's "Restart Backend" brings it back). All limits and the hard stop switch are environment variables (`CHAOSENGINE_STALL_*`). Tests: `tests/test_step_watchdog.py` (13, including a fake pipeline whose second step sleeps) and a route test for 504 versus 409. Live: with the limits lowered to 0.5 s, Wan TI2V got **HTTP 504 and the message after 50 s (load included), and the next request ran at once (4 s to the same answer)**, so the tracker resets between runs. The hard stop itself was not exercised on a real hang (it would end the backend); its exit path is covered by the unit test. No E2E check: there is no HTTP surface to assert beyond the 504 above. |
| 2 | Rebuild the installer with all post-installer fixes | **BUILT, NOT INSTALLED (needs your OK).** Built the way `release.yml` does (fresh Python 3.12 venv at `C:\Users\Dan\release-venv-w5` with `release-constraints.txt`, `npx tauri build --bundles nsis`, release staging, no updater signing; `tauri.conf.json` untouched): `src-tauri\target\release\bundle\nsis\ChaosEngineAI_0.10.0_x64-setup.exe`, **136.6 MB**. Manifest: `llamaServer: bin/llama-server.exe`, `warnings: []`; the runtime archive holds `install-mflux.sh`, `install-ltx2-mlx.sh`, `install-mtplx.sh`, `install-tensorfold.sh`. The section 7 smoke (install on top, backend start, GPU runtime, CUDA torch) needs the installed app replaced, so it waits for you. The build includes this pass's backend fixes except the last two route edits (504, env limits), which came after; rebuild once more before you ship. |
| 3 | Storyboard "renders appear in the project" | **Not a routing bug; descriptions and instructions tightened.** The agent guide's own `poster_set` prompt tells the model to run `generate_image` once per poster, so Nemotron choosing `generate_image` for "two posters" was following the guide: a poster is not a film, and its file lands in the image outputs by design. The failure was the expectation. What was ambiguous: `generate_image` did not say what it is for, and `storyboard_project` did not say it starts empty. `generate_image` now reads "...for a picture on its own (poster, cover). Not for film stills: storyboard_generate makes those", and the server instructions say "A poster or any single picture is generate_image, never a film project... Films: storyboard_project (it starts empty: add shots, then generate)" (`tests/test_mcp_bridge.py`, golden tool list updated). The two empty projects the 0.6B model made are the guard question already noted; not added. |
| 4 | Windows test portability | **Python 38 -> 16 failures; vitest 4 -> 1.** Fixed: `voice_env._is_windows()` replaces tests patching the global `os.name` (which made pathlib build `PosixPath` on Windows: 20 tests in `test_voice_kokoro_optional`, `test_voice_model_files`, `test_voice_io`); path separators in `test_measure_voice_models` and `test_music_sound_1_0` (`as_posix()`); `signal.SIGKILL` in `test_storyboard_v2` (`proc.kill()`); vitest `validateGate` (`fileURLToPath`, the `C:\C:\` spawn path, 3 tests). A real catalog bug the audit found: the `ltx-2-diffusers` family had no row in the size-limit table (`catalog/size_limits.py`). Left for the Mac session or host-bound: 5 `test_setup_routes` CUDA-index tests and `test_gpu_detection` (they call the real `nvidia-smi`), 4 `test_media_input` (Mac commands), 2 `test_size_limits` and `test_video_routes::...slow_on_labels` (Mac catalog rows), 2 `test_voice_speak`, and the `test_storyboard_v2` killed-render test (its child never reaches shot 3 here). The one remaining vitest failure is the Video Studio memory-tooltip test (the tooltip text carries a host-detected card). **A run while a model was loaded in a dev backend showed 13 more music failures (host memory gate, "only 11.7 GB available"); they pass with the backend stopped.** |
| 5 | Paste-image check | **Owner test:** Ctrl+V of a picture into a Cast reference box (the test browser cannot carry the OS clipboard). |
| 6 | Storyboard with LTX-2.3 4-bit as the project default | **PASS after three fixes.** Three shots (a lighthouse still, a harbour still, a shot continuing from the last frame), FLUX.2 Klein stills, an end frame on shot 1: both stills and all three clips with sound in **272 s** (one batch), **peak 22.6 GB**, each clip 2.71 s 864x480 with a 48 kHz stereo AAC track; export **8.13 s 854x480 H.264 + AAC stereo**. What it found: **(a)** the catalog called the pack "won't fit" on a 24 GB card (estimate 25 GB against 24 usable), so the agent bridge and Storyboard refused it ("not an installed video model") although it runs at 22.3 GB; the row now carries the measured 22.5 GB and the fit logic accepts a figure measured on a 24 GB NVIDIA card as "tight" (`footprintMeasuredOn: cuda`; a 16 GB card still says won't fit; `test_recommendations.py`); **(b)** `Shot.endImage` was read by clip generation, the staleness check and the bundle, but no op could set it, so no project could have an end frame: new op `set_end_image` (undoable, takes an upload handle or a file; empty clears it), route tests and an E2E check ("a shot takes an end frame and loses it again", phase 10: PASS, 9 of 9); **(c)** nothing in the UI or the agent's `storyboard_shot` tool offers the op yet (FU-162). |

**E2E on this PC:** phase 10 (Storyboard) PASS, 9 checks. Phase 5 (video): the new "cache strategy reaches the runtime" check and the Wan image-to-video check SKIP on the host memory gate (the installer build and a loaded model held the RAM); the LTX-2.3 diffusers check PASS; the LTX-2.5 check FAILS here because that row is Mac-only (not touched by this pass).


## Qwen3.8-27B + DFlash2 through llama.cpp CUDA (2026-11-11)

**Verdict: PASS (runs, 1.8 to 2.6x), far short of 700 tok/s.** The lane already existed (FU-089): llama.cpp b11981 reads DFlash 2 from the `dflash-` sidecar's GGUF metadata and the app pairs it as `--model-draft <sidecar> --spec-type draft-dflash --spec-draft-n-max 15`. Nothing needed converting. Weights: `ggml-org/Qwen3.8-27B-GGUF` `Qwen3.8-27B-Q4_K_M.gguf` (18.07 GiB) + `dflash-Qwen3.8-27B-Q8_0.gguf` (1.92 GiB); the drafter's Hub twin is `z-lab/Qwen3.8-27B-DFlash2` (safetensors, not used here).

Setup: b11981 CUDA build (`~/.chaosengine/test-llama-b11981/cuda`), `--parallel 1 --ctx-size 4096 --jinja --flash-attn on --reasoning off --n-gpu-layers 99`, drafter on GPU, greedy, 300 tokens, one warm-up, 3 runs per prompt, llama.cpp's own timings, card idle.

| Prompt | No drafter | DFlash 2 | Speed-up | Draft acceptance |
|---|---|---|---|---|
| Prose (essay) | 43.1 tok/s | 77.8 | 1.8x | 31 % |
| Code (LRU cache module) | 43.1 | 93.0 | 2.2x | 39 % |
| Structured (JSON array) | 43.1 | 111.8 | 2.6x | 50 % |
| Peak VRAM (whole card) | 19.8 GB | **24.0 GB of 24.5** | | |
| Load | 24.6 s | 10.5 s (warm cache) | | |

Runs were within 1 % of each other. Comparison, same card: TensorFold's Python engine with the same drafter measured 47 to 49 plain, 147 to 151 prose and about 300 on code (section above), so on a 4090 the llama.cpp route gets about half of TensorFold's drafted prose speed and a third of its code speed, for the same plain speed. Acceptance is low (31 to 50 %); the cause is not investigated (block size, drafter quant, or llama.cpp's DFlash 2 path against the vendor engine). The 700 tok/s headline is a 5090 number.

**Limits.** Only 4096 context was run; with the drafter the card is full (24.0 of 24.5 GB), so longer context needs the app's `--fit` to shrink GPU layers, and 12 to 16 GB cards cannot hold the 4-bit model at all (a 32 GB 5090 or a 128 GB DGX Spark can hold Q8_0). Output equality with plain decoding was not diffed (greedy verification should make it identical). Not run through the app: the E2E check is added but skips without the files in its model scan.

**Shipped:** catalog note on the `ggml-org/Qwen3.8-27B-GGUF` row (Q8_0 is 26.6 GB; Q4_K_M + sidecar measured), test `test_dflash2_sidecar_for_the_4bit_pack_picks_its_closest_quant` (real file names), E2E check "Qwen3.8-27B DFlash 2 drafter (GGUF sidecar)" in `scripts/e2e_test_suite.py` (SKIP unless the model with a `dflash-Qwen3.8-27B*` sidecar is on disk; fails if the note names no drafter). The mapping in `DFLASH2_DRAFT_MODEL_MAP` stays vLLM-only: GGUF needs no entry. Follow-up FU-163 (acceptance gap, GGUF row for 24 GB cards).

## Report back (speed-up and quantisation pass, W1 to W5)

Branch `test/rtx4090`. Every item is recorded with numbers in the sections above ("W1" to "W5").

### What is faster
- **Wan 2.2 TI2V 5B (35.5 s baseline):** First Block Cache 0.12 is **1.23x**, TaylorSeer 3 is **1.47x**, `torch.compile` is **1.22x** (after a 106 s compile), 12 steps is **1.37x**. Every one of them changes fine detail (PSNR 17 to 21 dB), so **none is a default**. VAE tiling and slicing: no change.
- **Text on CUDA:** the DFlash drafter gives **1.28x at 16K, 1.31x at 32K, 1.38x at 64K** on Nemotron 3.5 Bolt (0.83x at 4K, first request). A quantised KV cache is not faster on that model (11 to 36 % slower at 64K); on a KV-heavy dense model turbo3 decodes 25 % faster at 32K.
- **Nothing sped up:** FLUX.2 Klein (4 steps) and LTX-2.3 distilled (8 steps): every cache was slower, equal, refused, crashed or changed the picture.

### What is smaller
- **FP8 layerwise casting:** transformer halved (Klein 7.75 to 3.89 GB, Wan 10.0 to 5.1 GB), 28 dB on Klein, scene kept on Wan. Already in the app; confirmed.
- **FLUX.2 Klein:** torchao int8 weight-only is the best option measured (33 dB, peak 17.6 to 13.8 GB, +11 % time) and NF4 the smallest (1.94 GB, 22 dB). Neither is wired for FLUX.2 on CUDA (FU-160).
- **KV cache (Qwen3-0.6B, 33K tokens):** f16 3.7 GB, q8_0 2.0 GB (needle intact), turbo3 0.7 GB (needle lost, as q4_0 and turbo4 did).

### What was refused, and why
- **bitsandbytes int8 on FLUX.2:** pure noise. **NF4 on Wan 5B:** the lighthouse breaks. **GGUF Q4 on Wan:** 4.7x slower. All three stay out.
- **FBCache on FLUX.2 and LTX-2:** diffusers cannot compare dual-stream with single-stream blocks and has no LTX-2 block registration; refused with a note (it used to crash the run or answer HTTP 500). **TeaCache and MagCache:** no patch or calibration for these models; noted, not applied.
- **TaylorSeer, PAB and FasterCache on LTX-2.3 4-bit:** they stalled the run for more than 8 minutes against 60 s; refused when the pipeline is offloaded and under 6 GB of the card is spare (read from the card).
- **4-bit KV types on a 0.6B model:** lose the needle; the app should not offer them below a few billion parameters.
- **SageAttention 2, flash-attn, Nunchaku, TriAttention, DDTree, MTPLX, TensorFold on this PC:** SKIP with the reason in each section (no PyPI wheel for SageAttention 2; no Windows flash-attn wheel; the Nunchaku `torch2.11` wheel does not load against `cu128` torch; TriAttention needs vLLM and WSL2 has no distro; DDTree's verifier is MLX-only, MTPLX is Apple Silicon only, TensorFold's app path is arm64 only).

### Bugs found and fixed (each with a test)
Video cache setting never reached the runtime; its threshold was capped at 1 so interval caches got 422; hooks outlived the request; the video "not applied" note was dropped; TaylorSeer on Wan hooked nothing while claiming to apply; FasterCache crashed mid-run (now retried stock, re-seeded); the catalog called LTX-2.3 4-bit "won't fit" on its own target card, which made the agent bridge and Storyboard refuse it; Storyboard end frames had no op to set them; the `ltx-2-diffusers` family had no size-limit row; a step that never returns could not be stopped (step watchdog, 504 with a reason); the golden tool list and 23 Windows-only test failures.

### Defaults changed
FLUX.2 / schnell / turbo rows: the Studio offers only "Off" for caches. Wan 2.2 TI2V 5B: cache off, with the measured reason. LTX-2.3 4-bit: footprint 22.5 GB measured, "tight" on 24 GB. Agent and Storyboard instructions: a poster is `generate_image`, not a film project. Nothing else.

### Owner tests still needed
- **Blackwell (RTX 5090) and DGX Spark:** NVFP4 (the gate is written and tested, no live run), `torch.compile`, the bf16 LTX-2.3 rows, Nunchaku on the cu130 runtime.
- **12, 16 and smaller cards:** the cache headroom guard (6 GB), VAE tiling (decode peak matters more there), FP8 layerwise and NF4 on FLUX.2, `torch.compile` peak (23 GB here).
- **Linux CUDA:** SageAttention 2, flash-attn, TriAttention with vLLM, `torch.compile` without the Visual Studio shell.
- **A 7B to 14B dense model** at 32K and 64K: f16 against q8_0 against turbo4 (only a 0.6B and a 4B were available).
- **By hand:** Ctrl+V of a picture into a Cast box. **Your OK needed:** installing the rebuilt installer over the installed app and rerunning the section 7 smoke (the installer is built: `src-tauri\target\release\bundle\nsis\ChaosEngineAI_0.10.0_x64-setup.exe`).
- Leftover disk: the FLUX.1-schnell Nunchaku pack and text encoders (17 GB) and the Wan GGUF (3.4 GB) are in the Hugging Face cache; `C:\Users\Dan\release-venv-w5` holds the installer-build venv.


## TensorFold 1.x integrated as the CUDA text engine (2026-11-11, FU-111)

Built on `test/rtx4090`; design in the section "TensorFold 1.x on the RTX 4090" above.

- **Installer** (`backend_service/routes/setup/tensorfold_native.py`): Python thread, no bash, no venv. Downloads `tensorfold-<ver>-linux-x86_64|linux-aarch64.tar.gz`, checks sha256 against the published `.sha256`, refuses path traversal, unpacks to `~/.chaosengine/tensorfold-native/<ver>-<platform>/`, runs `capabilities --json`, then writes `install.json`. **Windows has no release archive**, so there it adopts a built `tensorfold-native.exe` (copy, hash, check, record) or downloads a self-hosted archive named by `CHAOSENGINE_TENSORFOLD_WINDOWS_URL` and `CHAOSENGINE_TENSORFOLD_WINDOWS_SHA256`. The Linux download path is covered by unit tests with a faked download, not run live here.
- **Gate** (`_tensorfold_native.evaluate_gate`): reads compute capability, driver and free VRAM from `nvidia-smi`, then the binary's own chip and back ends. Refuses: no NVIDIA GPU, driver below R580, compute capability under 8.0 or unreadable, a card smaller than the smallest model, a binary that does not report CUDA. Release image set sm_80/86/120/121 = "ready"; any other Ampere-or-newer card is "unqualified" until the binary reports a CUDA chip. Nothing names a card; the 4090 shows `ready` because its own source build reports `nvidia-sm89`.
- **Engine**: `TensorFoldNativeEngine` (engine id `tensorfold-native`) subclasses the Mac engine, so streaming, tool calls and the acceptance metric are shared. Serve flags (`--drafter`, `--context`, `--vision`, `--parallel`, `--snapshot-dir`, `--no-update-check`) go on the command line only when `capabilities --json` lists them. A load refuses when free VRAM is under the registry estimate (27B: 18.8 GB measured at 8192 tokens, rescaled for context, minus the drafter when it is off).
- **Routing**: `backend=tensorfold` picks it when the Mac engine is absent; `auto`/`mlx` picks it for a registry repo when native is installed and the Mac engine is not (MLX-format checkpoints have no other engine on a CUDA host). The Mac Python 0.6.6 path is unchanged and wins when installed.
- **Live, through the app (backend on 8876, RTX 4090, driver 621.14)**: status gate `ready` (`nvidia-sm89`), load of Qwen3.8-27B + `z-lab/Qwen3.8-27B-DFlash2` in 46 s (engine `tensorfold-native`, drafter engaged, 21.4 GB used on the card), then 300-token replies: **49.6 tok/s plain, 132 tok/s drafted on prose, 206 tok/s drafted on code** (app default sampling, not forced greedy; the standalone greedy runs read 147-151 and about 300). Unload returned the card to 1.8 GB.
- **Tests**: `tests/test_tensorfold_native.py` (43), vitest `tensorfoldSupport.test.ts` (+2), `tsc` clean, `npm run i18n:validate` OK (two new en keys, other locales `null`). E2E: phase 1 check "TensorFold 1.x native, CUDA (FU-111)" (skips without the binary); the Mac check and the catalog check skip or relax for the native engine.
- **Not done**: Linux archive installed live; `tfgrammar` (structured output) on the Windows source build; Gemma 4, Qwen3.6-MoE and LFM2.5 registry rows; image input run.


## Rebuilt installer installed over the installed app, section 7 smoke again (2026-11-11)

Installed over the existing 0.10.0 (silent `/S`, exit 0) with your OK. The build includes everything up to `18cb80e6`.

| Item | Result | Evidence |
|---|---|---|
| `./build.sh` as written | **FAIL on this PC, with a reason** | It installs into and stages the dev `.venv` (torch, diffusers, bitsandbytes and the rest): the runtime archive came out at **3.04 GB** and `makensis` stopped with "failed creating mmap" of it (NSIS cannot map past 2 GB). It also upgrades packages in the dev `.venv` without `release-constraints.txt`. A release build must stage a clean venv. |
| Installer built instead | PASS | Same steps as `release.yml` from a clean Python 3.12 venv (`C:\Users\Dan\release-venv-w5`, `release-constraints.txt`), `npx tauri build --bundles nsis`, release staging, `tauri.conf.json` untouched: `ChaosEngineAI_0.10.0_x64-setup.exe`, **136.6 MB**; runtime archive **131.9 MB**. |
| Install over the old app | PASS | Exit 0; `ChaosEngineAI.exe` product version 0.10.0, timestamp moved from Nov 10 12:10 to Nov 11 15:11. No app or backend process was running first. |
| Backend starts | PASS | `/api/health` 200 after **24 s**; `appVersion` 0.10.0; runtime unpacked to `%TEMP%\chaosengine-embedded-runtime\windows-x86_64-30e8dd42` (old extractions are still never cleaned). |
| Overlay guard | PASS | `extras\cp312\stale-core` holds 53 entries, all from Nov 4 and Nov 10; nothing new after the update, so nothing was quarantined. The overlay still has `torch 2.14.1+cu130`, `diffusers 0.42.1`, `transformers 5.22.2`, `bitsandbytes 0.50.2`. |
| GPU runtime present, torch is the CUDA build | PASS (existing runtime kept) | The installed app's own Python with the overlay: `torch 2.14.1+cu130`, CUDA 13.0, `cuda.is_available() True`, RTX 4090. Image and video runtimes report diffusers ready, expected device `cuda`, 23.99 GB. |
| GPU runtime install from scratch | **SKIP** | Not re-run: the overlay is kept across updates by design and the install (466 s, 17 packages) passed on the earlier build; reinstalling would only repeat it. |
| One image from the installed app | PASS | FLUX.2 Klein 4B 1024x640: `Diffusers local engine (cuda)`, **14.8 s** generation (65.3 s with the first load), peak 10.3 GB. |
| One video from the installed app | PASS | Wan 2.2 TI2V 5B 832x480, 49 frames, 20 steps: `Diffusers video engine (cuda)`, **42.3 s** generation (85.5 s total), peak 13.2 GB. |

The installed app was closed afterwards (backend down, GPU free for the TensorFold session). Not exercised against the installed app: the step watchdog's hard stop.
