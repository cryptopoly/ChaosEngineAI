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


## build.sh stages a clean release venv (2026-11-11)

`./build.sh` no longer installs into or stages the dev `.venv`. It creates `.venv-release` on demand (CPython 3.12 via `py -3.12` / `python3.12`, or `CHAOSENGINE_RELEASE_PYTHON`; git-ignored), installs with `-c release-constraints.txt` as `release.yml` does, and exports `CHAOSENGINE_EMBED_PYTHON_BIN` to it on all three platforms (the Mac and Linux package lists are unchanged apart from `babel`, which the workflow already installs). A leftover GPU-bundle venv is recreated when `CHAOSENGINE_BUNDLE_GPU` is not set; `CHAOSENGINE_BUNDLE_GPU=1` keeps its own warning and now notes that a Windows NSIS installer cannot hold it. `scripts/stage-runtime.mjs` fails right after archiving on Windows when the runtime archive is over 1.9 GB, naming a staged dev venv or the GPU bundle as the cause (`CHAOSENGINE_SKIP_ARCHIVE_SIZE_CHECK=1` skips it). Tests: `tests/test_stage_runtime_scripts.py` (+6, 18 pass). Checked with `bash -n` and the test only; the full Windows build was not rerun. An aborted dry run started a real stage step and was killed; it left no tracked changes.


## CUDA llama-server as an in-app lane on Windows (2026-11-11)

Built on `test/rtx4090`. The Linux-only installer in `helpers/llama_cuda.py` now covers Windows x64 and ARM64 too; Linux behaviour is unchanged.

- **Installer** (`POST /api/setup/install-llama-cuda`, background job, Python only, no bash, no `shell=True`): reads the driver's CUDA version from `nvidia-smi` (`nvidia_driver_cuda`), picks the matching official release asset of the pinned tag (`LLAMA_CPP_TAG` b11981): CUDA **13.4** build for a driver reporting CUDA 13 or newer, **12.4** build (x64) for 12.4 to 12.9, nothing below that. Downloads the build zip and the separate cudart zip, **checks each against its published sha256** (pinned table for b11981; for any other tag it asks the GitHub release for the digest; an archive with no published digest installs but is recorded `verified: false`), refuses zip members outside the folder, needs 3 GB free, unpacks both into one folder `~/.chaosengine/bin/llama-cuda/<tag>/` (DLLs beside `llama-server.exe`, so no `PATH` change), and writes `installed.json` (tag, OS, arch, CUDA, asset names, sha256 per archive, verified flags, driver, time). A mismatch installs nothing and leaves no staging folder. After the job the cached backend capabilities are refreshed, so the running backend picks the new server without a restart.
- **Status** (`GET /api/setup/llama-cuda-status`): new fields `state` (`unsupported`, `no-driver`, `driver-too-old`, `not-installed`, `installed-no-device`, `in-use`), `message`, `canInstall`, `driverTooOld`, `driverCuda`, `os`, `verified`. A too-old driver is `supported: true, canInstall: false` with a message naming the CUDA version it saw, and the install job fails with the same text before downloading anything. Nothing names a card: only the driver decides, and the build then has to list a CUDA device itself (`--list-devices`) before the app uses it, so a 12 GB card, a 5090 and a Spark take the same path.
- **Routing**: `binaries._resolve_llama_server` already preferred an installed CUDA server that lists a device; it now finds `llama-server.exe`. Not installed, no device or `CHAOSENGINE_LLAMA_CUDA=0` keeps the standard server (tests cover each).
- **Setup page**: the existing Accelerators panel shows on Windows too, with an explanation instead of a button when the driver is too old (en key `llamaCuda.driverTooOld`, other locales `null`).
- **Live, through the dev backend on 8876** (RTX 4090, driver 621.14, CUDA 13.5): install through the endpoint twice (second time to prove the live refresh; both sha256 verified, about 590 MB, roughly 1 min), status `in-use`, health `llamaServerPath` moved from `~/.chaosengine/bin/llama-server.exe` (CPU build, `--list-devices` empty) to `llama-cuda\b11981\llama-server.exe`. 300-token replies, default `--fit on` (no manual layer count), 3 runs each:

| Model (GGUF, ctx 4096) | CPU build | CUDA build | VRAM added |
|---|---|---|---|
| Qwen3-0.6B Q8_0 | 102 tok/s | **549 tok/s** (5.4x) | 1.5 GB |
| Nemotron 3.5 Bolt E4B Q4_K_M | 22.2 tok/s | **172 tok/s** (7.8x) | 3.7 GB (peak 3.7) |

  Unload returned the card to its 1.9 GB idle in both cases.
- **Tests**: `tests/test_llama_cuda_windows.py` (new, 31 incl. capability refresh) and `tests/test_llama_cuda.py` (Linux paths, updated) pass; `npx tsc --noEmit` clean; `npm run i18n:validate` OK. E2E: phase 1 check "CUDA llama-server lane (Windows and Linux)" (skips with the reason without an NVIDIA driver, with a too-old driver or when not installed; fails if installed but no device or unverified); it was not run as a full phase, only the status route it reads was exercised live.
- **Not verified / SKIP**: Linux install (unit tests with fake archives only, no Linux host here); the CUDA 12.4 and Windows ARM64 paths (asset names and digests checked against the release listing, no machine to run them); a too-old driver live (unit and route tests only); AMD ROCm: the release has a `win-rocm-10.0-x64` asset but no Linux ROCm one and nothing here to test it on, so it stays a documented SKIP. Reinstall while a GGUF model is loaded fails on Windows (locked DLLs) with a message to unload first. Two failures outside this change on this box: `test_setup_routes::test_cuda_torch_order_follows_the_driver_and_cpu` (reads the live driver, 13.5) and a vitest `VideoStudioTab` tooltip test.


## Qwen3-TTS on the NVIDIA GPU (2026-11-11, FU-148)

Built on `test/rtx4090`. The five Qwen3-TTS rows that were `lanes=['mlx']` now have a CUDA twin each (`qwen3-tts-0.6b-cuda`, `-1.7b-cuda`, `-custom-0.6b-cuda`, `-custom-1.7b-cuda`, `-design-1.7b-cuda`), upstream `Qwen/Qwen3-TTS-12Hz-*` bf16 weights pinned to full revisions (Hub API, no token; none gated, all Apache-2.0; only the 11 files the loader reads; 18.6 GB in total for all five, downloaded through the app's Models route at 2 to 8 MB/s). RTX 4090 24 GB, driver 621.14, voice venv torch 2.9.1+cu128, Windows 11, dev backend on 8876.

**What had to be built (the part that was not obvious).**
- `qwen-tts` 0.1.1 (Apache-2.0; its `sox` import is BSD-3-Clause, pure Python, both recorded in `THIRD_PARTY_NOTICES.md`) pins transformers 4.57.3. The voice venv has 5.2.0 because Chatterbox pins exactly that. Tried first on 5.2: four compatibility patches (the renamed `check_model_inputs`, the missing `pad_token_id` default, the missing `default` rope function, the `fix_mistral_regex` argument) get it to load and run, but it never stops generating and the result is hum ("Mmmmmm" in a Whisper transcript), because its prefill and rope bookkeeping rely on 4.x generation internals. transformers 5.22 (the app extras) was not tried live: the extras have no librosa, soundfile or onnxruntime, and the venv's torchaudio 2.9 does not load against their torch 2.14, so it would need a third environment.
- So Qwen3-TTS runs in its own **worker**: the same `voice_worker.py` and the same Python and PyTorch, with the packages `qwen-tts` needs (transformers 4.57.3, huggingface-hub 0.36.0, accelerate 1.12.0, qwen-tts, sox; 127 MB) installed `--no-deps --target` into `voice-venv/qwen-site` and put first on `PYTHONPATH` for that process only. The backend's own `PYTHONPATH` is never inherited: a first attempt leaked it and loaded tokenizers 0.23.3 from the app extras, found by the live run. `voice_client.speech_worker_for` picks the worker and unloads the other worker's model first, so two speech models never share the GPU (Kokoro stays); `stop_all`, `interrupt_all`, `worker_busy`, `loaded_sources` and the Unload panel cover both. Unticking the add-on deletes `qwen-site` (`voice_runtime_disk.remove_qwen_site`, path-checked). An add-on installed before this change tells the person to untick and tick it again.
- **GPU gate** (`backend_service/voice_gpu.py`, read from the hardware, never assumed): each row declares `minVramGb` (measured peak + 0.7 GB) and `minComputeCap` 7.0. Total memory comes from the existing GPU monitor (so a unified-memory board such as the DGX Spark is never blocked by a number that cannot be read), the compute capability from `nvidia-smi`. A row this GPU cannot run is dropped from the automatic picks when another runs, refused with the reason on Download and on Speak, and shown with the reason in the Models row (`unsupportedFacts`; en keys `voice.speak.gpuTooSmall` and `voice.speak.gpuTooOld`, other locales `null`). The worker loads bf16 where the card has it and fp16 below Ampere (the fp16 path is not run on any card here).
- Also fixed on the way: Apple-only Qwen rows no longer shadow their GPU twin on a PC (`_rows` keeps rows that run here, so Speak's default voice "Ryan" resolves to the GPU model instead of an undownloadable Mac one); `voice_clone.list_models` no longer lists a row of the other lane; the clone default is the same model on both lanes (`qwen3-tts-0.6b` and its `-cuda` twin).

**Live results through the app** (`POST /api/voice/speak`; a 12-word line cold and warm, then a 600-character text in 6 chunks; peak is the GPU memory the worker added over idle via `nvidia-smi` every 0.15 s, CUDA context included; the model stays resident, so idle equals peak; listen-check = rate, duration, `ffmpeg volumedetect`, and a Whisper Large v3 Turbo transcript of the warm clip):

| Row | Result | Cold / warm, short line | Per 10 s of speech (600 chars) | Peak GPU memory | Audio checks |
|---|---|---|---|---|---|
| CustomVoice 0.6B, Ryan | PASS | 17.3 s / 8.7 s for 5.04 s | 17.1 s (43.9 s of audio in 74.9 s) | 2.83 GB | 24 kHz, mean -19.9 dB, max -3.6 dB; transcript exact |
| CustomVoice 1.7B, Ryan neutral | PASS | 17.6 s / 8.3 s for 4.72 s | 17.2 s (40.0 s in 68.8 s) | 4.68 GB | 24 kHz, mean -22.5 dB, max -6.6 dB; exact |
| CustomVoice 1.7B, style "Sound cheerful and speak quickly with a big smile." | PASS | 15.7 s / 6.5 s for 3.68 s | 17.2 s | 4.57 GB | 24 kHz, mean -19.8 dB; the same line is 22 % shorter than neutral (4.72 s), so the style is followed; exact |
| VoiceDesign 1.7B, "A calm older man with a deep, slow, gravelly voice." | PASS | 16.9 s / 7.6 s for 4.32 s | 17.2 s | 4.58 GB | 24 kHz, mean -21.0 dB; exact |
| Base 0.6B clone (12 s synthetic reference made by CustomVoice Ryan, saved as a voice with the permission record, reference text auto-transcribed) | PASS with a flaw | 29.6 s / 9.7 s for 5.36 s | 19.1 s | 3.42 GB | 24 kHz, mean -24.9 dB; AI label and watermark applied; transcript "coffee. The quick brown fox ..." - the last word of the reference leaks into the start of the clip (same on every run with seed 7) |
| Base 1.7B clone (same reference) | PASS | 24.6 s / 8.9 s for 4.80 s | 24.1 s | 5.31 GB | 24 kHz, mean -23.2 dB; exact |

- **FU-148 answered: 0.6B is not faster on CUDA.** CustomVoice 0.6B 17.1 s and 1.7B 17.2 s per 10 s of speech, short line warm 8.7 s against 8.3 s. The only saving is memory (2.83 against 4.68 GB). Generation is launch-bound (the GPU shows about 40 % use during a run: Python and many small kernels per token), so it does not scale with the model.
- **Speed is the weak point**: about 1.7 s of compute per second of speech (0.58x real time) against 0.25 s on the M4 Max, so on this PC Qwen3-TTS is fine for Speak and slow for read-aloud and calls. Not optimised (CUDA graphs or `torch.compile` of the talker and the 15-step sub-talker are the obvious next steps; not tried).
- The reference was synthetic (generated by the model, no real person); a clip cloned from a person's own voice was not made.
- Switching between workers checked live: Turbo (6.1 GB used in total), then Qwen 0.6B CustomVoice (7.9 GB), then Turbo again (6.1 GB), all three answered; the Unload panel shows the model of whichever worker holds one.
- **E2E**: new check "speak with Qwen3-TTS on the GPU" in phase 8 of `scripts/e2e_test_suite.py` (CustomVoice 1.7B with a style and VoiceDesign, real WAV, 24 kHz, not silent; skips when no GPU row is downloaded and supported) PASS in 25 s; the clone-permission check now covers the GPU Base rows. Phase 8 overall, 18 checks: 16 PASS, 1 SKIP (Nemotron) and 1 FAIL, "speech model catalog: chatterbox-turbo-cuda is listed without measured numbers", which comes from the dev-only `CHAOSENGINE_VOICE_UNMEASURED=1` switch the backend was started with, not from the new rows.
- **Tests**: new `tests/test_voice_qwen_cuda.py` (install pins and folder, catalog twins and measured numbers, routing next to the Apple rows, the GPU gate for 3, 4, 8, 24 and 32 GB and compute capability 6.1 to 12.0, worker calls and loader, worker selection and unload, removal); `tests/test_voice_cuda_lane.py`, `test_voice_speak.py`, `test_voice_clone.py`, `test_voice_models_wave2.py` (these now pin the Apple lane for their Apple-catalog assertions), `test_voice_io.py`, `test_voice.py`, `test_voice_watermark.py`, `test_mcp_bridge.py` and `test_stage_runtime_scripts.py` pass, except `test_voice_speak.py::ChatStreamTests::test_a_cloned_voice_reads_aloud_marked_and_audited_but_is_refused_for_calls_and_agents` (`mark.call_count 1 != 2`), which fails the same way on the commit before this work. Vitest `src/features/voice` 144 passed (a new case for the GPU reasons), `npx tsc --noEmit` clean, `npm run i18n:validate` OK.
- **Setup string**: `llamaCuda.description` no longer says "about 680 MB" (the Windows download is about 590 MB); it now ends "The download includes the CUDA runtime." and every other locale is `null` again (the translated lines quoted 680 MB).
- **Not verified**: Linux and DGX Spark (the ARM64 add-on is not offered); cards other than the 4090 (the gate is unit-tested with mocked memory and compute capability only); the fp16 path of a pre-Ampere card; a fresh install of the add-on through the app's installer on a clean venv (the `--target` command and the import check were run by hand with the same arguments, and a unit test checks the command list); non-English CustomVoice speakers; the 25 Hz tokenizer models; a clone from a real person's sample; and the listening quality of style and design beyond duration, level and a transcript (the clips are in `%TEMP%\voice-checks\qwen-*.wav`, the reference is `synthetic-reference-ryan-12s.wav`).

## Two known failures resolved

- **`test_a_cloned_voice_reads_aloud_marked_and_audited...` (`mark.call_count 1 != 2`)**: a stale test, not a code bug. `voice_chat.speak_sentence` has passed `altered=` to `voice_clone.mark_cloned` since the "don't claim PerTh after speed or pitch changes" commit, but the test's mock was `lambda wav, r, vid`, so the first sentence raised a TypeError (counted as call 1), the stream fell back and the second sentence never reached the mark. The mock now takes `**kw`; the code still marks and audits every sentence of a cloned voice. `tests/test_voice_speak.py`: all pass.
- **E2E phase 8 "is listed without measured numbers"**: `GET /api/voice/models` now carries `unmeasuredListed` (true when `CHAOSENGINE_VOICE_UNMEASURED=1` is on). The check (`unmeasured_listed_rows` in `scripts/e2e_test_suite.py`) SKIPs with that reason and the row keys when the switch is on, and still FAILs when it is off and an unmeasured non-Qwen3-TTS row is listed. Covered by a unit test; phase 8 itself was not re-run (it needs the running backend and loads speech models on the GPU).


## Estimate audit: memory, size and time figures on CUDA (2026-11-11, FU-164, FU-165)

The Mac re-evaluation found many "will it fit / how much memory / how long" figures wrong. No Mac findings were on this branch, so every estimate a non-Apple host shows was inventoried and measured here, one model at a time, with `nvidia-smi` sampled every 0.25 to 0.5 s and the app's own figure computed from the same code. Card idle at 1.4 to 1.9 GB (desktop); "measured" below is the process's own share (peak minus idle). Commit `925f6f2f`.

### Inventory: what assumed a Mac

| Estimate | Where | Apple-only assumption off the Mac |
|---|---|---|
| Chat fit badge, `/api/models/memory-check` | `helpers/fit.py`, `routes/cache.py` | Weights + 10 % + KV in one pool. On CUDA llama.cpp keeps input embeddings in system RAM; the memory check compared against **system RAM**, not the card |
| KV cache size | `helpers/model_arch.py` | Ignored Gemma 4 shared-KV layers and narrow sliding heads (Nemotron 3.5 Bolt) |
| TurboQuant size and speed | `helpers/cache.py`, `cache_compression/turboquant.py` | MLX lane: keeps 2 x `fp16_layers` layers in f16, 0.13 to 0.18x decode. The llama.cpp lane compresses every layer and decodes at 0.85x |
| Decode tok/s | `helpers/cache.py` | 2 ms per-token overhead fitted on an M4 Max; the whole file read per token |
| Hardware class | `helpers/hardware_class.py` | A GPU that shares system memory (DGX Spark) got 100 % of RAM as "usable" |
| Image / video badge | `helpers/fit.py` | Whole pipeline on the card; the runtime offloads by model on CUDA |
| Studio peak (image, video) | `src/utils/images.ts`, `videos.ts` | Quadratic attention (score matrix held); CUDA SDPA never holds it |
| Time estimates | catalog `estimatedGenerationSeconds` | Flat per row, not scaled by size, frames or steps (Mac and PC alike) |
| Memory gates after Unload | `video_runtime`, `image_runtime` | (Windows only) freed RAM was not given back to the OS |

### Chat (llama-server b11981 CUDA and the TurboQuant fork, the app's launch flags, 256-token decode)

| Model, context, cache | Measured VRAM | App before | App after | Measured tok/s | Before | After |
|---|---|---|---|---|---|---|
| Qwen3-0.6B Q8_0, 4K f16 | 1.44 GB | 1.3 GB (-10 %) | 1.50 (+4 %) | 563 | 316 (-44 %) | 466 (-17 %) |
| Qwen3-0.6B, 32K f16 | 4.53 GB | 4.3 (-7 %) | 4.56 (+1 %) | 566 | 316 | 466 |
| Qwen3-0.6B, 32K turbo3 | 1.95 GB; KV 700 MiB | KV 1.37 GB (+96 %) | KV 0.68 GB (-3 %); total +6 % | 467 | **56 (-88 %)** | 396 (-15 %) |
| Qwen3-0.6B, 32K turbo4 | KV 952 MiB | KV 1.53 GB (+61 %) | 952 MiB (0 %) | 462 | 56 | 396 |
| Bolt E4B Q4_K_M, 4K f16 | 3.57 GB | **6.5 GB (+82 %)** | 3.73 (+5 %) | 174 | 112 (-36 %) | 182 (+5 %) |
| Bolt E4B, 64K f16 | 4.56 GB; KV 1040 MiB | 8.1 GB; KV 1.82 GB (+75 %) | 4.67 (+2 %); KV 1044 MiB | 174 | 112 | 182 |
| Bolt E4B, 64K turbo3 | 4.49 GB | 7.0 GB | 4.4 (-2 %) | 152 | 20 | 155 (+2 %) |

Why: llama.cpp keeps `token_embd` and Bolt's 3.5 GB `per_layer_token_embd` in system RAM and puts a copy of a tied output head on the card; the new `helpers/gguf_layout.py` reads that split from the GGUF tensor table (it matched llama.cpp's buffers to the MiB: 3023 / 3536 MiB and 604 / 158 MiB). Bolt caches only 4 global + 20 sliding layers of 42 (18 reuse an earlier layer's cache), sliding heads 256 wide. The TurboQuant fork holds one more compute buffer (+0.27 to +0.55 GB), now counted. TurboQuant on this lane saves real memory only on KV-heavy models: Bolt's turbo3 used **more** VRAM than f16 up to 32K and saved 0.08 GB at 64K, where the app promised 1.1 GB. Live check after the fix: `/api/cache/preview` for Qwen3-0.6B turbo3 at 32K reads 0.68 GB and 466 tok/s.

**The memory check judged the wrong memory:** on a 24 GB card with 50 GB of free system RAM, a 33 GB model read "fits"; llama.cpp then runs the layers that do not fit on the CPU (section 6: 21.6 tok/s on CPU against 170 on the card). It now judges the card's free memory plus what loaded chat models hold, says "GPU memory" and names the CPU slowdown, and separately checks the embeddings against system RAM.

### Image and video (dev backend, warm unless noted)

| Model and size | Measured peak | Badge before | Badge after | Studio before | Studio after |
|---|---|---|---|---|---|
| FLUX.2 Klein 4B 1024², 4 steps | 9.1 GB | 20.8 GB, "tight" (+129 %) | 11.4, fits (+25 %) | 9.2 | 8.7 (-4 %) |
| Klein 1536² | 10.0 GB | | | 17.3 (+73 %) | 11.2 (+12 %) |
| Klein 2048² | 15.6 GB | | | **39.2, "danger"** (+151 %) | 15.1 (-3 %) |
| Klein 1024² fp8 layerwise | 8.4 GB | | | | |
| Wan 2.2 TI2V 5B 832x480x49, 20 steps | 11.0 GB | 22 GB, "tight" | 17.6, tight | 22, caution | 13.5 (+23 %) |
| Wan TI2V 832x480x121 | 12.9 GB | | | 33, **danger** | 15.5 (+20 %) |
| Wan TI2V 1280x704x49 | 12.7 GB | | | 31, **danger** | 15.3 (+20 %) |
| Wan TI2V 1280x704x121 (its recommended size) | 17.3 GB | | | **119, danger** (+585 %) | 19.9 (+15 %) |
| LTX-2.3 4-bit 768x512x97 (cold / warm) | 21.7 / 16.9 GB | 22.5 measured, tight | unchanged | 22.5 | unchanged |
| LTX-2.3 768x512x49 / 1024x576x97 | 16.0 / 17.9 GB | | | 22.5 | unchanged |

The Studio's danger state blocks Generate until ticked, so Wan TI2V at its own recommended settings needed an override on a card where it uses 72 %. Badge: diffusers rows the runtime offloads by model on a discrete card (all video, the FLUX family) are judged at the offloaded share (image 0.55, video 0.8 of the full pipeline); rows measured on CUDA keep their measurement; Macs and shared-memory GPUs are unchanged.

**Time (catalog `estimatedGenerationSeconds`, shown flat in the Studio):**

| Run | Measured | Catalog | Verdict |
|---|---|---|---|
| Klein 1024², 4 steps, warm (cold) | 6.3 s (49.7 s, 33 s of it loading) | 5.5 s | correct warm (-13 %); load not in the estimate |
| Klein 2048² | 11.2 to 13.1 s | 5.5 s | wrong by -51 % |
| Wan TI2V 832x480x49, 20 steps | 34.5 s | 275 s | **wrong by 8x** (the catalog figure is for 1280x704x121) |
| Wan TI2V 832x480x49 10 steps / 832x480x121 / 1280x704x49 | 22.8 / 88.0 / 81.7 s | 275 s | wrong by 3 to 12x |
| Wan TI2V 1280x704x121 | 273 s (W1) | 275 s | correct at the reference |
| LTX-2.3 768x512x97 / x49 / 1024x576x97 | 59.5 / 36.3 / 86.7 s | 70 s | -15 % / +93 % / -19 % |

Not fixed (FU-165: needs a reference size and step count per row, and touches the Mac): a fit that works for both measured video models is `seconds = catalog x (0.04 + 0.96 x steps/ref_steps x w(T)/w(T_ref))` with `w(T) = T + T²/53,300` and `T = W/16 x H/16 x F/4`. It predicts Wan TI2V 33 / 22 / 84 / 78 s for the four runs above (within 5 %) and LTX at 49 frames 15 % low.

### Memory not given back after Unload (Windows)

After Klein, then Wan TI2V, and two Unloads, the dev backend still held **29.5 GB of system RAM (35 GB committed)** with nothing loaded, and the gates then refused FLUX.1 schnell ("needs about 26 GB, only 25.1 GB available") and LTX-2.3 ("only 23.3 GB"). Cause: PyTorch's Windows CPU allocator (mimalloc) keeps freed memory until its next allocation, and nothing allocates after an Unload (no Python object was left: no module or large tensor alive, `remove_all_hooks` and heap compaction changed nothing). `backend_service/__init__.py` now sets `MIMALLOC_PURGE_DELAY=0` and `MIMALLOC_RESET_DELAY=0` before torch loads (Windows only; a user's own values win). Measured: Klein unload 16.6 → **1.8 GB**, Wan unload 26.5 → **2.0 GB**; Klein warm 6.57 s with and without the setting (A/B, 5 runs each). Linux uses glibc, not mimalloc: not checked.

### Other changes

- DGX Spark and other GPUs that share system memory: usable memory is total minus max(8 GB, 10 %) (115 GB of 128) instead of 100 %. Not measured: **owner test**.
- `llama-server-turbo.exe` in `~/.chaosengine/bin` is now found by a backend run from source on Windows (the lookup tried only the name without `.exe`; the packaged shell passes an env var, so installed apps were not affected).
- Tests: `tests/test_chat_memory_cuda.py` (22: GGUF split, shared KV, TurboQuant lane, decode speed, VRAM table 4 to 96 GB and a 128 GB shared-memory machine, shared-memory class table 32/64/128, `.exe` lookup, mimalloc env), `tests/test_media_fit_cuda.py` (5: Klein, Wan TI2V and LTX tables at 8 to 96 GB, rows kept whole, Mac and Spark unchanged), 3 memory-check route tests in `test_model_arch.py`, vitest image (3 new) and video (4 updated or new). E2E: phase 4 "CUDA estimates judge the card's own memory (FU-164)" PASS here (it skips off a discrete card).
- Mac paths: unchanged except shared bugs: the shared-KV / sliding-head KV reading (Gemma 4 GGUFs and `num_kv_shared_layers` in config.json on any machine) and the TurboQuant llama.cpp-lane size (a GGUF on a Mac also runs the `turboN` types on every layer). The Mac's TurboQuant speed ratio for GGUF is unchanged (unmeasured there). The Mac-oriented tests (`test_speed_estimate`, `test_model_fit`, `test_model_arch`, `test_recommendations`, `test_memory_gate`) pass; the memory-check tests now pin the host instead of reading this PC.

### Not verified (owner tests)

| What | Why |
|---|---|
| Qwen3.8-27B Q4_K_M + DFlash sidecar | No longer in the HF cache (a 20 GB download, over this pass's limit) |
| Cards of 8, 12, 16 and 32 GB | Badges and Studio figures come from formulas fitted on 24 GB; Klein on 12 GB still reads "won't fit" (offloaded estimate 11.4 GB against a measured 9.1) |
| RTX 5090, DGX Spark | Spark's usable share; FLUX placement (the image runtime always offloads FLUX on CUDA, even where the whole 16 GB pipeline fits: resident Klein ran 1.42 s against 6.6 s offloaded at 1024², but spilled past 24 GB at 2048²) |
| Linux CUDA | glibc allocator behaviour after Unload; the CUDA decode overhead constant |
| Wan 2.1 1.3B at 720p | Filled the card before the offload rule existed (section 4); not re-run |
| Studio time scaling | FU-165 |

Pre-existing failures seen while testing, unrelated: vitest "keeps the device and capacity facts as the memory item's tooltip" (host card), pytest `test_video_routes::test_catalog_slow_on_labels_reach_the_payload` (Mac row), E2E phase 4 "new image families" (Mac-only mflux rows are filtered on Windows).

## llama.cpp b12292 CUDA lane on the 4090 (2026-11-12)

Checks the Mac session's move from b11981 to b12292 (`153751bf`; CUDA 13 archives renamed 13.4 to 13.5). RTX 4090, driver 621.14 (CUDA 13.5), dev backend on 8876.

| Check | Result | Detail |
|---|---|---|
| `test_llama_cuda.py`, `test_llama_cuda_windows.py`, `test_linux_arm64_build.py` | PASS | all pass, 5 skipped (Linux-only) |
| Pinned sha256 against the GitHub release API (`b12292`) | PASS | all 6 Windows CUDA digests in `_PINNED_SHA256` (12.4 x64, 13.5 x64, 13.5 arm64; build and cudart) equal the digests the release lists |
| `POST /api/setup/install-llama-cuda` | PASS | picked CUDA 13.5, both zips (178 MB + 426 MB) downloaded and verified (`installed.json`: `verified: true` for both, sha256 as pinned), about 3 min |
| `GET /api/setup/llama-cuda-status` | PASS | before: `not-installed` for tag b12292 (the b11981 folder is ignored, status is per pinned tag); after: `installed`, `verified`, then `in-use` after a load; health `llamaServerPath` = `llama-cuda\b12292\llama-server.exe`; `--list-devices` shows `CUDA0: NVIDIA GeForce RTX 4090 (24563 MiB)`; `--version` build 12292 |
| Old `llama-cuda\b11981` folder | left by the installer, not pruned | ignored (never used once b12292 is installed). Deleted by hand after b12292 worked (our own install); an app update leaves one stale 600 MB folder per tag. Not fixed |

**Decode, 300 tokens, greedy, 3 runs, default `--fit on`, through `/api/chat/generate`.** On b12292 the first request of a benchmark process read low on 3 of 3 Qwen3 passes and 2 of 3 Bolt ones (Qwen3 342 to 363 against 541, Bolt 135 against 160) even with the server already warm; the b11981 passes showed no such dip. Cause not found (idle GPU clocks after a pause between processes is the likely one, not proven); the table gives runs 2 and 3. Same-day A/B with the b11981 server (`CHAOSENGINE_LLAMA_CUDA=0` plus `CHAOSENGINE_LLAMA_SERVER` to force it, because an installed CUDA server wins over the env override by design):

| Model | b11981 today | b12292 | b11981 in the doc (11 Nov) | vs doc | vs same day |
|---|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | 493 to 510 | **541, 543** | 549 | -1.5 % | +7 % |
| Nemotron 3.5 Bolt E4B Q4_K_M | 153 to 157 | **160, 162** | 172 | **-6.6 %** | +3 % |
| Bolt + DFlash drafter (BF16) | 171, 173 | **180.5 to 183.9** (second and third) | 332 (400 tokens, prose, other prompt) | not comparable | +7 % |

Verdict: no regression from b12292. The 6.6 % gap to the doc's Bolt 172 is run-to-run and day-to-day drift: the old build measured 155 on the same card today. The DFlash figure uses a different prompt (a long essay on the printing press at temperature 0, 300 tokens) from the doc's 332, so only the same-day A/B counts: b12292 is +7 %. Drafter engaged ("Speculative decoding active (draft-dflash ...)").

| Check | Result | Detail |
|---|---|---|
| E2E phase 1 `CUDA llama-server lane (Windows and Linux)` | PASS | 0.4 s; also pass: GGUF llama.cpp, GGUF speculative lane, GGUF Nemotron 3.5 Bolt + pinned DFlash drafter (4.9 s), TensorFold 1.x native CUDA |
| E2E phase 1 as a whole | FAIL (not llama.cpp) | `unusable settings are corrected with a note`: the check loads an "MLX" model with `--backend mlx`; on this PC that routes to TensorFold native, which answers HTTP 500 "Structured output (a JSON schema) needs TensorFold's grammar support" to the check's `jsonSchema`. A Mac-shaped check on a CUDA host; not touched here |
| `cache-strategy-matrix.py --quick` | PASS | only the native GGUF smoke cell runs in quick (Qwen3-0.6B Q8_0, b12292 server): 522.5 tok/s; the other 15 cells skip ("deferred to full run") |
| Full matrix (extra) | PASS | 2 pass, 14 skip: native GGUF smoke 223 tok/s (b12292; 24-token reply, startup dominated), `turboquant GGUF 3-bit` 445 tok/s on `llama-server-turbo.exe`, which is its own pin: build 9079 (`69d8e4be4`), not b12292 |

Left: backend stopped, no llama-server running, card back to the desktop's idle 1.2 GB.

## E2E settings check skips off MLX; CUDA llama.cpp installer prunes old tags

- `unusable settings are corrected with a note` (E2E phase 1) now skips with a reason when the load landed on an engine other than `mlx` (on a CUDA host the MLX request routes to TensorFold native). The decision is `_non_mlx_engine_skip` in `scripts/e2e_test_suite.py`, tested in `tests/test_e2e_engine_skip.py`.
- Route: the TensorFold "needs grammar support" error stays an error on purpose (the engine docstring: dropping the schema would hand a JSON-parsing caller free-form text, and the guard already notes-and-drops only where the engine is known not to enforce, mlx/mtplx). It is now a `RequestNotUsable`, so the chat and OpenAI routes answer 400 with the in-app fix instead of 500. Asserted in `tests/test_tensorfold_engine_integration.py`.
- `llama_cuda.install` now calls `prune_old_installs` after the pinned tag's `installed.json` and binary verify. It removes only direct child folders whose `installed.json` names a different tag; skips folders with no marker, dot-folders, files, symlinks, folders with a running process (psutil) and anything that fails to delete (locked DLLs: logged, never fails the install). Tests: `tests/test_llama_cuda_prune.py`.
- Not applied to `llama-server-turbo` (installs flat into `~/.chaosengine/bin` and overwrites in place) or TensorFold native (one install root and one record): neither leaves per-version folders.


## Linux pass of the backend on WSL2 with the RTX 4090 (2026-11-12)

Ubuntu 24.04 (WSL2, Python 3.12.3, 32 cores, 31 GB RAM, the Windows driver 621.14 passed through) with a clone of `test/rtx4090` at `~/ChaosEngineAI`, a plain venv (pip from get-pip: the image has no `ensurepip`), the `release.yml` base list plus the image packages installed with `-c release-constraints.txt`, and CUDA torch from the app's own `cuda_torch_indexes()` order (cu134 first, which serves `2.11.0+cu130`; `torch.cuda.is_available()` True, sm_89). Models were read from the Windows HF cache over 9p (`HF_HOME=/mnt/c/...`, nothing copied; load is slower, steady state is not). No root: system packages came from `apt-get download` and `dpkg -x` into `~/syslibs` (libgomp1, the python3.12-dev headers, a relocated gcc 13), none installed system-wide.

**WSL2 is not bare Linux.** Host-to-device copies measured 10.5 GB/s pageable and 12.5 GB/s pinned (PCIe 4 x16 gives about 24), 156 TFLOPs bf16 matmul, 10 us per tiny kernel launch. Anything that offloads weights to system RAM is slower here than on Windows native (Wan TI2V 87 s against 35 s, Klein 10.0 s against 6.3 s, warm, same settings). Compare speed-ups inside this section, not with the Windows numbers.

### 1. Python test suite (`pytest tests/`, whole suite)

3740 passed, 14 failed, 71 skipped on the first run. After the fixes below: 11 failed.

| Failure | Verdict |
|---|---|
| `test_agent_files::test_bad_uploads_are_refused` (200 != 413) | **Linux-only, fixed (test).** A solid 32x24 PNG compresses to under the test's 100-byte cap with this zlib. The cap is 20 bytes now. |
| `test_storyboard_store::test_a_real_97_frame_clip_measures_4_0417_seconds` (3.95) and `test_storyboard_v2::OtioTests::test_document_shape_crossfade_and_lanes` ([27.84, ...] against [30, ...]) | **Linux-only, real bug, fixed.** `probe_seconds` read the `frame=` counter of `ffmpeg -c copy -f null -`. The static Linux ffmpeg 7.0.2 (imageio-ffmpeg) prints none for a stream copy and reports a time that ignores the last frames, so every clip measured about 0.09 s short and the film length, crossfades and OTIO export were off. It now decodes to count when no counter is printed. |
| `test_setup_routes` x5 (CUDA index order, `--target`) | Same as Windows: they call the real `nvidia-smi` (here `/usr/lib/wsl/lib/nvidia-smi`, driver 621, CUDA 13.5, cu134 first). Host-bound, not fixed. |
| `test_size_limits` x2, `test_video_routes::test_catalog_slow_on_labels...` | Same as Windows (the memory gate reads the card; a Mac catalog row). |
| `test_music_sound_1_0` (soundfile), `test_voice_watermark` x2 (librosa) | Same as Windows: the optional audio packages live in the voice venv, not in this test venv. |

### 2. Backend and E2E (port 8876, backend started from the clone)

The backend starts and serves on Linux. The CLI-driven E2E suite needs `CHAOSENGINE_REQUIRE_AUTH=0` (otherwise every call is 401). `--smoke` after the fixes: phases 0, 3, 6, 7, 8 PASS (phase 8 has 18 checks; the speech checks SKIP with their reasons because the voice runtime is not installed here). Phase 4 fails only on "new image families" (the Mac-only mflux rows are not in a Linux catalog; same as Windows). Phase 5: LTX-2.5 on ltx-2-mlx fails (Mac row, same as Windows); the real LTX-2.3 4-bit text-to-video check PASSES (after the SageAttention fix below); the checks after it SKIP with "host memory gate fired" because a 31 GB VM cannot hold LTX-2.3 and Wan together. Not run: the full suite (phases 1, 2, 9, 10 need the large chat models).

### 3. CUDA llama.cpp installer, live (`POST /api/setup/install-llama-cuda`)

| Item | Result |
|---|---|
| Asset | `llama-b12292-bin-ubuntu-cuda-13.5-x64.tar.gz` plus the `cudart-` archive; sha256 verified against the release digests (both "Verified"); 77.5 s end to end. |
| Install check | **FAIL, then fixed.** "Installed, but it found no CUDA device." The server could not start at all: this image has no `libgomp.so.1` (`error while loading shared libraries`), and the message sent the user to the driver. `llama_cuda.missing_system_library` now reads the loader error and says "this system lacks libgomp.so.1. On Ubuntu or Debian: sudo apt install libgomp1" in the install job and in the status (new `missingLibrary` field). The installer cannot apt-install without root, so the fix is the message. |
| Chat (Qwen3-0.6B Q8_0 from the Windows cache, ctx 4096, f16 KV, 300 tokens) | PASS once `libgomp1` was available: **322 / 489 / 480 tok/s** (the first run includes warm-up), +1.5 GB VRAM. Windows native b11981 measured 549 to 563 tok/s, so about 12 % lower under WSL2. |

### 4. TensorFold 1.2.7 native, Linux archive through the app (`POST /api/setup/install-tensorfold-native`)

- Gate before install: `unqualified` (sm_89 unconfirmed until the installed binary reports a CUDA chip). Install 9.2 s (archive and `.sha256` checked by the installer). After install the binary's own `capabilities --json` reports **`nvidia-sm89`, back end cuda, grammar true**: the sm_80/86 images in the release set do run on the 4090, no image-load failure.
- Serving Qwen3.5-2B (`mlx-community/Qwen3.5-2B-MLX-4bit` from the Windows cache) through `/api/models/load` with `backend: tensorfold`: load 3.3 s, engine label "TensorFold 1.x (CUDA, exact speculative decoding)", 3 x 300 tokens at **450.0 / 454.0 / 453.8 tok/s**, +2.1 GB VRAM. The sm_89 source build on Windows gave 460 tok/s, so the shipped image costs about 1.5 %.

### 5. Linux-only speed options

Wan 2.2 TI2V 5B 832x480, 49 frames, 20 steps; FLUX.2 Klein 4B 1024^2, 4 steps; warm = the second run in the same load.

| Option | Result | Numbers |
|---|---|---|
| SageAttention 2.2.0 (thu-ml master `d1a57a5`) | **PASS: 1.49x on Wan, no gain on Klein.** No PyPI wheel (PyPI stops at 1.0.6) and no Linux release wheel, so it was built for sm_89 only: 6 min with a relocated gcc 13 and `nvidia-cuda-nvcc==13.0.*` (plus the matching nvvm, crt, cccl) from pip; a 15.7 MB wheel. | Wan warm 87.2 s -> **58.7 s**, peak 12.9 -> 12.7 GB. Klein warm 10.0 to 10.5 s -> 9.9 to 10.5 s (offload-bound). Image quality not compared (no PSNR run). |
| `torch.compile` (`CHAOSENGINE_SPEEDUPS=compile`) | **PASS on Wan, 1.67x; no gain on Klein.** Needs a C compiler and Python.h on Linux; this image had neither (a stand-in compiler from `ziglang` proved the path first). | Wan cold 117.6 s -> 123.7 s, warm 87.2 s -> **52.3 s**, peak 12.9 -> 12.7 GB. Klein cold 63.8 s -> 72.1 s, warm 10.0 to 10.5 s -> 9.8 to 10.3 s. |
| flash-attn | **SKIP.** The newest prebuilt wheels are cu12 / torch 2.8 (v2.8.3.post2); none for torch 2.11 with cu13, and the app has no flash backend switch (only `sage` and SDPA), so a source build would have nothing to call it. | |
| vLLM and TriAttention | **SKIP (stopped at the disk floor).** `vllm` 0.33.0 pins torch 2.14 and the cu13 toolkit, so it needs its own venv (the test venv pins 2.11). That venv reached 6.0 GB and C: fell from 23.4 to 14.4 GB free, under the 15 GB stop line, so the install was killed and the venv deleted. Run the vLLM lane and TriAttention after the WSL disk is compacted (see the footprint). | |

### Linux-only bugs found and fixed here

1. `probe_seconds` was short by the last frames with the static Linux ffmpeg 7 (above).
2. A missing `libgomp.so.1` read as "no CUDA device" for the CUDA llama-server (above).
3. **Installing SageAttention 2 broke every LTX-2.3 clip**: `ValueError: attn_mask is not supported for sage attention`, raised in the text connectors, so the app's blanket `set_attention_backend("sage")` failed a model that passes a mask. The pipeline call is now guarded: on that error the transformer goes back to PyTorch attention and the call repeats (the call signature is preserved for the kwarg filter; five tests). Verified live: the LTX-2.3 4-bit E2E check passes with the package installed (it failed three times before). The note still reads "Attention: SageAttention" after a fall-back.
4. **System RAM not given back after Unload on glibc**: 12.0 GB stayed with the backend after Klein then Wan (nothing loaded), and the gates then refused the next model. `malloc_trim(0)` after the image and video pipelines are released (`helpers/malloc_release.py`) leaves 0.4 to 1.9 GB.
5. The `torch.compile` knob had no check for a C compiler or Python.h on Linux (the first step failed with a Triton error) and was read only by the video runtime. `compile_blocker()` now names the missing piece (`build-essential` or `python3-dev`) and the image runtime reads the knob too.
6. Test only: the 413 upload cap (above).

### Observed, not fixed

- A warm Wan pipeline (about 25 GB of system RAM on this 31 GB VM) makes the next Wan run fail the gate ("Only 2.6 GB of RAM available ... needs at least 6.0 GB free") although the pipeline is already resident: the gate counts the warm model against itself. The warm timings above needed the floor lowered in the clone (not committed). Same gate on any machine with little spare RAM.
- `bitsandbytes` and `ftfy` are not in the base venv; the Wan image-to-video and LTX 4-bit routes fail without them until the GPU bundle installs them. Not run here: the bundle installer itself (it would have overlaid a second torch, over the disk budget).
- Not run: ACE-Step and Stable Audio (music stays Apple-only), the voice runtime, Nunchaku, DDTree, MTPLX.

### WSL footprint

The distro's virtual disk is 23.1 GB (the guest holds about 14 GB after cleaning: venv 5.5 GB, `~/.chaosengine` 1.1 GB, scratch gcc and headers 0.3 GB, build caches). C: went from 42.4 GB free to 12.1 GB free, below the 15 GB stop line, so vLLM was not retried. The virtual disk does not shrink by itself: `wsl --shutdown`, then compacting `ext4.vhdx` (admin `Optimize-VHD` or `diskpart` "compact vdisk") gives back about 9 GB. Not done here (host-level change).

## TensorFold CUDA registry rows; image input on the native engine

| Item | Result | Notes |
|---|---|---|
| A. `NATIVE_MODELS` rows (FU-168) | PASS | Added Qwen3.6-35B-A3B + MTP (`qwen3_5_moe`, 20.9 GB, ~23 GB estimate) and Gemma 4 31B (`gemma4`, 18.44 GB, vision, ~21.5 GB estimate). Both repos are public (not gated) and listed by the 1.2.6 binary's `capabilities --json` (`qwen3_5_moe` and `gemma4`, mlx-q4g64); both `measured=False`, tight on 24 GB, upstream receipts are GB10. LFM2.5 already had its row. Gemma 4 26B-A4B stays Python-only (1.x refuses it), and the Gemma drafter is not wired on CUDA. Not loaded (over the size budget). |
| B. Image input, Qwen3.5-2B 4-bit (3 GB) via the backend | FAIL, then fixed, PASS | Before: the row had `vision=False`, `visionEnabled: false`, and the model answered "there is no image provided" (the app drops images). Binary check: `serve --vision` on the same checkpoint answered "Red" for a red square, so the checkpoint ships its tower. Fix: `vision=True` on the `qwen3_5_2b` row, with a test. After: `visionEnabled: true`, reply "red" through `/api/chat/generate` on `tensorfold-native`. |
| Registry/routing/setup/engine/native tests | PASS | 5 files, no failures (skips are macOS-only) |

Left: dev backend on 8876 and the side `serve` on 8899 stopped, model unloaded.


## Installer rebuilt with everything since 18cb80e6, installed, section 7 smoke (2026-11-12)

Installed over the existing 0.10.0 (silent `/S`, exit 0) with your OK. The installed backend ran on port 8890 (a throwaway `%APPDATA%\.chaosengine\settings.json` with `preferredServerPort`, removed afterwards) because the WSL pass holds 8876 and the app attaches to whatever answers there.

| Item | Result | Evidence |
|---|---|---|
| `./build.sh` | **FAIL, fixed** | `stage-runtime.mjs` crashed right after archiving: `ReferenceError: Cannot access 'NSIS_ARCHIVE_LIMIT_BYTES' before initialization` (the const sat below the top-level `main()` call). Moved above it; regression test `test_archive_limit_constant_is_declared_before_main_runs` (fails without the fix). Second run PASS, 3 m 42 s for the failed run, about 4 min for the good one. |
| Installer / archive | PASS | `ChaosEngineAI_0.10.0_x64-setup.exe` **131.8 MiB (138.2 MB)**, runtime archive **133.5 MB**; manifest `llamaServer: bin/llama-server.exe` (Vulkan b12292), `llamaServerTurbo` set, `warnings: []`; the four `install-*.sh` are in the archive. |
| Backend starts | PASS | `/api/health` 200 after 18 s (10 s on later starts). |
| Torch is the CUDA build | PASS | `torch 2.14.1+cu130`, CUDA 13.0, available, RTX 4090 (existing overlay kept; install from scratch not re-run). |
| `GET /api/setup/llama-cuda-status` | PASS | `supported:true, state:in-use, tag b12292, installedCuda 13.5, verified:true`. |
| `GET /api/setup/tensorfold-native-status` | PASS | installed 1.2.6 (the earlier manual adoption, pin is 1.2.7), gate `ready` on nvidia-sm89; install status route answers `idle`. |
| One image | PASS with a crash | FLUX.2 Klein 4B 1024x640, 4 steps: cold 54 to 81 s, warm **6.6 s** (`Diffusers local engine (cuda)`). One earlier run crashed the backend (below). |
| One video | **FAIL** | Wan 2.2 TI2V 5B 832x480, 49 frames, 20 steps: backend process dies right after "Loading pipeline components 100%" (52 s and 65 s, twice, on a fresh backend, GPU idle at start). Passed on the 11-11 installed build (42.3 s). |

Crash signature (Application log, same each time): `python.exe` faulting module `torch\lib\c10.dll`, exception `0xc0000005`, offset `0x91444`; the first one coincided with a `dwm.exe` crash (`udwm.dll`), so a driver reset is possible. The WSL pass was using the GPU at times (6 to 15 GB), which may be involved; the video runs started with about 1 GB used. Cause not found, not fixed. Next: run the same video request against the dev backend (`.venv`) to tell the new build from the machine.

## Wan 2.2 TI2V 5B crash on the installed build: commit exhaustion, fixed (2026-11-12)

Reproduced on the dev backend (port 8891), same request (832x480, 49 frames, 20 steps): the process died 53 to 70 s in, three runs out of three, with the same `c10.dll` `0xc0000005`. `faulthandler` put it in model CPU offload moving the text encoder to the GPU (`accelerate` `init_hook` -> `module.to`).

| Check | Result |
|---|---|
| `MIMALLOC_PURGE_DELAY=0` (925f6f2f) as the cause | **No.** With the mimalloc defaults preset (`PURGE_DELAY=10`, `RESET_DELAY=100`; `__init__` only uses `setdefault`) it crashed the same way. Kept. |
| WSL-pass commits (sage guard, `malloc_trim`, compile check) | Linux-only or inert here (the sage guard only runs when SageAttention is installed). Video and image runtimes are otherwise unchanged since the 11-11 build, and so are the overlay packages. |
| What ran out | System **commit**, not RAM or VRAM. Backend private bytes reached **59.3 GB**; the page file grew from 4 to 15.7 GB and stopped (C: has about 12 GB free); idle system commit is about 20 GB (WSL VM and the rest). `dwm.exe` died at the same moment with `0xc00001ad` (fatal memory exhaustion). |
| Why so much | Two Windows facts, measured with a probe: CUDA allocations count against the process's commit under WDDM (a 10 GiB CUDA tensor adds 10.5 GB private), and diffusers' memory-mapped safetensors load stays charged while the model lives. Wan's transformer alone (fp32 on disk, loaded as bf16, 9.3 GB of tensors) held **34.3 GB** private with the map and **16.0 GB** with `disable_mmap=True`; the whole pipeline **46.8 GB** vs **28.5 GB**. |
| Why it passed on 11-11 | Same code and packages; the disk had room then, so the page file could grow past the peak. |

Fix: `helpers/malloc_release.pipeline_load_kwargs()` returns `{"disable_mmap": True}` on Windows; the image and video runtimes pass it to the pipeline `from_pretrained`. Tests in `tests/test_malloc_release.py` (+3). The `-k "video or image or diffusion"` run has the same 7 Windows-only failures with and without the change.

Verified on the dev backend: Wan 2.2 TI2V 5B passed three times in a row (115 s cold, then **39.5 s** generation twice), peak private **43.9 GB** (was 59.3), and after Unload the backend held **4.1 GB** private (1.8 GB RSS), so the purge still returns the memory. Not verified: the installed app, which needs a rebuilt installer. Free disk on C: is still the real margin: with about 12 GB free the page file cannot grow much, so a bigger model can still run out of commit.

## Disk cleanup (2026-11-12)

C: free 13.6 GB before, 105.9 GB after (about 92 GB freed). Cache at `~/.cache/huggingface/hub`, removed with `huggingface_hub` `delete_revisions`; nothing was running (no backend ports, no python).

Deleted (GB): Wan 2.2 TI2V 5B Diffusers 34.2, Wan 2.1 T2V 1.3B Diffusers 28.9, TensorFold/Qwen3.8-27B-MLX-4bit 16.1, nunchaku-flux.1-schnell 6.8 (Nunchaku does not load on this torch), z-lab/Qwen3.8-27B-DFlash2 3.9, QuantStack/Wan2.2-TI2V-5B-GGUF 3.4; empty Wan2.1-T2V-1.3B stub dir; `~/.chaosengine/test-llama-b11981` 1.4; stale `Temp\pip-install-*` about 4.

E2E impact: the Wan 2.2 and Wan 2.1 checks already skip or pass without the model (`_wan_image_to_video`, cached live run "model is not installed", `_wan_not_burnt` "not downloaded"); `_qwen38_dflash2` skips without a GGUF. The 27B MLX + DFlash2 run (section above) is recorded here and no E2E check needs the weights.

Kept: FLUX.2-klein-4B 16, LTX-2.3 nf4 30.9 (live E2E row), Nemotron Bolt 6.2 + its DFlash drafter 0.7, Qwen3.5-2B MLX-4bit 1.75 (TensorFold), Qwen3-0.6B GGUF 2.35 (cache matrix smoke), voice models (Chatterbox, Chatterbox-turbo, VoxCPM2, Qwen3-TTS 0.6B/1.7B, faster-whisper, audioseal, VAD), Qwen-Image-2.1 GGUF stub, small test repos.

Re-download: the app's model Hub, or `huggingface-cli download <repo>` (Wan-AI/Wan2.2-TI2V-5B-Diffusers, TensorFold/Qwen3.8-27B-MLX-4bit with z-lab/Qwen3.8-27B-DFlash2, nunchaku-ai/nunchaku-flux.1-schnell).

## Warm-pipeline memory gate and GPU bundle packages

- Video/image gates counted the resident pipeline as used memory against the next request (Linux run: 25 GB held, "Only 2.6 GB of RAM available"). New `memory_gate.adjust_for_resident`: same model reuses the pipeline (no load term, floor and pressure ceiling drop to the chat gate's); another model unloads first, so the smaller of this process's RSS and the loaded model's catalog footprint counts as free, plus the VRAM this process reserved. Cold and genuinely too-big requests are refused as before. Nothing GPU-specific. Tests: `WarmPipelineGateTests` in `tests/test_memory_gate.py`.
- GPU bundle (`routes/setup/gpu_bundle.py`) already lists `ftfy` and `bitsandbytes>=0.43.0` with no platform marker; bitsandbytes ships a `win_amd64` wheel on PyPI (0.50.2 and since 0.43.1). They were absent only because the bundle had not run. `release-constraints.txt` has ftfy 6.3.1 but no bitsandbytes (not in the tested env); not re-frozen. Added `gpu_bundle_hint` (errors naming ftfy, bitsandbytes and the other bundle packages now say "Install GPU runtime, then restart") and a clearer NF4 fallback note. Licences: ftfy Apache-2.0, bitsandbytes MIT; noted in THIRD_PARTY_NOTICES.md.
- Not live-checked (no GPU run). `test_setup_routes.py` has 5 torch-index tests failing on this host with and without these changes.

## LTX-2.5 on Windows/CUDA (diffusers): scoping (2026-11-12, no rows added yet)

**Licence.** LTX-2.x Community License (`license: other`, binding text in the Lightricks/LTX-2 repo, `LICENSE-2_x`). Free commercial use under US$10M annual revenue, measured across the whole entity including affiliates; above that a paid Commercial Use Agreement; transferring fine-tunes may need a paid licence. Both Lightricks repos are `gated: auto`. Gates are per repo: the stored account can download from `Lightricks/LTX-2.5` but gets 403 on file reads in `Lightricks/LTX-2.5-Diffusers` (only the README is public), so the licence on the Diffusers repo has to be accepted on its own page.

**diffusers.** The official card says "install from main", but 0.42.1 (the version in the GPU extras) already has every class the card uses: `LTX2Pipeline`, `LTX2VocoderWithBWE`, `LTX2VideoDiffusionDecoderModel`, `LTX2LatentUpsamplePipeline`, `DISTILLED_SIGMA_VALUES`, `STAGE_2_DISTILLED_SIGMA_VALUES`, `LTX25ModularPipeline`; transformers 5.22.2 has `Gemma4*`. PyPI has 0.43.0; no upgrade needed or made. Not run (no weights): whether 0.42.1 loads the real checkpoint end to end.

**Layout of `Lightricks/LTX-2.5-Diffusers`** (distilled is the default; `transformer_full/` is the dev/SFT DiT): transformer 38.0 GB bf16 (22B, 48 layers; the listing holds a second shard set of the same size, probably stale), transformer_full 38.0, text_encoder (Gemma 4 12B) 23.9, connectors 6.3 (plus a duplicate 2-shard copy), vae 1.5, diffusion_decoder 0.8 (needs `kernels`/NATTEN, optional), latent_upsampler 1.0, temporal_latent_upsampler 0.3, audio_vae + vocoder 0.4, prompt_enhancer 10.2 and a distilled LoRA 9.7 (neither in `from_pretrained`). Card: 110 GB for everything `from_pretrained` needs, **72 GB** without `transformer_full/`. Single-stage distilled, sigmas `DISTILLED_SIGMA_VALUES`, no guidance; two-stage adds the x2 latent upsampler.

**Community builds checked.** `Lightricks/LTX-2.5` (single files for Comfy: bf16 42 GB, int8-convrot 21.5, nvfp4 18.7, Gemma int8 15.4; not diffusers layout). `KevinDowling/LTX-2.5-Diffusers-NVFP4` 21.2 GB, Blackwell only, gated. `agosh/...-Comfy-GGUF` and `unsloth/LTX-2.5-GGUF` (distilled transformer Q4_K_M 12.2 GB; Comfy key layout, transformer only). `unsloth/LTX-2.5-FP8` (ungated, diffusers layout mirror; FP8 19 GB / INT8-ConvRot 25 GB single-file DiTs that need a custom W8A8 runtime). **No bnb-nf4 diffusers pack of LTX-2.5 exists** (OzzyGT has 2.3 and 2 only).

**What fits 24 GB.** Gemma 4 12B bf16 alone is 24 GB, so on 24 GB both the transformer and the text encoder must be 4-bit. Pre-quantised pack (like the 2.3 row): loads with plain `from_pretrained`, no engine change; none exists. On-load 4-bit of both from the official repo (`PipelineQuantizationConfig`, bnb nf4): 72 GB on disk, new loader branch (today `useNf4` quantises the transformer only and `LTX2VideoTransformer3DModel` is not in `_BNB_NF4_VIDEO_TRANSFORMER_CLASSES`). Expected footprint ~31 GB pack-equivalent, ~22-23 GB peak like the 2.3 row (estimate; Gemma 4 vs Gemma 3 unmeasured).

**Proposed rows** (all repo `Lightricks/LTX-2.5-Diffusers`, cudaOnly, gated + licence note, `downloadFiles` pinned to the component folders, exact shard patterns to be read from the transformer index once the repo is accessible): (1) distilled 4-bit on load, 24 to 48 GB cards, ~22.5 GB peak, estimated until measured; (2) distilled bf16, 80 GB cards and unified memory (DGX Spark), ~76 GB in memory; (3) later: dev (`transformer_full`) bf16 for 80 GB cards, and NVFP4 for Blackwell once its layout is checked. Gates come from `variant_fit` and the hardware read, nothing keyed to a card.

**Not done:** no rows, tests or E2E check, no download. Blockers: licence not accepted on the Diffusers repo for the stored account; the 72 GB download would leave ~22 GB free on C: (floor is 25 GB), so it needs either space freed or the streaming-conversion route (download shard by shard, quantise to nf4, delete the shard: peak ~45 GB, network ~68 GB, output a ~31 GB local pack).

## Host-bound tests skip or pass everywhere (2026-11-12)

Full runs on Windows (RTX 4090): Python **3710 passed, 14 failed, 134 skipped** before, **3737 passed, 0 failed, 134 skipped** after; vitest 1528 passed, 1 failed before, **1529 passed** after. The WSL list had 3 more host-bound failures (soundfile, librosa) that now skip.

| Test | Cause | Fix |
|---|---|---|
| 5 CUDA torch index tests (`test_setup_routes`) | called the real `nvidia-smi` | mock "no NVIDIA driver" so they run everywhere |
| 2 `test_size_limits` gate tests | the size term used this card's free VRAM | pin `cuda_free_vram_gb` to None |
| `test_catalog_slow_on_labels_reach_the_payload` | Mac-only rows are filtered off other hosts | pin Apple Silicon for the catalog read |
| 4 `test_media_input` | path separator / `.exe` in command paths | compare with `Path` |
| Video Studio tooltip (vitest) | device guess read the host's navigator | stub a Mac navigator in that assertion |
| 3 voice/music tests | need soundfile / librosa | `skipIf` the package is missing, reason names it |
| E2E phase 4 mflux rows, phase 5 LTX-2.5 MLX row | catalog lists them on Apple Silicon only | new check "mflux image rows" and the LTX check skip off Apple Silicon |
| `test_a_killed_render_is_interrupted...` | **real bug**: the child process picked ffmpeg 9.0.2 from the extras, and Storyboard renders failed on it | see below |

Real bug: ffmpeg 9 prints two flag columns in `-filters` (every film failed "lacks the filters") and removed `-filter_complex_script`. `ffmpeg_bin._FILTER_LINE` reads both layouts and `assemble._filter_script_flag()` uses `-/filter_complex` from ffmpeg 7.0 on. Tests added in `test_ffmpeg_bin.py`. Not run against a live 9.x render end to end beyond the killed-render test.

E2E phases were not run (no backend started); the new skips are in `scripts/e2e_test_suite.py` (`_mflux_catalog_rows`, `_ltx2_engine`).

**Update 2026-11-12 (licence accepted, indexes read).** Access to `Lightricks/LTX-2.5-Diffusers` now works. `model_index.json` is `LTX2Pipeline`; the weight maps settle the duplicates: `transformer/` = the 4-shard set (`-0000N-of-00004`, 37.98 GB); the 8-shard set has no index and is stale. `connectors/` = the 2-shard set (6.34 GB); the single 6.34 GB file is stale. `text_encoder/` = 5 shards, 23.92 GB (Gemma 4, `Gemma4UnifiedForConditionalGeneration`). `processor`, `prompt_enhancer` (10.2 GB), `duration_head` are optional in `LTX2Pipeline` (`_optional_components`); the card's 72 GB leaves out `prompt_enhancer`, so pass `None` for it. Exact allow-patterns, distilled pipeline: `model_index.json`, `scheduler/*`, `tokenizer/*`, `processor/*`, `duration_head/*`, `transformer/config.json`, `transformer/*.index.json`, `transformer/*-of-00004.safetensors`, `text_encoder/config.json`, `text_encoder/generation_config.json`, `text_encoder/*.index.json`, `text_encoder/*.safetensors`, `connectors/config.json`, `connectors/*.index.json`, `connectors/*-of-00002.safetensors`, `vae/*`, `audio_vae/*`, `vocoder/*`, `latent_upsampler/*`. Sizes: **70.1 GB** without the latent upsampler, **71.1 GB** with it (route a, the whole download). Route b has the same network volume; it keeps a ~32 GB pack (transformer nf4 ~14.4, Gemma 4 nf4 ~8.4, connectors 6.35 kept bf16, VAEs/vocoder/upsampler ~2.8, from the 2.3 pack's proportions) and a peak of that plus one source shard (max 10 GB) and the HF partial file: **~45 GB on disk**. C: free is **61 GB** now (it was 94 GB earlier in the session; something else wrote ~33 GB), so route a (70 GB) does not fit and route b leaves ~16 GB, under the 25 GB floor. Nothing downloaded.

## TensorFold native on Windows: tfgrammar and the Gemma 4 / Qwen3.6 rows (2026-11-12, FU-111, FU-168)

**tfgrammar.dll builds on Windows.** `zig build grammar -Dxgrammar=DIR` (xgrammar `v0.2.8` cloned with submodules; same Zig 0.17.0 as the exe, no vcvars or nvcc needed, 32 s) wrote `zig-out\native\lib\tfgrammar.dll` (2.8 MB). Copied beside the installed exe (`~\.chaosengine\tensorfold-native\<version>\`; the server also opens `..\lib`). The app's detection (`grammar_library_present`, status `grammar: true`) saw it with no code change and the 400 refusal (`_require_grammar_for`) is skipped when present, kept when absent. Through the app (backend on 8894, Qwen3.5-2B MLX-4bit, `jsonSchema` name/age/tags, additionalProperties false): reply was schema-valid JSON in 0.9 s. Changes: the adopt step (`adopt_binary`) now copies a `*tfgrammar*` library found beside the exe, in `lib\` or in `..\lib` (a source build puts it in `native\lib`) into the data folder; the Windows "no build" install message names the `zig build grammar` step; test `test_adopting_brings_the_grammar_library_from_the_build_lib_folder`; E2E check "TensorFold 1.x native structured output (FU-111)" (skips without the binary, the library or Qwen3.5-2B).

**Gemma 4 31B** (`mlx-community/gemma-4-31B-it-4bit`, 18.4 GB, pulled 36 min, native engine, vision on, `--no-drafts`):

| Measure | Result |
|---|---|
| Load (8192 window) | 19.9 s; +21.8 GB on the card (23.1 GB total with the desktop's 1.3 GB), fits 24 GB |
| Plain greedy, 300 tokens, 3 runs | 42.5 / 42.5 / 42.6 tok/s |
| Quality | "The capital of France is Paris."; red and blue 64 px squares answered "Red" and "Blue" |
| 3K / 3.8K / 10K-token prompt | 41.2 / 41.2 / 39.0 tok/s decode (10K at a 12,288 window: +22.1 GB) |
| 16,384 window | **refused**: "the CUDA memory budget fits no stream beside the model: one at a 16384-token window takes 1.27 GiB of KV pages, and the budget leaves 0.95 GiB" |

Row: `measured=True`, `vram_gb` 21.8, KV 0.65 GB per 8K, note says keep the window at 12K or less on 24 GB.

**Qwen3.6-35B-A3B + MTP** (`TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP`, 20.9 GB, pulled about 45 min):

| Measure | Result |
|---|---|
| Load (8192) | 21 to 24 s; +20.1 GB (21.0 GB total), fits |
| Plain (`--no-drafts`), prose, 3 runs | 180 / 186 / 186 tok/s |
| MTP on (`speculativeDecoding`), prose | 178 / 187 / 187 tok/s (no gain on prose) |
| Code, 400 tokens, MTP on vs off | **417 to 425 vs 186 tok/s** (2.3x) |
| Context 3.8K / 14.5K / 30K prompt, MTP on | 172 / 167 / 156 tok/s; card +20.1 / +20.3 / +20.6 GB (windows 8K / 16K / 32K) |
| Quality | "The capital of France is Paris."; "a fox" from the long-context needle prompts |

The old row (23.0 GB estimate) made the app's own gate refuse the 16K window ("needs about 23.3 GB ... has 23.3 GB free") although it loads at 20.3 GB. Row now `measured=True`, `vram_gb` 20.2, KV 0.16 GB per 8K. One note: a first attempt while another session held 23.8 GB of the card was refused by that gate, as designed; rerun on the idle card.

Weights deleted afterwards from the HF cache (both; neither is on the Keep list); the cached Qwen3.5-2B stays. tfgrammar.dll stays installed beside the exe.

## W4 leftovers on Linux: vLLM and TriAttention (WSL2, 2026-11-12)

Own venvs in WSL (`uv`, no root): `~/venv-vllm` (vLLM 0.33.0, torch 2.14.0+cu130, Apache-2.0) and `~/venv-vllm19` (vLLM 0.19.1, torch 2.10.0+cu128, the release the TriAttention README names), each with the app installed (`-e .[desktop]`) and TriAttention from the pinned commit `a4bc3c8` (cloned without the unused LongLive submodule, whose clone made the git+https install hang). Model: `Qwen/Qwen3-0.6B` HF safetensors (1.4 GB, ungated) in the WSL HF cache. Backend run in WSL on port 8895 with `CHAOSENGINE_REQUIRE_AUTH=0`; every load and generate below went through `/api/models/load` and `/api/chat/generate/stream` (`vllm_engine.VLLMEngine`, the in-process Linux lane; the Windows host uses `vllm_wsl_engine`). GPU was under 1.4 GB used at every run start.

### 1. vLLM lane

| Item | Result | Evidence |
|---|---|---|
| `is_available` / install path | PASS | `vllmAvailable: true, vllmVersion 0.33.0` from `/api/health`. The in-process engine imports `vllm` from the interpreter running the backend, so the backend must run from the vLLM venv (the app venv pins torch 2.11). |
| Load + generate, native f16 KV, 8K and 16K | **PASS after 3 fixes** | Load 24 to 59 s. First run **FAIL**: `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` (FlashInfer sampler JIT during the memory profiling run). |
| Throughput | PASS | Short prompt (18 tokens, 600 to 750 generated): **413 to 439 tok/s** on 0.33.0, **320 to 325 tok/s** on 0.19.1 (includes first-token wait). |
| VRAM | not a comparison | vLLM reserves `gpu_memory_utilization` 0.92 on load: 0.9 GB before, **23.9 GB** after, whatever the context. The useful number is the KV pool it logs: **187,776 tokens** (20.1 GiB) on 0.33.0, 184,960 on 0.19.1. The app passes no `gpu_memory_utilization`, so another GPU job on the card makes the load fail. |
| Matrix `vllm native (Qwen3-0.6B)` | PASS | 277 to 300 tok/s. |

Surgical fixes (commits `08d6708c`, `ee2220d1`; tests in `tests/test_vllm_engine.py`, +8):
1. **FlashInfer sampler without nvcc** (load failed): `_prepare_vllm_env` sets `VLLM_USE_FLASHINFER_SAMPLER=0` when no nvcc is found (the WSL-bridge engine already did this).
2. **An installed TriAttention broke every vLLM load**: its `vllm.general_plugins` entry point patches every engine, `native` included. On 0.33 the engine core died on the first request (`AttributeError: 'EngineCoreProc' object has no attribute 'log_iteration_details'`) after the load had reported success. The engine now sets `ENABLE_TRIATTENTION=0` unless the strategy is `triattention`.
3. **Prompt format**: the engine fed `USER: ... ASSISTANT:` text; Qwen3 rambled and looped (needle check failed, 1500 tokens of repetition). It now applies the tokenizer chat template (plain text only when there is none). The needle answer went from FAIL to PASS.

### 2. TriAttention

| Item | Result | Evidence |
|---|---|---|
| Install + `is_available` | PASS | The pure-Python package installs on Linux; `TriAttentionStrategy.is_available()` is True, badge "Ready" (vLLM backend). It is not MLX-only. |
| vLLM 0.33.0 | **FAIL, now falls back** | The plugin patches a method vLLM renamed (fix 2). `apply_vllm_patches` now refuses when `EngineCoreProc.log_iteration_details` is missing, so the load comes back as native with the note "TriAttention's vLLM integration does not support vLLM 0.33.0" instead of dying on the first request (test added). |
| vLLM 0.19.1 | loads and generates, **compression never applies** | The plugin activates (`Runtime (V2) plugin activated`) and the scheduler signals every 128 tokens, but all **1885 of 1885** compaction events (long-decode run, budget 512) ended `no_compactable_groups`, 0 applied; same on the 5.5K-token needle prompt with budget 256. No KV was dropped. Cause not found (third-party runtime). |
| Stats file | n/a | TriAttention needs a per-model stats file; only Qwen3-32B and gpt-oss-120b ship for vLLM. Generated one for Qwen3-0.6B with `scripts/calibrate.py` (needed `--attn-implementation sdpa`; the default wants flash-attn) from 118 KB of docs text. |
| App wiring | gap | The app sets none of `TRIATTN_RUNTIME_KV_BUDGET`, `TRIATTN_RUNTIME_SPARSE_STATS_PATH`, `enforce_eager`, `enable_prefix_caching=False`. The runs below set the two env vars by hand. FU-171. |

Needle in a haystack (passphrase in the middle, ask for it, thinking left on, answer read from the end; Qwen3-0.6B, `contextTokens` 8192 / 16384, prompt 5,546 / 11,071 tokens). "Short" is tok/s for an 18-token prompt; "needle" is tok/s for the long prompt with 16 generated tokens, so it is prefill-bound.

| Run | 8K short / needle tok/s | 8K needle | 16K short / needle tok/s | 16K needle |
|---|---|---|---|---|
| native f16, vLLM 0.33.0 | 416 / 61 | PASS | 413 / 31 | PASS |
| native f16, vLLM 0.19.1 | 325 / 139 | PASS | 320 / 52 | PASS |
| triattention, budget 2048, 0.19.1 | 429 / 115 | PASS | not run | not run |
| triattention, budget 512, 0.19.1 | 411 / 123 | PASS | 418 / 61 | PASS |
| triattention on 0.33.0 | falls back to native (above) | PASS (native) | | |

Reading: the triattention rows are **not** evidence for the strategy, because nothing was compacted (with a 5.5K prompt and a 512 budget a working compressor would be expected to lose the needle). VRAM is 23.9 GB in every row (pre-allocated pool); the KV pool is 184,960 tokens in both 0.19.1 rows.

### 3. Cache-strategy matrix in WSL

`--quick` runs **0 cells**: the quick set is the MLX and GGUF cells (models absent, or turbo binary absent) and the vLLM cells are `quick=False`. Full run against the WSL backend (library: Qwen3-0.6B safetensors, plus the Qwen3-0.6B GGUF copied from the Windows cache):

| Cell | Result |
|---|---|
| native GGUF (smoke) | PASS, 525 tok/s |
| vllm native (Qwen3-0.6B) | PASS, 277 to 300 tok/s |
| vllm triattention (Qwen3-0.6B) | PASS on 0.19.1 (384 to 403 tok/s, same output hash as native: no compaction); on 0.33 the cell used to pass as native |
| the four turboquant cells, dflash, ddtree, mtplx, tensorfold, MLX cells, GGUF MTP, vllm dflash | skip (no turbo binary, runtimes not installed, models not in the WSL library) |

Fix (matrix): a `triattention` or `turboquant` cell whose load fell back to native is now a skip with the reason rather than a pass (2 tests).

Not done: a real TriAttention compression comparison (FU-171) and DFlash on vLLM (no Qwen3.5-4B copy, size).

Footprint: WSL `ext4.vhdx` 36.0 GB (23.9 before), C: free 56 GB; the two venvs are 9.4 and 9.6 GB, the WSL HF cache 3.7 GB.

## LTX-2.5 4-bit text encoder branch: written, not run live (2026-11-12)

**Verified: unit tests with fakes only.** No weights downloaded, no GPU used, no backend started. The tests (`tests/test_video_ltx25.py`, 25) cover: catalog shape (two rows on the one gated repo, estimated figures, licence note, no card names), the access gate, quantisation config (nf4, bf16 compute, skip list), the text-encoder class read from `text_encoder/config.json`, fallback notes (no bitsandbytes, class missing from transformers, no architecture, non-CUDA, load error), tier choice at 12/24/48/96 GB free and unified memory, Windows `disable_mmap` kwargs, and a snapshot without `prompt_enhancer` validating complete. Existing video, fit, size-limit and catalog test files pass.

**What changed.**
- `video_runtime/ltx25_loader.py` (new): transformer through `try_load_bnb_nf4_transformer` (`LTX2VideoTransformer3DModel` now registered for the LTX-2.5 and official LTX-2.3 repos), Gemma 4 encoder from the architecture named in its own config with `BitsAndBytesConfig(nf4, bf16 compute, llm_int8_skip_modules=[lm_head, embed_tokens, per-layer embeddings, vision/audio towers and projectors])`, `prompt_enhancer=None`. Encoder stays bf16 when free GPU memory >= 48 GB (`CHAOSENGINE_LTX25_TE_BF16_MIN_GB`) or on unified memory; a 4-bit transformer that fails raises (a 38 GB bf16 one would fill system RAM), an encoder that fails falls back to bf16 with a note.
- `loadQuantization: "nf4"` on a row makes `app.py` ask for NF4; the 4-bit row and the bf16 row share the repo, so the run key differs by `useNf4`. Footprint for placement is 22.5 GB (nf4) or 76 GB, so model CPU offload follows the same rule as LTX-2.3.
- `pipeline_load_kwargs()` (`disable_mmap` on Windows) now also goes to the NF4 transformer load.
- Catalog `catalog/video_ltx25.py`: `Lightricks/LTX-2.5-Diffusers~nf4` (22.5 GB peak, estimated) and `Lightricks/LTX-2.5-Diffusers` (76 GB, estimated), CUDA-only, allow-patterns from the scoping section, LTX-2.x Community License terms in the note. Mac MLX row untouched.
- Gate: gated rows with no complete weights and no saved token carry `accessBlockedReason`, are never a fit or the recommendation, and the download route answers 403 with the reason. A saved token is the only offline signal; the real answer comes from the download.
- `validate_local_diffusers_snapshot` ignores `prompt_enhancer` for this repo.
- E2E phase 5 check "LTX-2.5 on diffusers, 4-bit on load" skips off CUDA and without the weights.

**Not verified (live steps still needed).** Whether diffusers 0.42.1 and transformers 5.22.2 load the real checkpoint; that `Gemma4UnifiedForConditionalGeneration` quantises with this skip list (module names are guessed from Gemma 3n/4 conventions, read them from the first real load); that `disable_mmap` is accepted with a quantised diffusers load; the 22.5 GB peak, seconds and the 48 GB tier threshold (all estimates). Steps: free disk (route b: stream-convert ~45 GB peak) or fit 71 GB; download through the app (`/api/video/download`, ~71 GB, needs the token); load the 4-bit row; one 512x320, 25-frame clip with sound; record peak VRAM and seconds; then flip `footprintBasis` to measured.

## Installer rebuilt via `./build.sh`, installed over the installed app, section 7 smoke (2026-11-12, third run)

Installed over 0.10.0 (silent `/S`, exit 0) with your OK, build at `0326b5c2`. Ports 8876/8890/8891 were free, so no settings trick; the installed backend ran on 8876. Page file was system-managed. C: free 81.6 GB before the build.

| Item | Result | Evidence |
|---|---|---|
| `./build.sh` (clean `.venv-release`, `release-constraints.txt`) | PASS | No fix needed this time; about 25 min including a full Rust compile. Size guard passed. |
| Installer / archive | PASS | `ChaosEngineAI_0.10.0_x64-setup.exe` **131.75 MiB (138,149,774 B)**, runtime archive **133.5 MB**; manifest `llamaServer` and `llamaServerTurbo` set, `warnings: []`; the four `install-*.sh` are in the archive. |
| Backend starts | PASS | `/api/health` 200 after 28 s, `appVersion` 0.10.0, runtime `windows-x86_64-1e6de1b8`; health reports the CUDA llama-server `llama-cuda\b12292`. |
| Torch CUDA build | PASS | Overlay `torch 2.14.1+cu130`; with the app's python: CUDA available, RTX 4090 (4 s import); image runtime `expectedDevice cuda`, 23.99 GB. |
| `llama-cuda-status` | PASS | `in-use`, tag b12292, CUDA 13.5, verified. |
| `tensorfold-native-status` | PASS | installed 1.2.6, gate `ready` (nvidia-sm89), **`grammar: true`**. |
| FLUX.2 Klein 4B 1024x640, 4 steps, three runs | PASS, one oddity | Cold 84.3 s wall (10.0 s generation); warm 10.4 s (8.9 s); second warm 11.4 s (9.6 s). No crash. Peak backend private bytes 25.6, 23.7 and **48.9 GB** (the third is not explained; a sampling artefact or a transient, nothing failed). |
| LTX-2.3 4-bit text mode 768x512, 97 frames, clip 1 (FLUX pipeline still resident) | PASS | 151 s wall, **84.4 s generation** (baseline about 70 s), peak private **72.1 GB**. The cross-model warm gate let it through. |
| Clip 2 back to back, same model | **FAIL then PASS** | First attempt: HTTP 503 `Memory pressure is 98%` at 0.1 s: free RAM 1.0 GB of 63.9 GB (backend working set 45 GB, private 60.8 GB), above the 98 % chat ceiling that `reuse_resident` falls back to (98.4 %). Ten seconds later free RAM was 1.6 GB (97.5 %) and the same request ran: 81.6 s wall, **80.4 s generation**, peak private **76.3 GB**. So the warm same-model gate sits on the edge: the resident LTX pipeline itself keeps the host at about 98 % pressure and the request passes or is refused on noise. Not changed (needs a decision: for `reuse` the resident pipeline's own memory should probably not count as pressure). |
| Commit / page file | PASS | Commit 83 to 85 GB against a limit of 100.4 GB during and after the clips; the system-managed page file grew to 36.6 GB allocated but only 87 MB was ever used. Side effect: C: free fell from 98.7 to 72.7 GB (the page file grows on disk when commit passes RAM). After the runs the backend held 23.2 GB private. No `python.exe`, `dwm.exe` or other fault in the Application log (Event 1000/1001) in the previous two hours. |
| `scripts/e2e_test_suite.py --smoke` against the installed backend | **FAIL (1 of 7 phases)** | Phases 0, 3, 4, 5, 6, 8 PASS; phase 7 `diagnostics-snapshot` FAIL: the CLI timed out at 30 s. `GET /api/diagnostics/snapshot` answers in a constant **61.2 s** on the installed backend (twice, also with the backend at 23 GB). Not isolated: the one 30 s subprocess in that route (torch probe with `sys.executable`) runs in 4 s when started by hand with the overlay on `PYTHONPATH`. Phase 8's video check skipped on the memory gate (503) while the LTX pipeline was resident. Reports: `~/.chaosengine/test-results/e2e-20261112-145929.md`. |

Open: the snapshot latency (first suspect: the torch probe subprocess; run the route under a profiler on the installed backend) and the warm same-model video gate margin.

## Warm same-model gate: resident memory credited on reuse (2026-11-12)

The back-to-back LTX-2.3 refusal (98.4 % pressure, 1.0 GB free, 45 GB resident, 63.9 GB total) came from the resident pipeline itself holding the host at the pressure ceiling. `adjust_for_resident()` now credits the resident pipeline's memory on reuse as it already did on a swap: `min(process RSS, catalog footprint)` (RSS alone when the loaded row is unknown), added to available RAM (capped at total) and pressure recomputed. The reuse flag no longer relaxes the image/video floors to the chat gate's; the credited figures meet the normal floors, so other apps eating RAM still refuse (e.g. 4 GB free + 1 GB credit < 6 GB video floor) and a too-big model is still refused. Image and video routes share the helper, so both are covered. Reused pipelines keep their GPU memory as resident (no VRAM credit; that is swap-only). Tests: `ResidentCreditOnReuseTests` in `tests/test_memory_gate.py` (installed-app case allowed, credit caps, low headroom refuses, image gate); memory_gate, media_fit_cuda, chat_memory_cuda, backend_service all pass. Not re-run live (no GPU run).

## Diagnostics snapshot: 61 s to 4 s

`GET /api/diagnostics/snapshot` took a constant ~61 s on the installed backend and timed out E2E phase 7. Per-section timing on the dev backend: extras 26.0 s, gpu (torch subprocess probe) 2.4 s, hardware 0.9 s, rest under 0.1 s. Cause: `_extras_info` walked the whole extras site-packages tree (torch and CUDA wheels, tens of thousands of files) with `rglob` on every call; the installed app is slower per file than dev.

Fix in `backend_service/routes/diagnostics.py`: the walk is now an `os.scandir` pass run once in a background thread and cached for 10 min (the snapshot waits at most 2 s, else `extras.sizeNote` says it is still measuring); sections run in parallel under a 12 s total deadline and a section still running reports a timeout note. Dev backend after: 3.9 s on both calls (the torch probe now dominates). Tests: `tests/test_diagnostics_snapshot.py`.

## Final installer rebuild, installed, smoke at `16bd472b` (2026-11-12, fourth run)

`./build.sh` (size guard passed) then silent install (`/S`, exit 0) over the installed app, with your OK. Ports 8876/8890/8891 free, C: free 104 GB before. Installed backend on 8876. Generation ran only after the GPU was idle (nvidia-smi 1.3 GB used).

| Item | Result | Evidence |
|---|---|---|
| Build / installer | PASS | `ChaosEngineAI_0.10.0_x64-setup.exe` **131.78 MiB**; build exit 0. |
| Backend / CUDA | PASS | `/api/health` 200, 0.10.0, CUDA llama-server b12292; `torch-upgrade-available` reports `2.14.1+cu130` already latest; `llama-cuda-status` in-use CUDA 13.5 verified; `tensorfold-native-status` installed **1.2.6** (pin in CLAUDE.md says 1.2.7), gate `ready` on nvidia-sm89. |
| `GET /api/diagnostics/snapshot` | PASS | **4.4 s** (was 61 s), HTTP 200, 23.7 KB. |
| FLUX.2 Klein 4B 1024x640, 3 runs | PASS | Cold 90.5 s wall (26.2 s gen); warm 8.7 s (7.3 s); 9.0 s (7.4 s). Peak private 23.2 / 27.7 / **47.6 GB** (the third run shows the same jump as last run, no failure). |
| LTX-2.3 4-bit 768x512, 97 frames, twice back to back | **PASS** | Clip 1 138.2 s wall, 79.4 s gen, peak private **75.0 GB**. Clip 2 78.8 s wall, 77.2 s gen, peak **72.9 GB**; not refused (the warm-gate credit works). |
| `e2e_test_suite.py --smoke` | **FAIL (1 of 7)** | Phases 0, 3, 4, 6, 7, 8 PASS; **phase 7 diagnostics now PASS (6.2 s)**. Phase 5 FAIL: check "LTX-2.5 on diffusers, 4-bit on load": "a gated row with no weights and no token must say why it is blocked". The catalog row `Lightricks/LTX-2.5-Diffusers~nf4` has `gated: true` and no `accessBlockedReason` (and `hasLocalData` false) with no HF_TOKEN set. The same phase's generate check skipped on a 503 memory gate while the LTX pipeline was resident. Reports `~/.chaosengine/test-results/e2e-20261112-155607.{json,md}`. |
| Windows Application log | **FAULTS** | No `python.exe` crash (Event 1000/1001 for python). But Event 1001 `LiveKernelEvent` code **141** (GPU engine hang/reset), each with an `AppTermFailureEvent` report, at **15:52:16** (end of the LTX runs) and **15:57:51-52** (during E2E phase 8's video work); earlier one at 14:37:40-42 (previous run). Both generations returned HTTP 200 and the backend stayed up (health 200 afterwards). Dump files under `C:\ProgramData\Microsoft\Windows\WER\ReportQueue\Kernel_0_0_*`. Not analysed. |

Open: LTX-2.5 gated row without `accessBlockedReason` (either set it on the row or relax the E2E check); the GPU code-141 events during heavy video runs (check the NVIDIA driver and TDR delay; a 4090 at sustained load); TensorFold native runs 1.2.6, not the 1.2.7 pin.

- E2E video phase gated-row check fixed: it now treats a saved Hugging Face token (settings `hasHuggingFaceToken` flag or env var, flag only) as "token present" and skips; the reason is asserted only with no token. Backend gate unchanged; unit tests cover both branches and the download 403 / token-open route.

## Qwen3-TTS speed on CUDA (2026-11-12)

Standalone runs in the voice venv with the worker's own loader (`load_qwen`), RTX 4090, torch 2.9.1+cu128, same text (21 words), seed 7, one warm call then 3 timed runs; GPU use sampled with `nvidia-smi` every 0.2 s; quality = sample-for-sample comparison and a Whisper Large v3 Turbo transcript.

- **Launch-bound, confirmed** (cProfile of one request): about 300 k small GPU ops per request, ~45 us of host time each (Windows launch cost); the 5-layer code predictor runs 15 forward passes per talker step and takes about 70 % of them, the 28-layer talker the rest. The GPU sits at 25 to 40 %.
- **Kept: plain code-predictor loop** (`backend_service/qwen_tts_fast.py`, installed by `voice_worker_torch.speed_up_qwen`). `qwen-tts` calls the full `transformers` `generate()` for the 15 codes of every talker step; the loop does the same forward passes and the same sampling (temperature, top-k, top-p, one `torch.multinomial`) in the same order. Output is bit-identical for a seed (max sample difference 0.0, same transcript).

| Model | Before (s compute per s of audio) | After | Audio |
|---|---|---|---|
| CustomVoice 1.7B, Ryan | 1.913 (16.8 s for 8.80 s), GPU 38 % | 1.702 (15.0 s), GPU 39 % | identical, transcript exact |
| Base 1.7B clone (12 s synthetic reference, ICL with transcript) | 3.302 (23.8 s for 7.20 s), GPU 25 % | 2.940 (21.2 s), GPU 27 % | identical, transcript exact |

  The Base row's "after" also reuses the reference's speaker embedding and codes (`cached_clone_prompt`, per clip path, modification time, transcript and mode; cleared on unload). `CHAOSENGINE_QWEN_FAST=0` turns the loop off; if the loop fails it notes why and every later call uses the library's `generate()`.
- **Not kept: CUDA graphs of the code predictor** (static KV cache, one graph per position, sampling eager between replays so the random stream is unchanged). The prefill graph matched eager; decode graphs gave wrong logits when the 15 graphs shared a memory pool, and on the 1.7B model also with separate pools, while the same plan run eagerly on the same static cache was right. Not finished inside the time box; not shipped. `torch.compile` (inductor) needs Triton, which this Windows venv does not have.
- **Not tried in this pass**: attention backends (only SDPA and eager are installed; SageAttention and flash-attn are not on Windows, and the predictor attends over at most 16 keys, so attention is not where the time goes), fp16 against bf16, a static cache for the 28-layer talker.
- **0.6B Base "last word of the reference leaks"**: our worker passes the transcript stripped and unchanged, and `qwen-tts` cuts the reference part of the decoded audio by exact frame ratio, so no trimming bug was found on our side; the reference (transcript ends "...a short queue of early coffee.") ends on that word, and the model repeats it before the new text. Treated as model behaviour; not re-run on the GPU in this pass (no fix, no test).
- Tests: new `tests/test_qwen_tts_fast.py` (loop order, step numbers and one cache; other calls and failures go to the library; sampling order; install once; prompt cache keys; worker wiring and the off switch); `tests/test_voice_qwen_cuda.py` passes.

## Installer rebuilt at fa42c580, installed, Qwen3-TTS and GPU code-141 check (2026-11-12)

Rebuilt with `./build.sh` from `fa42c580` (Qwen3-TTS loop, E2E fix), installed over the app (`/S`, exit 0). Installer **131.8 MiB (138,156,687 B)**, runtime archive 133.5 MB, manifest `llamaServer` and `llamaServerTurbo` set, `warnings: []`. Health after 29 s; `/api/diagnostics/snapshot` **5.2 s** (was 61 s).

| Item | Result | Evidence |
|---|---|---|
| Qwen3-TTS CustomVoice 1.7B (voice Ryan, style calm, seed 7) through the app | PASS | `qwen3-tts-custom-1.7b-cuda`, `notes: []` (no fallback note), cold 19.2 s, warm 9.0 to 12.2 s for 5.52 s of audio (about 1.6 to 2.2 s per s; the standalone "after" figure was 1.70), 24 kHz, peak 22655, identical samples across runs for the seed. |
| Clone (existing 1.7B Base clone voice, `POST /api/voice/clone/speak`) | PASS | `Qwen3-TTS 1.7B Base (GPU)`, `notes: []`, 10.1 to 14.5 s for 6.2 s of audio (about 1.6 to 2.3 s per s); the saved WAV is 24 kHz, 6.16 s, not silent (peak 18372). The speed-up is not logged by name; "used" rests on the empty notes and the timing, not on a marker. |
| FLUX.2 Klein 4B 1024x640 | PASS | cold 58 to 59 s wall, warm 9.6 s wall (7.8 to 8.4 s generation), three runs. |
| LTX-2.3 4-bit 768x512, 97 frames | PASS | 69.5 s and 73.8 s generation (128 to 131 s wall), peak private 57.6 and 76.7 GB. |
| **GPU code-141 check** | **No new events** | Windows Application log before: last LiveKernelEvent 141 at 16:18 today (many earlier, from the earlier sessions). Nothing else on the GPU (idle 1.1 to 1.4 GB, desktop only; no compute processes holding memory). After 13 successful generations one at a time (3 image, 2 video, 6 TTS custom and clone in the series, 2 more TTS in the smoke) from 17:30 to the end: **LiveKernelEvent 0, AppTermFailureEvent 0, APPCRASH 0**, nothing from the display driver in the System log; checked after every run. So the 141s seen earlier came with the WSL pass and the 2 GB-card-spilling runs, not with these workloads on a clean GPU. |
| `--smoke` against the installed backend | not rerun | the diagnostics fix is covered by the 5.2 s measurement. |

**Found, not fixed (for a decision):** after a video clip (or image then video) the backend keeps its host memory after `POST /api/video/unload` and `/api/images/unload` (`loadedModelRepo` null): working set 45.7 GB, private 60.6 GB, only 3.3 GB RAM free. The next image or video request is then refused by the gate ("needs about 18 GB, only 3.3 GB available"; "26 GB ... only 3.3 GB"). 9 of 22 requests in the run hit this (3.3 to 14.7 GB free); only restarting the backend cleared it. The same happened after Qwen3-TTS runs (backend 30.8 GB working set, 14.6 GB free, image and video refused). Earlier today the dev backend returned to 4 GB after unload and the first installed run to 23 GB, so the release is not reliable on the installed build.

## Host RAM kept after Unload on Windows: root cause and fix (2026-11-12)

**Root cause: torch's allocator, not a Python reference.** PyTorch on Windows allocates CPU memory through mimalloc (linked into `c10.dll`). Weights are allocated by threads that exit soon after (the loader threads that read the weights, request workers that go idle), so their memory sits in mimalloc "abandoned" segments. Unload frees the weights from another thread, and by default mimalloc does not take an abandoned segment back when a block in it is freed, so `MIMALLOC_PURGE_DELAY=0` never ran for it and the memory stayed committed. Which thread allocated and whether it had exited decides it, so it hit some loads and not others (dev and installed alike; torch 2.11 and 2.14 carry the same mimalloc).

Evidence:

| Check | Result |
|---|---|
| Installed app at `fa42c580`, `img img vid img vid` with unload after each | 3rd image: unload 200, `loadedModelRepo` null, backend stayed at **16.9 GB working set / 23.1 GB private**; the next video loaded on top (45.9 / 61.0 GB) and its unload dropped back to 16.9 / 23.2 only. Second session: the first image already stuck at 16.6 / 22.8 GB. |
| In the stuck process (diagnostic route injected through a scratch `sitecustomize`, removed afterwards) | no pipeline, no `nn.Module` over 300 MB and no CPU tensor alive (`gc`); private commit 22.2 GB, 18.6 GB of it in 1 to 2 GB mimalloc arena regions; `gc.collect`, `torch.cuda.empty_cache`, `torch._C._host_emptyCache`, `_heapmin`, `HeapCompact` freed nothing; allocating in all 40 request workers freed nothing; the next load did not reuse it (private grew by a whole pipeline). |
| Standalone, embedded Python + torch 2.14.1 | tensors made in a thread that exits, freed from the main thread: kept **21.8 of 21.8 GB** (512 KB tensors) and **14.1 of 14.1 GB** (24 MB tensors) with no options or with purge delay 0; with `ABANDONED_RECLAIM_ON_FREE=1` plus purge delay 0: back to the **1.6 GB** baseline. Reclaim alone, or abandoned-page purge (with or without purge delay 0), changes nothing. Same result on the dev venv's torch 2.11 (2.86 GB kept vs 1.61 GB). |
| Voice worker | separate process, about 2 GB working set / 8.2 GB private while idle with Qwen3-TTS 1.7B on the GPU; not where the RAM was. |

**Fix** (`6d4b3fef`): `backend_service/__init__` applies `helpers/malloc_release.MIMALLOC_ENV` on Windows before torch loads: `MIMALLOC_PURGE_DELAY=0`, `MIMALLOC_RESET_DELAY=0` (as before) and **`MIMALLOC_ABANDONED_RECLAIM_ON_FREE=1`**; a value the user exported wins. Gate unchanged. Tests: `AllocatorEnvTests` and a real measurement `AbandonedSegmentMeasurementTests` in `tests/test_malloc_release.py` (about 1 GB made in an exited thread and freed elsewhere; fails without the fix: "kept 1.26 GB of 1.26 GB"). E2E phase 4 check "memory returns after unload; a second generation is not refused" (two generations of the smallest installed diffusers image model with unload; fails when unload keeps over 30 % of what the model took): PASS on the dev backend (FLUX.2 Klein: 4.0 GB before, 23.9 loaded, 4.0 after, twice). pytest: memory_gate, malloc_release, voice_qwen_cuda, qwen_tts_fast, image_runtime, video_runtime, media_fit_cuda, chat_memory_cuda, backend_service: **403 passed**.

**Rebuilt and installed** (`./build.sh`, exit 0; `ChaosEngineAI_0.10.0_x64-setup.exe` 131.75 MiB; manifest `llamaServer` and `llamaServerTurbo` set, `warnings: []`; archive holds the four `install-*.sh` and the fix; silent `/S` exit 0; backend up in 32 s). Mixed sequence through the installed app, unload after every image and video:

| Step | Loaded (backend WS / private GB) | After unload | Free RAM after |
|---|---|---|---|
| Klein 1024x640 x2 | 16.6 / 23.1 | 1.7 / 3.2 | 48.2 GB |
| LTX-2.3 4-bit 768x512, 97 frames x2 | 30.9-31.0 / 41.3-41.4 | 2.0 / 3.5 | 48.2 GB |
| Qwen3-TTS CustomVoice 1.7B x2 (worker) | backend 2.0 / 3.5; worker 1.9 / 8.2 | worker stays loaded | 46.2 GB |
| Klein, LTX, Klein, LTX, Klein | 16.9-17.0 / 23.4 and 31.2-31.3 / 41.4 | 2.0-2.4 / 3.5-3.6 | 45.5-46.0 GB |

11 of 11 requests HTTP 200, **0 refused** (before: 9 of 22 refused). Every unload back to about 2 GB working set.

Not verified: other cards and RAM sizes (the fix is an allocator setting, independent of card and RAM, but only this 4090 / 64 GB PC was measured); Linux and macOS (not affected: no mimalloc there, `malloc_trim` path unchanged); very small CPU allocations (32 KB tensors) still stay with the abandoned thread in the standalone probe (6.2 GB of 6.3 kept), which model weights do not hit in the runs above.

## FLUX.2 Klein through the NF4 FLUX loader (2026-11-12)

**Cause.** `image_runtime/transformer_loaders.py` loaded every repo with "flux" in its name (`_is_flux_repo`) with `FluxTransformer2DModel` (FLUX.1). Klein's snapshot declares `Flux2Transformer2DModel` (`transformer/config.json` `_class_name`). Direct repro with diffusers 0.42.1: every Klein key unused (`single_transformer_blocks.*.attn.to_qkv_mlp_proj`, `*.ff.linear_in/out`, `double_stream_modulation_*`, `time_guidance_embed.*`) and every FLUX.1 key newly initialised (`norm1_context`, `ff.net.*`, `attn.to_q/k/v`, `proj_mlp`). The NF4 load then raised `NotImplementedError: Cannot copy out of meta tensor`, the loader caught it ("falling back to bf16"), and the pipeline loaded the right class in bf16. So images were never garbage (current output is pixel-identical to a plain bf16 load), but Klein never got NF4 and every cold load paid for the failed attempt and printed the warnings. FLUX.2 Dev took the same path; the MPS int8wo loader had the same bug.

**Fix.** Both loaders take the class from the snapshot's `transformer/config.json` `_class_name` (FLUX.1 class when the file is missing or odd; a class the installed diffusers lacks returns the bf16 fallback note without loading). Tests: `FluxTransformerClassTests` in `tests/test_image_runtime.py` (6). `test_image_runtime.py` + `test_teacache.py` and `-k "image or nf4 or flux or quant"` pass.

**Evidence.** Klein 1024x640, 4 steps, seed 1234: after the fix, no unused / newly-initialised warnings, NF4 `Flux2Transformer2DModel` loads (13.3 s), denoise warm 4.6 s vs 6.3 s bf16 (script) and 4.6 s vs 7.8-8.4 s earlier through the dev backend; peak CUDA allocation 8.38 GB both ways (the Qwen3 text encoder dominates). Image: same composition and quality as bf16, small detail differences (PSNR 21 dB vs bf16), checked by eye. Not verified: FLUX.2 Dev (not downloaded), MPS int8wo, other cards.
## TriAttention on vLLM: root cause and a working combination (2026-11-13)

FU-171. WSL2 Ubuntu 24.04, RTX 4090, Qwen3-0.6B, the stats file made on 2026-11-12 with TriAttention's `scripts/calibrate.py`. Plugin at commit `a4bc3c8` (upstream `main` is unchanged since).

**Root cause.** The plugin copies vLLM 0.18-era model-runner internals; 0.19 changed two of them.

| vLLM | What happens | Evidence |
|---|---|---|
| 0.19.1 | compacts nothing | vLLM's `bind_kv_cache` sets `layer.kv_cache = [kv_cache]` up to v0.18.x and `= kv_cache` (bare tensor) from v0.19.0. The plugin's `kv_group_resolver.resolve_group_tensors` skips any layer whose `kv_cache` is not a list, so it found no KV tensors and `hook_group_pipeline` returned `no_compactable_groups` for every event (and that reason is on the plugin's allowed-skip list, so nothing failed). |
| 0.19.1 + a resolver shim | compacts once, then crashes | `compression applied ... before=7794 after=2048 reclaimed_blocks=360`, then on the next step `AttributeError: 'Tensor' object has no attribute 'np'` in the plugin's copied `_prepare_inputs` (`self.positions.np`): `positions` is a `_make_buffer` CPU/GPU buffer up to v0.18.1 and a plain GPU tensor from v0.19.0. Porting that copy is upstream work; the shim was dropped. |
| 0.33.0 | engine core dies on the first request | `EngineCoreProc.log_iteration_details` is gone (found 2026-11-12). |
| 0.18.1 | **compacts** | Scratch venv: a hardlinked copy of the 0.19.1 venv plus `vllm==0.18.1` (same torch 2.10), +3 GB on the WSL disk. |

**Our fix (adapter).** On 0.18.1 the first compaction worked and then no other: the plugin's `mark_compression_skipped` stamps the skipped step as `last_compression_step`, and its one-step dedup guard then skipped every following signal (budget 512, 3000 tokens: 1 applied, 2304 `batch_queue_dedup` skips). `_install_skip_stamp_fix` keeps the last applied step: 20 compactions on the same run, 39 at 6000 tokens.

**Prompt handling.** The plugin compacts before the step's forward pass. A 7.8K prompt prefilled in one step was compacted before its KV existed: garbage from the first token, then `effective_len_regressed` once the dedup fix let it continue. With 2048-token prefill chunks it compacts the prompt correctly, but a prompt in the budget loses detail. With the prompt protected (not counted in the budget) only generated tokens are compacted. The app uses chunks of 2048 and the protected prompt.

Needle (passphrase `PURPLE-OTTER-4821` in the middle of filler, thinking on, 1024 max tokens), vLLM 0.18.1, `enforce_eager` in every row:

| Run | Prompt | Compactions | Needle | e2e tok/s |
|---|---|---|---|---|
| native | 7,794 | 0 | PASS | 72.7 |
| native | 15,580 | 0 | PASS | 66.4 |
| prompt in budget 2048, one-chunk prefill | 7,794 | 1, then garbage | FAIL | |
| prompt in budget 2048, chunks 2048 | 7,794 | 2 | FAIL (`PURPLE-4821`) | 15.8 |
| prompt in budget 4096, chunks 2048 | 7,794 | 2 | FAIL (`PURPLE-4821`) | 28.6 |
| prompt in budget 4096, chunks 2048 | 15,580 | 5 | FAIL (made-up passphrase) | 21.0 |
| prompt in budget 2048, chunks 2048 | 15,580 | 14 | FAIL (never left thinking) | 73.1 |
| **prompt protected, budget 2048 (app setting)** | 15,580 | 0 (161 generated) | **PASS** | 66.5 |

Long generation (69-token prompt, up to 6000 tokens):

| Run | Generated | Compactions | tok/s | Per-request KV at the end |
|---|---|---|---|---|
| native, CUDA graphs | 5,779 (finished) | 0 | **340.5** | ~5,850 tokens |
| native, eager | 5,241 (finished) | 0 | 80.6 | ~5,310 tokens (332 blocks) |
| TriAttention budget 1024, eager | 6,000 (still reasoning) | 39, 312 blocks reclaimed | 76.9 | at most 1,152 tokens (72 blocks) |
| same, prompt protected | 6,000 (still reasoning) | 39 | 76.1 | same |
| **app `VLLMEngine`, budget 1024, 3000 tokens** | 3,000 | 15 | 70.1 | |

Reading:
- Compaction is real on 0.18.1: blocks go back to vLLM's pool, so a long generation holds about 4.6x fewer KV blocks at budget 1024.
- Peak GPU memory does not drop. vLLM reserves its pool at load (`gpu_memory_utilization` 0.6 here, torch peak 13.8 GB in every row). The gain is pool room for more or longer requests at once, not lower VRAM.
- The cost is eager mode, which the plugin requires: decoding is 80 tok/s against 340 with CUDA graphs on this small model. Compaction itself costs about 5 % on top (76.9 against 80.6).
- Answer quality is not measured. The math prompt was badly worded and Qwen3-0.6B answered it wrongly with and without compression. Text stayed coherent through 39 compactions, but the compressed run had not finished reasoning at 6000 tokens, while native finished at 5,241.

**App changes.**
- `cache_compression/triattention.py` refuses vLLM >= 0.19 with the reason. The Settings badge reads Unavailable, tone warning.
- It looks up the stats file `<repo name>.pt` in `~/.chaosengine/triattention/stats` (`CHAOSENGINE_TRIATTENTION_STATS_DIR` overrides). The pip install ships no stats, and without a file the load falls back to native with a note.
- It installs the dedup fix and returns the env and `LLM` kwargs: budget (`CHAOSENGINE_TRIATTENTION_KV_BUDGET`, default 2048), stats path, protected prompt, `enforce_eager`, no prefix caching, 2048-token chunks, and the engine core in process so the fix reaches the scheduler.
- `VLLMEngine` applies them and clears them for the next load.
- The `[triattention]` extra pins `vllm>=0.15,<0.19`, and the accelerator card copy no longer claims chat KV compression in general.
- Tests: `tests/test_vllm_engine.py`, 14 new tests replacing the 2 for the old engine-core check.

**Not done or not verified.**
- The app's own vLLM is >= 0.28, so a stock install shows TriAttention on vLLM as Unavailable. It works only from a hand-made vLLM 0.18 environment, and nothing in the app creates one.
- No E2E check added, since no user-reachable feature shipped.
- Not run through the HTTP backend or the UI. The engine was driven directly.
- A model larger than 0.6B was not tried.
- The scratch venv `~/venv-vllm18` is left in WSL.

## Qwen3.8-27B + DFlash2 on the vLLM lane (WSL2, vLLM 0.33.0, 2026-11-13)

**Verdict: the plain 27B runs through the app on a 24 GB card at about 53 tok/s. With the DFlash 2 drafter it does not fit 24 GB at any context; FAIL (memory), not a wiring fault.** Three app bugs on the way are fixed (below). The speed-up and acceptance on vLLM remain unmeasured (FU-172).

**Weights (downloaded through the WSL backend's `/api/models/download`, kept):** `grevinden/qwen3.8-27b-w4a16-autoround-rtx3090` 15.61 GiB (W4A16 compressed-tensors with int8 `lm_head` and `embed_tokens`, Apache-2.0) and `z-lab/Qwen3.8-27B-DFlash2` 3.58 GiB (revision `50307d4c`). The catalog has no vLLM row that fits 24 GB (its only vLLM row is FP8, 30.9 GB). `RedHatAI/Qwen3.8-27B-INT4` (19.5 GB, bf16 `lm_head` and embeddings) was not taken: about 21.5 GiB of weights with the drafter.

**The pack does not load on stock vLLM 0.33.** Error: `There is no module or parameter named 'embed_tokens.weight_packed' in Qwen3_5Model`. vLLM builds the Qwen3.5 embedding without a quant config, and the pack's card names a vLLM fork. For the runs, a local copy (`~/models/qwen3.8-27b-w4a16-bf16embed`, 2.4 GB of new data, the rest symlinked to the HF cache) has the int8 embedding dequantised to bf16 (group 128 scales). It was loaded through `/api/models/load` with `path` and `backend: vllm`.

Setup: dev backend in WSL from `~/venv-vllm` on port 8897. Each cell is one load, three 300-token replies per prompt (after one warm-up) at temperature 0 (the engine sends 0.01), then unload. VRAM is the whole card from `nvidia-smi`. Host RAM is the RSS of the backend plus the engine process. Windows shared GPU memory stayed at 0.1 GB (no spill).

| Context | Load | VRAM after load / peak | Host RAM | Prose tok/s | Code tok/s | Long prompt tok/s |
|---|---|---|---|---|---|---|
| 4K | 60 s (compile cache warm) | 20.8 / 20.8 GB | 5.3 GB | 52.9 | 52.9 | |
| 16K | 151 s | 20.9 / 23.2 GB | 6.6 GB | 52.9 | 52.8 | 49.1 (11.5K-token prompt; 29.3 on first sight) |
| 32K | 150 s | 20.7 / 23.0 GB | 6.5 GB | 52.9 | 53.1 | 47.8 (23K-token prompt; 19.6 on first sight) |

- vLLM's KV pool: 23,552 tokens at 4K, 47,104 tokens at 32K. The model is hybrid (16 attention layers of 64), so long context is cheap.
- Unload returns the card to 0.2 GB.
- Same-card reference for this model: llama.cpp Q4_K_M 43 tok/s and TensorFold native about 50 tok/s plain (sections above).
- Every reply began with reasoning text even with `thinkingMode: off`: the vLLM engine renders the chat template without `enable_thinking` (FU-172).

**DFlash 2 drafter on 24 GB: does not fit.**

| Attempt | Weights | KV budget | Result |
|---|---|---|---|
| App, 4K and 16K (bf16 drafter, share 0.92) | 20.21 GiB | -2.02 GiB | "No available memory for the cache blocks" |
| Standalone, fp8 drafter, 2048 batched tokens, share 0.93, 4K | 18.49 GiB | 0.18 GiB, 1.84 needed | refused |
| Same, 1024 batched tokens, 1K context | 18.49 GiB | 0.30 GiB, 1.60 needed | refused |

The recurrent-state blocks for draft verification need about 1.6 GiB even at 1K context, and CUDA reports only 22.35 of 23.99 GiB free on an idle card under WSL2. With the bf16 drafter the lane needs roughly 26 GB or more. The drafter pairing and config are correct: the app's load reached vLLM with `{'method': 'dflash', 'model': 'z-lab/Qwen3.8-27B-DFlash2', 'num_speculative_tokens': 7}`, and vLLM resolved `DFlash2DraftModel`, compiled `dflash_head` and `dflash2_candidate_selector`, and failed only at the KV allocation. On this card, DFlash on Qwen3.8-27B stays with TensorFold native (132 prose / 206 code tok/s through the app) and llama.cpp (78 to 112).

**Fixed (tests in `tests/test_dflash.py` and `tests/test_vllm_engine.py`):**
1. **Drafter not paired** (commit `56ea082d`): `get_draft_model(..., allow_dflash2=True)` matched only the exact id `Qwen/Qwen3.8-27B`. The catalog's own `Qwen/Qwen3.8-27B-FP8` row, every INT4/AWQ/W4A16 pack and local folders got no drafter, and the load fell back to plain decoding with "no compatible draft model is registered". It now matches the base name inside the ref.
2. **`num_speculative_tokens` missing**: vLLM 0.33 defaults it only from `n_predict` (MTP) or a DSpark block size. The app's `{"method": "dflash", "model": ...}` failed with "A speculative model was provided, but `num_speculative_tokens` was not provided" (reproduced). It is now the drafter's `dflash_config.block_size` minus one (7 for DFlash2, 15 for the b16 drafters), read from the drafter's `config.json`.
3. **`max_num_seqs` 256 on hybrid models**: "max_num_seqs (256) exceeds available Mamba cache blocks (31)" failed the plain 27B load at share 0.85. The engine now passes 8. That also cut CUDA graph capture at load from 224 s (51 graphs) to 3 s.
4. **GPU share**: vLLM refuses a load when share x total is more than CUDA's free memory ("Free memory on device (22.35/23.99 GiB) ... less than desired GPU memory utilization (0.97 ...)", reproduced at 0.97). The app passed nothing, so 0.92 failed whenever about 2 GB of the card was in use (the desktop held 2.7 GB earlier today). The engine now reads free memory through NVML at every load and lowers the share when needed (free less 1.5 GiB headroom: NVML read 23.38 GiB free where CUDA read 22.35 under WSL2). `CHAOSENGINE_VLLM_GPU_SHARE` overrides it. Nothing is fixed to one card.

**Not done or not verified:**
- DFlash speed-up and acceptance on vLLM (needs a 32 GB or larger card). Acceptance is not exposed in any case, because the in-process `LLM` runs with stats off.
- The Windows WSL-bridge engine (`vllm_wsl_engine.py`) still drops speculative decoding (not touched).
- No catalog row added: no 4-bit pack verified to load on stock vLLM with the drafter on 24 GB.
- No E2E check: the lane that runs (plain vLLM) already has the matrix cell, and the DFlash cell cannot run on 24 GB.
- The bf16-embedding copy is a test aid, not an app feature.

Disk: HF cache in WSL 16 GB + 3.6 GB, local copy 2.4 GB of new data, so 22 GB in all, inside the WSL `ext4.vhdx`. C: free went from 60.9 GB to 52.3 GB over this run (the vhdx grows and does not shrink; vLLM compile caches too). Nothing deleted.

## LTX-2.5 on Windows/CUDA (diffusers), first real run (2026-11-13)

Dev backend on port 8896 (the installed app held 8876), `.venv` torch 2.11+cu128 and diffusers 0.42.1, transformers 5.22.2 and bitsandbytes 0.50.2 from the GPU extras. **diffusers 0.42.1 loads the real checkpoint end to end; no upgrade needed.** Gemma 4 text encoder is `Gemma4UnifiedForConditionalGeneration` (11.96B; children `language_model`, `embed_vision`, `embed_audio`, `lm_head`); 4-bit load works with the skip list as written (`lm_head`, `embed_tokens`, `embed_vision`, `embed_audio` match; the other names match nothing and are harmless). `disable_mmap` is accepted with both quantised loads.

**Download** through `/api/video/download` (the app path, saved token): 66.2 GiB (71.1 GB), about 37 min at 26 to 30 MB/s. C: free 124.6 GB before, 49.8 GB after. Nothing deleted.

| Run (row `~nf4`, seed 42, 8 steps) | Load | Denoise | Wall | GPU peak (desktop 0.7 GB incl.) | Backend private peak | Commit peak |
|---|---|---|---|---|---|---|
| 512x320, 25 frames, cold | 124.5 s | 22.7 s | 149.3 s | 18.8 GB | 51.0 GB | 68.7 GB (limit 67.9, page file grew) |
| same, warm | 0 | 21.0 s | 21.5 s | 18.6 GB | 48.3 GB | 66.3 GB |
| 768x512, 97 frames, cold | 107.1 s | 62.4 s | 170.5 s | 20.1 GB | 51.4 GB | 69.3 GB |
| same, warm | 0 | 61.3 s | 62.2 s | 20.1 GB | | |
| `/api/video/preload` then 512x320 clip | 107.4 s (preload) | 21.6 s | 22.3 s | 18.8 GB | 51.0 GB | 68.7 GB |

Loaded and idle the backend holds 37.5 GB private (the offloaded components wait in system memory); the 51 GB peak lasts a few seconds while the pipeline is assembled. After `/api/video/unload`: private 4.2 GB, GPU 1.15 GB (memory returns). Minimum free RAM seen 22.9 GB. Every clip has a stereo 48 kHz AAC track, mean -28.6 / -32.3 dB, max -12.6 / -16.6 dB (not silent); frames looked at (first, middle, last): a fox moving through snow, coherent, no artefacts. LTX-2.3 4-bit for comparison: 70 to 80 s denoise at 768x512x97.

**Failed and fixed**
1. The download ended `failed`: "missing components: diffusion_decoder". `model_index.json` lists `diffusion_decoder`, which `LTX2Pipeline` in 0.42.1 does not take (and it needs NATTEN). `OPTIONAL_COMPONENTS_BY_REPO` now lists it with `prompt_enhancer`; re-requesting the download validated as complete with no new bytes.
2. `/api/video/preload` ignored the row's `loadQuantization`: preloading the 4-bit row would have read the bf16 repo (38 GB transformer + 24 GB encoder) into a 64 GB machine. Route, manager and engine now pass `use_nf4`; checked live (above: 18.8 GB on the card, then generate reused it, load 0 s).
3. The download bar was sized at 162.45 GB (the whole repo, stale shard sets and `transformer_full` included). A video or image row with pinned files now uses the catalog size (converted to GiB, as the bar counts); the Hub preview lists only top-level files, so it cannot be summed against the patterns.

**Catalog** `Lightricks/LTX-2.5-Diffusers~nf4` is now measured: `runtimeFootprintGb` 19.5 (offloaded generation peak), `loadMemoryGb` 38 (held while loaded), `estimatedGenerationSeconds` 62 (warm, 768x512, 97 frames), `footprintBasis` measured on cuda. The placement figure in `video_runtime/defaults.py` is now 27 GB (the whole pipeline in place: 18.1 GB of 4-bit models on the card after loading + 6.3 GB connectors + ~2 GB VAEs); offload is still chosen on cards up to ~40 GB. The bf16 row stays estimated. E2E check "LTX-2.5 on diffusers, 4-bit on load" now expects basis `measured`.

**Not verified:** the bf16 text-encoder branch (needs 48 GB free on the card; does not fit 24 GB, not attempted); the bf16 row; image-to-video on this row; E2E phase 5 itself not run (the live steps it makes were run by hand above). The commit peak passes the system-managed page file's current limit by ~1.5 GB while loading, so on a 64 GB machine the page file has to be allowed to grow.
## Installer rebuilt with today's fixes, installed, open items closed through the app (2026-11-13)

Built with `./build.sh` from `12288835` (mimalloc unload, Klein loader class, TriAttention, vLLM/DFlash, LTX-2.5), installed over the app (`/S`, exit 0), then rebuilt and installed twice more for the two fixes below (`e0467e43`, `36014c9b`). Every build: installer about 131.8 MiB, archive 133.5 MB, manifest `llamaServer` and `llamaServerTurbo` set, `warnings: []`, the four `install-*.sh` in the archive. Health in 17 to 33 s. Everything below ran through the installed app (backend on 8876, embedded runtime, extras overlay torch 2.14.1+cu130), one GPU job at a time. C: free about 49 GB throughout; nothing deleted.

| Item | Result | Evidence |
|---|---|---|
| FLUX.2 Klein 4B 1024x640, 4 steps, seed 1234 | **PASS** | Image looked at: a red fox sitting in snow at sunrise, sharp, no artefacts. No "weights not used / newly initialized" lines in this session's backend log (the same log file holds them for every earlier session). Load 36 to 50 s cold, 10.2 to 10.5 s reload; denoise **4.4 to 4.5 s on the first run after a load, 5.2 to 5.7 s on warm repeats** (dev backend: 4.6 s). |
| Unload returns memory (image) | **FAIL, fixed, PASS** | First install: after Unload the backend kept **16.5 GB private** (11.4 GB working set) from the first load on; later loads added on top, and one warm run grew private from 29.8 to 43.1 GB. Cause below. After the fix, three load / warm / unload cycles: 16.7 GB loaded, **3.2 GB after every Unload**. |
| LTX-2.5 4-bit text to video, 512x320, 25 frames, seed 42 | **FAIL, fixed, PASS** | First install: HTTP 500 "Gemma4VideoProcessor requires the Torchvision library" at the end of the load, and the failed load **kept 19 GB on the GPU and 21 GB private**, which Unload could not free. After the fixes: 200, load 183 s cold (132 s on a later cold load), **denoise 35.1 s / 33.0 s** (dev backend: 21 to 23 s, FU-173), peak GPU 19.1 GB, peak private 45.6 GB, min free RAM 22.7 GB. Stereo 48 kHz AAC, mean -47.6 dB, max -30.9 dB (not silent); first, middle and last frames: a fox walking through a snowy pine forest, coherent. |
| LTX-2.5 4-bit image to video (start = the Klein fox) | **PASS** | 200, 134 s load (the image-to-video class reloads the pipeline), 33.6 s denoise; frame 0 is the Klein still, then the fox gets up and walks, same light and scene. Sound: with a quiet prompt ("soft wind") mean -90 dB (near silent); with a loud prompt (howling wind, crunching snow, a crow, other seed) mean **-8.2 dB**, max 0 dB, 34.2 s wall warm. The audio path works; the level follows the prompt. |
| Unload after LTX-2.5, then the next generations | **PASS** | Video Unload: 36.6 GB private to **3.4 GB**, GPU 1.4 GB. Then Klein (200, 3.5 GB after Unload) and LTX-2.5 text to video again (200, not refused, 3.5 GB after Unload). |
| `e2e_test_suite.py --smoke` | **PASS, 7 of 7 phases** | 0, 3, 4, 5, 6, 7, 8. Phase 4 "memory returns after unload" PASS (3.5 GB before, 16.8 loaded, 3.5 after, twice). Phase 5 "LTX-2.5 on diffusers, 4-bit on load" **PASS live** (32.6 s). Skips: mflux rows and generate (Apple Silicon), sd.cpp FLUX.1 (GGUF not downloaded), LTX-2.5 on ltx-2-mlx (Apple Silicon), Wan burn check (MPS only), wave-2 speech models (none downloaded), Nemotron ASR (the portable lane has none). Report `~/.chaosengine/test-results/e2e-20261113-135204.md`. |
| The other phases (1, 2, 9, 10, 11, 16 to 19) | **PASS, 9 of 9** | Skips: MLX DFlash / DDTree / MTPLX / TensorFold-exact / Kolibri / LFM3 / prompt-cache (Mac, or no model on disk), GGUF MTP and the Qwen3.8 DFlash 2 sidecar (no such GGUF on disk), TensorFold 1.2.7 families (covered by the native check), settings correction (engine is tensorfold-native), 20 s instrumental (no music pack), captions from speech (needs `E2E_CAPTIONS_STT=1`). Report `e2e-20261113-135932.md`. |
| Qwen3-TTS CustomVoice 1.7B (`qwen3:Ryan`, calm, seed 7) | **PASS** | `notes: []`, 7.60 s of 24 kHz audio, peak 19071, identical across runs. Fresh backend, warm: **1.62 s per s of audio with the speed-up, 1.81 with `CHAOSENGINE_QWEN_FAST=0`** (11 % faster, as measured standalone). First call 23 s. On a worker started during the E2E suite the same request took 2.9 s per s; a backend restart brought it back (not explained). |
| GPU event log | **No new events during app runs** | Before: 18 LiveKernelEvent / AppTermFailureEvent today, the last at 08:28:49. This session: 5 x LiveKernelEvent 141 + 1 AppTermFailureEvent, all at **12:59:45**, when the standalone Python repro below started CUDA while the installed backend sat idle with its own CUDA context. Nothing during any generation through the app, the E2E suite or TTS. No APPCRASH. |

**Fix 1, memory kept after Unload (`e0467e43`).** The mimalloc options from `6d4b3fef` cover memory allocated by a thread that has since exited. Request handlers run on a pool of live worker threads: the worker that loaded the pipeline (and, with model CPU offload, moved the weights back to system memory after each run) was seldom the one that served Unload. Standalone, with the app's own Python and the installed backend code: Klein loaded on one live thread and unloaded on another kept **16.6 GB** (29.9 after a second cycle); load and unload on the same thread went back to **3.3 GB**; priming the unloading thread's heap did not help; the old bf16 path behaves the same (23.0 GB kept). Now `helpers/malloc_release.ModelThread` runs the image and video managers' `generate`, `preload` and `unload` on one dedicated thread per manager (Windows only; a call made on that thread runs inline; context variables travel with the call; `CHAOSENGINE_MODEL_THREAD=0` turns it off). Standalone manager test: 16.7 GB loaded, **3.4 GB** after Unload from another thread; switched off, 16.6 and then 29.9 GB kept. Tests: `ModelThreadTests` and the real measurement `LiveThreadMeasurementTests` in `tests/test_malloc_release.py` (1.26 GB allocated on one live thread and freed from another: 1.27 GB kept without, 0.01 GB with).

**Fix 2, LTX-2.5 on the installed app (`36014c9b`).** `model_index.json` lists `processor` (`Gemma4Processor`), whose video processor needs torchvision; the dev venv has torchvision, the GPU runtime does not. The processor serves only prompt enhancement, which the app does not use, so `LTX25_UNUSED_COMPONENTS` (`prompt_enhancer` and `processor` as `None`) now goes to every LTX-2.5 load, both rows, text and image to video (the bf16 row did not pass `prompt_enhancer=None` before either; not runnable on this card). Also: a failed video load re-raises without its traceback, after a collection, so components loaded before the failure are freed (the 19 GB GPU / 21 GB RAM case). Tests in `tests/test_video_ltx25.py` (the components; all three LTX-2 pipeline classes take them as optional) and `FailedLoadReleaseTests` in `tests/test_video_runtime.py`. pytest: malloc_release, image_runtime, video_runtime, video_ltx25, memory_gate, media_fit_cuda, backend_service, teacache, ltx2_mlx_runtime, mcp_bridge, storyboard_tools, job_queue pass.

**Not verified on this PC:** the LTX-2.5 bf16 row and the bf16 text-encoder branch (need 48 GB or more of VRAM); vLLM `thinkingMode` and the RedHat INT4 pack (need Linux, which was removed from this PC); FLUX.2 Dev NF4 (not downloaded); other cards and RAM sizes for the model-thread fix (an allocator matter, independent of the card). Open: LTX-2.5 and Klein warm denoise are slower on the installed runtime than on the dev venv (FU-173).

## FU-173: installed app slower than dev, root cause and fix (2026-11-13)

Same card, same clips (`klein.json`: 1024x640, 4 steps; `ltx_t2v.json`: LTX-2.5 `~nf4`, 512x320, 25 frames, 8 steps, seed 42), one backend at a time on 8876, the installed app closed for the standalone runs. "Standalone" is the installed runtime (embedded Python, extras overlay with torch 2.14.1+cu130, the installed backend code) started from a shell with the same `PYTHONPATH` the shell builds; "dev" is `.venv` (torch 2.11+cu128). diffusers 0.42.1 in both is byte-identical.

| Run | Klein warm denoise | LTX-2.5 warm denoise |
|---|---|---|
| Dev venv, torch 2.11+cu128, no UI | 4.74 s | 20.8 s |
| Installed runtime, torch 2.14.1+cu130, standalone, no UI | 4.72 s | 20.2 to 20.4 s |
| Installed app (its own UI polling) | 5.16 to 5.22 s | 22.1 to 22.3 s |
| Standalone + UI-like polling (`/api/metrics/gpu` 2 s, `/api/health` 1 s, `/api/workspace` 3 s) | 5.47 s | |
| same, `/api/metrics/gpu` only | 4.75 s | |
| same, `/api/health` only | 4.74 s | |
| same, `/api/workspace` only | 5.45 s | |
| **After the fix**, all three polls | **4.83 s** | **20.6 s** |
| After the fix, no polls | 4.77 to 4.80 s | |

**Not torch, not CUDA 13, not diffusers.** The runtime itself is as fast as the dev venv; the torch 2.11 swap was not needed. **Cause:** the UI polls `/api/workspace`, whose system snapshot lists LLM processes with `psutil.process_iter(["pid", "name", "cmdline", "memory_info", "cpu_percent", "ppid"])`. On Windows that opens every process on the machine: 0.44 s for 290 processes, holding the GIL, against 0.01 s for pid, name and command line. Every poll stalled the Python thread that queues the GPU work. The filter reads only name and command line, and `_describe_process` already fetches memory and CPU for the few matches, so the scan now asks for those three only (`helpers/system_processes.py`, `SCAN_ATTRS`). `/api/workspace` went from 0.56 to 0.15 s per call. Test: `RunningProcessListTests.test_scan_asks_only_for_name_and_command_line` in `tests/test_system.py`.

The 33 to 35 s LTX-2.5 denoise seen through the app earlier today did not come back (22 s through the app this time, before the fix); the scan cost grows with the number of processes and the polls with the number of open views, which is the best explanation, not proven. Remaining `/api/workspace` cost (0.15 s: `shutil.which` over `PATH` and the TurboQuant path probe) is mostly file-system calls that release the GIL; not measurable in denoise.

**Ships with:** the next installer build (backend code only, no runtime or pin change). Not yet checked through a rebuilt installer.
