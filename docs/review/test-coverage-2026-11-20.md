# Test coverage of the Windows / RTX 4090 pass (2026-11-20)

The owner's question: have tests been added to the suite for all new functionality? This audit lists every feature
built on `test/rtx4090` since the Windows pass began (handover 2026-11-18; `git log origin/release/v0.10.0..HEAD`
plus the release-branch commits from `3be6e496` on, and `docs/progress/rtx4090-progress.md`), then checks four test
layers by grepping the trees, not from memory:

- **pytest**: `tests/test_*.py`
- **vitest**: `src/**/*.test.ts(x)`
- **Playwright**: `e2e/tests/*.spec.ts`
- **E2E**: a check in `scripts/e2e_test_suite.py` or `scripts/e2e_phases/*.py` (the CLAUDE.md new-feature gate)

Verdicts: **covered** (unit tests and an E2E check, plus a browser spec where there is a UI flow), **thin** (some
layer the gate asks for is missing), **missing** (no test at all).

## Summary

| | Covered | Thin | Missing | Rows |
|---|---|---|---|---|
| Before this audit | 12 | 25 | 1 | 38 |
| After this audit | 25 | 13 | 0 | 38 |

What was added (all run, results below): 12 E2E checks (new file `scripts/e2e_phases/rtx4090_pass.py` plus two in
`scripts/e2e_phases/lipsync.py`), 1 Playwright spec with 5 tests (`e2e/tests/42-rtx4090-pass.spec.ts`), spec 24
brought up to the new H3 terms, 1 pytest case for the extras-folder fix (`tests/test_setup_routes.py`), and
`tests/test_e2e_rtx4090_checks.py` (10 cases) holding the new E2E checks' own logic.

The 13 rows still thin need hardware, a real download or install, a second backend, or a Mac (FU-430 to FU-441).
None is untested: each has unit tests; what it lacks is the E2E layer.

## Matrix

"Before" is the state at `8e331f8c`; the last column says what this audit added. Phase numbers are the suite's.

| # | Feature (commit or row) | pytest | vitest | Playwright | E2E | Before | After | Added |
|---|---|---|---|---|---|---|---|---|
| 1 | `/v1/chat/completions` streams tool calls (FU-258, `61c6e846`) | `test_backend_service.py` (stream + text-written calls) | n/a | n/a | phase 1 "/v1 chat completions stream tool calls on GGUF" | thin | covered | E2E check; passed live on Qwen3 0.6B GGUF through llama.cpp |
| 2 | Tool-call repair, lone backslash advice (FU-255, `ec51d5f9`) | `test_agent_call_repair_paths.py`, `test_agent_call_repair.py` | n/a | n/a | none | thin | thin | FU-438 |
| 3 | Plan card prices the settings that will run, Expert fields, `POST /api/chat/studio/price` (FU-240, `02413fa1`, `29e09e56`) | `test_chat_studio_tools.py`, `test_studio_tiers.py` | `studioChat.test.tsx` | spec 15 | phase 7 "plan card price route prices a render as typed" | thin | covered | E2E check |
| 4 | Draft, Standard, Final quality tiers (FU-241) | `test_studio_tiers.py` | `studioChat.test.tsx` | spec 15 | phase 7 "Chat quality tiers" | covered | covered | none; this check **fails** on this PC (FU-430) |
| 5 | Studio staging: Storyboard tools only for a film turn; ads and stated lengths (FU-231 part 2, `3359a5f9`, `14fd93c7`) | `test_studio_staging.py` | n/a | n/a | phase 7 "Chat Storyboard tools only for a turn about a film (in process)" | thin | covered | E2E check (in process: the turn's tool list is not visible over HTTP) |
| 6 | MLX tool block prints allowed values and limits (FU-231 part 1, `dd86468c`) | `test_tool_prompt_limits.py` | n/a | n/a | none | thin | thin | Mac-only gap (FU-437) |
| 7 | Windows Job Object ends workers when the backend is force-killed (`bdc369c0`) | `test_windows_job.py` (real `taskkill /F`, Windows only) | n/a | n/a | none | thin | thin | FU-431 (needs a backend the suite may kill) |
| 8 | Storyboard: a Windows path over 260 characters is a plain 400 (`df1ff8c8`) | `test_storyboard_path_errors.py` | n/a | n/a | none | thin | thin | FU-439 |
| 9 | A brand-new extras folder is on `sys.path` without a restart (`380cbd27`) | added: `test_setup_routes.py::test_a_brand_new_extras_folder_is_put_on_sys_path_before_the_reprobe` | n/a | n/a | none (phase 6 never installs, by design) | missing | thin | pytest case |
| 10 | Accelerators: install error shown, kvpress without deps, metadata version, vLLM and TriAttention refused on native Windows (`9f51918a`, `7e5a6a66`, `c1b4b59e`) | `test_setup_routes.py`, `test_accelerator_versions.py`, `test_accelerator_capabilities.py` | `acceleratorCatalog.test.ts` | spec 41 | phase 6 "fu-056-capability-flags", "vLLM and TriAttention refuse at once on native Windows" | thin | covered | E2E check (Windows only; skips elsewhere, where the call would install) |
| 11 | Nunchaku rows: pack plus base parts; a row not on disk refuses with 409, no placeholder (`6d343d34`, `b2bd1093`) | `test_nunchaku_download.py` | `imageAdapter.test.ts` | n/a | phase 4 "a Nunchaku row that is not downloaded refuses with a plain 409" | thin | covered | E2E check |
| 12 | FLUX offload by VRAM; full GPU when the card holds it (`14fd93c7`, `1c76b241`) | `test_image_cpu_offload.py` | n/a | n/a | phase 4 generate runs, offload not asserted | thin | thin | FU-434 |
| 13 | FLUX 3 Klein (diffusers, sd.cpp), Qwen-Image 2.1 Turbo, MOSS-Image, Kandinsky 6 rows (FU-273) | `test_new_media_rows.py`, `test_image_catalog_listing.py`, `test_video_runtime.py` | n/a | n/a | phase 4 and 5 `new_media_rows` (live when installed) | covered | covered | none |
| 14 | sd.cpp Windows CUDA build and staging (`9067494f`, `b2bd1093`) | `test_sdcpp_image.py`, `test_stage_runtime_scripts.py`, `test_check_upstreams.py` | n/a | n/a | phase 4 "generate via sd.cpp FLUX.1", sd.cpp rows in `new_media_rows` | covered | covered | none |
| 15 | sd.cpp device label read from the binary (cpu or cuda) | `test_sdcpp_image.py` | `VideoStudioTab.test.tsx` | n/a | none (no route exposes the sd probe) | thin | thin | FU-436 |
| 16 | WSL vLLM bridge preflight: C compiler and Python.h, root cause first (`f1d6d882`) | `test_vllm_wsl_prereqs.py`, `test_vllm_wsl_install.py` | n/a | n/a | phase 6 "vllm-wsl-status" (status shape only) | thin | thin | FU-435 |
| 17 | WSL vLLM bridge: GPU share, reasoning split, DFlash, library folder, local-folder name (`2d9f4abf`, `d8461064`) | `test_vllm_wsl_engine.py`, `test_accelerator_capabilities.py` | n/a | n/a | none | thin | thin | FU-435 |
| 18 | New chat GGUF rows: Gemma 4 QAT 12B / 26B with mtp drafters, Gemma 26B Q8_0, Lightning Q4_0, Qwen3.6 35B MTP IQ4_XS (`af2b6bae`, `511777aa`, `44cdd00f`) | `test_text_gguf.py`, `test_catalog_text_families.py` | n/a | n/a | phase 0 "gemma-4 QAT GGUF rows", "new model families" | covered | covered | none |
| 19 | Chat follow-ups: Qwen3.6 35B re-pinned to Q4_K_M, Bonsai 2 Q2_0, 31B drafter (FU-263, 267, 269, `cd802a6b`) | `test_catalog_text_families.py`, `test_text_gguf.py` | n/a | n/a | phase 0 "chat follow-up rows" | thin | covered | E2E check |
| 20 | MTP draft length 2 on the GGUF lane (FU-268) | `test_inference.py`, `test_catalog_text_families.py`, `test_llama_fu178.py` | n/a | n/a | phase 1 "GGUF MTP speculative" runs, the draft length is not asserted | thin | thin | FU-441 |
| 21 | Nemotron Lightning loads with a 16,384 minimum context and says so (FU-265) | `test_backend_service.py::test_a_row_with_a_min_context_loads_larger_and_says_so` | n/a | n/a | phase 0 "chat follow-up rows" (every Lightning row carries it) | thin | covered | E2E check |
| 22 | One mmproj per repo, `MTP/` folder sidecars (FU-264, FU-266) | `test_text_gguf.py`, `test_mmproj_vision.py`, `test_inference.py` | n/a | n/a | phase 1 GGUF checks load real files, the sidecar choice is not asserted | thin | thin | FU-441 |
| 23 | Speech to text on the GPU (stt-gpu add-on: Qwen3-ASR, Parakeet, Nemotron ASR, Granite, Transcribe) | `test_voice_stt_gpu.py` | `voiceModels.test.tsx` | n/a | phase 8 "stt on the NVIDIA GPU" | covered | covered | none |
| 24 | Word timings for the GPU dictation rows (FU-270) and pinned revisions (FU-271) | `test_voice_stt_gpu.py`, `test_check_upstreams.py` | n/a | n/a | phase 8 "stt on the NVIDIA GPU" (words in order, or the aligner named) | covered | covered | none |
| 25 | Word timings row: the forced aligner as its own download (D-101) | `test_voice_stt_gpu.py` | `voiceModels.test.tsx` | added: spec 42 | phase 8 "Word timings row where the GPU dictation rows are" | thin | covered | E2E check, Playwright test |
| 26 | ACE-Step 1.5 on CUDA, memory gate on host RAM, Windows worker hang, offload by free VRAM (FU-139) | `test_music_worker.py`, `test_music_routes.py`, `test_music_env.py`, `test_music_catalog.py`, `test_music_client.py` | `musicComponents.test.tsx`, `sounds.test.tsx` | spec 04 | phase 11 lane check and a real 20 s track | covered | covered | none |
| 27 | Stable Audio 3 and the CUDA sounds lane (FU-274) | `test_music_sound.py`, `test_music_sound_1_0.py`, `test_music_sound_steps.py` | `sounds.test.tsx` | n/a | phase 11 "a real 5 s Stable Audio 3 sound on CUDA" | covered | covered | none |
| 28 | Resume paused Hub downloads from the partial file (FU-275) | `test_download_resume.py` | n/a | n/a | none | thin | thin | FU-432 |
| 29 | ACE-Step XL refused with a plain message on a shared GPU (FU-278) | `test_music_worker.py` | n/a | n/a | none | thin | thin | FU-433 |
| 30 | Copy now for music weight files (D-100, FU-276) | `test_music_weight_copy.py`, `test_music_env.py` | `weightCopy.test.tsx`, `music.test.ts` | added: spec 42 (offer, progress, Done and retry; Cancel) | phase 11 "Copy now: weight-copy status fields, Cancel with nothing running" | thin | covered | E2E check, 2 Playwright tests |
| 31 | Voice: a PC shows only what it runs, Mac-only fold line, dictation panel guidance, fit chips, download bar (D-101, D-102 A/C/D) | `test_voice_stt_gpu.py`, `test_voice_qwen_cuda.py` | `voiceModels.test.tsx`, `VoicesScreen.test.tsx` | spec 21 (mocks pinned); added: spec 42 (fold line, GPU row name) | phase 8 "speech rows carry their lanes" | thin | covered | E2E check, Playwright tests |
| 32 | Chatterbox, VoxCPM2 and Orpheus on the GPU (D-102 B) | `test_voice_cuda_lane.py`, `test_voice_clone.py`, `test_voice_watermark.py` | `voiceModels.test.tsx`, `speakLogic.test.ts` | spec 18 | phase 8 "speak and clone with Chatterbox, VoxCPM2 and Orpheus on the GPU" | covered | covered | none |
| 33 | MiniMax H3: tick accepts licence and AUP, hosted-service notice, live licence link, outputs tagged machine-generated (D-99) | `test_licence_gate.py` | `Licence.test.tsx` | spec 24 (updated to the new sentence, notice and link) | phase 5 "MiniMax H3 terms" (in process while H3 is unreleased) | thin | covered | E2E check, spec 24 update; the tag on a real H3 video waits for release (FU-440) |
| 34 | Lip-sync engine, LTX-2.3 audio to video (D-103) | `test_lipsync.py` | `storyboardComponents.test.tsx` | n/a | phase 22 (catalog figures, refusal, a real 5 s shot) | covered | covered | none; the Mac path is untested (FU-370, Mac-only) |
| 35 | Storyboard vocal clip for a shot (`set_vocal_clip`, drawer field) | `test_lipsync.py` | `storyboardComponents.test.tsx` | added: spec 42 (attach, player, remove, real backend) | phase 22 "a Storyboard shot takes a WAV vocal, refuses a text file and clears it" | thin | covered | E2E check, Playwright test |
| 36 | `make_lipsync_shot` agent tool | `test_lipsync.py` | n/a | n/a | phase 22 "make_lipsync_shot is on the agent bridge and refuses a missing still in words" | thin | covered | E2E check |
| 37 | Speaker labels on the GPU dictation rows (`e7301f19`) | `test_voice_stt_gpu.py` | n/a | n/a | phase 8 (two-voice clip per speaker turn) | covered | covered | none |
| 38 | CUDA llama.cpp archive b12441 lane, GGUF turns on after the CUDA install (`99804088`) | `test_llama_cuda.py`, `test_llama_cuda_windows.py`, `test_llama_cuda_prune.py` | n/a | n/a | phase 1 "CUDA llama-server lane (Windows and Linux)" | covered | covered | none |

## Runs (2026-11-20, Windows 11, RTX 4090)

- pytest (throwaway HOME): `tests/test_e2e_rtx4090_checks.py` and `tests/test_setup_routes.py`, all pass.
- `npx tsc --noEmit`: clean. `npx vitest run` on music, voice models, storyboard components and the licence screen: 8 files, 167 tests pass.
- Playwright (own ports, throwaway servers): `42-rtx4090-pass` 5 of 5, `24-territory-gate` 2 of 2.
- `scripts/e2e_test_suite.py --smoke` against a throwaway backend (HOME and USERPROFILE under `E:\tmp`, port 8981): phases 0, 3 to 8 pass, exit 0.
- `--phases 5,11,22` against the same backend: pass (the model parts skip without weights).
- The phase 1 stream check alone, against a throwaway backend with the installed llama-server and Qwen3 0.6B Q4_K_M (GPU lock held): pass, the call arrives as a `tool_calls` delta and the stream ends with `tool_calls`.
- `--phases 7` against a throwaway backend reading the real model cache (no generation): the new checks pass; the existing **"Chat quality tiers" check fails**: `FLUX.1-schnell-nunchaku-int4 draft: time 0`. A Nunchaku draft is under half a second and rounds to 0, which the check (and a plan card showing "0 s") treats as no estimate. FU-430.

Not run: the full suite and phase 1 as a whole (the owner runs them at the end), and anything needing a Mac.

## Mac-only gaps

- MLX tool block limits (row 6): the block is printed by the MLX worker; a live check needs an MLX model on a Mac.
- Lip sync through `ltx-2-mlx a2v --distilled` (row 34, FU-370): wired, never run.
- Phase 1's MLX checks (unchanged by this pass) skip here by design.
