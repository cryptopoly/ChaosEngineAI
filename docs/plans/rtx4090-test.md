# RTX 4090 test plan for v0.10.0

For the Claude Code session running on the Windows PC. Work top to bottom. After each item, write PASS, FAIL or SKIP and one line of evidence in `docs/progress/rtx4090-progress.md`. Commit that file and any fixes to branch `test/rtx4090` and push after each section. Do not push to `staging`, `main` or `release/v0.10.0`.

Related: [WSL_CUDA_TESTING.md](../WSL_CUDA_TESTING.md) (earlier CUDA lane run on WSL2, Ubuntu 24.04), [E2E_TESTING.md](../E2E_TESTING.md).

## Start here (Windows, for the Claude Code session on the PC)

Read this section first, then work through sections 0 to 7.

**Ground rules**

- Work on branch `test/rtx4090`, cut from the latest `release/v0.10.0`. Push only that branch. Never push to `staging`, `main` or `release/v0.10.0`, never force-push, and never tag.
- The project rules in `CLAUDE.md` apply. Read `docs/hub/status.md` and `docs/hub/decisions.md` first. Never name other desktop AI apps in code, docs or commits.
- Run one test suite at a time, never several in parallel, and run Playwright with `--workers=1`.
- Never print or commit a Hugging Face token. If a model is gated, stop and ask the owner to accept the licence and sign in.
- Never kill a process you did not start.
- Record every item as PASS, FAIL or SKIP with one line of evidence in `docs/progress/rtx4090-progress.md`. Commit and push after each section. For a FAIL, add the log tail and the exact command.
- Fix small, clearly Windows-only bugs on `test/rtx4090` with a test. List anything bigger in the progress file for the Mac session instead of redesigning it.

**Windows commands** (PowerShell; the docs show macOS paths):

- Python is `.venv\Scripts\python` (macOS: `.venv/bin/python`). Create it with `py -3.11 -m venv .venv`, then `.venv\Scripts\python -m pip install -e .` with the extras the README names for CUDA, installed with `-c release-constraints.txt`.
- The app from source: `npm install`, `npm run stage:runtime`, `npm run tauri:dev` (README "Quick Start (from source)"). For sections 1 to 6 the backend alone is enough: `.venv\Scripts\python -m backend_service.app`, then open `http://127.0.0.1:8876/api/health`.
- Playwright E2E: `cd e2e`, `npm ci`, `npx playwright install chromium --only-shell`, then `npx playwright test --workers=1`.
- Use `nvidia-smi` for the GPU, driver and VRAM, and `nvidia-smi --query-gpu=memory.used --format=csv -l 1` for peak VRAM during a run.
- Linux is tested later in WSL2 (D-66): see [WSL_CUDA_TESTING.md](../WSL_CUDA_TESTING.md). Not in this pass.

**New since the plan was first written** (all on `release/v0.10.0`; the checks are in sections 2 to 4):

- Voice makeover (D-73):
  - Voice Models: Delete on speech and dictation models, a "Voice runtime" panel with add-on tick boxes, and "Remove voice runtime" under Troubleshooting.
  - Cancel and progress on a speech-model download, and "Show N sizes" on model families.
  - Speak: a Change voice drawer. Voices: a main-voices strip. Chat: a Voice popover.
- Chat read-aloud and calls use any installed voice (W6), with Kokoro as the fallback. Making Kokoro optional is D-75, which may land later.
- Sounds: Stable Audio Open 1.0 always renders its full 47 s window, so a 1 s clip costs as much as a 45 s one.
- Video Models: rows that are slow on the current machine carry a "Slow on this computer" note (D-44). On this PC with its NVIDIA GPU, no row should carry it unless a measurement says so.
- The eight closing improvements (D-84, FU-186, FU-188, FU-189; table rows 14 to 27 below; plan `docs/plans/v0.10.0-eight-improvements.md`, per-package notes `docs/progress/v0100-{j,k,l,m1,m2,n}.md`, each ends with what the Mac could not test):
  - Chat plan card survives a reload; running threads get a title; a refused tool call shows a card (D-85 replaces D-60).
  - Storyboard: keyboard roving through the shot list, primary pass button, time per still with a fast-model button, shot-count suggestion, redo-still refusal, drafts on the fastest installed still model.
  - Settings > Disk: a clean-up panel that deletes what the user ticks. **It deletes real data: on the PC use a throwaway data folder unless the owner says otherwise.**
  - Long projects: pause and resume across a restart, a song with sections and a cut plan, lyric captions, a freestyle Board tab, and Chat's `storyboard_song` tool.
  - Non-streaming replies on both engines wait for the longest reply the request allows (was a fixed 120 s).
- Windows things the Mac could not check: file locks on delete, killing the backend from Task Manager, non-ASCII paths and project names, the bundled ffmpeg and numpy under the packaged runtime, keyboard and screen reader behaviour in the desktop webview, and CUDA model choice for drafts.

## 0. Setup

- [ ] `git fetch` and check out `release/v0.10.0`, then branch `test/rtx4090`.
- [ ] Record: GPU model and driver (`nvidia-smi`), VRAM, RAM, Windows build, Python version, free disk.
- [ ] Create the venv and install as `README` says for Windows. Record any step that fails.
- [ ] Backend starts and `/api/health` answers. The Tauri shell is not needed for sections 1 to 6.

## 1. Automated checks (Windows native)

- [ ] `python -m pytest tests/ -q`. List every failure; mark the ones that are Windows-only.
- [ ] `npm test`, `npx tsc --noEmit`, `npm run i18n:validate`.
- [ ] `python scripts/e2e_test_suite.py --smoke`, then the full run. Save the report path.

## 2. Speech (Voice Studio)

For each: record the time for 10 s of speech, peak VRAM (`nvidia-smi`), and whether it plays.

- [ ] Kokoro: speak a paragraph.
- [ ] Qwen3-TTS CustomVoice 1.7B: speak with a preset speaker, with and without a style line. Save two WAVs with the same text, one neutral and one "angry". Report whether they differ in length and, if you can, in pitch or energy.
- [ ] Qwen3-TTS VoiceDesign 1.7B: speak with a description.
- [ ] Qwen3-TTS Base: clone from a short sample (own voice only), check the AI label and watermark appear.
- [ ] Qwen3-TTS 0.6B models: time against 1.7B. FU-148 asks whether they are faster on CUDA.
- [ ] Speed and pitch sliders; text over 400 characters (chunking); Cancel mid-run.
- [ ] Speech to text: transcribe a clip with the default model.

- [ ] Voice Models, Speech models table: start a download, check progress and GB shown, press Cancel; the row returns to Download, no stray download process is left in Task Manager, and Download again resumes.
- [ ] Delete a speech model: the confirm names it, its size and anything another model keeps; the folder under `%USERPROFILE%\.cache\huggingface\hub` is gone afterwards. Delete is refused while Speak is running.
- [ ] Delete a dictation model from its "···" menu.
- [ ] Voice runtime panel: parts and sizes are right, "In memory now" names the loaded model, Unload frees VRAM.
- [ ] Remove voice runtime (Troubleshooting), once with and once without the Kokoro tick: downloaded models stay, the voice venv folder is gone, reinstall works.
- [ ] Voices: set Speak, Chat read-aloud and Voice calls to different voices; Chat read-aloud uses the chosen one, and a voice that fails falls back with a note.
- [ ] Chat: the Voice popover shows the three choices and the settings; a voice call works end to end with a microphone.

### 2.1 Wave-2 speech models on the GPU (D-70 Phase 6, written on a Mac, never run)

Start the backend with `CHAOSENGINE_VOICE_UNMEASURED=1` so the unmeasured CUDA rows are offered. Record driver, CUDA and torch versions.

- [ ] Voice Models: the "More speech models on the NVIDIA GPU" add-on appears (it must not on a machine without nvidia-smi). Install it; the log ends with `speech gpu ok`. Record the install size of the voice venv before and after.
- [ ] If an import fails, note the missing package (the add-on installs `chatterbox-tts` and `voxcpm` with `--no-deps` plus a hand-picked list in `voice_env.SPEECH_GPU_PACKAGES`); add it there with its licence checked.
- [ ] Download each CUDA row: Chatterbox-Turbo (GPU), Chatterbox Multilingual V3 (GPU), VoxCPM2 (GPU), Orpheus 3B (GGUF). Check only the listed files arrive (the Chatterbox repo is 13.9 GB in all; V3 needs about 3.2 GB).
- [ ] Turbo: which decoder does `from_local` read, `s3gen.safetensors` or `s3gen_meanflow.safetensors`? Drop the other from `voice_models_cuda.py`.
- [ ] Speak with each (Turbo voice with `[laugh]`, Chatterbox in English and German with Expression 0.3 and 0.8, VoxCPM2 design, Orpheus `tara` with `[laugh]`). Japanese on Chatterbox: pykakasi is left out (GPL), so note whether Japanese still reads.
- [ ] Measure each with `.venv\Scripts\python scripts\measure-voice-models.py --rows chatterbox-turbo-cuda chatterbox-mtl-cuda voxcpm2-cuda orpheus-3b-gguf --json speech-4090.json` and also peak VRAM from `nvidia-smi --query-gpu=memory.used --format=csv -l 1`. Put the numbers in the rows' `estimates` (basis measured) so they are offered without the switch.
- [ ] Clone with each cloning row (own voice only): permission tick needed, AI label present, `GET /api/voice/clone/watermark/{id}` finds our AudioSeal mark after speed 1.3 and pitch +2.
- [ ] PerTh: on a Chatterbox clip after speed 1.3 and pitch +2, run `perth.PerthImplicitWatermarker().get_watermark(audio, sample_rate=sr)` in the voice venv; the AI tag of cloned Chatterbox speech must say `"builtinWatermark": "perth"`.
- [ ] Orpheus GGUF: llama-server starts on a free 127.0.0.1 port and stops when the voice worker stops (Task Manager / `ps` shows no stray llama-server); a 250-character chunk is not cut short.
- [ ] Same text through Qwen3-TTS and each wave-2 row, side by side.

## 3. Music and sound

- [ ] ACE-Step 1.5: first install into the music venv, then one 30 s track. Time and VRAM.
- [ ] Stable Audio Open: one sound effect.
- [ ] Stable Audio Open 1.0 at 2 s and at 30 s: both take about the same time, the run strip estimate matches, and the job does not time out. Record the time and peak VRAM.
- [ ] Music and Audio Library with 20 or more items: the page scrolls to the last card.

## 4. Image and video sizes (nothing is capped by the dev Mac)

- [ ] One image model at its largest listed size. Record VRAM.
- [ ] One video model at 720p, 1080p, and the largest size the Fits / Tight / Will not fit hint calls Tight. Record whether the hint matched what happened (FU-145).
- [ ] Frame rates 24, 30, 48, 60 in a Storyboard project: do the "converts from..." hints name the right native rate for the model used?
- [ ] Export a project at 1080p and at 2160p with the Upscale option.
- [ ] Video Models: note which rows, if any, show "Slow on this computer" and time one Wan and one LTX clip at the same size, so the NVIDIA rows get measured numbers.

## 5. Projects workspace

- [ ] Play through the flower-shop brief in Chat with studio tools on: plan card, approve, renders appear in the project.
- [ ] Cast: add a character with four pictures; run a shot with it.
- [ ] Mixer: voice, effect and music lanes; Export tab preflight; export MP4 and open it.
- [ ] Paste an image with Ctrl+V into a reference box.
- [ ] Open the project from a second app start (reload keeps it).

## 6. Text models on CUDA

- [ ] llama.cpp with CUDA: load a GGUF, chat, record tokens per second.
- [ ] vLLM lane if installed (see WSL doc).
- [ ] Cache strategies: `python scripts/cache-strategy-matrix.py --quick`.

## 7. Packaged installer (only after sections 1 to 6 are clean)

- [ ] Build or download the Windows installer for the release candidate.
- [ ] Install on a clean user profile. First launch, runtime extraction, sidecar start.
- [ ] Repeat: one speech item, one image, one project export.
- [ ] Uninstall leaves no running processes.

## Report back

Finish the progress file with: what failed (with logs), what is slower than expected, what the Mac values got wrong, and a one-line verdict per section.


## v0.10.0 all-singing: Windows pass

Branch `test/rtx4090`, recut from the finished `release/v0.10.0` (plan: `docs/plans/v0.10.0-all-singing.md` section 6). One suite at a time, Playwright `--workers=1`, PASS / FAIL / SKIP with one line of evidence in `docs/progress/rtx4090-progress.md`.

| # | Test | Expected |
|---|---|---|
| 1 | `pytest tests/ -q`, `npm test`, `npx tsc --noEmit`, `npm run i18n:validate`, full `scripts/e2e_test_suite.py`, `npx playwright test --workers=1` | No new failure against the last Windows run; the new checks (border setting, clone model, estimates, TensorFold report, music refusal, territory gate) pass |
| 2 | Build the installer, install over the app, start it | Starts; slow-start splash shows the plain note with no backend log lines (R2) |
| 3 | Settings, Accelerators: install the CUDA llama-server at `b12317` on a clean `~/.chaosengine/bin/llama-cuda/` | Archives match `_PINNED_SHA256`; a tampered archive is refused; `llama-server --list-devices` lists the 4090; a GGUF chat runs on the card |
| 4 | Storyboard shot editor: pick an end frame, render an LTX-2.5 clip with start and end stills | The last frame matches the end still; the agent `storyboard_shot` `end_image` action does the same |
| 5 | FU-155 / FU-158: remove the voice runtime, reinstall, tick the speech GPU add-on (section 2.1) | Log ends `speech gpu ok`; each CUDA row downloads only its listed files and speaks; VRAM and speed recorded into `estimates`; unticking removes only the listed packages and Kokoro, dictation and Qwen3-TTS still work; no stray llama-server |
| 6 | Voice call in Chat with a microphone (Kokoro, then a GPU voice) | Chosen voice heard, barge-in works, fallback note on a failing voice |
| 7 | New cloned voice form with two cloning models downloaded | Both listed with costs; the default `qwen3-tts-1.7b` (the `-cuda` twin on this host) is preselected and tagged Default, Save is enabled; picking the other model saves that model on the voice; a GPU too small for the default falls to a downloaded runnable model |
| 8 | Image and Video Studio estimates: Wan TI2V at 832x480x49 and at the reference, Klein at 1024 and 2048, LTX-2.5 at 49 and 97 frames | Shown time within 20 % of the run; report the error per case so the CUDA image and LTX constants can be set (below) |
| 9 | Chat with Studio tools: 4-shot storyboard, leave it waiting | One job card, time left from the first clip (basis `catalog` until a clip is measured), summary at the end |
| 10 | TensorFold: `check-upstreams.py`; status route family report; then, if weights are at hand, run one model each of `dory`, `kolibri1`, `lfm2_vl`, `nemotron_h_omni` | No uncurated family reported; each family run gets a `NATIVE_MODELS` row with measured VRAM, or its note says why not |
| 11 | Music and Sounds screens | The sentence "Music generation runs on Apple Silicon Macs in this version. Windows and Linux support (NVIDIA GPUs) is planned; nothing is installed or downloaded on this computer." shown once on Music Models and in both studios; each ACE-Step row reads "Not available on this computer yet"; no Install or Download can start (also `POST /api/music/download` is not refused server-side yet, FU-139: note what happens) |
| 12 | Stronger borders switch (Settings > General > Accessibility) | Fields and buttons visibly outlined by default; the switch reverts them without a reload; no layout shift |
| 13 | Territory gate (only if `licence_gate.H3_RELEASED` is true; otherwise SKIP) | Rows hidden until the tick; download refused through the API without it. **Do not download or run H3 in the UK.** |
| 14 | Storyboard keyboard use (item 2): a 30-shot project, Tab through the list | One Tab stop for the list; arrows, Home, End and Enter work; Tab goes into the current card's controls; on Windows with a screen reader (Narrator) card names are read |
| 15 | Draft then final (items 3, 4 and FU-189): with only a slow still model saved, open a fresh planned film | Make drafts is the primary button; the defaults panel shows the time per still and "Use the fast model"; drafts use the fastest installed still model and the estimate prices that model; on CUDA, rows with no catalogue time come last, so note which model it picks and whether that is what you would pick |
| 16 | Plan tab shot-count suggestion (item 5) and redo-still refusal (item 7) | "A 30 s ad is about 6 to 8 shots" for a 30 s brief; redo still with ids on a shot with no take is refused in the same words as the form without ids |
| 17 | Settings > Disk > Clean-up panel (item 6): tick one test project, one chat thread and one gallery picture, confirm | Count and size named; only the ticked items go; nothing is ticked at the start; **Windows file locks**: a picture open in another program, a project folder open in Explorer, and a project with a running batch are refused with a readable reason, not a stack trace; never run it against data you want to keep |
| 18 | Chat plan card after a reload (item 1): ask for a 4-shot storyboard, wait for the card, press F5 (and Ctrl+R in the desktop app) | The card is still there and answering it renders; Stop cancels the backend turn; a running thread shows its first message as the title; a refused call (for example `what: "stills"`) shows a "Not started" card |
| 19 | Pause and resume (item 8): start a clip batch of 3, press Pause; then start another and **kill the backend process from Task Manager** mid-step, restart the app | Pause stops after the current step and the bar says Paused; after the kill the batch comes back Paused with the reason "Paused when the app closed"; Resume finishes the remaining steps under the same batch id; the cut-off step is made again; Cancel works on a paused batch |
| 20 | Song (item 8): attach a WAV, an MP3 and an M4A (and a Music library track if one exists), then edit a section label | Seconds, bpm, sections and the rough-analysis note appear; the bundled ffmpeg and numpy are used on Windows (no console window flashes); a text file renamed `.wav` is refused; a path with spaces and non-ASCII characters works |
| 21 | Cut plan and lyric captions (item 8): "Make a cut plan", review, apply, Undo; type lyrics and time them from the sections; then, with a speech-to-text model installed, time them by ear | Shots follow the sections and are capped at the video model's clip length; Undo restores the old shots in one step; captions reach the exported video and the `.srt`; the by-ear option is disabled with a plain reason when no model is installed |
| 22 | Freestyle Board tab (item 8): drop pictures, drag cards, pan, zoom, select several, Order these, Undo | Order is left to right then top to bottom, one Undo restores it; the per-card chat reworks that shot's prompt; the board chat adjusts several shots; the song ruler runs along the bottom; arrow keys move a selected card by 8 px and Shift+arrows by 64 px; works with a mouse and with a touch screen if the PC has one |
| 23 | Bundle of a project with a non-ASCII name (the sample film is named in the user's language): export, download, import | Works on Windows paths; the song file is in the bundle |
| 24 | Chat tool `storyboard_song` (new in Chat): "cut this track into a 30 s ad" with a Music library track or an `upload:` handle | Attach, cut plan and apply work from Chat; applying shows as one Undo in the app; speech timing is not offered in Chat |
| 25 | Long non-streaming replies (FU-186): Chat with Studio tools on and a GGUF model that thinks for a long time (Qwen3.8-27B or similar), a request whose reply takes more than 2 minutes | The reply arrives instead of ending at 120 s; the model stays loaded |
| 26 | Path import check (FU-186): set a brand logo from a text file renamed `.png` | Refused with "Not a picture"; a real PNG, JPEG and WebP work |
| 27 | Speech and tokenizer repos in model lists (FU-186): with a chatterbox, orpheus, VoxCPM or S3Tokenizer repo in the cache | None of them appears in Chat's or the Plan tab's model list |

| 28 | CLI with auth on (S-O): leave `requireApiAuth` on, run `chaosengine-cli` commands and `scripts/e2e_test_suite.py --smoke` | No 401: the CLI reads the loopback session token after a 401 by itself, never prints it, never puts it in a URL |
| 29 | Unsaved text (S-O): in Plan, Cast, Brand, Mixer and section-label fields, cause a refused save (edit the same project in a second window, or stop the backend), keep typing | Your text is not lost; "Your text was not saved" shows with **Put it back** (after a conflict) or **Retry** (backend did not answer) |
| 30 | Plan tab keyboard (S-O): Tab through a 19-shot Plan | One Tab stop per shot row (about 6 stops, not about 95); arrows move inside a row |
| 31 | Song longer than shots (S-O): attach a 4-minute track, set Length to 15 s, "Make a cut plan" | The note offers a cut plan and a "Use the song's length" chip; the plan covers only the Length (S-P: `cut_plan` takes `seconds`, else uses Length); one Undo restores |
| 32 | Cut-plan clip length (S-P): render a 15 s cut plan with LTX on CUDA | Final MP4 is 15.000 s (clips round up, then are trimmed); cuts land on the beat within one frame. On the Mac every clip was about 0.3 s short before the fix |
| 33 | Lyric timing and notes (S-P): time 84 lyric lines from sections | No two captions overlap; the note separates "lines the film never reaches" from "lines waiting for clips". Known gap: Export says "84 captions" for a 15 s film that shows about 6 |
| 34 | Chat Studio tools and the prompt watchdog (S-P): a Studio-tools turn on a slow GGUF model | The 60 s prefill watchdog does not cancel the turn; Stop still works. A project name given as `project_id` is refused with the real id so the model can retry |
| 35 | Pause on a real batch (S-P, passed on Mac with Iris-3B Turbo): pause, resume, cancel a running batch of 6 drafts on a CUDA image model | "Pausing after this step…" at once; bar reads "Paused · N of 6 steps done"; Resume keeps the batch id; Cancel on a paused batch keeps finished stills; a pause on the last step ends the batch as done |
| 36 | Touch (S-P, passed in the iOS 27 Simulator only): on a touch screen, tap, long press, drag a card, pinch zoom on the Board, two-finger pan, scroll the shot list | Taps are 44 px on touch screens; long press does not select text; two fingers moved together pan, they do not zoom. Windows touch and pen were not tested |
| 37 | Compact layout (S-Q): shrink the app window below 1180 px (or use a tablet) | 56 px top bar with a Menu button; the nav opens as a drawer (Escape, backdrop, choosing a tab close it; Tab stays inside); no sideways scroll; at 1180 px and wider the sidebar column is unchanged; Chat thread list takes at most about a third of the height. Only Chromium emulation was tested; check the real Tauri window on Windows and the light theme |
| 38 | FP8 KV (FU-178): `llama-server -m <8B Q4_K_M> -c 32768 -fa on -ctk f8_e4m3 -ctv f8_e4m3`, one 400-token reply, compare with q8_0 | Loads, KV about half of f16. If it works and is not slower than q8_0, add an FP8 option gated to CUDA (`_CUDA_ONLY_CACHE_TYPES` already lets it through). Off CUDA the app turns FP8 into f16 with a note |
| 39 | Native KV type on CUDA (FU-178): E2E phase 1 `GGUF native q8_0 KV cache`, then the app's "KV cache type" slider (f16 / q8_0 / q4_0) on a GGUF model | Load reports cacheBits 8 / 4; the memory check halves / quarters the KV (Mac measured 4886 / 2728 / 1575 MB; decode 53.7 / 55.3 / 51.0 tok/s) |
| 40 | DFlash 2 through the app (FU-178): download `lmstudio-community/Qwen3.8-27B-GGUF` (now brings the z-lab Q8_0 drafter, +2.1 GB), chat with speculative decoding on | Note names `Qwen3.8-27B-DFlash2-Q8_0.gguf` and `draft-dflash`; acceptance shown; Mac measured 55 to 59 tok/s, 3.8x plain, 5.44 accepted per round |
| 41 | MTP vs DFlash 2 on CUDA (FU-178): ggml-org Q4_K_M with its `mtp-` and `dflash-` sidecars; the app now picks dflash; also run once with only the mtp- sidecar | The block drafter is also ahead on CUDA; if not, the order needs a per-platform rule |
| 42 | Acceptance on the GGUF lane (FU-178): E2E phase 1 `GGUF baked-in MTP head + draft acceptance` | `dflashAcceptanceRate` greater than 0 (Mac: 2.96 with MTP N=3, 2.2x plain) |
| 43 | MoE expert cache (FU-178): a MoE that does not fit 24 GB (Qwen3.6-35B-A3B Q8_0 or Nemotron 3.5 Lightning 30B-A3B Q8_0): `--fit on` vs `-cmoe --moe-cache-mib 12000 -lm none` | tok/s, load time, VRAM, RAM; decide whether to build the Fit rule (design in `docs/progress/v0100-fu178.md`) |
| 44 | New pins on CUDA (upstream bumps 2026-11-16): llama.cpp b12441 CUDA 12.4 and 13.5 archives, TensorFold 1.2.9, sd.cpp `532d2c1` | Installs and runs; Windows CUDA digests were checked against release metadata only; the CUDA capabilities fixture is still assembled by hand (see "Also for this pass") |

Not covered anywhere: Windows-only hazards (path length, Defender, console windows), pen input, VoiceOver, and the Tauri webview. The iOS 26.3 Simulator crashes Safari's web process on the app with a JavaScriptCore JIT assertion (a WebKit bug); a real iPad on iOS 26 might hit it.

### Also for this pass

- **Real CUDA capabilities fixture (R9, FU-168).** `tests/fixtures/tensorfold_caps_cuda_1_2_7.json` was pieced together from the v1.2.7 source and the GB10 receipts, so some format lists (`spark2_5`, `minimax_m3_vl`, `gemma4` on CUDA) are guesses. Run the installed binary's `capabilities --json` on the 4090, save the output verbatim in `tests/fixtures/` (name it for the binary version, for example `tensorfold_caps_cuda_win_<version>.json`), point `tests/test_tensorfold_family_report.py` at it in place of the assembled file, and run `pytest tests/test_tensorfold_family_report.py tests/test_tensorfold_native.py -q`. Note any family it lists that no row names.
- **FU-165 constants still to measure.** `estimateRef` rows for CUDA images and for LTX-2.5 / LTX-2.3 on CUDA were not measured (the LTX constants are the Mac `ltx-2-mlx` fit: `fixed` 0.1, `quad` 1e9, `ignoresSteps`). From row 8, record per case: shown seconds, real seconds, error. Then set `fixed`, `quad` and `ignoresSteps` per row in the image and video catalogues, add the cases to `tests/fixtures/time_estimate_cases.json`, and update `docs/research/time-estimate-fit-2026-11.md`.
