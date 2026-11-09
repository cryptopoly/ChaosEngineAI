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
