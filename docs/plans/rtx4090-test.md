# RTX 4090 test plan for v0.10.0

For the Claude Code session running on the Windows PC. Work top to bottom. After each item, write PASS, FAIL or SKIP and one line of evidence in `docs/progress/rtx4090-progress.md`. Commit that file and any fixes to branch `test/rtx4090` and push after each section. Do not push to `staging`, `main` or `release/v0.10.0`.

Related: [WSL_CUDA_TESTING.md](../WSL_CUDA_TESTING.md) (earlier CUDA lane run on WSL2, Ubuntu 24.04), [E2E_TESTING.md](../E2E_TESTING.md).

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

## 3. Music and sound

- [ ] ACE-Step 1.5: first install into the music venv, then one 30 s track. Time and VRAM.
- [ ] Stable Audio Open: one sound effect.

## 4. Image and video sizes (nothing is capped by the dev Mac)

- [ ] One image model at its largest listed size. Record VRAM.
- [ ] One video model at 720p, 1080p, and the largest size the Fits / Tight / Will not fit hint calls Tight. Record whether the hint matched what happened (FU-145).
- [ ] Frame rates 24, 30, 48, 60 in a Storyboard project: do the "converts from..." hints name the right native rate for the model used?
- [ ] Export a project at 1080p and at 2160p with the Upscale option.

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
