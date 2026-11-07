---
title: "Progress: V-audio (sound effects)"
status: review
---
# Progress: V-audio (sound effects, D-27)

Updated: 2026-11-07 11:30. Owner: agent. Branch: `feat/sound-effects` (off `feat/home-studio`).

## Goal

Stable Audio Open Small as the **Sounds** mode of the Music Studio: opt-in, gated, never bundled, with a licence
screen. Short clips up to 11 s, optional seamless loop, saved to the Music Library as kind `sound`, sent to the
storyboard's effects lane, and an agent tool body `make_sound` (not registered here).

## Findings (checked 2026-11-07)

- Repo `stabilityai/stable-audio-open-small`, `gated: auto`, revision `ea2772170fc06f80dfd02bea6e57a44d9d4bc9d3`.
  Needed files: `model.safetensors` (1.68 GB), `model_config.json`. The repo also holds a second 1.7 GB copy
  (`model.ckpt`), the base model and 1.4 GB of mobile builds; the catalog fetches only the two files plus LICENSE.
- The T5-base text encoder (Apache 2.0, `google-t5/t5-base`, 0.9 GB) is read at run time; it is a download companion.
- Licence: Stability AI Community Licence (updated 2024-07-05), text read from the repo's LICENSE file.
- **No MLX build we can pin.** The community port `sandst1/stable-audio-mlx` has no licence (GitHub: NOASSERTION),
  no package, 7 stars, and its converter needs the user's token. So the engine is `stable-audio-tools` 0.0.20 (MIT)
  on PyTorch: the Apple GPU (MPS) when present, else the CPU.
- stable-audio-tools 0.0.20 declares Python 3.10 and old exact pins, so the pack lists its imports in
  `backend_service/sound_requirements.txt` (frozen from a clean 3.12 venv) and installs it `--no-deps
  --ignore-requires-python`. Importing the model builders also needs `pytorch_lightning` (found by running it, not in
  the package metadata).
- The weights are gated and no Hugging Face token was available to the build run (the owner's was not used), so the
  real model was **not run**. The engine was run end to end with a random-weight mini model built from a hand-written
  config through the real worker (CPU and MPS, 8 steps, loop and plain, step events, range check). The mini model
  proves wiring, not sound quality or speed.
- Live on port 8903 (throw-away HOME): the app's own sounds-pack install job ran to the end in 180 s on Python 3.11
  (it first failed on a requirements file frozen on 3.12: `networkx 3.7` needs 3.12; re-frozen on 3.11, which also
  resolves on 3.12). With the mini model placed in the checkpoints layout, e2e phase 15 passed all 6 checks including
  the full run (HTTP, queue slot, memory gate, worker in the sounds venv, real stable-audio-tools engine, tagged FLAC
  in the Music Library, audio route). A real download without a token answered with the plain gated message.

## Done

- Catalog family `stable-audio-open` (`catalog/music_models.py`): gated licence block, attribution, US$1M cap,
  pinned model and text-encoder revisions, `downloadPatterns`, `downloadCompanions`, estimates marked **estimated**.
- `music_sound_env.py` (own venv `~/.chaosengine/sound-venv`, marker, install job, T5 mirror),
  `music_worker_stableaudio.py` (engine), worker limits per engine (`SECONDS_RANGE`, `TASKS`), `MusicWorker(pack=...)`.
- Runtime and routes: `POST /api/music/sound/generate`, `/api/music/sound/install[/status|/cancel]`, runtime
  `sound` block, estimate per model range, library kind `sound` with `loop` and `licence`, FLAC tags (`kind`,
  `loop`, `licence`, prompt, seed, machine_generated), plain gated-download message (`helpers/hf_errors.py`).
- UI: Music | Sounds switch in the Studio header (shown only when the catalog has a sound row), `SoundsStudio`,
  `SoundPackNotice` under the model row, licence chip and screen reused, "Use in storyboard" sends kind `sfx`.
- `mcp_bridge/sound_tool.py` (`make_sound`), not registered.
- Tests: `tests/test_music_sound.py`, `src/features/music/__tests__/sounds.test.tsx`; e2e phase 15
  (`scripts/e2e_phases/phase_15.py`) passed live on port 8903.

## Not done

- A real run with the real weights (needs the owner to accept the licence on Hugging Face and save a token).
  Then: measure seconds per clip on MPS and CPU, peak memory, replace the `estimates` block (basis `measured`).
- CUDA and CPU-only lanes (the sounds pack refuses non-Apple machines, like the music pack).
- Registering `make_sound` in the bridge (the bridge stream adds `SOUND_TOOLS` / `SOUND_HANDLERS`, and the scope
  name `make_sound` to `JOB_TOOLS`).
- Non-English prompts (the model is English only).

## How to resume

1. `cd` to the worktree and `git status`; branch `feat/sound-effects`.
2. Tests: `.venv/bin/python -m pytest tests/test_music_sound.py tests/test_music_*.py -q`, `npx vitest run src/features/music`,
   `npx tsc --noEmit`, `npm run i18n:validate`.
3. Live: start the backend on a spare port with a throw-away HOME, then
   `CHAOSENGINE_PORT=<port> .venv/bin/python scripts/e2e_test_suite.py --phases 15`.
4. Real run: accept the licence at huggingface.co/stabilityai/stable-audio-open-small, save a token in Settings,
   install the sounds pack and download the model in Music > Models, then the phase 15 full-run check runs by itself.

## Decisions to propose

- Keep the sounds pack separate from the music pack (own venv, 1 GB), so a songs-only user never installs Stability code.
- Revisit an MLX build when Stability or a licensed community package ships one for Small (Stable Audio 3 has MLX).
