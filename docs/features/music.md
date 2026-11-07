# Music: songs and instrumentals with ACE-Step 1.5

ChaosEngineAI makes songs with vocals and lyrics, or instrumentals, 10 seconds to 10 minutes long, on this
computer. This page covers the backend. For the Models, Studio and Library screens, see
[Music](../usage/music.md). Measurements are from the P0 spike (`docs/plans/music-spike.md`) and the P1 check
on an M4 Max with 64 GB.

## What you need

- **Apple Silicon.** The engine runs on MLX. On Windows and Linux the music worker starts and answers
  "engine unavailable" with the reason; the CUDA lane is later work.
- **The music pack** (Music > Models, `POST /api/music/install`): its own environment in
  `~/.chaosengine/music-venv`, built with plain pip from the app's own Python (3.11 or 3.12), with ACE-Step 1.5 at
  a pinned commit. About 1.6 GB. If the folder already holds a working pack, the install only checks it.
- **Weights**, downloaded through the app's downloader into the Hugging Face cache at pinned revisions:

| Model | Download | While generating | Idle | Speed (loaded) |
|---|---|---|---|---|
| ACE-Step 1.5 turbo (default) | 10.1 GB | about 22 GB | about 15 GB | about 1 s per 10 s of audio (60 s in 5 to 6 s; 600 s in 82 s) |
| ACE-Step 1.5 XL turbo (large) | 20.0 GB, plus turbo | about 47 GB | about 40 GB | about 2 s per 10 s; 64 GB Macs only, unloads every other model first |

The first track after a start takes about 25 s more (the model loads). On an NVIDIA GPU the XL card says 20 GB
or more; that is not measured here.

## What a track is

- FLAC, 24-bit, 48 kHz stereo, peak-normalised to -1 dBFS, exactly the length you asked for.
- Saved in the Music Library: `~/.chaosengine/music/library/<id>.flac` with `<id>.json` (title, style prompt,
  lyrics, seed, model, BPM, key, tags). Rename, tag and delete through `/api/music/library/{id}`; the audio route
  answers `Range` requests.
- The same seed with the same settings gives the same file, byte for byte.
- Each file carries `ENCODER` and `COMMENT=chaosengine:{...}` (app version, model, seed, length, BPM and key when
  given, the style prompt, and `machine_generated: true`). Lyrics and local paths are never written into the file.
  Settings switch `musicEmbedTag` (on by default, decision D-34) turns the tag off for private use.

## Controls

Style prompt (up to 512 characters), lyrics (up to 4,096, with `[verse]` / `[chorus]` tags) or Instrumental,
length 10 to 600 s, seed, variations 1, 2 or 4 (seeds `seed`, `seed+1`, ...), and optionally BPM (30 to 300), key
(for example "A minor") and time signature. Cover, repaint and reference audio exist in the engine but are not
offered yet; a request that sends them is refused, not ignored.

You are responsible for rights in lyrics you type. Describe a sound instead of naming an artist.

## Running and stopping

- Music is a heavy job on the shared queue: it waits for a running image, video or agent job, and they wait for it.
- Progress phases: load, encode, compose (8 real diffusion steps), decode, save. `GET /api/music/progress`.
- Cancel (`POST /api/music/cancel`) stops the diffusion within about a second; during the audio decode, which has
  no hook, the worker is stopped after 3 s and the next track starts it again.
- The memory gate asks for 23 GB free before a generation (8 GB when the model is already loaded) and says what to
  unload. An idle music worker is stopped when an image or video job needs the memory, and after 10 minutes unused
  (`POST /api/music/settings` with `idleUnloadMinutes`). `POST /api/music/unload` frees it at once; while a track
  is being made or a model loads it answers 409 (cancel the job first), so a job is never cut off as a failure.
- Agents: `make_music` on the agent bridge (see [agent bridge](agent-bridge.md)).

## Licence

ACE-Step 1.5 code and weights are MIT and not gated. The authors state the model was trained on legally compliant
data and that output may be used commercially. You are responsible for checking that your output does not copy
existing works. The pack leaves out `pytorch-wavelets` (restricted wording on PyPI); the optional DCW correction runs natively on
MLX without it (see `THIRD_PARTY_NOTICES.md`).
