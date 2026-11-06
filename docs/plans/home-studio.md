# Home studio: Music, Storyboard and the all-in-one pipeline

Status: draft 2 (2026-11-06), written on `feat/media-input`. Draft 1 was attacked by an Opus reviewer who checked it against the code; its findings are folded in (section 12). Decisions: D-25 to D-35 (section 9).
Research behind it: model and licence sweep (Opus agent, section 3), a code survey of how Voice, Image and Video were built (section 4), and the adversarial review (section 12).

## 1. Goal

One local app where a person (or an agent such as Claude Code) goes from an idea to a finished film with picture, speech, sound effects and music, and can change any single scene without redoing the rest.

The test we are building toward, already half proven by the 32 s lighthouse film:

> "Make a 30 s ad for X." Shots appear in a storyboard. The user likes shots A, B, E, F, and asks for new takes of C and D. Music runs under the whole film, speech and effects sit on their own lanes, and one button exports the result. Nothing leaves the machine.

Principles:

1. **Local first.** No hosted service, no credits. Models are free to download and run, and we say plainly what each licence asks.
2. **Every step is a tool.** Whatever a person can click, an agent can call (the agent bridge), through the same code.
3. **Never silently ignore.** A request a model cannot honour is refused with a reason, as image and video input already do.
4. **Per-shot control.** Prompts, seeds, reference images, takes and locks are saved, so redoing a shot is a button.
5. **One heavy job at a time** on the shared queue (`HEAVY_JOBS`). Music, sound and speech are heavy jobs too.
6. **Small steps, each shippable.** Every phase ends with a live check on a real machine and an E2E check.

## 2. Where we are (done, on `feat/agent-bridge` and `feat/media-input`)

- Agent bridge: 12 `/mcp` tools, shared queue, access switch and token, stdio relay, remote upload and download.
- Image and video from images: video start / end image (LTX-2), image references (FLUX.1, FLUX.2 Klein, Qwen, FLUX 3 Klein edit), Studio pickers, standard size buttons.
- 32 s eight-shot film made only through `/mcp` (`docs/examples/lighthouse-keeper-demo/`), with one consistent character and place. The script is already a crude storyboard.
- Not done, and the gap this plan closes: no music, no sound effects layer, no voice cloning, no saved shot list, no mixer, no join step inside the app.

## 3. What the research found

### 3.1 Models (Opus research agent, 2026-11-06; primary pages read; owner must re-verify at pin time)

| Need | Pick | Licence (as read) | Why / caveat |
|---|---|---|---|
| Songs and instrumentals | **ACE-Step 1.5** (2B turbo; XL 4B on big GPUs) | MIT, not gated; XL card says trained on "legally compliant datasets", output usable commercially | Songs 10 to 600 s with vocals and lyrics; BPM, key, duration, reference audio, cover, repaint, stems (base models), LoRA. Under 4 GB for 2B. MLX path in the repo. **No Mac or 4090 speed numbers found; spike first.** |
| Songs, second choice (CUDA) | HeartMuLa 3B | Apache 2.0 (changed 2026-01-20) | Slow (real-time factor about 1), no reference audio, no Mac mention |
| Instrumental beds, second choice (Mac) | Magenta RealTime 2 | code Apache 2.0, weights CC-BY 4.0 | Instrumental only, streams in 2 s chunks; credit Google |
| Sound effects | **Stable Audio Open** (Small 11 s, 1.0 47 s) | Stability Community License: gated; free under US$1M annual revenue; must show "Powered by Stability AI"; outputs may not train other foundation models; you own outputs | Not vocals; good at effects and field recordings. Needs a licence screen |
| Video to sound | LTX-2.3 **Foley V2A LoRA** (Lightricks) or ThinkSound | LTX-2 Community License (we already carry it) or Stability terms | Unclear if the LoRA works on footage we already have; spike |
| Voice cloning | **Qwen3-TTS** (0.6B / 1.7B), second VoxCPM2 | Apache 2.0 | Clone from about 3 s of audio; MLX ports exist; add a consent checkbox |
| Separation | python-audio-separator (BS-RoFormer), Demucs (archived) | package MIT; weight licences not stated | Optional tool; verify weights |
| Loudness | ffmpeg `loudnorm`, `sidechaincompress`, `xfade` | LGPL/GPL build we already ship (imageio-ffmpeg) | Check the bundled build has these filters (spike) |

Rejected on licence: MusicGen / AudioGen (CC-BY-NC), YuE2 (CC-BY-NC plus companies need a licence), SongGeneration / LeVo (academic only; repo gone), MMAudio and Foley-Omni (CC-BY-NC parts), HunyuanVideo-Foley (excludes EU, UK, South Korea), F5-TTS and Voxtral TTS (CC-BY-NC). MiniMax Music 3 is open-weight but needs on-screen attribution, a revenue authorisation above US$20M and an AI-disclosure rule, and says CUDA only: third choice, not now. Chatterbox is MIT but retired.

### 3.2 Why the film lost its music

LTX-2 makes sound per clip, from the clip's own prompt. The first clip happened to get music; nothing carries over. A separate music lane under the whole film fixes it, with a per-shot "native sound" level.

## 4. What the code survey found (the traps)

Voice (FU-117) is **not** a clone of Image and Video. Voice has no catalog, no Models page adapter, no memory gate, no progress tracker and no output-folder setting. Music is heavy generation, so it takes **the Image / Video pattern** for catalog, gate, progress and queue, and **the Voice pattern** for the isolated venv, the worker process and the library.

Traps a new area must not miss (full list in appendix A):

1. Tab and group ids are string literals in many places (`types/system.ts`, `constants/tabs.ts`, `sidebarGroups.ts`, `tabRedirects.ts` and its test, `SidebarIcons.tsx`, `App.tsx`, four locale blocks in `common.json`).
2. The kind literals `"image" | "video"` are hard-coded in `fit.py`, `mcp_bridge/models.py` (an unknown kind silently becomes video), `tools.py`, `useGenerationProgress.ts`, `types/progress.ts`, `ModelsArea`, `CHAT_EXCLUDED_MODEL_TYPES`.
3. Downloads (`state/downloads.py`, `huggingface.py`, `companions.py`) check image, then video; a music repo falls through both unless a third branch is added.
4. `mcp_bridge/files.py` only serves the image, video and voice folders; the music folder must be added or remote fetch returns 404.
5. `controller.py` has a fixed list of process markers for Stop.
6. `HEAVY_JOBS.slot` must wrap the model run before the memory gate.
7. Windows and Linux voice lanes are still untested, so a new venv inherits that uncertainty.

## 5. The design

### 5.1 Music area: Models, Studio, Library

**Backend** (mirrors Voice for the process, Image / Video for the rest):

- `catalog/music_models.py` plus `helpers/music.py` (payloads with fit and availability), exported from `catalog/__init__.py`.
- `music_env.py` (own venv `~/.chaosengine/music-venv`, Python 3.11 or 3.12, pinned ACE-Step commit, install script like `install-ltx2-mlx.sh`), `music_worker.py` (JSON-lines subprocess, no backend imports), `music_client.py` (spawn, idle stop), `music_runtime.py`, `music_library.py` (one JSON plus audio file per item).
- `routes/music.py`: `/api/music/{catalog,runtime,install,progress,cancel,preload,unload,generate,library,...}`; registered in `routes/__init__.py`.
- `MUSIC_PROGRESS` tracker, queue kind `"music"`, memory gate for music, `fit.py` kind `"music"`, new task value `txt2music`.
- Settings: `musicOutputsDirectory` (four backend and five frontend places), process markers, model classifier, `THIRD_PARTY_NOTICES.md`, `pre-build-check` key deps.
- Output: FLAC or WAV 48 kHz stereo as the master, MP3 on export. Tags carry prompt, lyrics, seed, model, "machine generated" (a metadata writer for audio, new, modelled on `output_metadata.py`).

**Frontend:**

- Sidebar group `music` with Models, Studio, Library, like Image and Video; Models page through the shared `ModelsPage` plus `adapters/music.ts`.
- **Music Studio:** style prompt (tags and free text), lyrics box with an "instrumental" switch, length (10 s to 10 min), BPM and key (optional), seed, variations (up to 4), waveform player with loop, "send to storyboard". Later: reference audio, cover, repaint and extend a section.
- **Music Library:** list with player, rename, tags, delete, drag into a storyboard.
- Strings in `studio.json` (en text, `null` elsewhere), reused `LiveProgress`, monochrome icons, tokens only (skills `design-md-enforce` and `ui-ux-pro-max`).

**Agent tools:** `make_music` (style, lyrics, seconds, bpm, key, seed, instrumental), `list_models kind music`, plus `music` in the output downloads.

#### Music engines: ACE-Step first, others as options

The worker takes an `engine` field, so ACE-Step is the first engine, not the only one. Each catalog row carries its licence, gated flag, platforms and what it makes (songs, instrumentals, loops), and the Models page shows them. Candidates for people who do not want ACE-Step, from the research:

| Engine | For whom | Caveat |
|---|---|---|
| ACE-Step 1.5 (2B turbo; XL 4B) | Default: songs with vocals, Mac and CUDA | MIT; speed on a Mac to be measured |
| HeartMuLa 3B | CUDA users who want a second song model | Apache 2.0; slow (about real time); no reference audio; no Mac note |
| Magenta RealTime 2 | Mac users who only want instrumental beds or loops | Weights CC-BY 4.0 (credit Google); instrumental only |
| DiffRhythm v1.2 | Light option (about 8 GB, Mac supported) | Apache 2.0 code and DiT; VAE licence unstated; quality behind the leaders |
| MiniMax Music 3 | Big-GPU users who accept its terms | Needs "MiniMax-Music3" shown in the product UI, an AI-disclosure rule, revenue authorisation above US$20M, CUDA, about 57 GB: opt-in only |
| Stable Audio Open (Small / 1.0) | Loops and effects, not songs | Gated; Stability Community License (see 5.2) |

Cloud services are out of scope (local-first).

### 5.2 Sound effects, video to sound, voice

- **Sound effects** as a mode of the Music area ("Sounds"), engine Stable Audio Open (Small first). A licence screen shows "Powered by Stability AI" terms and the US$1M cap; the model stays gated, so the user accepts on Hugging Face.
- **Video to sound:** spike the LTX-2.3 Foley LoRA on existing clips; if it works, expose "add sound to this clip" in the Video Studio and as a shot option. ThinkSound is the fallback under Stability terms.
- **Voice cloning:** add Qwen3-TTS as a second voice engine (MLX on Mac, CUDA elsewhere) in the Voice Studio; a consent checkbox ("I have the right to clone this voice") is required, and the audio is tagged machine generated.

### 5.3 Storyboard: the project and its shots

A **project** is one self-contained folder the user can move, back up, zip or send to another machine:

```
MyFilm/
  project.json          revision, schemaVersion, defaults, references, shots, audio lanes
  media/<shot>/<take>/  still.png, clip.mp4   (copied or hard-linked from the Studios, never only referenced)
  media/audio/          music, sfx, voice, imported sound
  cache/                normalised per-shot intermediates, proxies, waveforms (safe to delete)
  history/              one small file per change, for undo
```

Draft 1 only referenced library items; the review showed that breaks on a moved folder, a changed output folder, a Gallery delete, or another machine. So:

- **Self-contained.** Every take and audio clip is copied (or hard-linked on the same disk) into `media/`. Paths in `project.json` are project-relative, with a content hash. A missing file shows a "missing media" state and a relink action.
- **Hostile input.** `project.json` may come from another machine or an agent. Only project-relative paths and `upload:` handles are allowed; each path is resolved and checked to sit inside the project; text fields never reach an ffmpeg filter string unescaped.

```
Project  { schemaVersion, revision, id, name, createdAt,
           defaults{width,height,fps,imageModel,videoModel,voice,exportPreset},
           references{ characters[], places[] },
           shots[], audio[] }
Shot     { id, order, title, status(idle|queued|running|done|failed|stale|missing),
           stillPrompt, motionPrompt, referenceImages[],
           startFrom("generate"|"upload"|"previous-shot"), endImage?,
           seconds, seed, locked, trimIn, trimOut, speed,
           takes[{ id, still, clip, hashes, settings, measuredSeconds, createdAt }], activeTake,
           transition{ type(cut|crossfade), seconds }, nativeSoundDb }
AudioClip{ id, kind(music|sfx|voice|import), source, anchor{shotId, offset} | start,
           trimIn, trimOut, gainDb, fadeIn, fadeOut, duck, loop }
```

Sharing and layers: the folder is the working format (like a layered PSD, but for a film: shots, takes and audio lanes stay separate and editable). A single-file **bundle** (`.chaosfilm`, a zip of the folder) is for sharing and backup, and opens in the app. Later (P6): export the timeline as OpenTimelineIO, so a cut can be finished in DaVinci Resolve or Premiere, and export each audio lane as its own stem file. (OpenTimelineIO's licence to be read before use.)

Rules (each one answers a review finding):

- **Takes, not overwrites.** "Redo" adds a take; picking another take is an edit. "Keep A, B, E, F and redo C and D" leaves the other shots' bytes untouched (checked by hash in the tests).
- **Dependencies.** A shot chained with `startFrom: previous-shot` depends on the previous shot's *active take*. Redoing or re-picking a take marks every chained shot after it `stale` (not regenerated); the user chooses. Changing a reference or a default marks the shots that use it `stale` too.
- **Timing is measured, not assumed.** The timeline comes from each take's measured duration (a 97-frame LTX clip is 4.04 s, not 4.0). Voice and effects anchor to a shot by default, so a redo with a different length does not slide every later cue. Transitions shorten the timeline by their length and the mixer shows it.
- **Locks.** `regenerate unlocked` skips locked shots. A direct generate on a locked shot is refused unless the call says `force`.
- **Concurrency.** One backend writer per project (a lock), writes go to a temp file then rename, and every write carries a `revision`; a write against an old revision is rejected with the current one (UI and agent can edit at once). Each change appends to `history/` for undo.
- **Restart recovery.** On load, any shot left `queued` or `running` is reset to `idle` (the in-memory job store forgets jobs on restart). A film render keeps a manifest so it resumes at the first missing intermediate.
- **Versioned from day one.** `schemaVersion` plus a migration test in P3.
- **Optional director step:** write a shot list from a one-paragraph brief with a local chat model, or the agent's own model through the tools.

### 5.4 Assemble and mix

Draft 1 put everything in one ffmpeg graph. That keeps every decoder open, re-encodes the whole film on each preview, and hits Windows' 32 KB command-line limit near 20 shots. The design is now two stages:

1. **Per-shot intermediate** (cached, keyed by the take's hash and the project's export settings). Each take is normalised to one size, fps, `yuv420p`, timebase and 48 kHz stereo audio: `fps,scale,pad,setsar=1,format=yuv420p,settb=AVTB`. A clip without sound gets `anullsrc` for its length; audio and video are trimmed to the same length (`atrim` / `apad`); trims and speed apply here.
2. **Join and mix.** Short `xfade` / `acrossfade` joins (cuts are a plain concat). Audio is mixed in a second graph from the lanes: native shot sound at its per-shot gain, voice, effects, music; music ducked under voice with `sidechaincompress` (the voice bus is `asplit` so one copy is the key); `amix` with `normalize=0`; `loudnorm` then `aresample=48000` (single pass for preview, two-pass for export); fades; captions and title cards if asked.

Safety and portability:

- Always the **pinned bundled ffmpeg** (`imageio-ffmpeg`), never a Homebrew one. Today `output_metadata._ffmpeg_exe()` prefers `shutil.which`; assemble must not. The bundled macOS build is ffmpeg 7.1 and I confirmed it has `xfade`, `acrossfade`, `loudnorm`, `sidechaincompress` and `ebur128`. Windows and Linux builds are checked in P0. It ships no `ffprobe`, so durations come from ffmpeg output or PyAV.
- `-protocol_whitelist file,pipe` and `-safe 1`; the graph goes in a script file, not argv.
- A preview is a low-resolution render from cached intermediates, so redoing one shot only re-renders that shot.
- When a music lane exists, new clips are generated with "no music" in the sound prompt so lanes do not fight.
- Declare `imageio-ffmpeg` in `pyproject.toml` (it is only transitive today) and add the GPL notice and source offer to `THIRD_PARTY_NOTICES.md` (section 6).

### 5.5 Storyboard UI

New sidebar group "Storyboard": **Projects**, **Board**, **Mixer**.

- **Board:** shot cards (still, clip, status, takes strip, lock, redo, transition, "stale" badge with a one-click re-run); drag to reorder; a brief box and "write shots". Import own footage, audio and images into a shot or lane.
- **Mixer:** timeline with picture on top, lanes below, waveforms (cached), drag audio from the Music Library, per-clip gain, fades and trims, a ducking toggle, play preview, captions (from the existing Whisper STT, as an SRT sidecar and optional burn-in), export with presets (16:9, 9:16, 1:1; a size change reframes at export, not a full regenerate).
- **Batches:** a storyboard run is ordered by kind (all stills, then all clips, then audio) and unloads the previous model between kinds, so a 64 GB Mac is not thrashing models. It has one Cancel for the whole batch.
- **Long renders:** keep the machine awake, notify on finish, show disk used by takes with a purge action.

### 5.6 Agent bridge growth

Prerequisites found in review: `mcp_bridge/jobs.py` needs a persistent record of batch jobs, running-cancel for music and voice (it exists only for image and video), and cancel by ticket id (see section 5.9); `files.py` must serve project exports; `upload` needs audio and video, not only images.

Tools to add: `make_music`, `make_sound`, `storyboard_create`, `storyboard_get`, `storyboard_edit`, `storyboard_generate` (shot ids, what: still / clip / both, new take), `storyboard_assemble`, `storyboard_export`. That takes 12 tools to about 20. Every tool description costs an agent tokens, so group by verb (decision D-30) and keep descriptions short. All tools use the same code as the UI; the demo script becomes a short agent recipe in `SKILL.md`.

### 5.7 Quality and control pack (after the core)

Ordered by value: extend / continue a clip, locked-seed variations, video upscale and frame interpolation, image inpaint and outpaint, depth / pose control, character LoRA training, speed work (FIS-DiT, DisCa, lower-resolution-then-upscale; papers, code availability unverified), more quantisation. Each is its own FU row, not a phase gate.

### 5.8 Remote compute (the 4090)

Separate epic. Done already: network listening works, the token works, remote upload and download exist. Missing: a Windows install that listens on the network, the CUDA lanes for music and Qwen3-TTS, and (optional) a Mac-app "remote worker" setting that routes jobs to the PC. Plan: first prove the PC as a plain `/mcp` target for Claude, then decide whether the Mac app should route to it.

### 5.9 Queue and cancel (a bug found in review)

`routes/images.py` and `routes/video.py` cancel with `IMAGE_PROGRESS.request_cancel() or HEAVY_JOBS.cancel_waiting("image")`. One Studio Cancel therefore stops whatever image job is running, even an agent's, and removes every waiting image job whatever its source. A storyboard batch would die on one click. Fix first, before the storyboard: tickets get a `batch` id and `source`; Cancel takes a ticket or batch id; a Studio Cancel only touches Studio tickets; a storyboard batch has its own Cancel.

## 6. Licence and compliance plan

| Item | What the UI and repo must do |
|---|---|
| ACE-Step (MIT) | Notices entry; show: "Weights are MIT. Trained on data the authors describe as legally compliant. You are responsible for checking output does not copy existing works." |
| Stable Audio Open | Gated; licence screen with "Powered by Stability AI", the US$1M cap and the no-model-training clause; ship the agreement and Notice file with the notices |
| LTX-2 Foley LoRA | Covered by the LTX-2 licence we carry; check the LoRA clause |
| Qwen3-TTS (Apache) | Notices entry; consent checkbox; machine-generated tag |
| Anything gated or non-commercial | Not default; behind an explicit accept step; never bundled |
| All generated audio | Tag "machine generated" in the file metadata |
| Exported MP4 | Machine-readable marker in the MP4 metadata (`embed_mp4_settings` exists) and an optional visible "AI-generated" end card; ask the owner about an audio watermark. Tags are easy to strip, so EU AI Act Art. 50 style labelling is treated as a requirement to design for, not a tag |
| Voice cloning over `/mcp` | The UI checkbox is not enough: consent is a required tool argument, recorded with the sample, and cloning is off for remote agents by default |
| Lyrics, cover and reference-audio modes | UI copy: you are responsible for rights in lyrics you type and audio you supply; warn when an artist name is in the style prompt; the same notice for stem separation |
| Bundled ffmpeg | The `imageio-ffmpeg` build is `--enable-gpl` with x264 and x265, and `THIRD_PARTY_NOTICES.md` has no entry for it today (this exists now, not only for the plan): add the GPL notice and a source offer, or switch to an LGPL build |
| Catalog | The catalog has no licence field; add `licence`, `gated`, `commercial` so the Models page and the licence screens read data, not text in notes |

Rule: nothing ships as a default unless we have read its licence on the primary page. Re-read at pin time (the Stability and MiniMax licences have been misreported before).

## 7. Phases

Each phase is one branch, ends with a live run on this Mac, tests, an E2E check, a hub update. The review changed the order: the storyboard does not need music (voice, clip sound and imported audio fill the lanes), and the highest-risk unknown (music on a Mac) must not hold up the highest-value, lowest-risk feature.

Branching: the phases currently stack on `feat/media-input`, which is not merged, while v0.9.9 is still held. Decide the merge order first (D-32). Land the shared registries in one small scaffold branch so parallel branches do not collide (shared files: `App.tsx`, `constants/{tabs,sidebarGroups,tabRedirects}.ts` and its test, `types/system.ts`, `SidebarIcons.tsx`, the ten `common.json` and `studio.json`, `api/index.ts`, `mcp_bridge/{tools,models,jobs,files}.py`, `progress.py`, `routes/__init__.py`, settings files, `fit.py`, `state/downloads.py`, `controller.py`, `e2e_test_suite.py`, `CLAUDE.md`, `FOLLOW_UPS.md`, hub, notices). Only `features/music/*` and `storyboard/*` are disjoint.

E2E numbers are reserved now: phase 10 Storyboard core, phase 11 Music, phase 12 Storyboard UI.

| Phase | What | Exit criteria | Size | Model |
|---|---|---|---|---|
| **P-1 Prep** | (a) fix queue cancel (5.9); (b) extract `App.tsx` (2,269 lines, cap 500) before six more tabs; (c) scaffold branch with the tab, group, kind and tool registries and the catalog licence field; (d) GPL notice for ffmpeg, declare `imageio-ffmpeg`; (e) log D-27 to D-35 | Tests green; no behaviour change except Cancel; notices updated | 4 to 6 days | Sonnet; Opus reviews (a) and (b) |
| **P0 Spikes** (parallel with P3) | ACE-Step 2B turbo on this Mac (speed, RAM, sample rate, pinned commit, Python version); ffmpeg filters on **Windows and Linux** builds (macOS done); LTX Foley LoRA on an existing clip; Qwen3-TTS on MLX; Stable Audio Open Small on MLX | One report with numbers and go / no-go per item; D-25 closed | 3 to 4 days (5 spikes) | Opus reads results; Sonnet runs installs |
| **P3 Storyboard core (headless)** | portable project folder, revision and lock, history, takes and dependency stale marking, queue batches (by kind, unload between), per-shot cached intermediates, two-stage assemble, trims, ducking, loudness, export, persistent resumable render, agent tools | The lighthouse film rebuilt from a project; redo shots 4 and 5, rejoin, other shots byte-identical; a moved and zipped project still opens; a hostile `project.json` is refused; E2E phase 10 (3-shot mini film) | **10 to 14 days** | **Opus** for the model and assemble; Sonnet for routes and tests |
| **P1 Music backend** (after the P0 go) | venv following `voice_env.py` (builds from the backend's own Python, unlike `install-ltx2-mlx.sh`), worker, runtime, routes, progress, queue, library, settings, gate, fit, downloads, markers, running-cancel by killing the worker, notices, `make_music` | A 60 s song with vocals from API and `/mcp`, in the library with tags; E2E phase 11 | 4 to 6 days | Sonnet; Opus review |
| **P2 Music frontend** | sidebar group, Models page, Studio, Library, strings | Make, play, loop, save, delete a track from the UI | 4 to 5 days | Sonnet; `ui-ux-pro-max`, `design-md-enforce` |
| **P4 Storyboard UI** | Projects, Board, Mixer, preview, import, captions, export presets, locks, takes, long-render keep-awake and notification, disk view | A person makes the 30 s ad from a brief with no terminal and redoes two shots; E2E phase 12 | **12 to 18 days** | Sonnet builds; Opus reviews the interaction model |
| **P5 Sounds and voice** | Stable Audio Open mode and licence screen, V2A if the spike passes, Qwen3-TTS with recorded consent | Effects lane and a cloned voice in a mixed film | 4 to 6 days | Sonnet |
| **P6 Quality pack** | extend clip, seed variations, upscale and interpolation, inpaint, control, title cards, templates, beat-synced cuts | Each its own small branch | open | mixed |
| **P7 Remote and Windows** | Windows install that listens, CUDA lanes, remote-worker spike, audio and video upload, project export fetch | The mini film made on the 4090 and fetched to the Mac | 3 to 5 days plus the owner's hardware time | Opus design; owner tests |
| **P8 Polish and release** | docs, screenshots, release notes, notices, perf baseline, accessibility | Pre-build checklist green | 2 to 3 days | Haiku for chores |

Critical path: P-1, then P3, then P4, then the definition-of-done run. Music (P0, P1, P2) runs beside P3 and joins at the Mixer. Total about **8 to 11 weeks** of agent time with two streams in parallel; the first draft's 5 to 7 weeks was optimistic for P3 and P4.

## 8. Risks and unknowns

1. **ACE-Step on a Mac is unproven here.** No M-series speed number exists. The plan no longer waits on it: the storyboard runs first.
2. **Memory thrash** (raised from "needs measuring"). The queue serialises jobs but does not unload. Alternating still, clip, music and voice either reloads models every step or keeps several resident. Batch by kind and unload between kinds; measure on the 64 GB Mac in P3.
3. **Project integrity.** Concurrent edits, restarts, moved folders, hostile files. Covered by section 5.3; each rule needs a test.
4. **Filter and codec differences on Windows and Linux** (macOS confirmed). P0.
5. **Python versions.** ACE-Step wants 3.11 or 3.12; `voice_env.py` uses the backend's own Python (3.12 in release, 3.11 in dev).
6. **Download sizes.** XL is about 18.8 GB; show sizes and fit honestly, and show disk used by takes.
7. **Licence drift and gaps.** Re-read at pin time; gated models opt-in; the ffmpeg GPL notice is missing today.
8. **UI weight.** The Board and Mixer are the biggest UI we have built; first version has lanes only, no free multi-track editing.
9. **Windows and Linux** voice lanes are still untested; expect the same for Music and the PC install.
10. **Scope and parallel conflicts.** P6 is cut before P4. Shared files make parallel branches collide; the scaffold branch is the guard.
11. **Time:** the 45-minute film target holds only at 768x448 export; state the export size in every claim.

## 9. Decisions needed

| # | Question | Recommendation |
|---|---|---|
| D-25 | Music first model | ACE-Step 1.5 after the P0 spike; HeartMuLa second on CUDA; Magenta RT 2 for Mac beds |
| D-26 | Storyboard | Yes, self-contained project folder with takes, locks and dependencies (5.3) |
| D-27 | Sound effects despite the Stability terms | Yes, opt-in, gated, with the licence screen; never bundled |
| D-28 | Voice cloning | Yes, Qwen3-TTS; consent recorded; off for remote agents by default |
| D-29 | Remote compute | Prove the PC as a `/mcp` target first (P7); decide Mac-routes-to-PC afterwards |
| D-30 | Agent tool surface | Group storyboard tools by verb to keep the list near 20 and descriptions short |
| D-31 | Project format | **Decided.** Portable folder: `project.json` with `schemaVersion` and `revision`, media copied in, no database; plus a single-file `.chaosfilm` bundle for sharing |
| D-32 | Release and branch order | Merge agent bridge and media input as v0.9.10 first (owner's call, on hold); Music and Storyboard as v0.10.0 |
| D-33 | Storyboard before music | Yes: P3 runs beside the music spike, not after music |
| D-34 | AI labelling | Machine-readable marker in exported MP4 and audio tags, optional visible end card; ask about an audio watermark |
| D-35 | ffmpeg licence | Add a GPL notice and source offer now, or move to an LGPL build; owner to choose |

## 10. Definition of done for the first "home studio" release

From a one-paragraph brief, in the app, a person gets a 30 s film with picture, speech, effects and music; redoes any two shots without touching the others; exports an MP4; and the whole run is repeatable from an agent. The run is timed and recorded (target: under 45 minutes on a 64 GB M-series Mac for 8 shots plus music), with the exact licence notices visible in the app.

## 11. Test strategy

- Unit and contract tests per module (catalog, worker protocol, library, assemble filter-graph builder, project store, migrations, takes and locks).
- Hash check for "other shots untouched" after a redo.
- E2E: phase 10 (Music), phase 11 (Storyboard mini film), phase 9 stays.
- Live check on this Mac at the end of each phase, with what was not tested written down.
- Perf baseline for music (seconds of audio per second) added to `PERF_BASELINE.md`.

## 12. Review log (Opus adversarial review, 2026-11-06)

Checked by me before accepting: the Studio Cancel bug (confirmed at `routes/images.py:123`), `App.tsx` at 2,269 lines (confirmed), no ffmpeg entry in `THIRD_PARTY_NOTICES.md` (confirmed), the bundled ffmpeg 7.1 being GPL with `xfade`, `acrossfade`, `loudnorm`, `sidechaincompress`, `ebur128` (confirmed on macOS).

| Finding | Change made |
|---|---|
| Project folder not portable | Self-contained folder, relative paths, hashes, missing-media state (5.3) |
| Hostile paths into ffmpeg | Path rules, protocol whitelist, graph in a file (5.3, 5.4) |
| Continuity breaks "keep A, B, E, F" | Dependency graph, stale marking on take change (5.3) |
| One Cancel wipes a batch | Queue fix as first task (5.9, P-1) |
| Jobs and status lost on restart | Status reset on load, resumable render manifest, running-cancel for music and voice (5.3, 5.6) |
| Concurrent writes, undo | Writer lock, revision check, history folder (5.3) |
| Timing model gaps | Measured durations, anchors, trims (5.3) |
| Assemble pitfalls | Two-stage design with cached intermediates and the explicit filter rules (5.4) |
| Spike tests the wrong ffmpeg | Always the pinned binary; Windows and Linux probed in P0 (5.4) |
| GPL ffmpeg, no notice | Notice and source offer, D-35 (6) |
| Consent and AI labelling | Recorded consent, off remotely by default, MP4 marker, D-34 (6) |
| Order and estimates | Storyboard before and beside music; P3 10 to 14 days, P4 12 to 18; App.tsx extract; E2E numbers reserved (7) |
| Memory thrash | Batch by kind, unload between; risk raised (5.5, 8) |
| Remote agent gaps, kind fall-through, catalog licence field | Listed in 5.6, 6 and P-1 |
| Mixed Python venv pattern | Follow `voice_env.py` (P1) |
| D-27 to D-32 not logged | Logged with D-33 to D-35 |

Not taken as written: audio watermarking and the 20-shot single-graph claim are treated as design questions (D-34) and a fixed design (the two-stage assemble), not as separate work items.

Product gaps the reviewer ranked, now placed: import own footage and audio, captions from Whisper, 9:16 and 1:1 export, resumable render with keep-awake and notification, disk view and purge, cached proxies (all in P4); title cards, templates, beat-synced cuts, thumbnails (P6).

## Appendix A: touch-point checklist for a new area (from the code survey)

Backend: `mcp_bridge/jobs.py` (persistence, running-cancel for music and voice); `routes/music.py` plus `routes/__init__.py` registration; `catalog/music_models.py` and export; `helpers/music.py`; `fit.py` kind branch; `state/downloads.py`, `helpers/huggingface.py`, `helpers/companions.py` third branch; `progress.py` instance; `HEAVY_JOBS` kind; `memory_gate.py`; settings key in `helpers/settings.py`, `models/__init__.py`, `state/settings_state.py`; `controller.py` process markers; `model_classifier.py` and `discovery.py`; `mcp_bridge/{models,tools,files}.py`; `output_metadata.py` (audio); notices and `pre-build-check`; `check-platform-wheels.py` group; packaging in `stage-runtime.mjs` if a script is bundled.

Frontend: `types/system.ts` (`TabId`, `SidebarGroupId`); `constants/{tabs,sidebarGroups,tabRedirects}.ts` and the redirect test; `SidebarIcons.tsx`; `App.tsx` switch and runtime fetch; `modelEntry.ts` `ModelsArea`; `components/models/adapters/music.ts`; `features/music/*`; `hooks/useGenerationProgress.ts`, `types/progress.ts`; `api/music.ts` and `api/index.ts`; `types/music.ts`; locale keys in `common.json` (tabs, captions, short labels, groups) and `studio.json`; `utils/library.ts` exclusion.

Tests and docs: `tests/test_music*.py`; E2E phase 10; `docs/usage/music.md` and `mkdocs.yml`; `docs/E2E_TESTING.md`; `CLAUDE.md` test table; `FOLLOW_UPS.md` rows and `tools/hub_tracker.py`; hub status and decisions.
