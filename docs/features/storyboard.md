# Storyboard: a film as a project of shots, takes and audio lanes

A storyboard project is one folder you can move, back up, zip or send to another machine. This page
covers the backend. For the Projects, Board and Mixer screens, see [Storyboard](../usage/storyboard.md).

## The folder

```
<app data>/storyboards/<project-id>/
  project.json   schemaVersion, revision, defaults, references, shots, audio, export settings, batches
  media/         <shot>/<take>/still.png, clip.mp4; audio/; refs/   (copied in, never only referenced)
  cache/         render intermediates (safe to delete)
  history/       one small file per change, for undo
  exports/       rendered films and .chaosfilm bundles
```

- **Self-contained.** Importing a take, a still or a sound copies it into `media/` (a hard link when it is
  on the same disk) with a sha256. Paths in `project.json` are project-relative. A missing file shows the
  shot or clip as `missing`; `relink` points it at a new file.
- **Hostile input.** `project.json` and bundles are validated strictly: unknown fields, absolute paths,
  `..`, drive letters, URL schemes, a leading `-`, line breaks and symlinks leaving the folder are refused.
  A newer `schemaVersion` is refused; older ones are migrated (`store.MIGRATIONS`).
- **Revisions.** One writer per project at a time; every write is atomic and carries the `revision` the
  caller saw. An old revision answers 409 with the current project, so the UI and an agent can edit at once.
- **Undo and redo** cover shot edits, reorder, lock, pick take, add and remove shot, and audio edits.
  They put back only the fields the change made: takes, picks made by a batch and relinked files made
  since are kept, and a shot brought back by redo keeps the takes it had.
- **Restart.** Shots left `queued` or `running` go back to `idle`; an unfinished batch shows `interrupted`.

## Shots, takes and dependencies

A redo adds a take; picking another take is an edit. A shot with `startFrom: previous-shot` continues from
the previous shot's active take. Re-picking or redoing that take, or changing a reference or a picture
default, marks the shots that depend on it `stale`. Nothing regenerates by itself.

A clip that starts from a still or from the previous shot needs a video model that takes an image
(LTX-2.3 or LTX-2.5, `acceptsImageInput`). With any other video model the batch skips those shots with a
plain reason, and does not make their stills either: it never quietly makes a text-to-video clip instead.

## Format

`defaults` holds the aspect ratio (`exportPreset`), `resolution` (480p, 720p, 1080p, 1440p, 2160p), `fps` and the
pixel `width` and `height` they give (`backend_service/storyboard/formats.py`). A change to the aspect ratio or
resolution sets the size. A project saved without `resolution` (before D-64) reads as the same aspect ratio at 720p;
the old `shape` is accepted as a name for `aspect`. Clips are made at the size the video model supports and fitted
to the project size at join with ffmpeg `scale` and `pad`; a different frame rate is converted with `fps`.
`export.upscale` (`none`, `1440p`, `2160p`) adds a lanczos `scale` to the final encode, only to a larger size.

## Timing

The timeline uses each take's measured length (a 97-frame clip at 24 fps is 4.0417 s). `trimIn` and
`trimOut` are in and out points in the source, `speed` divides the length, a crossfade into a shot
overlaps the previous one. Audio anchored to a shot starts at that shot plus its offset, so it moves with
the shot.

## Batches and renders

A batch makes all stills, then all clips, and unloads the image model in between. Locked shots are
skipped; naming one needs `force`. It runs on the shared heavy-job queue as source `storyboard` with its
own batch id: the Studio's Cancel never touches it, and the batch's one Cancel stops all of it. Results
are kept in `project.json`. A render runs one at a time in the background (preview: small and fast).
A render does not take a heavy-queue slot: it is ffmpeg work on the CPU and can run while an image,
video or music job runs, so both are slower then. Render intermediates in `cache/render/` that the
current film does not use are removed when a render starts, once they are more than 7 days old.
A project cannot be deleted while one of its batches or renders runs (409).

## Captions, resume, OTIO and stems (v2)

- **Captions.** `project.captions[]` (`Caption {id, clipId?, start, end, text}`; with `clipId` the times are
  seconds in that clip's source, so a caption follows its clip; without, timeline seconds). Ops: `set_captions`
  (replace one clip's captions), `add_caption`, `edit_caption`, `remove_caption`; all undoable; removing a clip
  removes its captions. `POST .../captions/generate` runs `voice_runtime.transcribe_file` (the function behind
  `/api/voice/transcribe`) on each voice clip under the heavy-job slot, outside the project lock, then commits
  one `set_captions` per clip. `captions.cues()` maps rows to timeline seconds (trim and start applied);
  `to_srt` strips `{}`, `-->` and blank lines so libass reads plain text. `POST .../captions/export` writes
  `exports/<slug>-captions.srt`. Schema version 2 (migration 1 to 2 adds `captions: []`).
- **Burn-in.** `export.captionsBurn` writes `cache/render/captions.srt` and passes `captions_srt` in the plan; the
  mix stage draws it with the `subtitles` filter (pinned ffmpeg 7.1 macOS build has `subtitles`, `drawtext`,
  libass, fontconfig). If a build lacks `subtitles` the render drops burn-in, adds a warning and still writes
  the sidecar. Sidecar `.srt` (`export.captionsSrt`) is written by the render manager once the film exists.
- **Resumable render.** Per-shot intermediates are cached by content hash (unchanged). A manifest
  `cache/render/manifest-<preview|export>.json` records state, progress and the intermediate names. A manifest
  that says `running` with no live job in this process reads as `interrupted`; `describe()` then carries
  `resumable`, `shotsReady`, `shotsTotal`. Starting a render again is the resume. At the start of every render
  `assemble.remove_leftovers` deletes run-id scratch files and `.partial` outputs of a killed run. The render
  thread holds `longrun.keep_awake` (`caffeinate -w <pid>` on macOS, `SetThreadExecutionState` on Windows,
  `systemd-inhibit` on Linux) and a render of 30 s or more sends `longrun.notify` (`osascript` or
  `notify-send`, text passed as arguments). The Tauri shell has no notification plugin; if one is added the
  banner should move there.
- **OpenTimelineIO.** `otio_export.py` writes the JSON itself (no dependency; OTIO is Apache-2.0). `Timeline.1`,
  `Stack.1`, `Track.1`, `Clip.1` with `ExternalReference.1`, `Gap.1`, `Transition.1` (`SMPTE_Dissolve`, offsets
  half the fade, neighbours shortened by half the fade so the track equals the rendered film), `LinearTimeWarp.1`
  for speed. Gain, fades, duck and loop are in `metadata.chaosengine`.
- **Stems.** `export.stems` adds `stems_base` to the plan; after the film, `assemble._write_stems` renders the
  shots' own sound (from the joined film) and each lane kind through `build_mix_graph` with a silent first input,
  no loudness and no ducking, to `<film>-stem-<lane>.wav`. A stem failure is a warning, not a failed render.
- Routes: `POST /projects/{id}/captions/generate`, `/captions/export`, `/otio`; `/render` takes `notify`
  (default true; agent calls default to false) and `GET /render` falls back to the manifest after a restart.

## AI label (D-34)

The MP4 carries a prompt-free comment tag (app version, "machine generated", model names, a hash of the
project id and name; never a folder path) unless `export.aiLabel.embedTag` is off. A project with any
cloned-voice clip is always tagged. There is no visible end card; disclosure on screen is up to the author.

## API and agent tools

- HTTP: `/api/storyboard/projects` (list, create, get, delete), `/projects/{id}/ops` (`{revision, op, args}`),
  `/undo`, `/redo`, `/batches`, `/render`, `/bundle`, `/files?path=media/...|exports/...`, `/peaks?path=...&buckets=32..4000` (waveform peaks for the Mixer, cached in `cache/peaks/`), and
  `/api/storyboard/import` (a local path or the bundle bytes).
- Agent tools: `storyboard_project`, `storyboard_shot`, `storyboard_audio`, `storyboard_generate`,
  `storyboard_render` (see [Agent bridge](agent-bridge.md)).
