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

## AI label (D-34)

The MP4 carries a prompt-free comment tag (app version, "machine generated", model names, a hash of the
project id and name; never a folder path) unless `export.aiLabel.embedTag` is off. A project with any
cloned-voice clip is always tagged. `export.aiLabel.endCard` adds a visible end card.

## API and agent tools

- HTTP: `/api/storyboard/projects` (list, create, get, delete), `/projects/{id}/ops` (`{revision, op, args}`),
  `/undo`, `/redo`, `/batches`, `/render`, `/bundle`, `/files?path=media/...|exports/...`, `/peaks?path=...&buckets=32..4000` (waveform peaks for the Mixer, cached in `cache/peaks/`), and
  `/api/storyboard/import` (a local path or the bundle bytes).
- Agent tools: `storyboard_project`, `storyboard_shot`, `storyboard_audio`, `storyboard_generate`,
  `storyboard_render` (see [Agent bridge](agent-bridge.md)).
