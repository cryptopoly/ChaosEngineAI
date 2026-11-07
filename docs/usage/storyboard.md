# Storyboard

!!! note "Preview"
    Preview: this area is built but not yet switched on in the app (planned for v0.10.0).

Storyboard turns a set of shots into one film. It runs on your machine; nothing is uploaded.

The idea: a film is a **project folder** of **shots**, **takes** and **audio lanes**.

- A **shot** is one scene, with a still prompt, a motion prompt and a length.
- A **take** is one attempt at a shot: a still and a clip. A redo adds a take. It never overwrites one. You
  pick which take the film uses.
- **Audio lanes** hold voice, effects and music, and sit on a timeline under the picture.

You can change one shot without redoing the rest. Nothing regenerates by itself.

There are three screens under **Storyboard** in the sidebar: **Projects**, **Board** and **Mixer**.

## Projects

**Storyboard > Projects** lists your films as cards with thumbnails, the shot count, the length, the shape and a
**stale** count when shots are out of date.

- **New project.** Name, **Shape** (16:9, 9:16 or 1:1) and **Frame rate** (24 or 30).
- **Open** opens the project on the Board.
- **Export bundle** writes a single `.chaosfilm` file. See [Bundles](#bundles-for-sharing-and-backup).
- **Import bundle** reads a `.chaosfilm` file. The imported project gets a new id.
- **More actions** has **Rename** and **Delete**. Delete removes the whole folder and cannot be undone.

## Board

**Storyboard > Board** is where you write, make and arrange the shots.

At the top are **Project defaults** (shape, frame rate, image model, video model and voice) and **References**:
named characters and places, each with a description and an optional image, that you can attach to shots to
keep them consistent.

**Add shot** adds a card. Each shot card shows:

- A title you can edit, and a thumbnail. Click it to play the clip.
- A status: Not made yet, Queued, Making, Done (with the measured length, not the requested one), Failed, Stale
  or Take file missing.
- The **takes strip**. Click a take to use it; the active one is marked.
- **Lock.** A locked shot is skipped by every batch.
- **Redo.** Marks the shot for a new take.
- **Transition:** **Cut** or **Crossfade** with a length in seconds. A crossfade overlaps the previous shot.
- **More actions:** **Edit prompts**, **Import footage**, **Move up**, **Move down**, **Delete**. You can also
  drag a card by its handle to reorder.

### Make shots

- **Make missing** makes a take for every unlocked shot that has none yet.
- **Make all unlocked** adds a new take to every unlocked shot. Nothing is overwritten.

A batch makes all the stills first, then all the clips, and unloads the image model in between. A bar at the
top shows progress, for example "Shot 3 of 8", with **Cancel batch**. One Cancel stops the whole batch.

The batch runs on the shared queue, so it waits for other heavy jobs. A Studio's own Cancel never stops it.

### Edit prompts

**Edit prompts** opens a drawer for one shot: **Still prompt**, **Motion prompt**, **Start from** (a new
still, your image, or the last frame of the previous shot), length, seed, **Shot sound level**, trim in and
out, speed and the references to use. **Save changes** keeps them; **Make this shot** makes a new take of just
that shot.

### Import footage

**Import footage** adds a take from an image (png, jpg, webp) or a video (mp4, mov, webm) file, or from the
Image Gallery or Video Gallery. The file is copied into the project.

### Stale shots

A shot starting from the previous shot's last frame depends on that shot. If you give the earlier shot a new
take, or change a reference or a picture default, the later shot shows **Stale · after shot N**. You choose:

- **Re-run** marks it for a new take.
- **Keep** clears the badge and keeps what you have.

### Undo and redo

**Undo** and **Redo** in the header cover shot edits, reorder, lock, picking a take, adding and removing shots,
and audio edits. Takes made since stay in the folder.

If another window or an agent changes the project, the Board shows "Changed elsewhere; showing the latest".

### Keyboard

With a card focused: arrow keys move between cards, Enter opens the editor, L toggles lock, R toggles redo
and Delete asks to delete the shot.

## Use in storyboard

Other screens can send a file straight into a project:

| From | Button | What happens |
|---|---|---|
| Image Gallery | **Use in storyboard** | A new shot with the image as its still, or a new take of a shot. |
| Video Gallery | **Use in storyboard** | A new shot with the clip as its take, or a new take of a shot. |
| Voice Library and Voice Studio **Speak** | **Send to storyboard** | A clip on the Voice lane. |
| Music Studio and Music Library | **Send to storyboard** | A clip on the Music lane. |

The menu lists your three most recent projects, with the open one first, and **New project…**. With no
projects it reads **Start a storyboard with this**. Pictures and clips land on the Board; sound lands in the
Mixer. Sound is placed at the start of the first shot.

## Mixer

**Storyboard > Mixer** is where you hear the film, place sound and export. It shows the preview, a timeline,
the transport and the export panel.

**Preview.** Press **Render preview** to build a small, fast version. After you change something the screen
says "Out of date. Render again to hear your latest changes." Shots with no clip yet are left out.

**Lanes.** **Picture** and **Shot sound** are read from the shots; click a picture block to open that shot on
the Board. **Voice**, **Effects** and **Music** hold audio clips.

- **Add audio.** Each audio lane has a plus button. **Add from library** lists the Voice Library on the Voice
  lane. For Music and Effects, choose a file or type its path under **Import audio**. To bring a track from
  the Music Library, use **Send to storyboard** there.
- **Move and trim.** Drag a clip to move it. Drag its ends to trim. Clips show a waveform.
- **Zoom** in and out. The transport has Play, Stop and Loop. Space or K plays and pauses, J and L jump 5 s,
  Left and Right move 1 s (Shift 10 s, Alt one frame).

**Inspector.** Select a clip to edit it:

| Control | What it does |
|---|---|
| Name | The clip's label. |
| Gain | Level in dB. |
| Fade in, Fade out | Fade lengths in seconds. |
| Trim in, out | The part of the source to use. |
| Duck under voice | Lowers this clip while someone speaks. Not offered on voice clips. It is off until you tick it. |
| Loop to fill | Repeats the clip. |
| Anchor | Ties the clip to a shot, with an offset, so it moves with that shot. Or place it at a film time. |
| Remove from lane | Deletes the clip. |

Selecting a shot instead shows **Shot sound level for shot N** and **Open on Board**.

A clip whose file is missing shows **Missing** and **Relink**.

## Export

The export panel has:

- **Shape:** 16:9, 9:16 or 1:1. The pixel size shows beside it, taken from the project's size. A different
  shape is reframed at export; shots are not regenerated.
- **Loudness target:** -16 LUFS by default, which suits web video. Empty leaves the mix as it is.
- **Captions:** **SRT file** writes the captions next to the film; **Burn in** draws them into the picture.
  Both are off until you tick them (see Captions below).
- **Audio lanes as separate files** (stems): off until you tick it.
- **AI label** (below).
- **Export MP4.** Progress runs through Normalise shots, Join, Mix sound and Encode. **Cancel export** stops it
  and writes nothing. When it finishes you see the size and length, with **Reveal file** or **Copy path**, and
  the other files it wrote (the .srt and the stems).
- **Export for an editor (.otio).** Writes the cut as an OpenTimelineIO file for DaVinci Resolve, Premiere and
  other editors (see below).

The film is made from each take's measured length. Transitions shorten the film by their length; the ruler
shows the real total. Renders run one at a time.

### The AI label

- **Add a machine-readable AI tag.** On by default. A tag in the MP4 metadata with the app version, "machine
  generated", model names and a hash of the project id and name. It holds no prompts and no folder names.
- **A clip marked as a cloned voice forces the tag on.** When you add a voice clip you can tick **This clip uses
  a cloned voice**. The project then says "This project uses a cloned voice, so the AI tag is always written."
  The app does not clone voices itself yet; the box is a label you set.

### Captions

Open **Captions** above the export panel (it starts closed).

- **Make from voice** listens to your voice clips with the Voice Studio's speech model (the default one for
  this Mac, or Whisper elsewhere) and writes one caption per sentence. Long sentences are split in two lines of
  at most 84 characters. It needs a speech model installed (Voice > Models) and takes a few seconds per
  minute of speech. Running it again replaces the captions of those clips; captions you added by hand stay.
- Captions made from a clip **move with that clip**: move, trim or re-anchor the voice clip and its captions
  follow. A caption whose words you trimmed away shows **Trimmed away** and is left out.
- Edit the wording in the box (it saves when you leave it), **Remove** a caption, or **Add at playhead** for a
  line of your own (two seconds long; it is placed on the film's timeline).
- **Write .srt** saves `exports/<project>-captions.srt` right away, without a render.
- **Burn in** needs the `subtitles` filter in the bundled ffmpeg. If a build lacks it, the film is made
  without burned-in captions, the .srt is still written, and the export says so.

All of it can be undone with Undo.

### Resuming a long export

While an export runs the app asks the system not to sleep, and a long one (30 seconds or more) ends with a
system banner. If the app is closed or crashes mid-export, or you cancel it, the Export panel shows
**The export was interrupted** (or **did not finish**) with how many shots are already done, and **Resume
export**. Resume starts the export again and reuses every finished shot; only the rest is made. A shot you
changed in between is made again. Half-written files from the interrupted run are deleted when the next one
starts. Previews are quick and are not offered for resume.

### Exporting for an editor

**Export for an editor (.otio)** writes `exports/<project>.otio`: a picture track, a track of each shot's own
sound, and a track per lane (voice, music, effects, imported; a lane gets a second track only where its clips
overlap). Crossfades become dissolves. Each clip links to the file in your project folder, so the file opens on
this computer; to open it elsewhere, copy the project's `media/` folder and relink. Gain, fades and ducking
are written as notes on each clip, because the format has no field for them. A shot with a speed change
carries a speed effect that editors read differently; the app warns about it. Check the result in your
editor: this version has not been opened in Resolve or Premiere yet.

### Lane stems

With **Audio lanes as separate files** ticked, a full export also writes `<film>-stem-shot-sound.wav`,
`-stem-voice.wav`, `-stem-music.wav`, `-stem-sfx.wav` and `-stem-import.wav` (only lanes that have clips), each
as long as the film, 48 kHz stereo. A stem has its clips' gain, fades, trims and positions, but no ducking
and no loudness change. The shot-sound stem includes the crossfades. Stems are not made for previews.

## Bundles for sharing and backup

**Export bundle** writes a `.chaosfilm` file: one zip with the project file and every media file it uses,
nothing else. It leaves out unfinished batches and undo history, and holds no absolute paths from your
machine. **Import bundle** checks the file strictly before it unpacks anything: unsafe paths, links, wrong
checksums and newer formats are refused or dropped, and a file that does not match shows as **Missing**.

## What a project folder contains

```
storyboards/<project-id>/
  project.json   shots, takes, audio, defaults, export settings
  media/         every still, clip, sound and reference, copied in
  cache/         render intermediates (safe to delete)
  history/       one small file per change, for undo
  exports/       rendered films and .chaosfilm bundles
```

The folder is in the app's data folder, next to the settings. It is self-contained: you can move, zip or back
it up. Paths inside are relative to the folder.

## Where your files go

- **Stills and clips** a batch makes are saved by the same code as the Image and Video Studios, so each one also
  appears in the **Image gallery** or **Video gallery** as an ordinary item. Each is marked **From project X**,
  and the galleries' **Project** filter shows only one project's items. The project name, project id and shot id
  are also written into the file's own settings (the PNG text chunk, the MP4 comment) as names only, no folder
  paths. Items made outside a project have no label.
- **Copies in the project.** The project keeps its own copy of every still and clip under `media/`, so deleting
  an item from a gallery does not break the project, and the other way round.
- **Finished films** are written to the project's `exports/` folder and stay there; they are not added to the
  Video gallery. The project lists them under **Film**, newest first, each with a Reveal button.
- **Agents.** Outputs an agent makes through the Studios' tools land in the galleries like your own, without a
  project label unless a Storyboard batch made them.

## Limits and what is not done yet

- **Captions** have one style (white text with an outline at the bottom); there are no fonts or position
  settings yet, and times are not edited by hand (move the clip, or remove and add the caption).
- **Music cannot be sent as a sound effect** yet. Music Library items go to the Music lane.
- **Cloned voices** are not made by the app yet.
- **Resume** reuses finished shots only. The join and the sound mix are done again, which is the shorter part.
- A project can hold up to 500 shots, 500 audio clips and 50 takes per shot.
- After a restart, shots that were queued or running go back to idle, and an unfinished batch shows as
  interrupted.

## Agents

An agent can do all of this through the [agent bridge](../features/agent-bridge.md): five `storyboard_*` tools
for projects, shots, audio, batches and renders. For the backend facts, see
[Storyboard backend](../features/storyboard.md).
