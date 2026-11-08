# Music and Audio

!!! note "Preview"
    Preview: this area is built but not yet switched on in the app (planned for v0.10.0).

Music makes songs with vocals and lyrics, or instrumentals, from 10 seconds to 10 minutes long. It runs on
your machine; nothing is uploaded. The model is ACE-Step 1.5. The area also makes sound effects (see the Studio's
**Music | Sounds** switch). There are three screens under **Music and Audio** in the
sidebar: **Models**, **Studio** and **Library**.

The engine runs on Apple Silicon (MLX). On Windows and Linux the Music and Audio Models page says "Music needs an Apple
Silicon Mac for now."

## Install the pack and the model

Open **Music and Audio > Models**. Two things are needed, once:

1. **The music pack** (about 1.6 GB). A banner at the top of the page offers **Install the music pack**. It
   lives in its own folder (`~/.chaosengine/music-venv`), so it cannot disturb the rest of the app. If you
   cancel or it fails, the banner shows the log and **Retry**.
2. **A model**, with **Download** on its row.

| Model | Download | While generating | Idle in memory | Speed once loaded |
|---|---|---|---|---|
| ACE-Step 1.5 turbo (the default) | 10.1 GB | about 22 GB | about 15 GB | about 1 s of compute per 10 s of audio |
| ACE-Step 1.5 XL turbo (large) | 20.0 GB, plus turbo | about 47 GB | about 40 GB | about 2 s per 10 s |

The XL model is for 64 GB Macs. It unloads every other model first. Each row also shows whether the model
fits this machine, its licence (MIT) and a note about the download. The turbo download includes a planner
model of 3.7 GB that the app does not load by default.

Figures are from a test on an M4 Max with 64 GB. Smaller Macs have not been measured.

## Make a track

Open **Music and Audio > Studio**. If the pack or the model is missing, the screen says so with one button, **Open Music and Audio
Models**.

The default view has:

- **Song** or **Instrumental**. Instrumental hides the lyrics box.
- **Describe the music.** Up to 512 characters, for example "warm lo-fi hip hop, vinyl crackle, mellow piano,
  85 bpm". Describe the sound, not an artist. If the text names a known artist, a warning says that can
  produce a copy of their style. It does not block you.
- **Lyrics** (songs only). Up to 4,096 characters. Mark sections with `[verse]` and `[chorus]`. A song needs
  lyrics; if you have none, switch to Instrumental.
- **Length.** Pills for 15 s, 30 s, 1 min and 2 min, and **Custom** for 10 to 600 seconds.
- The run estimate (time, memory and disk; see [Studio layout](studio-layout.md)) and **Make track**.
  Cmd or Ctrl plus Enter does the same.

**Options** holds the things you change less often. Its summary line shows the current values.

| Option | What it does |
|---|---|
| BPM | 30 to 300, or Auto. |
| Key | A major or minor key such as "A minor", or Auto. |
| Variations | 1, 2 or 4 tracks per run, with seeds `seed`, `seed+1` and so on. Each one adds the same time again. |
| Seed | Random by default. The same seed with the same settings gives the same file. |

**Expert** holds:

- **Model**, when more than one is installed.
- The output format, which is fixed: FLAC, 24-bit, 48 kHz, stereo.
- **Mark the file as AI-generated (metadata).** See [The AI tag](#the-ai-tag).
- **Unload the model after this many idle minutes.** The default is 10; 0 keeps it loaded.

There are no steps, guidance, sound-effect or reference-audio controls. Cover, repaint and reference audio
are not offered.

### What you get

Each finished track appears under **Latest** with a waveform player, a loop button and a line such as "Done in
33 s · 6 MB · seed 8142". Every track is saved to the Library on its own. The buttons are:

- **Open in Library.**
- **Again:** the same settings with a new random seed.
- **Vary:** the same settings with the next seed.
- **Send to storyboard.** See [Storyboard](storyboard.md).

## Time and memory

- About 1 s of compute per 10 s of audio once the model is loaded: a 60 s track takes 5 to 6 s, a 600 s track
  about 80 s.
- The first track after the app starts takes about 25 s longer, because the model loads.
- While generating, the music worker uses about 22 GB. Idle it holds about 15 GB until it unloads.
- Before a run the app checks that enough memory is free (23 GB, or 8 GB when the model is already loaded). If not, it
  says what to unload.
- An idle music worker is stopped when an image or video job needs the memory, and again after the idle
  minutes in Expert.

## Cancel and the queue

**Cancel** appears under the action while a track is being made. During composing it stops within about a
second. During the final audio decode there is no hook, so the worker is stopped after about 3 seconds and
the next track starts it again.

Music shares one queue with images, video, storyboard batches and agent jobs. Only one heavy job runs at a
time. If another job is running, the screen shows "Queued: yours starts after the job ahead of it."

## The Library

**Music and Audio > Library** keeps every track and sound effect. Each card has a player, the style prompt, the tags and these actions:

- **Rename.** Edit the title in place.
- **Add tag** and remove a tag. Tags are free text.
- **Make another like this.** Opens the Studio with that track's settings filled in, ready to change.
- **Send to storyboard.** See [Storyboard](storyboard.md).
- **Delete**, with a confirmation. You can select several cards and delete them together.

The toolbar has a filter to show **All**, **Music** or **Sound effects** (each with its count; selecting then
deleting only touches what is shown), search over titles, prompts and tags, a tag filter, a sort (newest, oldest, longest) and a
grid or list layout. Files are in `~/.chaosengine/music/library/` as `<id>.flac` with a `<id>.json` beside
it.

## The AI tag

Every track carries a machine-readable tag in the file's metadata: the app version, the model, the seed, the
length, BPM and key when given, the style prompt, and `machine_generated: true`. Lyrics and local paths are
never written into the file.

The tag is on by default. For private use, turn it off in **Studio > Expert** with **Mark the file as
AI-generated (metadata)**. The Expert summary then reads "AI tag off".

## Rights in lyrics

You are responsible for the rights in lyrics you type. The Studio says so under the lyrics box. Write your own
lyrics, or use words you may use. Describe a sound instead of naming an artist.

## Licence

ACE-Step 1.5 code and weights are MIT and are not gated. Check that a track is original before you publish it,
and say clearly when AI was involved.

## Troubleshooting

| What you see | What to do |
|---|---|
| "The music pack is not installed yet. Install it from Music and Audio Models (about 1.6 GB)." | Open **Music and Audio > Models** and press **Install the music pack**. |
| "No music model installed yet." | Open **Music and Audio > Models** and download ACE-Step 1.5 turbo. |
| "There is not enough free memory for this track. Try a shorter length." | Close other heavy apps or unload an image or video model, then press **Use a shorter length** or **Retry**. |
| "Music needs an Apple Silicon Mac for now." | The engine is MLX only. A CUDA lane is later work. |
| "The music pack could not be installed." | Open the **Technical details** log on the banner, then press **Retry**. The install needs Python 3.11 or 3.12. |
| "The backend did not answer." | Check that the app's backend is running, then try again. |

For the same tool for agents, see [Agent bridge](../features/agent-bridge.md) (`make_music`). For the backend
facts, see [Music backend](../features/music.md).
