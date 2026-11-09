# Voice

Voice in ChaosEngineAI runs on your machine; nothing is uploaded. It covers
dictation and read-aloud in Chat, voice calls with the loaded model, and Voice
Studio for speaking text, choosing voices and transcribing files.

One rule says where things live: **a model is on Voice > Models, a voice is on
Voice Studio > Voices, and a setting sits next to the thing it changes.**

## Voice > Models

![Voice > Models](../screenshots/voice-models.png)

### The voice runtime

Install the **voice runtime** once (about 2.3 GB with the Kokoro quick voices,
about 2 GB without). It is everything voice needs that is not a model you
choose, in its own folder so it cannot disturb the rest of the app. The panel
lists its parts with their size on disk: the speech packages (about 1.9 GB),
the Kokoro quick voices (0.3 GB, when installed), turn detection for calls and
the speaker-label models (0.04 GB together). Dictation and speech models are
separate downloads on the same page.

- **In memory now** names the voice model the speech worker holds, with
  **Unload**, and the minutes of idle time after which it is freed.
- **Add-ons** are tick boxes: **Kokoro quick voices** (about 330 MB, ticked
  by default), **Mandarin Chinese quick voices** (8 voices, about 60 MB, needs
  Kokoro) on Apple Silicon, and on Windows and Linux with an NVIDIA GPU the
  speech GPU add-on when it is offered. Ticked before the install, they install
  with the runtime; afterwards, ticking installs the add-on and unticking
  removes it (refused, with the reason, while a voice model is loading or
  speaking). Kokoro is the only voice that runs fast without a GPU and the
  first voice a new user hears; leave it out if you only want other speech
  models.
- **Troubleshooting** has **Reinstall** and **Remove voice runtime**. Remove
  asks first and says what stops working (dictation, read-aloud, voice calls
  and Voice Studio) until the runtime is installed again. Downloaded dictation
  and speech models, cloned voices and the Library always stay; a tick box
  also deletes the runtime's own downloads (Kokoro, turn detection, speaker
  labels).

(The voice runtime was called the voice pack before v0.10.0; settings and the
install folder are unchanged.)

| | Dictation | Speech |
|---|---|---|
| **Apple Silicon** | Parakeet (25 European languages, the fastest), Nemotron 3.5 (40 languages, detects the language by itself) or Whisper (99 languages) | Kokoro (optional add-on), and the speech models below |
| **Windows and Linux** | Whisper through faster-whisper | Kokoro (optional add-on) |

Voice on Windows and Linux is new and has had less testing than on the Mac.
Please report anything that does not work. The first time you use the
microphone, macOS asks for permission.

### Dictation models

The speech-to-text models used by the microphone in Chat, voice calls and
Transcribe. The dictation **Language** sits in the panel header (Whisper uses
it; Parakeet and Nemotron find the language themselves). Each card has **Use
for dictation**, **Download**, and on a downloaded model a **···** menu with
**Delete model**, which asks first. A model you pick but have not downloaded
is fetched the first time you dictate.

### Speech models

The speech models are on the same table as Image, Video and Music Models:
search, Installed / Available / Downloading filters, a Makes filter, and sort
by family, fit, size, name or status. Each row has what it makes (Natural
voices, Style, Clone, Design, Sound tags, Expression), its download size, its
**Fit** on this machine from the measured peak memory, and its status. The row
detail has the measured memory and speed, cold start, licence and repository.

- **Kokoro** is the first row: the quick voices, an add-on of the runtime,
  fast on any machine. **Browse voices** opens Voices; it has no Delete
  (untick it under the runtime's add-ons instead). Left out, the row says
  **Not installed** and where to tick it.
- Downloaded rows have **Use in Speak** (a model that only clones has
  **Browse voices**), a **···** menu with the licence, and **Delete model**.
  Delete asks first, naming the model and its size; it removes the model and
  the companion files only it uses, keeps any companion another downloaded
  model still needs (the confirm lists them), and keeps the watermark
  component while any downloaded model can clone. It never deletes a cloned
  voice or a Library item. A delete is refused, with the reason, while a voice
  model is loading or speaking (read-aloud, a call, Speak, a transcript) or
  while a speech model downloads; it works again once that has finished.
- While no natural voice is downloaded, **Install a better voice** above the
  table offers Qwen3-TTS 1.7B CustomVoice with its size and measured cost.

| Model | What it adds | Download | Memory | Seconds per 10 s of speech |
|---|---|---|---|---|
| Qwen3-TTS 1.7B CustomVoice | Nine natural voices that follow a written style | 3.1 GB | 3.5 GB | 2.5 |
| Qwen3-TTS 1.7B VoiceDesign | A voice from a written description | 3.1 GB | 3.5 GB | 2.5 |
| Qwen3-TTS 0.6B / 1.7B Base | Cloning (0.6B is the default) | 2.5 / 3.8 GB | 4.3 / 6.5 GB | 3.2 / 3.6 |
| Chatterbox-Turbo | Its own English voice that performs sound tags; clones from 5 s | 1.2 GB | 1.7 GB | 2.1 |
| VoxCPM2 | Design a voice, or clone one, in 30 languages (found from the text); 48 kHz | 3.2 GB | 3.8 GB | 5.2 |
| Chatterbox Multilingual V3 | A default voice in 23 languages with an Expression dial; clones | 3.2 GB | 3.7 GB | 3.6 |
| Orpheus 3B | Eight expressive English voices (Tara, Leah, Jess, Leo, Dan, Mia, Zac, Zoe) with sound tags | 3.6 GB | 4.2 GB | 7.2 |

Measured on an Apple M4 Max (64 GB, 2026-11-08 and 2026-11-09). A model that
has not been measured says so instead of showing a guess. On Windows and Linux
with an NVIDIA GPU, a speech GPU add-on for the same models is being prepared
and is not offered until it has been measured.

## Voice Studio

![Voice Studio > Speak](../screenshots/voice-studio.png)

Voice Studio has three modes, in this order: **Speak**, **Voices** and
**Transcribe**. It opens on Speak the first time and then on the mode you used
last; opening a transcript from the Library always lands in Transcribe. The
**Models** and **Library** buttons in the Studio header go to the other two
Voice pages, and both of those have an **Open Studio** button back.

### Speak

Speak turns long text into speech, saved as WAV or MP3 and to the Voice
Library. The screen reads top to bottom: the text, **the voice row**, the
controls of that voice, speed and pitch, Advanced, the model line and
**Generate audio**, all on the first screen.

The **voice row** names the voice in use, its kind (Natural, Quick, Cloned,
Designed), one line about it, its cost chip and a sample button. **Change
voice** (Enter or Space work too) opens the picker in a drawer; a click picks
a voice and closes it, the arrow keys move between voices, and Escape closes
it with focus back on Change voice. The picker has five groups:

- **Recommended**: the best natural voice you have installed for the app's
  language, the best-graded quick voice, your newest cloned voice and Design a
  voice.
- **Quick**: the Kokoro voices, an add-on of the runtime (fast, under 1 GB,
  speed only). A language list, then the shortlist the model author grades
  best (for American English: Heart, Bella, Nicole, Fenrir, Michael, Puck);
  **All voices** shows the rest. Choosing Chinese offers the Mandarin add-on
  in place when it is missing.
- **Cloned**: your cloned voices and **New cloned voice**, which opens the
  form on Voices.
- **Designed**: a voice made from a written description; it changes on every
  run unless you set a seed under Advanced.
- **More**: the other natural voices, and **Get more voices**, which opens
  Voice Models.

The controls of the voice sit right under the voice row: **Cloning model** or
**Design model** when two are downloaded, the **sound tag** chips for Turbo
and Orpheus voices (inserted at the cursor and performed, not read), the style
box with its chips (free text: tone, pace, accent; for a designed voice it is
the description), and for Chatterbox the **Speech language** list and the
**Expression** slider. Under Advanced, **Follow the voice sample closely**
(Chatterbox: try 0.3 for a fast speaker, 0 for a sample in another language).
With VoxCPM2 a cloned voice can follow a style at a small cost in likeness.

While no natural voice is installed, **Install a better voice** under the
voice row names what Qwen3-TTS 1.7B CustomVoice adds and downloads it in
place; **Not now** hides it until the app restarts. Speed and pitch are an
ffmpeg step for every voice except quick voices; when ffmpeg is missing the
screen says so before Generate.

Speak starts on the voice you picked last time in Speak (kept apart from the
chat voice); before that, on the best installed natural voice for the app's
language, else the best-graded quick voice.

### Voices

Voices is where you choose which voice each part of the app uses.

- **Main voices** at the top: **Speak**, **Chat read-aloud**, **Voice
  calls** and **Fallback**, each with the voice's name and kind, its
  first-words time in Chat, and **Change**, which opens the picker drawer
  showing only the voices that use allows. The call voice starts as the chat
  voice ("same as Chat").
- **Fallback** is the voice that reads when the chosen chat or call voice
  cannot speak (its model is not downloaded, or it fails to load); Chat says
  so. Any installed voice that may read aloud can be the fallback (a cloned
  voice is used for read-aloud only, never in calls); its drawer also has
  **No fallback voice**. It starts as Kokoro's Heart, which counts only while
  Kokoro is installed. With no fallback, read-aloud and calls show "The
  chosen voice could not speak and no fallback voice is set. Pick one on
  Voice Studio > Voices." instead of switching voice. The line under the
  cards names the fallback in effect or says none is set.
- Without Kokoro, the Quick group says the quick voices are not installed,
  with **Open Voice Models**.
- **New cloned voice**, **Design a voice** and **Get more voices** in the
  header.
- **The list**, filtered by kind (All, Natural, Quick, Cloned, Designed) and
  language, grouped like the picker: Natural, Quick (the graded shortlist,
  **All voices**, and **Add Mandarin voices** while it is missing), Cloned,
  Designed. Each row has a sample button (a sample is never kept), badges for
  the uses it holds, its cost and measured first-audio time, and one **Use
  for…** menu: Speak, Chat read-aloud, Voice calls, Fallback. A use a voice cannot have
  is greyed with the reason: Describe a voice needs a description each time,
  cloned voices are not used in calls, a voice whose model is not downloaded
  offers **Download** instead. Cloned voices also have **Delete**.

### Transcribe

Transcribe takes a recording, or audio and video files dropped or chosen (WAV,
MP3, M4A, FLAC, OGG, MP4, MOV and more), in a queue, with speaker labels and
**Translate to English (Whisper Small or Base)**. Transcripts are kept in the
Library. Edit the transcript, click a word to jump to it in the audio, rename
speakers, then export TXT, Markdown, SRT, VTT or JSON, or send it to a chat.

### Clone a voice

Voice Studio > **Voices** > **New cloned voice** makes a voice from your own
voice, or from someone who has given you permission. Apple Silicon Macs only
for now. It uses Qwen3-TTS Base by default; VoxCPM2, Chatterbox Multilingual
V3 and Chatterbox-Turbo can clone too. Download them from Voice Models.

The form opens with the rules: only your own voice or that of someone who gave
permission; not public figures, children or people who have died; every clip
is labelled as AI-made and watermarked, and neither can be turned off.

1. Choose whose voice it is and type the speaker's name.
2. Add a voice sample: 3 to 30 seconds of the same person speaking clearly
   (10 to 20 seconds works best). Upload a WAV, MP3, M4A, FLAC or OGG file,
   or record it with the microphone.
3. Tick "This is my own voice, or the speaker has given me permission to
   clone it." and click **Save voice**.

The permission record (speaker's name, own voice or permission, date, a hash
of the sample, app version) is kept with the voice. **Delete** removes the
voice, its sample and the record; deleting a model never deletes a cloned
voice.

Every clip made with a cloned voice is saved to the Voice Library, labelled
as AI-made inside the file (WAV and MP3), and carries an inaudible watermark
(Meta's AudioSeal, MIT), whichever model made it. The watermark component
(about 100 MB) installs with the first cloning model and stays while any
downloaded model can clone; cloned speech is refused without it. A local log,
`~/.chaosengine/voice/clone-audit.jsonl`, records when voices are made, used
and deleted (ids, hashes and lengths, never the text); it is never uploaded.

Agents (the agent bridge) and other computers cannot use cloned voices unless
you turn on **Settings > Agent access > Allow voice cloning**, which only this
window can change. Agents can never make a new cloned voice, and each time
they speak with one they must confirm that the speaker's permission covers
that use. Agents cannot delete models or remove the runtime.

## In Chat

- **Dictation.** The microphone button in the composer.
- **Read-aloud.** A speaker button on every reply, or have replies read
  automatically. The bar above the composer names the voice reading, with a
  note when the fallback voice reads instead.
- **Voice calls.** Press **Call** and talk. The app hears when you stop, the
  model answers out loud, and you can talk over it to interrupt. Every turn is
  saved in the chat.
- **Voice** (next to Call) opens a popover: the read-aloud voice, the call
  voice, the fallback voice (or "None set") and the dictation model, each with
  **Change** (Voices, or Models for
  dictation); **Read every chat reply aloud**, **Send as soon as dictation
  finishes**, **Reading speed**, the **pause that ends your turn** in a call
  and **interrupt the model by speaking**; and what a call still needs, with
  **Install what is missing** when Voice Models can fix it.

Any voice that runs on this machine can read chat replies aloud or speak in
calls; cloned voices read aloud only (watermarked and labelled, not saved to
the Library) and are not used in calls; Describe a voice cannot be a chat
voice. One speech model is loaded at a time, next to the chat model, and freed
after the idle time set on the runtime panel.

What each voice costs in Chat, measured with
`scripts/measure-voice-models.py --chat` on an Apple M4 Max (64 GB,
2026-11-09), two runs, the slower kept. "First audio" is the time from the
request to the first sentence of a reply being ready; "cold" includes starting
the speech worker and loading the model.

| Voice | First audio (warm) | First audio (cold) | Speech worker memory |
|---|---|---|---|
| Kokoro (quick voices) | 0.13 s | 5.7 s | 0.9 GB |
| Chatterbox-Turbo | 0.6 s | 3.8 s | 1.7 GB |
| Qwen3-TTS 1.7B CustomVoice (Ryan and the other presets) | 1.3 s | 4.1 s | 3.5 GB |
| VoxCPM2 (designed or cloned) | 1.7 s | 3.8 s | 3.6 GB |

A voice whose warm first audio is over 2 s is labelled **Slow for calls**
(still allowed); none of the measured voices is.

## Voice Library

The Voice Library keeps every recording, transcript and generated clip. Filter
by kind (Transcripts, Speech), by the voice a clip was made with (Any voice,
Natural, Quick, Cloned, Designed) and by project. Speech cards name the voice
and its kind, mark clips made with a cloned voice as AI-made, and have **Speak
again**, which opens Speak with the clip's text and voice, and **Send to
project**. Items from the old Gallery move over the first time you open it.
