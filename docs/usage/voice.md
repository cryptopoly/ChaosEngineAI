# Voice

Voice in ChaosEngineAI runs on your machine; nothing is uploaded. It covers
dictation and read-aloud in Chat, voice calls with the loaded model, and Voice
Studio for transcribing and speaking files.

## Install the voice pack

Install the voice pack once from **Voice > Models** (about 2 GB, plus the
models you choose). It lives in its own folder, so it cannot disturb the rest
of the app. The same page lists the speech-recognition models and lets you
pick the one used for dictation. Each speech model row shows its download size, peak memory,
seconds per 10 s of speech, cold start and the machine it was measured on;
an installed row has **Use in Speak** (a cloning model has **Browse
voices**), which opens Voice Studio with that model's voice selected.

The quick (Kokoro) voices have one line here, with how many there are on this
computer, **Add Mandarin voices** while that add-on is missing, and **Browse
voices**, which opens Voice Studio > Voices. **Dictation and memory** holds the
dictation language, how long voice models stay loaded when idle, and **Unload
now**. How chat reads aloud and voice calls are set on Voice Studio > Voices.

![Voice > Models](../screenshots/voice-models.png)

| | Speech recognition | Speech |
|---|---|---|
| **Apple Silicon** | Parakeet (25 European languages, the fastest), Nemotron 3.5 (40 languages, detects the language by itself) or Whisper (99 languages) | Kokoro |
| **Windows and Linux** | Whisper through faster-whisper | Kokoro |

Voice on Windows and Linux is new and has had less testing than on the Mac.
Please report anything that does not work.

The first time you use the microphone, macOS asks for permission.

## In Chat

- **Dictation.** The microphone button in the composer.
- **Read-aloud.** A speaker button on every reply, or have replies read
  automatically. Chat reads with the chat voice, chosen with **Use in Chat**
  on Voice Studio > Voices (quick voices for now).
- **Voice calls.** Press **Call** and talk. The app hears when you stop, the
  model answers out loud, and you can talk over it to interrupt. Every turn is
  saved in the chat. An (i) next to Call lists anything still missing.

## Voice Studio

![Voice Studio with a transcript opened from the Library](../screenshots/voice-studio.png)

Voice Studio has three modes: **Speak**, **Transcribe** and **Voices**. It
opens on Speak the first time and then on the mode you used last; opening a
transcript from the Library always lands in Transcribe. The **Models** and
**Library** buttons in the Studio header go to the other two Voice pages, and
both of those have an **Open Studio** button back.

**Speak** turns long text into speech, with pauses and voice previews, saved
as WAV or MP3. The voice picker has five groups:

- **Recommended**: the best natural voice you have installed for the app's
  language, the best-graded quick voice, your newest cloned voice and Design a
  voice. While no natural voice is installed, the first card is **Install a
  better voice**: it names what Qwen3-TTS 1.7B CustomVoice adds, its download
  size and its measured memory and speed, and downloads it in place. **Not
  now** hides it until the app restarts.
- **Quick**: the Kokoro voices that come with the voice pack (fast, under
  1 GB, speed only). A language list, then a shortlist of the voices the model
  author grades best (for American English: Heart, Bella, Nicole, Fenrir,
  Michael, Puck); **All voices** shows the rest. Choosing Chinese offers the
  Mandarin add-on in place when it is missing. Japanese is not listed on
  Apple Silicon, where it cannot run.
- **Cloned**: the standing statement on cloning, your cloned voices, and
  **New cloned voice**, which opens the form on the Voices tab.
- **Designed**: a voice made from a written description. It changes on every
  run unless you set a seed under Advanced.
- **More**: the other natural voices, and **Get more voices**, which opens
  Voice Models.

Every card shows what it costs: memory and seconds per 10 s of speech for
natural, designed and cloned voices (measured on an Apple M4 Max, 64 GB, on
2026-11-08), or the download still needed. The line above Generate names the
model that will run with the same numbers, and **Manage in Models** goes to
the model list. A model that has not been measured says so instead of
showing a guess.

The style box takes free text: tone, pace and accent in your own words. For a
designed voice it is the description (age, gender, pitch, tone, accent, pace).
Speed and pitch are an ffmpeg step for every voice except quick voices, which
change speed themselves; when ffmpeg is missing the screen says so before
Generate.

Every result is saved to the Voice Library with the voice's name; **Saved to
Library** on the result opens the Library on Speech.

Speak starts on the voice you picked last time in Speak (kept apart from the
chat voice); before you have picked one, it starts on the best installed
natural voice for the app's language, else the best-graded quick voice. When
both CustomVoice models are installed the 1.7B one always runs, so typing a
style never swaps the model; the 0.6B one runs only when it is the only one
installed.

### More speech models

Voice Models also offers four more models on Apple Silicon, each measured on
an Apple M4 Max (64 GB, 2026-11-09) before it was offered:

| Model | What it adds | Download | Memory | Seconds per 10 s of speech |
|---|---|---|---|---|
| Chatterbox-Turbo | Its own English voice that performs sound tags; clones from 5 s | 1.2 GB | 1.7 GB | 2.1 |
| VoxCPM2 | Design a voice, or clone one, in 30 languages (found from the text); 48 kHz | 3.2 GB | 3.8 GB | 5.2 |
| Chatterbox Multilingual V3 | A default voice in 23 languages with an Expression dial; clones | 3.2 GB | 3.7 GB | 3.6 |
| Orpheus 3B | Eight expressive English voices (Tara, Leah, Jess, Leo, Dan, Mia, Zac, Zoe) with sound tags | 3.6 GB | 4.2 GB | 7.2 |

Their voices sit with the natural voices in the picker. What changes on Speak:

- **Sound tags.** With a Turbo or Orpheus voice, chips under the text box
  insert tags such as `[laugh]`, `[sigh]` or `[clear throat]` at the cursor;
  the model performs them instead of reading them. Every other voice skips
  them, with a note. Turbo also has ten style tags (`[whispering]`,
  `[angry]` ...) that are in the model but not documented by its authors, so
  results vary; put one at the start of a sentence.
- **Chatterbox.** A Speech language list (it starts on the app's language
  when Chatterbox speaks it, else English) and an Expression slider, calm to
  dramatic. Under Advanced, **Follow the voice sample closely**: try 0.3 for
  a fast speaker and 0 when the sample is in another language. A seed does
  not repeat a Chatterbox take exactly.
- **Designed voices** can run on VoxCPM2 as well as Qwen3-TTS VoiceDesign;
  with both downloaded, pick one in **Design model**. On a long text VoxCPM2
  designs the voice once and carries it into each later part.
- **Cloned voices** can use Qwen3-TTS Base, VoxCPM2, Chatterbox V3 or Turbo
  (pick in **Cloning model**). With VoxCPM2 the style box opens: leave it
  empty for the closest likeness, or type a style and VoxCPM2 follows it at a
  small cost in likeness. Turbo needs a sample of at least 5 seconds.

On Windows and Linux with an NVIDIA GPU, a **speech GPU add-on** for the same
models is being prepared and is not offered until it has been measured.

**Voices** is the voice manager: every voice in one list, filtered by kind
(Recommended, Quick, Natural, Cloned, Designed) and language. Each row has a
sample button (cloned and designed voices too; a sample is never kept),
**Use in Speak** and **Use in Chat**; cloned voices also have **Delete**. Chat
uses quick voices for now, so Use in Chat is greyed out on the others. Cards
at the top make a cloned voice, design one, add the Mandarin voices while
they are missing, or open Voice Models for more. **Chat and calls** at the
bottom shows the chat voice and holds reading speed, auto-read, auto-send,
the call pause and interrupt.

**Transcribe** takes a recording or audio and video files in a queue, with
speaker labels and translation to English. Edit the transcript, click a word
to jump to it in the audio, rename speakers, then export TXT, Markdown, SRT,
VTT or JSON, or send it to a chat.

### Clone a voice

Voice Studio > **Voices** > **Clone a voice** makes a voice from your own voice, or from someone
who has given you permission. Apple Silicon Macs only for now. It uses
Qwen3-TTS Base (0.6B is faster, 1.7B closer; Apache 2.0) by default; VoxCPM2,
Chatterbox Multilingual V3 and Chatterbox-Turbo can clone too. Download them
from Voice Models.

1. Click **New cloned voice**, choose whose voice it is and type the
   speaker's name. Public figures, children and people who have died are not
   allowed.
2. Add a voice sample: 3 to 30 seconds of the same person speaking clearly
   (10 to 20 seconds works best). Upload a WAV, MP3, M4A, FLAC or OGG file,
   or record it with the microphone.
3. Tick "This is my own voice, or the speaker has given me permission to
   clone it." and click **Save voice**.

The permission record (speaker's name, own voice or permission, date, a hash
of the sample, app version) is kept with the voice. **Delete** removes the
voice, its sample and the record. There is no import from web links and there
are no celebrity voices.

Every clip made with a cloned voice is saved to the Voice Library, labelled
as AI-made inside the file (WAV and MP3), and carries an inaudible watermark
(Meta's AudioSeal, MIT), whichever model made it. Neither can be
turned off; the watermark component (about 100 MB) installs with the cloning
model, and cloned speech is refused without it.
A local log, `~/.chaosengine/voice/clone-audit.jsonl`, records when voices
are made, used and deleted (ids, hashes and lengths, never the text); it is
never uploaded.

Agents (the agent bridge) and other computers cannot use cloned voices unless
you tick **Allow agents and other computers to use cloned voices**, which only
this window can change. Agents can never make a new cloned voice, and each
time they speak with one they must confirm that the speaker's permission
covers that use; the confirmation goes into the local log.

## Voice Library

The Voice Library keeps every recording, transcript and generated clip. Speech
cards name the voice and its kind (Quick, Natural, Designed, Cloned), mark
clips made with a cloned voice as AI-made, and have **Speak again**, which
opens Speak with the clip's text and voice. Items from the old Gallery move
over the first time you open it.
