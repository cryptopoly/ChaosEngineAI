# Voice studio UX audit (2026-11-25)

Scope: the Voice group (Models, Studio with Speak / Voices / Transcribe, Library) and the voice parts of Chat (mic,
read-aloud, calls). Audit from code. The dev server was started against an unused API port so nothing reached the
backend on 8876; the app did not mount outside the desktop shell (empty root), so no screen was seen. Nothing here
was checked in a running app. No code was changed.

## 1. Verdict

Each screen is careful and honest: sizes, measured memory, reasons on greyed controls, consent, watermark,
autosave. The confusion is the setup, which comes in three layers the user must assemble alone: the voice runtime,
then add-ons, then a model download, each in its own panel under its own word. On an NVIDIA PC the first install
gives quick voices only. Natural voices, cloning and design stay greyed until the user finds a 5.7 GB tick box,
and then they still need a model. The Studio is split by tool (Speak, Voices, Transcribe), not by task. Voices
mixes a catalogue, app-wide settings and the clone form. The fix: one setup card that knows what each task lacks,
plain task names, and controls hidden when the chosen voice cannot use them.

## 2. The flow today, and where people get lost

1. **Sidebar > Voice** ("Dictation, speech and voice chat") opens Models until the runtime is in, then Studio
   (`sidebarGroups.ts`). *Lost: calls happen only in Chat, and no Voice page says so or links there.*
2. **Models, before install.** "Voice runtime", subtitle "NVIDIA GPU. Everything voice needs that is not a model you
   choose." Add-on ticks: Kokoro (ticked), Mandarin, "Speech GPU add-on: Qwen3-TTS, Chatterbox, VoxCPM2 and
   Orpheus ...", "Speech-to-text GPU add-on: Qwen3-ASR, Parakeet ...". Only Kokoro is ticked
   (`VoicePackPanel.tsx` `wanted`). *Lost: family names instead of outcomes; nothing says "tick this to clone".*
3. **After install** the page grows to three long panels. *Lost: "Open Studio" sits in a panel header; the
   "Install a better voice" card is far down.*
4. **Dictation models.** Radios "Use for dictation", a language select in the header, a Word timings card, a
   footnote that an undownloaded pick "is fetched the first time you dictate". *Lost: the Studio calls this
   Transcribe. Without the speech-to-text add-on the GPU rows are absent, with no hint they exist.*
5. **Speech models.** A sortable table (Makes, Size, Fit, Status). On a PC without the add-on every row says "Tick it
   under Voice > Models > Voice runtime > Add-ons." *Lost: a path in words, not a button; "Makes" chips are
   features, not tasks.*
6. **Studio > Speak.** Text, voice row with "Change voice", Style box and six chips, Speed, Pitch, Advanced (Seed,
   Guidance), a model line, Generate. A new user starts on a quick voice, so Style and chips are greyed ("Quick
   voices have no style control"). *Lost: the first screen is half disabled. Pitch looks live but shows an ffmpeg
   error only after it moves (`ffmpegBlocked`). "Speech language" appears for some models only. On a PC without
   the add-on, Speak never says natural voices exist (the install card needs `modelsAvailable`).*
7. **Result.** Player, WAV, MP3, "Saved to Library", Send to Storyboard. *Lost: the next Generate replaces it;
   files are always "speech.wav"; nothing says where the file lives.*
8. **Studio > Voices.** "New cloned voice", "Design a voice", "Get more voices"; a strip of Speak, Chat read-aloud,
   Voice calls, Fallback with Change; filters, language, grouped lists with "Use for...". *Lost: "Fallback" is
   jargon. Greyed "New cloned voice" explains itself only in a tooltip; greyed "Design a voice" not at all. "Use
   for... > Speak" stays on Voices, so nothing seems to happen.*
9. **New cloned voice.** Statement, whose voice, names, cloning model (or a download), sample rules, file or
   Record, words spoken, permission tick, Save. *Lost: model before sample; after Save the form closes with no
   "Speak with it".*
10. **Transcribe.** Model and language selects, Translate, Label speakers, record-in-drop-zone, queue, editor.
    *Lost: undownloaded models are listed "(x GB download)" and the first job downloads silently; this language
    select does not save, while the Models one does; "Speaker labels need a voice runtime update" is a third kind
    of missing part.*
11. **Library** is clear. Only the Studio header links to it.
12. **Errors.** Worker failures arrive as raw text (`routes/voice.py` `_worker_error`, 503 with `str(exc)`). Out of
    GPU memory, missing model and a dead worker look alike.

## 3. Ranked changes (10)

"Keys" means new English strings in `src/locales/en/studio.json` (`chat.json` for the popover) with `null` in the
nine other locales, then `npm run i18n:validate`. Changing existing English also needs a new key.

| # | Change | Size | Risk | Keys |
|---|--------|------|------|------|
| 1 | One setup card, one Install | L | Medium | Yes (~25) |
| 2 | Plain words for tasks and parts | S | Low | Yes (~20) |
| 3 | Voices split: picker, Clone a voice, Chat's choices to Chat | M | Low | Yes (~10) |
| 4 | Speak shows only what the voice can do | M | Low | Yes (~10) |
| 5 | Clone ends on Speak; greyed buttons say why | S | Low | Yes (~6) |
| 6 | Plain errors with one action | M | Medium | Yes (~12) |
| 7 | "Pick this one" per task in Models | M | Low | Yes (~8) |
| 8 | One language rule per task | S | Low | Yes (~6) |
| 9 | Recent clips stay; named files; Show file | M | Low | Yes (~5) |
| 10 | Studio as landing; Chat link; no duplicate links | S | Low | Yes (~4) |

1. **Setup card.** Lists tasks (Quick speech, Natural voices and style, Clone or design, Speech to text on GPU,
   Calls), each Ready, Missing (what, how big) or Not on this machine (why), with one "Install what is missing"
   button that chains the existing install job and catalog download. Ticks move under "Choose parts". Files:
   `VoicePackPanel.tsx`, `VoiceModelsTab.tsx`, `VoiceStudioTab.tsx` (gate installs in place), new
   `setup/VoiceSetupCard.tsx`, `setup/voiceSetupLogic.ts`, `voicePackState.ts`, `runtimeSize.ts`.
2. **Words.** "Voice setup" for the runtime; "Optional parts" named by outcome ("Natural voices, cloning and design
   on this GPU, 5.7 GB"); "Backup voice" for Fallback; "Can do" for Makes; one name for speech to text (decision 3).
   Each stream renames in the files it owns.
3. **Voices split.** Studio tasks become Speak, Clone a voice, Transcribe. The catalogue lives in Speak's existing
   picker drawer. The Chat / Calls / Backup slots move into Chat's Voice popover, whose Change already routes here.
   Files: `voiceStudioMode.ts`, `VoicesScreen.tsx`, `VoicePicker.tsx`, `VoicePopover.tsx`, `NewVoiceForm.tsx`.
4. **Speak.** One "This voice can" line under the voice row (speed, style, sound tags, languages). Unusable controls
   hidden, replaced by one sentence and a button ("Style needs a natural voice. [Get one, 4.5 GB]"). Pitch moves
   to Advanced, disabled up front when ffmpeg is missing. Files: `SpeakScreen.tsx`, `SpeakControls.tsx`,
   `VoiceRow.tsx`, `speakLogic.ts`.
5. **Clone.** Order: permission, sample, name, model. After Save, "Voice saved. [Speak with it]". Greyed Clone and
   Design show their reason as text with the fix button. Files: `NewVoiceForm.tsx`, `VoicesScreen.tsx`,
   `CloneModelPicker.tsx`.
6. **Errors.** `_worker_error` adds a `code` (gpu_memory, model_missing, part_missing, ffmpeg_missing,
   unreadable_audio, worker_stopped, unknown). The UI shows a sentence and one action (Unload, Download, Open Voice
   setup, Retry), raw text under Details. Files: `routes/voice.py`, `voice_client.py`, `src/types/voice.ts`,
   `src/api/voice.ts`, new `voiceErrorText.ts`, `tests/test_voice_io.py`.
7. **Pick this one.** Each Models section opens with the one recommended row for this machine ("Qwen3-TTS 1.7B
   CustomVoice: natural voices and style, 4.5 GB, about 4.7 GB GPU memory"), the rest under "All models". Files:
   `SpeechModelsPanel.tsx`, `DictationModelsPanel.tsx`, `voiceModelsLogic.ts`.
8. **Language.** Transcribe gets "Remember as default" and the Models header select goes. Options say "finds the
   language itself" or "uses your choice". An undownloaded pick says "Downloads {size} before the first file".
   Speak shows the voice's languages. Files: `TranscribePanel.tsx`, `DictationModelsPanel.tsx`, `VoiceRow.tsx`.
9. **Results.** The session's last five clips stay under Generate; files named from the first words and the voice;
   "Show file" beside "Saved to Library". Files: `SpeakScreen.tsx`, `SpeakResultRow.tsx`, `audio.ts`.
10. **Navigation.** Voice always lands on Studio; drop the Studio header's Models / Library buttons (they repeat
    the sub-tabs); the Calls row says "happens in Chat [Open Chat]". Files: `sidebarGroups.ts`, `tabRedirects.ts`,
    `VoiceStudioTab.tsx`.

## 4. Proposed layout

```
Voice   [ Studio ]  Models  Library
+-- Voice setup ------------------------------------ NVIDIA RTX 4090, 24 GB --+
| Quick speech              Ready                                             |
| Natural voices, style     Missing: GPU speech part 5.7 GB + model 4.5 GB    |
| Clone or design a voice   Missing: same as above                            |
| Speech to text on GPU     Missing: GPU speech-to-text part 2.9 GB           |
| Voice calls               Ready, in Chat                     [Open Chat]    |
| [Install what is missing, 13.1 GB]                Choose parts...           |
+------------------------------------------------------------------------------+
[ Speak ]  [ Clone a voice ]  [ Transcribe ]
```

```
SPEAK
| Type or paste text...                                          |
(>) Heart  Quick · English (US), female              [Change voice]
    This voice can: speed. Style needs a natural voice. [Get one]
Speed [----o----] 1.0x                                Advanced v
[Generate audio]
Recent  0:12 "Welcome to the..."  Heart  [Play] [WAV] [MP3] [Send to project]
Saved to Library automatically. [Open Library] [Show file]
```

```
CLONE A VOICE
1 Permission  (o) My own voice  ( ) Someone who gave permission  [x] I confirm
2 Sample      [Record] or [Choose file]   0:14 ok (10 to 20 s is best)
3 Name        [ ]
4 Model       (o) default cloning model   ( ) others downloaded
[Save voice]  ->  Voice saved. [Speak with it]
```

```
TRANSCRIBE
Model [Qwen3-ASR 1.7B, finds the language itself v]  Language [Detect v]  [ ] Remember
( o ) Record, or drop audio and video files here
interview.m4a   Transcribing 0:41   [Cancel]          (editor below, unchanged)
```

Chat's Voice popover gains "Voices in Chat: Read-aloud, Calls, Backup", each with Change opening the picker there.

## 5. Quick wins (a day)

- Reason as text under greyed "New cloned voice" and "Design a voice", with the fix button (`VoicesScreen.tsx`).
- "Speak with it" after Save voice (`NewVoiceForm.tsx` `onCreated`, `VoicesScreen.tsx`).
- Turn "Tick it under Voice > Models > ..." into a button that scrolls to the tick box.
- Disable Pitch with the ffmpeg reason up front (`SpeakScreen.tsx`).
- For a quick voice, one line and a button instead of the greyed Style box and chips.
- Name downloads from the first words (`SpeakResultRow.tsx`).
- "Fallback" becomes "Backup voice".
- Pre-tick the speech GPU add-on where offered (`VoicePackPanel.tsx` `wanted`), if decision 1 says so.

## 6. Build order for agent streams

One worktree per stream; file sets do not overlap; English through `defaultValue` only, no locale edits.

1. **A, alone first:** error codes (change 6): `routes/voice.py`, `voice_client.py`, `src/types/voice.ts`,
   `src/api/voice.ts`, new `voiceErrorText.ts`, `tests/test_voice_io.py`, and an E2E check in
   `scripts/e2e_test_suite.py` (feature gate).
2. **Then four in parallel:**
   - **B, setup and navigation** (1, 10): `VoicePackPanel.tsx`, `VoiceModelsTab.tsx`, `VoiceStudioTab.tsx`,
     `voiceStudioMode.ts`, new `setup/`, `voicePackState.ts`, `runtimeSize.ts`, `sidebarGroups.ts`,
     `tabRedirects.ts`.
   - **C, Speak** (4, 9, Speak half of 8): `speak/SpeakScreen.tsx`, `SpeakControls.tsx`, `SpeakResultRow.tsx`,
     `VoiceRow.tsx`, `speakLogic.ts`, `audio.ts`.
   - **D, voices and clone** (3, 5): `speak/VoicesScreen.tsx`, `VoicePicker.tsx`, `clone/`, `VoicePopover.tsx`;
     B makes the mode change in `voiceStudioMode.ts`.
   - **E, models and transcribe** (7, rest of 8): `models/`, `studio/TranscribePanel.tsx`, `voiceLanguages.ts`.
   - Each stream applies change 2's words in its own files.
3. **F, last, alone:** `npm run i18n:extract`, the nine locales, `src/styles.css`, i18n validate, vitest, tsc,
   pytest, and screenshots of each task on a Mac, an NVIDIA PC and a CPU-only PC in the running app.

## 7. Decisions for the owner

1. **First install on an NVIDIA PC:** minimal (about 2.6 GB, quick voices only) or everything this GPU can run
   (about 16 GB with the recommended speech model)? Recommendation: two presets, "Basic" and "Full for this GPU",
   Full preselected when the GPU fits. Most confusion starts from the minimal default.
2. **Where Chat's voice choices live:** on Voices, or in Chat's Voice popover? Recommendation: the popover; they
   are Chat settings, and Voices becomes a plain catalogue.
3. **One name for speech to text:** "Transcribe", "Dictation" or "Speech to text"? Recommendation: "Speech to
   text" for models and setup, "Transcribe" for the Studio task, "Dictate" for the Chat mic.

## Summary

1. The screens are careful; the confusion is the layered setup: runtime, add-ons, then models.
2. On NVIDIA PCs the first install gives quick voices only; cloning and natural voices hide behind a 5.7 GB tick.
3. Fix: one setup card that lists tasks, says what is missing and installs it with one button.
4. Studio tasks become Speak, Clone a voice, Transcribe; the catalogue folds into Speak's picker.
5. Chat's voice choices (read-aloud, calls, backup) move to Chat's Voice popover.
6. Speak hides controls the chosen voice cannot use and says why, with a fix button.
7. A saved clone ends on Speak with that voice; greyed buttons explain themselves.
8. Worker errors get codes and plain sentences; raw text stays under Details.
9. Quick wins fit in a day; then one backend stream, four UI streams, one locale stream.
10. Not verified in a running app: the UI did not mount outside the desktop shell; the backend was untouched.
