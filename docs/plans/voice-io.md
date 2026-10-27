# Voice I/O plan

Status, 2026-10-27: P0, P1 and P2 are built on `feat/voice-io` and run on an M4 Max; Windows and Linux are not run yet. P3 and P4 are not started. Progress and open items: FU-117 in `CLAUDE.md`.

Differences from this plan in what was built:
- P2 turn detection runs Silero VAD over HTTP (`POST /api/voice/vad`, a second small worker), not a WebSocket: the backend has no WebSocket package on Windows / Linux.
- P2 has no live captions while the user is still talking; their words appear once the turn is transcribed (about 0.1 s after it ends).
- P1 voice messages are not tagged `inputMode: "voice"` yet.

## 1. Goal

Make voice a first-class way to use ChaosEngineAI, fully on-device:

- talk to any chat model and hear it answer, hands-free, with interruptions;
- dictate into the composer and have any reply read aloud;
- transcribe recordings and videos with timestamps, speakers and subtitles, then hand the transcript to a chat model;
- produce long-form speech, multi-speaker dialogue and (with consent) a cloned voice;
- expose the same engines to other programs through the local OpenAI-compatible server.

Where this puts us ahead of other local AI apps today: most offer at most one of these (a mic button, or read-aloud through the OS voice, or a call mode that needs separate servers). None of the ones reviewed combine in-chat voice, a hands-free call mode with barge-in, a transcription studio and an API from one install, with models tuned for Apple Silicon and a matching CUDA lane.

## 2. Where we start

What exists (backend `backend_service/voice_runtime.py`, `routes/voice.py`, frontend `src/features/voice/`):

- STT: Parakeet TDT 0.6B v3 and Whisper (mlx-whisper) on Apple Silicon; faster-whisper elsewhere.
- TTS: Kokoro 82M through mlx-audio (Apple Silicon) or kokoro-onnx; 5 voices exposed.
- Studio (record, transcribe, synthesize), Models (install, download) and Gallery tabs.

Blocking problems found in the review:

| # | Problem | Effect |
|---|---|---|
| B1 | No `com.apple.security.device.audio-input` entitlement and no `NSMicrophoneUsageDescription` | Recording fails, or the app is terminated, in the signed macOS build |
| B2 | mlx-whisper and parakeet-mlx call the `ffmpeg` CLI, which is not bundled and not on a Finder-launched app's PATH | Every transcription fails in the packaged app |
| B3 | Voice pack installs 7 packages into the shared extras overlay, one by one, no progress | Fragile, pollutes the overlay (FU-098), Japanese chain needs a compiler |
| B4 | `async def transcribe` runs blocking work on the event loop | All API calls stall during a transcription |
| B5 | Kokoro reloaded from disk on every request; STT and TTS models live in the FastAPI process | Slow, and a hang takes the backend down |
| B6 | No i18n, inline styles, no limits on text or upload size | Fails project rules |

## 3. Feature set

Phases are ordered so each one ships something usable. P0 and P1 are the next point release.

### P0 Foundation (required before anything is visible)

1. **Permissions.** Add the audio-input entitlement and a usage string (Info.plist via `bundle.macOS.infoPlist`). Check WebView2 (Windows) and WebKitGTK (Linux) microphone prompts; Linux needs the Tauri permission handler to allow `getUserMedia`.
2. **No ffmpeg dependency.**
   - Microphone: capture raw PCM with an AudioWorklet, resample to 16 kHz mono in the page, send WAV/PCM. No container decoding needed.
   - Files: decode with PyAV (ships its own FFmpeg libraries, LGPL) into a float32 array and pass arrays to Parakeet / Whisper, both of which accept arrays. Confirm PyAV wheel licence and size before adding.
3. **Voice worker.** `backend_service/voice_worker.py`, a subprocess like the MLX worker, speaking JSON lines: `load_stt`, `transcribe`, `transcribe_stream`, `load_tts`, `synthesize_stream`, `unload`, `status`. It keeps the chosen STT and TTS models warm, unloads after 10 idle minutes (setting), reports its memory to the Dashboard process list (so Stop works) and to the memory gate.
4. **Isolated environment.** Install into `~/.chaosengine/voice-venv`, like mflux and TensorFold, so voice packages never shadow the app's own transformers / mlx. One "Install voice pack" job with `PHASE:` progress lines, a size estimate and a retry. Japanese G2P becomes an optional extra.
5. **Correctness.** Run transcription off the event loop; cap text at 20 000 characters per request and uploads at 500 MB; label recordings with the real MIME type; confirm deletes; move every string through i18n.

### P1 Voice in Chat (catch up and pass)

1. **Mic in the composer.**
   - Click to dictate, click again to stop. Hold the hotkey (default: hold Space with the composer empty, or `Ctrl/Cmd+Shift+M`) for push-to-talk.
   - While recording: live level meter, timer, and partial text streaming into the composer (greyed until final).
   - Auto-send toggle (off by default): send when you stop talking.
   - Language: auto-detect, or a pinned language per user.
2. **Read aloud on every assistant message.**
   - Speaker button in the message actions; plays from the first sentence while the rest is synthesized.
   - The current sentence is highlighted; click any sentence to start there; stop, pause and speed controls in a small player bar.
   - Text clean-up before speech: drop code blocks (say "code block omitted"), read link text not URLs, expand units and numbers, skip tool-call JSON and reasoning.
3. **Auto-read replies** toggle per chat: replies are spoken as they stream.
4. **Voice settings** in Settings > Voice: default STT model, default voice, speed, auto-send, auto-read, push-to-talk key, idle unload time, microphone device.
5. Voice messages are stored like any other message, with `inputMode: "voice"` and the recording duration, so history shows a mic badge.

### P2 Conversation mode (the lead)

A hands-free call with the loaded chat model.

1. **Turn-taking.** Silero VAD (MIT, about 2 MB, ONNX) in the worker on 30 ms frames; end of turn after 700 ms of silence (setting). Echo cancellation and noise suppression requested from `getUserMedia`.
2. **Barge-in.** When the user starts speaking while the model talks, playback stops within 150 ms, generation is cancelled, and the partial reply is kept with a "interrupted" marker.
3. **Pipelined latency.** Final transcript -> chat stream -> sentence splitter -> TTS per sentence -> audio queue. Speech starts after the first sentence, not the full reply.
   - Target on an M4 Max with a 4B chat model: end of speech to first audio under 1.2 s. Budget: VAD tail 0.7 s is user-tunable; STT 0.15 s; first sentence from the LLM 0.25 s; first TTS chunk 0.1 s.
4. **Voice persona.** Conversation mode sends a short system addition ("answer in short spoken sentences, no markdown") and forces Thinking Off unless the user chose otherwise.
5. **Captions.** Live captions for both sides; the call lands in the chat thread as normal messages afterwards.
6. **UI.** Full-panel overlay (see section 5), keyboard: Space to mute, Esc to end.

### P3 Studio (beyond chat)

1. **Transcribe files.** Drag in audio or video (mp3, m4a, wav, flac, ogg, mp4, mov, mkv, webm). Queue with progress. Word timestamps, language auto-detect, translate-to-English (Whisper).
2. **Speakers.** Diarization so the transcript reads "Speaker 1 / Speaker 2", renameable. Candidate: NVIDIA Sortformer (CC-BY-4.0) or a pyannote pipeline; pick after a licence and accuracy check.
3. **Transcript editor.** Text synced to an audio player: click a word to seek; edits keep timestamps.
4. **Export.** TXT, Markdown, SRT, VTT, JSON (segments, words, speakers).
5. **Hand off to chat.** "Summarise", "Action items", "Meeting notes" buttons open a chat with the transcript attached as a document.
6. **Speak studio.** Long-form text with paragraph progress, pause markers, all Kokoro voices with one-click previews, export WAV and MP3.
7. **Dialogue.** Script with `[S1]` / `[S2]` lines rendered by a dialogue model (Dia2, Apache-2.0) into one file.
8. **Voice cloning (opt-in).** A 10-30 s reference clip produces a reusable voice (Chatterbox, MIT). Requires a consent checkbox stored with the voice, and the voice is marked "cloned" everywhere it appears.
9. **Library.** Replaces Gallery: search, filters (transcripts, speech, clones), source chat link, re-export, bulk delete.

### P4 Platform

1. **OpenAI-compatible endpoints** on the local server: `POST /v1/audio/transcriptions`, `POST /v1/audio/translations`, `POST /v1/audio/speech`; listed in the Server tab with curl examples.
2. **CLI:** `chaosengine-cli transcribe <file>`, `chaosengine-cli speak "<text>" -o out.wav`.
3. **System-wide dictation (later, opt-in).** A global shortcut records anywhere and types the text into the focused app. macOS needs Accessibility permission; treat as a separate feature flag.
4. **Agent tools:** `transcribe_file` and `speak` tools for the agent loop.

## 4. Models

Defaults are chosen for quality per byte and permissive licences. Rows marked "evaluate" need a licence and speed check on the M4 Max and the RTX 4090 before they ship.

### Speech-to-text

| Model | Platforms | Runtime | Languages | Size | Licence | Role |
|---|---|---|---|---|---|---|
| Parakeet TDT 0.6B v3 | Apple Silicon | parakeet-mlx | 25 European | 2.5 GB bf16 | CC-BY-4.0 weights | Default on Mac (fastest) |
| Parakeet TDT 0.6B v3 | Windows, Linux | sherpa-onnx or onnx-asr (evaluate) | 25 European | ~0.7 GB int8 | CC-BY-4.0 | Default elsewhere if it beats faster-whisper |
| Whisper large-v3-turbo | all | mlx-whisper / faster-whisper | 99 | 0.8-1.6 GB | MIT | Default for non-European languages, translate |
| Whisper small / base | all | same | 99 | 0.07-0.5 GB | MIT | Low-memory machines |
| Kyutai STT 1B, Voxtral Realtime | Apple Silicon | mlx-audio (evaluate) | EN/FR; multilingual | 1-4 GB | evaluate | Streaming partials in P2 if they beat chunked Parakeet |
| Silero VAD | all | onnxruntime | - | 2 MB | MIT | Turn detection |

### Text-to-speech

| Model | Platforms | Runtime | Size | Licence | Role |
|---|---|---|---|---|---|
| Kokoro 82M | all | mlx-audio / kokoro-onnx | 0.35 GB | Apache-2.0 | Default; all 50+ voices, 8 languages |
| Chatterbox (Turbo 350M EN, multilingual) | all | mlx-audio / PyTorch (evaluate) | 0.7-2 GB | MIT | Voice cloning, emotion |
| Dia2 1B/2B | all | evaluate | 2-4 GB | Apache-2.0 | Dialogue scripts |
| Kyutai Pocket TTS 100M | all (CPU) | evaluate | 0.2 GB | evaluate | Low-end cloning |

Excluded unless the user accepts a non-commercial licence: XTTS v2 (CPML), Fish Speech (CC-BY-NC-SA).

Every model row in the Voice Models tab shows: languages, size, licence, installed state, and speed measured on this machine (real-time factor for STT, time to first audio for TTS), recorded by a one-click "Benchmark" button.

## 5. UI

All strings through i18n; all colours from `styles.css` tokens; monochrome SVG icons.

### Composer

```
┌──────────────────────────────────────────────────────────────┐
│ Type a message…  (hold Space to talk)                         │
│ ▁▃▅▇▅▃▁  0:07   "so what I want is a list of…"   (greyed)     │
└──────────────────────────────────────────────────────────────┘
 Thinking [Off|Low|Med|High]  Temp  Samplers  KV  Tools   [mic] [Send]
```

- `[mic]` states: idle (outline), recording (filled, red ring pulses with level), processing (spinner), error (tooltip with the fix, for example "Allow microphone in System Settings > Privacy").
- Right-click `[mic]`: language, auto-send, open Voice settings.

### Message actions

```
Agent  …reply text…
[copy] [retry] [speaker] [fork]            Ternary-Bonsai-27B | 51 tok/s
```

Playback bar while speaking (sticky above the composer):

```
[pause] [stop]  ━━━━━━○──────  0:12 / 0:41   1.0x   Voice: Heart ▾
```

### Conversation mode

```
┌───────────────────────── Voice call ─────────────────────────┐
│                       ( animated orb )                        │
│                      Listening… / Thinking… / Speaking…       │
│                                                               │
│  You:   what's a good name for a coffee shop                  │
│  Model: How about "Second Pour"? It hints at…                 │
│                                                               │
│  [mute]   [end call]        Model: Qwen3.8 27B   Voice: Heart │
└───────────────────────────────────────────────────────────────┘
```

Entry point: a phone icon in the chat header, disabled with a tooltip until a chat model is loaded and the voice pack is installed.

### Voice section (sidebar)

- **Studio** with two tabs, Transcribe and Speak.
  - Transcribe: drop zone and record button on the left; queue below; editor on the right with player, speakers, timestamps; export menu and "Send to chat" actions.
  - Speak: text editor with paragraph markers; voice picker with preview buttons; speed; Generate; result list with play, export, save to Library.
- **Models:** STT, TTS, VAD, cloning groups; per model install, size, licence, languages, measured speed, Default radio.
- **Library:** search, filter chips, cards with play, open, export, delete.

### Empty and error states

- Voice pack not installed: one card, "Install voice pack (about 1.2 GB)", progress phases, Cancel.
- Microphone blocked: steps per OS with a button that opens the right settings pane (macOS `x-apple.systempreferences:` URL).
- Model downloading: inline progress where the user pressed the mic, not a silent wait.

## 6. Architecture and data

### Processes

```
WebView ──HTTP/WebSocket──> FastAPI (routes/voice.py)
                               │  JSON lines over stdin/stdout
                               ▼
                       voice_worker.py  (voice-venv)
                         STT model | TTS model | VAD
```

- FastAPI never imports mlx-audio, parakeet-mlx or faster-whisper.
- One worker per app; restarted on crash; memory shown on the Dashboard with Stop.
- On a 64 GB Mac the worker is counted by the chat memory check, so loading a 40 GB chat model warns when voice is warm.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/voice/runtime` | Capabilities, installed models, worker state |
| POST | `/api/voice/install` | Start voice pack install (background job) |
| GET | `/api/voice/install/status` | Phases and log tail |
| POST | `/api/voice/transcribe` | File or WAV upload, returns text + segments |
| WS | `/api/voice/stream` | PCM16 frames in; partial, final, VAD events out |
| POST | `/api/voice/speak` | Text in, WAV out |
| POST | `/api/voice/speak/stream` | Text in, chunked PCM out per sentence |
| POST | `/api/voice/benchmark` | Measure speed of installed models |
| GET/POST/DELETE | `/api/voice/library…` | Library items |
| POST | `/api/voice/clones` | Create a cloned voice (consent required) |
| POST | `/v1/audio/transcriptions`, `/v1/audio/speech` | OpenAI-compatible |

### Stored data (`~/.chaosengine/voice/`)

- `settings` (in the main settings file): `voice.sttModel`, `voice.ttsVoice`, `voice.speed`, `voice.autoSend`, `voice.autoRead`, `voice.pushToTalkKey`, `voice.idleUnloadMinutes`, `voice.micDeviceId`, `voice.vadSilenceMs`.
- `library/index.json`: `{id, kind: transcript|speech|dialogue, title, text, segments[{start,end,speaker,text,words[]}], voice, model, language, durationMs, createdAt, sourceChatId?, file}`.
- `library/<id>.wav|.mp3`, `cache/tts/<hash>.wav` (read-aloud cache keyed by text + voice + speed; size-capped, LRU).
- `clones/<id>/{reference.wav, voice.json}` with `{name, createdAt, consent: {confirmed: true, at}}`.
- Chat messages gain optional `inputMode: "voice"`, `audioDurationMs`.

## 7. Quality bar

- **Latency** (M4 Max, measured by the benchmark button and an E2E check): push-to-talk 10 s clip -> text under 0.6 s with Parakeet; read-aloud first audio under 0.4 s; conversation turn under 1.2 s.
- **Accuracy:** a small checked-in test set (5 clips, English, German, Japanese, noisy room, two speakers) with expected text; word error rate tracked per model.
- **Packaged-app check:** every release candidate records, transcribes and speaks in the signed macOS build, the Windows installer and the AppImage before Voice is listed in release notes.
- **Tests:** worker protocol, sentence splitter and text clean-up, VAD endpointing on recorded fixtures, library CRUD, route limits, i18n keys; E2E phase in `scripts/e2e_test_suite.py` for install, transcribe and speak.

## 8. Risks

| Risk | Mitigation |
|---|---|
| WebView differences (AudioWorklet, permissions) across WKWebView, WebView2, WebKitGTK | Capture through AudioWorklet only; test all three in P0 |
| Speaker audio feeding back into the mic | Browser echo cancellation; headphones hint; mute mic while speaking unless barge-in is on |
| Memory contention with large chat models | Worker counted in memory checks; idle unload; small default models |
| Licences (CC-BY attribution, cloning misuse) | Attribution in THIRD_PARTY_NOTICES and model cards; consent record; cloned voices labelled |
| Package churn (mlx-audio moves fast) | Isolated venv with pinned versions, bumped deliberately |

## 9. Milestones

| Release | Scope | Rough effort |
|---|---|---|
| v0.9.9 | P0 + P1 (mic, read-aloud, settings), Voice visible again | 4-6 days |
| v0.10 | P2 conversation mode, Studio file transcription + export | 6-8 days |
| v0.10.x | Diarization, editor, Library, OpenAI endpoints, CLI | 5-7 days |
| later | Cloning, dialogue, system-wide dictation, agent tools | 5+ days |

## 10. Decisions needed

| Question | Recommendation | Why |
|---|---|---|
| Isolated `voice-venv` or extras overlay | Isolated venv | Overlay already broke image/video once (FU-098) |
| PyAV for file decoding vs bundling ffmpeg | PyAV | Smaller, no CLI, no PATH problem |
| Parakeet on Windows/Linux via sherpa-onnx | Evaluate in P0, ship if faster than faster-whisper on the 4090 | Same model on every platform |
| Voice cloning in scope | Yes, opt-in, later phase | Clear differentiator; needs consent handling |
| System-wide dictation | Later, separate flag | Needs Accessibility permission and its own testing |
| Conversation mode default model | The loaded chat model, Thinking Off | No surprise loads |
