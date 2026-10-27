## What's new in v0.9.9

### Highlights

- **Talk to your models.** Voice comes to ChaosEngineAI, all on your machine: nothing is uploaded.
  - **Voice calls in Chat.** Press Call and talk. The app hears when you stop, the model answers out loud, and you can talk over it to interrupt. Every turn is saved in the chat. An (i) next to Call lists anything still missing.
  - **Dictation and read-aloud.** A microphone button in the composer, and a speaker button on every reply (or have replies read automatically).
  - **Voice Studio.** Transcribe files or recordings in a queue, with speaker labels and translation to English. Edit the transcript, click a word to jump to it in the audio, rename speakers, then export TXT, Markdown, SRT, VTT or JSON, or send it to a chat. Speak turns long text into speech, with pauses and voice previews, saved as WAV or MP3.
  - **Voice Library** keeps every recording, transcript and generated clip. Items from the old Gallery move over the first time you open it.
- **LFM3-1.7B** (Liquid AI): a small model for agents and tool calling, with a 256K context and 16 languages. The official MLX 4-bit build is about 1 GB and writes at about 250 tokens per second on an M4 Max. Tool calls work.

### Voice: what you need

- Install the voice pack once from **Voice > Models** (about 2 GB, plus the models you choose). It lives in its own folder, so it cannot disturb the rest of the app.
- **Apple Silicon:** Parakeet (25 European languages, the fastest) or Whisper (99 languages) for speech recognition, Kokoro for speech.
- **Windows and Linux:** Whisper (through faster-whisper) and Kokoro.
- Voice on Windows and Linux is new and has had less testing than on the Mac. Please report anything that does not work.
- The first time you use the microphone, macOS asks for permission.

### Chat

- **Unload / Cancel load** button in the chat header.
- Chat no longer asks to load the model again before every message.
- Tool calls written in XML (LFM3, Qwen3-Coder) now run. Before, they showed as text.

### Fixes

- **Discover went blank** when you typed in the search box.
- Apple Silicon: a model that reasons after a tool result, even with Thinking Off, no longer shows that reasoning in its answer.
- A reasoning level that a model's chat template does not accept is mapped to the nearest one it does.
