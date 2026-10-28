## What's new in v0.9.9

### Highlights

- **Talk to your models.** Voice comes to ChaosEngineAI, all on your machine: nothing is uploaded.
  - **Voice calls in Chat.** Press Call and talk. The app hears when you stop, the model answers out loud, and you can talk over it to interrupt. Every turn is saved in the chat. An (i) next to Call lists anything still missing.
  - **Dictation and read-aloud.** A microphone button in the composer, and a speaker button on every reply (or have replies read automatically).
  - **Voice Studio.** Transcribe files or recordings in a queue, with speaker labels and translation to English. Edit the transcript, click a word to jump to it in the audio, rename speakers, then export TXT, Markdown, SRT, VTT or JSON, or send it to a chat. Speak turns long text into speech, with pauses and voice previews, saved as WAV or MP3.
  - **Voice Library** keeps every recording, transcript and generated clip. Items from the old Gallery move over the first time you open it.
- **LFM3-1.7B** (Liquid AI): a small model for agents and tool calling, with a 256K context and 16 languages. On Apple Silicon the official MLX 4-bit build is about 1 GB and writes at about 250 tokens per second on an M4 Max. On every platform the official GGUF (Q4_K_M, 1.1 GB) runs through llama.cpp, about 190 to 240 tokens per second on the same Mac. Tool calls work on both.
- **Every screen in all 10 languages.** The whole app, including Voice, Settings, the model panels and the Image and Video Studios, is now translated into German, French, Spanish, Russian, Portuguese (Brazil), Japanese, Korean and both Chinese variants. The translations were machine-made; corrections from native speakers are welcome.
- **Security updates.** The libraries that read PDFs and images, handle uploads and make web requests are on their latest versions, and so are the certificate list used for HTTPS and the TLS stack in the desktop shell. See "Security" below.

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
- **Tools work on GGUF models.** llama.cpp returns tool calls as structured data, which the app ignored, so a GGUF model asked to use a tool answered with nothing and the tool never ran. Fixed for every GGUF model.
- llama.cpp is now build b11938, which adds LFM3.

### Security

- PDFs you add to a chat are read with pypdf 6.21, which fixes a long list of parser problems (a crafted PDF could hang the app or use all its memory). The app now refuses to read PDFs with an older pypdf.
- Pillow (images) moves to 13, python-multipart (uploads) to 0.0.31, and aiohttp, urllib3, anyio, idna, msgpack, fsspec and filelock to their fixed versions.
- The certificate bundle (certifi) no longer trusts a certificate authority whose keys were compromised, in the Python runtime and in the desktop shell.
- The desktop shell's TLS and async libraries (rustls, tokio and others) and the chat code highlighter are updated.
- The desktop shell moves to Tauri 2.12, which brings the updater plugin 2.14 and a patched `zip` library (the Windows update unpacker). Every advisory the pre-release audit finds is now fixed.

### Fixes

- **Wan 2.2 TI2V 5B video** no longer comes out over-bright with colour blocks at the default settings (the default guidance now matches the model card).
- **Discover went blank** when you typed in the search box.
- Apple Silicon: a model that reasons after a tool result, even with Thinking Off, no longer shows that reasoning in its answer.
- A reasoning level that a model's chat template does not accept is mapped to the nearest one it does.
