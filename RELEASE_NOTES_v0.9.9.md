## What's new in v0.9.9

### Highlights

- **Talk to your models.** Voice comes to ChaosEngineAI, all on your machine: nothing is uploaded.
  - **Voice calls in Chat.** Press Call and talk. The app hears when you stop, the model answers out loud, and you can talk over it to interrupt. Every turn is saved in the chat. An (i) next to Call lists anything still missing.
  - **Dictation and read-aloud.** A microphone button in the composer, and a speaker button on every reply (or have replies read automatically).
  - **Voice Studio.** Transcribe files or recordings in a queue, with speaker labels and translation to English. Edit the transcript, click a word to jump to it in the audio, rename speakers, then export TXT, Markdown, SRT, VTT or JSON, or send it to a chat. Speak turns long text into speech, with pauses and voice previews, saved as WAV or MP3.
  - **Voice Library** keeps every recording, transcript and generated clip. Items from the old Gallery move over the first time you open it.
- **Nemotron 3.5 Bolt** (NVIDIA): a 4B-effective assistant model for voice and device assistants, better at multi-turn tool use and following instructions. The official 6 GB build runs on macOS, Windows and Linux through llama.cpp. NVIDIA's own DFlash drafter downloads with it, and ticking speculative decoding takes an M4 Max from about 62 to about 160 tokens per second (2.6 to 3 times faster). An 8-bit build is also listed.
- **Nemotron 3.5 speech recognition** (Apple Silicon): a 0.6B model that transcribes 40 languages and detects the language by itself. On an M4 Max a 5-second clip takes about 0.14 seconds once the model is loaded. Pick it in **Voice > Models**.
- **Granite Switch 4.1** (IBM, Apache 2.0): the 8B and 30B Granite models in one checkpoint with built-in decision and safety adapters. The app runs the chat model.
- **Iris-3B Turbo** on Apple Silicon: Apache 2.0, four steps, a 1024 by 1024 image in about 18 seconds on an M4 Max using 7.5 GB of memory. The MLX image engine moves to mflux 0.24; update it from the Image Studio.
- **LFM3-1.7B** (Liquid AI): a small model for agents and tool calling, with a 256K context and 16 languages. On Apple Silicon the official MLX 4-bit build is about 1 GB and writes at about 250 tokens per second on an M4 Max. On every platform the official GGUF (Q4_K_M, 1.1 GB) runs through llama.cpp, about 190 to 240 tokens per second on the same Mac. Tool calls work on both.
- **LTX-2.5 video with sound on Apple Silicon** joins LTX-2.3 and becomes the default LTX model: the same prompt and seed took 128 seconds against 155 for 2.3 on an M4 Max (4 seconds, 768 by 512, 17 GB of memory) and looks cleaner. It is a gated Hugging Face repo: accept its licence on the model page and give your token read access to public gated repos, then download it from the Video Studio. The notes below describe the engine both share.
- **LTX-2.3 video with sound on Apple Silicon** replaces Wan 2.2 TI2V 5B. On its own MLX engine a 4-second 768 by 512 clip with stereo sound takes about 2.5 minutes on an M4 Max (the same model through the older route took 19 minutes for a 1.4-second clip). Install the engine from the Video Studio (about 1 GB), then download **LTX-2.3 distilled q4** (28 GB, including its text encoder). Sizes are multiples of 64, and the steps, guidance and negative prompt controls are hidden because this pipeline has none. The Wan 2.2 5B models are removed from the catalog.
- **Every screen in all 10 languages.** The whole app, including Voice, Settings, the model panels and the Image and Video Studios, is now translated into German, French, Spanish, Russian, Portuguese (Brazil), Japanese, Korean and both Chinese variants. The translations were machine-made; corrections from native speakers are welcome.
- **Security updates.** The libraries that read PDFs and images, handle uploads and make web requests are on their latest versions, and so are the certificate list used for HTTPS and the TLS stack in the desktop shell. See "Security" below.

### Voice: what you need

- Install the voice pack once from **Voice > Models** (about 2 GB, plus the models you choose). It lives in its own folder, so it cannot disturb the rest of the app.
- **Apple Silicon:** Parakeet (25 European languages, the fastest), Nemotron 3.5 (40 languages, automatic language detection) or Whisper (99 languages) for speech recognition, Kokoro for speech.
- **Windows and Linux:** Whisper (through faster-whisper) and Kokoro.
- Voice on Windows and Linux is new and has had less testing than on the Mac. Please report anything that does not work.
- The first time you use the microphone, macOS asks for permission.

### Chat

- **Unload / Cancel load** button in the chat header.
- Chat no longer asks to load the model again before every message.
- Tool calls written in XML (LFM3, Qwen3-Coder) now run. Before, they showed as text.
- **Tools work on GGUF models.** llama.cpp returns tool calls as structured data, which the app ignored, so a GGUF model asked to use a tool answered with nothing and the tool never ran. Fixed for every GGUF model.
- llama.cpp is now build b11981, which adds LFM3 and carries Granite Switch and the Nemotron 3.5 Bolt architecture.

### Security

- PDFs you add to a chat are read with pypdf 6.21, which fixes a long list of parser problems (a crafted PDF could hang the app or use all its memory). The app now refuses to read PDFs with an older pypdf.
- Pillow (images) moves to 13, python-multipart (uploads) to 0.0.31, and aiohttp, urllib3, anyio, idna, msgpack, fsspec and filelock to their fixed versions.
- The certificate bundle (certifi) no longer trusts a certificate authority whose keys were compromised, in the Python runtime and in the desktop shell.
- The desktop shell's TLS and async libraries (rustls, tokio and others) and the chat code highlighter are updated.
- The desktop shell moves to Tauri 2.12, which brings the updater plugin 2.14 and a patched `zip` library (the Windows update unpacker). Every advisory the pre-release audit finds is now fixed.

### Fixes

- **A setting a model cannot use no longer breaks the reply.** A repeat penalty of 0 (llama.cpp refused it), XTC with a threshold of 0 (the text turned to noise), an image sent to a model that cannot read images, a prompt larger than the context, and a reply that spent its whole budget thinking all used to fail or come back empty with no reason. They are now corrected or explained in the reply's note. Image and video models are only given the arguments they accept, sizes and frame counts are rounded to what the model needs (multiples of 16, LTX frames 8k+1, Wan 4k+1), and speech recognition falls back to automatic language detection when the language does not fit.
- **Options for hardware you do not have are hidden.** CUDA-only boosters and toggles no longer show on a Mac or a PC without an NVIDIA GPU, MLX rows no longer show on Windows and Linux, and Settings > Accelerators lists only what your machine can run.
- **Iris-3B Turbo images** came out as noise or blocks at small sizes: it now renders at 768 px or more and says so when it raises a smaller size. The gallery also records the steps and guidance the model really used.
- **Voice Models** cards no longer overflow when the DOWNLOADED badge is shown.
- **Discover went blank** when you typed in the search box.
- Apple Silicon: a model that reasons after a tool result, even with Thinking Off, no longer shows that reasoning in its answer.
- A reasoning level that a model's chat template does not accept is mapped to the nearest one it does.
- **Speculative decoding on Apple Silicon works again, and faster.** DFlash had been quietly falling back to plain decoding on every reply; it now runs (Qwen3-4B: about 113 tokens per second against 53) and verifies 8 tokens per step instead of 16, which was slower than plain decoding on prose. DDTree (the draft tree) is back, with a corrected tree verification, for models made of plain attention layers; hybrid models such as Qwen3.5 and 3.6 run linear DFlash (Qwen3.6-27B: about 4 accepted tokens per step; with the newer MLX libraries it had produced garbage on them until the app corrected which output head it reads, and every load now checks the draft runtime against the model before using it). On a 30B-class model (Qwen3-Coder-30B-A3B, 4-bit) code and reasoning replies run 1.4 to 1.7 times faster; creative writing is still slower with speculative decoding on, so leave it off for stories.
- **Clips with sound say so.** LTX clips show a Sound badge in the gallery and the result window, which plays them with sound; the Video Studio reminds you that the sound follows your prompt.
- **A JSON schema on a reasoning model** (llama.cpp) no longer returns nothing: if the grammar leaves the model with no text, the reply is asked for as plain JSON and the note says so. A conversation larger than the loaded context now answers with a 400 and the sizes, not a 500.
- **A failed mflux update keeps your working engine.** The old install is moved aside and put back if the new one does not verify.
- **Gated models:** when your Hugging Face token cannot open gated repos, the download error now says exactly which token setting to enable.
- **Wan** default guidance follows the Wan-AI model cards (5.0 for Wan 2.1, 4.0 for 2.2 A14B); the old values over-saturated Wan 2.2 5B clips. These two were not run here.
- The import-models panel labels each group with the folder it found them in instead of naming other apps.
