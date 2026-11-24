## What's new in v0.10.0

### Highlights

- **Make a film from a song.** Storyboard turns a project into shots, takes and audio lanes. Attach a song and the app plans cuts on its phrases and captions the lyrics. Draft first on a fast model, keep the shots you like, then make the final pass. A Mixer with ducking, a rough-cut player and MP4 export at 16:9, 9:16 or 1:1 finish the job, and a bundled four-shot sample film shows how it works.
- **Let an agent drive the app.** An optional MCP endpoint (off until you switch it on in Settings > Integrations > Agent access) lets an agent such as Claude Code make images, video, speech, music, sounds, lip-sync shots and storyboards, with 28 tools, on the same queue as the Studios.
- **Keep going in Studio chats.** Ask for a film or a set of images in Chat and the app nudges the model until the job is done, frees GPU memory while a render runs, and shows a plan card with Draft, Standard and Final buttons and the time and memory each will cost.
- **Music and sound effects.** Songs with lyrics or instrumentals, 10 to 600 seconds, with ACE-Step 1.5 on Apple Silicon and, new, on Windows and Linux with an NVIDIA GPU (a 3-minute song in about 3 to 4 seconds once loaded on an RTX 4090). Stable Audio 3 adds sound effects and short music on NVIDIA.
- **Lip sync (preview).** Give Video Studio a vocal and a start image and LTX-2.3 makes a clip where the mouth follows the audio. Each shot can make several takes and keep the best by a sync score.
- **Windows and Linux with an NVIDIA GPU are first-class.** CUDA llama.cpp, LTX-2.3 and LTX-2.5 video, FLUX 3 Klein, Nunchaku FLUX.1, GPU speech-to-text, natural voices, music, TensorFold 1.x and vLLM in WSL2.
- **Linux ARM64 and DGX Spark** packages, with the GPU detected on shared-memory machines (from the v0.9.10 beta; see "Known limits").
- **Bigger models on a 24 GB card.** Mixture-of-experts models can keep their expert weights in system RAM ("Experts in system RAM" in the launch dialog), and llama.cpp can run its KV cache at q8_0 or q4_0.
- **Every screen in the nine translations is filled**, and a Klingon locale is registered as an experiment.

### Chat

- **Keep going** has its own control and chips in the thread; the nudges the app writes are marked. It stops when a waited job makes no progress.
- **Waiting on a job** blocks up to 30 minutes with a live progress line (elapsed and time left). Messages and tool calls show their times.
- **Skills**: import a folder or zip. They are scanned on import, off by default, and a High-rated skill needs an "I read the findings" confirm before you can switch it on. Settings > Skills lists them and the composer has a per-chat switch.
- **Prompts** now live in Chat. The Plugins page is gone.
- **Models**: recommended rows come first on a new machine, with Best fit, Fastest, Newest and Most popular chips. The launch dialog shows one memory bar (in use now, model, cache, limit) for every engine.
- New GGUF rows measured on an RTX 4090: Gemma 4 12B QAT (93 tokens per second, 115 to 120 with its MTP drafter), Gemma 4 26B-A4B QAT (160, about 175 with its drafter), Nemotron 3.5 Lightning Q4_0 (232), and Qwen3.6 35B-A3B MTP UD-IQ4_XS (170, no real gain from its heads). Qwen3.8-27B pairs with z-lab's DFlash 2 drafter (43 plain, 78 to 112 drafted on CUDA). Speculative decoding stays off by default. Larger quants of these families are listed with sizes and the fit check but were not run; the Qwen3.8 Flash Next Q2_0 row for a 24 GB card is untested.
- Tools: the tool prompt on Apple Silicon lists allowed values and ranges, more kinds of malformed tool calls are mended, and tool calls stream from the OpenAI-compatible endpoint.

### Studios

- **Calmer Image and Video Studios**: the default view holds what a first-time user needs; **Options** and **Expert** folds hold the rest. A run estimate strip above Generate shows time, memory and disk, tagged Measured or Estimated.
- **Standard size buttons**: aspect and size rows in Image Studio (0.5 to 8 MP) and named resolutions in Video Studio; sizes past a model's limit are greyed out with the reason.
- **Reference images** in Image Studio (FLUX.1, FLUX.2 Klein, Qwen Image 2.1, FLUX 3 Klein) and a **start image and optional end image** in Video Studio (LTX-2.3 and LTX-2.5).
- **NVIDIA video**: LTX-2.3 (text, start picture, start plus end picture, with sound; about 70 seconds for a 97-frame clip on an RTX 4090) and LTX-2.5 4-bit (gated; 62 seconds for 768 by 512 by 97 frames on a 24 GB card), and Kandinsky 6.0 Lite distilled video with sound (110 seconds for 5 seconds at 864 by 480).
- **NVIDIA images**: FLUX 3 Image Klein (7 seconds for 1024 px at 8 steps), Qwen-Image-2.1 Turbo as a Q4 row (7.5 seconds), MOSS-Image SFT (53 seconds, 22 GB), and **Nunchaku (SVDQuant INT4) FLUX.1** (schnell at 1024 px in about 6 seconds warm). Nunchaku is not on PyPI under that name, so you install its wheel yourself; its rows are CUDA-only. FLUX runs fully on the card when it fits and offloads to system memory only when it cannot.
- Removed: **Enhance** (the button, the auto-enhance switch and the prompt enhancer). Video prompts are sent as typed.

### Voice, music and sounds

- **Music** (Apple Silicon and NVIDIA): a Music Studio with Options and Expert, a Library with rename, tags, filter and "Make another like this", and a prompt-free AI tag in the FLAC file with a switch for private use. The NVIDIA music pack is 6.6 GB in its own folder; turbo needs about 8 GB of video memory and XL 12 to 14 GB.
- **Stable Audio 3** (small SFX, small music, medium) beside Stable Audio Open on NVIDIA. The models are gated on Hugging Face: accept Stability's licence on each model page and add your token in Settings.
- **Speech-to-text on the NVIDIA GPU**: an add-on (Voice > Models > Add-ons) with Qwen3-ASR, Parakeet TDT 0.6B v3, Nemotron 3.5 ASR and Granite Speech 5.0. On a 9-second clip the warm time is 0.15 to 0.2 seconds for Parakeet and 1 to 2.6 seconds for the others. Word timings and speaker labels included.
- **Natural voices on the GPU**: Chatterbox (Turbo and Multilingual V3), VoxCPM2 and Orpheus, with voice design and cloning for the first three, and Qwen3-TTS. Cloned speech is watermarked as before.
- **Voice Studio** has Models and Library, a voice manager (filters, samples, Use in Speak or Chat), delete for speech and dictation models, add-ons and the voice runtime. Any listed voice can read aloud in Chat or run a call, with a Kokoro fallback and a note. Every Speak result is saved to the Library.

### Lip sync (preview)

- On an NVIDIA card with about 22 GB free, a 5-second 768 by 512 shot takes about 70 seconds warm (157 the first time).
- Each shot can make several takes (3 for a sung vocal, 1 for speech, 4 for best quality) and keep the best by a sync score. The score comes from an optional add-on (about 90 MB, "Install sync scoring" in the lip-sync panel); without it, take 1 is used and the panel says so.
- In Storyboard, "song to lip-sync shots" slices the vocal at each shot. Singing works only partly: on one sung 4-second window none of three takes reached the usable bar. Sync scores are better than chance in our measurement, but a person has not judged them.

### Storyboard and agent bridge

- Plan tab (brief, proposed shots with a suggested count, style references, brand kit), a freestyle board tab, take compare, durable batches that pause and resume across a restart, burnt-in captions with a look, a corner logo and title and end cards.
- An AI label (a metadata tag, forced when a clip is marked as a cloned voice) and `.chaosfilm` bundles for sharing and backup. "Use in storyboard" appears in the Image and Video Galleries, the Voice Library and Music.
- The agent bridge serves an `AGENTS.md` guide. Files can be sent in and fetched out for an agent on another machine. A Studio's Cancel stops only its own job.

### Engines and accelerators

- **llama.cpp b12651.** Settings > Accelerators installs the CUDA llama-server on Windows and Linux (archives checked by sha256); Qwen3.6-35B-A3B Q4_K_M answered at 41.6 tokens per second on an RTX 4090.
- **TensorFold 1.x** (1.3.4) as the CUDA text engine, with its own speculative drafter. On Linux Setup downloads the release archive; Windows has no archive, so you supply a source build or your own archive. Through the app on an RTX 4090 with Qwen3.8-27B: 49.6 tokens per second plain, 132 to 206 drafted (a 1.2.x build). The launch dialog now sizes it against the real GPU limit.
- **vLLM in WSL2**: "Install vLLM in WSL" in Settings > Accelerators installs it in your Ubuntu distro. A 0.6B model answers at about 80 tokens per second through the bridge. Only one small model on a 24 GB card was tried.
- **stable-diffusion.cpp** builds with CUDA on Windows. dflash-mlx moves to v0.1.10 on Apple Silicon.

### Linux ARM64, DGX Spark and Linux NVIDIA

- **Linux ARM64 packages** (`_aarch64.AppImage`, `_arm64.deb`) with the Python runtime and a CPU llama.cpp, so GGUF chat works on the CPU out of the box.
- **CUDA llama-server on Linux** (ARM64 and x86_64): an Install button in Settings > Accelerators when `nvidia-smi` is present. The app uses it only when it sees a CUDA device; otherwise the CPU build stays.
- **Shared-memory GPUs** (DGX Spark GB10, GH200): `nvidia-smi` reports their memory as "Not Supported", so the app reads system memory, detects the GPU and sizes estimates against the shared pool. llama-server loads with `--no-mmap` there (`CHAOSENGINE_LLAMA_MMAP=1` keeps mmap).
- The GPU runtime install picks the PyTorch build that matches the CPU and driver (CUDA 13.0 first on ARM64 and on drivers 580 and newer), which also helps RTX 50-series cards.
- Testers can run `scripts/linux-gpu-check.sh`; it writes a report on your machine and uploads nothing.

### Windows

- The NVIDIA features above run on Windows 11. The app now ends `llama-server` and the other workers if the backend is killed from Task Manager, and no longer hangs for 300 seconds on the first music or lip-sync load.
- Unloading an image or video model gives its memory back, Wan 2.2 TI2V 5B no longer exhausts system commit on load, and a GPU runtime install no longer leaves the packaged backend unable to start.
- An installer built with the CUDA `sd.exe` carries NVIDIA's CUDA runtime DLLs under NVIDIA's licence (see `THIRD_PARTY_NOTICES.md`).

### Settings, Server and languages

- Server > Activity shows a per-model timeline and event list (Logs folds into it); `chaosengine-cli server-events` streams them.
- 110 % and 125 % UI scale, a stronger-borders accessibility switch, and a clean-up panel in Settings > Disk. Below 1180 px the sidebar becomes a top bar with a menu drawer.
- The sidebar keeps only the active group open (a Settings switch keeps every group open).
- Nine translations were filled on 2026-11-23 (the 13 newest strings show in English until the last pass); a Klingon (`tlh`) locale is registered as experimental. The language you choose and the clock format are now saved.
- API keys are masked on screen with a Reveal control, and Hugging Face library usage reporting is off by default (set `HF_HUB_DISABLE_TELEMETRY=0` to opt back in).

### Security

- The WSL bridge no longer allows command injection; agent audio for lip sync is held to uploads and outputs; ffmpeg is restricted to a protocol whitelist and refuses archive paths with backslashes or drive letters.
- Images and videos made from an input image no longer carry the full local path (and the account name) in their embedded settings.

### Fixes worth knowing

- A chat whose model cannot load now says why instead of "Internal Server Error".
- The model library follows a moved Hugging Face cache, and a repo downloaded only for its drafter is no longer listed as a model.
- A local model no longer takes another catalogue row's repo (a small Qwen had been refused as "runs only on TensorFold").
- A Storyboard shot lasts the seconds you asked for, and lyric lines no longer overlap.
- Benchmarks and Conversion no longer invent figures: the constant Quality 100 % is gone and task accuracy scores the answer the model rates highest.
- Automatic and agent picks skip stable-diffusion.cpp rows when no `sd` binary is installed.
- Paused Hugging Face downloads resume from the partial file.

### Upgrading

- **Plugins are removed** and **Fine-Tuning is hidden** (turn on Settings > General > Appearance > Show experimental tabs to see it). **Enhance** is removed.
- **Logs** is now Server > Activity; old links redirect.
- Cache strategy ids `chaosengine` and `rotorquant` load as `turboquant`.
- Nunchaku and some other NVIDIA-only rows are hidden on Apple Silicon.

### Known limits

- The NVIDIA features were run on one Windows 11 PC with one RTX 4090. Native Linux x86_64, RTX 50-series, cards below 24 GB, and AMD or Intel GPUs have not been tried, and the new features were run from the development tree rather than the installed app.
- **Linux ARM64 is experimental.** It was tested on an Ubuntu 26.04 ARM64 virtual machine without a GPU (deb and AppImage install and start; a GGUF chat ran on the CPU at about 29 tokens per second). Nothing was run on a real NVIDIA GPU on ARM64, including a DGX Spark: GPU detection, the CUDA llama-server, `--no-mmap` and the PyTorch CUDA 13 install are built from documentation. Image and video on the GB10 and vLLM on ARM64 are untested, and voice transcription runs on the CPU there.
- The Apple Silicon checks (MLX, mflux, ltx-2-mlx, Music on MLX, the packaged app, the microphone) were not re-run on the combined build. Lip sync has never run on the Mac, and Stable Audio 3 does not run on the Mac yet.
- Voices and music were checked by measurement and read-back, not by listening. Lip sync is a preview and sung shots often have no usable take.
- Several downloads are gated by Hugging Face licences (Stable Audio 3, the FLUX.1 schnell base, FLUX 3 Image Klein, LTX); FLUX 3 Image Klein and the FLUX.1 [dev] Nunchaku pack carry non-commercial or revenue-limited licences.
- TensorFold on Windows needs a binary you build or host yourself, and the 1.3.4 family rows were not compared with the binary.
- MiniMax H3 video stays off in this release: its licence does not apply in several regions and a check on its machine-generated tag is open.
- The machine-made translations have not been read by native speakers; corrections are welcome.
