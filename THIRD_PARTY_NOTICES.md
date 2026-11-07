# Third-Party Notices

ChaosEngineAI incorporates or depends on the following third-party
projects. Each is subject to its own licence as noted below.

---

## Bundled / Built-from-Source Binaries

These may be compiled from source and shipped alongside ChaosEngineAI.

### llama.cpp (upstream)

- **Repository:** <https://github.com/ggml-org/llama.cpp>
- **Licence:** MIT
- **Copyright:** Copyright (c) 2023-2026 The ggml authors
- **Binary:** `llama-server`, `llama-cli`
- **Usage:** Standard GGUF inference backend.

### llama-cpp-turboquant (TurboQuant fork of llama.cpp)

- **Repository:** <https://github.com/TheTom/llama-cpp-turboquant>
- **Licence:** MIT (inherited from upstream llama.cpp)
- **Copyright:** Copyright (c) 2023-2026 The ggml authors
- **Binary:** `llama-server-turbo`, `llama-cli-turbo`
- **Usage:** Adds turbo2/3/4 KV cache quantisation types used by the
  TurboQuant cache strategy. Actively maintained fork with support for
  recent model architectures (Gemma 4, etc.).

> **MIT licence notice (applies to both llama.cpp and the TurboQuant fork):**
>
> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

### stable-diffusion.cpp

- **Repository:** <https://github.com/leejet/stable-diffusion.cpp>
- **Licence:** MIT
- **Copyright:** Copyright (c) 2023 leejet
- **Binary:** `sd` (built from `sd-cli` by `scripts/build-sdcpp.sh`; staged into the
  desktop bundle when it is present at build time)
- **Usage:** Cross-platform image and video generation lane (FLUX.1, Qwen-Image-2.1
  and the Wan GGUF rows). The MIT permission notice above applies to it, with the
  copyright holder named here.

---

## Optional Third-Party Cache Strategies

ChaosEngineAI supports optional cache/compression strategy backends.
If installed by the user, each is subject to its own licence:

| Strategy | Package | Repository | Licence |
|----------|---------|-----------|---------|
| TriAttention | `triattention` | <https://github.com/WeianMao/triattention> | See upstream |
| TurboQuant MLX | `turboquant-mlx-full` | <https://github.com/arozanov/turboquant-mlx> | MIT |
| MegaKernel | — | <https://github.com/Luce-Org/luce-megakernel> | See upstream |
| TeaCache (diffusion) | vendored patches | <https://github.com/ali-vilab/TeaCache> | Apache 2.0 |

### TeaCache (vendored forward patches)

TeaCache is distributed upstream as a collection of per-model Python files
(one ``teacache_forward`` per DiT class) rather than a pip package. When
ChaosEngineAI ships support for a specific diffusion pipeline it vendors
the corresponding ``teacache_forward`` into
``cache_compression/_teacache_patches/`` under the upstream Apache 2.0
licence, preserving the original copyright header in the vendored file.

- **Repository:** <https://github.com/ali-vilab/TeaCache>
- **Licence:** Apache 2.0
- **Usage:** Training-free timestep caching for FLUX, Wan2.1, Wan2.2,
  HunyuanVideo, Mochi, CogVideoX, LTX-Video, and other DiT-based diffusion
  pipelines. 1.5–2.0× image speedup and 1.6–2.1× video speedup with
  negligible visual quality loss at ``rel_l1_thresh=0.4``.

## Optional Speculative Decoding

| Package | Repository | Licence |
|---------|-----------|---------|
| `dflash-mlx` | <https://github.com/bstnxbt/dflash-mlx> | MIT |
| `mtplx` | <https://github.com/youssofal/mtplx> | Apache 2.0 |
| `tensorfold` | <https://github.com/ashhart/TensorFold> | MIT |

These libraries are **not bundled** with ChaosEngineAI. They are
optional pip dependencies that the user may install independently.

### mtplx (MTP speculative decoding on Apple Silicon)

`mtplx` provides native in-model Multi-Token Prediction speculative
decoding for models that ship baked-in MTP heads (Qwen3.5/3.6,
DeepSeek V3/R1, Qwen3-Coder-Next, Youssofal's MTPLX-Optimized
variants). It bundles a forked ``mlx-mtplx`` runtime that conflicts
with upstream ``mlx``, so the install lives in an **isolated venv**
at ``~/.chaosengine/mtplx-venv/`` — never co-installed with our main
``.venv``. ChaosEngineAI shells out to the ``mtplx start --model X
--port N`` CLI from ``backend_service/inference/mtplx_engine.py`` and
proxies via the package's own OpenAI-compatible HTTP server. Not
bundled in the desktop ``.app``; installed on demand from the Setup
page on Apple Silicon hosts. See FU-028 in CLAUDE.md.

> **Apache 2.0 licence summary**: free use, modification, and
> redistribution permitted with attribution preserved. Source:
> ``~/.chaosengine/mtplx-venv/lib/python*/site-packages/mtplx-*.dist-info/licenses/``
> (full LICENSE file shipped with the wheel).

### tensorfold (exact speculative decoding on Apple Silicon)

`tensorfold` is a local OpenAI-compatible inference server with per-family
kernels whose speculative drafts are verified exactly against the target
model, so drafted replies are identical to serial decoding. It pins its own
``mlx`` / ``mlx-lm`` ranges, so the install lives in an **isolated venv** at
``~/.chaosengine/tensorfold-venv/`` — never co-installed with our main
``.venv``. ChaosEngineAI shells out to ``tensorfold serve <model dir>`` from
``backend_service/inference/tensorfold_engine.py`` and proxies via the
package's own HTTP server. Not bundled in the desktop ``.app``; installed on
demand from the launch settings or the Setup page on Apple Silicon hosts.

> **MIT licence summary**: free use, modification, and redistribution
> permitted provided the copyright and permission notice are preserved.
> Source: ``~/.chaosengine/tensorfold-venv/lib/python*/site-packages/tensorfold-*.dist-info/``
> (the LICENSE file shipped with the package).

Two optional TensorFold extras can be added on demand from the launch settings
or ``chaosengine-cli tensorfold-install --extras``. They are installed into the
same isolated venv and are never bundled: ``vision`` (image input) brings
Pillow (HPND, a permissive MIT-style licence), ``transformers`` (Apache 2.0) and
``mlx-vlm`` (MIT); ``grammar`` (structured output) brings ``xgrammar``
(Apache 2.0) and, through it, PyTorch (BSD-3-Clause).

## Optional Apple Silicon Image Runtime

| Package | Repository | Licence |
|---------|-----------|---------|
| `mflux` | <https://github.com/filipstrand/mflux> | MIT |

`mflux` runs image models (Qwen-Image-2.1, FLUX.1, FLUX.2, Z-Image, ...) natively
on MLX. It pins ``mlx >=0.32`` and ``torch >=2.13``, newer than the stack the
app runs, so it installs into an **isolated venv** at
``~/.chaosengine/mflux-venv/`` and is never co-installed with the main
``.venv``. ChaosEngineAI shells out to its ``mflux-generate-*`` commands from
``backend_service/image_runtime/mflux_engine.py``, one process per image. Not
bundled in the desktop ``.app``; installed on demand from the Image Studio
(``scripts/install-mflux.sh``) or with ``chaosengine-cli mflux-install``.

> **MIT licence summary**: free use, modification, and redistribution
> permitted provided the copyright and permission notice are preserved.
> Source: ``~/.chaosengine/mflux-venv/lib/python*/site-packages/mflux-*.dist-info/``
> (the LICENSE file shipped with the package).

The weights it runs keep their own licences. In particular Qwen-Image-2.1 is
under the Qwen Research Licence (non-commercial use only).

## Optional Apple Silicon Video Runtime

| Package | Repository | Licence |
|---------|-----------|---------|
| `mlx-video` | <https://github.com/Blaizzy/mlx-video> | MIT |

`mlx-video` is an MLX-native video generation runtime for Apple Silicon
covering Wan2.1 / Wan2.2 / LTX-2 T2V, I2V, and A2V. ChaosEngineAI drives
it as a subprocess from ``backend_service/mlx_video_runtime.py`` — not
bundled, installable on demand from the Setup page on Apple Silicon
hosts. See FU-009 in CLAUDE.md.

| Package | Repository | Licence |
|---------|-----------|---------|
| `ltx-2-mlx` (`ltx-core-mlx`, `ltx-pipelines-mlx`) | <https://github.com/dgrauet/ltx-2-mlx> | MIT |

`ltx-2-mlx` is a pure MLX port of Lightricks LTX-2.3 / 2.5 (text to video with
stereo sound, no PyTorch). It pins its own ``mlx`` / ``mlx-lm`` floors, so it
installs into an **isolated venv** at ``~/.chaosengine/ltx2-venv/`` from a pinned
commit (it is not on PyPI) and is never co-installed with the main ``.venv``.
ChaosEngineAI shells out to its ``ltx-2-mlx generate --distilled`` command from
``backend_service/ltx2_mlx_runtime.py``, one process per clip. Not bundled in
the desktop ``.app``; installed on demand from the Video Studio
(``scripts/install-ltx2-mlx.sh``) or with ``chaosengine-cli ltx2-install``.

The weights it runs keep their own licences: ``dgrauet/ltx-2.3-mlx-q4`` is
derived from Lightricks LTX-2.3 under the LTX-2 community licence (free below
US$10 million annual revenue), and ``mlx-community/gemma-3-12b-it-4bit`` (the
text encoder) is under the Gemma terms of use. Neither is bundled.

---

## Ported Algorithms

### DDTree (Diffusion Draft Tree)

- **Upstream:** <https://github.com/liranringel/ddtree>
- **Licence:** MIT
- **Port location:** `backend_service/ddtree.py`
- **Usage:** The tree-building and tree-mask compilation logic is ported
  to ChaosEngineAI's MLX runtime. The draft model bundle is reused from
  DFlash. No upstream code is bundled verbatim; this is a re-implementation
  of the published algorithm.

### Kolibri 1 architecture for mlx-lm (vendored)

- **Upstream:** the community port proposed in
  <https://github.com/ml-explore/mlx-lm/pull/1945>, itself ported from Aleph Alpha's
  vLLM plugin <https://github.com/Aleph-Alpha/aleph-alpha-inference>
- **Licence:** Apache 2.0 (SPDX header kept in the file)
- **Location:** `backend_service/mlx_models/kolibri1.py`, copied unmodified
- **Usage:** mlx-lm ships no `kolibri1` architecture yet, so `mlx_lm.load` reads this
  file through its `model_file` hook for Kolibri 1 checkpoints. It is no longer used
  once an mlx-lm release carries the architecture. The weights keep their own
  licence (Apache 2.0, Aleph Alpha).

### LFM3 architecture for mlx-lm (vendored)

- **Upstream:** Liquid AI's reference `modeling_lfm3.py` in
  <https://huggingface.co/LiquidAI/LFM3-1.7B> and the mlx-vlm port of it
  (<https://github.com/Blaizzy/mlx-vlm>, `mlx_vlm/models/lfm3`, commit `701456f1`)
- **Licence:** Apache 2.0 (SPDX header kept in the file)
- **Location:** `backend_service/mlx_models/lfm3.py`, rewritten against mlx-lm's
  model interface (reuses mlx-lm's LFM2 `ShortConv`)
- **Usage:** mlx-lm ships no `lfm3` architecture yet, so `mlx_lm.load` reads this
  file through its `model_file` hook for LFM3 checkpoints. It is no longer used
  once an mlx-lm release carries the architecture. The weights keep their own
  licence (LFM Open License v1.0, Liquid AI).

## Internationalization (FU-042)

### i18next (frontend i18n framework)

- **Upstream:** <https://github.com/i18next/i18next>
- **Licence:** MIT
- **Usage:** Core i18n runtime loaded at frontend boot
  (`src/i18n/index.ts`). Powers namespace bundles + locale switching
  for all React surfaces.

### react-i18next

- **Upstream:** <https://github.com/i18next/react-i18next>
- **Licence:** MIT
- **Usage:** React bindings (`useTranslation` hook, `<Trans>` component)
  consumed throughout `src/components/` and `src/features/`.

### i18next-icu

- **Upstream:** <https://github.com/i18next/i18next-icu>
- **Licence:** MIT
- **Usage:** ICU MessageFormat backend for plural / select rules.
  Required for correct Slavic 4-form plurals (`ru`), Polish 4-form
  plurals (`pl`), and the zero/two/few categories needed for Arabic
  (FU-046 RTL phase).

### i18next-browser-languagedetector

- **Upstream:** <https://github.com/i18next/i18next-browser-languageDetector>
- **Licence:** MIT
- **Usage:** Reads `navigator.language` + `navigator.languages` for the
  initial locale negotiation when no OS / Tauri locale is available
  (browser dev mode). In packaged builds the Tauri `tauri-plugin-os`
  locale takes priority and this is a fallback only.

### intl-messageformat

- **Upstream:** <https://github.com/formatjs/formatjs>
- **Licence:** BSD-3-Clause
- **Usage:** Transitive dep of `i18next-icu` for the ICU parser. Listed
  explicitly so its licence is acknowledged.

### Babel (Python message catalogs)

- **Upstream:** <https://github.com/python-babel/babel>
- **Licence:** BSD-3-Clause
- **Usage:** Backend gettext-style translation catalogs under
  `backend_service/locales/{lang}/LC_MESSAGES/messages.{po,mo}`. The
  `pybabel extract / update / compile` toolchain drives the workflow.
  Lazy-imported per CLAUDE.md performance guidelines so it doesn't
  cost startup on workers (mlx_worker, vllm, ddtree) that never read
  translations.

### rust-i18n

- **Upstream:** <https://github.com/longbridgeapp/rust-i18n>
- **Licence:** MIT
- **Usage:** Compile-time message catalog macro for the Tauri shell
  (`src-tauri/locales/*.yml`). Powers localized native menu / tray /
  updater dialog strings via `t!("menu.file")`.

### fluent-bundle

- **Upstream:** <https://github.com/projectfluent/fluent-rs>
- **Licence:** Apache-2.0
- **Usage:** Runtime ICU-equivalent for plural / select / select-ordinal
  in Rust. Complements `rust-i18n` for dynamic strings (e.g. updater
  progress with plural categories) that need runtime composition.

### unic-langid

- **Upstream:** <https://github.com/zbraniecki/unic-locale>
- **Licence:** MIT / Apache-2.0 dual
- **Usage:** BCP-47 language tag parsing for `fluent-bundle`. Listed
  explicitly as a transitive dep with permissive licensing.

---

## Voice I/O (STT + TTS backends)

Installed on demand by "Install voice pack" (Voice > Models) into a separate
environment, `~/.chaosengine/voice-venv`; not bundled with the app. Run by
`backend_service/voice_worker.py`.

### mlx-whisper (Apple Silicon STT)

- **Upstream:** <https://github.com/ml-explore/mlx-examples>
- **Licence:** MIT
- **Usage:** Whisper speech-to-text on MLX.

### parakeet-mlx (Apple Silicon STT)

- **Upstream:** <https://github.com/senstella/parakeet-mlx>
- **Licence:** Apache 2.0
- **Usage:** NVIDIA Parakeet TDT speech-to-text on MLX, the default on Apple
  Silicon. Weights: `mlx-community/parakeet-tdt-0.6b-v3` (NVIDIA Parakeet TDT
  0.6B v3, CC-BY-4.0, attribution: NVIDIA).

### faster-whisper (Windows / Linux STT)

- **Upstream:** <https://github.com/SYSTRAN/faster-whisper>
- **Licence:** MIT
- **Usage:** CTranslate2 Whisper. Weights:
  `mobiuslabsgmbh/faster-whisper-large-v3-turbo` and
  `Systran/faster-whisper-{small,base}` (MIT), the repos faster-whisper itself
  maps its size names to. On a machine with an NVIDIA GPU the install also adds
  NVIDIA's `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` wheels from PyPI
  (NVIDIA proprietary licence; downloaded by the user's machine, never
  redistributed by us).

### mlx-audio (Apple Silicon TTS)

- **Upstream:** <https://github.com/Blaizzy/mlx-audio>
- **Licence:** MIT
- **Usage:** Kokoro text-to-speech on MLX. Weights: `prince-canuma/Kokoro-82M`
  (Kokoro-82M by hexgrad, Apache-2.0).

### kokoro-onnx (Windows / Linux TTS)

- **Upstream:** <https://github.com/thewh1teagle/kokoro-onnx>
- **Licence:** MIT
- **Usage:** ONNX Kokoro; model and voice files from the project's GitHub
  release (Kokoro-82M, Apache-2.0).

### misaki and spaCy (text front-end for Kokoro on MLX)

- **misaki** — <https://github.com/hexgrad/misaki> — Apache-2.0. Installed as
  `misaki[en]`; `misaki[zh]` with the optional Mandarin Chinese voices.
- **spaCy** — <https://github.com/explosion/spaCy> — MIT, with its
  `en_core_web_sm` 3.8.0 model (MIT), installed with the voice pack so misaki
  never downloads it at run time.
- **espeak-ng** (through `espeakng-loader`) — GPL-3.0-or-later; a separate
  library used for languages misaki has no rules for. Installed on the user's
  machine by the voice pack, not shipped by us.

### Silero VAD and ONNX Runtime (voice-call turn detection)

- **Silero VAD** — <https://github.com/snakers4/silero-vad> — MIT. The ONNX
  model file is downloaded from `onnx-community/silero-vad` (MIT) at a pinned
  revision when the voice pack is installed.
- **ONNX Runtime** — <https://github.com/microsoft/onnxruntime> — MIT. Runs the
  VAD model in the voice environment.

### Speaker labels (diarization)

- **sherpa-onnx** — <https://github.com/k2-fsa/sherpa-onnx> — Apache-2.0.
  Installed in the voice environment; runs the two models below on the CPU.
- **pyannote segmentation 3.0** — <https://huggingface.co/pyannote/segmentation-3.0>
  — MIT. ONNX export from `csukuangfj/sherpa-onnx-pyannote-segmentation-3-0`,
  downloaded at a pinned revision when the voice pack is installed.
- **3D-Speaker CAM++ (VoxCeleb, English)** — <https://github.com/modelscope/3D-Speaker>
  — Apache-2.0. ONNX file from `csukuangfj/speaker-embedding-models`,
  downloaded at a pinned revision.

### FFmpeg video tools (joining, mixing, tagging)

FFmpeg is **not bundled** with ChaosEngineAI and we host no FFmpeg build (decision D-49). The app finds it in this
order: a file the user chose, a download the app made, the `imageio-ffmpeg` package, and (for tagging only) a program
on the system `PATH`. The app runs it as a separate program and never links it into our code. Its Setup row and the
"Video tools are needed" prompt install it on request.

- **imageio-ffmpeg** (all systems, the one-click install). Upstream: <https://github.com/imageio/imageio-ffmpeg>
  (the Python wrapper, BSD-2-Clause) and <https://ffmpeg.org> (the program). The user's machine installs it from PyPI
  when they choose Install video tools; we distribute nothing. The macOS build we checked (ffmpeg 7.1,
  `ffmpeg-macos-aarch64-v7.1`, imageio-ffmpeg 0.6.0) reports `--enable-gpl` with `libx264` and `libx265`, so the
  binary as a whole is licensed under the **GNU General Public License, version 2 or later**; its configuration line
  is printed by `ffmpeg -version`. Source: <https://ffmpeg.org/releases/>, x264 at
  <https://code.videolan.org/videolan/x264>, x265 at <https://bitbucket.org/multicoreware/x265_git>. The ltx-2-mlx
  environment keeps its own libx264 ffmpeg and is not affected by this entry.
- **BtbN LGPL builds** (Windows and Linux, optional, on request in Setup). Upstream:
  <https://github.com/BtbN/FFmpeg-Builds>, release `autobuild-2026-10-31-12-59` (ffmpeg n9.0.2, `lgpl` variant, built
  without `--enable-gpl` and `--enable-nonfree`; licence **LGPL-2.1-or-later**; source and build scripts at the
  release page and in that repository). The app downloads the pinned file from that release, checks its SHA-256
  against a value stored in the app, and runs it only after the check. The `lgpl` variant includes libass (ISC),
  FreeType (FTL: "Portions of this software are copyright The FreeType Project"), HarfBuzz (MIT), FriBidi (LGPL-2.1),
  libvpx (BSD-3-Clause) and openh264 (BSD-2-Clause, compiled from source by the builder; Cisco's patent licence
  covers only Cisco's own binary); the full list is in the build's `ffmpeg -buildconf`. BtbN, not us, distributes
  these files. This software uses code of FFmpeg licensed under the LGPLv2.1.
- **H.264 encoding:** libx264 when the ffmpeg in use has it, otherwise the system's encoder (VideoToolbox on macOS;
  NVENC, AMF or Quick Sync where the machine has them; openh264 as a last resort).

### PyAV (audio decoding)

- **Upstream:** <https://github.com/PyAV-Org/PyAV>
- **Licence:** BSD-3-Clause for PyAV itself. Its PyPI wheels bundle FFmpeg libraries built with `--enable-gpl`
  (libx264 and libx265; checked on the macOS arm64 wheel of av 14.1.0, where libavutil reports "GPL version 3 or later"),
  so treat the bundled libraries as **GPL-3.0-or-later**, not LGPL. They are installed from PyPI on the user's machine
  (voice venv), not shipped by us. Sources: <https://github.com/PyAV-Org/pyav-ffmpeg>.
- **Usage:** Decodes recordings and dropped audio / video files, so no
  `ffmpeg` command is needed.

### Japanese G2P chain (jf_* voices, Setup page only)

Installed individually from the Setup page, instead of `misaki[ja]` — see the comment in
`backend_service/routes/setup/__init__.py` for the unidic-lite rationale.

- **fugashi** — <https://github.com/polm/fugashi> — MIT AND BSD-3-Clause
- **jaconv** — <https://github.com/ikegami-yukino/jaconv> — MIT
- **mojimoji** — <https://github.com/studio-ousia/mojimoji> — Apache-2.0
- **pyopenjtalk** — <https://github.com/r9y9/pyopenjtalk> — MIT (bundles
  OpenJTalk components under their modified-BSD licence)
- **unidic-lite** — <https://github.com/polm/unidic-lite> — MIT wrapper
  (bundled UniDic dictionary data under the BSD-style UniDic licence)

## Music (ACE-Step 1.5)

Installed on demand by "Install music pack" (Music > Models) into a separate environment,
`~/.chaosengine/music-venv`; not bundled with the app. Run by `backend_service/music_worker.py`.
Weights are downloaded by the user into the Hugging Face cache at pinned revisions; none ship with the app.

### ACE-Step 1.5 (code)

- **Upstream:** <https://github.com/ace-step/ACE-Step-1.5>, pinned to commit
  `ca1e85fe9430179831e6bc6be790c332190a3866` (package `ace-step` 1.5.0).
- **Licence:** MIT ("Copyright (c) 2026 ACEStep").
- **Usage:** text-to-music library driven through its Python API by our worker. The library copies its own
  model code into the checkpoint folder before loading, so the code that runs is from the pinned commit.

### ACE-Step 1.5 weights

- `ACE-Step/Ace-Step1.5` (revision `19671f406d603126926c1b7e2adc169acbcade22`: 2B turbo DiT, VAE,
  Qwen3-Embedding-0.6B text encoder, 1.7B planner LM) and `ACE-Step/acestep-v15-xl-turbo`
  (revision `d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee`).
- **Licence:** MIT on every model card (`license: mit`), not gated. The authors state the model was trained on
  legally compliant data and that generated music may be used commercially; the README asks users to check that
  a track is original, to disclose AI involvement and to get permission when adapting protected material.

### Packages the music pack installs (not bundled)

`backend_service/music_requirements.txt` (exported from the pinned `uv.lock`) installs, among others: PyTorch and
torchaudio (BSD-3-Clause), torchcodec (BSD-3-Clause), MLX and mlx-lm (MIT), transformers, diffusers, accelerate and
huggingface-hub (Apache-2.0), numpy (BSD-3-Clause), soundfile (BSD-3-Clause; bundles libsndfile, LGPL-2.1),
gradio, lightning, peft, torchao and modelscope (Apache-2.0 or BSD; pulled in by the library, not used for
inference). All are MIT, BSD or Apache except the one below.

- **pytorch-wavelets 1.3.0: not installed (dropped 2026-11-07, D-37, FU-138).** PyPI says "Free To Use But Restricted".
  Its LICENSE file is MIT (copyright 2020 Fergal Cotter) for the parts that are new, but the dual-tree (DTCWT) parts derive
  from Nick Kingsbury's MATLAB toolbox ("for research purposes only"), and importing the package loads that code. ACE-Step
  uses it only for the optional DCW correction, which the MLX path computes natively (Haar), so we removed it from
  `music_requirements.txt` and verified a generation without it (`docs/research/licences-wavelets-ffmpeg.md`).
  PyWavelets (MIT AND BSD-3-Clause) stays.
