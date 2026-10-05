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

Installed on demand via the Voice Models tab (Setup install allowlist);
not bundled with the app.

### mlx-whisper (Apple Silicon STT)

- **Upstream:** <https://github.com/ml-explore/mlx-examples>
- **Licence:** MIT
- **Usage:** Whisper speech-to-text on MLX. Loaded by
  `backend_service/voice_runtime.py` for `/api/voice/transcribe`.

### faster-whisper (cross-platform STT)

- **Upstream:** <https://github.com/SYSTRAN/faster-whisper>
- **Licence:** MIT
- **Usage:** CTranslate2 Whisper backend for Windows / Linux hosts.
  Weights: `mobiuslabsgmbh/faster-whisper-large-v3-turbo` and
  `Systran/faster-whisper-{small,base}` (MIT) — the same repos
  faster-whisper resolves its own size tokens to.

### parakeet-mlx (Apple Silicon STT)

- **Upstream:** <https://github.com/senstella/parakeet-mlx>
- **Licence:** Apache 2.0
- **Usage:** NVIDIA Parakeet TDT speech-to-text on MLX — the fast default
  STT lane on Apple Silicon. Weights: `mlx-community/parakeet-tdt-0.6b-v3`
  (NVIDIA Parakeet TDT 0.6B v3, CC-BY-4.0 — attribution: NVIDIA).

### mlx-audio (Apple Silicon TTS)

- **Upstream:** <https://github.com/Blaizzy/mlx-audio>
- **Licence:** MIT
- **Usage:** Kokoro text-to-speech on MLX. Loaded by
  `backend_service/voice_runtime.py` for `/api/voice/synthesize`.

### kokoro-onnx (cross-platform TTS)

- **Upstream:** <https://github.com/thewh1teagle/kokoro-onnx>
- **Licence:** MIT
- **Usage:** ONNX Kokoro backend for Windows / Linux hosts.

### misaki (G2P text processing for Kokoro)

- **Upstream:** <https://github.com/hexgrad/misaki>
- **Licence:** Apache-2.0
- **Usage:** Grapheme-to-phoneme front-end mlx-audio's Kokoro pipeline
  requires but does not declare. Installed as `misaki[en]`.

### Japanese G2P chain (jf_* voices)

Installed individually instead of `misaki[ja]` — see the comment in
`backend_service/routes/setup/__init__.py` for the unidic-lite rationale.

- **fugashi** — <https://github.com/polm/fugashi> — MIT AND BSD-3-Clause
- **jaconv** — <https://github.com/ikegami-yukino/jaconv> — MIT
- **mojimoji** — <https://github.com/studio-ousia/mojimoji> — Apache-2.0
- **pyopenjtalk** — <https://github.com/r9y9/pyopenjtalk> — MIT (bundles
  OpenJTalk components under their modified-BSD licence)
- **unidic-lite** — <https://github.com/polm/unidic-lite> — MIT wrapper
  (bundled UniDic dictionary data under the BSD-style UniDic licence)
