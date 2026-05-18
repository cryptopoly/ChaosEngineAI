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
