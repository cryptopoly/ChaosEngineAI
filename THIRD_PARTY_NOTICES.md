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
  RotorQuant and TurboQuant cache strategies. Actively maintained fork
  with support for recent model architectures (Gemma 4, etc.).

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

## Vendored Packages

### ChaosEngine (PCA-based KV cache compression)

- **Repository:** <https://github.com/cryptopoly/ChaosEngine>
- **Licence:** Apache 2.0
- **Submodule:** `vendor/ChaosEngine`
- **Usage:** Desktop builds may bundle this into the runtime via
  `npm run stage:runtime`.

---

## Optional Third-Party Cache Strategies

ChaosEngineAI supports optional cache/compression strategy backends.
If installed by the user, each is subject to its own licence:

| Strategy | Package | Repository | Licence |
|----------|---------|-----------|---------|
| TriAttention | `triattention` | <https://github.com/WeianMao/triattention> | See upstream |
| RotorQuant (marker) | `turboquant` | <https://github.com/back2matching/turboquant> | Apache 2.0 |
| TurboQuant MLX | `turboquant-mlx` | <https://github.com/sharpner/turboquant-mlx> | MIT |
| MegaKernel | — | <https://github.com/Luce-Org/luce-megakernel> | See upstream |

## Optional Speculative Decoding

| Package | Repository | Licence |
|---------|-----------|---------|
| `dflash-mlx` | <https://github.com/bstnxbt/dflash-mlx> | MIT |

These libraries are **not bundled** with ChaosEngineAI. They are
optional pip dependencies that the user may install independently.
