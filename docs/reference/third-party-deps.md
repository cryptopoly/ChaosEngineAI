# Third-party dependencies

ChaosEngineAI incorporates or depends on the following third-party projects.
Each is subject to its own licence. The canonical source for this list is
[`THIRD_PARTY_NOTICES.md`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/THIRD_PARTY_NOTICES.md)
in the repo root; this page is a docs-side mirror.

## Bundled / built-from-source binaries

These may be compiled from source and shipped alongside ChaosEngineAI.

### llama.cpp (upstream)

- **Repository:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **Licence:** MIT
- **Binary:** `llama-server`, `llama-cli`
- **Usage:** Standard GGUF inference backend.

### llama-cpp-turboquant (TurboQuant fork)

- **Repository:** [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
- **Licence:** MIT (inherited from upstream llama.cpp)
- **Binary:** `llama-server-turbo`, `llama-cli-turbo`
- **Usage:** Adds turbo2 / 3 / 4 KV cache quantisation types used by the
  TurboQuant cache strategy. Actively maintained fork with support for
  recent model architectures (Gemma 4, etc.).

### stable-diffusion.cpp

- **Repository:** [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- **Licence:** MIT
- **Binary:** `sd` (built by `scripts/build-sdcpp.sh`)
- **Usage:** Cross-platform diffusion inference for image and video.
  Image lane covers FLUX, SD3.5, SDXL, SD2.1, Qwen-Image, Z-Image. Video
  lane covers Wan-family GGUF quants.

## Optional third-party cache strategies

| Strategy | Package | Repository | Licence |
|---|---|---|---|
| TriAttention | `triattention` | [WeianMao/triattention](https://github.com/WeianMao/triattention) | See upstream |
| TurboQuant MLX | `turboquant-mlx-full` | [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) | MIT |
| TeaCache (diffusion) | vendored patches | [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) | Apache 2.0 |

TeaCache is vendored as per-model `teacache_forward` Python files under
`cache_compression/_teacache_patches/` (the upstream isn't published as a pip
package). The Apache 2.0 copyright headers are preserved in each vendored
file.

## Optional speculative decoding

| Package | Repository | Licence |
|---|---|---|
| `dflash-mlx` | [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) | MIT |
| `dflash` (CUDA) | upstream of `dflash-mlx` | MIT |
| `mtplx` | [youssofal/mtplx](https://github.com/youssofal/mtplx) | Apache 2.0 |

`mtplx` ships a forked `mlx-mtplx` runtime that conflicts with upstream
`mlx`, so it lives in an isolated venv at `~/.chaosengine/mtplx-venv/`.
See the [MTPLX page](../features/mtplx.md).

## Optional Apple Silicon video runtime

| Package | Repository | Licence |
|---|---|---|
| `mlx-video` | [Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video) | MIT |

Subprocess engine for Wan 2.1 / 2.2 / LTX-2 T2V / I2V / A2V. Not bundled;
installed on demand from the Setup page on Apple Silicon hosts.

## Ported algorithms

### DDTree (Diffusion Draft Tree)

Ported from [liranringel/ddtree](https://github.com/liranringel/ddtree) into
[`backend_service/ddtree.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/ddtree.py)
with adapter to dflash-mlx's per-family target ops. MIT licensed upstream.

## Frontend / shell

- [Tauri 2](https://tauri.app/) — Apache 2.0 / MIT dual.
- [React 18](https://react.dev/) — MIT.
- [Vite](https://vitejs.dev/) — MIT.
- [shadcn/ui](https://ui.shadcn.com/) (selected components) — MIT.

## Internationalization

The FU-042 i18n infrastructure landed:

| Layer | Package | Licence |
|---|---|---|
| Frontend i18n | `i18next` | MIT |
| Frontend React bindings | `react-i18next` | MIT |
| Frontend ICU formatter | `i18next-icu` | MIT |
| Frontend lang detection | `i18next-browser-languagedetector` | MIT |
| ICU message format | `intl-messageformat` | BSD-3-Clause |
| Python message catalogs | `Babel` | BSD-3-Clause |
| Rust localization | `rust-i18n` | MIT |
| Fluent bindings | `fluent-bundle` | Apache 2.0 |
| Language identifier | `unic-langid` | Apache 2.0 / MIT dual |

## Adding a new dependency

When adding a new direct or vendored dependency:

1. **Check the licence.** Must be permissive (MIT, Apache 2.0, BSD, or
   similar). GPL / AGPL / SSPL are not compatible.
2. **Add an entry** to [`THIRD_PARTY_NOTICES.md`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/THIRD_PARTY_NOTICES.md).
3. **Pip package?** Add to `_INSTALLABLE_PIP_PACKAGES` in
   `backend_service/routes/setup/_install_helpers.py` so the Setup tab
   can pre-stage it.
4. **System binary?** Add to `_installable_system_packages()` in the same
   file.
5. **Update the docs.** Mirror the entry here.
6. **Pre-build check.** Phase 4 (Licence notices) validates that
   `THIRD_PARTY_NOTICES.md` lists every direct + vendored dep.

## See also

- [Coding guidelines → Adding dependencies](../contributing/coding-guidelines.md)
- [Pre-build check](../testing/pre-build-check.md)
