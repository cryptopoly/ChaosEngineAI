# Changelog

## v0.6.0 - 2026-04-19

- Renamed the local `compression/` package to `cache_compression/` so it no longer shadows Python 3.14's PEP 784 stdlib `compression` namespace package. Fixes a `ModuleNotFoundError: No module named 'compression._common'` surfacing on Windows with Python 3.14 when PyTorch's import chain reached into the shadowed package.
- Made the My Models library RAM estimate use the actual on-disk size + KV cache heuristic instead of the catalog flagship's `estimatedMemoryGb`, so differently-sized variants of the same family no longer all render as the same ~76 GB value. Added a parallel compressed-cache estimate for the Compressed column.
- Video diffusion models (HunyuanVideo, Mochi, Wan2.x, LTX-Video, CogVideo, etc.) are now tagged `modelType="video"` during discovery and kept out of the chat-oriented My Models list and chat picker. They continue to surface under the dedicated Video section.
- Video-gen memory safety now includes the model footprint (with device-class fragmentation factors) in the safety verdict, preventing the 40-frame Wan 2.1 T2V 1.3B MPS crash on 64 GB Macs.
- Hardened Windows staging: `scripts/stage-runtime.mjs` now clears read-only attributes and retries on transient EPERM/EBUSY during `.runtime-stage` cleanup, and skips the dev-mode tar archive that Tauri ignores anyway. `build.ps1` pre-clears stale staging and installs the project via `pip install -e ".[desktop,images]"` so strict validation has its required extras.
- Bumped the application version to `0.6.0` across the npm, Python, and Tauri package metadata.

## v0.5.3 - 2026-04-18

- Fixed the GitHub Actions release workflow to use the valid `includeUpdaterJson` input for `tauri-apps/tauri-action@v0.6.0`, removing the repeated `uploadUpdaterJson` warnings from release builds.
- Bumped the application version to `0.5.3` across the npm, Python, and Tauri package metadata in preparation for the next release.
