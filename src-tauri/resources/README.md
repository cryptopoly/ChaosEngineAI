This directory is populated by `./scripts/stage-runtime.mjs`.

ChaosEngineAI stages an embedded runtime archive here before `tauri dev` and `tauri build`:

- `embedded/runtime-<platform-tag>.tar.gz` contains the backend payload, embedded Python home, pinned site-packages, and `llama-server` assets
- `embedded/runtime-<platform-tag>.manifest.json` describes how the native launcher should extract and run that payload

The staged payload is intentionally ignored by git.
