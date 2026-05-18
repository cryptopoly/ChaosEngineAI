# Development setup

ChaosEngineAI is a polyglot project — Rust (Tauri shell), TypeScript (React),
and Python (FastAPI backend). The dev loop optimises for fast iteration on
all three at once.

## One-time bootstrap

```bash
git clone https://github.com/cryptopoly/ChaosEngineAI.git
cd ChaosEngineAI

# Python backend
python3 -m venv .venv
.venv/bin/pip install -e .

# JS dependencies
npm install

# Stage the bundled runtime (only needed once, or after pulling)
npm run stage:runtime
```

You also want the [optional extras](../getting-started/installation.md#optional-extras)
for whichever features you'll touch.

## Daily dev loop

```bash
# Full desktop app — Tauri shell + Vite HMR for the React frontend
npm run tauri:dev
```

This starts the Tauri shell, which starts the Python backend, which starts
Vite. Changes to:

- `src/**/*.{ts,tsx,css}` → hot-reload through Vite.
- `backend_service/**/*.py` → backend restart required. The Tauri shell
  doesn't watch Python files; kill + relaunch (or run the backend
  separately and point the Tauri shell at it).
- `src-tauri/**/*.rs` → Tauri recompile + relaunch.

For backend-only work, skip the Tauri shell:

```bash
./scripts/chaosengine-cli serve
```

Then drive the backend with the CLI. This is the fastest loop for
inference / routing / cache strategy work.

For frontend-only work, use Vite standalone:

```bash
npm run dev
```

This runs Vite at `http://localhost:5173/` against an already-running
backend on 8876.

## Running tests

```bash
# Python tests
.venv/bin/python -m pytest tests/ -q

# Single file / class / test
.venv/bin/python -m pytest tests/test_cache_strategies.py -v
.venv/bin/python -m pytest tests/test_dflash.py::TestDraftMap::test_qwen3_canonical

# TypeScript tests
npm test
npm test -- src/utils/__tests__/format.test.ts

# Type-check
npx tsc --noEmit

# Pre-build gate (everything)
./scripts/pre-build-check.sh

# E2E smoke (60s)
./scripts/e2e_test_suite.py --smoke
```

See [Testing overview](../testing/overview.md) for the full taxonomy.

## Useful CLI commands during dev

```bash
# Live diagnostics tail in another shell
watch -n2 './scripts/chaosengine-cli diagnostics-snapshot | jq ".runtime, .capabilities"'

# List recent log entries
./scripts/chaosengine-cli diagnostics-log-tail --lines 200

# Drive a load + prompt without the UI
./scripts/chaosengine-cli load "Qwen/Qwen3-4B"
./scripts/chaosengine-cli prompt "hi" --max-tokens 8 --metrics
./scripts/chaosengine-cli unload

# Watch hardware metrics during a workload
watch -n1 './scripts/chaosengine-cli metrics-gpu | jq "."'
```

## Linting

```bash
# Python — ruff
.venv/bin/ruff check backend_service/ tests/ scripts/

# TypeScript — eslint
npm run lint
```

Pre-commit hooks (`.pre-commit-config.yaml`) run a subset of these
automatically — install them with `pre-commit install` once after
cloning.

## Editor setup

We don't enforce a specific editor. The repo includes:

- A `pyproject.toml` configured for `ruff` (line length 100, target
  Python 3.11+).
- A `tsconfig.json` configured for `tsc --noEmit` strict mode.
- A `vite.config.ts` for the React build.
- A `vitest.config.ts` for the TypeScript test suite.

VS Code users will get the same checks via the Python + TypeScript
extensions. Vim / Neovim users with `pylsp` / `pyright` + `tsserver`
get the same.

## Building a release

```bash
# Stage for release (bundles Python runtime + llama.cpp binaries)
npm run stage:runtime:release

# Build the signed bundle
npm run tauri:build

# Or, for unsigned local macOS app + DMG:
npm run release:macos -- --skip-sign --skip-notarize
```

Release artifacts land in `src-tauri/target/release/bundle/`.

The full release flow is tag-driven — push `vX.Y.Z` and the GitHub
Actions release workflow builds signed bundles for macOS, Linux, and
Windows in parallel, generates the `latest.json` updater manifest,
and stages a draft release.

## See also

- [Coding guidelines](coding-guidelines.md) — the human version of `CLAUDE.md`.
- [Adding a feature](adding-a-feature.md) — the E2E coverage gate.
- [Pre-build check](../testing/pre-build-check.md).
