# Testing overview

ChaosEngineAI has three complementary test surfaces. Each one catches a
different class of regression; together they're the safety net before a
release.

## Test taxonomy

| Suite | Tool | Scope | Speed |
|---|---|---|---|
| **Python unit / integration** | `pytest` | Pure-function helpers, FastAPI route shape, engine adapters with mocked subprocess, cache strategy contracts. | ~30 s on a warm box. |
| **TypeScript unit** | `vitest` | React utilities, hooks, API client shapes, format helpers. | ~5 s. |
| **TypeScript type-check** | `tsc --noEmit` | Compile-time type safety across the React + Tauri IPC surface. | ~10 s. |
| **End-to-end (E2E)** | `scripts/e2e_test_suite.py` | Real-engine routing, model loads, cache strategies, speculative decoding, image / video generation when models are on disk. | 60 s (smoke) → 25 min (full). |

## When to run what

| Task | Run |
|---|---|
| Editing a pure-function helper | `pytest tests/<file>::TestClass` for that module. |
| Touching a FastAPI route | `pytest tests/test_backend_service.py` + the route's specific test. |
| Touching `inference/controller.py` or an engine adapter | Full `pytest tests/` + `./scripts/e2e_test_suite.py --phases 0,1,7`. |
| Bumping `mlx-lm`, `llama.cpp`, `diffusers`, `dflash-mlx`, `turboquant-mlx-full`, or `mtplx` | Full `pytest tests/` + full `./scripts/e2e_test_suite.py`. |
| Changing TypeScript | `npm test` + `npx tsc --noEmit`. |
| Cutting a release | Everything — see [Pre-build check](pre-build-check.md). |

## Required commands

ChaosEngineAI tests run against **the installed app's runtime** — the
same torch / diffusers / mlx / nunchaku / etc. wheels users have
installed via the in-app "Install GPU runtime" + per-feature install
buttons. No custom dev setup. The flow is:

1. Open the ChaosEngineAI app (the Tauri shell launches the backend
   on port 8876 and adds the persistent extras dir to its `PYTHONPATH`).
2. From any shell, run the test suites below.

```bash
# Python tests — auto-loads the app's extras dir via tests/conftest.py
.venv/bin/python -m pytest tests/ -q

# TypeScript tests — no backend dependency
npm test

# Type-check
npx tsc --noEmit

# E2E smoke — talks to the running app on 127.0.0.1:8876
.venv/bin/python scripts/e2e_test_suite.py --smoke
```

### Why the app's extras, not the dev venv?

The dev `.venv` ships with FastAPI + pytest + huggingface-hub but
deliberately **without** torch / diffusers / mlx / nunchaku /
sageattention / triattention / vllm. Those heavy packages live in the
persistent extras directory at:

- Windows: `%LOCALAPPDATA%\ChaosEngineAI\extras\cp{XY}\site-packages`
- macOS: `~/Library/Application Support/ChaosEngineAI/extras/cp{XY}/site-packages`
- Linux: `${XDG_DATA_HOME}/ChaosEngineAI/extras/cp{XY}/site-packages`

`tests/conftest.py` auto-discovers that path at pytest collection time
and adds it to `sys.path` (via [`ensure_extras_on_sys_path`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/backend_service/runtime_paths.py)),
so `import torch` in a test resolves against the same wheel a user
runs. A torch upgrade landing via the in-app installer is reflected in
the next `pytest` run automatically — no `pip install` dance required.

Set `CHAOSENGINE_TEST_TRACE_EXTRAS=1` to log which extras path got
prepended for a given run (useful when debugging "is this test
hitting the install I think it is?").

### Headless dev backend (advanced)

Contributors who want to run the suite without the Tauri shell open
can stand up the backend headlessly:

```bash
# One shell — runs the FastAPI app under the dev venv
.venv/bin/python -m backend_service.app --port 8876

# OR (gets the embedded runtime via Tauri's stage script)
npm run tauri:dev
```

This works, but won't exercise the exact `python-build-standalone`
binary the desktop bundle ships — for release-blocking validation,
prefer the production-app path above.

## Where the tests live

| Path | What's tested |
|---|---|
| [`tests/test_backend_service.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_backend_service.py) | FastAPI route shape + FakeRuntime contracts. |
| [`tests/test_services.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_services.py) | Backend service helpers. |
| [`tests/test_inference.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_inference.py) | Engine routing, binary resolution, `_select_engine`. |
| [`tests/test_setup_routes.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_setup_routes.py) | Install endpoints + job pattern. |
| [`tests/test_cache_strategies.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_cache_strategies.py) | Cache strategy registry + flags. |
| [`tests/test_dflash.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_dflash.py) | DFlash draft model registry. |
| [`tests/test_agent.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_agent.py) | Tool-call parser + dispatch. |
| [`tests/test_cache_strategy_matrix_runner.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/tests/test_cache_strategy_matrix_runner.py) | Cross-strategy sweep runner. |
| [`src/**/*.test.ts`](https://github.com/cryptopoly/ChaosEngineAI/tree/staging/src) | Frontend unit tests. |
| [`scripts/e2e_test_suite.py`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/e2e_test_suite.py) | End-to-end suite. |

## Patterns

### Python — `FakeRuntime`

`test_backend_service.py` uses a `FakeRuntime` that fulfils the
`RuntimeController` interface without spawning real subprocesses. Most
route tests stand up a `TestClient` against a FastAPI app with a
`FakeRuntime` wired in via dependency override.

### TypeScript — factories + `vi.stubGlobal`

Frontend tests prefer `makeVariant()` / `makeSession()` factory helpers
over hand-rolled fixtures. `vi.stubGlobal("fetch", mockFn)` is the
canonical API-mocking pattern.

### E2E — phase functions + skip semantics

Each phase is a function in `scripts/e2e_test_suite.py` that appends
named checks. A check returns `("pass" | "fail" | "skip", reason, detail)`.
Skips are first-class — when a required model isn't on disk, the check
reports `skip` (not `fail`) and the suite stays green.

## What unit tests don't cover

- **Real subprocess interactions.** MLX worker IPC, `llama-server`
  startup, MTPLX subprocess routing — all mocked in pytest. The E2E
  suite covers the real paths.
- **Real GPU.** Capabilities probes are mocked; the E2E suite exercises
  actual hardware.
- **Real Hugging Face downloads.** Stubbed in pytest; the E2E suite
  skips download-dependent checks when the model isn't already on disk.

For anything that requires the real backend behaviour, write an E2E
check — see [Adding checks](adding-checks.md).
