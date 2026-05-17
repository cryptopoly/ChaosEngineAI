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

```bash
# Python tests
.venv/bin/python -m pytest tests/ -q

# TypeScript tests
npm test

# Type-check
npx tsc --noEmit

# E2E smoke
./scripts/chaosengine-cli serve &  # one shell
./scripts/e2e_test_suite.py --smoke  # another shell
```

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
