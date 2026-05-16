# Pre-build check

`scripts/pre-build-check.sh` is the canonical gate that runs before every
release. It bundles every automated check the project ships into a single
9-phase sequence, prints a `PASS / FAIL / WARN` summary, and exits non-zero
on any failure.

```bash
./scripts/pre-build-check.sh
```

A second runner — `scripts/pre-build-check.mjs` — invokes the same set
from Node for the desktop release workflow. Both runners must stay in
sync; the workflow gate runs the `.sh` form locally and the `.mjs` form
from CI.

## What each phase validates

### 1. Python tests
```
.venv/bin/python -m pytest tests/ -q --tb=line
```
Every unit + integration test must pass.

### 2. TypeScript tests
```
npm test
```
Vitest runs every `src/**/*.test.ts` file.

### 3. TypeScript type-check
```
npx tsc --noEmit
```
Compile-time type safety.

### 4. Licence notices
Probes [`THIRD_PARTY_NOTICES.md`](../reference/third-party-deps.md) for
the expected dependency entries. Fails when a new entry is missing or a
removed entry is still listed.

### 5. Cache strategy validation
Imports each registered cache strategy and asserts its contract:

- `native` strategy's `llama_cpp_cache_flags()` only emits standard
  types: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`.
- TurboQuant strategy returns `required_llama_binary() == "turbo"`.
- Legacy `chaosengine` / `rotorquant` ids coerce to `turboquant` via
  `registry.resolve_legacy_id`.
- DFlash `_COMMUNITY_PREFIXES` includes all common model repo prefixes.

### 6. Upstream dependency check
Reads the `dflash-mlx` commit pin in both
[`pyproject.toml`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/pyproject.toml)
and
[`scripts/stage-runtime.mjs`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/scripts/stage-runtime.mjs).
Fails if they've drifted (this caught a real bug — the dev `.venv` was
on 0.1.5.1 while `npm run stage:runtime` was still bundling 0.1.4.1).

### 7. Binary availability
Confirms the bundled binaries exist or are resolvable:

- `llama-server` (Homebrew or bundled)
- `llama-server-turbo` (when `~/.chaosengine/bin/` is populated by
  `scripts/build-llama-turbo.sh`)
- `sd` (stable-diffusion.cpp, when populated by
  `scripts/build-sdcpp.sh`)

### 8. i18n locale validation
```
npm run i18n:validate
```
Every locale parity check + ICU syntax compile + orphan key scan.

### 9. E2E smoke
```
./scripts/e2e_test_suite.py --smoke
```
The smoke run validates that the backend reaches model load, generates
tokens, runs an image / video preflight probe, and the diagnostics
snapshot reports `recentOrphanedWorkers == []`. Wall time: < 60 s.

## Output

The script prints a summary at the end:

```
=== Summary ===
  PASS  Python tests
  PASS  TypeScript tests
  PASS  TypeScript type checking
  PASS  Licence notices
  PASS  Cache strategy validation
  PASS  Upstream dependency check
  PASS  Binary availability
  PASS  i18n locale validation
  PASS  E2E smoke — all phases green

9 passed, 0 failed, 0 warnings
```

Any `FAIL` line is a release blocker. `WARN` lines are advisory — they
flag issues that don't block a release but should be addressed soon.

## Exit codes

- **0** — every check passed (warnings are allowed).
- **non-zero** — at least one check failed.

## Required state

Some checks need state that may not exist in CI by default:

- **E2E smoke** needs the backend running on `127.0.0.1:8876`. The
  runner starts it as a side-effect for the duration of phase 9.
- **Cache strategy validation** needs the `turboquant`, `triattention`,
  etc. extras importable (or it'll downgrade to a `WARN` instead of
  asserting on the cache flags).
- **Binary availability** is a `WARN` when binaries aren't present —
  developer environments don't always have a full release setup.

In CI, run the pre-build check with the same set of extras you ship in
release builds.

## Skipping phases

There's no built-in `--phases` flag for the pre-build check today —
the phases are a tight bundle. If you need to isolate a single phase
during development, run the individual command directly (see the per-
phase sections above).

## See also

- [E2E testing](e2e-testing.md) — phase 9's contract.
- [Adding checks](adding-checks.md) — how to extend the E2E suite.
- [Contributing → Adding a feature](../contributing/adding-a-feature.md).
