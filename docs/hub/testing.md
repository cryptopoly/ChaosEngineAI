---
title: Testing
order: 6
summary: How to run and test, the release gate, definition of done, launch and debug commands.
---
# Testing

Every command here is copy-pasteable from `/Users/dan/ChaosEngineAI`. Add each new test command the day you add the test. The release checklist and the table of "which tests for which area" are in `CLAUDE.md` (Build Checklist, Testing Requirements); this page is the short version.

## How to run and test

```bash
# everything Python (about 3 minutes; mocks the memory gate, writes nothing to real settings)
.venv/bin/python -m pytest tests/ -q
# one file
.venv/bin/python -m pytest tests/test_ltx2_mlx_runtime.py -q
# frontend tests and types
npm test
npx tsc --noEmit
# translations: parity, ICU syntax, orphans
npm run i18n:validate
# end-to-end against a live backend: fast smoke (60 s), then the full sweep
./scripts/e2e_test_suite.py --smoke
./scripts/e2e_test_suite.py
# Tauri shell
cargo test --manifest-path src-tauri/Cargo.toml
# every Hub repo and pinned file in the catalogs exists (needs network)
.venv/bin/python scripts/audit-catalog-repos.py
# security and freshness of pinned dependencies (needs network)
.venv/bin/python scripts/audit-dependencies.py
# the whole pre-release gate
./scripts/pre-build-check.sh
# the hub itself
./tools/run_tests.sh
```

The cross-strategy runner `scripts/cache-strategy-matrix.py` (`--quick` about 5 minutes) sweeps cache strategy by speculative method by model through a live backend.

## Launch and debug commands

| What | Command or flag | Notes |
|---|---|---|
| Backend on the usual port | `CHAOSENGINE_PORT=8876 .venv/bin/python -m backend_service.app` | add `CHAOSENGINE_EXTRAS_SITE_PACKAGES=/tmp/ce-no-extras` to ignore the extras overlay |
| Frontend dev server | `npm run dev -- --port 5175` | set `CHAOSENGINE_ALLOWED_ORIGINS=http://localhost:5175` on the backend |
| Desktop app in dev | `npm run tauri:dev` | the owner tests from the main checkout |
| Use a test llama-server | `CHAOSENGINE_LLAMA_SERVER=/path/to/llama-server` | needed when the system one lacks a new architecture |
| Test environment from the release pins | a venv built with `pip install -c release-constraints.txt` | what release builds install; the dev `.venv` can lag it |

## Definition of done

- [ ] Python, frontend and type tests pass, on the working copy and for risky changes on a clean copy.
- [ ] The change was **observed working** in the app or against a live backend (screenshot, log line or measured number), not only compiled.
- [ ] A user-visible feature has an E2E check in the right phase of `scripts/e2e_test_suite.py`.
- [ ] Catalog rows pass `scripts/audit-catalog-repos.py`; locales pass `npm run i18n:validate`.
- [ ] The tracker row, release notes and this hub are updated and rebuilt.
- [ ] What was **not** tested is stated in the report (Windows and Linux most often).

## Per-artefact checklists

- **New model row:** repo and pinned files exist (audit script), size and footprint measured or marked estimated, licence in the note, gated repos say so, Download button fetches only the files the runtime reads.
- **New engine lane:** installer script staged in the release bundle (`BUNDLED_INSTALL_SCRIPTS`), status route, CLI command, Setup banner, E2E check, third-party notice.
- **New setting:** what happens when a model cannot use it (dropped with a note, not an error); see `backend_service/helpers/request_guards.py`.

> Note: an agent's report is a claim. Re-run the important checks before marking a task done.
