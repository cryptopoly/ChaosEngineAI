# Review stream G: integration and verification

Branch `review/integrate`, worktree `/Users/dan/ChaosEngineAI-review-integrate`, cut from `release/v0.10.0` at `1bcb4e21`.
Plan: [overnight-e2e-review.md](../plans/overnight-e2e-review.md). Backend port 8910. Python `/Users/dan/ChaosEngineAI/.venv/bin/python`,
pytest always with a throwaway `HOME`.

## Log

| # | Step | Result | Evidence |
|---|---|---|---|
| 1 | Merge `review/a` (`02d41b40`) | PASS | no conflict |
| 2 | Merge `review/b` (`c4982f56`) | PASS | conflicts in `scripts/e2e_test_suite.py`, `docs/E2E_TESTING.md`, `e2e/README.md` (both sides kept); locales, `tests/test_route_contracts.py` merged clean |
| 3 | Merge `review/c` (`169be582`) | PASS | same three files, both sides kept |
| 4 | Merge `review/e` (`1410fa08`) | PASS | same three files; `App.tsx`, `styles.css`, `tabs.ts`, locales merged clean |
| 5 | Merge `review/f` (`072d68b7`) | PASS | no conflict |
| 6 | One phase scheme | PASS | review group 21, one key per stream: 211 A Benchmarks and Conversion, 212 B Fine-Tuning, 213 C Prompts and Plugins (was a "joined" part of 21), 214 E Server Activity (was 21). No key 21 any more. `docs/E2E_TESTING.md`, the CLAUDE.md testing table and `tests/test_prompts_plugins_e2e_phase.py` follow |
| 7 | Stream tests after the five merges | PASS | pytest (throwaway HOME) on the 12 stream test files: 312 passed, 1 skipped, 1 xfailed; `npx vitest run` 160 files, 1872 tests; `tsc --noEmit` clean; `i18n:validate` OK |
