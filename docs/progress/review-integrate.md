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
| 8 | "Show experimental tabs" switch | PASS | `7c0556ba`: Settings > General > Appearance checkbox, off by default, stored with the sidebar preferences (`chaosengine.sidebar.prefs.v1`, the "Keep all sidebar groups open" pattern); `experimental` flag on `TabConfig`; Sidebar and sub-tab bar leave flagged tabs out; the App's one tab setter sends a hidden tab to the Dashboard. vitest `experimentalTabs.test.ts` |
| 9 | Hide Fine-Tuning and Plugins (separate commit, reverts alone) | PASS | `ce589407`: two `experimental: true` flags, README lines; Prompts, Benchmarks, Conversion stay. Playwright 01, 28, 29, 36: 15 + 5 passed (`--workers=1`, ports 8919/8910) after one race fix in the new spec |
| 10 | Owner decision (coordinator, 09:13): the Connect snippet rename is cancelled (compatibility targets are allowed); masking stays | DONE | nothing had been renamed |
| 11 | API keys masked on screen | PASS | `d6aa020b`: `maskSecret` (8 dots + last 4; dots only when the key is 8 characters or shorter), Reveal button, Copy copies the real text. Server key field (was first 4 + last 4), Connect samples, Test commands, Settings > Integrations, Agent access. vitest `secretMask.test.ts` + ConnectPanel test; Playwright `37-key-masking` 2 passed, `31-server-activity` 6 passed |
| 12 | Dashboard copy | PASS | `59da5ffa`: MLX title "Recommended target: 70B class at 4-bit" (en; other locales null so they fall back); the "Hardware profile refreshed" activity is written with the readiness line after the MLX probe. `tests/test_dashboard_recommendation.py` 3 passed; with `test_backend_service.py` and `test_licence_gate.py` 132 passed |
