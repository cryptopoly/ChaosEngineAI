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
| 9 | Hide Fine-Tuning and Plugins (separate commit, reverts alone) | PASS | first cut `ce589407` also carried the spec and README lines, so `git revert` hit a conflict in `e2e/README.md` once spec 37 sat next to it; reverted (`c3df1018`) and re-cut: `a992f154` (helper, specs 28 and 29 turn the switch on, spec 36 reads the flags from `tabs.ts`) and **`41f5a92a`** (two `experimental: true` flags, README lines, vitest block). `git revert --no-commit 41f5a92a` on the tip applies clean. Prompts, Benchmarks, Conversion stay. Playwright 01, 28, 29, 36: 15 passed |
| 10 | Owner decision (coordinator, 09:13): the Connect snippet rename is cancelled (compatibility targets are allowed); masking stays | DONE | nothing had been renamed |
| 11 | API keys masked on screen | PASS | `d6aa020b`: `maskSecret` (8 dots + last 4; dots only when the key is 8 characters or shorter), Reveal button, Copy copies the real text. Server key field (was first 4 + last 4), Connect samples, Test commands, Settings > Integrations, Agent access. vitest `secretMask.test.ts` + ConnectPanel test; Playwright `37-key-masking` 2 passed, `31-server-activity` 6 passed |
| 12 | Dashboard copy | PASS | `59da5ffa`: MLX title "Recommended target: 70B class at 4-bit" (en; other locales null so they fall back); the "Hardware profile refreshed" activity is written with the readiness line after the MLX probe. `tests/test_dashboard_recommendation.py` 3 passed; with `test_backend_service.py` and `test_licence_gate.py` 132 passed |
| 13 | Full pytest on `68e35083` (throwaway HOME) | PASS | exit 0: 4684 passed, 17 skipped, 1 xfailed (counted from the progress line; the repo's `addopts` plus `-q` hides the summary). The real `~/.chaosengine/prompt_templates.json` was not touched (mtime 08:25, before the run) |
| 14 | `e2e_test_suite.py --smoke` (own backend 8910, throwaway HOME, `HF_HUB_OFFLINE=1`) | PASS | phases 0, 3, 4, 5, 6, 7, 8: 81 checks; phase 212 alone: 5 checks |
| 15 | Merge `review/plan` (`1ce012ab`, one plan file) | PASS | clean. Its rows D-89 to D-92 and FU-230 to FU-246 added (links repointed from `review/plan` to `release/v0.10.0`); the integrator's own owner-decision row moved from D-89 to D-93 and its FU-230 to FU-247 (FU-221 to FU-229 did not clash). Hub rebuilt, 0 warnings; `test_hub_tracker.py` 5 passed |
| 16 | Full Playwright suite on `0e1d6fdb` (A, B, C, E, F merged; switch, hide, masking, Dashboard), `--workers=1`, ports 8919/8910 | PASS | 213 passed in 25.1 min, no retries |
| 17 | `scripts/check-upstreams.py` | PASS (report only) | 1 behind: llama.cpp pinned b12441, newest b12503 (pre-existing; not this run's to bump) |
| 18 | `scripts/audit-dependencies.py` | PASS | 1006 packages; nothing blocking (one high, `picomatch` 4.0.4 ReDoS, is dev-only); 109 out of date, 26 by a major version |
| 19 | Merge `review/d` (`31e9d194`) | PASS | conflicts: 11 `common.json` (3-way JSON merge by key, 0 leaf clashes), `ModelLaunchModal.tsx` (D's dialog focus handling replaces F's Escape effect; same behaviour, one handler), `e2e/README.md` (both), one test screenshot (D's). My specs renumbered to 38 and 39 (D added 36 and 37). tsc clean; vitest 165 files, 1899 tests; `i18n:validate` OK; Playwright 01, 27 to 39: 67 passed (4.7 min), D-13's spec `37-chat-local-model` included |
| 20 | D's left items | DONE | FU-248 (backend trusts `canonicalRepo`, L-16), FU-249 (Diagnostics strings, L-2), FU-250 (language banner, L-1), FU-251 (the rest); L-12 is the masking above. L-18 placeholder half fixed (`composerPlaceholderKind`, vitest); L-20 not trivial (a loading flag through `useImageState`, the tab and a new string), left in FU-251 |
| 21 | Merge `review/gguf` (`f7283367`) | PASS | conflicts: `docs/E2E_TESTING.md` (its phase was 213, which stream C holds: now **215**, registered in the review block; `phase=215` in the phase file), `src/utils/__tests__/library.test.ts` (imports from both). CLAUDE.md testing row lists `test_gguf_routing.py` and phases 211 to 215. `test_gguf_routing`, `test_conversion_helpers`, `test_inference`, `test_tensorfold_routing`, `test_benchmark_eval`, `test_backend_service`, `test_prompts_plugins_e2e_phase`: 264 passed; vitest `src/utils` 342 passed; tsc clean; Playwright 01, 27, 37: 18 passed |
