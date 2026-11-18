# Review stream D: whole-app sweep

Branch `review/d`, worktree `/Users/dan/ChaosEngineAI-review-d`, cut from `release/v0.10.0` at `b31c4ea5`.
Plan: `docs/plans/overnight-e2e-review.md`. Scope: every tab except Benchmarks, Conversion, Fine Tuning, Prompts, Plugins, Logs and Server.
Budget cut by the coordinator to 3.5 hours (start 07:36, 2026-11-18), so the sweep is ordered by user impact:
Chat, Models, Studios, first run, backend-down recovery, then the long tail.

## Setup (how to resume)

- Python: `/Users/dan/ChaosEngineAI/.venv/bin/python`. This worktree has no `.venv`, so Playwright needs `E2E_PYTHON` set to it.
- `e2e/node_modules` is a symlink to the main checkout's (git-ignored), Playwright 1.65.1, Chromium headless shell already installed.
- Ports: backend 8903, Vite 8933. One backend at a time: the Playwright run and the hand walk both use 8903, never together.
- Playwright: `cd e2e && E2E_PYTHON=/Users/dan/ChaosEngineAI/.venv/bin/python E2E_WEB_PORT=8933 E2E_API_PORT=8903 npx playwright test --workers=1`
- Hand walk: backend with a throwaway `HOME` (mktemp), `CHAOSENGINE_PORT=8903`, dummy token; Vite with `VITE_CHAOSENGINE_API_BASE=http://127.0.0.1:8903`.

## Log (newest last)

| Time | Check | Result | Evidence |
|---|---|---|---|
| 07:40 | `npx tsc --noEmit` on the untouched tip | PASS | exit 0 |
| 07:40 | `npm run i18n:validate` on the untouched tip | PASS | en 4951/4951, nine locales 4947/4951 (99.9 %), tlh 82.6 % (outside the gate) |
| 07:41 | First Playwright start | FAIL (setup, not the app) | `ModuleNotFoundError: No module named 'fastapi'`: the config walks up for `.venv` and a sibling worktree has none. Fixed by `E2E_PYTHON`. |
| 07:42 | Baseline Playwright run on the untouched tip (179 tests, one worker) | running | log in the scratchpad, result recorded below |

## Notes found while reading (before the browser walk)

- The app has **no light theme**: `src/styles.css` starts with `color-scheme: dark` and there is no theme switch or `data-theme`. The brief asks for light and dark; the sweep checks instead that an OS set to light changes nothing.
- `scripts/i18n-pseudo.mjs` documents `?devLocale=en-XA`, but nothing in `src/` reads that parameter, so the pseudo-locale cannot be opened in the app. The build checklist item "Pseudo-locale en-XA smoke-tested on Setup + Diagnostics" has no way to run by hand.
- `e2e/tests/helpers.ts` `openApp` waits 5 s for the terms dialog on every test even when the terms are accepted, so each test costs about 6 s (179 tests, about 18 minutes).
