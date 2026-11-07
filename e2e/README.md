# Browser E2E (Playwright)

Dev-only. Not part of `npm test`, `npm install` or the release build; the root `package.json` has no Playwright dependency. Conditions and risks: [docs/research/playwright-security.md](../docs/research/playwright-security.md).

## One-time setup

```
cd e2e
npm ci --ignore-scripts
npx playwright install chromium --only-shell   # explicit browser download, headless shell only
```

Never use `--with-deps` outside CI, the Chrome/Edge channel reinstall commands, or `@playwright/mcp`.

## Run

From the repo root: `npm run e2e` (the root needs its own `npm ci` so Vite is available). Or `cd e2e && npm test`.

Playwright starts both servers itself:

| Server | Port | Notes |
|---|---|---|
| Vite dev server | 8909 | `VITE_CHAOSENGINE_API_BASE` points it at the backend |
| Backend (repo `.venv`, or `E2E_PYTHON`) | 8910 | fresh `HOME` at `e2e/.run/home` on every run, dummy token, empty HF tokens |

Two runs at once (another worktree): `E2E_WEB_PORT=8911 E2E_API_PORT=8912 npm run e2e`.
The Vite server for the run sets `VITE_CHAOSENGINE_HOME_STUDIO=1` so the Music and Storyboard tabs exist.

Every run starts as a first launch, so the terms screen appears. Specs run in file order with one worker and share
one backend, so `01-smoke` (which accepts the terms last) must stay before specs that assume they are accepted.
Anything that is not loopback is aborted in `tests/fixtures.ts`; import `test` from there, not from `@playwright/test`.

Output (git-ignored): `e2e/test-results/` (screenshots, traces on failure only), `e2e/playwright-report/`, `e2e/.run/`.
Traces contain request bodies and headers; do not share them from a run that used real tokens (there are none by default).

## Specs

- `01-smoke`: shell renders, sidebar order, terms screen on a clean first run, accepting it sticks.
- `02-gallery-project-filter`: the Project dropdown in the Image gallery (open, search, keyboard, Escape returns focus). Gallery outputs are mocked at `/api/images/outputs`.
- `03-libraries-projects`: Video, Music and Voice libraries with a Storyboard project: dropdown, filtering, the "From project X" badge, header counts, Open in Storyboard (library data mocked, the project seeded through the backend).
- `04-music-sounds-licence`: Music | Sounds switch, the Stable Audio Open licence screen and its "Powered by Stability AI" line (the backend's own catalog).
- `05-storyboard`: create a project, add shots, takes and lock, captions panel, export panel (forced AI tag, run estimate, Film list). Projects are seeded with a tiny still and a tone; nothing is rendered.
- `06-busy-advice`: `/api/busy-advice` mocked for ok, slower and risky: the Chat load dialog ("Load anyway" gating) and the Music Studio estimate strip.
- `07-voice-clone`: the clone statement, consent gates, consent sentence with code, no save without a recording, AI label on cloned speech.
- `08-settings-terms`: Settings terms panel (read only), and an older accepted version bringing the terms back.
- `09-mobile`: 375 px wide, no sideways scroll, a full-page screenshot per screen in `e2e/screenshots/` (git-ignored). One known failure is marked `test.fail` (F-1 in [docs/research/e2e-findings.md](../docs/research/e2e-findings.md)).

Spec order matters: `01-smoke` accepts the terms; `08` accepts them again against the same backend and leaves them accepted. `tests/helpers.ts` has the shared helpers (open a tab, seed a project, fixture media under the run's throwaway HOME).
