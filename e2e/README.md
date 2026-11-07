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

Every run starts as a first launch, so the terms screen appears. Specs run in file order with one worker and share
one backend, so `01-smoke` (which accepts the terms last) must stay before specs that assume they are accepted.
Anything that is not loopback is aborted in `tests/fixtures.ts`; import `test` from there, not from `@playwright/test`.

Output (git-ignored): `e2e/test-results/` (screenshots, traces on failure only), `e2e/playwright-report/`, `e2e/.run/`.
Traces contain request bodies and headers; do not share them from a run that used real tokens (there are none by default).

## Specs

- `01-smoke`: shell renders, sidebar order, terms screen on a clean first run, accepting it sticks.
- `02-gallery-project-filter`: the Project dropdown in the Image gallery (open, search, keyboard, Escape returns focus). Gallery outputs are mocked at `/api/images/outputs`.
