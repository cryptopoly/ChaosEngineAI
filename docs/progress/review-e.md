# Stream E progress: Server page and Activity (D-88)

Branch `review/e`, worktree `/Users/dan/ChaosEngineAI-review-e`, backend port 8904. Spec: [server-activity-design](../plans/server-activity-design.md).

Order (coordinator, 07:40): backend events, collapsed connect block, timeline and event list, Activity tab fold-in. Report due by 11:05.

| Time | Step | Result | Evidence |
|---|---|---|---|
| 07:42 | Spec | DONE | `docs/plans/server-activity-design.md` |
| 07:45 | Backend events (`model_events.py`, controller hooks, `/api/server/events` + stream, log stream backlog fix) | PASS | `pytest tests/test_model_events.py` 26 passed; `test_inference test_backend_service test_services test_route_contracts test_diagnostics_routes test_mcp_contract test_cli_smoke` no failures |
| 07:55 | Server page split: Overview / Activity views, collapsed connect line (remembered), timeline, event list with detail, copy/export selected, pause, latest failure, old log lines under Activity | PASS | `npx vitest run src/features/server` 23 passed; `npx tsc --noEmit` clean; `npm run i18n:validate` OK (en filled, 9 locales null) |
| 07:57 | Logs fold-in: `logs` is a legacy id that opens Server > Activity (`tabRedirects.ts`), saved ids map to `server`, sidebar entry removed, `LogsTab.tsx` kept unreferenced | PASS | full `npx vitest run` 153 files, 1838 tests passed; tsc clean; i18n OK |
| 07:58 | Playwright `e2e/tests/31-server-activity.spec.ts` (mocked event stream) | PASS | `E2E_PYTHON=... E2E_WEB_PORT=8924 E2E_API_PORT=8904 npx playwright test tests/31-server-activity.spec.ts --workers=1`: 6 passed |
| 07:59 | E2E phase 21 (`scripts/e2e_phases/phase_21_server_activity.py`, registered as `PHASES[21]`) | WRITTEN | run pending the model lock |
| 08:03 | Overview screenshots (dark, light emulation, 390 px) | DONE | `docs/progress/review-e/server-overview-*.png`. The app has one (dark) theme; the light shot shows it ignores `prefers-color-scheme` |
| 08:03 | Real model run | WAITING | model lock held by another stream since 07:51; backend 8904 (PID 6396) and Vite 8924 up |
