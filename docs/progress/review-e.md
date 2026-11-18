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
| 08:09 | Real model run | WAITING | lock held by review-f since 07:51; retrying every 60 s for up to 50 min; backend restarted on new code (PID 28291) |
| 08:16 | Model lock taken (08:15:54), backend 8904 restarted with `HF_HOME` on the real cache, read only, `HF_HUB_OFFLINE=1` | DONE | first try failed to load because the throwaway HOME had no HF cache; the failure itself showed up as a `load_failed` event with command, env and error (kept in the history) |
| 08:16 | E2E phase 21 with a real model (`CHAOSENGINE_PORT=8904 e2e_test_suite.py --phases 21`) | PASS | Qwen3-0.6B-4bit MLX: kinds requested, resolving, loading, loaded (3.09 s, 1.04 GB), generation (301 tok/s), idle, unloaded (user); 5/5 checks |
| 08:17 | Real timeline: MLX load + streamed answer, GGUF load (MLX parked warm) + streamed answer, answer from the warm MLX engine, unload both | PASS | events 12 to 31; llama.cpp `loaded` detail carries the full `llama-server` command line; warm-engine answer attributed to the MLX model |
| 08:18 | Models unloaded, lock released (08:18:06) | DONE | no stream E worker left; the remaining `mlx_worker` (PID 3081) belongs to stream A's backend on 8900 |
| 08:19 | Fix: a warm model that answers stays warm on the timeline (was drawn as loaded-idle); detail fields labelled through i18n | PASS | vitest `src/features/server` 24 passed |
| 08:20 | Screenshots of the real timeline and detail | DONE | `docs/progress/review-e/timeline-real-load-*.png`, `server-activity-*.png`, `server-activity-detail-dark.png` |
| 08:22 | Full pytest (throwaway HOME) | PASS | `pytest tests/ -q -rfE`: exit 0, no FAILED or ERROR |
| 08:24 | Playwright 01-smoke, 31-compact-sidebar, 31-server-activity on 8904/8924 | PASS | 19 passed |
| 08:25 | Docs: README Server Activity, `docs/usage/server-mode.md` views, routes, CLI | DONE | |
| 08:27 | A load replaced or cancelled is a `warning` (reason `cancelled`), not an error | PASS | `tests/test_model_events.py` 27 passed |
| 08:30 | Arrow keys on the Overview/Activity tabs | PASS | Playwright 31-server-activity 6 passed; vitest 153 files, 1839 tests |

## Left for the owner or a follow-up

- Delete after the owner agrees: `src/features/logs/LogsTab.tsx`, `LogsIcon` in `src/components/icons/SidebarIcons.tsx`, locale keys `diagnostics.logsTab.*`, `common.tabs.logs`, `common.tabCaptions.logs` (all unreferenced now). `src/hooks/useServerLog.ts` was already unused before this stream.
- No agent-bridge tool reads the events yet (golden MCP list unchanged); a `server_events` tool would let an agent see why a load failed.
- The app has one (dark) theme; the light screenshots show it ignores `prefers-color-scheme`. The new CSS uses theme tokens only.
- Pre-existing editorial issue moved, not added: the connect snippets are titled with other apps' names (`serverTab.connect.openWebui`, `ollamaApps`, `ollamaHint`, `continueDev`). Rename needs the owner's call (it touches 10 locales).
- `PHASES[21]` may clash with stream D's planned phase 21; the integrator renumbers one of them.
- Not run: `e2e_test_suite.py --smoke` and the full suite (stream G), Windows and Linux, the packaged Tauri window.
