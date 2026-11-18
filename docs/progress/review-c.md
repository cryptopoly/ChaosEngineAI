# Overnight review, stream C: Prompts and Plugins

Branch `review/c`, cut from `release/v0.10.0` at `b31c4ea5`. Not merged. Plan: `docs/plans/overnight-e2e-review.md`.
Started 2026-11-18 07:36. Budget cut by the coordinator to 3.5 hours (report due about 11:05).

How to resume: backend on port 8902 with a throwaway HOME (below), Vite on 8942 (8903 was taken by another
stream's backend). Model work needs the shared lock `review-lock/model.lock` in the session scratchpad.

```
S=<scratchpad>/review-c
HOME=$S/home CHAOSENGINE_PORT=8902 CHAOSENGINE_HOST=127.0.0.1 CHAOSENGINE_API_TOKEN=review-c-token \
  CHAOSENGINE_ALLOWED_ORIGINS=http://127.0.0.1:8942 HF_HUB_OFFLINE=1 HF_TOKEN= \
  /Users/dan/ChaosEngineAI/.venv/bin/python -m backend_service.app
VITE_CHAOSENGINE_API_BASE=http://127.0.0.1:8902 VITE_CHAOSENGINE_API_TOKEN=review-c-token \
  npx vite --host 127.0.0.1 --port 8942 --strictPort
```

Two small MLX models are linked into the throwaway HOME's Hub cache (`Qwen3-0.6B-4bit`, `Qwen3-8B-4bit`); nothing
is copied, downloaded or deleted.

## Verdict (plan rubric)

| Feature | Works | Honest | Valuable | Clutter cost | Verdict |
|---|---|---|---|---|---|
| **Prompts** | Partly. Save, edit, delete, search, variables and "Use in Chat" work (see the log). Twelve defects are fixed here; the main one is only eased (the prompt is invisible in Chat and app-wide, checks 31 and 39). | No. It shows a "preset model, applied when you Use in Chat" badge and stores preset samplers and few-shot examples, and none of the three is ever applied. "Use in Chat" sets a system prompt that Chat never shows. | Yes, in one respect: it is the **only** way in the app to give Chat a system prompt. The template store itself (five generic personas, tags, categories) is not something a user would miss. | One sidebar slot under Tools; 438 + 301 + 104 lines; raw JSON boxes for variables and samplers; a label that still says "Phase 2.7". | **Merge into Chat**, then remove the sidebar entry. Proposal: `docs/plans/review-c-prompts-into-chat.md`. Until then keep it (it works after the fixes). |
| **Plugins** | No. Until this branch the page always said "No plugins found" (it read a key the route never sent; broken since the first release in April). With that fixed it lists the 9 cache strategies and 3 chat tools the app already has, with ON/OFF switches that change nothing and are forgotten on restart. A folder put where the page says is never read. | No. The page, the README (three places) and the docs promise external plugins "discovered from the app's plugin directory" and five extension types. No code path loads a plugin; three of the five types have no rows and no consumer. | No. Nothing a user can do needs it: chat tools from outside come through MCP servers (`settings.mcpServers`, `backend_service/mcp/loader.py`), instructions through Skills, outside agents through the bridge (`/mcp`), cache strategies through the model launch settings. | One sidebar slot; 181 + 132 + 50 lines; a public promise that is false. | **Hide behind the experimental switch now; remove after v0.10.0** unless a real plugin loader with a trust model is planned. Files that go with it are listed below. |

## Log

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | `GET /api/plugins` on a fresh data dir | PASS | 12 rows: `cache_strategy` 9, `tool` 3; the answer is the map itself, no `plugins` key |
| 2 | Plugins page, before the fix | **FAIL** | "0 plugins registered / No plugins found." with 12 rows in the backend. `PluginsTab.tsx` read `data.plugins`; the route has returned the bare map since `1d50fb51` (2026-04-14). Fixed in `01d79f7f` |
| 3 | Plugins page, after the fix | PASS | "12 plugins registered", all rows; screenshot looked at |
| 4 | Plugins page rows cut off | **FAIL**, fixed | The panels sat in a flex column inside a clipped frame: scroll height 876 in a 679 px panel, no scrollbar, the tools row unreachable at 1400 x 900. Fixed in `9cbb2dc9` (`.plugins-page` grid that scrolls) |
| 5 | Install a sample plugin | **FAIL** (nothing loads) | Wrote `<data>/plugins/review-sample/manifest.json` (`type: tool`, `entry_point: plugin:register`) and `plugin.py` that writes a marker file when imported. Restarted the backend. `GET /api/plugins` still 12 rows, no `review.sample`; no marker file. `discover_from_directory` has no caller (`git grep`), and it would only read the manifest: `entry_point` is never imported |
| 6 | Run a plugin | SKIP | Nothing can be run; there is no loader |
| 7 | Switch a built-in off | **FAIL** (no effect) | `POST /api/plugins/tool.calculator/disable` answers `enabled: false`; `GET /api/tools` still offers `calculator`. `cache.native` switched off still shows as the active strategy in the header. The flag is in memory only (back on after a restart) |
| 8 | Error path | PASS | Unknown id: 404 with a localized message. A path-shaped id (`..%2F..%2Fetc`): 404. No token: 401 on list and on enable |
| 9 | Uninstall | SKIP | There is no install, so nothing to uninstall; deleting the folder changes nothing |
| 10 | Trust and permission model | n/a | There is none because no outside code is loaded. That is the safe state. A unit test now pins it: a manifest is listed without importing its code, and nothing in the app scans a plugin folder (`tests/test_plugins.py`) |
| 11 | Safe-mode story | n/a | Nothing to disable at start-up: the registry is rebuilt from built-ins on every start |
| 12 | `GET /api/prompts` on a fresh data dir | PASS | 5 built-ins seeded into `<data>/prompt_templates.json` |
| 13 | Prompts page: dates | **FAIL**, fixed | "Created: 1794987548.93451 \| Updated: 1794987548.93451" (epoch seconds printed raw). Fixed in `01d79f7f` |
| 14 | Prompts page: save with no name | **FAIL**, fixed | The route answers 422 and the page did nothing: no message, editor still open. Now "Give the template a name before saving." (and a sentence for any other refused save) |
| 15 | Prompts page: search | **FAIL**, fixed | Search read name, category and tags, not the prompt text (the route's `query` does). Now it reads the text too |
| 16 | Delete every template, restart | **FAIL**, fixed | The five built-ins came back, because an empty library was re-seeded on load. Now only a missing or unreadable file seeds |
| 17 | Hand-edited library file with a bad row | **FAIL**, fixed | A row that is a number made every `/api/prompts` call a 500 (`TypeError` in `load`). Bad rows are now skipped |
| 18 | Prompts at 375 px | **FAIL**, fixed | The detail panel was 65 px wide beside a fixed 300 px list. The two now stack below 760 px and the page scrolls; the editor scrolls inside its panel on a short window (`9cbb2dc9`) |
| 19 | Plugins at 375 px | PASS after #4 | No sideways scroll, every row reachable |
| 20 | Light theme | n/a | The app has one theme (`color-scheme: dark`, no light tokens). Both tabs hard-code about 30 hex colours instead of the tokens, which will matter the day a light theme exists |
| 21 | Playwright `29-prompts-plugins` | PASS | 4 passed (25.6 s): built-ins, dates, refused save, variables, search by text, Use in Chat sends the filled prompt as `systemPrompt`, edit, delete; 375 px for both pages; Plugins lists what the backend registers and a switch round-trips |
| 22 | Import and export of templates | SKIP | The feature does not exist (no route, no button). The brief asked for it to be tested |
| 23 | Use from Studio | SKIP | Not offered: templates are Chat system prompts only. Image, Video, Voice and Music studios have their own presets and no link to this page |
| 24 | CLI: `prompts-list`, `prompts-create`, `prompts-delete` | PASS | list 5; create answers the template; delete rc 0; delete of an unknown id rc 1 with the 404 text |
| 25 | CLI: `plugins-list`, `plugins-disable`, `plugins-enable` | PASS | `{'cache_strategy': 9, 'tool': 3}`; switch off and on rc 0; unknown id rc 1 with the 404 text |
| 26 | **The test suite writes into the real prompt library** | **FAIL**, fixed | `~/.chaosengine/prompt_templates.json` on this Mac holds 510 templates; 505 are "Phase 0 test template". `tests/test_route_contracts.py::PromptsRouteTests` reset the module-level library to `None`, which then opened `DATA_LOCATION` (the real data directory), and its create test never deleted its row: one junk template per `pytest tests/` run without a throwaway HOME (the command in CLAUDE.md's build checklist). Proven both ways: the old test file run with `HOME=<tmp>` leaves `<tmp>/.chaosengine/prompt_templates.json`; the fixed one leaves none. Fixed in `28ce7b20`. **The 505 rows are still in the owner's library** (not touched: real data); clean-up command under "For the owner" |
| 27 | Prompts and Plugins with the list route failing (500) | **FAIL**, fixed | Prompts showed an empty list, Plugins "No plugins found." Both now say the list could not be loaded (`7c757fe6`); Playwright covers it |
| 28 | Prompts: empty list, search with no match | **FAIL**, fixed | A blank panel. Now "No templates yet. Use New to make one." / "No template matches the search." |
| 29 | Prompts: Delete | **FAIL**, fixed | One click deleted a template with no confirm and no undo. It now asks (`window.confirm`, the pattern Diagnostics and HTML Challenge use) |
| 30 | Prompts editor accessibility | **FAIL**, fixed | No field had an accessible name (labels were not tied to inputs). Each field now carries its label (`de0b5e6b`). The variables label said "Phase 2.7" in every locale; removed |
| 31 | After "Use in Chat", what Chat shows | **FAIL**, left | Applied "Translator", landed on Chat: the page has no mention of the template or of a system prompt (checked by text and by eye). The prompt is one `useState` in `useSettings.ts`: every thread gets it, nothing shows it, nothing clears it, a reload drops it. See the proposal |
| 32 | Preset model, preset samplers, few-shot examples | **FAIL**, wording fixed | Stored and shown, never applied (`onApplyTemplate` passes the prompt text only; no editor exists for few-shot examples). The badge's tooltip said "applied when you Use in Chat"; it now says Chat does not apply it yet (`ac9ead34`) |
| 33 | Plugins page text | **FAIL**, wording fixed | "Place plugin directories with a manifest.json in ~/.chaosengine/plugins/" pointed at a folder nothing reads (and the wrong folder when the data directory is moved). Now "External plugins are not loaded in this version." (`ac9ead34`, en only; other locales fall back to en). The README still makes the promise (lines 97, 168, 286 to 288) |
| 34 | Phase 21 (prompts and plugins part), in process | PASS | `tests/test_prompts_plugins_e2e_phase.py`: 9 checks pass, the real-model check skips (no model in the test), nothing left behind. The in-process run caught a mistake in the phase itself (a non-object variable is a 422, not a dropped row) |
| 35 | Prompts editor: text that is not valid JSON in the Variables or Preset samplers box | **FAIL**, fixed | Save succeeded and silently dropped what was typed (`parseEditVariables` returned `[]` on a parse error). Save is now refused with a sentence and nothing is lost |
| 36 | **Phase 21 against the running backend, with a real model** (model lock held 08:11 to 08:14) | PASS | `scripts/e2e_test_suite.py --phases 21`: 10 checks pass in 6.4 s. Real-model check: `mlx-community/Qwen3-0.6B-4bit`, prompt 19 tokens without the template and 43 with it, reply "pineapple" (the template's one word). Model unloaded after |
| 37 | **Prompts to Chat in the UI with a real model** | PASS, with the defect of check 31 | Template "reply with exactly one word: {{word}}", filled with MANGO in the form (the preview showed the resolved text), Apply to chat, asked "What is the capital of France?": the reply was "MANGO" (426 tok/s). **A new thread** asked for a planet also answered "mango": the prompt is app-wide. **After a reload** the same question answered "One planet in our solar system is Mars.": the prompt is gone, with nothing on screen in either case |
| 38 | Full `pytest tests/` (throwaway HOME) | PASS | exit 0, no FAILED or ERROR line; one expected failure (the plugin switch, on purpose) |
| 39 | Stopgap for check 31 | fixed in part | The Prompts page now shows the system prompt Chat is sending, with "Stop using it" (it could only be dropped by a reload before). Chat itself still shows nothing, and the prompt is still app-wide and lost on reload: the real fix is the proposal's step 1. Playwright: after Stop, the next message carries no `systemPrompt` |

## Commands

```
HOME=$(mktemp -d) .venv/bin/python -m pytest tests/test_prompts.py tests/test_plugins.py tests/test_route_contracts.py -q
npx vitest run src/features/plugins src/features/prompts
npx tsc --noEmit
npm run i18n:validate
cd e2e && E2E_PYTHON=/Users/dan/ChaosEngineAI/.venv/bin/python E2E_WEB_PORT=8952 E2E_API_PORT=8953 \
  npx playwright test tests/29-prompts-plugins.spec.ts --workers=1
```

(`e2e/node_modules` in this worktree is a link to the main checkout's, like the root `node_modules`; nothing was installed.)

## For the owner

- **Your own Prompts page has 505 junk rows.** `~/.chaosengine/prompt_templates.json` holds 510 templates and 505 of
  them are "Phase 0 test template", written by the test suite (check 26). The cause is fixed; the rows are not
  removed, because that is your data. With the app closed, this keeps a backup and removes them:

  ```
  .venv/bin/python - <<'PY'
  import json, pathlib, shutil
  p = pathlib.Path.home() / ".chaosengine" / "prompt_templates.json"
  shutil.copy(p, p.with_name("prompt_templates.backup.json"))
  rows = [t for t in json.loads(p.read_text(encoding="utf-8")) if t.get("name") != "Phase 0 test template"]
  p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
  print(len(rows), "templates kept")
  PY
  ```

- Decisions needed (proposal in `docs/plans/review-c-prompts-into-chat.md`): move Prompts into Chat, or only make
  the applied prompt visible per thread; hide Plugins behind the experimental switch now and remove it after the
  release, or plan a real loader with a trust model.
- Two wording changes were made without asking, each its own commit so it can be reverted (`ac9ead34`): the Plugins
  page no longer gives install instructions for a folder nothing reads, and the preset badge no longer says Chat
  applies it. The README still makes both promises (lines 64, 97, 130, 168, 282 to 288).

## For the integrator

- `scripts/e2e_test_suite.py` gets two lines: an import after the `phase_15` import and
  `PHASES[21] = _phase_21_prompts_plugins.joined(PHASES.get(21), sys.modules[__name__])` after `PHASES[15]`. `joined`
  runs whatever is already under 21 first, then this part, and merges the checks. If another stream assigns
  `PHASES[21] = ...` on a later line, it replaces this part without an error: keep this line last, or make the other
  stream's line the one before it.
- `src/styles.css`: one block added after `.workspace-content-frame:has(.discover-page)` (`.prompt-library-layout`,
  `.plugins-page`).
- Locale files: ten new keys under `common.promptLibrary` and one under `common.pluginsTab`, en filled and null
  elsewhere; four keys reworded in en and set to null elsewhere (`promptLibrary.presetModelTitle`,
  `pluginsTab.externalIntro`, `pluginsTab.externalManifestHint`, `panels.installFromDirectory`); the " — Phase 2.7"
  suffix removed from `promptLibrary.editor.variables` in every locale.
- CLAUDE.md's testing table has no row for Prompts and Plugins; suggested row (not added, the file is the lead's):
  `| Prompts and Plugins (template store, plugin registry and routes, phase 21 part) | test_prompts.py, test_plugins.py, test_route_contracts.py, test_prompts_plugins_e2e_phase.py | pytest tests/test_prompts.py tests/test_plugins.py tests/test_route_contracts.py tests/test_prompts_plugins_e2e_phase.py -v |`
- Suggested follow-up rows (not added to `FOLLOW_UPS.md`, to keep the merge clean):
  1. Prompts: the applied system prompt is app-wide, invisible in Chat, never cleared and lost on reload; make it
     per thread with a chip (proposal section 1, step 1).
  2. Prompts: preset model, preset samplers and few-shot examples are stored and never applied; apply or drop.
  3. Plugins: switches do nothing (`tests/test_route_contracts.py` has the expected-failure test) and no plugin can
     be loaded; hide, then remove or build a loader with a trust model.
  4. README and docs promise a plugin system with five extension types and discovery from a folder.
  5. Both tabs hard-code about 30 hex colours instead of the theme tokens.
  6. No "Show experimental tabs" switch exists yet; the plan's "hide" verdicts need it.
