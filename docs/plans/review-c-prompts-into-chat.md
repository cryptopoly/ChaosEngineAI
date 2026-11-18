# Proposal: Prompts moves into Chat; Plugins is hidden, then removed

Status: **proposal only, nothing here is built.** From the overnight review, stream C (2026-11-18). Evidence:
`docs/progress/review-c.md`. Needs the owner's decision.

## 1. Prompts: merge into Chat

### Why

- "Use in Chat" is the only way to give Chat a system prompt, and Chat does not show it. The prompt lives in one
  `useState` in `useSettings.ts` (`systemPrompt`): it applies to **every** thread, including old ones and voice
  calls, it is not shown anywhere in Chat, it cannot be cleared except by applying another template, and it is
  gone after a reload. A user who applied "Translator" yesterday cannot tell why today's thread behaves differently
  from yesterday's, in either direction.
- The page stores three things it never uses: a preset model (the badge says "applied when you Use in Chat"), preset
  samplers and few-shot examples. Chat already has per-thread samplers (`chat.samplers.<sessionId>`), a per-thread
  model and a temperature override.
- What overlaps today:

| Surface | What it holds | Where it is used |
|---|---|---|
| Prompts page | Named system prompts with `{{variables}}`, tags, a category | Chat, as a hidden app-wide system prompt |
| Skills (Settings > Skills, per-chat switch) | Folders of instructions (`SKILL.md`) with a security scan | Chat: the model reads a skill when the task matches |
| Chat slash menu | Ten commands (`/clear`, `/think`, `/tools`, `/model`, `/cancel`, `/export`) | Chat composer |
| HTML Challenge prompt library | Curated task prompts in a searchable modal | Fills the challenge prompt box |
| Studio presets (Image, Video, Voice, Music) | Style and size presets per studio | Their own studios; no link to Prompts |

A template is the same kind of thing as a skill (text that steers the chat model), chosen by hand instead of by the
model. It belongs next to the composer, where the Skills switch and the slash menu already are.

### What moves where

1. **System prompt becomes a per-thread setting shown in Chat.**
   - Storage: `chat.systemPrompt.<sessionId>` in local storage, the same pattern as `samplerOverrides.ts`,
     `temperatureOverride.ts` and `kvStrategyOverride.ts` (new file `src/features/chat/systemPromptOverride.ts`).
     `useChat.ts` reads it when it builds the request (it already does this for the samplers at the same place,
     line 778) instead of the app-wide `systemPrompt` argument.
   - UI: a chip in the composer's control row, next to Tools, Studio and Skills: "Prompt: Translator" with a clear
     button, or "Prompt" when none is set. Clicking it opens the picker below. Files: `ChatComposer.tsx` (the chip),
     `ChatTab.tsx` (state, as for `samplerOverrides`).
   - Remove `systemPrompt` / `setSystemPrompt` from `useSettings.ts`, `App.tsx` (two places) and the `useChat`
     signature.
2. **The template picker becomes a modal opened from Chat**, built like `ChallengePromptLibraryModal.tsx` (category
   tabs, search, card grid, Escape closes): pick a template, fill its variables, see the resolved text, Apply to
   this chat. "Edit" and "New" open the editor in the same modal. New folder `src/features/chat/prompts/`
   (`PromptPickerModal.tsx`, `PromptEditor.tsx`, and `promptLogic.ts` moved from `src/features/prompts/`).
3. **Slash menu:** `/prompt` opens the picker; `/prompt <name>` applies a template with no variables straight away;
   `/prompt off` clears it. One entry in `slashCommands.ts` plus `onOpenPromptPicker` and `onSetSystemPrompt` in
   `SlashCommandContext`.
4. **Editor clean-up while it moves:** variables as rows (name, type, default) instead of a JSON box; drop the
   "Phase 2.7" label; a confirm before Delete; an empty state and an error state for the list.
5. **Drop what is never used:** `presetModelRef`, `presetSamplers`, `fewShotExamples` from the editor and the
   request model (old files keep loading: unknown keys are ignored). Or wire the preset samplers into
   `writeSamplerOverrides` on Apply if the owner wants them; that is about 15 lines.
6. **Backend:** `routes/prompts.py` and `helpers/prompts.py` stay as they are (the store is sound after this
   branch's fixes). Optional: `apply_variables` / `extract_placeholders` have no caller outside the tests and can go.
7. **Sidebar:** remove `prompt-library` from `src/constants/tabs.ts`, `src/types/system.ts`, `src/app/tabViews.ts`
   and `PromptLibraryView` from `src/app/SystemTabs.tsx`; add a redirect to Chat in `src/constants/tabRedirects.ts` so a saved
   "last tab" still opens. Locale keys `tabs.promptLibrary`, `tabCaptions.promptLibrary`,
   `tabShortLabels.promptLibrary` go; `promptLibrary.*` stays (the modal uses it).

### Tests that move or change

- `e2e/tests/29-prompts-plugins.spec.ts`: the Prompts tests open the picker from Chat instead of the sidebar; add
  "the chip shows the template name", "a second thread has no prompt", "the prompt survives a reload", "`/prompt off`
  clears it".
- `src/features/prompts/__tests__/promptLogic.test.ts` moves with the file. New vitest for
  `systemPromptOverride.ts` (read, write, clear, per thread).
- `scripts/e2e_phases/phase_21_prompts_plugins.py`, `tests/test_prompts.py`, `tests/test_route_contracts.py`:
  unchanged (backend).
- README lines 64, 130 and 282 to 284, `docs/reference/api.md`: reword to "prompt templates in Chat".

Size: about two days with tests. Risk: `useChat.ts` (1166 lines) and `ChatTab.tsx` are over the soft cap, so the
change should add the new file and thread it through, not grow them.

### If the owner prefers the smallest change

Keep the page, and do only step 1 (per-thread prompt, visible chip with a clear button). That removes the one
dishonest behaviour for about half a day's work.

## 2. Plugins: hide now, remove after v0.10.0

### Why

Nothing can be installed, loaded or run (progress file, checks 5 to 9). The page lists what the app already shows
elsewhere (cache strategies in the model launch settings, tools in Chat) with switches that change nothing. The
README and the page promise external plugins. Every real extension need has a working home:

| Need | Where it is done today |
|---|---|
| Give the chat model a new tool | MCP servers (`settings.mcpServers`, `backend_service/mcp/loader.py`, tools appear as `mcp:<server>`) |
| Give the chat model instructions | Skills (import, scan, per-chat switch) |
| Let an outside agent drive the app | Agent bridge (`/mcp`, Settings > Integrations) |
| Add a cache strategy | `cache_compression/` registry (a code change, and it should be) |
| Add a model source or an engine | Code in `backend_service/inference/` |

### Hide (one commit, easy to revert)

- Put `plugins` behind the "Show experimental tabs" switch from the review plan (the switch does not exist yet; the
  integrator builds it once for every stream).
- Replace the "External Plugins" panel text, which tells people to put folders in `~/.chaosengine/plugins/`; nothing
  reads that folder. Until the switch exists, removing that one panel is the cheapest honest fix
  (`PluginsTab.tsx`, lines 163 to 178, and the two `pluginsTab.external*` keys).
- README lines 97, 168 and 286 to 288: remove "plugin system with 5 extension types" and "external plugins are
  discovered from the app's plugin directory".

### Remove (after the release, if no loader is planned)

Files: `src/features/plugins/` (3 files), `backend_service/plugins/__init__.py`, `backend_service/routes/plugins.py`
and its line in `backend_service/routes/__init__.py`, `PluginsView` in `src/app/SystemTabs.tsx`, the `plugins` rows in
`src/constants/tabs.ts`, `src/types/system.ts`, `src/app/tabViews.ts`, locale keys `tabs.plugins`,
`tabCaptions.plugins`, `tabShortLabels.plugins`, `pluginsTab.*`, `panels.pluginSystem`, `panels.pluginsRegistered`,
`panels.externalPlugins`, `panels.installFromDirectory`. Tests: `tests/test_plugins.py`, `PluginsRouteTests` in
`tests/test_route_contracts.py`, `src/features/plugins/__tests__/`, the Plugins tests in
`e2e/tests/29-prompts-plugins.spec.ts`, the plugin checks in `scripts/e2e_phases/phase_21_prompts_plugins.py`.
Docs: `docs/cli/reference.md` (three rows), `docs/architecture/overview.md`, `docs/reference/api.md`, `docs/index.md`.
The CLI has `plugins-list`, `plugins-enable` and `plugins-disable` (`scripts/chaosengine-cli`, lines 1113 to 1123 and
1815 to 1817, and `docs/cli/overview.md`); they go with the routes. The agent bridge (`backend_service/mcp`) does not
call `/api/plugins`. The CLI's `prompts-list`, `prompts-create` and `prompts-delete` stay (the prompt store stays).

### If a real plugin system is wanted later

It needs, before any code is loaded from a folder: a manifest with declared permissions, a scan like the Skills one
(`backend_service/skills/scan.py`), off by default with a confirm for High findings, process isolation (the MLX
worker pattern), a start-up switch that skips all plugins, and uninstall. `tests/test_plugins.py` now has a test
that fails the day a caller of `discover_from_directory` is added, so this cannot arrive by accident.
