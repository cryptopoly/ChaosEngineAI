---
title: "Progress: ui-chat (Chat > Models page)"
status: done
---
# Progress: ui-chat

Updated: 2026-10-29 19:40. Owner: ui-chat agent (Opus). Worktree `/Users/dan/ChaosEngineAI-wt/ui-chat`, branch `feat/models-ui-chat` (from `feat/models-ui-core` at `23363c9`). Not pushed.

## Goal

Replace Chat Discover (`OnlineModelsTab`) and Chat My Models (`MyModelsTab`) with one `ChatModelsPage` (tab `chat-models`) on the shared `ModelsPage` shell, keeping every behaviour (FU-129, map R1-R27).

## Done

| Commit | What |
|---|---|
| `b0f301b` | Additive shell options (below). **Also carries the deletion of the four old chat files** (they were staged early), so this commit alone does not build; the branch tip does. Not rewritten: history rewrites need the owner's yes. |
| `94220c6` | `src/components/models/adapters/chat.ts`: the join rule, search, acceleration predicate, favourites id, partial-download inference, `findLibraryItemForRepo`; 24 tests |
| `36422e9` | `ChatModelsPage.tsx`, `ChatHubResults.tsx`, `memoryFit.ts`; App wiring; old tabs and their tests removed and ported |
| `5632c80` | A catalog row is installed only through its own files (see rule) |

Checks at `5632c80`: `npx tsc --noEmit` clean, `npx vitest run` 57 files / 719 tests pass, `npm run i18n:validate` OK.

### The join rule (R1), `adapters/chat.ts`

1. Every catalog variant (`workspace.featuredModels`, all of them; host gating is per row) is a row, id `catalog:<variant id>`.
2. A chat library item merges into a catalog row when **both** hold:
   - mutual best fuzzy match (`libraryVariantMatchScore` >= 12): the item's best variant is that variant AND the variant's best item is that item (first-wins ties, as `findLibraryItemForVariant` / `findCatalogVariantForLibraryItem`). Scores are computed once per pair (`prepareChatJoin`, memoised on catalog + library, not on download polling);
   - identity confirmed: the backend's `availableLocally` is true, or the item IS the repo (HF cache folder `models--org--name`, an item named `org/name`, or a folder path ending in `/org/name`, case-insensitive, `\` normalised).
   The fuzzy score alone is too loose to merge on: any GGUF folder scores 14 against any GGUF variant from the format alone.
3. Every other library item is its own row, id `library:<path>`; it keeps its best fuzzy match for display only (capabilities, context, note, release), as old My Models did.
4. A download in progress (downloading / paused / failed) whose repo no row shows yet is a `library:download://<repo>` row (no Reveal, no path, no favourite).

Installed state: from the files the join gave the row (installed when not broken) or a completed download; an active transfer wins (shared ladder). `availableLocally` alone does not make a catalog row installed: it is a name-substring check, and on this Mac it flagged "Qwen3.6 27B" BF16 (from the 4-bit conversion), "Qwen3.8 27B" BF16 and "Gemma 4 26B A4B" (from their DFlash drafter repos). Each download shows on exactly one row: the catalog row of its repo, else the library row that is that repo (never one reached by fuzzy match), else a download row. A broken HF-cache item with a "partial blob"/"incomplete" reason is a paused download with progress from the bytes on disk (old My Models), total from the matched variant's size.

Checked in the app (this Mac): 21 chat library items, 21 rows with files (9 merged, 12 library rows), 113 of 118 catalog rows shown (the 5 hidden are CUDA-only), 125 rows, all ids unique.

Tests (`chatAdapter.test.ts`): merged, library-only, catalog-only, download-only; fuzzy match without identity stays two rows; merge on `availableLocally`; `/org/name` folder; two items wanting one row (only the variant's best merges, the other stays); every item and variant exactly once; broken beats `availableLocally`; partial cache → paused; `availableLocally` alone → available; download on one row only; loaded flag. Mutation checks: dropping the identity check, the mutual-best check or the one-row download rule each fails a test.

### Search (R8)

One input (`useModels.searchInput`). A catalog row matches when every token of the query (old Discover tokeniser) is in its family's text or **its own variant's** text (old Discover matched per family and then showed every variant of the family). A row with files also matches the old My Models substring text (name, path, format, quant, backend, source kind, directory label). The same text drives the debounced Hugging Face search (unchanged, `useModels`); its results show as cards **under** the table. When the query matches no row but Hugging Face has results, the empty state says the results are below.

### Filters (R7) and state (R16)

- Capability: the row's capabilities (variant's, else family's; a library row's from its fuzzy match; an unmatched local model has none, as before). Chips without emoji.
- Format / Backend: the row's display format / backend (the files' when the row has files, else the catalog's). Hidden below two values.
- Acceleration: rows with files use the old My Models rule (`modelSupportsStrategy`: what can run now); rows without use the old Discover rule per variant. MTPLX / TensorFold chips hidden off Apple Silicon (old Discover). DFlash empty tooltip and TurboQuant "memory saver" label kept. Chip counts are over the host-visible rows matching the search.
- Survive tab switches: search, capability, format, acceleration (`useModels`, as before), backend filter and sort + direction (moved into `useModels`), availability (App). Row expansion resets (page-local, as Images).

### Actions

- Installed: primary "Chat in a new thread" (`thread`), menu "Use in this chat" (`chat`) and "Load for server" (`server`), all preselecting `library:<path>` (or `catalog:<id>` without files) and disabled with "Cannot run on this machine" when the host cannot run the row. Convert (R4) on installed catalog rows with `launchMode: "convert"`. Delete model (handler confirms; no second confirm), favourite star (rows with files; persisted to settings, pinned to the top whatever the sort), Reveal (rows with real files).
- Downloading / paused / failed: shared Pause+Cancel / Resume+Delete download / Retry+Delete download, on the download's own repo. Failed shows the error under the row.
- Broken with a known repo: Retry + Delete download + Delete model + Reveal; BROKEN badge and reason. Broken with no repo: Delete model + Reveal only (`canDownload: false`).
- Expanded row: closest catalog entry (library rows), note, repository, released, on-disk path, directory, modified, fit with reason and basis, family readme (2 lines), Details (the App-level family window, R6), Open model card on Hugging Face.
- Hub cards (`ChatHubResults.tsx`, moved from the old Discover): file groups, collapse above 8 weights, same download states; "Chat with model" / "Load for server" now preselect that model (R3): the library item that is the repo, else one whose name or path contains it (backend rule), else the catalog variant with that repo, else no preset.
- Run from Hugging Face and Import panels stay, above the table.
- Fit (R14): backend `fit` with `FitBadge` on catalog rows; `memoryFitBucket` fallback (moved to `src/features/models/memoryFit.ts`, test moved) only without it; best-fit ribbon with a chat tooltip.

### App wiring

- `chat-models` renders `ChatModelsPage`; the chat parts of the TEMPORARY switch are gone (header switch now `video-models` only; `legacyModelsView` still used for video).
- R18: on `chat-models`, unless the availability is "available", the cache preview follows the newest chat library item matching the search (`chatModelsLibraryVariant`, replaces `selectedLibraryPath` + its effect); with "available" it follows the search selection (old Discover).
- R17: empty-state buttons and redirects were already wired by ui-core (`modelsPageRequest`), unchanged.
- Removed from App: library search / sort / cap / format / backend / expansion state, `libraryRows`, synthetic rows and their per-render O(items × variants) match, `localVariantCount`, `libraryTotalSizeGb`, and imports they used. Removed from `useModels`: `expandedFamilyId` / `expandedVariantId` (only the old Discover used them).

## Deliberate changes

- Flat rows per variant instead of family cards; family headline, readme and Details live in the expanded row. The collapsed-family download chip is gone (each row shows its own status).
- Search is per variant (above).
- A catalog row is not installed on `availableLocally` alone (above). The App-level Details window still trusts it, so for those three models it disagrees with the page.
- Library rows retry / show downloads only for their own repo; old My Models used the fuzzy match's repo, which downloaded a different repo into the HF cache without fixing the folder.
- Primary action is a new thread on every installed row (old My Models' primary was "use in this chat").
- Launching is disabled for rows the host cannot run (old My Models allowed it; the launch picker already hid them).
- Hub results sit under the table (old: above curated results while searching).
- Search error callout keeps the error text but drops "Showing the last successful Discover results." (the error already says what is shown).
- Format and quant share a column; params sit under the size.
- Default sort: newest released (as Images). Old My Models defaulted to recently modified; that sort is still in the list.
- Download rows have no format (old: "Downloading", which then appeared as a format chip).
- Broken rows read "Broken" in the status chip (shared label); the BROKEN badge stays.
- Hub card chevron is an SVG instead of ▲/▼ characters.

## Shared changes (lead merges; all additive, Images unchanged)

- `downloadActions.ts`: option `canDownload` (default true); `false` → only Delete model outside the installed state.
- `ModelDownloadControls.tsx`: prop `canDownload`, passed through.
- `modelFilters.ts`: `BuildModelRowsInput.pinned` (rows kept on top, order inside each band follows the sort).
- `ModelsPage.tsx`: `subtitle` may be a function of `{shown, total, installed}` (new exported type `ModelsPageCounts`), `pinned`, `emptyFilters`.
- `index.ts`: exports `ModelsPageCounts`.
- New in the shared dir: `adapters/chat.ts`, `__tests__/chatFixtures.ts`, `__tests__/chatAdapter.test.ts`.
- `src/hooks/useModels.ts` and `src/App.tsx` (see App wiring). Expect conflicts with ui-video in `App.tsx`: the header switch block (I narrowed it to `video-models`; ui-video removes it) and the LegacyModelsSwitch import.

## New i18n keys (no locale JSON touched)

All `library:` namespace:
- `models.chat.intro = Every chat model in one list: the ones on disk, the catalog, and Hugging Face search results. Filter by installed, available or downloading.`
- `models.chat.subtitle = {shown} of {total, plural, one {# model} other {# models}} · {installed} installed · {size} on disk · {dirs, plural, one {# directory} other {# directories}}`
- `models.chat.searchPlaceholder = Search name, provider, capability, path or format (also searches Hugging Face)...`
- `models.chat.newThread = Chat in a new thread`
- `models.chat.useInThisChat = Use in this chat`
- `models.chat.bestFitTitle = The pick in this family for this machine: the newest release that fits comfortably; among sizes released together, the largest that fits.`
- `models.chat.closestMatch = Closest catalog entry: {name}`
- `models.chat.sourceDownload = Download`
- `models.chat.sizeTitle = {onDisk} on disk; the catalog lists {catalog}`
- `models.chat.detail.released = Released`
- `models.chat.detail.directory = Directory`
- `models.chat.detail.modified = Modified`
- `models.chat.filter.capability = Capability`
- `models.chat.filter.format = Format`
- `models.chat.filter.backend = Backend`
- `models.chat.filter.acceleration = Acceleration`
- `models.chat.emptyCatalog = No chat models in the catalog or on disk yet. Check that the backend is running, or add model directories in Settings.`
- `models.chat.emptyDownloading = No chat model is downloading.`
- `models.chat.emptyAttention = No chat model needs attention.`
- `models.chat.emptySearchHub = No model on disk or in the catalog matches "{query}". Results from Hugging Face are below.`
- `models.chat.sort.release = Newest released`
- `models.chat.sort.name = Name`
- `models.chat.sort.format = Format`
- `models.chat.sort.formatDir = format {direction}`
- `models.chat.sort.backend = Backend`
- `models.chat.sort.backendDir = backend {direction}`
- `models.chat.sort.size = Largest size`
- `models.chat.sort.ram = Highest RAM`
- `models.chat.sort.ramDesc = highest RAM first`
- `models.chat.sort.ramAsc = lowest RAM first`
- `models.chat.sort.compressed = Highest compressed RAM`
- `models.chat.sort.compressedDesc = highest compressed RAM first`
- `models.chat.sort.compressedAsc = lowest compressed RAM first`
- `models.chat.sort.context = Longest context`
- `models.chat.sort.contextDesc = longest context first`
- `models.chat.sort.contextAsc = shortest context first`
- `models.chat.sort.modified = Recently modified`
- `models.chat.sort.modifiedDesc = recently modified first`
- `models.chat.sort.modifiedAsc = least recently modified first`
- `models.chat.sort.status = Status`

Reused (translations exist): `library:onlineModels.*` (capability, filter, accelFilter, memory, action, hub, status), `library:myModels.*` (column, tooltip, badge, download, action, filter.allBackends, strategy.dflashEmptyTooltip, empty, status.incompleteReason), `library:imageModels.sort.*` summaries, ui-core's `library:models.*`, `common:tabs.chatModels`. Now unused but kept (R22; checked by grepping `src`): `library:onlineModels.subtitle`, `.searchPlaceholder`, `.variantCount`, `.section.curatedAria`, `.status.countInstalled`, `.status.notInstalled`, `.column.*`, `.searchError.fallback`, `.empty.*`; `library:myModels.subtitle`, `.searchPlaceholder`, `.filter.all` / `.allFormats` / `.allStrategies`, `.capability.*`, `.strategy.chip` / `.compatTooltip` / `.memorySaverLabel` / `.memorySaverTooltip`, `.column.quant`, `.status.*` (except `incompleteReason`), `.download.failed` / `.failedShort`, `.action.*` (except `starModel` / `unstarModel` / `deleteModel`); `common:tabs.myModels`, `common:panels.discoverModels`.

## Not tested

- Windows and Linux: host rules unit-tested with host descriptions only (Mac arm64, Windows x86_64 24 GB VRAM).
- A real download, pause, resume, retry or delete: not clicked (multi-GB models). Delete, Reveal and Convert were not clicked in the app. Covered by render tests.
- The backend `fit` payload: this worktree's backend has no fit code, so the app showed the fallback badge; the backend verdict path is covered by a render test.
- The chat empty-state buttons ("Browse Discover" / "Load Model"): they appear only in an empty thread, and making one writes a session. Redirect tests cover them.
- Keyboard-only use, screen readers, phone widths, very large libraries (render with 125 rows: 25-155 ms per filter change in dev mode).
- Dev console: every render logs i18next `missingKey` for the new `library:models.*` keys (ui-core's and mine) until they are in `en`; with App re-rendering every second (ui-core's note on `useWorkspace`) this was about 200 000 lines in a session.

## How to resume

1. `cd /Users/dan/ChaosEngineAI-wt/ui-chat && git status && git log --oneline -5`.
2. `npx tsc --noEmit && npx vitest run src/components/models src/features/models src/constants && npm run i18n:validate`.
3. Run it: `CHAOSENGINE_ALLOWED_ORIGINS=http://localhost:5179,http://127.0.0.1:5179 CHAOSENGINE_EXTRAS_SITE_PACKAGES=/tmp/ce-no-extras CHAOSENGINE_PORT=8879 /Users/dan/ChaosEngineAI/.venv/bin/python -m backend_service.app`, then `VITE_CHAOSENGINE_API_BASE=http://127.0.0.1:8879 npm run dev -- --port 5179`, open Chat > Models.
4. Next: lead merges with ui-video (App.tsx header block, LegacyModelsSwitch removal once both areas are ported).

## Requests (for the lead)

- Locale strings that still name the old tabs and now show on this page: `library:myModels.strategy.dflashEmptyTooltip` ("Download a base model from Discover"), `common:runHf.downloadStarted` ("track progress in My Models").
- The App-level Details window (`App.tsx`, `detailFamilyId`) still treats `availableLocally` as installed; consider giving it the page's rule.
- `COVERAGE_BASELINE.md:90` lists the deleted `MyModelsTab.tsx`; `src/components/CapabilityStrip.tsx:16` comment names both deleted tabs.
- Backend: `_variant_available_locally` (`helpers/model_family_payload.py`) and the hub `available_locally` check match drafter repos and other quants by substring (e.g. hub `RecursiveSI/flint-int4` shows "Installed" because `RecursiveSI/flint-int4-gguf` is on disk). Worth tightening at the source.

## Decisions to propose

- A chat catalog row is installed only through its own files; `availableLocally` only confirms a merge (evidence above).
- Library rows never take a download or a retry through a fuzzy match.
- Hub results under the table; Run from Hugging Face and Import above it.
