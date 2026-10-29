---
title: "Progress: ui-core (shared Models page + Images port)"
status: done
---
# Progress: ui-core

Updated: 2026-10-29 19:05. Owner: ui-core agent (Opus). Worktree `/Users/dan/ChaosEngineAI-wt/ui-core`, branch `feat/models-ui-core` (from `feat/models-fit` at `870c0a1`). Not pushed.

## Goal

Replace Discover + My Models with one "Models" page per area (FU-129). Build the shared parts in `src/components/models/`, port Images first as the reference, and do the routing (new tab ids, one redirect map with presets, sidebar, deep links) so Chat and Video can be ported on top by `ui-chat` and `ui-video`.

## Done

- `233dfc2` shared shell + Images > Models page + routing; old `ImageDiscoverTab.tsx` / `ImageModelsTab.tsx` deleted.
- `29308e5` tests (status, filters, download actions, delete confirm, FitBadge, redirects, sidebar, Images page) + chat Discover prefers the backend `fit` (R14).
- `a2c61a2` row detail built only while expanded.
- Checks at `a2c61a2`: `npx tsc --noEmit` clean, `npx vitest run` 57 files / 675 tests pass, `npm run i18n:validate` OK.

### Files

| File | What |
|---|---|
| `src/components/models/modelEntry.ts` | `ModelEntry<TSource>`, `ModelsArea`, `ModelInstallState`, `ModelStatusBucket`, `ModelAvailabilityFilter` |
| `src/components/models/modelStatus.ts` | `resolveModelStatus(entry, ctx)`, `countBuckets`, `matchesAvailability`, `statusSortRank`, `isVisibleOnHost` |
| `src/components/models/modelFilters.ts` | `buildModelRows` (host rule, search, groups, counts, availability, sort), `sortModelRows`, `ModelSortOption`, `ModelFilterGroup`, `hasActiveModelFilters` |
| `src/components/models/downloadActions.ts` | pure `downloadActionsFor(status, {trackedOnly, canDeleteModel})` |
| `src/components/models/deleteConfirm.ts` | `buildDeleteConfirmMessage`, `confirmAndDelete(entry, onDelete, confirm?)` |
| `src/components/models/fit.ts` | own `Fit` type, `readFit`, `readVariantFit`, `readBestFitVariantId`, `fitSortRank` |
| `src/components/models/FitBadge.tsx` | `FitBadge`, `BestFitRibbon`, `fitLabel`, `fitTooltip` |
| `src/components/models/ModelStatusChip.tsx` | `ModelStatusChip`, `modelStatusLabel`, `modelBucketLabel` |
| `src/components/models/ModelDownloadControls.tsx` | `ModelDownloadControls`, `ModelActionMenu` (overflow) |
| `src/components/models/ModelFilterBar.tsx` | availability segmented control with counts, search, select / chip groups, sort, Clear |
| `src/components/models/ModelCard.tsx` | `ModelCard`, `ModelMemoryCell`, `ModelDetailList`, `defaultModelSubtitle` |
| `src/components/models/AcceleratorPills.tsx` | host-aware pills (`acceleratorsForHost` → `getApplicableAcceleratorsForHost`) |
| `src/components/models/ModelsPage.tsx` | the page shell |
| `src/components/models/adapters/image.ts` | `collectImageVariants`, `imageVariantToEntry`, `buildImageEntries`, `imageEntryMatchesQuery` |
| `src/components/models/LegacyModelsSwitch.tsx` | **TEMPORARY**, see below |
| `src/features/images/ImageModelsPage.tsx` | Images > Models |
| `src/constants/tabRedirects.ts` | the redirect map, `TabRequest`, `modelsPageRequest`, `resolveTabRequest`, `migrateLastChildByGroup` |
| `src/styles.css` | one block at the end, "Models pages (FU-129)" |

## Decisions (as given by the lead, and how they landed)

1. Chat launch: `ModelDownloadControls` takes `launch` (primary, installed rows) and `overflow` (menu). For chat: `launch = {id:"thread"...}`, `overflow = [{id:"chat", label:"Use in this chat"}, {id:"server", ...}]`. Rendered and tested in `ModelsPage.test.tsx`; chat not implemented.
2. Status ladder (`resolveModelStatus`), first match wins: active transfer (downloading, or `ctx.installing` for LongLive) > paused > installed (`availableLocally`, download `completed`, or `installedByRuntime`) > failed > broken > partial (`hasLocalData`) > available. Buckets: installed / downloading (downloading + paused) / attention (failed + partial + broken) / available. `loaded` is a flag on installed rows (chip icon "loaded"). **My call:** an active transfer wins over installed files (old My Models order) because the row needs Pause / Resume; a failed download on an installed model keeps it installed (old Discover order).
3. Delete model files: `confirmAndDelete` asks first; the message names the path and every sibling sharing the download (up to 6, then "and N more"). "Cancel download" / "Delete download" of an in-flight, paused or failed download do not ask (as before, in every area).
4. Host-incompatible: rows with nothing on disk are hidden (`isVisibleOnHost`); installed / partial / paused ones stay with a "Cannot run on this machine" chip, Generate disabled with that tooltip, Delete still available.
5. Subtitle: "provider · family name" (`defaultModelSubtitle`, repeats dropped). Note moved to the expandable detail.
6. Tab ids: `TabId` lost `online-models`, `my-models`, `image-discover`, `video-discover`; gained `chat-models`. One wrapper `setActiveTab` in `App.tsx` (~line 164) runs `resolveTabRequest`; old Discover ids preset "available", old `my-models` presets "installed". Persisted `lastChildByGroup` is migrated in `useSidebarPrefs.readStoredSidebarPrefs` (id only, no preset: a remembered sidebar child is not an intent). Buttons that meant "installed models" now pass an explicit `modelsPageRequest(area, "installed")`, because `image-models` / `video-models` are the merged page ids and no longer mean "installed".
7. Accelerator pills: host-aware on Images (no CUDA pills on a Mac, none on mflux / sd.cpp rows). The old Video tabs still use the old helper until ported.
8. Paused rows: Resume + Delete download (`downloadActionsFor`), tested.
9. Behaviour table coverage for Images: see "Dropped or changed" below; everything else in map sections 1.3 / 1.4 is kept.

### Dropped or changed on purpose (Images)

- The Status select (7 values: installed, not installed, downloading, paused, failed, incomplete) became the four availability buckets. Paused vs downloading and failed vs incomplete can no longer be filtered apart; the Status column sort still orders them (installed, loaded first; then downloading, paused, failed, broken, partial, available).
- Hero button "Installed Models" (Discover) and "Browse Catalog" (Models) are gone: the availability control on the same page does it. Hero text replaced by one intro line; "Open Studio" kept in the panel header.
- Status sort now defaults to ascending (installed first); old Discover defaulted to descending (not-installed first).
- The old Discover subtitle "… / live Hugging Face metadata" is replaced by "{shown} of {total} models · {installed} installed" (total = rows this host can show).
- Shared-storage badge now shows on any row with files (installed, partial, paused, failed), not only installed; it warns about deletion, which applies to all of them.
- Installed rows now also offer Delete model (Image Discover offered none; Image Models did, without a confirm).
- Added: "Loaded in memory" chip for the image the runtime holds (`imageRuntimeStatus.loadedModelRepo`); expandable detail (note, repo, engine, recommended size, licence, fit with reason and basis, shared siblings, path, metadata warning, Open model card).

## Temporary (until ui-chat / ui-video finish)

- `chat-models` and `video-models` still render the OLD tabs (`OnlineModelsTab` / `MyModelsTab`, `VideoDiscoverTab` / `VideoModelsTab`). `src/components/models/LegacyModelsSwitch.tsx` + the switch in the workspace header (`App.tsx`, `header-badges`) pick the view from the area's availability: "available" → old Discover, anything else → old My Models (matches the old sidebar default). Marked `TEMPORARY (FU-129)` in App.tsx (3 places), the component and the CSS block.
- `previewVariant` (R18) uses `activeTab === "chat-models" && legacyModelsView(...) === "installed"`; ui-chat must replace it when the page is ported.
- `OnlineModelsTab` keeps `memoryFitBucket` as a fallback for backends without `fit`; remove with the chat port.
- Old video tabs: their "Browse Catalog" / "Installed Models" buttons now pass `modelsPageRequest("video", ...)` so they switch view; ui-video replaces these files.

## New i18n keys (no locale JSON touched)

Tab registry (`src/constants/tabs.ts`, resolved with `t(labelKey, {defaultValue})`):
- `common:tabs.chatModels = Chat Models`
- `common:tabCaptions.chatModels = Find, download and manage language models`
- `common:tabCaptions.imageModelsPage = Find, download and manage image models`
- `common:tabCaptions.videoModelsPage = Find, download and manage video models`
- `common:tabShortLabels.models = Models`
(`tabs.imageModels` = "Image Models" and `tabs.videoModels` = "Video Models" are reused.)

`library` namespace (same English as `imageDiscover.action.*` / `.status.*` where the text matches, so those translations can be copied):
- `library:models.action.cancel = Cancel download`
- `library:models.action.deleteDownload = Delete download`
- `library:models.action.deleteModel = Delete model`
- `library:models.action.download = Download model`
- `library:models.action.more = More actions`
- `library:models.action.pause = Pause download`
- `library:models.action.resume = Resume download`
- `library:models.action.retry = Retry download`
- `library:models.action.trackedOnly = Tracked only — not yet launchable`
- `library:models.availability.all = All`
- `library:models.availability.attention = Needs attention`
- `library:models.availability.available = Available`
- `library:models.availability.downloading = Downloading`
- `library:models.availability.installed = Installed`
- `library:models.availability.label = Show`
- `library:models.card.collapse = Hide details`
- `library:models.card.expand = Show details`
- `library:models.card.incompatible = Cannot run on this machine`
- `library:models.card.incompatibleTitle = Its engine does not run on this machine. You can still delete its files.`
- `library:models.delete.confirmShared = {count, plural, one {# other model uses} other {# other models use}} the same download and will be removed too:`
- `library:models.delete.confirmSharedMore = and {count} more`
- `library:models.delete.confirmTarget = This removes {target} from disk. It cannot be undone.`
- `library:models.delete.confirmTitle = Delete the files of "{name}"?`
- `library:models.detail.engine = Engine`
- `library:models.detail.fit = Fit on this machine`
- `library:models.detail.license = Licence`
- `library:models.detail.localPath = On disk`
- `library:models.detail.openModelCard = Open model card`
- `library:models.detail.repo = Repository`
- `library:models.detail.resolution = Recommended size`
- `library:models.detail.sharedWith = Shares its download with`
- `library:models.detail.warning = Note`
- `library:models.empty.availability = No {bucket} models under the current filters.` (`{bucket}` is the lower-cased bucket label; translators may prefer a per-bucket sentence)
- `library:models.empty.catalog = No models in the catalog yet.`
- `library:models.empty.filters = No models match the current filters. Try broadening the filters or search terms.`
- `library:models.empty.showAvailable = Show available models`
- `library:models.filter.clear = Clear Filters`
- `library:models.filter.search = Search`
- `library:models.filter.sortBy = Sort by`
- `library:models.fit.bestFit = Best fit`
- `library:models.fit.bestFitTitle = The pick in this family for this machine: the newest release that fits comfortably.`
- `library:models.fit.estimated = estimated`
- `library:models.fit.estimatedNote = Estimated from the model size; not measured on this machine.`
- `library:models.fit.estimatedShort = est.`
- `library:models.fit.fits = Fits`
- `library:models.fit.measured = measured`
- `library:models.fit.measuredNote = Measured on this machine.`
- `library:models.fit.numbers = Needs about {need} GB of {usable} GB usable on this machine.`
- `library:models.fit.tight = Tight`
- `library:models.fit.unknown = Not for this machine`
- `library:models.fit.wontFit = Won't fit`
- `library:models.image.emptyAttention = No image model needs attention.`
- `library:models.image.emptyCatalog = No image models in the catalog yet. Check that the backend is running.`
- `library:models.image.emptyDownloading = No image model is downloading.`
- `library:models.image.emptyInstalled = No image models installed yet. Pick one from the available models to download it.`
- `library:models.image.intro = Every image model in one list. Download one to use it in Image Studio; filter by what is installed, available or downloading.`
- `library:models.legacy.catalog = Browse catalog` (temporary)
- `library:models.legacy.installed = Installed` (temporary)
- `library:models.legacy.label = View` (temporary)
- `library:models.sort.fewestDownloads = fewest downloads first`
- `library:models.sort.leastLiked = least liked first`
- `library:models.status.available = Not installed`
- `library:models.status.broken = Broken`
- `library:models.status.downloading = Downloading`
- `library:models.status.failed = Failed`
- `library:models.status.failedDetail = Download failed`
- `library:models.status.installed = Installed`
- `library:models.status.loaded = Loaded in memory`
- `library:models.status.partial = Incomplete`
- `library:models.status.paused = Paused`
- `library:models.subtitle = {shown} of {total, plural, one {# model} other {# models}} · {installed} installed`
- `library:models.summary.availability = Showing: {bucket}`
- `library:models.summary.count = {count, plural, one {# model} other {# models}} · {sortLabel}`
- `library:models.summary.search = Search: {query}`
- Already called in the old Discover code but never in `en`: `library:imageDiscover.sortLabel.likes = most liked first`, `library:imageDiscover.sortLabel.downloads = most downloads first`.

Reused existing keys (translations already exist): `library:imageDiscover.{filter.*, task.*, access.*, sort.*, column.*, memory.*, trackedOnly.*, action.generate|openHuggingFace|huggingFace|settings, hero.openStudio, unknown}`, `library:imageModels.{sort.*, sharedRepo, sharedRepoTitle}`. Old keys are untouched (R22).

## Not tested

- Windows and Linux: host rules are unit-tested with host descriptions only (Mac arm64 vs Windows x86_64 with 24 GB VRAM).
- A real download, pause, resume or delete in the app: I did not start a download (all image models are GBs). Paused / failed / gated rows are covered by render tests with hand-made download states. Delete was exercised in the app with `window.confirm` stubbed to "No": the message named the shared sibling and no DELETE request was sent.
- FitBadge in the app used a scratchpad proxy (`fit_proxy.py`, not committed) that added sample `fit` / `bestFitVariantId` to `/api/images/catalog`, because my worktree backend has no fit code. The real backend payload was not rendered.
- The chat empty-state links ("Browse Discover" / "Load Model"): the banner did not show on this install; covered by the redirect tests only.
- Screen readers, keyboard-only use of the `<details>` overflow menu, the 1280 px breakpoint of the filter row.

## How to resume

1. `cd /Users/dan/ChaosEngineAI-wt/ui-core && git status && git log --oneline -4`.
2. Tests: `npx tsc --noEmit && npx vitest run src/components/models src/features/images src/constants src/features/models && npm run i18n:validate`.
3. Run it: backend `CHAOSENGINE_ALLOWED_ORIGINS=http://localhost:5177,http://127.0.0.1:5177 CHAOSENGINE_EXTRAS_SITE_PACKAGES=/tmp/ce-no-extras CHAOSENGINE_PORT=8877 /Users/dan/ChaosEngineAI/.venv/bin/python -m backend_service.app`, then `VITE_CHAOSENGINE_API_BASE=http://127.0.0.1:8877 npm run dev -- --port 5177`, open http://127.0.0.1:5177 > Images > Models.
4. Next step for ui-video / ui-chat: write `adapters/video.ts` / `adapters/chat.ts`, a `VideoModelsPage` / `ChatModelsPage` like `ImageModelsPage.tsx`, then delete `LegacyModelsSwitch` and its three `TEMPORARY (FU-129)` spots in `App.tsx`.

## Requests (for the lead)

- Merge with `settings-rec`: `ImageStudioTab.tsx` and `VideoStudioTab.tsx` changed in three small places each (one import line, the `onActiveTabChange` prop type `TabId` → `TabRequest`, four `onActiveTabChange(...)` calls now pass `modelsPageRequest(area, preset)`). `ImageGalleryTab.tsx`, `VideoGalleryTab.tsx` likewise.
- Types: my `Fit` (`src/components/models/fit.ts`) is structurally the same as `ModelFit` in `src/types/models.ts` on `feat/models-fit`; after the merge `fit.ts` can re-export `ModelFit` and the `as Partial<ImageModelVariant>` casts in the tests can go.
- Locale strings that still say Discover / My Models (map 8.2): `chat:composer.placeholderNoModel` ("…from My Models or Discover…", visible in Chat today), `chat:swapMenu.noWarmModels`, `chat:emptyBanner.*`, `common:runHf.downloadStarted`, `library:imageModels.empty.noModels` (now unused), `studio:imageStudio.*` / `videoStudio.*` Discover / Installed button labels and callouts, `studio:*.safety.browseSmaller`, `studio:imageGallery|videoGallery.actions.installedModels`. Also docs / README / `scripts/capture-screenshots.sh` (clicks "Discover" / "My Models"), backend strings in map 8.5.
- Now unused, kept on purpose: `useImageState` outputs `combinedImageDiscoverResults`, `imageDiscoverHasActiveFilters`, `imageDiscoverSearchQuery`, `filteredImageDiscoverFamilies`, `filteredLatestImageDiscoverResults`, `installedImageVariants`; `common:panels.installedImageModels`, `common:tabs.imageDiscover` etc. Prune after Video / Chat land.
- Pre-existing, not mine: `useWorkspace`'s startup elapsed-seconds timer (`src/hooks/useWorkspace.ts` ~151) is never cleared after the workspace loads, so App re-renders every second for the life of the app (twice in dev StrictMode). Every page re-renders with it. Worth a separate fix.
- Possible follow-up: a "Best fit first" sort (needs `"fit"` in `DiscoverSort`, which Video shares; left out).

## Decisions to propose

- Keep the "active transfer wins over installed" order in `resolveModelStatus` (reason above).
- Persisted sidebar ids are migrated without preset; navigations carry presets.
- Status select replaced by four buckets (paused / failed no longer filterable on their own).
