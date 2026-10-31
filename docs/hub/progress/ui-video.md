---
title: "Progress: ui-video (Video > Models page)"
status: done
---
# Progress: ui-video

Updated: 2026-10-29 19:35. Owner: ui-video agent (Opus). Worktree `/Users/dan/ChaosEngineAI-wt/ui-video`, branch `feat/models-ui-video` (from `feat/models-ui-core` at `23363c9`). Not pushed.

## Goal

Replace Video Discover + Video Models with one `VideoModelsPage` on the shared Models shell (FU-129), keeping every behaviour of both old tabs (map sections 1.5 / 1.6, R9-R13, R21, R25, R26).

## Done

- `a49f325` `src/components/models/adapters/video.ts` (adapter, R12 handlers, delete-confirm entry), `src/features/video/VideoModelsPage.tsx`, App.tsx wiring, old `VideoDiscoverTab.tsx` / `VideoModelsTab.tsx` deleted, `videoDeleteRepoKind` in `src/utils/videos.ts`, tests.
- `d9569ee` shared shell fix: an empty Installed (or other bucket) view says the filters hide the rows when a search or filter group is active, instead of "No video models installed yet".
- Checks at `d9569ee`: `npx tsc --noEmit` clean; `npx vitest run` 59 files / 715 tests pass; `npm run i18n:validate` OK.

### Files

| File | What |
|---|---|
| `src/components/models/adapters/video.ts` | `VideoModelEntry` (adds `longLive`), `collectVideoVariants` (tracked de-dup, R26), `videoVariantToEntry`, `buildVideoEntries`, `videoEntryMatchesQuery`, `videoSizeSortKey` (GGUF file first, R13), `isLongLiveRepo`, `hasVisibleLongLiveRow` (R25), `videoRepoDependencies`, `buildVideoRepoUsers`, `videoSharedForDelete`, `videoDeleteConfirmEntry`, `videoRowActions` (R12 handlers) |
| `src/features/video/VideoModelsPage.tsx` | the page; `videoSortOptions`, `videoDeleteLabel` |
| `src/utils/videos.ts` | `VideoDeleteRepoKind`, `videoDeleteRepoKind`; `videoDeleteLabelForRepo` now uses it (same output) |
| `src/App.tsx` | one `video-models` branch; header `LegacyModelsSwitch` now chat only; imports |
| `src/components/models/ModelsPage.tsx` | empty-state message (see Shared changes) |
| tests | `src/components/models/__tests__/videoAdapter.test.ts` (18), `src/features/video/__tests__/VideoModelsPage.test.tsx` (22), fixtures in `src/components/models/__tests__/videoFixtures.ts` (separate file so it does not collide with ui-chat edits to `fixtures.ts`) |

### Behaviour kept (map 1.5 / 1.6)

- R12 keys: `downloadKey` = variant id; status read from `activeVideoDownloads[variant.id]` only (a repo-keyed status does not light up every row of that repo); Download / Resume / Retry call `onVideoDownload(variant.repo, variant.id)`; Pause calls `onCancelVideoDownload(download?.repo ?? variant.repo)`; Cancel / Delete download / Delete model use `videoDeleteRepoForVariant(variant, download)`. Tested with a mutation check (switching `downloadKey` to the repo or Pause to the variant repo fails two tests).
- Delete labels: "Delete shared GGUF download" / "Delete shared text encoder download" / "Delete shared component download", else "Delete model" or "Delete download".
- Size column: on-disk size first (GGUF file for pinned quants, FU-054), tooltip with the full-repo size; the catalog estimate is now also in the expandable detail ("Catalog size"). The old tabs did not show the catalog estimate inline either.
- "shares storage · +N" badge (FU-054, backend `sharedRepoSiblings`).
- "Loaded in memory" chip; loaded sorts first under Status.
- LongLive: Install button (disabled while installing, label "Installing"), "Installing" status chip, install note, `InstallLogPanel`, Generate once `longLiveStatus.realGenerationAvailable`; no Download / Delete; status never taken from the download map or the catalog flags; CUDA-gated (hidden on a Mac). `onRefreshLongLiveStatus` runs when a LongLive row can show on this host (R25).
- `localStatusReason` callout on rows that are not installed and not downloading; failed-download error callout.
- Task filter, search (variant + family text, as old Discover), all nine sorts with header click, Clear Filters, summary chips, release / downloads / likes labels, Reveal, model card link, Open Studio, Generate opens the Studio on that variant, platform gating (rows the host cannot run are hidden unless something is on disk; then "Cannot run on this machine", Generate disabled, Delete kept), tracked rows de-duplicated and `trackedOnly` honoured if a backend ever sends it (video `latest` is always empty today).
- Paused rows: Resume + Delete download (R10 fixed, the old Models tab showed Download).

### Engine banners (mission item)

The old Video Discover tab had no engine banner or install button besides LongLive. `WanInstallPanel` was removed from it on 2026-05-04 (`e228e41`, the component file is gone). `Ltx2EngineBanner`, the mlx-video install and `WanRuntimeInstaller` live in Video Studio (`VideoStudioTab.tsx`, `VideoStudioRuntimeBanner.tsx`) and are untouched. Nothing to move.

## Deliberate changes (follow ui-core unless noted)

1. The 7-value Status select (incl. "In memory") is replaced by the four availability buckets. Paused vs downloading and failed vs incomplete can no longer be filtered apart; the Status sort still orders them, loaded first.
2. Hero "Installed Models" (Discover) and "Browse Catalog" (Models) buttons are gone; "Open Studio" sits in the panel header; one intro line.
3. Subtitle is "provider · family" (old Discover showed the note, old Models the family name). The note, repo, GGUF file, text encoder, engine, recommended size, catalog size, licence, fit, shared siblings and path are in the expandable detail. The LTX-2.5 gated note is there.
4. Shared-storage badge shows on any row with files (was installed only) and is translated (was hardcoded English).
5. Delete model asks first (the old video tabs did not). **Video-specific:** the message names every catalog row that needs the deleted repo, not only the backend's `sharedRepoSiblings` group: deleting the base repo of a GGUF quant breaks the base row, the other quants and CausVid, and the message lists all of them. The badge keeps the backend's count, so badge and message can differ (badge "+1", message "4 other models"). The snapshot path is shown only when the target is that snapshot's repo; otherwise the repo id.
6. A failed download offers Retry + "Delete download" (old Discover said "Delete model"); a gated-access failure is explained with Hugging Face and Settings buttons, as on Images.
7. Accelerator pills are host-aware (no CUDA pills on a Mac, none on MLX rows).
8. "Loaded" row: when the Studio's selected variant is the one whose repo is loaded, that row; otherwise the first row with that repo (old behaviour). The runtime reports only a repo, which several GGUF rows share.
9. LongLive status is probed once per page visit and only when its row can show on this host. The old effect ran on every App re-render (a new callback each time), also on a Mac where the row is hidden.
10. LongLive's `InstallLogPanel` moved from the actions cell to the full-width area under the row, with the install note.
11. Licence shown under the release date (Images had it, Video did not).
12. The model-card button keeps the video icon and label ("Open model card"); Images uses the Hugging Face icon.

## Shared changes (for the merge with ui-chat)

- `src/components/models/ModelsPage.tsx` (empty state, ~6 lines): `emptyByAvailability[bucket]` is used only when no search or filter group is active; otherwise the generic "No {bucket} models under the current filters." Found in the app: Video Gallery > Installed Models with a leftover Image-to-video task filter said "No video models installed yet" while two were installed. Images tests pass unchanged; the new behaviour is pinned in `VideoModelsPage.test.tsx`.
- `src/utils/videos.ts`: new `videoDeleteRepoKind` (additive; `videoDeleteLabelForRepo` output unchanged, its tests pass).
- `src/App.tsx`: the header block (`header-badges`) now shows `LegacyModelsSwitch` only for `chat-models` (ui-chat removes it entirely); the two video branches became one; imports `VideoDiscoverTab` / `VideoModelsTab` → `VideoModelsPage`. Expect a textual conflict with ui-chat in the header block and the import list.
- Not touched: `fixtures.ts`, `index.ts`, `modelEntry.ts` (its comment still says "adapters/video.ts ... next"), `LegacyModelsSwitch.tsx` (its doc comment still mentions video; delete the file with the chat port).

## New i18n keys (no locale JSON touched)

`library` namespace:
- `library:models.video.intro = Every video model in one list. Download one to use it in Video Studio; filter by what is installed, available or downloading.`
- `library:models.video.emptyCatalog = No video models in the catalog yet. Check that the backend is running.`
- `library:models.video.emptyInstalled = No video models installed yet. Pick one from the available models to download it.`
- `library:models.video.emptyDownloading = No video model is downloading.`
- `library:models.video.emptyAttention = No video model needs attention.`
- `library:models.detail.ggufFile = GGUF file`
- `library:models.detail.textEncoder = Text encoder`
- `library:models.detail.catalogSize = Catalog size`
- `library:videoModels.sharedRepo = shares storage · +{count}` (same English as `imageModels.sharedRepo`)
- `library:videoModels.sharedRepoTitle = Uses the same download as {count} other variant(s) of {repo}. Deleting this row removes those files too.` (same as `imageModels.sharedRepoTitle`)
- `library:videoModels.action.deleteSharedGguf = Delete shared GGUF download`
- `library:videoModels.action.deleteSharedTextEncoder = Delete shared text encoder download`
- `library:videoModels.action.deleteSharedComponent = Delete shared component download`
- Called before and still not in `en` (inline default only): `library:videoDiscover.status.installing = Installing`, `library:videoDiscover.sortLabel.likes = most liked first`, `library:videoDiscover.sortLabel.downloads = most downloads first`.

Reused: ui-core's `library:models.*` keys (actions, availability, card, detail, fit, status, sort.leastLiked / fewestDownloads, delete.*); existing `library:videoDiscover.{sort.*, column.*, task.*, filter.task, filter.searchPlaceholder, memory.*, action.generate|install|installing|openModelCard, hero.openStudio, longLive.installNote, unknown}`, `library:videoModels.sort.*`; cross-area with identical text: `library:imageDiscover.{access.gated, access.open, trackedOnly.badge, trackedOnly.tooltip, action.huggingFace, action.settings}`; `common:tabs.videoModels`. Old keys untouched (R22).

Now unused, kept on purpose: `library:videoDiscover.{subtitle, hero.heading, hero.body, hero.installedModels, filter.status, filter.clear, status.*, summary.*, empty.body}`, most of `library:videoModels.*` (subtitle, empty, action.browseCatalog / clearFilters, filter.*, summary.*, column.*, status.*), `common:tabs.videoDiscover`, `common:panels.installedVideoModels`.

## Not tested

- Windows and Linux. LongLive is unit- and render-tested only: the backend catalog has `_LONGLIVE_ENABLED = False`, so no LongLive row renders anywhere today, and its CUDA path cannot run on this Mac.
- The R25 effect itself (vitest runs in a node environment, effects do not run); `hasVisibleLongLiveRow` is tested.
- A real download, pause, resume, cancel or delete. In the app the confirm was stubbed to No (no delete request was sent), and the shared-sibling, paused and partial rows came from a scratch proxy (`video_proxy.py` in the session scratchpad, not committed) that edited `/api/video/catalog` and `/api/video/download/status` and refused every download / delete write.
- The not-installed LTX-2.5 row: on this Mac LTX-2.5 is installed, so it showed Generate; the not-installed state (Download, gated note in the detail) is covered by a render test. Its expanded detail with the gated note was checked in the app.
- Fit badges against a real backend payload (this worktree's backend has no fit code); render-tested.
- Keyboard-only use, screen readers, narrow widths.

## How to resume

1. `cd /Users/dan/ChaosEngineAI-wt/ui-video && git status && git log --oneline -4`.
2. Tests: `npx tsc --noEmit && npx vitest run src/components/models src/features/video src/features/images src/utils && npm run i18n:validate`.
3. Run it: backend `CHAOSENGINE_ALLOWED_ORIGINS=http://localhost:5178,http://127.0.0.1:5178 CHAOSENGINE_EXTRAS_SITE_PACKAGES=/tmp/ce-no-extras CHAOSENGINE_PORT=8878 /Users/dan/ChaosEngineAI/.venv/bin/python -m backend_service.app`, then `VITE_CHAOSENGINE_API_BASE=http://127.0.0.1:8878 npm run dev -- --port 5178`, open http://127.0.0.1:5178 > Video > Models.
4. Next: the lead merges with ui-chat (conflicts expected in `App.tsx` header and imports), then removes `LegacyModelsSwitch` once Chat is ported.

## Decisions to propose

- Delete confirm on video names every row that needs the deleted repo (wider than the FU-054 badge count). Alternative: make the backend's `sharedRepoSiblings` dependency-based so the badge and the confirm agree.
- Prune the now-unused `useVideoState` outputs (`combinedVideoDiscoverResults`, `videoDiscoverHasActiveFilters`, `videoDiscoverSearchQuery`, `filteredVideoDiscoverFamilies`, `filteredLatestVideoDiscoverResults`, `installedVideoVariants`) and the dead `src/features/video/VideoPlaceholderTab.tsx` after the merge.
- Hook-level filters (search, task, sort) persist across tab switches, so a Studio "Installed Models" click can land on an Installed view narrowed by an old task filter. The empty state now says so; resetting area filters on a preset navigation is an option.
