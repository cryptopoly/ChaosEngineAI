---
title: "Progress: map of the six model pages"
status: done
---
# Progress: map-models-pages

Updated: 2026-10-29. Owner: Explore agent (read-only research). Branch: `feat/models-fit`. Nothing was edited except this file's content (written by the lead from the agent's reply). I read code only; I did not run the app, vitest, tsc or any backend, and I did not test Windows or Linux.

## Goal

A precise map so `ui-core`, `ui-chat` and `ui-video` can merge each area's Discover tab and My Models tab into one "Models" page without losing behaviour or breaking deep links.

## How to resume

1. `cd /Users/dan/ChaosEngineAI && git status` (branch `feat/models-fit`).
2. Read sections 1 to 4 of this file, then `docs/plans/models-fit.md`.
3. Next step for `ui-core`: decide the "Decisions to propose" at the bottom, then build the shared components in section 6.

---

## 1. The six tabs

Abbreviations: DL = download, HF = Hugging Face, "dead" = destructured but never used in the render.

### 1.1 Chat Discover: `src/features/models/OnlineModelsTab.tsx` (819 lines)

**Props (28-70):** `searchResults` (families, already host-filtered in App by `filterFamiliesForHost`, `App.tsx:1267`), `searchInput`/`onSearchInputChange`, `searchError`, `localVariantCount`, `discoverCapFilter`/`discoverFormatFilter`/`discoverAccelFilter?` plus change handlers, `accelCompat?` `{dflashModels, mtplxModels, tensorfoldModels?, turboInstalled, appleSilicon?}`, `expandedFamilyId`/`expandedVariantId` plus change handlers, `onDetailFamilyIdChange`, `library` (= `chatLibrary`), `activeDownloads` (keyed by repo), `onDownloadModel`, `onCancelModelDownload` (this is PAUSE), `onDeleteModelDownload` (this is CANCEL or DELETE-partial), `onPrepareCatalogConversion`, `onOpenModelSelector(action, preselectedKey?)`, `onOpenExternalUrl`, `hubResults`, `expandedHubId`, `onToggleHubExpand`, `hubFileCache`/`hubFileLoading`/`hubFileError`, `availableMemoryGb?`.

**Hooks and state:** only `useTranslation("library")` (143). No local state. All state lives in `useModels` (`src/hooks/useModels.ts:28-60`, destructured at `App.tsx:259-281`): `searchInput`, `searchResults`, `hubResults`, `searchError`, `expandedHubId`, hub file caches, `detailFamilyId`, `expandedFamilyId`, `expandedVariantId`, `activeDownloads`, `discoverCapFilter`, `discoverFormatFilter`, `discoverAccelFilter`. Search: curated families filtered locally by `modelFamilyMatchesDiscoverQuery` (`useModels.ts:62-71`); a 120 ms debounced `searchHubModels(query)` network call (`useModels.ts:87-124`). DL polling every 2 s while any DL is `downloading` (`useModels.ts:~160`).

**Exports:** `memoryFitBucket` (85-107), imported by `__tests__/memoryFitBucket.test.ts`.

**Derived data:** `filteredResults` = cap filter (family or variant caps), format filter, accel filter (`familyMatchesAccel` 235-261: DFlash/MTPLX fuzzy `candidateKeys` match on any variant repo; TensorFold exact repo via `isTensorfoldRepo`; TurboQuant = family has a GGUF variant), then sorted by newest variant release (263-278). `filteredHubResults` = format filter only, sorted by `lastModified`, downloads, likes (279-290). With a non-empty query the Hub section renders before curated (795-797).

**User actions:**
- Search (738); capability chips (145-181); format chips, hidden if fewer than 2 formats (183-213); acceleration chips with per-chip counts, Apple-only chips hidden when `appleSilicon === false`, TurboQuant labelled "memory saver" (746-784).
- Expand or collapse family card (316-325); expand variant row (399-404); "Details" button opens the App-level modal (349-359).
- Variant actions (447-476): installed gives Convert (only if `launchMode==="convert"`), Chat (`onOpenModelSelector("thread", library:path or catalog:variantId)`), Load for server (`"server"`). Downloading gives Pause + Cancel. Paused gives Resume + Delete download. Failed gives Retry + Delete download. Completed gives nothing. Otherwise Download.
- Expanded variant (483-496): note, release label, local path, "Open model card on HF".
- Hub card (552-717): expand, lazy file list fetch, license/pipeline/size/tags, weights/tokenizer/other groups (collapse if more than 8 weights), same 5 DL states, "Chat with model" and "Load for server" with NO preselected key (689-690), "Open on HF" (712).
- Embedded panels: `<RunFromHuggingFace />` (785), `<ImportModelsPanel />` (786).

**Data shown:**
- Family header: name, provider, param range, formats, "N installed" chip, first active DL chip (collapsed progress), headline, `CapabilityStrip(8)`, variant count, `updatedLabel`.
- Variant row columns: name + caps(4), `format / quantization`, backend, params, size, RAM estimate + fit badge, compressed RAM, context, status icon, actions.
- Family `readme` (first 2 lines).
- Hub card: name, provider, format badge (accent if GGUF), status, release label, `updatedLabel`, `downloadsLabel`, `likesLabel`.

**Status logic (388-396):** installed if `variant.availableLocally` or DL completed; else downloading, paused, failed, or "Not installed".

**Shared helpers:** `Panel`, `RunFromHuggingFace`, `ImportModelsPanel`, `IconActionButton`, `StatusIcon`, `CapabilityStrip`, `candidateKeys`, `isMemorySaverStrategy` (`runtimeSupport`), `isTensorfoldRepo` (`tensorfoldSupport`), `CAPABILITY_META`, utils `number`, `sizeLabel`, `findLibraryItemForVariant`, `downloadProgressLabel`, `formatReleaseLabel`, `handleActionKeyDown`.

**i18n:** `library:onlineModels.*` (84 keys: subtitle, searchPlaceholder, variantCount, filter, capability.title/short, status, column, memory, action, hub, searchError, empty, accelFilter, section), `common:panels.discoverModels` (727), `library:download.*` via `downloadProgressLabel`.

**Tests:** `OnlineModelsTab.test.tsx` (7 tests at 175, 189, 204, 231, 275, 325, 390). They pin: "Preparing download...", "Downloading 42%", "Downloading 17%", `aria-label="Retry download"` and `"Delete download"`, "HuggingFace Hub", hub ordering by date, "sorted by most recent update", `<details class="hub-file-group hub-file-group--collapsible">`, "Weights (12)", "Largest shard 9.5 GB", absence of `class="discover-card expanded"`. `memoryFitBucket.test.ts` (7 tests).

### 1.2 Chat My Models: `src/features/models/MyModelsTab.tsx` (603 lines)

**Props (47-80):** `filteredLibraryRows: LibraryRow[]` (`{item, matchedVariant, displayFormat, displayQuantization, displayBackend, sourceKind, estimatedRamGb, estimatedCompressedGb}`, 26-35), `libraryTotalSizeGb`, `enabledDirectoryCount`, `librarySearchInput`, cap/format/backend filter pairs, `strategyCompat?`, `activeDownloads`, `expandedLibraryPath`, `fileRevealLabel`, DL handlers (download/cancel=pause/delete), `onPrepareLibraryConversion` (**dead**, 67/115), `onOpenModelSelector`, `onRevealPath`, `onDeleteModel(item)`, `librarySortKey`/`librarySortDir` plus setters, `favoriteModelRefs?`, `onToggleFavoriteModel?`.

**Hooks and state:** `useTranslation("library")`, local `useState strategyFilter` (252; resets on tab switch). Library state lives in `App.tsx:291-299` (search, sort key/dir, cap/format/backend filter, `expandedLibraryPath`). Row building is in `App.tsx:319-393`: `libraryRows` (319-331, with `findCatalogVariantForLibraryItem` fuzzy match), `syntheticDownloadRows` (339-375, `download://<repo>` sentinel paths for in-flight DLs not yet on disk), sort switch (381-393), search haystack (376-380: name, path, format, quant, backend, source kind, directory label).

**Local logic:**
- Favourite ref `rowFavoriteRef` (82-93); starred rows lifted to the top (330-344).
- `inferredPartialDownload` (157-179): a "broken" HF-cache row with "partial blob"/"incomplete" reason is shown as a PAUSED DL with computed progress.
- `modelSupportsStrategy` (255-304).
- `libraryDownloadDetail` (142-155).

**User actions:**
- Search (360-366); cap chips (182-218), format chips and backend chips (220-250, hidden if fewer than 2), strategy chips (372-425: all, DFlash, MTPLX, TensorFold, TurboQuant with counts, DFlash-empty tooltip).
- Sort by clicking headers: name, format, backend, size, RAM, compressed, context (428-439; default direction asc for name, else desc). The "modified" sort exists in state but has no header.
- Expand row (484-489).
- Per-row actions (527-573): star/unstar; DL overlay gives Pause or Resume/Retry + Cancel/Delete download; broken-with-repo gives Retry + Delete download; otherwise (not broken) Chat (`"chat"`, `library:path`) + Load for server (`"server"`); Reveal (hidden for `download://` rows); Delete model (hidden during a DL overlay; confirm in `modelActions.ts:57`).

**Data shown:**
- Name; source kind badge; "Active download" badge with size tooltip; `CapabilityStrip(5)` from `matchedVariant`; DL detail text (500-505); BROKEN badge + reason (506-511).
- Columns: format, quant, backend, size, RAM (tooltip), compressed (tooltip), context, status icon.
- Expanded: path, variant note, release label (576-588).
- Subtitle: "N models / X on disk / D directories".

**Status (467-475):** downloading, paused, failed, incomplete (broken without DL), else installed.

**Shared helpers:** `Panel`, `IconActionButton`, `StatusIcon`, `CapabilityStrip`, `candidateKeys`, `isMemorySaverStrategy`, `isTensorfoldRepo`, `CAPABILITY_META`, utils `number`, `sizeLabel`, `parseContextK`, `compareOptionalNumber` (unused here, used in App), `inferHfRepoFromLocalPath`, `downloadProgressLabel`, `downloadSizeTooltip`, `formatReleaseLabel`.

**i18n:** `library:myModels.*` (64 keys: subtitle, searchPlaceholder, empty, filter, capability, strategy, column, status, badge, tooltip, download, action), `common:tabs.myModels` as panel title (351).

**Tests:** `MyModelsTab.test.tsx`, 5 tests at 102, 126, 157, 179, 224. They pin: "Paused 36%", "23.9 / 67.0 GB downloaded.", aria-labels "Resume/Pause/Cancel/Retry/Delete download", "BROKEN" present or absent, no "CHAT"/"SERVER" on broken rows, synthetic row hides "Reveal" and the `download://` path, GGUF dir maps to matched repo ("Downloading 1%", "2.5 / 494.1 GB downloaded.").

### 1.3 Image Discover: `src/features/images/ImageDiscoverTab.tsx` (793 lines)

**Props (42-71):** `combinedImageDiscoverResults`, search input, task filter, access filter, sort (value plus setter each), `imageDiscoverHasActiveFilters`, `imageDiscoverSearchQuery`, `activeImageDownloads` (keyed by repo), `selectedImageVariant`, `fileRevealLabel`, `nativeBackends?`, `hostSystem?`, `onActiveTabChange`, `onOpenImageStudio(modelId?)`, `onImageDownload`/`onCancelImageDownload`(pause)/`onDeleteImageDownload`, `onOpenExternalUrl`, `onRevealPath`.

**Hooks and state:** `useTranslation("common")` as `t`, `useTranslation("library")` as `tLib`; local `statusFilter` (242) and `sortDir` (243); `useMemo filteredResults` (244-297): maps variants to `{variant, status, memoryEstimate}`, host-gates by `imageOrVideoVariantPlatformGate`+`isVariantCompatibleWithHost`, status-filters, sorts again. Hook state in `useImageState` (`src/hooks/useImageState.ts:79-263`): search, task, access, sort (persist across tab switches); `combinedImageDiscoverResults` = every curated variant as its own row plus tracked "latest" results not already curated (de-duped by repo, 241-258), sorted by `compareDiscoverVariants`.

**Actions:**
- Hero buttons: "Installed Models" goes to `image-models` (340), "Open Studio" calls `onOpenImageStudio(selected?.id)` (343).
- Search, task select, access select (all/open/gated), status select (any/installed/not-installed/downloading/paused/failed/incomplete), sort select (9 keys), Clear Filters (472-486).
- Header click sort (550-578).
- Row actions (674-768): tracked-only gives a disabled download button; installed gives Generate; downloading gives Pause + Cancel; paused gives Resume + Delete download; failed gives Retry + Delete download; else Download (or Resume if partial) and Delete model if `hasLocalData`; Reveal if `localPath`; "Open on HF".
- Failed DL callout with a friendly gated-access message, plus "Hugging Face" and "Settings" buttons when the error is a gated-access error (770-784).

**Data shown:** name, `note`, `styleTags(4)`, Open/Gated badge, "Watching upstream" badge (`trackedOnly`), accelerator pills, provider, task chips, size (compact label, title shows secondary size), RAM/VRAM estimate (`imageDiscoverMemoryEstimate`), released label + `downloadsLabel` + `likesLabel` + license, status icon.

**Status (145-155):** installed (`availableLocally` or completed) is checked first, then downloading, paused, failed, `hasLocalData` gives "incomplete", else "not-installed".

**Shared helpers:** `Panel`, `IconActionButton`, `StatusIcon`, `AcceleratorCard` (pill) + `getApplicableAccelerators`/`getAccelerator`, utils `compactModelSizeLabel`, `compactReleaseLabel`, `downloadProgressLabel`, `downloadSizeTooltip`, `formatImageAccessError`, `formatImageLicenseLabel`, `formatReleaseLabel`, `imageDiscoverMemoryEstimate`, `imagePrimarySizeLabel`, `imageSecondarySizeLabel`, `isGatedImageAccessError`, `imageOrVideoVariantPlatformGate`, `isVariantCompatibleWithHost`. Local copies of `sortIndicator`, `defaultSortDir`, `releaseSortKey`, `sizeSortKey`, `compareNullableNumber*`, `statusSortKey`, `memoryParts` (duplicated in all 4 media tabs).

**i18n:** `library:imageDiscover.*` (67 keys), `common:tabs.imageDiscover`/`tabs.imageModels` (empty-state panel title), `common:panels.noModelsMatchFilters`. `common:imageDiscover.*` (16 keys, `common.json:982`) looks unused by this tab. **Tests:** none for the tab; helpers covered in `utils/__tests__/images.test.ts` (`imageDiscoverMemoryEstimate`, `compareDiscoverVariants`).

### 1.4 Image Models: `src/features/images/ImageModelsTab.tsx` (483 lines)

**Props (34-55):** `installedImageVariants` (catalog + tracked, `availableLocally || hasLocalData`, `useImageState.ts:196-203`), `imageCatalog` (not the "with latest" version), `activeImageDownloads`, `fileRevealLabel`, `nativeBackends?`, `hostSystem?`, `onActiveTabChange`, `onOpenImageStudio`, DL handlers, `onOpenExternalUrl`, `onRevealPath`.

**State:** all LOCAL: search, task, status, sort, sortDir (163-167; reset on tab switch). `useMemo rows` (181-245): family lookup, host gate, task/status/search filter (haystack: name, provider, repo, runtime, family name, recommended resolution, style tags, tasks), sort.

**Actions:** Browse Catalog goes to `image-discover` (260); Clear Filters; header sort; row: Generate (installed), Pause (downloading), else Download/Retry/Resume icon (**paused rows get the Download icon and label**, 450-454), Cancel (downloading) or Delete model (installed, paused, failed, partial) (456-460), Reveal, "Open model card" (464).

**Data shown:** name; family name or provider (391); style tags; accelerator pills; provider; tasks; size + "shares storage +N" badge when `sharedRepoSiblings>0` and installed (420-437, tooltip warns deleting removes siblings' files); RAM/VRAM; released; status; failed DL callout (467-471). Not shown here but shown in Discover: note, gated badge, license, likes, downloads, tracked-only.

**Status (96-102):** downloading is checked BEFORE installed; anything else not installed becomes "incomplete". No "loaded" status for images.

**i18n:** `library:imageModels.*` (71 keys), `common:panels.installedImageModels` (250). **Tests:** none.

### 1.5 Video Discover: `src/features/video/VideoDiscoverTab.tsx` (777 lines)

**Props (50-82):** same shape as Image Discover minus the access filter, plus LongLive: `longLiveStatus`, `installingLongLive`, `longLiveJob`, `onRefreshLongLiveStatus`, `onInstallLongLive: () => Promise<InstallResult>`; `onVideoDownload(repo, modelId?)`.

**State:** same as Image Discover (local `statusFilter`/`sortDir`, hook state in `useVideoState.ts:82-87, 277-315`). Download map is keyed by `variant.id` first (`videoDownloadStatusForVariant`, `utils/videos.ts:41-48`); `handleVideoDownload` writes both `[modelId]` and `[download.repo]` (`features/video/downloadActions.ts`). Effect on mount: if any row is a LongLive repo, `onRefreshLongLiveStatus()` (274-276).

**Actions (beyond Image Discover):**
- LongLive row (672-694): installed gives Generate; else an Install button (disabled while installing) + `<InstallLogPanel job variant="longlive">`, plus an install-note callout (751-759: isolated venv, CUDA only, 5-15 min).
- Pause uses `downloadState?.repo ?? variant.repo` (707); Cancel/Delete uses `videoDeleteRepoForVariant` and `videoDeleteLabelForRepo` (shared GGUF / text encoder / component labels).
- `localStatusReason` callout (764-768) when not installed and not downloading.

**Data shown:** like Image Discover but no gated/trackedOnly/license; shows downloads and likes labels.

**Status (156-173):** LongLive: installed if `longLiveStatus.realGenerationAvailable`, downloading if `installingLongLive`; else the image logic.

**Shared helpers:** + `InstallLogPanel`, `videoDiscoverMemoryEstimate`, `videoDeleteLabelForRepo`, `videoDeleteRepoForVariant`, `videoDownloadStatusForVariant`, `videoPrimarySizeLabel`, `videoSecondarySizeLabel`.

**i18n:** `library:videoDiscover.*` (58 keys), `common:tabs.videoDiscover`/`tabs.videoModels`. **Tests:** none for the tab; `utils/__tests__/videos.test.ts` covers `videoDownloadStatusForVariant`, delete-target helpers, discover search, memory estimate.

### 1.6 Video Models: `src/features/video/VideoModelsTab.tsx` (513 lines)

**Props (38-64):** `installedVideoVariants`, `videoCatalog`, `activeVideoDownloads`, `loadedVideoVariant`, `fileRevealLabel`, `nativeBackends?`, `hostSystem?`, `onActiveTabChange`, `onOpenVideoStudio`, DL handlers, `onOpenExternalUrl`, `onRevealPath`. **Dead props (destructured at 169-171, 181-182, never used):** `videoRuntimeStatus`, `videoBusy`, `videoBusyLabel`, `onPreloadVideoModel`, `onUnloadVideoModel`.

**State:** local, as Image Models. Adds a "loaded" status and an "In memory" filter option (327). Size sort key includes `ggufFileGb` first (FU-054, line 74); Discover's does not.

**Actions:** as Image Models, with video delete target/labels (413-416, 484-486). No LongLive handling (an installed LongLive is `availableLocally` only if the backend says so; its readiness comes from the runtime probe, so it may not appear here).

**Data shown:** as Image Models; the shared-repo badge is hardcoded English (453-464), not i18n'd; `localStatusReason` callout (497-501).

**Status (109-120):** loaded, downloading, paused, failed, installed, else incomplete.

**i18n:** `library:videoModels.*` (71 keys), `common:panels.installedVideoModels`. **Tests:** none.

---

## 2. Duplicated versus unique, per area

| Area | Duplicated (Discover and My Models) | Only in Discover | Only in My Models |
|---|---|---|---|
| **Chat** | search box (different semantics), capability chips, format chips, accel/strategy chips (different matching), DL controls (5 states), Chat + Load-for-server buttons (**different launch action**), status icons, RAM + compressed RAM columns, context column, expand-for-detail, release label, note, capability strip, sizes | family grouping + family card + variant table, **Hub live search results with file lists**, **Run from HF**, **Import existing models**, **Details modal**, **Convert**, fit-in-memory badge, hub sort, newest-first sort, accel chip Apple-only hiding, "Open model card on HF", family readme, "N installed" chip, collapsed-header DL chip | **backend filter**, sortable column headers (8 keys), **favourite stars + lift-to-top**, BROKEN badge + retry, partial-HF-cache inference, **synthetic DL rows**, source-kind badge, directory-label search, **Reveal**, **Delete model files** (with confirm), local path, quant column, total size / directory count subtitle |
| **Image** | search, task filter, status filter, sort, Clear Filters, summary line, header sort, table columns, accel pills, style tags, size, RAM/VRAM, released, status, Generate, DL controls, Reveal, model-card link, host gating | hero + "Open Studio", **access filter**, likes/downloads sorts, `note`, gated badge, **tracked-only** disabled CTA, license, likes/downloads labels, **gated-access error help buttons**, not-installed rows, `hasLocalData` delete | "Browse Catalog", family name subtitle, **shared-storage badge**, delete for installed/paused/failed/partial |
| **Video** | as Image | hero + "Open Studio", likes/downloads sorts, not-installed rows, **LongLive Install + log + install note**, LongLive status refresh effect | "Browse Catalog", family subtitle, **"Loaded in memory" status + filter**, shared-storage badge, `ggufFileGb` size key |

---

## 3. App.tsx wiring and navigation

**Tab state:** `activeTab` is `useState<TabId>("dashboard")` (`App.tsx:148`). It is NOT persisted and there is NO URL or hash routing (grep found none).

**Render branches (all in one `if/else` chain):** `online-models` 1264-1306, `my-models` 1307-1355, `image-discover` 1356-1383, `image-models` 1384-1401, `video-discover` 1519-1549, `video-models` 1550-1573. Imports at 49-56. The chain is continued by `image-studio` 1402, `image-gallery` 1486, `video-studio` 1574.

**Props the branches pull from (so the merged branch needs the same):**
- Chat: `searchResults` filtered with `filterFamiliesForHost(…, workspace.system)`, `localVariantCount` (720), `accelCompat` from `workspace.system.dflash/mtplx/tensorfold/llamaServerTurboPath` + `isAppleSilicon` (710), `chatLibrary` (299), `activeDownloads`, `prepareCatalogConversion` (1051), `openModelSelector` (1126), `handleOpenExternalUrl`, hub state, `availableMemoryGb`.
- My Models: `filteredLibraryRows` (376-393), `libraryTotalSizeGb` (719, excludes synthetic rows), `enabledDirectoryCount` (718), `strategyCompat` (1321-1328), `fileRevealLabel` (721), `handleRevealPath`, `handleDeleteModel` (1041), favourites toggle (1344-1353, calls `updateSettingsApi({favoriteModelRefs})` then `refreshWorkspace`).
- Image: `imgState.*` (172), `nativeBackends`, `workspace.system`. Video: `videoState.*` (173) including LongLive.

**Special cases keyed on tab id:**
- `App.tsx:765`: `previewVariant` for `my-models` = first filtered library row's matched variant. `selectedLibraryPath` is only ever auto-set (292, 312-315), never by click. The default branch (online-models and others) uses `selectedVariant` from search results (286-288). `previewVariant` feeds `previewControls` (873-883).
- Header title and caption come from `tabs.find(activeTab)` (2031-2041); `ErrorBoundary key={activeTab} scope={tab.label}` (2172) so every tab switch remounts the page (local state resets).
- Details modal `App.tsx:2244-2351` is a THIRD copy of the variant download and launch controls (opened by `setDetailFamilyId` from Chat Discover).

**Sidebar definition:**
- `src/constants/tabs.ts:35-42` registers: `my-models` (group chat, shortLabel "My Models"), `online-models` ("Discover"), `image-models` ("My Models"), `image-discover` ("Discover"), `video-models` ("My Models"), `video-discover` ("Discover"). Each has `labelKey`, `captionKey`, `shortLabelKey` under `common:tabs.*`, `tabCaptions.*`, `tabShortLabels.*`.
- `src/constants/sidebarGroups.ts:11-13`: `defaultChild` is `chat` for Chat, **`image-models`** for Images, **`video-models`** for Video.
- `src/types/system.ts:1-27` is the `TabId` union (6 old ids at lines 3-6, 9-10). `SidebarGroupId` (29-37) already contains an unused `"models"` (icon at `SidebarIcons.tsx:148`, locale `tabGroups.models`).
- `Sidebar.tsx:56-72` `resolveGroupTarget`, `SubtabBar.tsx` (children = tabs of the group, uses `shortLabel`).
- `src/constants/__tests__/sidebarGroups.test.ts`: every `defaultChild` must be a real tab in that group; every grouped tab needs a `shortLabel`; every group needs at least one tab.

**Persisted state mentioning tab ids:** only `localStorage["chaosengine.sidebar.prefs.v1"]` (`useSidebarPrefs.ts:4`), shape `{mode, collapsedGroups, lastChildByGroup}`. `lastChildByGroup` can hold `"my-models"`, `"online-models"`, `"image-models"`, `"image-discover"`, `"video-models"`, `"video-discover"`. A stale id is harmless today (`resolveGroupTarget` falls back to `defaultChild`) but `rememberLastChild` will keep writing whatever id is clicked. Other localStorage keys (`chat.sidebarCollapsed`, per-session chat overrides, UI scale, locale banner, GPU banner) do not contain tab ids. No other persisted layout.

**Deep links and cross-links (code):**

| From | To | Where |
|---|---|---|
| Chat empty state "Browse Discover" (`onBrowseDiscover`) | `online-models` | `App.tsx:1798` → `ChatTab.tsx:136,200,530` → `ChatThread.tsx:54,490-494` → `ChatEmptyStateBanner.tsx:29,60` |
| Chat empty state "Load Model" (`onOpenModels`) | `my-models` | `App.tsx:1799`, `ChatEmptyStateBanner.tsx:78` |
| Image Studio buttons | `image-discover` (472, 585, 1088 "Browse smaller models"), `image-models` (475) | `ImageStudioTab.tsx` |
| Image Gallery "Installed Models" | `image-models` | `ImageGalleryTab.tsx:98` |
| Image Discover hero / Image Models "Browse Catalog" | `image-models` / `image-discover` | `ImageDiscoverTab.tsx:340`, `ImageModelsTab.tsx:260` |
| Video Studio | `video-discover` (703, 788, 1553), `video-models` (706) | `VideoStudioTab.tsx` |
| Video Gallery "Installed Models" | `video-models` | `VideoGalleryTab.tsx:65` |
| Video Discover hero / Video Models "Browse Catalog" | `video-models` / `video-discover` | `VideoDiscoverTab.tsx:382`, `VideoModelsTab.tsx:284` |

All callers go through the single `setActiveTab` from `App.tsx:148` (passed to `useImageState`, `useVideoState`, `useChat`, conversion actions, Sidebar, SubtabBar, modals). **A redirect map applied by wrapping that setter once covers every caller**, including future ones. Voice has the same pattern (`voice-models`, `VoiceStudioTab.tsx:52,83`) but is out of scope.

---

## 4. Behaviour table

Legend: D = Discover tab, M = My Models tab. `dup` = present on both tabs of the area (UI duplicated; logic may differ), `uniq-D`/`uniq-M` = only one tab. Chat = Chat area, Img = Image, Vid = Video.

| # | Behaviour | Chat | Img | Vid | Class and notes |
|---|---|---|---|---|---|
| 1 | Free-text search | D `searchInput` (token match + HF network); M `librarySearchInput` (substring haystack) | D hook query (`imageDiscover*MatchesQuery`); M local substring | same as Img | dup, **different semantics and state** |
| 2 | Live HF Hub search results | D only (`OnlineModelsTab.tsx:516-722`) | n/a | n/a | uniq-D |
| 3 | Capability chips | D (family or variant caps), M (`matchedVariant` only, unmatched rows drop out) | n/a | n/a | dup, semantics differ |
| 4 | Format chips | D (family has format; hub by `model.format`), M (`displayFormat`) | n/a | n/a | dup |
| 5 | Backend chips | M only | n/a | n/a | uniq-M |
| 6 | Acceleration / strategy chips | D `familyMatchesAccel` + Apple-only hiding; M `modelSupportsStrategy` + dflash-empty tooltip | n/a | n/a | dup, **different logic and tooltips** |
| 7 | Task filter | n/a | D, M | D, M | dup |
| 8 | Access filter (open / gated) | n/a | D only | n/a | uniq-D |
| 9 | Status filter | none | D (7 values incl. not-installed), M (6 values) | D (7), M (7 incl. loaded) | dup, **value sets differ**; Chat has none |
| 10 | Sort | D fixed newest-first; M 8 header keys + dir | D 9 keys; M 7 keys (no likes/downloads) | same as Img; M size key adds `ggufFileGb` | dup/uniq mix |
| 11 | Clear Filters button | none | D, M | D, M | dup |
| 12 | Result summary line and chips | none | D, M | D, M | dup |
| 13 | Platform gating of rows | D via `filterFamiliesForHost`; **M none** | D, M | D, M | dup (Chat asymmetric) |
| 14 | Favourite star + lift to top | M | n/a | n/a | uniq-M |
| 15 | Expand row / card for detail | D family + variant + hub; M row | none (no expand) | none | dup (Chat only) |
| 16 | Details modal (App-level) | D | n/a | n/a | uniq-D |
| 17 | Family header DL chip | D | n/a | n/a | uniq-D |
| 18 | Fit-in-memory badge (`memoryFitBucket`) | D | n/a | n/a | uniq-D (superseded by backend `fit`) |
| 19 | RAM / compressed RAM columns | D, M (M has tooltips) | n/a | n/a | dup |
| 20 | RAM/VRAM estimate | n/a | D, M | D, M | dup |
| 21 | Capability strip | D, M | n/a | n/a | dup |
| 22 | Accelerator pills | n/a | D, M | D, M | dup |
| 23 | Style tags | n/a | D, M | D, M | dup |
| 24 | Gated badge, Watching-upstream badge, license, likes/downloads labels | n/a | D | D (downloads, likes) | uniq-D |
| 25 | Shared-storage badge | n/a | M | M (hardcoded English) | uniq-M |
| 26 | `localStatusReason` callout | n/a | none | D, M | dup (Video) |
| 27 | BROKEN badge + reason | M | n/a | n/a | uniq-M |
| 28 | Source kind badge, directory-label search | M | n/a | n/a | uniq-M |
| 29 | Synthetic in-flight DL rows (`download://`) | M (App) | n/a | n/a | uniq-M |
| 30 | Partial HF cache shown as paused DL | M | n/a | n/a | uniq-M |
| 31 | Download | D, M (retry broken) | D, M | D, M | dup |
| 32 | Pause (= `onCancel*Download`) | D, M | D, M | D, M | dup |
| 33 | Cancel / delete partial DL (= `onDelete*Download`) | D, M | D, M | D, M | dup |
| 34 | Resume | D (paused), M | D (paused, partial), M (**only partial; paused shows Download**) | same as Img | dup, **M tabs inconsistent** |
| 35 | Retry (failed) | D, M | D, M | D, M | dup |
| 36 | Delete installed files | M (confirm) | D (if `hasLocalData`), M (**no confirm**) | D, M (**no confirm**, shared-repo labels) | dup, **asymmetric** |
| 37 | Reveal in file manager | M (not on `download://`) | D, M (if `localPath`) | D, M | dup |
| 38 | Open model card on HF | D | D, M | D, M | dup |
| 39 | Chat with model | D → `"thread"`; M → `"chat"` | n/a | n/a | dup, **different action** |
| 40 | Load for server | D, M (`"server"`) | n/a | n/a | dup |
| 41 | Convert model | D only | n/a | n/a | uniq-D (`onPrepareLibraryConversion` is dead in M) |
| 42 | Generate / Open in Studio | n/a | D, M | D, M | dup |
| 43 | Hero "Open Studio" + "Installed Models" | n/a | D | D | uniq-D (become self-links) |
| 44 | "Browse Catalog" | n/a | M | M | uniq-M (self-link) |
| 45 | Gated-access error help (HF + Settings buttons) | n/a | D | n/a | uniq-D |
| 46 | LongLive install + `InstallLogPanel` + status refresh | n/a | n/a | D | uniq-D |
| 47 | "Loaded in memory" status + filter | n/a | n/a | M | uniq-M |
| 48 | Run from Hugging Face panel | D | n/a | n/a | uniq-D |
| 49 | Import existing models panel | D | n/a | n/a | uniq-D |
| 50 | Hub card file preview (weights collapsed above 8) | D | n/a | n/a | uniq-D (test-pinned) |
| 51 | Empty states | D (3 strings), M (1) | D, M | D, M | dup, different text |
| 52 | Summary subtitle | D "families / downloaded"; M "models / size / dirs" | D "x of y"; M "rows of total" | same | dup, different |

---

## 5. Gaps, dead code and inconsistencies worth knowing

- Chat Hub-card "Chat with model" / "Load for server" call `onOpenModelSelector("thread")` / `("server")` with no preselected key (`OnlineModelsTab.tsx:689-690`); the launch modal falls back to the loaded model, not this model.
- `RunFromHuggingFace` calls `api.downloadModel` directly (`RunFromHuggingFace.tsx:~50`), not `useModels.handleDownloadModel`, so `activeDownloads` in App does not learn about it and polling does not start. Its notice says "track progress in My Models".
- Accelerator pills in all four media tabs use `getApplicableAccelerators(repo)` (not host-aware); only `MediaStudioBoosters.tsx:139` uses `getApplicableAcceleratorsForHost`.
- `VideoPlaceholderTab.tsx` has `models` and `discover` variants and locale keys (`common.json:1189`) but nothing imports it.
- `common:imageDiscover.*` (982) and `common:videoDiscover.*` (1004) appear unused (these tabs read `library:`).
- `MyModelsTab` `StrategyCompatInfo.mtplxInstalled` is carried but not used in matching.
- `library:filter.installed/uninstalled/favourites/...` (generic filter keys) have no `t()` caller in `src`.

---

## 6. Smallest shared component set

Goal: cover all six tabs with the fewest parts. Per-area differences go through adapters and slots, not forks.

**1. `ModelEntry` (type) + four adapters** (`chat catalog+library join`, `libraryRowToEntry`, `imageVariantToEntry`, `videoVariantToEntry`):
`{ id, area: "chat"|"image"|"video", name, subtitle?, provider, familyName?, repo, downloadKey, deleteTarget, installState: "installed"|"available"|"downloading"|"paused"|"failed"|"partial"|"loaded"|"broken", download?: DownloadStatus, sizeGb, memoryLabel?, memoryTitle?, fit?: Fit, releasedLabel?, tasks?, capabilities?, styleTags?, acceleratorRepo?, localPath?, link, flags: {trackedOnly?, gated?, shared?: {count,key}, favourite?, source?: string, broken?: {reason}, longLive?}, extraColumns }`. `downloadKey` is needed because video keys downloads by `variant.id`, chat and image by `repo`.

**2. `resolveModelStatus(entry, ctx)` (pure) + `<ModelStatusChip>`**
- Props: `status: ModelStatusKind | "notInstalled"`, `label`, `detail?`. Wraps `StatusIcon`.
- The function replaces `imageVariantStatus`, `imageStatus`, `videoVariantStatus`, `videoStatus`, and the three inline chat ladders (`OnlineModelsTab.tsx:388-396`, `540-550`; `MyModelsTab.tsx:467-475`). It also yields the filter bucket: Installed / Available / Downloading / Needs attention (paused, failed, partial, broken). Must handle the LongLive runtime-probe case, `loaded`, and `hasLocalData`.

**3. `<ModelDownloadControls>`**
- Props: `state: "none"|"downloading"|"paused"|"failed"|"completed"|"partial"`, `installed: boolean`, `onDownload`, `onPause`, `onCancel`, `onResume`, `onRetry`, `onDeleteModel?`, `deleteLabel?`, `deleteConfirm?: string`, `disabled?`, `trackedOnly?` (disabled CTA with tooltip), `extra?` slot (Generate / Chat / Server / Convert).
- Replaces the 4-state cluster in `OnlineModelsTab.tsx:447-476, 686-713`, `MyModelsTab.tsx:543-566`, `ImageDiscoverTab.tsx:674-768`, `ImageModelsTab.tsx:444-465`, `VideoDiscoverTab.tsx:671-749`, `VideoModelsTab.tsx:472-491`. Uses Discover's behaviour for paused (Resume + Delete download).

**4. `<ModelFilterBar>`**
- Props: `search: {value, onChange, placeholder}`, `availability: {value: "all"|"installed"|"available"|"downloading", onChange, counts}`, `groups: Array<{id, label, kind: "chips"|"select", options: Array<{value, label, count?, color?, title?, hidden?}>, value, onChange, hideBelow?: number}>`, `sort?: {options, value, dir, onChange}`, `hasActive`, `onClear`, `summary?: ReactNode`.
- Covers: capability, format, backend, acceleration/strategy chips (colour and "memory saver" tooltips), task, access, status, sort selects, Clear Filters, summary line.

**5. `<ModelCard>` (row or card)**
- Props: `entry`, `columns`/`layout`, `badges?`, `fit?` (`<FitBadge fit>` from ui-core), `pills?` (`<AcceleratorPills repo host nativeBackends>`), `status`, `actions` (slot, usually `ModelDownloadControls`), `detail?` (expandable slot: note, release label, path, model-card link), `callouts?` (error, quiet `localStatusReason`, LongLive install note, gated-access help), `className` (`downloaded`, `download-active`, `download-warning`).

**6. `<AcceleratorPills>`**: wraps `getApplicableAccelerators` (or the host-aware variant) plus `AcceleratorCard variant="pill"`. Used in 4 tabs today.

**Stay specialised (not forced into the shared card):** the Hub result card with file groups (`OnlineModelsTab.tsx:552-717`), `RunFromHuggingFace`, `ImportModelsPanel`, LongLive install control (`InstallLogPanel`), the Details modal (could be removed once the card is expandable, but see R6).

---

## 7. Risks (where merging could drop behaviour or break deep links)

- **R1 Chat join.** Installed state has three sources: catalog `availableLocally`, library scan row (fuzzy `matchedVariant` via `findCatalogVariantForLibraryItem`/`findLibraryItemForVariant`, score threshold 12), and `activeDownloads`. Local models with no catalog match exist only in My Models; catalog variants not installed only in Discover. Needs one explicit join rule or rows duplicate or vanish (`App.tsx:319-331`, `OnlineModelsTab.tsx:381-388`).
- **R2 Launch action.** Discover "Chat" = `"thread"` (new thread patched from the variant, `App.tsx:1177-1189`), My Models "Chat" = `"chat"` (re-point the active session, 1190-1215). A merged row must pick or keep both.
- **R3 Hub card** buttons have no preselected key (see section 5).
- **R4 Convert** exists only on Discover rows (`OnlineModelsTab.tsx:450-452`). Unmatched local models cannot be converted from either page today (the Conversion tab does it).
- **R5 My Models-only features** that disappear if the page is catalog-first: BROKEN badge and reason, Retry, partial-HF-cache inference (157-179; test), synthetic `download://` rows (hide Reveal and path; test), source badge, directory-label search, Reveal, delete with confirm, favourites (persisted to settings), backend filter, modified/context/compressed sorts, size and directory count subtitle.
- **R6 Discover-only features:** Hub network search, Run from HF, Import, Details modal (App-level, the only place family readme and per-variant pause/resume text show together), fit badge, readme, Convert, Apple-only accel chip hiding.
- **R7 Filter semantics differ** between D and M for capability, format, strategy (see rows 3-6). Merging must keep each rule on its own data (family vs library row).
- **R8 One search box vs two states.** `searchInput` triggers a debounced network search; `librarySearchInput` is a local filter. One merged input changes network behaviour and match rules (`modelFamilyMatchesDiscoverQuery` vs substring).
- **R9 Status buckets.** Image Discover checks installed first; Image/Video Models check downloading first. LongLive "installed" comes from the runtime probe, not `availableLocally`. An "Installed" filter built from `availableLocally` hides LongLive; built from `installed*Variants` (`availableLocally || hasLocalData`) it shows partials as installed. Define buckets before building.
- **R10 Paused label bug** in both Models tabs (Download icon and label instead of Resume; delete labelled "Delete model"). Use the Discover behaviour.
- **R11 Delete confirmation asymmetry.** Chat deletes files after `window.confirm` (`features/app/modelActions.ts:57`); image/video delete model files (including shared siblings and, for video, shared GGUF or text-encoder repos) with no confirm (`ImageModelsTab.tsx:456-460`, `VideoModelsTab.tsx:484-486`, `deleteImageDownloadById`, `handleDeleteVideoDownload`). Decide, do not drift.
- **R12 Download key divergence.** Chat and image key by `repo`; video keys by `variant.id` first (`utils/videos.ts:41-48`) and cancels with `downloadState?.repo ?? variant.repo`; video delete uses `videoDeleteRepoForVariant`/`videoDeleteLabelForRepo`; `onVideoDownload(repo, modelId)` has two args.
- **R13 Video size sort** uses `ggufFileGb` only in Models (`VideoModelsTab.tsx:74`); keep it in the merged sort.
- **R14 `memoryFitBucket`.** Exported from `OnlineModelsTab.tsx:85`, tested (7 tests), CSS `.memory-fit-badge*` (`styles.css:9012-9036`), keys `onlineModels.memory.*`. It compares against currently available RAM (volatile) while backend `fit` uses static usable memory. Replace deliberately, update or replace its test.
- **R15 Tests pin markup.** Strings in section 1.1/1.2 plus class `discover-card expanded` and `<details class="hub-file-group hub-file-group--collapsible">`. Port the assertions, do not drop them.
- **R16 State lifetime.** `ErrorBoundary key={activeTab}` (`App.tsx:2172`) remounts on tab switch. Local state (status filter, sort dir, all of Image/Video Models filters, chat `strategyFilter`) resets; hook-level state (search, task, access, sort, cap/format/backend filters, expansion) persists. A merged page should decide which survive.
- **R17 Self-links and presets.** "Installed Models", "Browse Catalog", Studio "Discover"/"Installed", "Open Image/Video Discover", "Browse smaller models" (`ImageStudioTab.tsx:1088`, `VideoStudioTab.tsx:1553`) all land on the same page. Redirects should carry a preset (`installed` vs `available`) or the buttons lose meaning. Same for Chat empty-state: Browse Discover = available, Load Model = installed.
- **R18 Preview variant.** `App.tsx:765` special-cases `my-models`; the new chat Models page falls into the default branch (search selection) unless handled. Affects `previewControls` sync (873-883).
- **R19 Platform gating asymmetry.** Chat M shows MLX/vLLM rows that the host can't run; image/video M hide them. Choose per area.
- **R20 Accelerator pills** not host-gated (section 5).
- **R21 Dead props.** `VideoModelsTab`: `videoRuntimeStatus`, `videoBusy`, `videoBusyLabel`, `onPreloadVideoModel`, `onUnloadVideoModel` (App passes them at `App.tsx:1556-1558, 1568-1569`). `MyModelsTab`: `onPrepareLibraryConversion` (passed at `App.tsx:1336`). Dropping is safe; `App.tsx` must stop passing them.
- **R22 i18n rules.** `scripts/i18n-validate.mjs` flags any key present in some locale but missing from `en` as an orphan, so removing an `en` key means removing it from all 10 locales (en, zh-CN, zh-TW, ja, de, ru, ko, fr, es, pt-BR). Keep the old keys until the cut-over is done. `locales/__tests__/parity.test.ts` enforces leaf-key parity. Video's shared-storage badge is hardcoded English (`VideoModelsTab.tsx:453-464`).
- **R23 Sidebar test and persisted ids.** Update `defaultChild` for images and video (`sidebarGroups.ts:12-13`) or `sidebarGroups.test.ts` fails. Keep or migrate stale `lastChildByGroup` values.
- **R24 CSS scroll containers.** `.workspace-content-frame:has(.image-discover-stack)` (`styles.css:7178`), `:has(.image-discover-page)` (7196), `:has(.discover-page)` (435) make the page scroll. The new shell needs one of these classes or its own rule. `.media-model-table` is an 8-column grid with `min-width: 1000px` rows (7576-7603); chat uses `.discover-variant-table` and `.library-full-table`. Class names with `image-discover-` are shared by video tabs.
- **R25 LongLive.** `onRefreshLongLiveStatus` must still run on the merged Video page whenever a LongLive row can render; its CUDA gate hides it on Mac.
- **R26 Tracked-latest de-dup** (`useImageState.ts:241-248`) and the `trackedOnly` disabled CTA (FU-061) must survive. Image Models uses `imageCatalog` (not "with latest") so tracked rows show provider as subtitle; Discover shows `note`. Pick one subtitle.
- **R27 Editorial rule.** `ImportModelsPanel` labels stores by folder and does not name other apps in UI strings; API fields `ollama`/`lmstudio` exist in code (`ImportModelsPanel.tsx:63-64`). Do not add UI strings that name them.

---

## 8. Everything that references the six old tab ids or labels

### 8.1 Code that holds the ids
- `src/types/system.ts:3-6, 9-10` (TabId union).
- `src/constants/tabs.ts:35-38, 41-42` (labels, captions, short labels), `src/constants/sidebarGroups.ts:12-13` (`defaultChild`).
- `src/App.tsx:49-56` imports; 765 preview special-case; 1264, 1307, 1356, 1384, 1519, 1550 render branches; 1798-1799 chat deep links; 2031-2041 header; 2172 error boundary scope.
- Literal id call sites (14 in 8 files): `ImageDiscoverTab.tsx:340`, `ImageModelsTab.tsx:260`, `ImageStudioTab.tsx:472, 475, 585, 1088`, `ImageGalleryTab.tsx:98`, `VideoDiscoverTab.tsx:382`, `VideoModelsTab.tsx:284`, `VideoStudioTab.tsx:703, 706, 788, 1553`, `VideoGalleryTab.tsx:65`.
- Callback names that carry the labels: `onBrowseDiscover`, `onOpenModels` (`ChatTab.tsx:136-137,200-201,530-531`, `ChatThread.tsx:54-55,110-111,490-494`, `ChatEmptyStateBanner.tsx:29-38,60,78`).
- Persisted: `chaosengine.sidebar.prefs.v1` → `lastChildByGroup` (`useSidebarPrefs.ts:4`).
- Tab titles used as panel titles: `common:tabs.myModels` (`MyModelsTab.tsx:351`), `t("tabs.imageDiscover")`/`tabs.imageModels` (`ImageDiscoverTab.tsx:318, 535`), `tabs.videoDiscover`/`tabs.videoModels` (`VideoDiscoverTab.tsx:360, 550`), `panels.discoverModels` (`OnlineModelsTab.tsx:727`), `panels.installedImageModels` (`ImageModelsTab.tsx:250`), `panels.installedVideoModels` (`VideoModelsTab.tsx:274`).
- CSS class names (not ids, but named after the tab): `image-discover-*` (`styles.css:6113-6118, 7159-7296, 7556-7570, 7714-7723, 7918-7939`).
- Comments naming the tabs: `App.tsx:335, 401, 969`; `types/settings.ts:72`; `types/image.ts:111`; `RunFromHuggingFace.tsx:13-14`; `ImportModelsPanel.tsx:8`; `AcceleratorCard.tsx:23, 61`; `tensorfoldSupport.ts:140`; `platform.ts:174-200`; `hooks/useFileActions.ts:13`; `hooks/useModels.ts:67, 91`; `VideoStudioTab.tsx:334`; `ImageStudioTab.tsx:258`.

### 8.2 Locale keys (`src/locales/<lang>/`, 10 locales; `en` line numbers)
- `common.json`: `tabs.myModels/onlineModels/imageModels/imageDiscover/videoModels/videoDiscover` (12-19), `tabCaptions.*` (45-52), `tabShortLabels.*` (72-79), `tabGroups.models` (unused group), `panels.discoverModels` (186), `panels.noModelsMatchFilters` (201), `panels.installedImageModels/installedVideoModels` (203-204), `runHf.downloadStarted` (1043, "track progress in My Models"), `importModels.subtitle` (1075), `discoverModal.*` (1097), `imageDiscover.*` (982) and `videoDiscover.*` (1004) (likely unused), `videoPlaceholderTab.*` (1189, dead component).
- `library.json`: `myModels` (47), `onlineModels` (135), `imageModels` (247), `videoModels` (344), `imageDiscover` (441), `videoDiscover` (534); `onlineModels.searchError.fallback` ("Showing the last successful Discover results."); `imageModels.empty.noModels` and `videoModels.empty.noModels` ("…from Image Discover / Video Discover…"); `*.hero.installedModels`, `*.action.browseCatalog`.
- `chat.json`: `composer.placeholderNoModel` (4, "My Models or Discover"), `swapMenu.noWarmModels` (117, "from My Models"), `emptyBanner.welcomeBody` (403), `emptyBanner.browseDiscover` (404), `emptyBanner.loadModel` (406).
- `studio.json`: `image.noModelsOption` (30), `runtimeBanner.summaryNotInstalled` (155), `imageStudio.actions.discover/installed` (176-177), `imageStudio.noModelsCallout.message/openDiscover` (191-193), `imageStudio.download.notInstalledMessage` (199), `imageStudio.safety.browseSmaller` (262), `videoStudio.actions.browseCatalog/installedModels` (398-399), `videoStudio.noModelsCallout.*` (410-412), `videoStudio.safety.browseSmaller` (516), `imageGallery.actions.installedModels` (576), `videoGallery.actions.installedModels` (630), `voice.studio.needsPack` (754, "Voice > My Models", leave).
- `src/locales/__tests__/parity.test.ts` and `scripts/i18n-validate.mjs` guard parity, ICU syntax and orphans.

### 8.3 Tests
- `src/features/models/OnlineModelsTab.test.tsx`, `MyModelsTab.test.tsx`, `__tests__/memoryFitBucket.test.ts` (import `memoryFitBucket` from `../OnlineModelsTab`).
- `src/constants/__tests__/sidebarGroups.test.ts` (defaultChild, shortLabel, group membership).
- Related util tests that should keep passing: `utils/__tests__/images.test.ts`, `videos.test.ts`, `models.test.ts`, `platform.test.ts`; `components/__tests__/AcceleratorCard.test.tsx`, `acceleratorCatalog.test.ts`, `runtimeSupport.test.ts`, `tensorfoldSupport.test.ts`.
- `src/__tests__/featureTabsSmoke.test.ts` only imports the two Studio tabs (not the six).
- No test references the six tab ids as strings. No test covers the four media tabs themselves.

### 8.4 Docs and scripts
- `README.md` lines 49-50, 60, 131, 161-179, 431 (names Discover, My Models, Image Discover, Image Models; screenshots `Screenshots/2. Discover_v2.png`, `3. My_Models_v2.png`, `5. Image Discover.png`).
- `docs/getting-started/first-run.md:44`, `docs/usage/image-studio.md:20`, `docs/features/cache-strategies.md:42` ("Models and Discover filters"), `docs/features/tensorfold.md:164`, `docs/image-generation-mvp.md:57-68, 297-298` (lists `image-discover`, `image-models`), `docs/image-discover-next-level.md` (a feature name, not a tab), `mkdocs.yml:135`.
- Hub: `docs/hub/ideas.md:14`, `docs/hub/tracker/open.md` (FU-129, FU-133), `tracker/README.md:24`, `tracker/closed.md` (FU-054, FU-061, FU-108 mention the tab names), `CLAUDE.md` (FU-054 line 189, FU-108 222, FU-129 240, FU-133 244, FU-061 271), `COVERAGE_BASELINE.md:90` (lists `MyModelsTab.tsx`), `RELEASE_NOTES_v0.9.8.md:56, 74`, `CHANGELOG.md`.
- `scripts/capture-screenshots.sh:20-21` (pages `Discover:discover`, `My Models:my-models`, clicks sidebar buttons by label).
- `scripts/e2e_test_suite.py:305, 446, 517` (comments only).

### 8.5 Backend user-facing strings (no tests assert on them)
- `backend_service/state/lifecycle.py:79` ("Open the Discover tab and download it first").
- `backend_service/state/sessions.py:340` ("via My Models").
- `backend_service/inference/mlx_engine.py:210, 283, 352, 387, 440` ("reload the model from My Models").
- Comments in `helpers/model_classifier.py`, `helpers/settings.py:254`, `models/__init__.py:251`.

### 8.6 Tauri
`src-tauri/locales/*.yml` and `src-tauri/src/{lib,i18n,settings}.rs` contain only generic menu, tray, updater strings. No tab ids or labels. Nothing to change.

### 8.7 Redirect completeness checklist
1. Add one `normalizeTabId` / legacy map and wrap `setActiveTab` at `App.tsx:148`.
2. Map: `online-models`, `image-discover`, `video-discover` → area Models page with "available" preset; `my-models`, `image-models`, `video-models` → area Models page with "installed" preset.
3. Migrate `lastChildByGroup` values on read (`useSidebarPrefs.ts:readStored`).
4. Update `sidebarGroups.ts` `defaultChild`, `tabs.ts`, `TabId`; remove or keep old ids in `TabId` (removing makes the compiler list all 14 call sites).
5. Update the 14 literal call sites and the 2 App deep links; reword chat empty-state strings.
6. Reword locale strings in 8.2 (lead owns locale JSON; tell the lead the list), docs in 8.4, backend strings in 8.5.

---

## 9. Decisions to propose (for the lead to log)

1. Which launch action an installed Chat row's primary button uses (`"thread"` vs `"chat"`), or both (primary + menu).
2. Status buckets and their exact definitions (Installed / Available / Downloading / Needs attention) and where LongLive and `loaded` fit.
3. Delete confirmation for image and video model files (add, or keep as is).
4. Whether installed-but-host-incompatible models stay visible on the Models page (Chat shows them today, Image/Video hide them).
5. Single subtitle on the shared card (`note` vs family name).
6. Whether to keep the old ids in the `TabId` union (compat) or remove them and rely on the redirect map plus compiler errors.
7. Whether accelerator pills become host-aware (`getApplicableAcceleratorsForHost`).

## Not done / not tested

- Nothing was run (no app, no vitest, no tsc, no backend). Line numbers are from the working tree at `4b2762b` plus uncommitted `docs/hub` edits.
- Windows and Linux behaviour (host gating, CUDA-only rows such as LongLive) is read from code only.
- I did not inspect the backend catalog payload builders for `fit` / `maxSide` / `recommendedSettings` (other streams own them).
- The `docs/site` and `site` built output were not scanned.
