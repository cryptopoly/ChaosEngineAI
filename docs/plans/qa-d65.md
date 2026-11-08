# QA review: Projects workspace rework (D-65 phases 1 and 2)

Date 2026-11-08. Branch `feat/project-workspace`, worktree `.claude/worktrees/agent-a0601f472d24874f1`,
range `c95112a..22e07f6` (6 commits). Read-only review: nothing edited or committed in the worktree.
Paths below are relative to the worktree.

## Verdict

One blocker (undo after the format 3 upgrade), four majors. Nothing in the keep-list is lost. The voice free
text is the only capability that is no longer reachable from the UI; it is still kept in stored projects and an
agent can still set it.

## What I ran

| Check | Result |
|---|---|
| `.venv/bin/python -m pytest tests/test_storyboard*.py tests/test_mcp_bridge.py tests/test_mcp_contract.py` | 201 passed, 41 subtests passed (106 s) |
| `npx tsc --noEmit` | clean |
| `npx vitest run src/features/storyboard src/app src/constants` | 12 files, 257 passed |
| `npm run i18n:validate` | OK (warn mode); other locales at 93.6 % (base: about 96.6 %) |
| Scratch reproduction: format 2 project and history, cast edit, upgrade, Undo | **fails**, see B1 |
| Scratch: format 3 add / removeImage / undo / redo, fingerprint | works; fingerprint restored on undo |
| Scratch: add_reference with a repeated source, then refused at the limit | refused correctly, leaves one orphan file (m10) |

Not run: Playwright e2e (05, 09), the full `scripts/e2e_test_suite.py`, the Tauri build. The 375 px screenshots
were not looked at. Paste in WKWebView not tried (M1).

## Findings

| # | Severity | File:line | Problem | Fix |
|---|---|---|---|---|
| B1 | blocker | `backend_service/storyboard/store.py:446-464` (`_replay`), `:37-42` (`MIGRATIONS`) | History entries written before the upgrade hold format 2 references (`image`, `hash`). The migration only rewrites `project.json`, so undoing or redoing a cast change made before the upgrade replays a format 2 `references` value and validation refuses it: "Cannot undo that change: Extra inputs are not permitted". Because undo goes down the stack, every earlier change is then blocked too. Reproduced with a scratch script on temp folders. | In `_replay`, run both history sides' `references` through `_refs_to_lists` before `_apply_side`, or do it in `_history_entry`. Add a test: write a format 2 project and history entry, load, undo, redo. |
| M1 | major | `src/features/storyboard/ReferenceCard.tsx:83-90,106`; `ReferenceImagePicker.tsx:84-89,106` | Paste relies on a `paste` event reaching a non-editable element. The card `<article>` has no `tabIndex`, so "paste with the card focused" (docs/usage/storyboard.md, Cast) only works when a button inside it has focus. WebKit (the macOS Tauri webview) normally fires `paste` only where something is editable, so Cmd+V is likely to do nothing in the shipped app. No test covers paste (vitest, e2e or the suite). | Make the card focusable (`tabIndex=0`) and handle Cmd/Ctrl+V in `keydown` with a `navigator.clipboard.read()` fallback for image blobs. Try it in the Tauri build, add a Playwright test with clipboard permission, and reword the docs to say what is actually true. |
| M2 | major | `ReferenceImagePicker.tsx:211-228` via `ProjectMedia.tsx:9-36` | The "From a shot" grid loads up to 48 stills (`PICKER_GRID_LIMIT`) at full resolution as blob URLs as soon as the picker opens. `loading="lazy"` has no effect because the fetch happens in the hook. At 1080p to 4K that is hundreds of MB. Cast cards also load each picture (up to 4 per reference) at full size. | Serve a downscaled thumbnail (a `?w=256` variant on the file route), or start each fetch from an IntersectionObserver and lower the limit to about 24. |
| M3 | major | `ExportTab.tsx:31`, `mixer/MixerTab.tsx:99`, `useStoryboardRender.ts:76-95` | Each tab mounts its own `useStoryboardRender`, and on mount it only adopts the newest job. After a preview is rendered in the Mixer, the Export tab no longer shows the last export's result card (size, Reveal file, Copy path, Also written), which the single Mixer hook used to keep. The elapsed clock also restarts on every tab switch, and the Mixer and Export tabs each poll. | Lift `useStoryboardRender` into `ProjectShell` and pass it to both tabs, or have `GET .../render` return the last finished export as well as the newest job. |
| M4 | major | `tests/` (gap) | Nothing tests B1, the upload size cap (413), an empty body, the shell's states (offline, not found, error) or `StoryboardView`'s choice between list and shell, including a stored project that was deleted. Nothing tests the picker as a component (choose file, then upload, then `onPick`; Escape; source keys), Cast drop and paste, `edit_reference.image` replacing every picture, or `removeImage` out of range. e2e covers the gallery and shot-still sources but not Choose file, drop or paste. | Add pytest for B1, 413 and an empty body, plus Testing Library tests for `ProjectShell` states, `ReferenceImagePicker` and `CastTab` drop. Add one Playwright `setInputFiles` test for Choose file. |
| m1 | minor | `ProjectCard.tsx:31-42`, `ProjectListTab.tsx:102` | A card whose folder was moved or deleted has no menu, and `confirmDelete` needs `card.project`, so it can never be removed. Design 4.1 promises "only Delete in the menu". This was already true at base. | Show a menu with Delete only, and allow the delete route without a revision for an unreadable project. |
| m2 | minor | `ProjectDefaultsPanel.tsx:102`, `VoicePicker.tsx` | The voice free text is gone, but design section 2 says it is "demoted and stays reachable". A stored custom value still shows (`voicePickerLogic.ts:36-39`) and an agent can set it; a user cannot type one. | Add a collapsed "Type a voice id" link like "Paste a path", or log the cut as a decision. |
| m3 | minor | `VoicePicker.tsx:21,33` | The voice list is cached at module level for the whole session, so a voice designed or a model installed later is missing until the app reloads. | Fetch again each time the picker opens, or when the cache is older than about 60 s. |
| m4 | minor | `projectRoute.ts:96`, `useSendToStoryboard.ts:89` | "Projects" (the back link) sets `projectId` to null. The send menu's "open project first" ordering then loses the project just used. Design 4.3 says that project stays "last open". | Keep a `lastOpen` (the last key of `lastTabs`) and have `readOpenProjectId` fall back to it. |
| m5 | minor | `src/constants/tabRedirects.ts` (`normalizeStoredTabId`) | A stored active tab of `story-mixer` or `story-projects` becomes `storyboard` without its route. It reopens the last project on its remembered tab instead of the Mixer or the list. Low impact; the legacy key migration covers the common case. | Apply `LEGACY_TAB_REDIRECTS[id].storyboard` when normalizing at startup. |
| m6 | minor | `ProjectShellHeader.tsx:127-138` | The format chips go to Overview but leave Project settings closed (design 4.3: "opens Overview > Project settings"). | Pass an "open settings" request so the Disclosure expands and takes focus. |
| m7 | minor | `ProjectShellHeader.tsx:84,177`; `ProjectShell.tsx:122` | An `<h2>` inside a `<button>` is invalid content and loses the heading. `aria-controls` on unselected tabs points at panels that are not rendered. The tabpanel has no heading (design section 7). | Put the heading outside the button (the button inside the h2), set `aria-controls` only on the selected tab, and add a visually hidden heading to the panel. |
| m8 | minor | `ProjectDialog.tsx` (Start from) | The template pills are `role="radio"` but arrow keys do nothing (the WAI radiogroup pattern expects them). | Reuse `nextTabIndex` for arrow keys with a roving tabIndex. |
| m9 | minor | `ReferenceCard.tsx:98-103`, `CastTab.tsx:71,92-100` | When a card is full, its `dragover` returns without `stopPropagation`, so a drop lands on the tab and creates a new character named after the file. Files beyond the room, or beyond 8 on the tab, are dropped without a word. | Stop propagation on a full card and say "This card holds 4 pictures"; report the skipped files. |
| m10 | minor | `backend_service/storyboard/ops.py:269-274,316-321,366` | When the same source appears twice (`image` plus `images`, or twice in `images`), it is staged once; the second copy is ingested outside staging. If the op is then refused (for example a fifth picture), that copy stays in `media/refs` as an orphan. Reproduced. The same picture can also be added twice. | Remove duplicates in `AddReference` and check `len(image)+len(images) <= 4` in a model validator, before staging. |
| m11 | minor | `backend_service/routes/storyboard.py:316-348` vs `routes/agent.py:106-133` | The streamed upload loop is copied. Only magic bytes are checked: a PNG header followed by junk is accepted and copied into the project. The cap (30 MB), path handling (random id, never the client's name, inside `agent-inbox`) and cleanup (part file always removed, inbox cleared after a day) are fine. | Extract a shared `files.receive(request, allowed)`. Optionally `PIL.Image.verify()` on finish, imported inside the function. |
| m12 | minor | `backend_service/storyboard/ops.py` | 609 to 645 lines; it was already over the 600 cap and the brief said not to make it worse. (`MixerTab.tsx` 902 to 699: better.) | Move the reference ops (`_source`, `_ref_image`, `add/edit/remove_reference`) to `storyboard/ref_ops.py` and re-export them. |
| m13 | minor | `src/locales/*/studio.json`, `common.json` | 132 new keys are `null` in 9 locales: coverage 96.6 % to 93.6 %, under the 95 % release gate. The orphan keys `tabs.storyProjects`, `storyBoard` and `storyMixer` (and captions and short labels) remain. | Add an FU row for the translations before the next release, and drop the orphan keys. |
| m14 | minor | `ProjectShellHeader.tsx:136`, `ProjectCard.tsx:51` | The "fps" unit is hardcoded in JSX. | Use an i18n key `{fps} fps`. |
| m15 | minor | `StoryboardView.tsx:54-62` | `fallbackName` is never passed, so the header is blank while the project loads (design section 4: "Header with the name from the list"). | Pass the name from the list cache (`readCards`) or from the route. |
| m16 | minor | `ProjectListTab.tsx:67-99` | A template project is made as create plus N `add_shot` calls. If one fails, the project already exists but the dialog stays open with an error, and Create again makes a duplicate. | When a create succeeded, close the dialog and open the project with a status line even if a template shot failed. |
| m17 | minor | `backend_service/mcp_bridge/storyboard_tools.py:174-188` | An agent cannot use a shot still as a cast picture: `_agent_source` refuses `media/...`, though the UI can, and `ops._source` already confines it with `media.inside`. Argument caps (maxItems 4, maxLength, remove_image 0..3) are enforced by the bridge, and the scope is the tool's Edit scope. | Allow a `media/` path through unchanged for `add_ref` and `edit_ref`, or say in the tool description that it is refused. |
| m18 | minor | `storyboard_tools.py:47-51` | `add_ref` finds `refId` by comparing against a load taken outside the lock, so a concurrent add can return the wrong id. | Have the op return the new id, or pick the new reference whose pictures match the arguments. |
| i1 | info | (whole app) | Browser back and forward do not apply: the app has no history routing (no `pushState` anywhere), so mouse back buttons do nothing. This was also true before. | None needed now; log it if wanted. |

Checked and fine: the format 2 to 3 migration; a bundle with at most one picture per reference is written as
format 2 (`bundle.py:76-80`), so an older app can import it; a format 3 bundle is refused with "Update the app"
(`store.py:115-116`); a single-picture fingerprint is unchanged, so takes do not go stale on upgrade
(`deps.py:31-35`, tested); format 3 undo and redo restore the pictures and the fingerprint; reference paths are
capped at 4, matching `ImageGenerationRequest.referenceImagePaths`; a project still as a source goes through
`media.inside` (no `..`, stays in the folder, symlinks resolved); upload handles are random and the type comes
from the bytes; deletion and rename go through the existing ops and confirm; the stored route parses safely and
migrates the old key once; a deleted stored project shows "This project is not there any more" with Back;
gallery sends open the right project on the right tab (pictures to Storyboard, sound to Mixer); no emoji, no
other products named; nothing requires a character (Overview, an empty
Storyboard, the drawer and the templates all say a cast is optional).

## Keep-list

Legend: KEPT = same control, MOVED = same control in a new place, REPLACED = same capability, new control,
MISSING = gone. Lines refer to the worktree.

| Old control or capability (base) | Status | New place |
|---|---|---|
| Project card: thumbnails, shots, seconds, aspect, stale, edited, missing-folder note | KEPT (plus resolution and fps) | `ProjectCard.tsx:44-111`, missing `:31-42` |
| New project: name, aspect, resolution, frame rate with fit hints | KEPT, extended (templates, models) | `ProjectDialog.tsx`, `ProjectListTab.tsx:67-99` |
| Open | REPLACED (click the thumbnails button) | `ProjectCard.tsx:55-74` |
| Export bundle | MOVED (card menu and Export tab) | `ProjectCard.tsx:81-86`, `ExportTab.tsx:160-174` |
| Import bundle | KEPT, plus drop on the list | `ProjectListTab.tsx:169-187,202-210` |
| Rename | KEPT, plus click the name in the header | `ProjectCard.tsx:80`, `ProjectShellHeader.tsx:37-87` |
| Delete with confirm | KEPT | `ProjectListTab.tsx:316-333` |
| List states: offline, loading, empty, error, import failed, "Exported X · size" | KEPT | `ProjectListTab.tsx:212-283` |
| Board header facts (shots · seconds · stale) | MOVED | `StoryboardTab.tsx:153-165` |
| Undo, Redo | MOVED to the shell header | `ProjectShellHeader.tsx:153-158` (Mixer Cmd+Z kept, `MixerTab.tsx:393-396`) |
| "Changed elsewhere" chip | MOVED | `ProjectShellHeader.tsx:140-152` |
| Projects and Mixer buttons (Board), Board and Projects buttons (Mixer) | REPLACED by the back link and the tab row | `ProjectShellHeader.tsx:122-125,161-192` |
| Project defaults: aspect, resolution, fps, image model, video model | MOVED to Overview > Project settings | `OverviewTab.tsx:97-105`, `ProjectDefaultsPanel.tsx` |
| Voice (free text) | REPLACED by the voice picker; free text not reachable in the UI (m2) | `ProjectDefaultsPanel.tsx:102`, `VoicePicker.tsx` |
| References: name, description, image path, remove | MOVED to Cast, extended to 4 pictures | `CastTab.tsx`, `ReferenceCard.tsx:134-171`; path at `ReferenceImagePicker.tsx:234-242` |
| Add shot | KEPT, plus a ghost card | `StoryboardTab.tsx:166,272` |
| Shot card (title, thumbnail, status, takes, Lock, Redo, transition, menu, drag) | KEPT (unchanged `ShotCard.tsx`) | `StoryboardTab.tsx:228-268` |
| Stale Keep and Re-run, Failed Retry, Missing Relink | KEPT | `StoryboardTab.tsx:250-256` |
| Make missing, Make all unlocked | KEPT (the run bar is phase 3) | `StoryboardTab.tsx:186-205` |
| Estimate strip | KEPT (still empty; phase 3) | `StoryboardTab.tsx:179` |
| Batch bar with Cancel | KEPT | `StoryboardTab.tsx:152` |
| Keyboard: arrows, Enter, L, R, Delete | KEPT | `StoryboardTab.tsx:112-147` |
| Delete-shot confirm | KEPT | `StoryboardTab.tsx` (`removeId` with ConfirmDialog) |
| States: no project, not found, offline, loading, error with Reload | MOVED to the shell | `ProjectShell.tsx:55-76` |
| Shot editor drawer, every field, the references checkbox list | KEPT (chips with thumbnails and the shared picker not done yet) | `ShotEditorDrawer.tsx`, `StoryboardTab.tsx:280` |
| Mixer subtitle (seconds · aspect · preview size) | MOVED under the tab row | `MixerTab.tsx:474-482` |
| Preview: Render, Out of date, Rendering, did not render, Download | KEPT | `MixerTab.tsx:515-560` |
| "n shots have no clip" notice | KEPT | `MixerTab.tsx:566` |
| Transport, lanes, timeline, drag, trim, waveform | KEPT | `MixerTab.tsx:608-640` |
| Picture block opens the shot | KEPT (now opens it focused on the Storyboard tab) | `MixerTab.tsx:597,648`; `ProjectShell.tsx:49-52` |
| Add audio per lane (library, file, path, cloned tick) | KEPT | `MixerTab.tsx:654` |
| Clip inspector, Relink | KEPT | `MixerTab.tsx:577,633,667` |
| Captions panel | KEPT | `MixerTab.tsx:678` |
| Export panel: aspect with size, loudness, SRT, burn-in, stems, AI label, Export MP4 with phases and Cancel, Resume, result (Reveal, Copy path, Also written), .otio, films list | MOVED to the Export tab (the result card is lost after a preview, M3) | `ExportTab.tsx:131-159`, `mixer/ExportPanel.tsx` |
| Galleries "Use in storyboard" and Voice and Music "Send to storyboard" | KEPT; open the project on Storyboard or Mixer | `useSendToStoryboard.ts:123-124`, `sendToStoryboardLogic.ts:63` |
| Galleries "From project X" chip | KEPT (opens the Storyboard tab) | `ImageGalleryTab.tsx:289`, `VideoGalleryTab`, `VoiceLibraryTab`, `MusicLibraryTab` |
| Agent tools `storyboard_*` | KEPT, plus add_ref, edit_ref and remove_ref | `storyboard_tools.py:22-36,254-272` |
| Folder layout, bundles, limits | KEPT; format 3 only when a reference has more than one picture | `bundle.py:76-80`, `store.py:37-76` |

Not built yet, as planned for later phases (not blockers): run bar with estimate, preflight, next-step guide and
checklist, recent activity, "Use still as reference", reference chips with thumbnails in the drawer, the picker
for "Start from: my image", Cmd+K, Cmd+N, Cmd+[ and Cmd+].
