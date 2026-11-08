# Project workspace rework (D-65): design

Date 2026-11-08. Status: proposal for the owner. Mockups: `mockups.html` in this folder (open in a browser; no
network needed). Research notes informed this but are not cited: patterns are described generically.

## 1. Recommendation in short

- **One sidebar entry, "Projects"** (section Make, after Music and Audio). It opens the project list. The three
  items Projects, Board and Mixer go away; their ids redirect.
- **A project is a place, not a mode.** Opening a project swaps the whole content area for a project shell: a
  header with a back link to the list, the project name, Undo and Redo, and a tab row. The shell keeps its own
  route (`project id + tab`) in local storage, so the sidebar entry reopens where you left off.
- **Five tabs in pipeline order: Overview, Cast, Storyboard, Mixer, Export.** Each works on that project only.
  Nothing in the app acts on a hidden "current project" any more.
- **The Board becomes the Storyboard tab and shows only shots.** Project defaults move to Overview, collapsed
  after setup. References move to the Cast tab with a real picker. The run estimate moves onto the Make button in
  a sticky run bar at the bottom.
- **Cut:** the Projects and Mixer buttons in the Board header, the Board and Projects buttons in the Mixer
  header, the raw `/path/to/file` box as the main way to set a reference image, the blank Voice text field, the
  estimate strip in the middle of the page, the Export panel inside the Mixer (it becomes the Export tab).
- **Everything the app can do today stays.** Section 2 maps every control to its new place.

Why this shape: the owner's brief (one entry, pick a project, tabs inside) matches what every mature editing
tool does: a project manager screen, then a workspace whose tabs follow the order of the work (gather, write,
make, mix, deliver). The order itself teaches a first-time user what to do next.

## 2. Keep-list: every capability today and where it lives after

Legend: **same** = unchanged control, **moved** = same control, new place, **replaced** = same capability, new
control.

### 2.1 Projects screen (today `story-projects`)

| Today | After | Note |
|---|---|---|
| Card: thumbnails, shots, seconds, aspect, stale count, "edited when", missing-folder note | Project list card | same, plus resolution and fps in the meta line |
| New project (name, shape 16:9 / 9:16 / 1:1, fps 24 / 30) | New project dialog | replaced: aspect ratio, resolution, frame rate with fit hints (D-64), models, start-from template |
| Open | Click the card, or Enter | same |
| Export bundle (.chaosfilm download) | Card menu and Export tab | moved |
| Import bundle | List header, secondary button; also drop a .chaosfilm on the list | same |
| Rename | Card menu; also the name in the project header (click to edit) | same |
| Delete (confirm dialog, cannot be undone) | Card menu | same |
| States: offline, loading, empty, error, import failed, status line "Exported X · size" | Project list | same |

### 2.2 Board (today `story-board`)

| Today | After | Note |
|---|---|---|
| Header: "{project} · Board", shots · seconds · stale | Project header (name) + Storyboard tab subtitle | same facts |
| Undo, Redo | Project header, every tab | moved up; covers the same ops |
| "Changed elsewhere; showing the latest" chip | Project header | moved |
| Projects button, Mixer button | Back link "Projects" and the tab row | replaced |
| Project defaults disclosure (shape, fps, image model, video model, voice text) | Overview > Project settings (collapsed after setup) | moved; voice becomes a picker; resolution and more fps values added |
| References disclosure (characters, places: name, description, image path, remove) | Cast tab | moved; image set by picker; path kept under "Paste a path" |
| Add shot | Storyboard tab: button in the list head and a ghost card at the end | same |
| Shot card: number, editable title, thumbnail, play clip | Shot card | same |
| Status: Not made yet, Queued · n, Making, Done · s, Failed, Stale · after shot n, Take file missing | Shot card | same |
| Takes strip, pick a take | Shot card | same |
| Lock, Redo toggles | Shot card | same |
| Transition cut / crossfade + seconds | Shot card | same |
| Menu: Edit prompts, Import footage, Move up, Move down, Delete | Shot card menu | same, plus "Use still as reference" |
| Drag handle reorder | Shot card | same |
| Stale: Keep, Re-run; Failed: Retry (+ technical details); Missing: Relink | Shot card | same |
| Make missing, Make all unlocked | Run bar (sticky, bottom) | moved; estimate on the button |
| Estimate strip (time, memory, disk, basis; empty today) | Run bar | moved and filled (section 4.6) |
| Batch bar: "Making shots · Shot n of total · stills, then clips", Cancel batch | Run bar during a run | moved |
| Keyboard: arrows, Enter, L, R, Delete | Shot list | same, extended (section 6) |
| Delete-shot confirm (takes stay, Undo brings it back) | same dialog | same |
| States: no project, not found, offline, loading, error + Reload | Project shell handles them once for all tabs | moved |

### 2.3 Shot editor drawer

| Today | After | Note |
|---|---|---|
| Still prompt, Motion prompt | same | |
| Start from: a new still, my image, last frame of the previous shot | same | "my image" uses the same picker as Cast |
| Length (s), Seed + Random, Shot sound level (dB), Trim in, Trim out (End of take), Speed | same | |
| References (checkbox list of names) | Reference chips with thumbnails | replaced |
| Import footage: file, Image Gallery, Video Gallery, Use as clip | same | |
| Relink (missing take) | same | |
| Save changes, Make this shot, Revert, Close, Discard confirm | same; Make this shot shows its own estimate | |

### 2.4 Mixer (today `story-mixer`)

| Today | After | Note |
|---|---|---|
| Header "{project} · Mixer", seconds · aspect · preview size; Projects, Board buttons | Project header + tab row | replaced |
| Preview video, Render preview, "Out of date", "Rendering preview…", "did not render", Download | Mixer tab | same |
| "n shots have no clip yet and are left out" notice | Mixer tab (and Export preflight) | same |
| Transport: Play, Pause, Stop, Loop, zoom, keyboard hint | same | |
| Lanes Picture, Shot sound, Voice, Effects, Music (+ import clips) | same | |
| Picture block click opens the shot on the Board | opens the Storyboard tab with that shot focused | same |
| Add audio per lane: Add from library (voice), Import audio (file chooser or path), cloned-voice tick | Add audio sheet: From library, Choose file, Paste a path (collapsed), cloned-voice tick | same, path demoted |
| Drag to move, drag ends to trim, waveform | same | |
| Inspector: Name, Gain, Fade in/out, Trim in/out, Duck under voice, Loop to fill, Anchor (shot + offset or film time), Remove from lane | same | |
| Selected shot: Shot sound level, Open on Board | same, "Open in Storyboard" | |
| Missing clip: Relink | same | |
| Captions panel: Make from voice, list, edit text, Remove, Add at playhead, Write .srt, Trimmed away | Mixer tab, "Captions" disclosure under the lanes | same |
| Export panel (everything in 2.5) | Export tab | moved |
| Undo, Redo, conflict chip, offline, loading, not found | Project shell | moved |

### 2.5 Export (today inside the Mixer)

| Today | After | Note |
|---|---|---|
| Shape 16:9 / 9:16 / 1:1 with pixel size, "Reframed at export; shots are not regenerated." | Aspect ratio, plus Resolution with an upscale note (D-64) | extended |
| Loudness target (LUFS, empty = as is) | same | |
| Captions: SRT file, Burn in | same, with the caption count and "Edit in Mixer" | |
| Audio lanes as separate files | same | |
| AI label: machine-readable tag on by default; forced on with a cloned voice | same | |
| Export MP4; phases Normalise shots, Join, Mix sound, Encode; Cancel export | same, with an estimate on the button | |
| Interrupted / did not finish, "n of total already done", Resume export | same | |
| Result: size · duration, Reveal file, Copy path, Also written (.srt, stems) | same | |
| Export for an editor (.otio), warnings | same | |
| Film list (exports, newest first, Reveal) | same, "Films" | |
| Export bundle (.chaosfilm) | added here too | |

### 2.6 Cross-area and agents

| Today | After | Note |
|---|---|---|
| Use in storyboard (Image and Video galleries): 3 recent projects, New project…, Start a storyboard with this | same menu; lands on the project's Storyboard tab | same |
| Send to storyboard (Voice Library, Speak, Music Studio and Library): lands in the Mixer | same; lands on the project's Mixer tab | same |
| Galleries: "From project X" chip and Project filter | same | |
| Agent tools `storyboard_project`, `_shot`, `_audio`, `_generate`, `_render`, `_captions`, `_export_otio` | unchanged | the UI shows their changes through the existing conflict chip |
| Project folder layout, bundles, limits (500 shots, 500 clips, 50 takes) | unchanged | |

Nothing is dropped. Two things are demoted (the path box, the voice text) and stay reachable.

## 3. Flow and information architecture

```
Sidebar: Make > Projects
  └ Project list  ───────────── search, sort, cards, New project, Import bundle
       └ Project shell (one project)
            header: [< Projects]  Name (click to rename)  [Undo] [Redo]  [conflict chip]
            tabs:   Overview | Cast | Storyboard | Mixer | Export
```

- **Overview**: what to do next, progress, the shots at a glance, project settings (collapsed), recent activity.
- **Cast**: characters and places, each with a name, a description and a picture. Set once, used by shots.
- **Storyboard**: the shots, their takes, the run bar.
- **Mixer**: preview, lanes, captions.
- **Export**: preflight, settings, render, films, bundle.

Where a project opens: the tab you last used in it; a new project opens on Overview; a send from a gallery
opens Storyboard; a send from Voice or Music opens Mixer. The sidebar entry opens the last project if one was
open, otherwise the list (the same "ready" rule the sidebar already has for Voice and Music).

The tab row is the project's own, not the app's sub-tab bar. The app's `SubtabBar` renders nothing for the
Projects entry. On narrow screens the project tabs scroll horizontally and the sidebar is the existing drawer.

## 4. Screens

Common states, handled once by the shell (every tab inherits them):

| State | What shows |
|---|---|
| Backend offline | Centered "Waiting for the backend…" with the header still visible |
| Loading | Header with the name from the list, then skeleton rows in the tab |
| Not found (folder moved or deleted) | "This project is not there any more." [Back to projects] |
| Error | Callout with the message, [Reload] |
| Conflict (another window or an agent changed it) | Chip "Changed elsewhere; showing the latest" with Dismiss, in the header |
| Busy machine (another heavy job, a batch or a render running) | The run bar says "Another job is running. Yours starts after it." and Make stays enabled; the estimate adds the wait |
| Batch running | Run bar shows progress; shot cards show Queued · n / Making; the Overview guide step 4 shows "Making…" |
| Stale shots | Cards show Stale · after shot n with Keep / Re-run; the run bar counts them; Export preflight lists them |

### 4.1 Project list (`storyboard` tab, no project open)

Purpose: find and open a project; make or import one. Primary action: **New project**.

| Control | Behaviour | States |
|---|---|---|
| Title "Projects", subtitle "{n} projects" | | |
| Search field | Filters by name as you type; `/` focuses it; Esc clears | empty result: "No project matches {query}." |
| Sort menu: Last edited (default), Name, Created, Length | Remembered in local storage | |
| New project (primary) | Opens the dialog (4.2) | |
| Import bundle (secondary) | File chooser for .chaosfilm; the list is also a drop target | importing: busy; failed: callout "This file is not a ChaosEngineAI project bundle." + technical details; done: status line "Imported {name}" |
| Card: poster (first done shot or last thumbnails), name, meta line "8 shots · 32 s · 16:9 · 1080p · 24 fps", stale badge "2 stale", "edited 2 h ago" | Click or Enter opens the project on its last tab | missing folder: card greyed, note "This project's folder was moved or deleted.", only Delete in the menu |
| Card menu (More actions): Rename, Export bundle, Delete | Rename opens the name dialog; Export downloads and shows "Exported {file} · {size}"; Delete confirms "{name} and every take, still and sound in its folder will be deleted. This cannot be undone." | exporting: "Exporting…" |
| Empty state | "No projects yet. A project holds your shots, takes and sound in one folder you can move or back up." [New project] [Import bundle] | |
| Loading | "Loading your projects…" | |
| Offline | "Waiting for the backend…" | |

Wide screens: 3 to 4 cards per row. 375 px: one card per row, the search field full width, New project full
width under the title.

### 4.2 New project dialog

Purpose: name the film and fix the picture format once. One screen, no wizard. Everything can be changed later
on Overview.

| Control | Behaviour |
|---|---|
| Name | Required; autofocus |
| Start from: Blank, 30 s ad, Vertical social, Short story | Pills. A template pre-fills aspect, frame rate and three empty shots with example prompts in the title. Blank is default. (Phase 2; cheap: client-side data only.) |
| Aspect ratio: 16:9, 9:16, 1:1 | Pills; the 9:16 and 1:1 sizes follow the chosen resolution |
| Resolution: 480p, 720p (default), 1080p, 1440p, 4K | Pills showing the pixel size for the chosen aspect and a fit hint under each: **Fits** (green), **Tight** (amber), **Will not fit** (red, still selectable), or "Not measured yet" (grey). Hint source: the same memory estimator the Video Studio uses for the chosen video model, plus the model's own maximum size. A size above the model's maximum reads "Made at {max}, upscaled at export" (D-64). |
| Frame rate: 24, 25, 30, 48, 50, 60 | Pills; a rate the chosen video model cannot make natively shows "Resampled from {native}". |
| Models (collapsed, summary line "Automatic · Automatic") | Image model, Video model selects, prefilled from the Studios' current selection. Changing the video model recomputes the hints. |
| Create (primary), Cancel | Create opens the project on Overview |

Never capped by this machine: every size and rate stays selectable; the hint is advice (D-64).

### 4.3 Project shell

| Element | Behaviour |
|---|---|
| Back link "Projects" (chevron + label) | Returns to the list; the project stays "last open" |
| Name | Click or Enter turns it into a field; Esc cancels; blur saves via `edit_project` |
| Meta chips: 16:9 · 1080p · 24 fps | Click opens Overview > Project settings |
| Undo, Redo | As today; disabled when the stacks are empty or the project is busy |
| Conflict chip | As today |
| Tab row (role tablist): Overview, Cast, Storyboard, Mixer, Export | Left/Right arrow keys move between tabs; the active tab is remembered per project. Count badges: Cast shows the reference count, Storyboard the stale count (amber) when > 0, Export a red dot when the last export was interrupted |

### 4.4 Overview tab

Purpose: tell the user what to do next and show the whole film at a glance. No editing of shots here.

| Block | Content | States |
|---|---|---|
| Next step card | One primary action computed from data: "Add your cast" (no references, no shots) → "Write your shots" (no shots) → "Make 3 shots · about 14 min" (missing takes) → "Pick a take for shot 4" (a shot has several takes and none active) → "Add voice or music" (no audio) → "Render a preview" (no preview or out of date) → "Export the film" → "Done. Export again or make a new take." Each step is a button that opens the right tab. Under it a 7-step checklist with ticks (Name and format, Cast, Shots, Takes, Sound, Preview, Export); hidden once all are ticked; a "Show guide" link in Project settings brings it back. | busy: the Make step reads "Making shot 3 of 8…" with Cancel |
| Progress strip | "8 shots · 6 made · 2 stale · 32 s · voice, music · last preview 2 h ago" | |
| Contact sheet | Thumbnails of every shot in order with the number; click opens that shot in the Storyboard tab (focused); stale and missing shots carry a small badge | empty: "No shots yet." [Write your shots] |
| Project settings (Disclosure, collapsed when name, format and at least one shot exist) | Aspect ratio, Resolution with fit hints, Frame rate, Image model, Video model, Voice (picker, 4.9), "Show guide" | changing aspect, resolution or a model after takes exist marks dependent shots stale, as today |
| Recent activity | Last 3 lines from batches and renders: "Batch of 3 shots finished · 2 h ago", "Preview rendered", "Export interrupted at shot 5" | |
| Folder | "Project folder · 640 MB" with [Export bundle] | |

### 4.5 Cast tab

Purpose: define the people and places once so every shot keeps the same look.

| Control | Behaviour | States |
|---|---|---|
| Two groups, Characters and Places, each with "Add character" / "Add place" | Adds a card with the name field focused | empty: "None yet. A character or place is a name, a few words and a picture. Shots that use it keep the same look." |
| Reference card: picture (square thumb), Name, "What it looks like" (2-row textarea), "Used by n shots", Remove | Fields save on blur, as today | no picture: dashed placeholder "Add picture"; picture changed since a take: the shots that used it show Stale (exists) |
| Picture slot: click, or drop an image on the card, or paste (Cmd+V) with the card focused | Opens the picker (4.9) | small image (< 512 px short side): note "Small picture; results may drift." |
| Remove | Removes the reference; shots that used it lose the chip (Undo brings it back) | |
| Drop zone for the whole tab | Dropping an image file on empty space adds a character named after the file | |

Kept out: multi-image references (front, side, detail). The data model holds one image per reference. Open
question 2.

### 4.6 Storyboard tab

Purpose: write, make and arrange the shots. The page is only the list plus the run bar.

| Control | Behaviour |
|---|---|
| List head: "Shots" with "8 shots · 32 s · 2 stale", [Add shot], [Paste a list] | Paste a list opens a small sheet: one line per shot; each line becomes a shot title and still prompt (phase 4) |
| Shot card | As today (2.2). Status pill, takes strip, Lock, Redo, Transition, More actions. New menu item "Use still as reference" adds a Cast entry with the active take's still |
| Ghost card at the end: "Add shot" | Same as the button; keeps the hand at the bottom of a long list |
| Length strip under the list head | A thin bar proportional to shot lengths with the numbers; click a segment to focus that card (phase 4; cheap) |
| Run bar (sticky at the bottom of the tab, above the window edge) | Left: need line "3 shots need a take · 2 stale". Middle: estimate "About 14 min · Peak 9 GB of 64 GB · Writes 120 MB" with the Measured / Estimated chip, grey "Time: unknown until the first run" when nothing is measured. Right: [Make all unlocked] (secondary, tooltip "Adds a new take to every unlocked shot. Nothing is overwritten.") and the primary **Make 3 missing · about 14 min**. |
| Run bar while running | "Making shots · Shot 3 of 8 · stills, then clips" with the meter, [Cancel batch] |
| Run bar when memory will not fit | Red border; "Run anyway. I accept that the backend may crash." tick before Make enables, as in the Studios |
| Run bar when another job runs | "Another job is running. Yours starts after it." |
| Run bar when every shot has a take | "Every unlocked shot has a take." Make missing disabled; Make all unlocked enabled |
| Drawer (Edit prompts) | As today; references as chips with thumbnails; "Make this shot · about 2 min" |

Estimate basis: a new `POST /api/storyboard/projects/{id}/estimate` sums per-shot still and clip estimates from
the same helpers the Image and Video Studios use (time from measured runs on this machine when available,
memory from the model card), plus the model load. Until the first measured run it says Estimated. Nothing new in
the estimator maths; it is a sum.

### 4.7 Mixer tab

As today minus the Export panel. Changes:

- The header line "{seconds} s · {aspect} · preview {w} × {h}" sits under the tab row.
- Add audio on the Voice lane: **From Voice Library**, **Choose file**, "Paste a path" (collapsed). Phase 4 adds
  **Speak a line** (text field, the project voice, Speak, lands on the lane) using the existing speak endpoint.
- Add audio on Music and Effects: **From Music and Audio Library** (when the library is available), **Choose
  file**, "Paste a path" (collapsed).
- Captions panel stays here as a disclosure under the lanes ("Captions (12)"), because captions follow voice
  clips and the playhead; Export shows the count and the options.
- Picture block click opens the Storyboard tab with that shot focused.

### 4.8 Export tab

Purpose: deliver. One screen, in order: check, choose, render, collect.

| Block | Content |
|---|---|
| Preflight | Lines with a tick or a warning: "8 of 8 shots have a clip" / "2 shots have no clip and will be left out"; "2 shots are stale" [Open in Storyboard]; "12 captions from 3 voice clips" / "No captions"; "Preview is out of date" (informational); "A cloned voice is used; the AI tag is always written". |
| Format | Aspect ratio pills (reframed at export, shots not regenerated); Resolution pills with the project size marked "project" and larger sizes marked "Upscaled" (D-64); Loudness target |
| Extras | Captions: SRT file, Burn in; Audio lanes as separate files; AI label (tag on by default; forced with a cloned voice) |
| Action | **Export MP4 · about 3 min** (primary), estimate chip; during: phases Normalise shots, Join, Mix sound, Encode with [Cancel export]; interrupted: "The export was interrupted. 5 of 8 shots are already done and will be reused." [Resume export] |
| Result | "Exported · 48 MB · 32 s" [Reveal file] [Copy path]; "Also written: film-captions.srt, stem-voice.wav…" |
| Other exports | [Export for an editor (.otio)] with the warnings list; [Export bundle (.chaosfilm)] |
| Films | The exports list, newest first, with Reveal |

### 4.9 Pickers

**Reference image picker** (used by Cast, by "Start from: my image" in the drawer, and by Import footage for
stills). A popover anchored to the slot, or a sheet at 375 px.

| Source | Control |
|---|---|
| Choose file | Native chooser (png, jpg, webp). The picker shows the thumbnail at once; the file is copied into the project by the existing op |
| Drop or paste | The slot and the popover are drop targets; Cmd+V with the slot focused |
| Image gallery | Thumbnail grid with a search field; the gallery's Project filter preset to this project; click picks |
| From a shot | Grid of this project's takes (active still first) labelled "Shot 3 · take 2"; click picks |
| Paste a path (collapsed link at the bottom) | The existing PathField; kept for agents' outputs and remote folders |

**Voice picker** (Overview > Project settings > Voice; later Speak a line). A popover with a search field and
four groups from `GET /api/voice/voices`: Built-in, Preset, Designed, Cloned. Each row: name, language, a
**Play** button (speaks a fixed sample sentence through the speak endpoint; cached per session; shows a stop
state while playing), and a note when `available` is false ("Needs the {model} model"). A cloned voice row
carries a "Cloned" badge and picking it tells the user the AI tag will be forced on voice clips. "No voice
(automatic)" is the first row.

## 5. What was cut, and why

| Cut | Why |
|---|---|
| Three sidebar items | They were two views of one thing plus a list; the project shell makes the relationship visible |
| Board and Mixer header buttons to each other and to Projects | The tab row and the back link do it |
| Estimate strip between defaults and shots | It is advice about one button; it now sits on that button |
| Project defaults at the top of the Board | Set once, read daily; collapsed on Overview |
| References on the Board | A library, not a per-shot control; now a tab with a picker |
| Blank Voice text field | Replaced by the picker over the real voice list |
| Path box as the main image input | Kept as a collapsed "Paste a path" for agents and remote folders |
| Export panel inside the Mixer | Export is the last step and has its own checks; a crowded Mixer hid it |
| Duplicate project, Reveal folder, Purge cache | No ops exist for them; not in the keep-list; log as follow-ups if wanted |
| A node canvas, a script importer | Out of scope; the linear board is right for a first-time user |

## 6. Keyboard flow

| Where | Keys |
|---|---|
| Anywhere in the Projects area | `/` focuses search; `Cmd+N` new project; `Cmd+K` project switcher (type, Enter opens) |
| Project list | Arrows move between cards, Enter opens, `Cmd+Backspace` asks to delete |
| Project shell | `Cmd+[` / `Cmd+]` previous / next tab; `Cmd+Z` / `Shift+Cmd+Z` undo / redo (exists); Esc closes a drawer or popover; `?` shows the shortcut sheet |
| Storyboard list (exists) | Arrows, Enter opens the editor, L lock, R redo, Delete asks to delete |
| Storyboard list (new) | N adds a shot after the focused one; `Cmd+Enter` makes the focused shot; `Shift+Cmd+Enter` makes missing; `[` / `]` previous / next take; `Alt+Up` / `Alt+Down` move the shot; Space plays the clip |
| Mixer (exists) | Space or K play, J and L 5 s, arrows 1 s (Shift 10 s, Alt one frame), Home, End, + and - zoom |
| Mixer (new) | M mutes the selected clip's lane in the preview (phase 4); `Cmd+R` renders the preview |
| Cast | Tab order: picture, name, description, remove; Enter on the picture opens the picker; `Cmd+V` pastes a picture |

The app has no global `Cmd+digit` shortcuts today, so `Cmd+1..5` for the project tabs is free; it is listed as
optional because browser shells on Windows and Linux use `Ctrl+digit` for window tabs.

## 7. Small screens and accessibility

- 375 px: the project header wraps (back link and name on one line, Undo/Redo and chips on the next); the tab row
  scrolls horizontally with the active tab scrolled into view; shot cards are one column with the thumbnail on
  top; the run bar stacks (need line, estimate, buttons full width); pickers open as bottom sheets; the Mixer
  keeps its lanes with horizontal scroll and the inspector becomes a sheet.
- Every tab row is `role="tablist"` with `aria-selected`, arrow-key movement and a `tabpanel` with a heading.
- The run bar's estimate is `aria-live="polite"` and the batch progress `role="status"`; the "will not fit" state
  is announced as text, not only colour.
- Pickers are dialogs with focus trap and Esc; drop zones also have a button; the voice Play button has an
  accessible name "Play sample of {voice}".
- Labels are short nouns (Overview, Cast, Storyboard, Mixer, Export) to survive translation; counts use ICU
  plurals as the existing keys do.
- Icons are monochrome SVG from `components/icons`; no emoji; sentence case throughout.

## 8. Agents and chat (D-60)

No tool changes. The design keeps every mutation on the existing ops, so a chat model using `storyboard_*`
tools produces exactly what the UI produces. Two small UI additions make agent-driven use visible:

- The project list refreshes when the tab becomes visible and every 10 s while it is visible, so a project an
  agent just created appears without a reload.
- The shell's conflict chip already covers edits; the Overview "Recent activity" lines name the source when a
  batch record carries one ("Batch of 3 shots started by an agent").

Later (not in this plan): a "Plan" preview that an agent fills before a render, the one-screen approval D-60
asks for, would be a sixth tab or a sheet on Overview.

## 9. Left-menu migration and redirects

| Today | After |
|---|---|
| Group `storyboard` with children `story-projects`, `story-board`, `story-mixer` | Standalone tab `storyboard`, label "Projects", caption "Your films: shots, sound, export", section Make |
| `readyChild: story-board` | The view itself decides: last open project → shell, else list |
| `localStorage` key for the open project id | `storyboard.route` = `{ projectId, tab }`; the old key is read once and migrated |
| `lastChildByGroup.storyboard` | Dropped by `migrateLastChildByGroup` (no group) |

Redirects (`tabRedirects.ts`): `story-projects` → `storyboard` (list); `story-board` → `storyboard` with
`storyboard: { tab: "storyboard" }` (last project, or the list when none); `story-mixer` → `storyboard` with
`{ tab: "mixer" }`. `TabNavigation` gains an optional `storyboard?: { projectId?: string; tab?: ProjectTabId }`
so galleries' "Use in storyboard" and the Dashboard "Continue" tile can deep-link.

## 10. Code changes by file

Frontend

| File | Change |
|---|---|
| `src/types/index.ts` (TabId) | add `"storyboard"`; the three old ids move to `LegacyTabId` |
| `src/constants/tabs.ts` | one row `storyboard` (label Projects); `TabConfig` gains `section?: "make" \| "run"` for a standalone tab inside a section |
| `src/constants/sidebarGroups.ts` | remove the `storyboard` group |
| `src/constants/features.ts` | `HOME_STUDIO_TABS` = music ids + `storyboard`; `HOME_STUDIO_GROUPS` = `["music"]`; add `HOME_STUDIO_STANDALONE_TABS` if the sidebar needs it |
| `src/constants/tabRedirects.ts` | legacy ids and the `storyboard` navigation field (section 9) |
| `src/components/Sidebar.tsx` | place standalone tabs with a `section` inside that section; `tabs` mode shows the entry as a plain button |
| `src/components/SubtabBar.tsx` | no change needed (no group → null) |
| `src/app/tabViews.ts` | `storyboard: StoryboardView` |
| `src/app/StoryboardTabs.tsx` → `src/app/StoryboardView.tsx` | the switch between list and shell; reads `storyboard.route` and the navigation request |
| `src/features/storyboard/projectRoute.ts` (new) | route type, read/write/migrate; replaces `readOpenProjectId` / `writeOpenProjectId` (re-exported for one release) |
| `src/features/storyboard/ProjectShell.tsx` (new) | header, name edit, Undo/Redo, conflict chip, tab row, common states; owns `useStoryboard(projectId)` |
| `src/features/storyboard/useStoryboard.ts` | take `projectId` as an argument instead of reading storage |
| `src/features/storyboard/ProjectsTab.tsx` → `ProjectListTab.tsx` | search, sort, drop-to-import, visibility refresh |
| `src/features/storyboard/ProjectDialog.tsx` | aspect, resolution with hints, frame rate, models, templates (`newProjectDefaults(aspect, resolution, fps)`) |
| `src/features/storyboard/storyboardLogic.ts` | `ASPECT_RATIOS`, `RESOLUTIONS`, `FPS_CHOICES = [24, 25, 30, 48, 50, 60]`, `sizeFor(aspect, resolution)`; `EXPORT_PRESETS` kept as an alias for the Export tab |
| `src/features/storyboard/OverviewTab.tsx` (new) | next step, checklist, progress, contact sheet, settings disclosure (moves `ProjectDefaultsPanel` in), activity |
| `src/features/storyboard/CastTab.tsx` (new, from `ReferencesPanel.tsx`) | cards, drop zone, used-by counts |
| `src/features/storyboard/ReferenceImagePicker.tsx` (new) | the four sources + path; reused by the drawer |
| `src/features/storyboard/VoicePicker.tsx` (new) | groups, search, Play sample |
| `src/features/storyboard/BoardTab.tsx` → `StoryboardTab.tsx` | drop defaults and references; add the run bar and the ghost card |
| `src/features/storyboard/RunBar.tsx` (new, absorbs `BatchBar.tsx`) | need line, estimate, Make buttons, progress, ack |
| `src/features/storyboard/useProjectEstimate.ts` (new) | calls the estimate endpoint, debounced on project changes |
| `src/features/storyboard/ShotCard.tsx` | menu item "Use still as reference" |
| `src/features/storyboard/ShotEditorDrawer.tsx` | reference chips; per-shot estimate on Make this shot |
| `src/features/storyboard/mixer/MixerTab.tsx` | remove the Export panel and the header buttons; AddAudioPanel sources |
| `src/features/storyboard/ExportTab.tsx` (new, from `mixer/ExportPanel.tsx` + `FilmList.tsx`) | preflight, resolution, bundle |
| `src/features/storyboard/useSendToStoryboard.ts` | open via the route (`{ projectId, tab }`) |
| `src/locales/*/studio.json`, `common.json` | new keys with `null` placeholders outside `en` |
| `src/features/storyboard/__tests__/*` | route, redirects, run bar states, picker sources, dialog hints |

Backend

| File | Change |
|---|---|
| `backend_service/routes/storyboard.py` | `POST /projects/{id}/estimate` (phase 3) |
| `backend_service/storyboard/estimate.py` (new) | sum of per-shot estimates using the Studios' helpers |
| `backend_service/storyboard/models.py` | none for sizes (4096 cap already allows 4K) and fps (60 cap) |
| `backend_service/mcp_bridge/storyboard_tools.py` | none |

Tests, docs, hub

| File | Change |
|---|---|
| `e2e/tests/helpers.ts` | `goTo(page, "Projects")`; `openProject(page, name, tab?)` clicks the card and the tab |
| `e2e/tests/05-storyboard.spec.ts`, `09-mobile.spec.ts` | new navigation; screenshots `storyboard-projects`, `storyboard-overview`, `storyboard-shots`, `storyboard-mixer`, `storyboard-export` |
| `scripts/e2e_test_suite.py` | add a phase check for the estimate endpoint |
| `docs/usage/storyboard.md`, `docs/usage/studio-layout.md` | rewrite for the shell |
| `docs/hub/decisions.md` | D-65 decision and the open questions below |

## 11. Build phases and risks

| Phase | Scope | Verify | Risks |
|---|---|---|---|
| 1. Shell | One sidebar entry, list with search and sort, shell with Overview (settings only), Storyboard (defaults and references gone), Mixer (export panel kept for now), redirects, route storage, send-to-storyboard, e2e | vitest for redirects and route; e2e 05 and 09 pass; 375 px screenshots looked at | stale stored tab ids; galleries' menu opening the wrong tab; a11y of the new tab row |
| 2. Cast and setup | Cast tab with the picker, voice picker, new-project dialog with D-64 choices and hints, settings collapsed on Overview | component tests for picker sources and hints; e2e creates a 1080p 9:16 project and adds a character from a shot still | hint source for models without a card; the voice sample playback needs a TTS model installed (show "Needs the model" instead of failing) |
| 3. Run bar and Export | Estimate endpoint, run bar with the estimate on Make, Export tab with preflight and resolution upscale, Mixer loses the panel | pytest for the estimate sum; e2e export flow unchanged; estimate within the Studios' own numbers | estimate wrong before the first run (say Estimated); the upscale path must exist in the render for sizes above the model maximum |
| 4. Guide and extras | Next-step card and checklist, contact sheet, recent activity, Paste a list, Use still as reference, length strip, Speak a line | e2e first-run walk: new project → guide steps tick in order | scope creep; keep each item under a day |

Phase 1 alone fixes the owner's three complaints (hidden current project, clunky defaults, estimate in the
middle). Phases 2 and 3 complete the brief. Phase 4 is the "maximise" list and can be trimmed.

## 12. Open questions for the owner (with recommendations)

1. **Sidebar label: "Projects" or "Storyboard"?** Recommendation: **Projects**. D-65 says "Projects in the menu
   opens a list", the galleries already say "From project X", and it avoids a Storyboard tab inside a
   Storyboard entry. The caption keeps the word: "Storyboard films: shots, sound, export".
2. **One picture per character, or up to four (front, side, detail)?** Recommendation: **one now**. The data
   model, the stale logic and the bridge tools all assume one image; a list is a model change that touches the
   bundle format. Add it as a follow-up once a video model that takes several references is wired.
3. **Fit hints before any run: show "Estimated" hints, or hide hints until the first measured run?**
   Recommendation: **show them**, labelled Estimated, from the model card and memory table, because the point of
   D-64 is to warn before a batch, and the first batch is the one that hurts.
4. **Templates in New project (Blank, 30 s ad, Vertical social, Short story)?** Recommendation: **yes, in
   phase 2**, as client-side data only (aspect, fps, three titled empty shots). It is the cheapest guided first
   run; it can be cut without touching anything else.
5. **Five tabs, or fold Export back into the Mixer (four)?** Recommendation: **five**. Export has its own checks
   and a long result block; inside the Mixer it competed with the timeline for height and was below the fold at
   375 px. The pipeline order Overview → Cast → Storyboard → Mixer → Export is also the guide's order.

## 13. Owner decisions (2026-11-08)

1. Sidebar label: **Projects**.
2. Characters take **up to four pictures** each (for example front, side, detail), not one. This is a data model change: the reference holds a list of images, old projects and bundles with one image read as a list of one, and the stale logic and the bridge tools follow the list. A project can also have **no named character at all**: a shot can stand alone (a busy street, a nightclub crowd), so Cast is optional everywhere, the guide never asks for a character, and the Make button never waits for one.
3. Fit hints are shown before any run, labelled Estimated.
4. Templates in New project: yes, in the second phase (Blank is the default and includes a no-character template such as a scene or b-roll set).
5. Five tabs: Overview, Cast, Storyboard, Mixer, Export.
