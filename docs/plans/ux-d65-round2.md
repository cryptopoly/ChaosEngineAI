# Projects workspace (D-65): UX and QA verification, round 2

Date 2026-11-08. Branch `feat/project-workspace`, worktree `agent-a0601f472d24874f1`, head `dcc4a821`. Read-only: nothing
edited or committed. Verified in a running build (Vite 8999, backend 9000, throwaway HOME, no models) at 1400 × 900 and
375 × 812 through the browser pane, screenshots looked at. Fresh first-time pass: "Petal & Stem" (30 s ad, 16:9, 1080p,
24 fps), character Maya with four pictures, place "Shop front", a second project "Busy street" (Scene / b-roll, no cast),
one gallery image sent to a shot, one still imported into a shot, rename, search, sort, delete.

Also run: `npx vitest run src/features/storyboard src/app src/constants` (13 files, 263 passed), `npx tsc --noEmit`
(clean), `pytest tests/test_storyboard_store.py tests/test_storyboard_routes.py tests/test_storyboard_tools.py
tests/test_storyboard_timeline.py` (78 passed). Not run: Playwright, the full e2e suite, the Tauri build.

Not tested: native file chooser, drag-and-drop, a real Cmd+V paste or "Paste from clipboard" with a picture on the
clipboard, a Make run, a render, Export MP4, Export bundle from the UI (it triggers a browser download; the endpoint
was exercised with curl instead and writes `exports/Petal---Stem-….chaosfilm`, 38 KB).

## Earlier P0 / P1 / blocker / major: status

| Item | Status | Evidence |
|---|---|---|
| UX P0-1 Cast description lost on Add picture | FIXED | Typed name + description, clicked Add picture from the description field: picker title "Add a picture of Maya", description intact after picking and after Cancel (character and place). `project.json` holds both descriptions. |
| UX P0-2 Shell wider than 375 px | FIXED | List, Overview, Cast, Storyboard, Mixer, Export at 375 px: `scrollWidth` 375, no element past the right edge, no panned box, cards one column, header wraps (chips and Undo/Redo on their own lines). See P2-1 below for an app-wide height issue that is not a regression. |
| UX P1-1 Overview has no next step | FIXED | "Next step: Make 6 shots" card with "Make 6 missing", 7-step checklist; Name and format and Shots ticked at creation, Cast ticked after adding Maya, "Cast (optional)" and "Sound (optional)" wording; the no-cast project shows the same card with Cast unticked and nothing blocking. |
| UX P1-2 Cards hide prompt and cast | FIXED | Every card shows the template prompt line; after ticking Maya in the drawer the card shows a "Maya" chip. |
| UX P1-3 Redo means two things | FIXED | Card toggle reads "New take" (tooltip "The next Make adds a new take of this shot. Nothing is overwritten."); header keeps Undo / Redo. |
| UX P1-4 "Done" on a still-only shot | FIXED (see P1-1 below) | Imported still: card says "Still only"; Overview "0 with a clip · 1 still only"; Mixer "6 shots have no clip yet". |
| UX P1-5 Gallery send menu cut off | FIXED | From a card at the bottom of the window the project menu and then the shot list open upward; six rows "1 · Hook (new take)" … all visible. Landing: Storyboard tab, shot 3 becomes "Still only". |
| UX P1-6 Export preflight | FIXED | "Before you export" with "No shot has a clip yet, so there is nothing to export. Open in Storyboard" and "No captions."; disabled Export MP4 has the helper "Nothing to export yet: no shot has a clip. Make clips on the Storyboard tab."; stems under "More". No "No films yet" line (no Films block at all when nothing was exported; P2). |
| UX P1-7 Create below the fold with Models open | FIXED | At 1400 × 900 with Models expanded the dialog grows to 740 px and Cancel / Create stay visible. |
| UX P1-8 List menus unnamed, empty state | FIXED | "More actions for Petal & Stem"; empty state has New project, Import bundle and "Or drop a .chaosfilm file here." |
| UX P1-9 Three "not estimated" lines | PARTLY | One chip "Estimate: unknown until the first run" on Storyboard, drawer and Export. The Storyboard button still reads "Make missing" without the count (Overview says "Make 5 missing", the subtitle "5 shots need a new take"). |
| QA B1 Undo after the format 3 upgrade | FIXED (by tests) | Commit 718212f4; store tests pass (78). Not reproduced by hand. |
| QA M1 Paste relies on a paste event | PARTLY | Picker has "Click here, then paste a picture (Cmd+V)" box and "Paste from clipboard". Not tried with a real clipboard; the card-focus paste path was not retested. |
| QA M2 Full-size grids | FIXED | Picker grid requests `files?…&thumb=256`, cast cards `thumb=512`, `loading="lazy"`. Each cast picture is requested twice when the picker opens (same URL, both 200): harmless, but worth a look. |
| QA M3 Render tracker per tab | NOT VERIFIED | One tracker lifted per commit fa7d690d; a render was not run (no model). |
| QA M4 Test gaps | FIXED (by count) | 263 vitest, 78 pytest pass, including the new picker, shell-state and upload tests. |

Minor items from round 1 that were checked: m2 voice id typeable (FIXED, "Type a voice id instead", `af_sky` resolves to
"Sky"); m6 header chips open Overview but Project settings stays closed (NOT FIXED); m12 ops.py split (FIXED); P2-1
banner names the open project (FIXED); P2-4 "Automatic · Automatic · voice none" (FIXED); P2-5 voice search (FIXED,
plus "Play sample" buttons); P2-6 short Cast groups (FIXED); P2-7 name focused on Add (FIXED); P2-8 remove buttons 25 px
(FIXED), small-picture note still wraps under the thumbnail; P2-9 picker wording (NOT FIXED: "Shot 2 · take 1" without
the title, second Choose file next to the path box); P2-10 contact sheet aspect (FIXED); P2-13 drawer title "Shot 1",
bare checkboxes with no accessible name (NOT FIXED); P2-14 header and tab row not sticky (NOT FIXED); P2-18 reload
returns to the open project and tab (FIXED).

## New findings

### P1

**P1-1. "Still only" shots are not "missing", so nothing makes their clip.** `missingShotIds` only takes shots with no
active take. After importing two stills, Overview says "Make 4 shots" and the subtitle "4 shots need a new take" while
the counter says "0 with a clip" and Export says nothing can be exported. The two still-only shots are left to "New take"
per card, which makes a new still rather than a clip for the one they have. Fix: count still-only shots in the next-step
card ("4 shots have no take, 2 have a still but no clip") and either include them in Make missing (clip from the
existing still) or give the card a "Make clip" action; say which in the checklist row "A clip for every shot".

**P1-2. Templates change the frame rate silently.** Picking "30 s ad" switched 24 fps to 30 fps and 720p to 1080p with
no caption or notice; the pill says "16:9 · 6 shots". A user who set the frame rate first ships a 30 fps ad. Fix: put the
format in the pill caption ("16:9 · 1080p · 30 fps · 6 shots") or apply the template's format only when the user has not
touched those rows.

### P2

**P2-1. 375 px: the whole app reads through a 250–300 px window (app-wide, not a regression).** The sidebar stacks
above the content and takes 410 px, the page header another 120 px, and `.content-grid` becomes a 268–298 px scroll box
with the rest of the viewport below it empty (measured on Projects and on the Image gallery). The shell fits the width
now, but a shot card needs two scrolls to read. Fix (shell, separate ticket): collapse the sidebar to a top bar at
< 768 px, or let the page scroll as one document instead of nesting the scroll box.

**P2-2. Project settings does not open from the header chips** (m6). The chips go to Overview; `<details>` stays
closed. Fix: pass an open request to the Disclosure (set `open` and focus the summary).

**P2-3. Typed voice id field is appended at the bottom of the voice list.** "Type a voice id instead" adds the field
inside the picker's scroll box under 60 rows, so only its top edge shows; the summary line then prints the raw id
("voice af_sky") while the field shows "Sky". Fix: put the id field above the list or in the picker footer; show the
resolved name in the summary.

**P2-4. Escape does not close the drawer after an Import.** After Import footage finished, Escape did nothing (the
Close button worked); `drawerEscapeAction` closes only when the keydown reaches the dialog, and focus was no longer in
it. Fix: keep focus inside the drawer after an action, or listen on `document` while the drawer is open.

**P2-5. Picker labels and buttons** (P2-9 again): "Shot 2 · take 1" tiles without the shot title; a second "Choose
file" next to the path box; drawer references are checkboxes named "on".

**P2-6. Header and tab row scroll away** on Cast with two cards and on a six-shot Storyboard (P2-14).

**P2-7. "No films yet" is missing on Export**; the Films block only appears after an export.

**P2-8. Bundle file name.** "Petal & Stem" exports as `Petal---Stem-20261108-130412.chaosfilm`. Fix: collapse runs of
separators to one dash.

**P2-9. Radios without names.** The template, aspect, resolution and frame-rate pills are `role="radio"` with no
accessible name beyond their text (fine) but the groups have no `aria-label`; the sidebar buttons are still unnamed
(P2-17, app-wide).

### First-time-user notes

Creating the ad, adding Maya with four pictures and a place, and reading the Storyboard took about six minutes and
nothing blocked. What still confused: the fps switch (P1-2); "Make 6 missing" on Overview against "Make missing" on
Storyboard; after importing stills the next step still says "Make 4 shots" (P1-1); the fourth picture's "Small picture;
results may drift." in 9 px under the thumbnail; the Mixer at 375 px shows an empty preview box with the controls two
scrolls down. The no-cast project needed no explanation: "Cast (optional)" stays unticked and nothing nags.

## Keep-list check (design.md section 2)

Walked the 2.1 to 2.6 rows against the build: list card facts, New project (aspect, resolution, fps, templates,
models), Open, Export bundle (card menu and Export tab), Import bundle (button, drop hint), Rename (card menu and
header), Delete with confirm, list states, header facts, Undo / Redo in the header, Project defaults under Overview
(voice now a picker with a typed id fallback), Cast (name, description, up to 4 pictures, path under "Paste a path"),
Add shot and ghost card, shot card controls, Lock and New take, transition, menu (no "Use still as reference" yet, as
planned), Make missing / Make all unlocked, estimate chip, keyboard, delete-shot confirm, shell states, drawer fields,
Import footage (file, path, galleries), Mixer preview, lanes, Open in Storyboard, captions, Export options, .otio,
bundle, gallery "Use in storyboard" to the Storyboard tab. Nothing from the keep-list is missing. Not exercised:
render, export, batch bar, relink, conflict chip, voice/music sends to the Mixer.

## Go / no-go

**Go** for merging into `feat/home-studio`, with P1-1 and P1-2 as follow-ups before the next release (both are small
copy and count changes; neither loses data). The round-1 P0s are gone, every P1 is fixed or reduced to wording, tests
and types pass, and the keep-list holds. Log P2-1 (375 px scroll window) as a shell ticket, not a Projects one.
