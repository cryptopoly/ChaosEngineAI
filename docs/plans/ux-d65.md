# Projects workspace (D-65): UX review

Date 2026-11-08. Branch `feat/project-workspace` (worktree `agent-a0601f472d24874f1`, head `22e07f6`). Read-only review
of a running build: Vite on 8997, backend on 8998 with a throwaway HOME, no models installed. Desktop at 1400 × 900
and phone at 375 × 812. Projects used: a "30 s ad" (Coffee shop ad), a "Vertical social" (Autumn walk), a
"Scene / b-roll" (Night market), a Blank one, and one seeded through the API with two shots that have stills
(Lighthouse keeper). Compared against `docs/plans/project-rework/design.md` (sections 4, 7, 13) and `mockups.html`.

Not tested: dropping files on the list or the Cast tab, the native file chooser, pasting a picture with Cmd+V,
a real Make run, a render, an export, the offline and not-found states, the conflict chip.

Verdict in one line: the shell (one entry, list, five tabs, route memory, pickers) is sound and already better
than the three-item version; what is missing is the guidance layer that the design leans on (next step, preflight,
cast on the cards), there is one data-loss bug on Cast, and the shell breaks at 375 px.

## Ranked findings

P0 blocks a task. P1 confuses or costs real time. P2 polish.

### P0

**P0-1. Cast: typed description is lost when you click "Add picture" straight from the field.**
Screen: Cast tab, character or place card. What happens: type a name, Tab or click into "What it looks like",
type a sentence, click "Add picture" on the same card. The picker opens; on Cancel (or after picking) the
description is empty again. Reproduced twice (character and place). The name survives because the click on the
description field blurred it first. Why it matters: the natural order is name, description, picture; every
first-time user will do exactly this and lose the sentence. Fix: save the field on `change` as well as `blur`,
or flush the pending edit before the picker opens (the picker button's `onMouseDown` can call the same save).
Also save on Enter in the name field.

**P0-2. 375 px: the project shell is wider than the screen and pans sideways.**
Screen: every tab inside a project. What happens: the header chips (16:9 · 1080p · 30 fps) clip at "30 f", the
tab row shows "Expo", the "Add shot" button reads "Add sh", a shot card's Move down arrow and More actions sit
off-screen, and focusing a tab scrolls the whole shell panel left so the back link and Overview disappear. The
Mixer lanes push the panel wider still. The document itself is 375 wide, so the e2e "no sideways scroll" check
passes while the user still cannot reach the controls. Why it matters: the design promises wrapping header,
scrolling tab row, stacked cards and a stacked run bar; none of that is there, and a phone user cannot finish a
shot. Fix: `min-width: 0` on `.project-shell`, `.project-shell__panel` and `.board-tab`; let the header wrap
(`flex-wrap`) with Undo/Redo and chips on a second line; `overflow-x: auto` on the tab row only, with the active
tab scrolled into view; shot cards one column with `width: 100%`; give the Mixer timeline its own horizontal
scroll container so it does not widen the panel.

### P1

**P1-1. Overview has no next step.**
Screen: Overview. What happens: a project with shots shows a counter line ("6 shots · 0 made · 0 stale · 30 s ·
no cast · no sound yet"), a row of hatched squares, and a collapsed Project settings row. There is no "Make 6
shots" button, no checklist, no activity, no folder size. Only a project with zero shots gets the "Start here"
card. Why it matters: the design's whole argument for an Overview tab is that it tells a first-time user what to
do next; without it the tab is a dead end that a user visits once. The template projects (6 shots, nothing made)
are exactly the case where "Make 6 shots" should be the one obvious button. Fix: ship the next-step card from
design 4.4 with at least the three states that exist today (write shots, make missing, add sound / render) and
the 7-step checklist; it can compute from data the Overview already has. Keep "Start here" as the zero-shot
variant of the same card.

**P1-2. Storyboard cards hide the prompt and the cast.**
Screen: Storyboard tab. What happens: a card shows number, title, a status pill, Lock, Redo, Transition and a
menu. The template's example prompt ("A striking close-up that stops the scroll") and the attached cast (Mara)
are only visible inside Edit prompts. After attaching Mara the card does not change at all. Why it matters: the
board is supposed to be readable as a storyboard; today the user must open every drawer to know what a shot is,
and cannot see which shots share a character (the reason Cast exists). The mockup shows a one-line prompt and
name chips on every card. Fix: one truncated line of the still prompt under the title (placeholder "No prompt
yet. Press Enter or Edit prompts.") and small name chips for attached references; both are read-only on the card.

**P1-3. "Redo" means two different things on the same screen.**
Screen: project header and every shot card. What happens: the header has Undo / Redo (history); each card has a
"Redo" toggle that means "make a new take next run". A user who wants to undo a change and one who wants another
take both reach for "Redo". Fix: rename the card toggle to "New take" (tooltip "The next Make adds a new take.
Nothing is overwritten.") and keep Redo for history only.

**P1-4. "Done" on a shot that has only a still.**
Screen: Storyboard card, Overview counter ("2 made"), project list ("4 shots · 16 s"). What happens: a shot with
an imported still and no clip reads "Done"; the Mixer then says "6 shots have no clip yet and are left out", and
Export MP4 is disabled. Why it matters: the user thinks the shot is finished and cannot see why the export is
dead. Fix: status "Still only" (or "Still · no clip") for takes without a clip; count them separately in the
counter line ("2 stills · 0 clips"); and make the Export preflight say it (P1-6).

**P1-5. The gallery's "Use in storyboard" menu is cut off at the bottom of the window.**
Screen: Images > Gallery, card actions. What happens: the project list and then the shot list open downward
from a button at the bottom of the page; only "Back to projects" and "New shot" are visible, the per-shot rows
("Shot 1 (new take)" …) are below the viewport and there is no scroll. Why it matters: the deep link lands on the
right tab and highlights the new shot (good), but the per-shot path is unreachable on short windows. Fix: flip
the menu upward when there is no room below (the app's menu primitive probably already supports a placement);
list shot titles ("1 · Hook") rather than "Shot 1 (new take)".

**P1-6. Export has no preflight and disabled buttons say nothing.**
Screen: Export tab. What happens: Export MP4 and Export for an editor are disabled with no reason; the three
"not estimated" lines sit where the reason should be; "Audio lanes as separate files" is listed under the
"Captions" heading. The design's "Before you export" list (clips, stale, captions, cloned voice, interrupted
export) is absent, and there is no Films block when nothing has been exported. Fix: add the preflight list above
Format, with "n shots have no clip and will be left out" linking to Storyboard; when nothing can be exported put
the reason on the button's helper line ("Nothing to export yet: no shot has a clip"); move the stems tick under
its own "More" heading; show "No films yet" under Films.

**P1-7. New project dialog: Create is below the fold on a 900 px tall window once Models is open.**
Screen: New project. What happens: with the Models disclosure expanded the dialog grows past the viewport and
scrolls internally; Cancel and Create are hidden and nothing hints that the dialog scrolls. Fix: pin the
footer (Cancel, Create) and let only the body scroll; or cap the template row to one line (see P2-3) so the
dialog fits. Enter in the name field does submit, which helps keyboard users.

**P1-8. Project list: five identical "More actions" buttons.**
Screen: Project list. What happens: every card's menu button is named "More actions" (the open button is
"Open {name}"); screen-reader and `find`-style navigation cannot tell them apart. Also the empty-state block
offers only New project, not Import bundle, and there is no hint that a .chaosfilm can be dropped. Fix: name the
button "More actions for {name}"; add Import bundle to the empty state and the line "or drop a .chaosfilm here"
from the mockup.

**P1-9. The run estimate reads as three failures.**
Screen: Storyboard run bar, drawer footer, Export action. What happens: "Time: unknown until the first run ·
Memory: not estimated · Disk: not estimated" appears three times across the project, in grey, before any
button. With no video model it cannot say more, but three "not" phrases next to the primary button read as
errors. Fix: one chip, "Estimate: unknown until the first run" (or "Install a video model to see an estimate"),
and put the shot count on the button ("Make 6 missing"), as the mockup does.

### P2

**P2-1. Two page headers.** The app banner "Workspace / Projects / Storyboard films: shots, sound, export" stays
above the project shell, so an open project shows "Projects" twice (banner and back link) and the project name
only on the third line. ~90 px of the 900 are spent before the tabs. Fix: when a project is open, the banner
shows the project name as the title and the list's "Projects" moves into the breadcrumb, or the banner collapses
to one line.

**P2-2. The list card duplicates the banner.** On the list the panel starts with another "Projects / 4 projects"
heading directly under the banner's "Projects". Fix: drop the panel heading; keep the count in the banner
subtitle ("4 projects · each one a folder you can move or back up").

**P2-3. Template pills wrap into a lonely second row.** Five templates at ~470 px put "Scene / b-roll" alone on
line two, and the slash is unusual in the app's copy. The pills also omit the frame rate they set ("30 s ad"
silently switches to 30 fps). Fix: four pills on one line by renaming to "Scene" (caption "b-roll, no cast") and
shortening "Vertical social" to "Vertical"; put "16:9 · 30 fps · 6 shots" in the caption.

**P2-4. Models summary "auto · auto".** The collapsed Models row and the Project settings summary say
"auto · auto · voice auto". Fix: "Automatic · Automatic · no voice", or hide the row until a model is chosen.

**P2-5. Voice picker is a 60-row flat list with no search.** Built-in alone has ~50 voices; the design asks for a
search field and a Play sample button per row. Only the "needs an extra install" notes exist. Fix: search at the
top, groups collapsible, Play sample when a TTS model is installed.

**P2-6. Cast empty states repeat the same sentence three times.** The tab intro, the Characters group and the
Places group each explain what a reference is; the group text still says "a picture" while the intro says "up to
4 pictures". Fix: keep the intro, shorten the groups to "None yet." with the Add button.

**P2-7. Add character / Add place does not focus the name field.** Typing straight after the click goes nowhere.
Fix: focus the new card's Name input.

**P2-8. Tiny targets on the picture strip.** The remove "x" on each cast thumbnail is ~14 px, and the "Small
picture; results may drift." note wraps under a 64 px thumbnail in 9 px text. Fix: 24 px remove button shown on
hover/focus; move the small-picture note to a line under the strip.

**P2-9. Picker wording.** Image gallery empty state says "No saved images match." before any search; "From a
shot" labels tiles "Shot 1 · take 2" without the shot title; the Paste a path row also has its own "Choose file"
button next to the Choose file tab. Fix: "No saved images yet. Make one in Images > Studio."; "1 Hook · take 2";
drop the second Choose file.

**P2-10. Contact sheet tiles ignore the aspect ratio.** A 9:16 project shows 16:9 hatched tiles on Overview.
Fix: tile aspect from the project format.

**P2-11. Counter line noise on an empty project.** "0 shots · 0 made · 0 stale · 0 s · no cast · no sound yet"
above the Start here card. Fix: hide the counter line until there is a shot.

**P2-12. "Start here" promises a list paste that does not exist.** The copy says "Write your shots, one line
each" but Write your shots opens the Storyboard with Add shot only (the Paste a list sheet is phase 4). Fix: "Add
your shots and give each a line of prompt" until Paste a list ships.

**P2-13. Shot drawer.** Title is "Shot 1" not the shot's title; references are bare checkboxes without
thumbnails or kind; the Import footage block keeps the raw path box as its first control. Fix: title "1 · Hook";
reference rows with a 24 px thumbnail and "character" / "place" captions; Choose file first, path collapsed.

**P2-14. Tab row and header are not sticky.** On a long Cast or Storyboard tab the header, Undo/Redo and the
tab row scroll away. Fix: sticky header inside the shell (the sidebar already is).

**P2-15. Sidebar entry style.** "Projects" renders like Server / Logs (mixed case, bold) between the uppercase
group rows IMAGES, VIDEO, VOICE, MUSIC AND AUDIO. It stands out, which may be intended, but it reads as a
different kind of item. Fix: either style standalone-in-section tabs like group headers, or accept and document
it.

**P2-16. Menu duplicates.** Move up / Move down exist both as card arrows and as menu items. Fix: keep one.

**P2-17. Sidebar buttons have no accessible names** (app-wide, not new): every nav button is an unnamed
`button` with the label in a child span, so assistive tech reads "button". Fix: `aria-label` or put the label
text directly in the button. Also the "Make all unlocked" button's accessible name is its tooltip sentence, not
its label.

**P2-18. Reload lands on Dashboard.** The app has no URL routing, so F5 and browser back lose the place; the
Projects entry then correctly reopens the last project and tab. Not Projects-specific, but the project shell is
where a user most expects a reload to come back to the same shot.

What works well and should stay: the five-tab order and arrow-key tab row; opening the last tab per project and
the list when none is open; rename from the header with Enter/Esc; the delete confirm copy; the picker's three
sources with "Paste a path" demoted; the 4-picture strip; the fit-hint sentence when no model is installed; the
gallery send landing on Storyboard with the new shot highlighted; the New project dialog at 375 px.

## The five changes that would most improve the experience

1. **Fix the Cast save race (P0-1)** and focus the name field on Add (P2-7). Ten minutes of work, removes the only
   data-loss path.
2. **Make the shell fit 375 px (P0-2)**: min-width 0, wrapping header, scrolling tab row only, one-column cards,
   Mixer lanes in their own scroll box. Then add the 375 px screenshots of Overview, Cast and Storyboard to the
   e2e mobile spec so it cannot regress silently.
3. **Ship the Overview next-step card (P1-1)** with the three computable states and the checklist. It turns the
   tab from a counter into the guide the design is built around, and it gives the template projects their
   obvious first button.
4. **Put the prompt line and cast chips on shot cards (P1-2)** and rename the card toggle to "New take" (P1-3).
   The board becomes readable without opening a drawer, and the Cast tab's effect becomes visible.
5. **Honest status and preflight (P1-4, P1-6, P1-9)**: "Still only" instead of "Done", one estimate chip with the
   count on the Make button, and a Before-you-export list that explains a disabled Export MP4.
