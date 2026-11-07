# Accessibility pass on the home studio screens (2026-11-07)

Scope: Storyboard (Board, Mixer, FilmList), Music Studio (Music | Sounds, licence screen), Voice (clone flow), first-run terms, ProjectFilter and ProjectHeader (spot check), sidebar. Method: reading the code against WCAG 2.2 AA. No screen reader, axe or browser run was done, so everything below is by inspection.

## Fixed in this branch

| Area | Finding (WCAG) | Fix |
|---|---|---|
| Terms (first run) | No focus placed in the dialog, no trap, no Escape in the read-only view, error not announced (2.4.3, 4.1.3) | `useDialogFocus`: focus starts on the dialog (title and points read first, not on a button, so an Enter cannot accept by accident), Tab wraps, Escape closes the read-only view only, focus returns to the opener. Dialog described by the points list. |
| Confirm / New project dialogs | Own Escape handler, no trap, no return of focus; destructive confirm focused the Delete button (2.4.3, 3.3.4) | Same hook. A danger confirm now starts on Cancel. Error text is `role="alert"`. |
| Licence screen | Trap existed, focus did not return to the opener (2.4.3) | Moved onto the shared hook. |
| Generation progress (`LiveProgress`, used by Music, Sounds, Image, Video, Storyboard) | Bar was a bare div; phase changes silent; decorative dot grids and the clock were in the reading order; no reduced motion (1.3.1, 4.1.2, 4.1.3, 2.3.3) | `role="progressbar"` with name and value (no value while indeterminate), a polite status line that changes only on phase or queue change (not the 250 ms clock), decoration `aria-hidden`, active phase `aria-current="step"`, looping animations off under `prefers-reduced-motion`. |
| `RunEstimateStrip` | Queue and busy lines were live regions that mounted with their text, so they were never announced; memory verdict by colour only (4.1.3, 1.4.1) | Notes region is always mounted (`display: contents`). Caution and danger add a screen-reader-only word. |
| Sidebar | Current page only by colour; unnamed child group; logo alt repeated the heading (1.3.1, 4.1.2) | `aria-current="page"` on nav and child buttons, group labelled, logo `alt=""`. |
| Sub tab bar | `role="tab"` with every tab in the Tab order and no arrow keys (2.1.1, ARIA tabs pattern) | Roving tabindex, Arrow / Home / End move and open the tab. |
| Voice clone | Delete buttons all named "Delete"; voice radios without a group; text area, file input and recording player unnamed; download error silent (1.3.1, 4.1.2, 4.1.3) | Names added (voice name in Delete), radiogroup, `role="alert"` on errors. Speak text area and the transcript line editor named too. |
| Target size | Conflict-chip dismiss was about 18 px (2.5.8) | 24 px minimum. |

## Checked and fine

- Contrast of the dark tokens (computed, WCAG relative luminance): `--text` 15.3 to 17.3, `--muted` 6.9 to 7.8, `--accent` 8.0 to 9.1, `--success` 9.2 to 10.4, `--warning` 9.4 to 10.7, `--danger` 7.3 to 8.3, all against `--app-bg`, `--panel`, `--panel-alt`. No token change was needed.
- Storyboard: takes strip is a radio group with roving tabindex; shot cards have a keyboard alternative to drag reorder (Move up / Move down buttons and the action menu); icon buttons are named; failed shots are `role="alert"`; running bar is a progressbar with a name.
- Mixer: the ruler is a keyboard slider, clips are focusable list items with full labels, clip start, trim, fades and gain are numeric fields in the inspector (the keyboard alternative to dragging and trimming, WCAG 2.5.7), transport buttons are named, a polite announcer exists.
- Music | Sounds switch and the track-kind buttons use `aria-pressed` in a labelled group. Export panel uses `role="switch"` and a labelled radio group.
- ProjectFilter (listbox, search, Escape, return focus), ActionMenu and SendToStoryboardMenu follow the menu pattern.

## Deferred

| # | Severity | Item | Proposed approach |
|---|---|---|---|
| 1 | Medium | Timeline clips can be moved and trimmed only with the pointer or via the inspector. Arrow-key nudging on a focused clip (and Shift for coarse steps, Alt+Arrow to trim) would be faster for keyboard users. | In `MixerTab` `onKeyDown`, map Arrow keys to the same `commitClip` call the pointer path uses (0.1 s fine, 1 s with Shift), announce the new start through the existing polite announcer. The inspector already meets 2.5.7, so this is an efficiency gain, not a blocker. About a day with tests. |
| 2 | Low | Preset pills in the project dialog and Export panel are `role="radio"` but each is its own Tab stop and ignores arrow keys. | Reuse `nextTakeIndex` and roving tabindex as in `TakesStrip`. |
| 3 | Low | Sub tab bar tabs have no `aria-controls` / tabpanel because the panel is a whole screen. | Either name the main region by the active tab, or switch the bar to a nav with `aria-current`. The e2e smoke spec looks tabs up by role, so change both together. |
| 4 | Low (app-wide) | Input and button borders (`--border` #27303a) are 1.2 to 1.4:1 against the panels, under the 3:1 for UI component boundaries (1.4.11). Text and focus rings pass. | Needs a theme decision (a stronger `--border-control` token for fields). Not changed here: it would retheme the app. |
| 5 | Low | Voice call overlay and a few older modals (`ModelLaunchModal`, `ImageGenerationModal`) were out of scope and do not use `useDialogFocus`. | Adopt the hook one by one. |
| 7 | Info | Dropped files on Transcribe and drag-to-reorder have button alternatives; checked by reading only. | Cover in the Playwright pass with axe once the e2e work lands. |

## Not verified

No VoiceOver, NVDA or axe run; no browser check of focus order or the new focus rings; the `display: contents` notes wrapper was not checked in the WebKit build of the desktop shell (it relies on WebKit 16+ keeping the element in the accessibility tree).
