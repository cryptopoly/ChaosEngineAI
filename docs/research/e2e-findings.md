# Browser E2E findings

Things the Playwright run in `e2e/` turned up. Each has a file, a repro and a status. Newest first.

## F-1: Music Models table keeps its desktop width on a phone (page scrolls sideways)

- Status: fixed. Below 760 px the header row is hidden and each row wraps (`.models-table` in `src/styles.css`); the `test.fail` pin is gone.

- Where: `src/features/music/MusicModelsPage.tsx:47-50` (`MUSIC_GRID`, `MUSIC_GRID_NARROW`) with the shared table in `src/components/models/ModelsPage.tsx:216` and `src/styles.css:11421-11441` (`.models-table`). The narrow template only helps down to about 1100 px; its columns have minimums that add up to about 650 px, plus the sidebar stacking above on a phone.
- Repro: open the app at 375 px wide, Music > Models. `document.documentElement.scrollWidth` is 626 px (251 px too wide); `.media-model-head` and `.media-model-rows` end at x=653.
- Impact: low. The desktop window cannot normally be that narrow. It shows up in a narrow window or a future mobile or web build.
- Not fixed: it needs a visual decision (hide more columns below ~760 px, or let the table scroll inside its own box), more than a one-line change.
- Covered by: `e2e/tests/09-mobile.spec.ts`, "375 px: Music Models fits the width", marked `test.fail`. It turns red the day the bug is fixed; then remove the `test.fail` line.
- Not checked: the Image, Video and Chat Models tables at 375 px (a probe of Image Models read 375 px, but it may have been measured before the rows rendered).

## F-2: Project card says "no shots yet" for a project that has shots without takes

- Status: fixed. The card shows "no shots yet", "no pictures yet" or "no preview" (`emptyThumbState` in `storyboardLogic.ts`).

- Where: `src/features/storyboard/ProjectCard.tsx:45-47`, string `storyProjects.card.noShots`.
- Repro: create a project, add two shots, do not make anything, go to Storyboard > Projects. The card meta says "2 shots · 8 s" and the thumbnail area says "no shots yet".
- Impact: copy only, but the two lines contradict each other. "no pictures yet" would be right.
- Not fixed: an i18n string change touches ten locale files.

## Notes (not bugs)

- `HOME_STUDIO_ENABLED` (`src/constants/features.ts`) was a hard-coded `false`, so Music and Storyboard could not be reached in any browser run. It now reads `VITE_CHAOSENGINE_HOME_STUDIO === "1"`; a normal build leaves it false. The E2E Vite server sets the variable (`e2e/playwright.config.ts`).
- Chat's empty-state "Load Model" button opens Chat > Models rather than the launch dialog; the dialog opens from the chat button on a model row. That is as designed, but it matters when writing a test.
- The AI-label switch in the Mixer export panel (`role="switch"`) updates after the backend round trip, so a test must wait for the new state instead of using `uncheck()`.
