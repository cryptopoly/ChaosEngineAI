# v0.10.0 task A: field border contrast

Branch `v0100/border` (worktree `/Users/dan/ChaosEngineAI-v0100-border`).

## Done
- `src/styles.css`: `--border-control: #66727f` in `:root`; `:root[data-strong-borders="off"]` sets it back to `var(--border)`.
- Control selectors moved to `--border-control` (only their `border` declaration): `.primary-button, .secondary-button, .segment` (`.primary-button` keeps its accent border-color rule), `.text-input, .text-area, select.text-input`, `.thread-icon-button`, `.thread-mode-toggle`, `.cap-filter-btn`, `.challenge-prompt-tab`, `.preset-button`, `.cache-strategy-info-btn`, `.slider-input`, `.bm-run-select`, `.sampler-panel__clear`, `.sampler-row__reset`, `.kv-chip__bits-button`, `.kv-chip__reset`, `.swap-menu__reset`, `.composer-mic__button`, `.voice-row__play`.
- Left on `--border` on purpose: panels, cards, tables, dividers, chips and badges, popovers, menus, drop zones, list rows (`.session-button`), the `.subtab` and view-toggle containers, hover-only rules.
- Setting `strongControlBorders` (default true): `helpers/settings.py` (default + loader), `state/settings_state.py` (read + update), `models/__init__.py` (request), `src/types/settings.ts` (both interfaces).
- `src/App.tsx`: one effect sets/removes `data-strong-borders="off"` on `<html>`.
- `src/features/settings/AccessibilityPanel.tsx` (+ test), mounted in General after `LanguagePanel`. The panel has no props: it reads and follows the `<html>` attribute (MutationObserver), so SettingsDraft and `utils/runtime.ts` (outside the ownership row) are untouched. Switch shows an On/Off segment next to the label (the plain button had no visible state).
- Locale keys `accessibilityPanel.{title,strongBorders,note,on,off}`: English in `en`, null in the other ten (incl. tlh).
- Tests: `src/features/settings/AccessibilityPanel.test.tsx`, `tests/test_strong_control_borders.py`, `e2e/tests/23-border-contrast.spec.ts`.
- E2E suite: `scripts/e2e_phases/border_settings.py`, one call in phase 7 (`strongControlBorders round trip`; restores the user's value).
- Screens in `docs/plans/v0100-screens/`: `border-{chat,image-studio,settings,modal}-{on,off}.png`, `border-settings-en-XA.png`.

## Tests run
- `pytest tests/test_strong_control_borders.py tests/test_backend_service.py tests/test_services.py`: 134 passed.
- `npm test`: 1537 passed (118 files); `npx tsc --noEmit` clean; `npm run i18n:validate` OK.
- Playwright `23-border-contrast.spec.ts`: 2 passed (desktop, 375 px): computed border-color of `#settings-language-select` and a `.secondary-button` is rgb(102,114,127) on, rgb(39,48,58) off, survives reload, no sideways scroll at 375 px. Full `npx playwright test --workers=1`: 134 passed (15.2 min).
- E2E suite phase 7 against a throwaway backend: the new check passes (the other four phase-7 checks failed there only because the CLI had no API token for that backend; not related).

## Bug found and fixed on the way
After a reload the panel showed "on" while the page was off (state read before the saved value reached the page). Fixed by syncing from the attribute in an effect.

## Deviations
- Screenshots were taken with the Vite dev server and test backend through Playwright (headless Chromium), not `npm run tauri:dev`; the app shell was not launched.
- en-XA: no runtime switch exists (`?devLocale` is not wired), so the generated pseudo-locale files were copied over `en` temporarily and reverted (`git checkout`); nothing from that is committed. My panel shows no clipping; the Language panel's help text is cut off at the card bottom in en-XA (existing, not touched).
- The panel's row layout uses an inline style (flex, gap) so no layout rule was added to `styles.css` outside the token and control rules.
- `.session-button`, view-toggle and other container-like buttons were left on `--border` (list rows, not field boundaries). Say if you want them strong.

## NOT tested
Windows and Linux WebView rendering; OS high-contrast modes; `npm run tauri:dev`; the other eight Settings sections and other studios at the new border (only Chat, Image Studio, Settings, one modal were looked at).

## Open questions
None blocking.
