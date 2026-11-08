# Klingon locale: self-QA

Branch `feat/klingon-locale`, 2026-11-08. Decision D-69.

## Coverage

| Surface | Strings | Translated | Note |
|---|---|---|---|
| Frontend `src/locales/tlh/*.json` | 4157 (incl. the new `languagePanel.communityNote`) | 4157 (100 %) | `npm run i18n:validate` shows `tlh 100%  (experimental, outside the strict gate)` |
| Tauri `src-tauri/locales/tlh.yml` | 35 | 35 | menu, tray, updater, dialog |
| Backend `backend_service/locales/tlh/LC_MESSAGES/messages.po` | 11 literal `localized_detail()` strings | 11 | the ~68 `str(exc)` call sites are dynamic and untranslatable in any locale; `en`, `ja`, `zh_CN` catalogs were never extracted (0 strings) |

## Checks run

| Command | Result |
|---|---|
| `npm run i18n:validate` | OK; `tlh 100%`; other nine locales 94.2 % because `en` already carries keys they lack (not from this branch, except `languagePanel.communityNote`, added as `null` to them) |
| `npm test` | 107 files, 1391 tests passed (includes the new `validateGate.test.ts`, the tlh ICU-compile test, the fallback test and the Intl fallback pin) |
| `npx tsc --noEmit` | clean |
| `pytest tests/i18n tests/test_backend_service.py -k settings` | passed, incl. the new `test_klingon_catalog_is_compiled_and_falls_back` and `test_settings_persist_locale_and_clock_format` |
| `E2E_WEB_PORT=8971 E2E_API_PORT=8972 npx playwright test tests/17-klingon.spec.ts` | 3 passed: pick tlh in Settings (note visible, sidebar flips, `html lang=tlh`), six surfaces screenshotted at desktop and at 375 px with no sideways scroll, switch back to English |

Screenshots looked at (`e2e/screenshots/klingon-*.png`, git-ignored): Settings general, Settings diagnostics, Settings accelerators (hosts the Setup install strings), Dashboard, Chat, Image Models, each at 1280 and 375 px. Findings from looking:

- Layout holds at 375 px; Klingon labels are shorter than English on every tab and button.
- The CSS `text-transform: uppercase` on eyebrows, badges and sidebar section labels turned `yIchenmoH` into `YICHENMOH`, which in Klingon spelling changes the letters. Fixed with `html[lang="tlh"] * { text-transform: none !important; }`.
- Strings that stay English on screen are hard-coded in components or come from the backend without a key: `Idle`, `Refreshing...`, `Copy diagnostics`, `Gathering system info...`, `Settings updated`, `0 models discovered across 3 configured directories.`, `Recommended target: 70B GGUF`, the DFlash row text `Apple Silicon — any LLM with a registered draft model`, the `yes`/`no` values in the backend-readiness line. These are i18n gaps for every locale, not tlh-specific; listed for a follow-up.
- Infra bug found and fixed on this branch: `PATCH /api/settings` dropped `locale` and `clockFormat` (not in `UpdateSettingsRequest`, not carried over in `update_settings`, not returned by `settings_payload`), so the Settings language choice never reached `settings.json` and did not survive a restart, for any locale. Fixed with a backend test; the Playwright spec now reloads and finds `lang=tlh`.

## ICU apostrophe rule

ICU MessageFormat treats `'` as a quote when it is followed by `{`, `}`, `#` or `|`. Klingon uses `'` as a letter, so 29 strings had `qonlu'ta'}` swallow the closing brace. Rule applied (and worth adding to the translator notes for any future re-sync): inside a string that contains `{`, an apostrophe directly before `{ } # |` is written `''`. Plain strings are not ICU-processed and keep single apostrophes. The `i18n.test.ts` ICU-compile test catches a regression.

## Back-translation spot check

Every 15th string across the nine text namespaces (276 strings) was back-translated to English from the Klingon alone, then compared with the source. The reviewer is the same model that wrote the translation, so this measures consistency and lost information, not fluency.

Result: 276 compared, 0 with an inverted or wrong meaning, 9 with a drift worth fixing or flagging, the rest equivalent in meaning (wording differs: "finished" for "complete", "folder" for "directory", "look for" for "browse").

| # | Key | Klingon | Read back as | Source | Action |
|---|---|---|---|---|---|
| 116 | studio.imageStudio.samplers.default.hint | yab wIv | Model's choice | Whatever the model shipped with | fixed to `yab motlh` (model default) |
| 170 | studio.board.shot.redoTitle | … pagh Qaw'lu'. | Nothing is deleted. | Nothing is overwritten. | fixed to `pagh ghItlhqa'lu'` (nothing is rewritten) |
| 202 | studio.overview.activity.interrupted | (SoQ ngoq) | (closed code) | (the app closed) | fixed to `(SoQpu' app)` |
| 36 | common.pluginsTab.toggleOff | CHU'HA' | OFF | OFF | all-caps breaks Klingon spelling; now `chu'Ha'`; same for `toggleOn`, the three server badges and `brokenBadge` |
| 57 | common.modelLaunchModal.busyConfirm | yIchu'; qaSjaj | Load; so be it | Load anyway | kept; `-jaj` ("may it happen") is canon but the idiom is a stretch, for the speaker |
| 113 | studio.imageStudio.diffusionCache.teaCacheHidden | … mIlloghvaD FLUX ghom neH Qap | works only on the FLUX family for images | its image-side calibration only covers the FLUX family | "calibration" dropped; acceptable simplification, flagged |
| 206 | studio.templates.shots.adProblem.title | qay' | Problem | The problem | `qay'` is the verb "be a problem" used as a label; a speaker may prefer `Seng` (trouble) |
| 207 | studio.templates.shots.socialShow.prompt | wa' mIllogh Sayqu'Daq qech tIn | One big idea on a very clean image | The main idea shown in one clear picture | creative one-liner, loose; acceptable |
| 265 | runtime.tensorfold.grammarLabel | ngoD chen: | Output shape: | Structured output: | coined; acceptable |

Observed in passing: `{count, plural, other {# may''}}` and the other doubled apostrophes render as a single `'` (checked with `intl-messageformat` directly).

## Coined terms for a speaker to review

The full list with meanings is in `glossary.md` ("Coined terms" plus the "Added during translation" table). The ones most worth a second opinion:

- `yab` for model (mind); the alternative is `velqa'` (replica).
- `ghom` for library and gallery, `Qu'` for project, `lut` for film, `'ay'` for shot, `nID` for take.
- `lI'` download / `lab` upload (transmit to / away), `qon` copy, `jom` install, `chu'` load.
- `vumpa'` workspace / Studio, `qoD pat` backend, `jabwI'` server, `chelwI'` plugin, `nommoHwI'` accelerator, `jonta'` engine.
- `mIllogh vIH` video, `QoywI'` microphone, `ghogh velqa'` voice clone, `QIch ghItlhwI'` speech-to-text, `ghItlh jatlhwI'` text-to-speech.
- `-ghach` nominalisations the studio translator used (`chenmoHghach`, `lI'ghach`, `jomghach`, `qonghach`, `Delghach`, `wuqghach`); a speaker may prefer verb phrasing.
- `Daw'` and `SaHbe'` for random (two different coinages in studio.json; `Qoj` in chat.json). Unify on a reviewer's pick.
- `tInghach` size, `patlh` quality, `Hat` temperature, `ghu'` context, `Seg` variant, `Hap` file format, `mIr` sort, `chevwI'` filter.
- Comparatives built with `law'/puS` in musicStudio issue texts.

## What was not tested

- The Tauri menu and tray in Klingon (needs a desktop build; `cargo test` not run in this pass, the YAML mirrors `en.yml` key for key).
- Backend strings in the real app (unit test only; the e2e backend never raised one of the 11 messages).
- Any voice feature: there is no Klingon speech model, voice stays English by design.
- The nine other locales' 94.2 % coverage is pre-existing on `release/v0.10.0` and untouched here.
