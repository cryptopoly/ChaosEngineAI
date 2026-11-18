# Review stream I18NFIX: Diagnostics in every language, the language banner, copy items

Branch `review/i18nfix` (pushed), worktree `/Users/dan/ChaosEngineAI-review-i18nfix`, cut from `review/d` at `31e9d194`.
Fixes defects L-1, L-2, L-4, L-5 (frontend part) and L-7 from `docs/progress/review-d.md`.
Started 10:17 on 2026-11-18.

## Headline

Settings > Diagnostics now reads in the app's language in all nine shipping locales, and the first-launch banner no longer asks a German user, in German, whether to switch to German.
99 strings were written or rewritten in English and translated into each of the nine locales (891 translations). None has been read by a native speaker: see the list at the end.

## What changed

| Defect | Was | Now | Commit | Test |
|---|---|---|---|---|
| L-2 | About 100 labels on Settings > Diagnostics were hard-coded English: section titles, row labels, values (installed, missing, yes, no, not set), the log tail line, the three recovery actions with their confirm dialogs and results. | 80 keys under `diagnostics.json` `panel`, all through `t()`. Names of packages, files and variables stay verbatim (`torch`, `llama-server`, `cwd`, the torch probe keys, environment variable names). The copied markdown stays English, as the review asked. The capture time uses the app language; the log line count is an ICU plural. | `b604c698` | vitest `DiagnosticsPanel.i18n.test.tsx`, e2e spec 38 |
| L-1 | On a German system the app starts in German and the banner asked "App auf Deutsch umstellen?" with "Zu Deutsch wechseln" and "Englisch beibehalten". | See the decision below. | `2bc88f26`, `08d8f58d` | vitest `firstLaunchLocaleBanner.test.ts`, e2e spec 34 |
| L-7 | The Language panel was titled "Settings"; the clock-format and UI-scale rows had only an aria-label. | Title "Language and time"; both rows show their label (the same string, now visible and tied with `aria-labelledby`). | `2bc88f26` | e2e spec 38; spec 33 still finds both radio groups by name |
| L-4 | Chat welcome: "We recommend a small Qwen3 or Llama 3 variant". Checked against the catalogue: no Qwen 3.x or Llama 3.x row has a variant under 3 GB any more (the small rows are Gemma 4 E2B, Bonsai, MiniCPM 5 and LFM3, 1.0 to 1.6 GB). | "For a first run, choose one of the small models (1 to 2 GB)". No family is named, so it cannot go stale again. | `2bc88f26` | e2e spec 38 |
| L-5, frontend part | Compare: result panel titles, "Waiting for Model A to finish...", "Loading...", "Generating...", the picker title "Select Model A" and its button "Use for Model A" were English. HTML Challenge used the same English label table, so its picker read "Für Model A verwenden". | All through `t()`; one helper, `compareSlotLabels(t)`, gives both views the catalogue's slot names. | `2bc88f26`, `fce56ddc` | e2e spec 38 |
| Null placeholders | Five keys other streams added were `null` in the nine locales and showed English: keep-going stall (`keepGoing.stop.job_stalled`, `keepGoing.waitingOn`), the backend-offline banner (title, detail), the size-under-minimum note. | Translated. | `2bc88f26` | `i18n:validate` |

Code shape: `DiagnosticsPanel.tsx` was 857 lines (soft cap 500). The read-only snapshot moved to `DiagnosticsSnapshotView.tsx` (with `diagnosticsSections(snapshot, t)`, the one place that lists the rows) and the three recovery buttons with their state to `DiagnosticsRecoveryActions.tsx`. That is also what lets a test render them. `DiagnosticsPanel.tsx` is 529 lines and still holds the Accelerators and Disk usage panels and the markdown formatter.

## Decision L-1: what the banner says (for the Decisions page)

The app boots in the detected language (`main.tsx` and the banner both call `detectLocale()`), so the banner always appears in the language it is about. Two options were on the table:

1. Start in English until the banner is answered. Rejected: a user who does not read English gets an English app and an English question; it also undoes D-1's "never flash English on a non-English machine".
2. Keep starting in the detected language and say so. **Chosen.**

Wording (English source, nine translations):

- "The app is in {endonym}, your system language."
- "You can change this any time in Settings → Language."
- Primary button: "Keep {endonym}". Saves that locale.
- Secondary button: "Use English". **Always written in English**, whatever the app language (the component asks for `lng: "en"` and every catalogue carries the string verbatim), so someone whose system is set to a language they do not read can find the way out. Switches at once and saves `en`.

The four old keys (`detected`, `prompt`, `switch`, `keep`) are removed from all eleven catalogues. Both buttons still dismiss the banner for good.

## Counts

- New keys: 92 (80 Diagnostics, 4 banner, 8 Compare). Rewritten: 2 (`languagePanel.title`, `emptyBanner.welcomeBody`). Filled from `null`: 5. Removed: 4.
- Strings written per locale: **99** in each of zh-CN, zh-TW, ja, de, ru, ko, fr, es, pt-BR (98 translated, 1 kept in English on purpose: "Use English"). A few are the same word in that language as in English and are correct so ("GPU", "Python", "Hardware" and "Version" in German, and so on).
- `npm run i18n:validate`: en 5042/5042; each of the nine 5040/5042, shown as 100 %. The two "missing" are not strings: the validator counts the Klingon row of `meta.json`'s `supportedLocales` array, which only exists in `en/meta.json`.
- Klingon (experimental, outside the gate): the 92 new keys are `null`; `languagePanel.title` and `emptyBanner.welcomeBody` went back to `null` because their English changed meaning. 4092 of 4954 before, 4086 of 5042 now (81 %).

## Tests added

- `src/features/settings/DiagnosticsPanel.i18n.test.tsx` (12 tests). (a) A marker catalogue: every Diagnostics string is wrapped in `[[ ]]`, the panel is rendered in its empty, snapshot and recovery states, and any text that is not marked, not fixture data and not a name kept verbatim fails. (b) For each of the nine locales, every key the panel uses has a string. (c) Rendered in each locale, every section title and row label is that locale's catalogue value, and no English value that the locale words differently is left. Checked by mutation: a hard-coded `"Driver"` and a mistyped key each fail it.
- `src/components/__tests__/firstLaunchLocaleBanner.test.ts` (10 tests): the banner's key set, and in each locale the statement names the language, is not a question, has no unfilled placeholder, and "Use English" is English.
- `e2e/tests/38-diagnostics-i18n.spec.ts` (3 tests): Diagnostics in German and Japanese against a real backend, labels compared with `src/locales/<lang>/diagnostics.json`; the Language panel; Compare and HTML Challenge in German.
- `e2e/tests/34-locales.spec.ts`: the banner test asserts the new German wording and both button labels; a second test covers "Deutsch beibehalten" (saves `de`, stays German, does not come back after a reload).
- Found while running it: spec 34's ten locale tests only passed when an earlier spec in the same run had accepted the terms. Run on its own (`playwright test tests/34-locales.spec.ts`) each one sat behind the terms dialog until its 180 s timeout, about 30 minutes in all. Each now opens the app in English first. Not caused by this branch; fixed here because it cost a run.

## Verification

All run on the final tree: see "Final checks" at the end.

Seen in a browser (headless Chromium, screenshots looked at): the banner on a German system; Settings > General in German (new title, both labels); Settings > Diagnostics in German and Japanese, top to the recovery actions; Compare in German.

## Not done, and why

- **L-5, backend-made text**: activity feed lines, folder labels, "Released Nov 2026", the DFlash row. These are sentences built in Python (`backend_service/catalog/*.py` carry about 20 hand-written `updatedLabel` strings in three shapes, "Released Aug 2026", "Released 2025-12", "Updated 2 weeks ago"). A front-end patch for the date alone would leave "Veröffentlicht Nov. 2026" next to "Updated 2 weeks ago", so it was left whole. Proper fix as the review said: the backend sends a key plus values (and a `releaseDate`), the front end formats with `Intl`. It touches the catalogue payloads and three model pages; about half a day with tests.
- **Klingon** strings for the new keys (fan translation, falls back to English).
- **`npm run i18n:extract`** cannot be used as the "zero new keys" check in CLAUDE.md section 6: it ignores the namespace given to `useTranslation("...")` and reports 2977 keys as missing and 4086 as unreferenced on an untouched tree (for example `diskUsage.title`, which exists). The tracked `src/locales/.extract-report.json` is long out of date; it was not regenerated (a 12 000-line diff of noise). Worth a follow-up row.
- `backend_service/locales` and `src-tauri/locales/*.ftl`: nothing changed there (no backend or menu string was touched), so no `pybabel compile`.

## What was not tested

- No native speaker has read any of the 891 translations.
- The recovery actions' result messages (after an install or a re-extract) and the three `window.confirm` texts are covered by the key checks only; nothing was installed or deleted.
- The CUDA block (Windows / Linux hosts) was rendered in vitest, not seen in a browser: this is a Mac.
- The banner in the Tauri shell (OS locale through the shell rather than `navigator.languages`); only Chromium with `locale: de-DE` and `ja-JP`.
- The other seven locales' Diagnostics in a browser (vitest renders all nine; the browser check is German and Japanese). Spec 34 still boots all ten non-English locales through every Settings section for raw keys and sideways scroll.
- 375 px and 1100 px layouts of the new visible labels beyond what specs 33 and 34 measure (they pass).
- The full Playwright suite (specs 01 to 31), pytest, cargo and the E2E suite: not run; no Python, Rust or inference code changed.

## Needs native-speaker review

Everything is machine-written. These are the ones I am least sure of. Keys are in `src/locales/<lang>/`.

Wording that is close to safety (destructive or long-running actions; no voice-clone or legal text was touched):

1. `diagnostics:panel.reextract.confirm` and `.hint`, all nine: "delete the cached embedded runtime", "runtime tree looks corrupted", "crashed install" (de "Laufzeitbaum", "abgestürzten Installation"; fr "arborescence", "interrompue par un plantage"; es "se bloqueó"; pt-BR "travou"; ko "런타임 트리"; ja "ランタイムのディレクトリ").
2. `diagnostics:panel.cuda.confirm` and `.hint`, all nine: "imports as CPU-only", "stale / mismatched torch files", "~2 GB, 1-5 minutes".
3. `diagnostics:panel.mlxVideo.confirm` and `.hint`, all nine: "shadowed the real one" (de "verdeckt", fr "masqué", es "ocultó", pt-BR "encobriu", ja "隠しています", zh "遮蔽", ko "가린"), "partially staged".
4. `diagnostics:panel.reextract.deleted`, all nine: it quotes the Restart Backend button; the quoted text must stay equal to `panel.restart`.

Terms:

5. `diagnostics:panel.rows.release` (the OS release number): fr "Publication", es "Lanzamiento", pt-BR "Lançamento", ru "Выпуск", zh "发行版本 / 發行版本", ko "릴리스".
6. `diagnostics:panel.rows.machine` (the CPU architecture string): de "Maschinentyp", ru "Тип машины", zh "机器类型 / 機器類型", ja "マシン", ko "머신".
7. `diagnostics:panel.rows.prefix`, `.basePrefix` (Python's `sys.prefix`): translated literally in all nine.
8. `diagnostics:panel.rows.warmPool`: follows the existing term for "warm pool" in each locale (zh-CN "热池", fr "pool chaud", es "grupo en caliente", pt-BR "pool quente", ru "тёплый пул"); those were never reviewed either.
9. `diagnostics:panel.sections.environment`: translated as "environment variables" (what the section lists) rather than "environment".
10. `diagnostics:panel.sections.extras`: ru "Дополнительно (каталог установки пакетов для GPU)" is long.
11. `diagnostics:panel.backendOffline`: keeps "sidecar", as the three sibling strings in `common.json` do (zh and ja leave it in Latin letters).

Grammar:

12. `diagnostics:panel.values.none` after "Loaded model": gender picked for "model" (de "keines", fr "aucun", es "ninguno", pt-BR "nenhum").
13. `diagnostics:panel.values.installed` / `.missing` after a package name: fr "installé / manquant", ru "установлен / отсутствует", es "instalado / falta".
14. `diagnostics:panel.cuda.unknownPython`: feminine for "version" in fr "(inconnue)", es "(desconocida)", pt-BR "(desconhecida)".
15. `diagnostics:panel.logTail`: Russian plural forms (one / few / many / other); Korean "#줄"; Chinese and Japanese use the single `other` form.
16. `diagnostics:panel.captured`: followed by a date and time (de "erfasst am", fr "capturé le", es "capturado el", ru "получено", ko "{timestamp}에 수집").
17. `common:localeBanner.inUse`, `.keepCurrent`, all nine. The language name is the capitalised endonym from `meta.json` ("Français", "Español", "Русский", "Português (Brasil)"). French, Spanish and Portuguese put it after a colon so the capital reads as a name; the buttons "Garder Français", "Mantener Español", "Manter Português (Brasil)", "Оставить Русский" still carry a capital a native writer would not use mid-phrase. Korean fixes the particle for "한국어" ("{endonym}로").
18. `common:localeBanner.changeHint`: must name the sidebar entry as each locale has it ("首选项", "偏好設定", "환경설정", "Préférences", "Preferencias", "Preferências", "Einstellungen", "Настройки", "設定").
19. `chat:compareView.useForSlot`, `.pickerTitleForSlot`, `.waitingFor` in Russian: the slot name is inserted in the nominative ("Использовать для Модель A"); the older `htmlChallenge.picker.*` strings have the same flaw.
20. `common:statusBanners.backendOffline.detail`: "until it is back" (de "bis es wieder da ist", ru "пока он не вернётся", pt-BR "até ele voltar").
21. `chat:keepGoing.stop.job_stalled`, `chat:keepGoing.waitingOn`: lower-case fragments that follow "Keep going stopped:"; "job" follows each locale's earlier choice (de "Auftrag", fr "tâche", es "trabajo", zh-TW "工作", zh-CN "任务").
22. `chat:emptyBanner.welcomeBody`, all nine: new second sentence; fr writes "Go", ru "ГБ", the others "GB".

Register:

23. German: Diagnostics uses "du" (as the Disk usage panel in the same file does); the Language panel's older strings use "Sie"; the banner is written without a pronoun. The mix is older than this branch.

## Final checks

Run at 10:55 to 10:59 on 2026-11-18, on `08d8f58d` (the last code commit).

| Check | Result |
|---|---|
| `npx tsc --noEmit` | exit 0 |
| `npx vitest run` | 156 files, 1850 tests, all pass (154 files / 1828 tests before this branch; 22 new) |
| `npm run i18n:validate` and `node scripts/i18n-validate.mjs --strict` | OK; nine locales 5040/5042 (100 %), no ICU error, no orphan |
| Playwright specs 32 to 38, `--workers=1`, backend 8950, Vite 8951 | 34 passed in 4.3 minutes (30 from specs 32 to 37 as stream D left them, 1 new banner test in 34, 3 in 38) |
| Playwright specs 34 and 38 on their own | 17 passed in 2.8 minutes |
| `pytest tests/i18n tests/test_hub_tracker.py` (throwaway `HOME`) | 21 passed |
| Editorial rule (no other desktop AI app named) | no hit in the branch diff |

How to rerun the browser part from this worktree (it has no `.venv`; `e2e/node_modules` is a git-ignored link to the main checkout's):

```
cd e2e && E2E_PYTHON=/Users/dan/ChaosEngineAI/.venv/bin/python E2E_WEB_PORT=8951 E2E_API_PORT=8950 \
  npx playwright test tests/34-locales.spec.ts tests/38-diagnostics-i18n.spec.ts --workers=1
```

The translations were applied from one table (key, English, nine translations) by a throwaway script that checks the placeholders of every translation against the English; it is not in the repo.
