# Klingon locale (tlhIngan Hol): assessment

Decision D-69. Date 2026-11-08. Branch `feat/klingon-locale` off `release/v0.10.0`.

## How locales work today

| Surface | Mechanism | Files | Strings |
|---|---|---|---|
| React UI | i18next + `intl-messageformat` post-processor (ICU plurals, `{var}`), lazy `import.meta.glob` per locale, `fallbackLng: "en"` | `src/i18n/index.ts`, `src/locales/<locale>/{common,chat,studio,library,setup,diagnostics,runtime,errors,meta,dashboard}.json` | 4134 leaf strings, 119 455 characters (studio 1633, common 1112, library 582, chat 401, runtime 169, setup 71, dashboard 66, diagnostics 62, errors 29, meta 9). 640 strings carry ICU placeholders, 85 are ICU plurals, 3 contain `<` tags. |
| Tauri shell (menu, tray, updater, dialogs) | `rust-i18n` compiled in from YAML, fallback `en`, `set_app_locale` IPC command | `src-tauri/locales/<locale>.yml` (not `.ftl`; the brief said `.ftl`, the repo uses YAML) | 35 strings |
| Backend (FastAPI) | gettext through Babel, `I18nMiddleware` negotiates `Accept-Language` or `settings.locale`, `localized_detail()` wraps user-facing `HTTPException` text | `backend_service/i18n.py`, `backend_service/locales/<locale>/LC_MESSAGES/messages.{po,mo}` | 11 literal strings reach `localized_detail(request, "...")`; the other ~68 call sites pass `str(exc)` (dynamic, untranslatable). `messages.pot` and the `en`, `ja`, `zh_CN` catalogs are header-only: nothing has ever been extracted, so every shipping locale has 0 backend strings. |

Validators: `npm run i18n:validate` (`scripts/i18n-validate.mjs`: parity, ICU compile with `intl-messageformat`, orphan scan, coverage table; the 95 % gate only bites with `--strict`), `src/locales/__tests__/parity.test.ts` (exact key parity, `null` allowed as placeholder), `src/i18n/__tests__/i18n.test.ts` (negotiation, hard-codes the 10-locale list), `scripts/i18n-pseudo.mjs` (en-XA / en-XB, dev-only), `tests/i18n/test_i18n_locale.py` (backend negotiation). `scripts/pre-build-check.sh` step 9 runs the validator warn-only.

Hard-coded locale lists that must all gain `tlh`: `src/i18n/index.ts` `SUPPORTED_LOCALES`, `src/locales/en/meta.json` `supportedLocales` (drives the picker and the first-launch banner), `scripts/i18n-validate.mjs` `LOCALES_LIST`, `src/locales/__tests__/parity.test.ts` `LOCALES`, `src/i18n/__tests__/i18n.test.ts` (asserts exactly 10), `backend_service/i18n.py` `SUPPORTED_LOCALES`. `normaliseLocale` already matches an unknown tag against the list by base, so `tlh` and `tlh-Latn` resolve once registered. The Tauri side needs only the new YAML file; `rust-i18n` picks up every file in the directory.

## What registering `tlh` needs

- **Picker and banner.** One `supportedLocales` entry with `endonym: "tlhIngan Hol"`. The picker shows the endonym only, plus a short note under the dropdown while `tlh` is active or selected ("fan translation, machine-assisted, corrections welcome"). The note is one new `common.json` key (`languagePanel.communityNote`), `null` in the nine other locales per the documented workflow. The first-launch banner never fires for `tlh` because no OS reports it.
- **Fallback to `en`.** Already the default (`fallbackLng: "en"`, `returnEmptyString: false`); a missing or `null` key renders English. Covered by a new vitest that initialises i18next with `tlh` resources and a deliberately missing key.
- **Intl formatting.** V8 has no CLDR data for `tlh`: `new Intl.NumberFormat("tlh").resolvedOptions().locale` is `en-US`, so numbers, bytes, dates and relative times silently use English formatting. Plural rules also fall back to English `one`/`other`; Klingon nouns do not mark plural, so every tlh plural uses a single `other` branch, which satisfies both the ICU grammar and the English rule set. `Intl.DisplayNames(["en"]).of("tlh")` is "Klingon", so the voice language list stays legible. A vitest pins the fallback behaviour.
- **Coverage gate.** Add an `EXPERIMENTAL_LOCALES` set in `scripts/i18n-validate.mjs`; those locales are still parity-, ICU- and orphan-checked and still appear in the coverage table (marked), but are skipped by the `--strict` 95 % threshold. Covered by a test that runs the validator against a fixture tree where the experimental locale is at 50 %.
- **Pseudo-locale.** No change; en-XA stays the layout stress test. Klingon romanisation is on average shorter than English, so layout risk is low; the long strings are the help texts, which wrap anyway.
- **Fonts.** None. Okrand's romanisation is plain Latin: the letters `a b ch D e gh H I j l m n ng o p q Q r S t tlh u v w y '`. It is case-sensitive (`q` and `Q` are different consonants, `I` is always capital, `D`, `H`, `S` always capital, everything else lower case). The private-use pIqaD code points are not used: no shipped font has them and they are not Unicode. Sentences do not start with a capital; only English proper nouns and technical tokens keep their case.
- **Backend.** Add `tlh` to `SUPPORTED_LOCALES`, write `backend_service/locales/tlh/LC_MESSAGES/messages.po` by hand with the 11 literal `msgid`s, compile with the main checkout's `pybabel`. `gettext.find` accepts a bare `tlh` directory.
- **Tauri.** `src-tauri/locales/tlh.yml`, 35 keys, same shape as `en.yml`.

## Quality limits, stated plainly

This is a fan translation produced by a language model that knows Okrand's published vocabulary and grammar but is not a fluent speaker and has no corpus for software. Expect:

- correct grammar for the common patterns (imperatives on buttons, `-lu'` passives for status lines, `-meH` purpose clauses, noun-noun compounds) and occasional errors in long help texts;
- around 80 coined terms, all marked in `glossary.md`, chosen so that a speaker recognises the parts even when the compound is new;
- English kept for everything that is a name or a token: model names, file formats, GPU, MLX, GGUF, CUDA, Metal, units, keyboard keys, API and product names, placeholders.

The back-translation check (`qa.md`) is the only quality measure in this pass; a speaker review is the next step and the UI says so.

## Scope of this pass

1. Register the locale, exclude it from the gate, add tests.
2. Translate the 10 frontend namespaces (4134 strings), then the Tauri YAML (35), then the backend `.po` (11).
3. Screenshots at 375 px and on Setup and Diagnostics through the Playwright harness on ports 8971/8972.
4. Self-QA: validators, tests, back-translation of every 15th string, usage page.

Out of scope: pIqaD, a Klingon speech model (voice stays English), the pseudo-locale, fixing the empty `en`/`ja`/`zh_CN` backend catalogs (a separate chore: the backend has never been extracted).
