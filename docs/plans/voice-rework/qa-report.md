# Voice rework QA report (branch feat/voice-rework, base 2b7b9e2)

Read-only review. Checked: diff, plan checklist, audit 1.x, pytest (test_voice_speak, test_mcp_bridge pass), vitest src/features/voice (65 pass), `npm run i18n:validate` (OK, 97.6 % non-en).

## Findings

| # | Sev | Where | Problem | Fix |
|---|---|---|---|---|
| 1 | blocker | `backend_service/routes/voice.py` `synthesize` (about line 283) | `POST /api/voice/synthesize` now calls `voice_speak.speak(...)` with the default `origin="ui"`. A body `{"voice":"clone:<id>"}` reaches `voice_clone.speak` with origin `ui`, skipping `clone_allowed`, the agent switch and `permissionConfirmed`. Reproduced with TestClient and `clone_allowed` forced False: the request reached `voice_clone.speak(origin='ui', permission_confirmed=False)`. Any agent token or other computer can make cloned speech. | In `synthesize`, reject ids that start with `clone:` (400) or run the same origin/`clone_allowed` guard as `/api/voice/speak`. Better: make `voice_speak.speak` require `origin` (no default) so a caller cannot forget. |
| 2 | blocker | `routes/voice.py` `speak` (the `body.voice.startswith(CLONE_PREFIX)` test) vs `voice_speak.py` `speak` (`voice.strip()`) | The route tests the raw string, `speak()` strips first. `" clone:abc"` (leading space) skips the guard and is treated as a clone with origin `ui`. Reproduced: `'clone:abc'` 403, `' clone:abc'` 200 with origin `ui`. | Strip in the route (or a Pydantic validator on `voice`) before the prefix test, and pass the stripped value on. Add a test for whitespace and for `/synthesize`. |
| 3 | major | `SpeakScreen.tsx:99-102`, `speakLogic.ts:chooseModel`, `voice_clone.py:271` | The UI picks the installed model (for clones it can be the 1.7B if only that is downloaded) and says "Runs with 1.7B", but the request carries no `model`. The backend uses `clone_model("")`, the default 0.6B, and fails with "not downloaded". Also `list_voices` marks clones `available` from the default base model only. The old "Cloning model" select is gone, so the user cannot choose 0.6B or 1.7B for clones (checklist item: MISSING as a control). | Send `model: note.model?.key` in `speakVoice`; make `voice_speak.list_voices` use "any base model installed"; if a choice is wanted, add a small model select to the model line for clones. |
| 4 | major | `SpeakScreen.tsx:131-134`, `routes/voice.py speak` | Cancel aborts the browser fetch only. The sync route keeps running (up to 15 min) and holds the `HEAVY_JOBS` "voice" slot, so the next Generate after Cancel waits behind it. | Abort should reach the worker: on client disconnect call `voice_client.stop_all()` (as MCP job cancel does) or expose a cancel route. At least say "Stopping after this part" in the UI. |
| 5 | major | Plan B / checklist "Voice Models on the shared ModelsPage shell" | Not done: `TtsModelsPanel` is added to the bespoke `VoiceModelsTab`. The plan lists this as a work package. Not a regression. | Either do the shell move or record the deviation in decisions.md. |
| 6 | minor | `speakLogic.ts` / `SpeakScreen.tsx:80` | For a cloned voice, `[pause]` becomes a blank line and the part-by-part progress and per-part cancel disappear (one request, up to 20,000 characters). Checklist "part-by-part with cancel" holds for built-in and preset voices only. 200k characters holds for those too; clones cap at 20k (was 5k). | Accept and document, or split clones across requests as well (each is saved to the Library, so would need a join). |
| 7 | minor | `SpeakScreen.tsx:55`, `:182-194` | Style text is shared across voices: a preset's "Speak in an angry tone." becomes the voice description when the user switches to Design; and stays hidden but set on built-in voices. | Keep one style value per mode, or clear on voice-kind change. |
| 8 | minor | `SpeakScreen.tsx:244-266` | Model line shows only "Downloading {name}…"; the old "Installing the watermark component…" phase text is missing in Speak (still in Models rows). Disabled style box has its reason in a separate paragraph, not `aria-describedby`. | Show `download.phase`; add `aria-describedby` to the style box. |
| 9 | minor | `voice_clone.py` (error text "Download it in Clone a voice first."), `voice_speak.py` | Stale wording: there is no Clone screen now. | "Download it in Voice > Models." |
| 10 | minor | `VoicePicker.tsx:39` | Cards are `role="radio"` buttons in a `radiogroup` with no arrow-key roving, every card is a tab stop. | Add roving tabindex or drop the radio roles (use `aria-pressed`). |
| 11 | minor | `src/styles.css` (`.voice-card--selected`) | Hardcoded `rgba(143, 180, 255, 0.12)` instead of a token. | Use an accent token with alpha. |
| 12 | minor | `SpeakScreen.tsx` (split in browser, 4000-char parts) | A code fence cut across two 4000-char parts is cleaned per part, so the second half is read as prose. Rare. | Split on paragraph boundaries outside fences, or send the whole text for non-clone voices (backend already chunks at 400). |
| 13 | minor | `useVoiceSample.ts` | A preset sample loads a 2.5-3 GB model with no loading indicator beyond the disabled state, and keeps playing when the user switches Studio mode. | Show a spinner state; stop on mode change. |

## Checklist (plan.md "Must-keep")

| Item | Status | Where |
|---|---|---|
| Transcribe (all controls) | KEPT | `TranscribePanel` untouched (not in diff); still mounted as a Studio mode |
| Speak: 200k chars, blank-line paragraphs, `[pause]`, speed | KEPT | `SpeakScreen.tsx` (`MAX_SPEAK_CHARS`, `splitSpeech`, slider) |
| Speak: voice preview | KEPT | `useVoiceSample`, card play button |
| Speak: part-by-part with cancel | KEPT for built-in and preset; changed for clones (finding 6); backend does not stop (finding 4) | `SpeakScreen.tsx` |
| WAV, MP3, Save to Library, Send to Storyboard | KEPT | `SpeakResultRow.tsx` |
| Clone: standing statement | KEPT, shown on Voices and when a clone is selected (hidden when cloning is not allowed here) | `CloneStatement` |
| Clone: model select | MISSING as a control (auto choice, see 3); download MOVED to the model line and Voice Models | `SpeakScreen`, `TtsModelsPanel` |
| Clone: voice list with consent line, delete with confirm | MOVED | `VoicesScreen.tsx`, `VoicePicker.tsx` |
| Clone: new-voice form, AI label badge | KEPT / MOVED | `NewVoiceForm` (unchanged) in Voices; badge in `SpeakResultRow` |
| Clone: agent-cloning switch | MOVED | `VoicesScreen.tsx` bottom (only when cloning is allowed) |
| Clone: not-Apple-Silicon message | KEPT | `VoicesScreen` `cloneNote` |
| Voice Models: pack, STT, voices, add-on, settings, unload | KEPT | `VoiceModelsTab.tsx` (one added panel) |
| Chat voice (mic, read-aloud, call): `ttsVoice`, `speed` | KEPT | `/api/voice/speak/stream` and settings untouched |
| Library speech and transcripts | KEPT | `saveSpeech` payload unchanged; clone items saved by backend |
| Storyboard send | KEPT | `SendToStoryboardMenu`, same `SendItem` shape |
| MCP `speak`, `list_voices` | KEPT, fields added (`pitch`, `style`, `kind`); golden updated | `mcp_bridge/tools.py`, `models.py`; `speak_cloned` unchanged |
| Nav: Models first, opens on Models until the pack is installed, stale copy fixed | KEPT | `sidebarGroups.ts`, `tabs.ts`, `App.tsx`, `studio.json` |

## Review items with no finding

Request shape matches (`SpeakBody` vs `SpeakRequest`: text, voice, style, speed, pitch, seed, model, permissionConfirmed). Bounds agree (speed 0.5-2, pitch -12..12, seed 0..2^31-1, text 20,000 per request, style 500). Style gating is consistent between `styleState` and `build_plan` (0.6B adds a note and drops the instruction). WAV join checks matching formats. ffmpeg is called list-form, in a temp dir, with no shell; the filter chain is correct for pitch plus speed. `_free_tts` runs before each model load. Heavy imports are lazy. No emoji, no competitor names, no legal-style wording in added code, docs or commit messages. i18n validate passes; chip and filter fallbacks have keys.

## Visual check

See the end of this file (appended after the browser run).

Run: the app's own e2e mocks (`e2e/tests/voiceMocks.ts`) on Vite port 8941 / backend 8942 (loopback, throwaway HOME), headless Chromium at 1280x900 and 375x812, screenshots read from `/private/tmp/claude-501/-Users-dan-ChaosEngineAI/f7a58925-b662-4665-b502-558c156b2bac/scratchpad/qa/` (`desktop-speak.png`, `desktop-voices.png`, `desktop-models.png`, `mobile-*.png`). Voice pack, voices and models were mocked, so no audio was made.

- No horizontal page scroll on Speak, Voices or Voice Models at 1280 or 375 px (measured `scrollWidth - clientWidth` = 0 on all six).
- Speak (desktop): your voices, natural voices, design card and quick voices as cards; style box and chips greyed out for a quick voice with the reason line; sliders, Advanced, model line and a "Generate audio (2 parts)" button all line up. No clipping.
- Voices (desktop): three action cards, statement, filters, one list with Play sample, Use and Delete, agent switch at the bottom. Fine. At 375 px the action cards wrap to two columns, no overlap.
- Voice Models at 375 px: pack panel fits. On the full Voice Models page the new speech-model rows were not inspected below the fold beyond the e2e assertions.
- Not visible in the screenshots but seen: on phones the sidebar sits above the page and shows only the active child (existing behaviour, not from this branch).
- The repo's own `e2e/tests/14-voice-speak.spec.ts` failed 7 of 7 when run cold here: the Terms dialog appears after the 3 s `openApp` check on a cold Vite start, so the dialog blocks the run. A scratch run that waited for it passed. This looks like a flaky helper (`e2e/tests/helpers.ts` openApp, 3000 ms timeout), not a defect in the voice code. Minor: raise that timeout.
