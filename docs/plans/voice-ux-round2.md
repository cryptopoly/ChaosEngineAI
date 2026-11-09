# Voice UX, round two: review and redesign

Date: 2026-11-08. Branch `docs/voice-ux-round2`, cut from `release/v0.10.0` (`e86e56a9`). Review and design only; no source changed. Inputs: the owner's four complaints, the code on the branch, `docs/plans/voice-rework/` (plan, audit, model research, QA report), and screenshots taken with the Playwright harness in `e2e/` against the shared voice mocks (`e2e/tests/voiceMocks.ts`, Vite 8941, backend 8942, throwaway HOME, desktop 1280 px and 375 px). Mock-ups of the proposed screens: [voice-ux-round2/mockups.html](voice-ux-round2/mockups.html).

Line numbers are from this branch's working tree.

D-67 progress:
- 2026-11-08: W1-W4 built on feat/voice-ux-a.
- 2026-11-08: W5, W7, W9, W10 built on feat/voice-ux-b.
- 2026-11-09: W6 built on feat/voice-w6 (D-72): any voice in Chat read-aloud, a separate call voice (`callVoice`), Kokoro fallback with a note, measured chat cost in the pickers (`measure-voice-models.py --chat`, table in `docs/usage/voice.md`). E2E is spec 19 (16 was taken). Designed voices wait for W8; the owner's real call is FU-156.

## 1. What is true today, per complaint

### 1.1 "Speak offers only Kokoro voices and defaults to Kokoro"

Partly true. The better models are on the Speak screen, but a Kokoro voice is always what is selected when you arrive, whatever is installed.

- The Speak screen's starting voice is `settings.ttsVoice` (`src/features/voice/VoiceStudioTab.tsx:31`), whose default is `af_heart` (`backend_service/voice_runtime.py:197`, `:232`). When that voice is not in the list, `defaultVoice()` falls back to the first voice whose `kind === "builtin"`, which is always Kokoro (`src/features/voice/speak/speakLogic.ts:119-122`). There is no "best installed voice" rule anywhere. Screenshot `d08`: with every Qwen3-TTS model installed and a cloned voice present, Speak still opens on "Heart".
- The picker (`src/features/voice/speak/VoicePicker.tsx:148-237`) shows four groups in this order: Your voices (clones), Natural voices (the nine Qwen3-TTS CustomVoice presets), one "Design a voice from a description" card, then Quick voices (Kokoro) behind a language select. "Natural" and "Quick" are never explained; the word "Kokoro" appears only in the model line at the bottom ("Runs with Kokoro, part of the voice pack", `SpeakScreen.tsx:283`).
- A preset is clickable even when its model is not downloaded (cards are disabled only on non-Apple machines, `VoicePicker.tsx:174`). The person finds out at the bottom of the form: "Needs Qwen3-TTS 0.6B CustomVoice (2.5 GB), not downloaded yet. Download" (`SpeakScreen.tsx:291-297`). Screenshot `d06`: with nothing downloaded, choosing Ryan shows a greyed style box with the note "The small model ignores style. The 1.7B model reads it." before any model exists, which reads as a riddle.
- Which model runs is decided by `chooseModel()` (`speakLogic.ts:23-43`) and mirrored by `_pick_custom()` (`backend_service/voice_speak.py:157-164`): smallest installed model first, 1.7B only when the style box has text. With both CustomVoice models installed, typing a style switches the running model, which means a reload (about 4 s, FU-141). FU-148 records that the 0.6B rows are not quicker on the M4 Max, so "smallest first" buys nothing.
- English presets are only Ryan and Aiden, both male (`backend_service/catalog/voice_models.py:163-164`, research-models section 1). A female English natural voice needs VoiceDesign, which is not saved between runs (FU-141, "saved designed voice" not built), so each Generate re-rolls the voice unless a seed is set under Advanced.
- Chat is Kokoro-only by construction. Read-aloud (`src/features/voice/ReadAloud.tsx:29`), auto-read (`src/features/chat/ChatTab.tsx:228-233`) and voice calls (`src/features/voice/conversation/useVoiceCall.ts:110,140`) send `settings.ttsVoice` to `POST /api/voice/speak/stream` (`backend_service/routes/voice.py:375-400`), which calls `voice_runtime.synthesize`, and that rejects anything that is not a Kokoro id (`voice_runtime.py:428-430`). `SettingsRequest.ttsVoice` is capped at 32 characters (`routes/voice.py:56`). So the better voices cannot be the chat voice at all.

What a first-time user picks: the pre-selected Kokoro "Heart". The natural voices sit above it but say nothing about why they are better, what they cost, or that they need a download; the cost line is below the fold at 1280x900 (screenshot `d05`, first run).

### 1.2 "The 54 Kokoro voices are listed on the Voice Models page"

True. `VoicesPanel` on the Models page (`src/features/voice/VoiceModelsTab.tsx:319-405`) renders every Kokoro voice as a chip grouped by language, with a play button, and clicking a chip sets the chat voice (`saveSettings({ ttsVoice })`, line 376). Its subtitle hardcodes "Kokoro: 54 voices in 9 languages. Pick the one chat reads aloud with." (`src/locales/en/studio.json:691`). The same voices are listed again on Voice Studio > Voices (`src/features/voice/speak/VoicesScreen.tsx`), so there are two voice lists with two purposes, two preview sentences (`VoiceModelsTab.tsx:26-36` and `useVoiceSample.ts:9-21`) and no link between them.

Also on that panel: the Mandarin add-on button (`VoiceModelsTab.tsx:350-364`), Japanese voices listed but permanently unavailable on the MLX lane (`voice_runtime.py` comment near line 200; screenshot `d13` shows a greyed Japanese row), and play/stop buttons drawn as the text glyphs "▶" and "■" (`VoiceModelsTab.tsx:394`), which breaks the monochrome SVG icon rule.

### 1.3 "Models to Studio to Library is clunky"

Mostly true. The pieces exist; the connections between them do not.

Sidebar and order. The Voice group lists Models, Studio, Library (`src/constants/tabs.ts:46-48`) and opens on Models until the pack is installed, then on Studio (`src/constants/sidebarGroups.ts:18`). That matches Music. Voice Studio then opens on Transcribe (`VoiceStudioTab.tsx:28`), so a person who came for speech is two clicks from Speak and sees a file-drop zone first (screenshot `d04`).

Studio header. Music Studio uses `StudioShell` with Models and Library buttons in its header (`src/features/music/MusicStudioTab.tsx:158`); Voice Studio does not use the shell and has no such buttons (`VoiceStudioTab.tsx:78-118`). Music Models and Music Library both have an "Open Studio" button (screenshots `d16`, `d18`); Voice Models and Voice Library have none (screenshots `d13`, `d14`). The only links out of Voice Studio are the gate ("Open Voice Models", `VoiceStudioTab.tsx:69`) and the "Get more voices" card on Voices (`VoicesScreen.tsx:99-105`).

Models page. A custom page, kept by the owner's decision (`docs/plans/voice-rework/plan.md:44`). Four panels: voice pack, speech-to-text models, speech models (`TtsModelsPanel`), Kokoro voices, voice settings. The speech-model rows show size, language count and the catalog summary with capability chips (`src/features/voice/speak/TtsModelsPanel.tsx:19-28`); no "use it" action, so after downloading a model there is no path back to Speak. The settings panel (`VoiceModelsTab.tsx:407-494`) holds chat behaviour (auto-send, auto-read, reading speed, call pause, interrupt) next to runtime housekeeping (idle unload, unload now).

Voices tab. One list with Quick / Natural / Cloned badges and All / Mine / Built-in filters (`VoicesScreen.tsx:13-14,155-191`). "Built-in" includes the Qwen3 presets (`speakLogic.ts:107-111`), the design voice is left out of the list (`VoicesScreen.tsx:39`), and cloned voices have no sample button (`useVoiceSample.ts:24-26`, FU-146). "Get more voices" says "Add Mandarin Chinese voices and more from the voice pack" and opens the Models page at the top, not at the add-on.

Library. Speech made with a quick or natural voice is only saved when the person presses Save to Library (`src/features/voice/speak/SpeakResultRow.tsx:42-47,73`); cloned speech is always saved by the backend (`voice_speak.py:439-456`). Library cards print the raw voice id in the meta line (`af_heart`, `clone:voice-1`; `src/features/voice/VoiceLibraryTab.tsx:85-91`, screenshot `d14`), carry no AI-made badge for cloned clips, and offer Play, Export, Delete and Send to Storyboard but no way to reopen the text in Speak or speak it again with another voice (`VoiceLibraryTab.tsx:118-172`). Transcripts do have Open (back to the Transcribe editor via `voiceOpenRequest`, `src/app/VoiceTabs.tsx:47-60`).

Hand-offs. Transcribe to Chat exists (Summarise, Action items, Meeting notes, `src/features/voice/studio/TranscriptEditor.tsx:121,259-263`). Speak to Projects exists (Send to Storyboard on the result row and on Library cards). Chat to Voice exists only as "open Voice Models" from the mic, the speaker button and the Call info (`ChatTab.tsx:217-218`: "speech recognition (Voice Models)", "a voice for replies (Voice Models)"). Nothing goes from Chat to Speak, and the Call info points at a page that lists 54 chips to choose the call voice from.

Music and Audio. The same group pattern (Models, Studio, Library) is better connected: `StudioShell` header buttons, "Open Studio" on Models and Library, a model picker with the install state in the Studio header, measured peak memory and a Fit badge per model row on the shared Models shell (screenshot `d16`). Two flow problems are shared with Voice: the Studio gate when the pack is missing sends you to Models with no way to install from the Studio itself, and the Library's empty state says "Open Studio" without saying which mode (Music or Sounds). Sounds has its own licence screen and model picker in the header, which works. Music is the pattern Voice should copy, not the other way round.

Mobile. At 375 px the expanded sidebar fills the first screen on every page (screenshot `m03b`), so Speak starts below the fold. This is app-wide, noted in the QA report, not a voice defect. No sideways scroll on any voice screen (the e2e `375 px` test in `e2e/tests/14-voice-speak.spec.ts:207-237` holds).

### 1.4 "Make the most of the model capabilities, with measured numbers"

The capabilities are wired; the explanation and the numbers are not.

- Style instruction: the box and six chips exist (`SpeakScreen.tsx:196-233`, chips in `speakLogic.ts:11-18`), greyed with a reason for quick, cloned and small-model cases (`speakLogic.ts:63-71`). Correctly 1.7B only. The chip wording is sent verbatim ("Speak in an angry tone."); nothing shows an example of what free text can do (pace, accent, whisper).
- Voice design: one card and a description box (`SpeakScreen.tsx:198-211`). Nothing says the voice changes on every run or how to keep it (seed under Advanced, or FU-141's design-then-clone). No sample.
- Cloning with consent: the form, the standing statement and the permission tick are intact (`src/features/voice/clone/NewVoiceForm.tsx`, `CloneStatement`). On Speak the statement is injected between the picker and the style box when a clone is selected (`SpeakScreen.tsx:194`, screenshot `d11`), which pushes the controls down every time you switch voice.
- Speed and pitch: two sliders (`SpeakScreen.tsx:235-249`). For every non-Kokoro voice both are ffmpeg post-processing (`voice_speak.py:14-15,323-344`); if ffmpeg is missing the error appears only after generation (`voice_speak.py:55`). The screen does not say which voices do speed natively.
- Languages: Kokoro voices are per language (nine), Qwen3 presets are one language each and the design voice says "Any"; every Qwen3 request is sent with `language: auto` (FU-141 lists a `language` field as not built). The picker has a language select for quick voices only.
- Mandarin add-on: only reachable from the Kokoro panel on the Models page.
- Kokoro quick English voices: all 20 American voices are shown as equals; the model author's grades (af_heart A, af_bella A-, af_nicole B-, most others C or below; research-models section 2) are not used anywhere.
- Measured numbers: the catalog has them for all five Qwen3-TTS rows (basis "measured", M4 Max 64 GB, 2026-11-08: cold start, seconds per 10 s of speech, peak and idle memory; `backend_service/catalog/voice_models.py:40-50,66-76,92-102,118-128,144-154`). `voice_clone.list_models()` does not copy `estimates` into the API row (`backend_service/voice_clone.py:70-93`), `VoiceModelRow` has no such field (`src/types/voice.ts:178-190`), and `grep estimates src/features/voice` finds nothing. So Models shows "3.08 GB · 10 languages" and Speak shows "Runs with Qwen3-TTS 1.7B CustomVoice." with no speed or memory. Music already has the pattern to copy: `MusicEstimates` (`src/types/music.ts:32-34,78`) and the estimate strip (`src/features/music/musicLogic.ts:193-233`). Kokoro has no measured row at all (research-models calls its speed "very fast, unmeasured on this Mac").

## 2. Problems, ranked

Blockers (they defeat the point of the rework: nobody will use the better voices).

1. Speak always starts on a Kokoro voice; there is no best-installed rule and no prompt to install a better voice when none is there (`speakLogic.ts:119-122`, `voice_runtime.py:197`).
2. Chat read-aloud, auto-read and calls cannot use a natural, designed or cloned voice: the stream route accepts Kokoro ids only (`routes/voice.py:375-400`, `voice_runtime.py:428-430`, `routes/voice.py:56`).
3. The measured cost of each model never reaches the screen (`voice_clone.py:70-93`); the picker and the model line give the person no basis for choosing.

Major.

4. Two voice lists with different jobs: the Kokoro chip panel on Models (chat voice) and the Voices tab in Studio (Speak voice), with different preview sentences and no link (`VoiceModelsTab.tsx:319-405`, `VoicesScreen.tsx`).
5. No cross-links: Studio has no Models or Library buttons, Models and Library have no Open Studio, Library speech cards cannot return to Speak (`VoiceStudioTab.tsx:78-118`, `VoiceLibraryTab.tsx:118-172`). Music has all of these.
6. Studio opens on Transcribe (`VoiceStudioTab.tsx:28`); Speak is second.
7. Picker groups are unexplained ("Natural", "Quick"), presets are selectable before download with the consequence shown at the bottom, and the design card hides that the voice is not stable (`VoicePicker.tsx:158-202`, `SpeakScreen.tsx:279-303`).
8. Model choice flips between 0.6B and 1.7B as the style box is typed in, forcing a reload, and the 0.6B rows are not faster (FU-148; `speakLogic.ts:34`, `voice_speak.py:159`).
9. Chat behaviour settings (auto-read, reading speed, call pause, interrupt, auto-send) live on the Models page, two groups away from Chat (`VoiceModelsTab.tsx:407-494`).
10. Library meta prints raw voice ids and no AI-made badge; non-cloned speech is lost unless Save to Library is pressed (`VoiceLibraryTab.tsx:85-91`, `SpeakResultRow.tsx:42-47`).
11. The Mandarin add-on and the "Get more voices" path are only discoverable on the Models page (`VoiceModelsTab.tsx:350-364`, `VoicesScreen.tsx:99-105`).
12. Speed and pitch need ffmpeg for every non-Kokoro voice and the screen does not say so until a failure (`voice_speak.py:55`).

Minor.

13. Text glyph play/stop buttons on the Models page (`VoiceModelsTab.tsx:394`); icon rule.
14. Hardcoded "54 voices in 9 languages" (`studio.json:691`).
15. Japanese Kokoro voices listed although never available on the MLX lane.
16. The clone statement is injected mid-form on Speak (`SpeakScreen.tsx:194`).
17. Cloned and designed voices have no sample button (FU-146).
18. The Kokoro quality grades are unused; all 20 American voices look equal.
19. The e2e voice mocks list two Kokoro voices (`voiceMocks.ts:64-65`), so the real density of the quick-voice group (20 cards) is never checked at 375 px.
20. The chips send fixed sentences; no free-text examples for style; no mention that the description also sets pace and accent for a designed voice.
21. The word "Voices" names both the Studio tab and the Kokoro panel on Models.

## 3. Proposed redesign

### 3.1 Information architecture

Keep the three sidebar children, in the same order as every other area, and make Studio open on Speak once the pack is installed.

| Sidebar child | Holds | Opens on |
|---|---|---|
| Voice > Models | Voice pack (install, update, repair). Speech models (the five Qwen3-TTS rows with capability chips and measured cost, each with "Use in Speak"). Speech-to-text models. One line for Kokoro ("Quick voices: 54 in 9 languages, part of the voice pack", with "Add Mandarin voices" and "Browse voices"). Runtime: idle unload, unload now. Header button: Open Studio. | top |
| Voice > Studio | Modes Speak, Transcribe, Voices in that order, with the shared `StudioShell` header (Models and Library buttons). Last mode remembered per window. | Speak |
| Voice > Library | Transcripts and speech. Header button: Open Studio. Speech cards: Play, Speak again, Export, Send to project, Delete. | top |

Where the pieces move:

- The 54 Kokoro voices leave the Models page. They live in Speak's picker (Quick group, curated shortlist with "All voices" expander) and on the Voices tab (filter Quick).
- The chat voice is chosen on the Voices tab ("Use in Chat" on any row; the current chat voice has a badge). The Kokoro chip panel's job moves here.
- Chat behaviour settings (auto-send, auto-read, reading speed, call pause, interrupt) move to a "Chat and calls" panel at the bottom of the Voices tab, since that is where the chat voice is chosen. Runtime housekeeping (idle unload, unload now, worker status) stays on Models. If the owner prefers Settings > Voice for the chat behaviour, that is open question 4.
- The Mandarin add-on becomes a card in the Quick group when the language select is Chinese, and a row on the Voices tab under Quick, as well as the line on Models.

### 3.2 The Speak screen

Top to bottom: text box; voice picker; style (or description) box with chips; speed and pitch; cost and model line; Generate; result row. The picker is the change.

Picker groups, in this order, each a row of cards with a heading and a one-line explanation:

1. Recommended. Three or four cards computed for this machine and the UI language: the best installed natural voice (the 1.7B CustomVoice preset that matches the UI language, or the first preset), the top-graded quick voice for the UI language (af_heart for English), the most recent cloned voice, and Design a voice. When no natural model is installed, the first card is "Install a better voice" with the 1.7B CustomVoice download inline, its measured cost (3.08 GB download, 3.5 GB in memory, about 2.5 s per 10 s of speech on the M4 Max) and one sentence on what it adds (style instructions, more natural prosody). The card dismisses for the session.
2. Quick (Kokoro). "Fast, under 1 GB, speed only. 54 voices in 9 languages." A language select, the shortlist for that language (graded B- or better; for American English: Heart, Bella, Nicole, Michael, Fenrir, Puck), then an "All voices in this language" expander. For Chinese, an "Add Mandarin voices" card when the add-on is missing. Japanese is hidden on lanes where it can never be available.
3. Cloned. Your cloned voices with the consent line, then a "New cloned voice" card that opens the existing form in a drawer. The standing statement sits inside this group's heading, not in the form below.
4. Designed. Saved designed voices (once FU-141's design-then-clone lands) and the "Design a voice" card. The card's detail says "A new voice every run unless you set a seed or save it."
5. More. The natural presets in other languages (Chinese, Japanese, Korean), and a "Get more voices" card that opens Models at the speech-model rows.

Each card: name, language and gender, one-line feel, a cost chip (Quick: "fast, 0.3 GB"; Natural: "3.5 GB, 2.5 s per 10 s"; Clone: "4.3 GB"), a play button where a sample exists, and the install state inline ("Download 3.08 GB" on the card itself when the model is missing, replacing the bottom-of-form discovery). A card whose model is missing is still selectable so the download can be started from it, but Generate stays off with the reason on the card.

Default-voice rule (replaces `defaultVoice()`): the voice last used in Speak (a new `speakVoice` setting, separate from the chat voice); else the best installed natural voice for the UI language; else the top-graded quick voice for the UI language; else the first available voice. The chat voice setting is no longer the Speak default.

Model choice (replaces the smallest-first rank): when a style model is installed, always run it for presets; the 0.6B CustomVoice is used only when it is the only one installed. No model switch when the style box is typed in.

Cost and model line: "Runs with Qwen3-TTS 1.7B CustomVoice, 3.5 GB in memory, about 2.5 s per 10 s of speech on this Mac (measured 2026-11-08). Manage in Models." For Kokoro: "Runs with Kokoro, part of the voice pack, under 1 GB." When ffmpeg is missing and speed or pitch is not neutral on a non-Kokoro voice: "Speed and pitch for this voice need ffmpeg. Install it from Settings." shown before Generate, not after.

Style box: unchanged behaviour, plus a placeholder that shows free text covering tone, pace and accent ("A tired, gentle bedtime voice, slow, slight Irish accent."). Chips stay. For a designed voice the label is "Describe the voice" and the hint says what to describe (age, gender, pitch, tone, accent, pace) and that the seed keeps it.

Result row: unchanged controls, plus "Saved to Library" as a link that opens the Library filtered to Speech when autosave is on (open question 1).

### 3.3 The Voices tab

The voice manager. Filters: All, Recommended, Quick, Natural, Cloned, Designed; a language select. Each row: name, kind badge, language and gender, feel, cost chip, Play sample (also for clones and designed voices, FU-146), Use in Speak, Use in Chat, and Delete for clones. Action cards at the top: New cloned voice, Design a voice, Add Mandarin voices (when missing), Get more voices (opens Models at speech models). The "Chat and calls" panel at the bottom holds the chat voice (read-only, set by Use in Chat), reading speed, auto-read, auto-send, call pause and interrupt. The agent-cloning switch stays where it is.

### 3.4 The Models page

Same page, trimmed and linked: header button Open Studio; speech-model rows gain the measured cost cells (download, memory, seconds per 10 s, cold start, measured on) and a "Use in Speak" button (installed) or "Download" (missing), and the row for the 0.6B CustomVoice carries the catalog's own note that it is no quicker than the 1.7B; the Kokoro panel shrinks to one line with "Add Mandarin voices" and "Browse voices"; the settings panel keeps only idle unload, unload now and dictation language and model. Play buttons that remain are SVG.

### 3.5 Models and Studio link both ways

- Studio header (`StudioShell`): Models and Library buttons, as Music.
- Models: Open Studio in the header; "Use in Speak" per speech-model row (opens Speak with that model's first preset, or the design voice, selected); "Browse voices" opens Studio > Voices.
- Speak: "Manage in Models" on the cost line; the install card in Recommended downloads without leaving the screen; "Get more voices" in More opens Models anchored at the speech models.
- Library: Open Studio in the header; "Speak again" on a speech card reopens Speak with the saved text and voice (extends `voiceOpenRequest` with a speech variant).
- Chat: the Call info and the read-aloud tooltip name the voice in use and link to Studio > Voices ("Change voice"), not to Models, once the pack is installed.

### 3.6 The Library speech view

Cards show the voice name (looked up from the voice list, with the kind badge Quick / Natural / Cloned / Designed), duration, date, the AI-made badge for cloned clips, the project chip, and the first line of the text. Actions: Play, Speak again, Export (WAV, MP3), Send to project, Delete. Filter chips: All, Transcripts, Speech; a voice filter when there are more than ten speech items. If autosave is adopted, everything Speak makes appears here within a second, and the result row's Save button becomes a link.

### 3.7 Music and Sounds

No structural change. Two small fixes for parity: let the Studio gate install the pack from the Studio itself (Voice gets the same), and have the Music Library empty state name the mode ("Open Studio > Music").

## 4. Work breakdown

Sizes: S under half a day, M one to two days, L three days or more, including tests. Every package keeps the must-keep checklist in 4.1 green.

| # | Package | Size | Files | New tests |
|---|---|---|---|---|
| W1 | Measured cost on screen: pass `estimates` through the API, add it to `VoiceModelRow`, cost chips on picker cards, the cost line on Speak, cost cells on Models rows | S | `backend_service/voice_clone.py` (`list_models`), `src/types/voice.ts`, `src/features/voice/speak/speakLogic.ts` (`costLabel`), `VoicePicker.tsx`, `SpeakScreen.tsx`, `speak/TtsModelsPanel.tsx` | pytest `tests/test_voice_clone.py` (rows carry `estimates` with basis and measuredOn); vitest `speak/__tests__/speakLogic.test.ts` (`costLabel` for measured, unmeasured, Kokoro); e2e `14-voice-speak` row text "3.5 GB" and "2.5 s per 10 s" |
| W2 | Default-voice rule and the install card: new `speakVoice` setting, `defaultVoice(voices, models, settings, uiLanguage)`, the "Install a better voice" card with inline download, no model flip on style (prefer the installed style model) | M | `backend_service/voice_runtime.py` (`DEFAULT_SETTINGS`), `backend_service/routes/voice.py` (`SettingsRequest`), `backend_service/voice_speak.py` (`_pick_custom`), `speakLogic.ts` (`defaultVoice`, `chooseModel`), `VoiceStudioTab.tsx`, `VoicePicker.tsx`, `SpeakScreen.tsx`, `VoiceContext.tsx` | pytest `tests/test_voice_speak.py` (style model preferred when installed; 0.6B only when alone), `tests/test_voice_io.py` (settings round trip); vitest `speakLogic.test.ts` (rule order: last used, best natural, top quick, first); e2e: fresh install shows the card, download body, Speak default after install |
| W3 | Picker regrouping: Recommended, Quick (shortlist plus expander, Mandarin card, Japanese hidden), Cloned (statement in the heading, New voice drawer), Designed, More; grades on Kokoro rows from the backend | M | `backend_service/voice_runtime.py` (`list_tts_voices` gains `grade`), `voice_speak.py` (`list_voices` passes it), `src/types/voice.ts`, `speakLogic.ts` (`groupVoices`, `KOKORO_SHORTLIST` by grade), `VoicePicker.tsx`, `SpeakScreen.tsx`, `src/styles.css`, `e2e/tests/voiceMocks.ts` (all 20 American voices) | pytest `tests/test_voice.py` (grade present, Japanese hidden on MLX); vitest `speakLogic.test.ts` (grouping, shortlist, expander); e2e `14-voice-speak` groups and 375 px with 20 quick cards |
| W4 | Links and order: Studio on `StudioShell` with Models and Library, opens on Speak and remembers the mode; Open Studio on Models and Library; "Use in Speak" per model row; "Manage in Models" on the cost line | S | `VoiceStudioTab.tsx`, `VoiceModelsTab.tsx`, `TtsModelsPanel.tsx`, `VoiceLibraryTab.tsx`, `src/app/VoiceTabs.tsx`, `src/app/context.ts` (open request carries a mode and a voice), `e2e/tests/helpers.ts` | vitest `VoiceStudioTab.test.ts` (default mode, remembered mode); e2e `14-voice-speak` (Models > Use in Speak lands on Speak with the preset selected), `11-voice-pack` (Open Studio) |
| W5 | Voices tab as the manager: filters, Use in Chat, samples for clones and designed voices, Mandarin and Get-more cards, the "Chat and calls" panel; Kokoro panel off Models (one line left) | M | `VoicesScreen.tsx`, `VoiceModelsTab.tsx` (remove `VoicesPanel`, keep the add-on line and runtime settings), `useVoiceSample.ts` (clone and design samples through `/api/voice/speak`, not saved), `voicePackState.ts`, `src/locales/*/studio.json` | vitest new `speak/__tests__/VoicesScreen.test.tsx` (Use in Chat writes `ttsVoice`; filters); e2e `11-voice-pack` (add-on from Voices and from Models), `14-voice-speak` (Use in Chat badge) |
| W6 (done 2026-11-09, `feat/voice-w6`) | Any voice in Chat: `speak/stream` plans through `voice_speak` per sentence, `ttsVoice` up to 64 characters and validated against `list_voices`, cloned voices allowed for read-aloud only when the clone rules pass, calls keep the natural or quick voice; Call info names the voice and links to Voices | L | `backend_service/routes/voice.py` (`speak_stream`, `SettingsRequest`), `voice_speak.py` (a `render_sentence` entry for the stream), `voice_runtime.py`, `src/features/chat/ChatTab.tsx`, `ReadAloud.tsx`, `useVoiceCall.ts`, `speechPlayer.ts`, `docs/usage/voice.md` | pytest `tests/test_voice_speak.py` (stream with a preset, with a clone refused for agents), `tests/test_voice_io.py` (settings accept `qwen3:Ryan`); vitest `src/api/voice.test.ts`; e2e new `16-chat-voice.spec.ts` (read-aloud request body with a preset voice; Call info text); a real run on the M4 Max: memory held by the 1.7B worker during a call, barge-in latency |
| W7 | Library speech view: voice name and kind badge, AI-made badge, Speak again, optional autosave of every Speak result with "Saved to Library" as a link | M | `VoiceLibraryTab.tsx`, `SpeakResultRow.tsx`, `SpeakScreen.tsx`, `src/app/VoiceTabs.tsx`, `backend_service/voice_library.py` (store the voice name and kind), `backend_service/routes/voice.py` (`/library/speech` fields) | pytest `tests/test_voice_library.py` (name and kind persisted; autosave item shape); vitest `VoiceLibraryTab` card meta; e2e `03-libraries-projects` (badge, Speak again lands on Speak with text and voice) |
| W8 | Saved designed voices: design, then clone from the clip, listed under Designed (FU-141) | L | `backend_service/voice_clone.py`, `voice_speak.py`, `routes/voice_clone.py`, `NewVoiceForm.tsx` (a "from this designed clip" path), `VoicePicker.tsx` | pytest `tests/test_voice_clone.py` (design-then-clone record, consent basis "designed"); e2e `07-voice-clone` extension |
| W9 | Hygiene: SVG play and stop icons, remove the hardcoded "54", hide Japanese where never available, ffmpeg check before Generate, clone statement out of the form body, style placeholder and design hint copy | S | `VoiceModelsTab.tsx`, `SpeakScreen.tsx`, `voice_speak.py` (`list_voices` returns `needsFfmpeg`), `src/locales/*/studio.json` | vitest snapshot-free checks on the hint text; e2e `14-voice-speak` (ffmpeg note when the mock says missing) |
| W10 | Copy, i18n in all ten locales with `null` placeholders, `npm run i18n:validate`, `docs/usage/voice.md` and the two screenshots, FOLLOW_UPS rows (FU-146, FU-148 closed or updated) and the hub rebuild | S | `src/locales/*/studio.json`, `src/locales/*/common.json`, `docs/usage/voice.md`, `docs/screenshots/voice-*.png`, `FOLLOW_UPS.md` | `npm run i18n:validate`; `pytest tests/test_hub_tracker.py` |

Order: W1, W2, W4 first (small, and they answer complaints 1, 3 and 4 directly), then W3 and W5 together (one picker, one list), then W7, W9, W10. W6 and W8 are the large items and can follow in their own PRs; W6 needs the real-Mac memory run before it is on by default.

### 4.1 Must-keep checklist

Copied from `docs/plans/voice-rework/plan.md`; QA ticks each line against the new screens.

- Transcribe: model, language, translate, label speakers, speaker count, record, drag and drop, multi-file queue, cancel, remove, mic settings button, transcript editor, hand-offs to Chat, exports.
- Speak: 200k characters, blank-line paragraphs, `[pause]`, voice preview, speed, part-by-part generation with cancel, WAV, MP3, Save to Library, Send to Storyboard.
- Clone: standing statement, model select and download, voice list with consent line and delete, new-voice form (basis, speaker name, name, file or microphone sample, sample text, permission tick), AI label badge, agent-cloning switch.
- Voice Models: voice pack install, update, repair and cancel, speech-to-text models, voices and Mandarin add-on, all voice settings (auto-send, auto-read, dictation language, reading speed, call pause, interrupt, idle unload, unload now). Under this design the voices, the add-on and the chat settings move to the Voices tab; they must still exist and still write the same settings keys.
- Chat voice: mic button, read-aloud, voice call. These read `settings.ttsVoice` and `speed`, so they keep working.
- Library, MCP tools, CLI commands and the Storyboard voice lane keep their current fields.

Added for this round: `POST /api/voice/speak` request shape unchanged (W6 adds nothing to it); the golden MCP list (`tests/golden/mcp_tools_list.json`) unchanged unless `list_voices` gains the cost fields, in which case it is regenerated on purpose; the e2e `goTo(page, "Voice", ...)` labels unchanged.

## 5. Open questions for the owner

1. Save every Speak result to the Library automatically (cloned speech already is)? Recommendation: yes. The result row's Save button becomes a "Saved to Library" link, and a Delete on the row removes the item. It makes "Speak again" and the project hand-off work from one place and costs disk only (a 1 min clip is about 3 MB as WAV).
2. Hide the 0.6B CustomVoice model from the Speak picker and the Recommended card (FU-148)? Recommendation: yes. Keep it on Models for small-memory machines with the catalog's own note; the picker never offers a model that ignores the style box when a 1.7B is installable.
3. Allow natural voices in Chat read-aloud and calls (W6, large)? Recommendation: yes for natural and designed voices, after the real-Mac memory run (the 1.7B worker holds 3.5 GB while a chat model is also loaded). Cloned voices: read-aloud yes under the existing clone rules, calls no in the first pass.
4. Where do the chat behaviour settings live: a "Chat and calls" panel on the Voices tab (where the chat voice is chosen), or a Voice section on the Settings page? Recommendation: the Voices tab, with a "Voice settings" link from the Call info in Chat; one place to find both the voice and how it behaves.
5. Studio opens on Speak and remembers the last mode? Recommendation: yes. Transcribe users lose one click once; the Library's Open on a transcript still lands in Transcribe.
6. The Quick group shows a graded shortlist with an "All voices" expander, rather than all 54? Recommendation: yes, shortlist by the model author's grades (B- and better), so a first-time user sees six English voices that sound acceptable instead of twenty where most do not.

## 6. What was not verified

- No audio was made or listened to; all screens ran against the mocked voice API (the backend had no voice pack in the throwaway HOME), so quality claims about Kokoro versus Qwen3-TTS rest on the research notes and the owner's ear.
- The mocks list two Kokoro voices; the real picker density at 375 px with 20 American voices was not seen (listed as problem 19).
- Memory and speed numbers are quoted from the catalog's measured rows; nothing was re-measured here.
- The Windows and Linux lanes (kokoro-onnx, no Qwen3-TTS) were not looked at; the design assumes the picker shows Quick and Cloned only where the natural models are unsupported, as the current `modelsAvailable` flag already does.
- The light theme in the mock-ups is a proposal; the app has no light theme today.
