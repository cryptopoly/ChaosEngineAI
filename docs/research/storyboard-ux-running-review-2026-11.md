# Storyboard UX: running-app review (2026-11-25)

Scope: the new Storyboard flow, viewed in a running app (Vite dev server on port 5174, test backend on 8876, test
data folder). The main subject is the finished 10-shot lip-synced music video "Sunrise Road Trip" (`f19c0475882c5a18`),
plus one new throwaway project, "UI review", made from the Music video template. This was a read-only review: nothing
was generated, installed or deleted, and no setting was changed except the Terms of use, which the owner had already
accepted for the test backend. It is checked against `docs/research/storyboard-ux-audit-2026-11.md`. Screenshots are in
`E:\tmp\ui-review\`.

Not tested:
- Failure cards (no shot is failed).
- A running batch.
- The Upscale install card (the add-on is already installed).
- A light theme (the app is dark only).
- German at 900 px.

## Verdict

The new flow mostly works and mostly looks right. The stage rail, the "Does someone sing on screen?" question, the
lip-sync coverage table, the Sings / No character chips, the draft tags, "Choose for final", the thinner run bar,
"Muted under the song", the three-group readiness check, the upscale method choice and Board behind "More" all render
as intended. None of the CSS is broken or unstyled. At 700 px nothing overflows the page.

It is not ready to show yet. On the finished, exported film the rail's one main button is wrong. It reads
"Drafts: Making the shots is paused" with **Resume**, and the Final stage says "10 shots changed since they were made"
with **Make them again**. Both come from backend rules that misread this project, not from anything the user did. One
click would re-queue 13 old draft jobs; the other would remake 10 lip-synced finals that would come out "stale" again.
German and Japanese also mix in English: every backend sentence, plus a few new frontend strings that never reached
the locale files.

## Findings

Severity is one of: Blocks launch, Should fix, Polish.

| # | Screen | What I saw | Severity | Suggested fix | Files |
|---|--------|-----------|----------|---------------|-------|
| 1 | Rail, finished project (07) | Current stage "Drafts · Making the shots is paused." with primary **Resume**. Cause: two paused draft batches from 2026-11-24 morning (13 queued jobs), superseded by later finished batches. Storyboard shows no paused batch. | Blocks launch | Ignore a paused batch whose shots a later batch covers. Show paused batches on Storyboard with Resume / Discard. | `backend_service/storyboard/stages.py` (line 33, `live`), `BatchBar.tsx` |
| 2 | Rail Final, Export "Worth fixing" (08, 20) | "10 shots changed since they were made" and **Make them again**. Cause: the final lip-sync clips are 1248 x 704 and `_stale` asks for exactly 1280 x 720, so the video model's 32 px rounding counts as stale. Remaking gives 1248 x 704 again, so this loops. Overview and Storyboard both say "0 stale" at the same time. | Blocks launch | Accept model rounding for final halves (the same tolerance as drafts, or compare with the size the batch asked for). Word it as a size problem, not "changed". Use one stale count everywhere. | `backend_service/storyboard/quality.py` `_stale` |
| 3 | German and Japanese (40-51) | Rail sentences, backend readiness items, coverage state "Synced", the tempo note and the length rule stay English, sometimes mid-sentence. `board.song.lipSyncRange` and `board.song.coverage.state.*` exist only as `defaultValue`, in no locale file (not even `en`), yet `i18n:validate` reports 100 %. | Should fix | Run `i18n:extract`, then translate. Translate backend items by their `code`. Make the validator catch code-only keys. | `coverageLogic.ts`, `LipSyncCoverage.tsx`, `readiness.py`, `stages.py`, `scripts/i18n-*` |
| 4 | Rail Check (09, 25) | Check stays "To do" after the film is exported. On the empty project its tooltip says "1 point is worth a look", but the item is in **Stops the export**. | Should fix | Say "stops the export" when a stop exists. Mark Check done once an export exists and nothing new stops it. | `stages.py` |
| 5 | Shot cards (10) | "No character" shows on shots 1-6 although no character is "In every shot". The prompts of those shots describe the singer in words. The tooltip ("No character is in this shot…") is wrong for them. | Should fix | Show the chip only when a lock exists and the shot left it out, as the audit planned. Otherwise show nothing. | `ShotCard.tsx` |
| 6 | Shot cards (10) | Every card offers **Choose for final**, even when its active take is already a final take. | Should fix | When the active take is final, show "Final" (and its size) instead of the button. | `ShotCard.tsx` |
| 7 | Plan, Song (13, 14) | On a finished, cut project, "Yes" is still selected and **Cut the song into shots** is the primary button, inviting a re-cut. There are three buttons for the same idea: Cut the song into shots, Keep my shots and give each its vocal, Give each shot its vocal. | Should fix | Once cut, show "Cut: 10 shots, 10 sing" with a secondary "Cut again". Fold "Give each shot its vocal" into the coverage table. | `song/SongPanel.tsx`, `song/SongVocals.tsx` |
| 8 | Plan, Song (13) | The Yes hint says "shots of 3 to 5 s". The next sentence says 2 s or more can lip-sync, and the table shows 2.3 to 2.7 s shots. The length rule is repeated three times (Song, Lip-sync section, shot editor). | Should fix | Make the hint "2 to 5 s". Say the rule once. | `song/SongPanel.tsx`, `coverageLogic.ts` |
| 9 | Plan, Storyboard, editor (14, 15, 17) | Shot 1's length is 2.7 s in the coverage table, 3.24 s in the Plan list and 3.38 s on its card. The Plan list shows 3.431 and the editor 4.453 (three decimals). | Should fix | Label which length is shown (song slot or clip). Round to 0.1 s. | `LipSyncCoverage.tsx`, `PlanTab.tsx`, `ShotEditorDrawer.tsx` |
| 10 | Rail action "Add the song" (05, 06) | It opens Plan at the top (chat model, brief). The Song section is below the fold. | Should fix | Scroll to and focus the Song section. | `guide/useGuide.ts` |
| 11 | Rail, every tab (17) | The rail is not sticky. It scrolls away on long tabs, and the rail and its button take about 90 px at the top of every tab. | Polish | Make it sticky in a compact form, or collapse it to a single row after the first stage. | `guide/guide.css` |
| 12 | Mixer (19) | Lip-sync clips in the inspector have no "carries the vocal, muted in the film" line (audit change 7). The preview label "preview 640 × 360" is small grey text, not "not the film". | Polish | Add both lines. | `mixer/ClipInspector.tsx`, `mixer/MixerTab.tsx` |
| 13 | Mixer at 700 px (30) | "-60 dB" labels overlap each other and the waveforms. The lane header wraps onto four lines. The Voice, Effects and Music "+" buttons become large squares. | Polish | Hide dB labels on narrow clips. Keep "+" a fixed size. | `mixer/Timeline.tsx`, `styles.css` |
| 14 | Export (21) | "Estimate: unknown until the first run" after six exports. | Polish | Use the measured time of the last export. | `mixer/ExportPanel.tsx` |
| 15 | Export upscale (26) | "Sharper (AI, needs the Upscale add-on)" is shown while the add-on is already installed. | Polish | Drop "needs…" when it is installed. | `mixer/upscaleMethod.ts` |

**Other polish (16 to 22):**

- **Upscale tab (24):** "Fast, stronger clean-up" runs at 1 to 3 frames a second. Rename it "Stronger clean-up (slow)". The page and card titles repeat.
- **Music video template (05):** it still uses the tea-shop brief placeholder and caption sample.
- **More menu (22):** the popover opens under Mixer and Export, not under "More".
- **Rail at 700 px (29):** chips wrap to three lines. In German, "Lippensynchronisation" is very wide and "Einstellungen" means both "shots" and Settings.
- **Japanese (46 to 51):** letter-spacing on small labels splits katakana apart. Drop it under `:lang(ja)`, `:lang(zh)` and `:lang(ko)` in `styles.css`.
- **Cast caption fonts (16):** the list includes symbol fonts (Wingdings, Marlett).
- **Template pills:** they lack `aria-pressed`.

## Matched the audit, and did not

**Matched:** everything listed in the verdict's first paragraph. "Style as words" also shows when the still model
cannot take pictures, and the run bar keeps one sentence with the rest under "More ways to make".

**Did not match:**

- Wrong rail state on a finished project (findings 1, 2 and 4).
- "No character" with no lock (finding 5).
- Draft tags read only "Draft"; the size is in the tooltip.
- No "carries the vocal" line in the Mixer (finding 12).
- "3 to 5 s" is still in the cut hint (finding 8).
- Three vocal buttons, not one step (finding 7).
- Strings missing from the locale files (finding 3).
- Failure cards are untested.

## Five most important fixes, in order

1. **Paused batches:** stop old paused batches from driving the rail and its Resume button (`stages.py`).
2. **False "stale":** fix the stale rule for model-rounded final sizes, and use one stale count across rail, Overview, Storyboard and Export (`quality.py`).
3. **Missing translations:** extract the missing keys, translate backend sentences by code, and make the i18n gate catch keys that exist only in code.
4. **Plan song section:** once the song is cut, show what was done instead of a primary re-cut button, fix "3 to 5 s", and say the length rule once.
5. **Chips and the Check stage:** show "No character" and "Choose for final" only when they are true, and make the Check stage match the readiness list.

## Screenshots (`E:\tmp\ui-review\`)

The ones that matter most are in bold.

- 01: Terms gate, before it was accepted.
- 02: project list.
- 03: new project dialog with Music video.
- 04: new project, Song current.
- 05, 06: "Add the song" lands at the top of Plan.
- **07: finished film, "Drafts paused", Resume.**
- **08: Final, "10 shots changed", Make them again.**
- 09: Check stays To do.
- 10: shot cards.
- 11: run bar.
- 12: Plan top.
- **13: sing question with "3 to 5 s".**
- 14: coverage table.
- 15: three-decimal lengths.
- 16: Cast and "In every shot".
- 17: shot editor.
- 18, 19: Mixer lane and inspector.
- **20: readiness check.**
- 21: export panel.
- 22: More popover.
- 23: Board.
- 24: Upscale tab.
- 25: empty project, stop.
- 26: upscale method.
- 27, 28: 900 px.
- 29: 700 px rail wrap.
- **30: 700 px Mixer, dB labels overlap.**
- 31: 700 px Plan.
- 40 to 45: German.
- **45: mixed English.**
- 46 to 51: Japanese.
- **51: mixed English and katakana spacing.**
