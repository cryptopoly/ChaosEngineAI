# Storyboard UX audit and guided-flow plan (2026-11-24)

Scope: the Projects entry (project list, then one project with seven tabs: Overview, Plan, Cast, Storyboard, Board,
Mixer, Export) and the backend it drives. Audit from code only: the backend on port 8876 was mid-render, and opening
the app fires requests at it on load (estimate, plan status), so the UI was not run. Nothing here was checked in a
running app. No code was changed.

## 1. Verdict

The parts are good (estimates, undo everywhere, takes never overwritten, a preflight list), but the project is laid out as seven equal tabs and the user has to know the order. The guide on Overview
knows only the plain film path (shots, clips, sound, preview, export): it has no idea a song is attached, never
mentions the cut plan, lip sync, drafts versus final, or the cast, and for a music video it says "Add voice or
music (optional)". The most important facts about a music video (which shots lip-sync and why not, that a take is a
half-size draft, that clip sound is muted under the song, that no character is attached to a shot) are either
invisible, shown once and lost on a tab switch, or shown only on the Export tab. The fix is not more controls: it is
one stage rail that owns the order, and four facts made visible where the work happens.

## 2. The flow today, and where the user gets lost

1. **New project.** Templates are Blank, 30 s ad, Vertical social, Short story, Scene / b-roll, Sample film
   (`projectTemplates.ts`). There is no music video start, so the user picks Blank. *Lost: nothing says a song goes
   in later, or where.*
2. **Overview.** "Write your shots" with "Plan from an idea" as the primary button, a checklist (Name and format,
   Cast, Shots, A clip for every shot, Sound, Preview, Export) and Project settings. *Lost: no row for song, cut
   plan, lip sync or quality (`overviewLogic.ts` `nextStep`, `checklist`).*
3. **Plan tab.** Chat model bar, Brief, then the Song section, then Shots, with a direction chat on the side. The
   Song section holds two different cut buttons: "Make a cut plan" and, further down under "Lip-sync the song",
   "Cut for singing (3 to 5 s shots)", then an optional stem field, then "Give each shot its vocal". *Lost: the
   lip-sync path is a numbered sentence in grey text; the user who already pressed "Make a cut plan" has to cut
   again, and the brief's "Propose shots" competes with both.*
4. **Lip-sync result.** After "Give each shot its vocal" a line says how many shots got a vocal and lists skipped
   shots. It lives in component state (`SongVocals.tsx` `result`), so it is gone after a tab switch or reload. The
   shot card shows nothing about a vocal (no file in the card reads `vocalClip`). The hint still says shots last
   "3 to 5 s" and the shot editor says "3 to 8 s", while the backend now accepts 2 s and up (`song_vocals.py`
   `MIN_SHOT_SECONDS`). *Lost: friction (b).*
5. **Cast tab.** Characters, places and styles are cards; a shot uses one only when the user opens each shot's
   editor drawer and ticks it. The planner sees the cast but its proposals carry no cast (`PlanTab.use` sends title,
   prompts and seconds only), and a cut plan keeps cast only from the shot it replaces. *Lost: friction (d).*
6. **Storyboard tab.** Shot cards and a sticky run bar with up to five actions: Make all unlocked, Make N missing,
   a "Drafts with clips" tick, Make drafts, Final pass on kept shots. A draft take has a small grey "Draft" badge on
   the card; the size is only in a tooltip. Two different "Keep" buttons exist: "Keep as is" (dismisses a stale
   badge) and "Keep for final". *Lost: friction (a); the clip player and the rough cut show no quality or size.*
7. **Mixer.** Lanes: Picture, Shot sound, Voice, Effects, Music. Under a song each new shot's own sound starts at
   -60 dB (`SONG_SHOT_NATIVE_SOUND_DB`), shown only as a number on the lane. The rendered preview is 360 px on the
   short side with the size in small text. *Lost: friction (c), and a second "is this the final?" moment.*
8. **Export.** "Before you export" lists format, clips, stale, drafts, loudness, captions, AI tag, preview. Every
   line is advisory and the list knows nothing about lip sync, the song's length against the film, failed shots or
   cast. The drafts warning appears here for the first time as a sentence.
9. **Failures.** A failed shot shows "The shot did not render. Try again or edit the prompts." with the raw
   exception under Technical details (`batch.py` stores `str(exc)`; `ShotCard.tsx`). Out of memory, a missing
   model, a missing lip-sync model and a gated download all look the same and all suggest editing the prompt.
   *Lost: friction (f).*

## 3. Ranked changes (12)

Translation: "keys" means new English strings in `src/locales/en/studio.json` with `null` placeholders in the nine
other locales, then `npm run i18n:validate`; a follow-up fills them.

| # | Change | Size | Risk | Keys |
|---|--------|------|------|------|
| 1 | Stage rail and song-aware guide | L | Medium | Yes (about 40) |
| 2 | Quality made visible (Draft / Final) | M | Low | Yes (about 12) |
| 3 | Lip-sync coverage that persists | M | Medium | Yes (about 15) |
| 4 | Character lock | M | Medium | Yes (about 10) |
| 5 | Plain-language failures with one action | M | Low | Yes (about 14) |
| 6 | Readiness check before export | M | Low | Yes (about 12) |
| 7 | Clip sound under a song shown and switchable | S | Low | Yes (about 5) |
| 8 | One "Sing on screen?" question replaces two cut buttons | S | Low | Yes (about 6) |
| 9 | Music video start in the new-project dialog | S | Low | Yes (about 4) |
| 10 | One-go "Make the film" run with a review stop | L | High | Yes (about 12) |
| 11 | Rename the two Keeps; thin the run bar | S | Low | Yes (about 4) |
| 12 | Tab tidy: Board behind "More", names that do not collide | S | Low | Yes (about 2) |

1. **Stage rail.** Problem: seven equal tabs, a guide blind to songs. Proposed: a rail above the tabs with the
   stages Song, Look, Shots, Drafts, Final, Lip sync, Sound, Check, Export; each shows done, current, needs
   attention or skipped, with one sentence and one primary button for the current stage. Stages that do not apply
   (no song: Song and Lip sync) are hidden. The tabs stay as the places the rail opens. Files: `overviewLogic.ts`
   (stage model replaces `nextStep` and `checklist`), `OverviewTab.tsx`, new `guide/StageRail.tsx` and
   `guide/stageLogic.ts`, `ProjectShell.tsx`, `ProjectShellHeader.tsx`.
2. **Quality.** Problem: drafts are half size (`batch.py` `draft_size`: 1280 x 720 gives 640 x 384) and look like
   the film. Proposed: a labelled corner tag on every draft thumbnail, clip player, take strip item and rough cut
   ("Draft 640 x 384"); a line in the run bar "Drafts are half size to choose shots fast. The film will be 1280 x
   720."; the Mixer preview titled "Preview, 640 x 360, not the film". Files: `ShotCard.tsx`, `TakesStrip.tsx`,
   `RunBar.tsx`, `preview/RoughCutPlayer.tsx`, `preview/RoughCutStage.tsx`, `mixer/MixerTab.tsx`, `styles.css`.
3. **Lip-sync coverage.** Problem: result shown once; skip reasons are backend English sentences. Proposed: the
   backend returns a reason code per shot and a read-only coverage call; a table (shot, song time, state: will
   sing, sung, no singing here, too short, too long, locked, other speed) with a fix button per row (lengthen,
   unlock); a "Sings" chip on the shot card; the stale hints corrected to the real range. Files:
   `backend_service/storyboard/song_vocals.py`, `routes/storyboard_long.py`, `mcp_bridge/storyboard_tools.py`,
   `src/types/storyboard.ts`, `src/api/storyboard.ts`, `song/SongVocals.tsx`, new `song/LipSyncCoverage.tsx`,
   `ShotCard.tsx`, `ShotVocalFields.tsx`.
4. **Character lock.** Problem: cast is opt-in per shot through a drawer. Proposed: "In every shot" switch on a
   cast card (adds the reference to all current and future shots, one op, one Undo); plan proposals and cut plans
   carry cast; the shot card shows "No character" as a quiet chip when a lock exists but the shot opted out; when
   the still model cannot take pictures, say the look goes in as words and may drift. Files: `CastTab.tsx`,
   `ReferenceCard.tsx`, `PlanTab.tsx`, `PlanProposal.tsx`, `backend_service/storyboard/ref_ops.py`, `ops.py`,
   `plan_ops.py`, `planner.py`, `cutplan.py`, `models.py`.
5. **Failures.** Problem: raw errors, wrong advice. Proposed: the batch records an error code beside the text
   (out of memory, model not installed, lip-sync model missing, video tools missing, gated or offline download,
   disk full, unknown); the card says what happened and offers the one action that helps ("Make it at draft size",
   "Open Video Models", "Install video tools", "Retry"). Raw text stays under Technical details. Files:
   `backend_service/storyboard/batch.py`, new `backend_service/storyboard/failure.py`, `models.py`, new
   `src/features/storyboard/failureText.ts`, `ShotCard.tsx`, `BatchBar.tsx`, `OverviewTab.tsx` (activity lines).
6. **Readiness check.** Problem: preflight is advisory and incomplete. Proposed: three groups (Stops the export,
   Worth fixing, For your information) adding: failed shots, drafts, lip-sync coverage ("9 of 12 sung shots have a
   synced clip"), film length against song length, clip sound audible over the song, shots without the locked
   character. With warnings the button reads "Export anyway". Files: new `exportReadiness.ts` (moves `preflight`
   out of `overviewLogic.ts`), `ExportTab.tsx`, `mixer/ExportPanel.tsx`.
7. **Clip sound.** Proposed: the Shot sound lane header says "Muted under the song" with a switch "Let clip sound
   through"; lip-sync shots are labelled "carries the vocal, muted in the film". Files: `mixer/MixerTab.tsx`,
   `mixer/Timeline.tsx`, `mixer/ClipInspector.tsx`, `ShotEditorDrawer.tsx`.
8. **One question.** Proposed: after the song is analysed ask "Does someone sing on screen?" Yes cuts 3 to 5 s
   shots and assigns vocals in one step; No makes the ordinary cut plan. The stem field sits under Yes. Files:
   `song/SongPanel.tsx`, `song/SongVocals.tsx`, `song/CutPlanReview.tsx`.
9. **Music video start.** A "Music video" template that opens the project on the Song stage. Files:
   `projectTemplates.ts`, `ProjectDialog.tsx`, `ProjectListTab.tsx`.
10. **One-go run.** One button queues drafts, stops for review, then finals, clips and lip sync, with one estimate
    and one progress list by stage. Files: new `guide/MakeFilmPanel.tsx`, `useStoryboard.ts`,
    `backend_service/storyboard/batch.py`, `batch_queue.py`, `routes/storyboard.py`. High risk: batch state,
    pause and resume, GPU swaps. Build last, after 1 to 6 are proven.
11. **Keeps and run bar.** "Keep for final" becomes "Choose for final"; the run bar shows one primary action and
    puts the rest under "More ways to make". Files: `ShotCard.tsx`, `RunBar.tsx`.
12. **Tabs.** Board and Storyboard read as the same thing. Board moves behind "More" until it has cards. Files:
    `ProjectShellHeader.tsx`, `projectRoute.ts` (remembered routes must still resolve).

## 4. Guided flow

The rail is always visible. One stage is current; its panel sits under the rail.

```
Projects > Night Drive          16:9 · 1280 x 720 · 24 fps        Undo  Redo
[1 Song ok] [2 Look ok] [3 Shots ok] [4 Drafts >] [5 Final] [6 Lip sync] [7 Sound] [8 Check] [9 Export]
------------------------------------------------------------------------------
```

**1 Song.** "Add the song this film is cut to." Library picker or file. After analysis:

```
Night Drive.flac   0:30 · 96 BPM · 4 sections      [Change] [Remove]
|intro|--verse-----|--chorus------|outro|
Does someone sing on screen?
( ) Yes, lip-sync the singing     vocal stem (optional, follows best) [Browse]
( ) No, pictures only
[Cut the song into shots]         12 shots of 2 to 5 s · nothing is made yet
```

**2 Look.** "Who and what is in it." Character, place and style cards.

```
[picture] Mara  "short red hair, denim jacket"     [x] In every shot
[picture] Rooftop at night                          [ ] In every shot
Still model takes 2 reference pictures: Mara goes in as a picture.
[Next: review the shots]
```

**3 Shots.** The cut plan grouped by section, prompts editable, "Rework with the model" per shot. Each row shows
the cast chips and, for a music video, "sings" or "no singing here".

**4 Drafts.** One choice, stated in sizes and time:

```
How do you want to start?
(o) Drafts first   640 x 384 · about 4 min · choose the shots you like, then make them full size
( ) Straight to final   1280 x 720 · about 38 min · peak 21 GB of 24 GB
[Make 12 drafts]
```

Every draft picture carries the tag "Draft 640 x 384". The panel then says "Choose the shots to make full size"
with "Choose all".

**5 Final.** "Make 9 chosen shots at 1280 x 720, then their clips · about 31 min." Failures show in place:

```
Shot 7 did not fit in GPU memory (needed about 26 GB, 24 GB here).
[Make it at 960 x 544]  [Retry]  [Technical details]
```

**6 Lip sync.** The coverage report, kept with the project:

```
Lip sync: 9 of 12 shots will sing                       vocal: stem
 #  Song time    Length  State
 1  0:00-0:03    3.0 s   Synced (take 2 of 3 kept, score 0.52)
 4  0:09-0:11    1.6 s   Too short (needs 2 s)        [Lengthen to 2 s]
 6  0:14-0:18    4.0 s   No singing here              plays as an ordinary clip
 9  0:22-0:26    4.0 s   Waiting                      3 takes · about 4 min
[Make 6 lip-sync clips · about 22 min]
```

**7 Sound.** The Mixer with the song lane and the line "Clip sound is muted under the song. [Let clip sound
through]". Captions from the lyrics are offered here.

**8 Check.**

```
Ready to export?
Stops the export     none
Worth fixing         2 shots are still drafts (640 x 384)            [Choose for final]
                     Shot 4 is not lip-synced: too short             [Open shot 4]
                     The film is 0:28, the song 0:30                 [Open Mixer]
For your information 1280 x 720 · 24 fps · -14 LUFS · AI tag written · no captions
[Watch the rough cut]   [Export anyway]
```

**9 Export.** The existing panel and result.

## 5. Build order for a team of agents

One worktree per stream; file sets do not overlap. Every stream adds English strings through `defaultValue` only and
does not edit locale files.

1. **Stream A, contracts (first, alone).** `backend_service/storyboard/song_vocals.py`, `batch.py`, new
   `failure.py`, `models.py`, `ref_ops.py`, `ops.py`, `plan_ops.py`, `planner.py`, `cutplan.py`,
   `routes/storyboard_long.py`, `mcp_bridge/storyboard_tools.py`, `src/types/storyboard.ts`,
   `src/api/storyboard.ts`, the pytest files, and the new E2E checks in `scripts/e2e_test_suite.py` (the feature
   gate). Delivers reason codes, coverage, error codes, the "in every shot" op and cast on proposals.
2. **Then four streams in parallel.**
   - **B, guide:** `overviewLogic.ts`, `OverviewTab.tsx`, new `guide/`, `ProjectShell.tsx`,
     `ProjectShellHeader.tsx`, `projectRoute.ts`, `projectTemplates.ts`, `ProjectDialog.tsx`,
     `ProjectListTab.tsx` (changes 1, 9, 12).
   - **C, shots and quality:** `ShotCard.tsx`, `TakesStrip.tsx`, `RunBar.tsx`, `BatchBar.tsx`, `preview/`, new
     `failureText.ts` (changes 2, 5, 11, and the card chips of 3 and 4).
   - **D, song and cast:** `song/`, `CastTab.tsx`, `ReferenceCard.tsx`, `PlanTab.tsx`, `PlanProposal.tsx`,
     `ShotVocalFields.tsx` (changes 3, 4, 8).
   - **E, mixer and export:** `mixer/`, `ExportTab.tsx`, `ShotEditorDrawer.tsx`, new `exportReadiness.ts`
     (changes 6, 7). B deletes `preflight` from `overviewLogic.ts` only after E's file is merged.
3. **Stream F, last, alone:** `npm run i18n:extract`, the nine locale files, `src/styles.css` conflicts,
   `npm run i18n:validate`, full vitest, tsc, pytest, and screenshots of each stage in a running app.
4. **Stream G, after the owner has used 1 to 9:** the one-go run (change 10).
