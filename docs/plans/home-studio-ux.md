# Home studio UX: Music, Storyboard and a calmer Studio

Status: draft 1 (2026-11-06), companion to [home-studio.md](home-studio.md). Scope: the screens of P2 (Music frontend), P4 (Storyboard UI) and P5 (licence screen), plus a ranked list of de-cluttering changes for the existing Image and Video Studios that are **not** to be built now (section H).

The brief from the owner: "The GUI is already crowded enough, so we need to make sure we give the user the best representation of options and data." Everything below answers that sentence.

Rules that apply to every screen (from `CLAUDE.md`): monochrome SVG icons, no emoji, colours and spacing from the `:root` tokens in `src/styles.css`, no inline styles, every string through i18n (`en` text, `null` elsewhere), no third-party desktop app named anywhere.

---

## A. Audit of the existing Studios

Method: counted every interactive control a user can click, type in or toggle at first load, on this Mac (Apple Silicon, backend online, one installed model, no danger-level warning). Counted from the component source, not a screenshot, so nothing hidden by a conditional is miscounted. Files: `src/features/images/ImageStudioTab.tsx` (1,380 lines) and `src/features/video/VideoStudioTab.tsx` (1,840 lines).

### A.1 Image Studio (FLUX.1-dev on diffusers, reference images supported, recommendation present)

| Group | Controls | Count | First-time user needs | Expert |
|---|---|---|---|---|
| Header actions | Find models, Installed, Gallery | 3 | Gallery | Find models, Installed (both are on the Models page already) |
| Hero + runtime banner | 5 status badges; Load into memory / Unload / Restart (banner) | 2 to 3 buttons, 5 badges | one status line | the rest |
| Model | select | 1 | yes | |
| Prompt | prompt, negative prompt | 2 | prompt | negative prompt |
| Reference images | Choose file, From gallery, (Strength slider when one is set) | 2 (+1) | yes when the model takes one | strength |
| Recommended for this machine | Apply recommended + one Use button per row | 1 + about 5 | the one Apply button | per-row Use |
| Aspect pills | 1:1, 3:2, 2:3, 4:3, 3:4, 16:9, 9:16, 21:9 | 8 | 3 or 4 of them | |
| Size pills | 0.5, 1, 2, 4, 8 MP | 5 | one choice (S / M / L) | |
| Quality pills | Fast, Balanced, High Quality, Preview | 4 | yes | |
| Diffusion cache | select + threshold number when on | 1 (+1) | no | yes |
| CFG decay, Preview VAE | two checkbox rows with a full sentence each | 2 | no | yes |
| Numbers | Width, Height, Steps, Guidance, Images, Random seed (+ Seed) | 6 (+1) | Images | Width, Height, Steps, Guidance, Seed |
| Generate | primary button | 1 | yes | |
| **Total** | | **about 44 controls, 13 of them for the size alone** | **about 9** | **about 35** |

On a PC with an NVIDIA GPU add FP8 layerwise (+1); on an SDXL model add the Sampler select (+1).

### A.2 Video Studio (Wan 2.2 on diffusers, Apple Silicon, recommendation present)

| Group | Controls | Count | First-time user needs | Expert |
|---|---|---|---|---|
| Header actions | Find models, Installed | 2 | | both |
| Engine banner (LTX-2 chip, mlx-video chip, install buttons, restart) | 1 to 3 buttons, 2 to 4 chips | 2 | one status line | the rest |
| Model | select, 6-badge stats row | 1 (+6 badges) | yes | |
| Prompt | prompt, negative prompt | 2 | prompt | negative |
| Start / end image | 2 pickers, 2 buttons each | 4 | start image | end image |
| Recommended | Apply + about 5 Use | 6 | Apply | per row |
| Quality pills | Draft, Standard, High, Max | 4 | yes | |
| Aspect pills | 16:9, 9:16, 1:1, 4:3, 21:9 | 5 | 2 or 3 | |
| Resolution pills | SD, HD, Full HD, 2K, 4K, 8K | 6 | 2 (most are greyed by the model limit anyway) | |
| Numbers | Width, Height, Frames, FPS | 4 | length in seconds | all four |
| Sliders | Steps (slider + number), Guidance (slider + number) | 4 | no | yes |
| Toggles | CFG decay, Preview VAE | 2 | no | yes |
| Diffusion cache | select (+ threshold) | 1 (+1) | no | yes |
| Seed | Random seed (+ Seed) | 1 (+1) | no | yes |
| Actions | Load into memory, Generate, Model Card | 3 | Generate | the other two |
| **Total** | | **about 47 controls** (+2 NF4 / FP8 on CUDA, +3 STG and Fast preview on LTX-2) | **about 8** | **about 39** |

Both Studios also show six to eight lines of informational text at first load (capacity line, clip-length line, will-load hint, disabled-reason hint, model notes, max-pixels note, engine notes).

### A.3 The three biggest sources of clutter

1. **One decision, four controls.** Size is chosen with aspect pills, size or resolution pills, a Width box and a Height box (13 to 15 controls for one choice); video adds Frames and FPS for a second choice ("how long") the user thinks of in seconds. Each control re-states the others.
2. **Expert engine knobs sit at the top level with paragraph-long labels.** Diffusion cache, threshold, CFG decay, Preview VAE, FP8, NF4, STG, Sampler: eight knobs that a first-time user cannot evaluate, each with a checkbox row that reads like documentation plus a tooltip that reads like a changelog. They take about 40 % of the form's height and most of its words.
3. **Status is told in seven places.** Hero badges, the runtime banner, the model stats row, the capacity line, the safety callout, the "Generate disabled" hint and the "will load on next generate" hint all describe the same three facts: which model is selected, whether it is loaded, and whether it fits. A user has to read all seven to know if clicking Generate is safe.

### A.4 Data the user needs that is hidden or buried

| Needed at the moment of decision | Where it is today |
|---|---|
| Will this model fit on my machine (fit verdict and reason) | Models page only (`FitBadge`); not in either Studio |
| How long will this take | Nowhere before the click; after it, "{value}s" on the output card |
| Peak memory for this run | Video: the capacity line, always; Image: only inside a warning callout |
| Licence, commercial use, gated | Models page, inside the expandable row detail (`ModelDetailList`) |
| What a setting costs (steps, size, frames) | Tooltip prose ("linearly slower"); no number |
| Disk used by outputs | Nowhere (Gallery counts items, not bytes) |
| Queue position and what is running ahead | Only inside `LiveProgress` once the job has started |
| Why Generate is disabled | Video: an inline hint; Image: the button tooltip only |

---

## B. Design principles for the whole product

1. **Progressive disclosure, three levels.** The default view of any Studio has at most **six controls and one primary action**. Everything else is behind `Options` (per-group disclosures, remembered per Studio) or `Expert` (one disclosure, collapsed by default, with a one-line summary of what is set to a non-default value). The levels are fixed across Image, Video, Voice, Music and the Storyboard so a user who learns one learns all.
2. **Presets are the main way to choose.** A preset is a named bundle of expert values (quality, size, length). Changing one expert value turns the preset pill to "Custom"; the pill row stays, so the way back is one click. Presets come from the catalog's `recommendedSettings` first, then from the static table.
3. **Smart defaults from the fit rating.** On first selection of a model the Studio applies the machine's recommendation (steps, size, cache) and says so in one line with an Undo. The fit verdict decides the default preset: `fits` gets the balanced preset, `tight` gets the fast one, `wont-fit` blocks the run with the reason and the nearest model that fits.
4. **Show the cost before the click, the result after.** Every primary action has a **run estimate strip** beside it: estimated time, peak memory of total, disk written, with a Measured / Estimated basis. After the run, the output card shows the measured time and size next to the estimate so the estimate gets better (the backend records it per model and preset).
5. **Plain-language labels, technical name in the tooltip.** The label says what the user gets ("Sharper, slower"); the tooltip says the knob ("Steps: 36"). Numbers are shown next to the plain label so experts are not slowed down.
6. **One screen anatomy.** Header (what and status) → inputs → preset row → estimate and primary action → Options → Expert → results. Same order, same class names, on every Studio (section C.5).
7. **Status in one place.** One status chip row in the header (model · engine · Installed / Loaded / Downloading / Needs licence · fit badge). Banners appear only when something needs an action (install, restart, licence) and disappear when done.
8. **Never silently ignore** (plan principle 3): a control the selected model cannot honour is not shown; if the user's saved value would be ignored, the estimate strip says so in one line.
9. **Every state has a sentence.** Empty, loading, queued, running, done, error, offline, missing media: each has a plain line and, where there is one, the one action that fixes it.

---

## C. Data-presentation spec

### C.1 The model card (every Models page: Chat, Image, Video, Voice, Music)

Implemented on the existing `ModelsPage` + `ModelCard` shell (`src/components/models/`). Fields, in this order, left to right in the row, then the expandable detail. Fields marked (new) need a backend field (the plan's `licence`, `gated`, `commercial`, `platforms`, `makes`).

| # | Field | Component | Rule |
|---|---|---|---|
| 1 | Name, with `Best fit` ribbon | `models-card__title`, `BestFitRibbon` | Name only; provider · family in the `small` subtitle |
| 2 | What it makes (new) | `MakesChips` (`.makes-chip`) | 1 to 3 chips: "Songs with vocals", "Instrumental", "Sound effects", "Loops"; for Image / Video reuse the tasks column |
| 3 | Size on disk | existing `sizeLabel` cell | "3.6 GB"; title has weights / full repo split |
| 4 | Fit badge with reason | `FitBadge` inside `ModelMemoryCell` | Fits / Tight / Won't fit / Not for this machine; tooltip = reason + numbers + basis |
| 5 | Licence chip (new) | `LicenceChip` (`.licence-chip`) | Text: licence short name; a second word states commercial use: "MIT · commercial", "Community · under US$1M", "CC-BY-NC · no commercial", "Gated" prefix when a Hub accept is needed. Tooltip: the one-sentence licence summary from the catalog. Click opens the licence screen (C.6) |
| 6 | Platform (new) | `AcceleratorPills` style chips | "Apple Silicon", "CUDA", "CPU"; a row that cannot run here already gets the incompatible chip from `ModelCard` |
| 7 | Status | `ModelStatusChip` | Installed / Not installed / Downloading 42 % / Paused / Failed / **Needs licence** (new state when gated and not accepted) / Loaded |
| 8 | Actions | `ModelDownloadControls` | Download / Pause / Resume / Delete / **Open in Studio** / **Accept licence** (gated, not accepted) |

Expandable detail (`ModelDetailList`): note, measured speed ("about 6 s of audio per second on this machine", or "not measured yet"), licence full line with "Read the licence" link, model card link, local path, shared-download siblings.

Sort options: Best fit first (default), Size, Released, Name. Filters: Makes (chips), Licence (Commercial ok / Any), Platform.

### C.2 The run estimate strip

One component, `RunEstimateStrip`, rendered directly above the primary action on every Studio and on the Mixer export panel.

```
About 40 s  ·  Peak 9 GB of 64 GB  ·  Writes 12 MB      [Estimated]      [ Generate ]
```

Rules:
- Three items, in this order: time, memory, disk. An item whose value is unknown reads "Time: unknown until the first run" (never a blank).
- Basis chip: `Measured` (timed on this machine for this model and preset; tooltip names the date) or `Estimated` (from the model card or a similar machine). Same chips as `RecommendedSettingsPanel`.
- Memory colour follows the fit verdict tokens: `--success` under 70 % of total, `--warning` 70 to 92 %, `--danger` above (the existing safety heuristic's bands). Danger keeps the existing acknowledgement checkbox, rendered inside the strip, not in a separate callout.
- Changing any control updates the strip live (`aria-live="polite"`).
- Source of the numbers: `assessImageGenerationSafety` / `assessVideoGenerationSafety` for memory; a new `estimateRun(kind, modelId, settings)` helper that reads `recommendedSettings.measuredOn` timings and a per-machine table the backend keeps (`~/.chaosengine/perf/{kind}.json`, written after every run).

### C.3 The queue and waiting note

Shown inside `LiveProgress` (existing queue line) and, before the click, as a second line of the estimate strip when the shared queue is busy:

```
Another job is running (Video Studio clip, about 3 min left). Yours starts after it.
```

Strings: "Waiting for another job to finish (position {n})" (exists), "Waiting for an agent job to finish" (exists), new: "{what} is running, about {time} left" and "Starts after the current job". A storyboard batch shows "Shot 3 of 8 · stills, then clips, then audio".

### C.4 Error and empty states, in plain language

| State | Pattern | Example |
|---|---|---|
| Empty (nothing made yet) | `.empty-state` with one sentence and one primary button | "No music yet. Describe a track and press Make." |
| Empty (no model) | same, button goes to the Models page | "No music model installed. Open Music Models to download one (about 3.6 GB)." |
| Loading | `.empty-state` muted line | "Loading your library…" |
| Queued | estimate strip second line | see C.3 |
| Running | `LiveProgress` with phases and queue line; Cancel | "Making your track · 00:42 · step 12 / 27" |
| Done | result card with measured time and size; actions | "Done in 38 s · 11 MB" |
| Error | `.callout.error`, headline in plain words, the fix as a button, technical text under "Technical details" (`.debug-details`) | "The model ran out of memory. Try a shorter track or the Fast preset." [Use Fast preset] |
| Offline | one line, no form | "Waiting for the backend…" (exists) |
| Missing media (Storyboard) | shot card state + button | "Take file missing. Relink or make a new take." |
| Needs licence | `.callout.warning` with one button | "Stable Audio Open needs a licence accept on Hugging Face before it can be downloaded." [Accept licence] |

Headline rule: a user-facing error never starts with an exception class or a path. The raw text stays under Technical details.

### C.5 Screen anatomy (every Studio)

```
┌ Panel header ───────────────────────────────────────────────────────────┐
│ Music Studio    ACE-Step 1.5 turbo · Installed · Fits (est.)  [Models] [Library] │
├─────────────────────────────────────────────────────────────────────────┤
│ Inputs           (prompt / lyrics / pickers)                             │
│ Preset row       (3 to 5 pills + Custom)                                 │
│ Estimate strip   About 40 s · Peak 9 GB of 64 GB · 12 MB   [Make track]  │
│ > Options        (collapsed groups with a value summary each)            │
│ > Expert         (collapsed; summary of non-default values)              │
├─────────────────────────────────────────────────────────────────────────┤
│ Results          (latest first; Open library)                            │
└─────────────────────────────────────────────────────────────────────────┘
```

Classes: `.studio-shell`, `.studio-shell__header`, `.studio-shell__inputs`, `.studio-shell__presets`, `.studio-shell__estimate`, `.studio-shell__options`, `.studio-shell__results`. A disclosure is a native `<details>` styled as `.disclosure` (keyboard and screen reader for free), with `.disclosure__summary` showing the title and a muted value summary ("BPM 120 · Key A minor · 2 variations").

### C.6 The licence screen (gated or restricted models)

A modal (`.modal-overlay` / `.modal-content`, existing) opened from the licence chip, the `Accept licence` action, or automatically on first Download of a gated row.

```
┌ Stable Audio Open Small ─────────────────────────────────────────┐
│ Stability Community License                                        │
│                                                                    │
│ What this licence asks of you                                      │
│  • Free for personal use and for organisations under US$1M a year  │
│  • Show "Powered by Stability AI" where you publish the output     │
│  • Do not use the output to train other foundation models          │
│  • You own what you make                                           │
│                                                                    │
│ This model is gated: accept the terms on Hugging Face, then        │
│ download here with your token (Settings > Hugging Face).           │
│                                                                    │
│ [Read the full licence]   [Open on Hugging Face]                   │
│                                      [Not now]  [I have accepted]  │
└────────────────────────────────────────────────────────────────────┘
```

Rules: bullet text comes from the catalog (`licence.summaryLines`), not from component code, so a changed licence is a catalog change. "I have accepted" records `{modelId, licenceId, acceptedAt}` in settings; the status chip then changes from Needs licence to Not installed and Download is enabled. The same modal shows the ACE-Step MIT note from the plan ("Trained on data the authors describe as legally compliant…") with only an OK button.

---

## D. Screen-by-screen wireframes

Widths: 1280 × 800 (the design width) and 760 (the narrowest supported, below the `@media (max-width: 820px)` break that stacks the two-column `content-grid`). Every screen reuses `.content-grid` + `Panel`.

### D.1 Music Models (`music-models`)

Purpose: find, download and manage music and sound models. Primary action: Download (per row). Visible by default: the `ModelsPage` shell with the C.1 card. Behind disclosure: row detail. Uses `ModelsPage` with a new `adapters/music.ts`.

```
1280 ─────────────────────────────────────────────────────────────────────────
Music Models      4 of 6 models · 1 installed                 [Open Studio]
[search          ] [All ▾ Installed Available Downloading Attention] [Makes ▾] [Licence ▾] [Sort: Best fit ▾]
Model                     Makes             Size    Fit          Licence                Status
▸ ACE-Step 1.5 turbo ★   Songs · Instr.    3.6 GB  Fits  est.   MIT · commercial       Installed   [Open in Studio] [Delete]
▸ ACE-Step 1.5 XL        Songs · Instr.    18.8 GB Tight est.   MIT · commercial       Not installed [Download]
▸ Magenta RealTime 2     Instrumental      2.1 GB  Fits  est.   CC-BY 4.0 · credit     Not installed [Download]
▸ Stable Audio Open S.   Sound effects     1.2 GB  Fits  est.   Gated · under US$1M    Needs licence [Accept licence]
```

760: columns Makes and Licence drop (`narrowHide`), licence stays as a chip under the name; actions wrap under the row.

States: empty catalog ("No music models in the catalog yet"), filtered empty (existing), offline (page shows "Waiting for the backend…"), downloading (progress in the status chip, Pause / Cancel), failed (row callout with the friendly reason and Retry), needs licence (C.6).

Keyboard: the shell already gives sortable headers and buttons; the row toggle is a button with `aria-expanded`. Screen reader: the fit badge carries `aria-label` with the reason (exists).

### D.2 Music Studio (`music-studio`)

Purpose: make a track or a sound effect. Primary action: **Make track** (or **Make sound** in the Sounds mode). Default view: 6 controls.

Simple view (default):

```
1280 ─────────────────────────────────────────────────────────────────────────
Music Studio    ACE-Step 1.5 turbo · Installed · Fits (est.)         [Models] [Library]
[ Song | Instrumental | Sound effect ]                       ← segmented, 1 control
Describe the music
[ warm lo-fi hip hop, vinyl crackle, mellow piano, 85 bpm                    ]  ← 2
Lyrics                                       (hidden in Instrumental and Sound effect)
[ (verse) ...                                                                ]  ← 3
Length   [15 s] [30 s] [1 min] [2 min] [Custom]                               ← 4
Quality  [Fast · draft] [Balanced] [Best · slower]                            ← 5
About 35 s · Peak 7 GB of 64 GB · Writes 6 MB   Estimated        [ Make track ]  ← 6 + action
> Options     BPM auto · Key auto · 1 variation · Random seed
> Expert      Model defaults
──────────────────────────────────────────────────────────────────────────────
Latest
┌ lo-fi piano · 30 s · 00:30 ─────────────────────────────────────────────┐
│ ▁▂▃▅▆▅▃▂▁▂▃▅▇▅▃▁  ▶  ⟲ loop   00:12 / 00:30                              │
│ Done in 33 s · 6 MB · seed 8142          [Save to Library] [Send to storyboard ▾] [Again] [Vary] │
└──────────────────────────────────────────────────────────────────────────┘
```

Options open (one `.disclosure` per group, each with a value summary in the summary line):

```
▾ Options
  Tempo and key     BPM [auto ▾ | 85 ]      Key [auto ▾ | A minor ]
  Variations        [1] [2] [3] [4]          (each adds the same time again: "about 2 min 20 s for 4")
  Seed              [x] Random    Seed [        ]
  Reference         [Choose audio] [From library]   (shown when the model supports it; "later" per plan)
```

Expert open:

```
▾ Expert       Steps 27 · Guidance 7 · Engine ACE-Step
  Model         [ACE-Step 1.5 turbo ▾]   (moves here; the header chip shows the choice)
  Steps         ━━━━●━━━━ 27            Sharper, slower  (tooltip: inference steps)
  Guidance      ━━━●━━━━━ 7             Follows the prompt more closely
  Sample rate   48 kHz (fixed by the model)
  Output        [FLAC] [WAV]
```

Sound effect mode: the same screen with "Describe the sound" ("rain on a tin roof, distant thunder"), Length pills 2 s / 5 s / 10 s / Custom (max from the model, 11 s for Small), no Lyrics, no BPM / Key. When Stable Audio Open is not installed the mode button is enabled but the form is replaced by the Needs-licence callout (C.4) with [Accept licence] and [Download].

Lyrics and rights: a one-line note under Lyrics, muted: "You are responsible for rights in lyrics you type." When the style prompt contains a known artist name (backend check), a `.callout.warning`: "Naming an artist can produce a copy of their style. Describe the sound instead." (not blocking).

States:
- Empty results: "No music yet. Describe a track and press Make track."
- No model: Models-page callout with the size.
- Queued: strip second line "Another job is running (Image Studio, about 1 min left). Yours starts after it."
- Running: `LiveProgress` (new accent `audio`, the existing waveform animation) with phases Load model → Compose → Decode audio → Save, step counter, queue line, Cancel.
- Done: result card above, estimate replaced by the measured line.
- Error: `.callout.error` "The model ran out of memory. Try a shorter length or the Fast preset." [Use Fast preset] [Technical details].
- Offline: "Waiting for the backend…".

Keyboard: Cmd/Ctrl+Enter makes; Space on a focused player toggles play; L toggles loop. Screen reader: the player exposes `aria-label="Play lo-fi piano, 30 seconds"`, the waveform canvas is `aria-hidden` with a `<progress>`-like text equivalent "00:12 of 00:30".

760: segmented control wraps to two rows, Length and Quality pills wrap, estimate strip stacks the three items above a full-width button, result card actions wrap.

### D.3 Music Library (`music-library`)

Purpose: find, play, tag and reuse tracks and sounds. Primary action: Play (per card). Built on the Voice Library pattern (`VoiceLibraryTab.tsx`, `.voice-library-card`), renamed to `.media-library-card` so Voice and Music share it.

```
1280 ─────────────────────────────────────────────────────────────────────────
Music Library     12 tracks · 4 sounds · 86 MB on disk                [Open Studio]
[search titles, prompts, tags   ] [All | Tracks | Sounds]  [Tag ▾]   [Delete selected (2)]
┌ [ ] lo-fi piano               Track · 30 s · ACE-Step 1.5 turbo · 6 MB · 2 Nov ───┐
│ ▁▂▃▅▆▅▃▂▁▂▃▅▇▅▃▁ ▶ ⟲   00:00 / 00:30           tags: lo-fi, piano  [+ tag]        │
│ [Open in Studio] [Send to storyboard ▾] [Export ▾] [Rename] [Delete]              │
└───────────────────────────────────────────────────────────────────────────────────┘
┌ [ ] rain on tin roof          Sound · 8 s · Stable Audio Open · 1 MB · 2 Nov ─────┐
```

Drag: a card is draggable (`draggable`, `dataTransfer` type `application/x-chaos-media` with `{kind, id}`) onto a Mixer lane; the keyboard path is "Send to storyboard ▾" (project · lane).

States: loading, empty ("Nothing here yet. Tracks you save in Music Studio are kept here."), filtered empty ("Nothing matches."), offline, delete confirm (existing dialog text pattern), export working.

760: cards stack; the toolbar wraps; actions wrap into two rows.

### D.4 Storyboard Projects (`story-projects`)

Purpose: open, create, import or back up a project. Primary action: **New project**.

```
1280 ─────────────────────────────────────────────────────────────────────────
Storyboard Projects    3 projects · 1.2 GB on disk          [New project] [Open folder] [Import bundle]
┌ Lighthouse keeper ───────────┐ ┌ Coffee shop ad ─────────────┐ ┌ Untitled 3 ────────────────┐
│ [still] [still] [still]      │ │ [still] [still]             │ │ (no shots yet)             │
│ 8 shots · 32 s · 16:9        │ │ 4 shots · 2 stale · 15 s    │ │ 0 shots                    │
│ 640 MB · edited 2 h ago      │ │ 410 MB · edited yesterday   │ │ 2 MB · edited 3 d ago      │
│ [Open]  [Export bundle] [⋯]  │ │ [Open]  [Export bundle] [⋯] │ │ [Open] [Delete]            │
└──────────────────────────────┘ └─────────────────────────────┘ └────────────────────────────┘
```

`⋯` (icon button, monochrome) holds Rename, Duplicate, Reveal folder, Purge cache ("frees 120 MB"), Delete. New project dialog: name, aspect preset (16:9 / 9:16 / 1:1), fps (24 / 30), default image and video model (prefilled from the Studios' current selection), a "Write shots from a brief" checkbox that opens the Board with the brief box focused.

States: empty ("No projects yet. A project holds your shots, takes and sound in one folder you can move or back up." [New project]), loading, missing folder ("This project's folder was moved or deleted." [Locate] [Remove from list]), import failed ("This file is not a ChaosEngineAI project bundle." with Technical details), offline.

760: one card per row.

### D.5 Storyboard Board (`story-board`)

Purpose: write, generate and arrange the shots. Primary action: **Make shots** (generates every unlocked shot that is idle or stale; batch, one Cancel). Default view: the brief box, the shot list, the batch estimate and the primary action. Behind disclosure: per-shot settings, project defaults.

```
1280 ─────────────────────────────────────────────────────────────────────────
Lighthouse keeper · Board     8 shots · 32.4 s · 2 stale                [Projects] [Mixer]
Brief  [ A 30 s ad for a quiet lighthouse hotel on a stormy coast, warm and hopeful  ] [Write shots]
> Project defaults    16:9 · 24 fps · FLUX.1-dev · LTX-2.3 · voice Heart
2 shots need a new take · About 9 min · Peak 21 GB of 64 GB · Writes 180 MB  Estimated   [ Make shots ]
──────────────────────────────────────────────────────────────────────────────
┌ 1  Arrival at dusk ────────────────┐ ┌ 2  The keeper lights the lamp ──┐ ┌ 3  Storm rolls in ──── STALE ─┐
│ ┌───────────────┐  Done · 4.04 s   │ │ ┌───────────────┐  Done · 4.0 s │ │ ┌───────────────┐ Stale · 4 s │
│ │   still/clip  │  seed 118        │ │ │   still/clip  │  locked       │ │ │   still/clip  │ after shot 2│
│ └───────────────┘                  │ │ └───────────────┘               │ │ └───────────────┘             │
│ takes  [t1][t2*][t3]  [+ Redo ▾]   │ │ takes  [t1*]        [+ Redo ▾]  │ │ takes  [t1*]  [Re-run] [Keep] │
│ [lock] [cut ▾] [⋯]                 │ │ [locked] [crossfade 0.5 s] [⋯]  │ │ [lock] [cut ▾] [⋯]            │
└────────────────────────────────────┘ └─────────────────────────────────┘ └───────────────────────────────┘
```

Shot card (`ShotCard`), top to bottom: order number and title (editable on click), the still or clip thumbnail (hover shows a monochrome play icon; click plays in place), status line with measured seconds (never the requested seconds), the **takes strip** (thumbnails, active take marked, keyboard selectable), **Redo ▾** (Still only / Clip only / Both; "New take" never overwrites), **lock** toggle (icon button, `aria-pressed`), **transition** picker (Cut / Crossfade with seconds), `⋯` menu (Edit prompts, Import footage, Add sound to this clip (when V2A exists), Move up / Move down, Duplicate, Delete).

Stale badge (`.stale-badge`, `--warning` token): shown when a dependency changed; text "Stale · after shot 2" with tooltip "Shot 2 got a new take; this shot starts from its last frame." Two actions: Re-run (new take) or Keep (clears the badge).

Shot editor (a drawer, `.drawer`, opened by Edit prompts or by double-click): still prompt, motion prompt, reference images (`InputImagePicker`), start from (Generate / Upload / Previous shot), end image, length pills, seed, native sound level; the same estimate strip for this one shot and a **Make this shot** button.

Drag to reorder: HTML5 drag on the card handle; keyboard: Move up / Move down in the menu; `aria-live` announces "Shot 3 moved to position 2".

States per shot: idle (grey dot), queued ("Queued · 3rd"), running (`LiveProgress` mini bar on the card, step counter), done, failed (`.callout.error` on the card: "The clip did not render: out of memory. Try a shorter shot." [Retry] [Technical details]), stale, missing ("Take file missing" [Relink] [New take]), locked.

Board states: empty ("No shots yet. Write a brief and press Write shots, or add a shot." [Write shots] [Add shot]), loading, batch running (one sticky bar at the top: "Making shots · stills 3 of 8 · about 6 min left · [Cancel batch]"), offline, project revision conflict ("Another window changed this project. Reload to continue." [Reload]).

Keyboard: cards are a `role="list"`; arrow keys move focus between cards (roving tabindex); Enter opens the editor; L toggles lock; R redoes; Delete asks to delete. Screen reader: each card is `aria-labelledby` the title and `aria-describedby` the status line.

760: one card per row, takes strip scrolls horizontally, the defaults disclosure and the estimate strip stack.

### D.6 Storyboard Mixer (`story-mixer`)

Purpose: hear and see the film, place sound, export. Primary action: **Export**. Default view: the preview, the timeline with four lanes, transport, the export strip. Behind disclosure: clip inspector (opens on selection), export options.

```
1280 ─────────────────────────────────────────────────────────────────────────
Lighthouse keeper · Mixer     32.4 s · 16:9 · preview 768 × 432                  [Board] [Projects]
┌────────────────────────── preview ──────────────────────────┐  ┌ Selected: music "lo-fi piano" ─────┐
│                                                             │  │ Gain      ━━━━●━━━━  -6 dB           │
│                   (frame at 00:12.4)                        │  │ Fade in   [0.5 s]  Fade out [1.5 s]  │
│                                                             │  │ Trim      in 00:00  out 00:30        │
└─────────────────────────────────────────────────────────────┘  │ [x] Duck under voice   [ ] Loop      │
[▶] [■]  00:12.4 / 00:32.4   [⟲]   zoom [-][+]                   │ Anchor  shot 1 + 0.0 s               │
                                                                  │ [Remove from lane]                   │
0s        5s        10s       15s       20s       25s       30s   └──────────────────────────────────────┘
Picture  [ 1 Arrival ][ 2 Keeper  ][ 3 Storm ][ 4 ... ]  ▌ playhead
Shot sound [▁▃▅▃▁   ][▁▂▁       ][▅▇▅▃     ][ ...     ]  per-shot level in the inspector
Voice    [ "Some nights…" ▁▃▅▃▁ ]            [ "…the lamp" ▁▂▃ ]
Effects          [ thunder ▅▇▅ ]       [ gull ▁▂ ]
Music    [ lo-fi piano ▁▂▃▅▆▅▃▂▁▂▃▅▇▅▃▁ ... (ducked under voice: shaded) ...  ]
                                                                 [+ Add from library ▾] [Import audio]
──────────────────────────────────────────────────────────────────────────────
Captions [ ] burn in   [x] SRT file        AI label  [x] file metadata  [ ] visible end card
Export  [16:9 1080p] [9:16 1080p] [1:1 1080p] [Custom]     About 2 min · Writes 48 MB   [ Export MP4 ]
```

Lanes (`TimelineLane`): Picture (read-only here; click a block to jump to that shot on the Board), Shot sound (the clips' own audio, per-shot gain), Voice, Effects, Music. Four audio lanes, fixed in the first version (plan risk 8). Clips (`TimelineClip`) show a cached waveform (`WaveformCanvas`, peaks from the backend `cache/`), name, and drag handles for trim at the ends and move in the middle. Dropping a Library card onto a lane creates a clip anchored to the shot under the drop point.

Ducking: a lane-level toggle on Music ("Duck under voice"), shown in the inspector and as a small shaded region on the music clip where voice plays. Transitions shorten the timeline; the ruler shows the real total.

Transport: Play / Stop, time, loop toggle, zoom. Preview is the low-resolution cached render (plan 5.4); when a shot's intermediate is missing, the picture lane block shows "Rendering preview…" and plays black with sound.

Export panel: presets 16:9 / 9:16 / 1:1 (a change reframes at export, the note says so: "Reframed at export; shots are not regenerated"), captions (SRT sidecar on by default, burn-in off), AI label options per D-34 (metadata on by default with a Settings switch for private use; visible end card off; cloned-voice projects force the metadata on and say so), estimate strip, Export. After export: "Exported · 47 MB · 00:32" [Reveal file] [Open]. Long renders keep the machine awake and notify on finish (plan 5.5).

States: empty (no shots: "Nothing to mix yet. Make shots on the Board first." [Open Board]); loading waveforms (clip shows a muted bar "Reading…"); preview rendering (progress on the transport line); exporting (`LiveProgress`, phases Normalise shots → Join → Mix sound → Encode; queue line; Cancel); done; failed ("Export failed while mixing sound. Nothing was written." [Retry] [Technical details]); missing media (clip hatched with "Missing" and a Relink button); offline.

Keyboard: Space play / pause; Left / Right nudge the playhead 1 s (Shift: 10 s; Alt: one frame); Home / End; `[` and `]` set trim in / out of the selected clip; Delete removes the selected clip (undo with Cmd/Ctrl+Z, from `history/`); `+` / `-` zoom; Tab moves between transport, lanes, clips (roving) and the inspector. Screen reader: lanes are `role="list"`, clips are `role="listitem"` with `aria-label="Music, lo-fi piano, from 0 to 30 seconds, minus 6 dB"`; the inspector is a `role="region"` labelled "Selected clip"; drag-free equivalents for every edit live in the inspector (anchor shot, offset, trim numbers).

760: inspector becomes a drawer under the timeline; lanes keep their names in a sticky left column of 72 px; the export row wraps into three lines (captions, AI label, presets + button).

### D.7 The licence screen

See C.6. Reached from: Models page chip or action, the Music Studio Sound effect mode when the model is missing, and the first Download of a gated row. Keyboard: a modal with focus trapped, Escape = Not now. Screen reader: `role="dialog"`, `aria-labelledby` the model name, `aria-describedby` the licence name.

### D.8 First-run empty states (the four doors)

| Screen | Line | Button |
|---|---|---|
| Music Models, nothing installed | "Pick a music model to start. ACE-Step 1.5 turbo (3.6 GB, MIT) fits this machine." | [Download ACE-Step 1.5 turbo] |
| Music Studio, nothing installed | "No music model installed yet." | [Open Music Models] |
| Storyboard Projects, none | "No projects yet. A project holds your shots, takes and sound in one folder you can move or back up." | [New project] |
| Mixer, no shots | "Nothing to mix yet. Make shots on the Board first." | [Open Board] |

Each state is one `.empty-state` with one sentence and one primary button; no illustrations, no second paragraph.

---

## E. Component inventory

### E.1 Reuse (file path · how)

| Component | Path | Use |
|---|---|---|
| `Panel` | `src/components/Panel.tsx` | every screen's frame, header actions |
| `InfoTooltip` | `src/components/InfoTooltip.tsx` | technical names behind plain labels |
| `ModelsPage`, `ModelCard`, `ModelDetailList`, `ModelMemoryCell`, `ModelStatusChip`, `ModelDownloadControls`, `ModelFilterBar` | `src/components/models/` | Music Models page through `adapters/music.ts`; add `"music"` and `"voice"` to `ModelsArea` in `modelEntry.ts` |
| `FitBadge`, `BestFitRibbon` | `src/components/models/FitBadge.tsx` | model card, Studio header chip row, Board defaults line |
| `AcceleratorPills` | `src/components/models/AcceleratorPills.tsx` | platform chips on the model card |
| `RecommendedSettingsPanel` | `src/components/RecommendedSettingsPanel.tsx` | inside the Expert disclosure, not at the top level (see H.6) |
| `LiveProgress` | `src/components/LiveProgress.tsx` | add accent `"audio"` (reuses the waveform animation) and `"storyboard"`; the queue line already exists |
| `InputImagePicker` | `src/components/InputImagePicker.tsx` | shot editor references, start / end images; pattern copied for `InputAudioPicker` |
| `ImageOutputCard` meta row | `src/components/ImageOutputCard.tsx` | the "Done in 33 s · 6 MB · seed" line style |
| Segmented control | `.segmented` / `.segment` (Voice Studio) | Song / Instrumental / Sound effect; Library filters |
| Preset pills | `.preset-row`, `.preset-pill`, `.preset-pill-sub` (Video Studio) | Length, Quality, Export presets |
| Library card | `VoiceLibraryTab.tsx` `LibraryCard` | extracted to `MediaLibraryCard` (Voice keeps its exports menu, Music adds tags and waveform) |
| `SliderField` | `src/components/SliderField.tsx` | gain, fades, steps, guidance |
| Modal | `.modal-overlay`, `.modal-content`, `.modal-header/body/footer` | licence screen, new project dialog |
| `debug-details` | `.debug-details` | Technical details under every error |
| Callouts and empty states | `.callout` (`.warning`, `.error`, `.quiet`), `.empty-state` | all states in C.4 |

### E.2 New components (name · props · behaviour · classes)

| Component | Props | Behaviour | Classes |
|---|---|---|---|
| `StudioShell` | `header, inputs, presets, estimate, options?, expert?, results` (ReactNode slots), `studioId` | lays out C.5; remembers open disclosures per `studioId` in `localStorage` under `chaos.studio.<id>.disclosures` | `.studio-shell`, `.studio-shell__*` |
| `Disclosure` | `id, title, summary?: string, defaultOpen?, open?, onToggle?, children` | native `<details>`; summary line shows `title` and a muted `summary` of current values; chevron is the existing `models-card__chevron` SVG | `.disclosure`, `.disclosure__summary`, `.disclosure__values`, `.disclosure__body` |
| `RunEstimateStrip` | `seconds: number\|null, peakGb: number\|null, totalGb: number\|null, diskMb: number\|null, basis: "measured"\|"estimated"\|null, risk: "safe"\|"caution"\|"danger", queue?: {what, secondsLeft}\|null, ack?: {checked, onChange}, action: ReactNode` | C.2; `aria-live="polite"`; danger renders the acknowledgement checkbox | `.run-estimate`, `.run-estimate__item`, `.run-estimate__item--time\|--memory\|--disk`, `.run-estimate--caution\|--danger`, `.run-estimate__basis`, `.run-estimate__queue`, `.run-estimate__action` |
| `LicenceChip` | `licence: {id, name, commercial: "yes"\|"limited"\|"no", gated: boolean, summary}, onOpen?` | text + tooltip; click opens `LicenceScreen` | `.licence-chip`, `.licence-chip--yes\|--limited\|--no`, `.licence-chip--gated` |
| `LicenceScreen` | `entry, licence, accepted: boolean, onAccept, onDismiss, onOpenHub, onOpenFull` | C.6 modal | `.licence-screen`, `.licence-screen__terms` |
| `MakesChips` | `makes: Array<"songs"\|"instrumental"\|"sfx"\|"loops">` | chips with i18n labels | `.makes-chip` |
| `AudioPlayer` | `src, durationS, peaks?: number[], loop?, onLoopChange?, label` | play / stop, loop, time, waveform from `peaks` or a flat bar; one player plays at a time (reuse `speechPlayer` owner pattern) | `.audio-player`, `.audio-player__wave`, `.audio-player__time`, `.audio-player__controls` |
| `WaveformCanvas` | `peaks: number[], progress?: number, height?` | canvas, `aria-hidden`; colours from `--muted` and `--accent` | `.waveform` |
| `InputAudioPicker` | same shape as `InputImagePicker` with `library: MusicLibrarySummary[]` | Choose file (wav, mp3, flac, m4a) / From library | `.input-audio-picker` (reuses `.input-image-*` layout) |
| `MediaLibraryCard` | `item, selected, onSelect, actions: ReactNode, player?: ReactNode, tags?` | extracted from Voice `LibraryCard` | `.media-library-card`, `__head`, `__title`, `__meta`, `__tags` |
| `SendToStoryboardMenu` | `item: {kind, id}, projects, onSend(projectId, lane)` | menu button; the keyboard path for drag | `.send-to-menu` |
| `ProjectCard` | `project, onOpen, onExport, menu` | D.4 | `.project-card`, `__thumbs`, `__meta`, `__actions` |
| `ShotCard` | `shot, index, selected, onSelect, onRedo(what), onLock, onTransition, onReorder(dir), onOpenEditor, onKeep, onRelink` | D.5 | `.shot-card`, `--idle\|--queued\|--running\|--done\|--failed\|--stale\|--missing\|--locked`, `__media`, `__status`, `__actions` |
| `TakesStrip` | `takes, activeId, onPick, onDelete` | horizontal thumbnails; roving tabindex; active marked | `.takes-strip`, `.takes-strip__take`, `.takes-strip__take--active` |
| `StaleBadge` | `reason: string` | `--warning` badge with tooltip | `.stale-badge` |
| `TransitionPicker` | `value: {type, seconds}, onChange` | Cut / Crossfade + seconds | `.transition-picker` |
| `ShotEditorDrawer` | `shot, defaults, gallery, onChange, onMake, estimate` | D.5 drawer | `.drawer`, `.drawer__head`, `.drawer__body` |
| `BatchBar` | `batch: {done, total, kind, secondsLeft}, onCancel` | sticky bar during a Board batch | `.batch-bar` |
| `Timeline` | `durationS, zoom, playheadS, lanes: LaneModel[], selectedClipId, onSelect, onMove, onTrim, onDrop, onSeek` | D.6 | `.timeline`, `.timeline__ruler`, `.timeline__lanes`, `.timeline__playhead` |
| `TimelineLane` | `lane: {id, kind, label, clips}, …` | one lane row; drop target | `.timeline__lane`, `--picture\|--native\|--voice\|--sfx\|--music`, `.timeline__lane-label` |
| `TimelineClip` | `clip, pxPerSecond, selected, …` | draggable block with trim handles and waveform | `.timeline__clip`, `--selected\|--missing\|--ducked`, `.timeline__clip-handle` |
| `TimeRuler` | `durationS, pxPerSecond` | ticks every 1 / 5 / 10 s by zoom | `.time-ruler` |
| `Transport` | `playing, timeS, durationS, loop, onPlay, onStop, onLoop, onZoom` | D.6 transport row; keyboard handlers live here | `.transport` |
| `ClipInspector` | `clip, shots, onChange, onRemove` | gain, fades, trim, duck, loop, anchor | `.clip-inspector`, `__row` |
| `ExportPanel` | `presets, value, captions, aiLabel, estimate, onExport, result?` | D.6 export row | `.export-panel`, `__presets`, `__options`, `__result` |
| `AiLabelNotice` | `metadata: boolean, endCard: boolean, forced: boolean, onChange` | the AI-label options with the D-34 copy | `.ai-label-notice` |

### E.3 Tokens and new CSS

All colours through existing tokens. Three additions to `:root` so lanes never hardcode a hex outside `:root`:

```
--lane-voice:  color-mix(in srgb, var(--accent) 22%, var(--panel));
--lane-sfx:    color-mix(in srgb, var(--warning) 18%, var(--panel));
--lane-music:  color-mix(in srgb, var(--success) 18%, var(--panel));
--lane-native: var(--panel-alt);
```

Spacing from `--panel-gap` / `--panel-padding`; radii from `--radius-sm/md`; the stale badge uses `--warning`, missing uses `--danger`, done uses `--success`. Icons: add `MusicIcon` (a single note), `StoryboardIcon` (three frames) and small action icons (lock, redo, play, loop, duck) to `src/components/icons/SidebarIcons.tsx` (or a sibling `ActionIcons.tsx`), all `stroke="currentColor"`, no fill colours.

---

## F. Navigation and information architecture

### F.1 Sidebar

Today: four single tabs (Dashboard, Server, Logs, Settings) and six groups (Chat, Images, Video, Voice, Benchmarks, Tools). Adding Music and Storyboard makes eight groups; with every group open that is 32 rows, more than an 800 px window shows.

Proposal, in order of least change:

1. **Accordion by default.** `Sidebar.tsx` already keeps `collapsedGroups` and opens the group with the active tab. Make "only the active group open" the default state (a Settings switch "Keep all sidebar groups open" restores today's behaviour). Result at 800 px: 4 singles + 8 group headers + 3 children = 15 rows.
2. **Two thin section labels** (`.nav-section-label`, uppercase muted eyebrow, not clickable): "Make" above Images (Images, Video, Voice, Music, Storyboard) and "Run" above Chat and Benchmarks (Chat, Benchmarks, Tools). Dashboard stays first; Server, Logs, Settings stay last.
3. **Not now, but logged:** move Benchmarks under Tools (saves a group header).

Group order within Make: Images, Video, Voice, Music, Storyboard (the pipeline order: stills → clips → speech → music → assemble).

### F.2 Tab ids and labels

| Tab id | Group | Label | Short | Caption |
|---|---|---|---|---|
| `music-models` | `music` | Music Models | Models | Find, download and manage music and sound models |
| `music-studio` | `music` | Music Studio | Studio | Make tracks and sound effects |
| `music-library` | `music` | Music Library | Library | Tracks and sounds you made |
| `story-projects` | `storyboard` | Projects | Projects | Your films, one folder each |
| `story-board` | `storyboard` | Board | Board | Shots, takes and prompts |
| `story-mixer` | `storyboard` | Mixer | Mixer | Picture, sound and export |

Group `music` default child: `music-studio` when a model is installed, else `music-models` (same rule the plan should apply to Images and Video; today both open on Models). Group `storyboard` default child: `story-projects`, or `story-board` when a project is open.

### F.3 Moving between areas

| From | Action | Lands on |
|---|---|---|
| Music Studio result / Music Library card | Send to storyboard ▾ → project → lane (Music or Effects) | Mixer, the new clip selected, anchored at the playhead |
| Image Gallery card, Video Gallery card | Use in storyboard ▾ → project → "New shot" or "Shot n (new take)" | Board, the shot card selected; an image becomes the shot's still, a clip becomes a take |
| Board shot card ⋯ | Open in Image Studio / Open in Video Studio | the Studio with the shot's prompt, seed and references loaded (`onUseSameSettings` pattern) and a "Back to shot n" link in the header; Generate there offers "Save as take of shot n" |
| Mixer picture block | click | Board with that shot selected |
| Voice Studio Speak result | Send to storyboard ▾ | Mixer, Voice lane |
| Dashboard | "Continue: Lighthouse keeper · 2 shots stale" tile | Board |

The menu lists the three most recent projects and "New project…". When no project exists, the item is "Start a storyboard with this" and creates one.

---

## G. Copy deck

Keys follow `studio.json` (`area.section.key`) and `common.json` (tabs). English text only; other locales get `null`. Tooltips carry the technical name.

### G.1 common.json

```
tabs.musicModels "Music Models" · tabs.musicStudio "Music Studio" · tabs.musicLibrary "Music Library"
tabs.storyProjects "Projects" · tabs.storyBoard "Board" · tabs.storyMixer "Mixer"
tabShortLabels.musicStudio "Studio" · tabShortLabels.musicLibrary "Library" · tabShortLabels.storyProjects "Projects" · tabShortLabels.storyBoard "Board" · tabShortLabels.storyMixer "Mixer"
tabCaptions.musicModels "Find, download and manage music and sound models" · tabCaptions.musicStudio "Make tracks and sound effects" · tabCaptions.musicLibrary "Tracks and sounds you made"
tabCaptions.storyProjects "Your films, one folder each" · tabCaptions.storyBoard "Shots, takes and prompts" · tabCaptions.storyMixer "Picture, sound and export"
tabGroups.music "Music" · tabGroups.storyboard "Storyboard"
sidebar.sectionMake "Make" · sidebar.sectionRun "Run"
```

### G.2 studio.json: shared patterns

```
estimate.time "About {time}" · estimate.timeUnknown "Time: unknown until the first run"
estimate.memory "Peak {peak} of {total}" · estimate.memoryUnknown "Memory: not estimated"
estimate.disk "Writes {size}" · estimate.measured "Measured" · estimate.estimated "Estimated"
estimate.measuredTitle "Timed on this machine for this model and preset." · estimate.estimatedTitle "From the model card or a similar machine; not timed here yet."
estimate.queueRunning "{what} is running, about {time} left. Yours starts after it."
estimate.queuePosition "Waiting for another job to finish (position {position})"
estimate.dangerAck "Run anyway. I accept that the backend may crash."
disclosure.options "Options" · disclosure.expert "Expert" · disclosure.defaults "Model defaults"
preset.custom "Custom" · preset.customTitle "A value was changed by hand. Pick a preset to go back."
error.technicalDetails "Technical details" · error.retry "Retry"
licence.chip.commercial "{name} · commercial" · licence.chip.limited "{name} · limited commercial" · licence.chip.noCommercial "{name} · no commercial use" · licence.chip.gated "Gated"
licence.screen.asks "What this licence asks of you" · licence.screen.gatedNote "This model is gated: accept the terms on Hugging Face, then download here with your token (Settings > Hugging Face)."
licence.screen.readFull "Read the full licence" · licence.screen.openHub "Open on Hugging Face" · licence.screen.notNow "Not now" · licence.screen.accepted "I have accepted" · licence.screen.ok "OK"
licence.status.needs "Needs licence" · licence.action.accept "Accept licence"
models.makes.songs "Songs with vocals" · models.makes.instrumental "Instrumental" · models.makes.sfx "Sound effects" · models.makes.loops "Loops"
models.speed.measured "About {rate} s of audio per second on this machine" · models.speed.unknown "Speed not measured yet"
```

### G.3 studio.json: music

```
musicModels.title "Music Models" · musicModels.subtitle "{shown} of {total} models · {installed} installed"
musicModels.search "Search models, licence, what they make" · musicModels.openStudio "Open Studio"
musicModels.empty.catalog "No music models in the catalog yet."
musicModels.empty.firstRun "Pick a music model to start. {name} ({size}, {licence}) fits this machine." · musicModels.empty.firstRunAction "Download {name}"
musicModels.filter.makes "Makes" · musicModels.filter.licence "Licence" · musicModels.filter.commercialOnly "Commercial use ok"

musicStudio.title "Music Studio" · musicStudio.subtitle "Describe a track, pick a length, press Make."
musicStudio.mode.song "Song" · musicStudio.mode.instrumental "Instrumental" · musicStudio.mode.sfx "Sound effect"
musicStudio.prompt.label "Describe the music" · musicStudio.prompt.placeholder "warm lo-fi hip hop, vinyl crackle, mellow piano, 85 bpm"
musicStudio.prompt.sfxLabel "Describe the sound" · musicStudio.prompt.sfxPlaceholder "rain on a tin roof, distant thunder"
musicStudio.lyrics.label "Lyrics" · musicStudio.lyrics.placeholder "[verse] ... [chorus] ..." · musicStudio.lyrics.rights "You are responsible for rights in lyrics you type."
musicStudio.artistWarning "Naming an artist can produce a copy of their style. Describe the sound instead."
musicStudio.length.label "Length" · musicStudio.length.custom "Custom" · musicStudio.length.seconds "{n} s" · musicStudio.length.minutes "{n} min"
musicStudio.quality.fast "Fast" · musicStudio.quality.fastSub "draft" · musicStudio.quality.balanced "Balanced" · musicStudio.quality.best "Best" · musicStudio.quality.bestSub "slower"
musicStudio.make "Make track" · musicStudio.makeSound "Make sound" · musicStudio.making "Making…"
musicStudio.options.tempoKey "Tempo and key" · musicStudio.options.bpm "BPM" · musicStudio.options.key "Key" · musicStudio.options.auto "Auto"
musicStudio.options.variations "Variations" · musicStudio.options.variationsCost "Each variation adds the same time again."
musicStudio.options.seed "Seed" · musicStudio.options.randomSeed "Random seed"
musicStudio.options.reference "Reference audio" · musicStudio.options.referenceRights "You are responsible for rights in audio you supply."
musicStudio.expert.model "Model" · musicStudio.expert.steps "Sharper, slower" · musicStudio.expert.stepsTooltip "Inference steps: {n}" · musicStudio.expert.guidance "Follows the prompt more closely" · musicStudio.expert.guidanceTooltip "Guidance scale: {n}"
musicStudio.expert.sampleRate "Sample rate {rate} kHz (fixed by the model)" · musicStudio.expert.output "Output"
musicStudio.result.done "Done in {time} · {size} · seed {seed}" · musicStudio.result.save "Save to Library" · musicStudio.result.saved "Saved" · musicStudio.result.send "Send to storyboard" · musicStudio.result.again "Again" · musicStudio.result.vary "Vary"
musicStudio.result.loop "Loop" · musicStudio.result.play "Play {title}, {seconds} seconds" · musicStudio.result.stop "Stop"
musicStudio.empty "No music yet. Describe a track and press Make track." · musicStudio.noModel "No music model installed yet." · musicStudio.openModels "Open Music Models"
musicStudio.sfxNeedsModel "Sound effects use {name}, which needs a licence accept before download."
musicStudio.progress.load "Load model" · musicStudio.progress.compose "Compose" · musicStudio.progress.decode "Decode audio" · musicStudio.progress.save "Save"
musicStudio.error.oom "The model ran out of memory. Try a shorter length or the Fast preset." · musicStudio.error.useFast "Use Fast preset"
musicStudio.error.cancelled "Cancelled." · musicStudio.error.generic "The track could not be made."

musicLibrary.title "Music Library" · musicLibrary.subtitle "{tracks} tracks · {sounds} sounds · {size} on disk"
musicLibrary.search "Search titles, prompts, tags" · musicLibrary.filter.all "All" · musicLibrary.filter.tracks "Tracks" · musicLibrary.filter.sounds "Sounds" · musicLibrary.filter.tag "Tag"
musicLibrary.kind.track "Track" · musicLibrary.kind.sound "Sound"
musicLibrary.addTag "Add tag" · musicLibrary.rename "Rename" · musicLibrary.export "Export" · musicLibrary.delete "Delete" · musicLibrary.deleteSelected "Delete selected ({n})"
musicLibrary.openInStudio "Open in Studio" · musicLibrary.confirmDelete "Delete this item and its audio? This cannot be undone." · musicLibrary.confirmDeleteMany "Delete {n} items and their audio? This cannot be undone."
musicLibrary.empty "Nothing here yet. Tracks you save in Music Studio are kept here." · musicLibrary.noMatch "Nothing matches." · musicLibrary.loading "Loading your library…"
sendTo.title "Send to storyboard" · sendTo.newProject "New project…" · sendTo.startWith "Start a storyboard with this" · sendTo.lane.music "Music lane" · sendTo.lane.sfx "Effects lane" · sendTo.lane.voice "Voice lane" · sendTo.newShot "New shot" · sendTo.takeOf "Shot {n} (new take)"
```

### G.4 studio.json: storyboard

```
storyProjects.title "Storyboard Projects" · storyProjects.subtitle "{count} projects · {size} on disk"
storyProjects.new "New project" · storyProjects.openFolder "Open folder" · storyProjects.import "Import bundle" · storyProjects.exportBundle "Export bundle"
storyProjects.card.meta "{shots} shots · {seconds} s · {aspect}" · storyProjects.card.stale "{n} stale" · storyProjects.card.edited "edited {when}" · storyProjects.card.noShots "no shots yet"
storyProjects.menu.rename "Rename" · storyProjects.menu.duplicate "Duplicate" · storyProjects.menu.reveal "Reveal folder" · storyProjects.menu.purge "Purge cache (frees {size})" · storyProjects.menu.delete "Delete"
storyProjects.empty "No projects yet. A project holds your shots, takes and sound in one folder you can move or back up."
storyProjects.missing "This project's folder was moved or deleted." · storyProjects.locate "Locate" · storyProjects.remove "Remove from list"
storyProjects.importFailed "This file is not a ChaosEngineAI project bundle."
storyProjects.dialog.name "Name" · storyProjects.dialog.aspect "Shape" · storyProjects.dialog.fps "Frame rate" · storyProjects.dialog.imageModel "Image model" · storyProjects.dialog.videoModel "Video model" · storyProjects.dialog.voice "Voice" · storyProjects.dialog.fromBrief "Write shots from a brief" · storyProjects.dialog.create "Create"

board.title "{project} · Board" · board.subtitle "{shots} shots · {seconds} s · {stale} stale"
board.brief.label "Brief" · board.brief.placeholder "A 30 s ad for a quiet lighthouse hotel on a stormy coast, warm and hopeful" · board.brief.write "Write shots" · board.brief.writing "Writing shots…"
board.defaults "Project defaults" · board.defaults.summary "{aspect} · {fps} fps · {imageModel} · {videoModel} · voice {voice}"
board.make "Make shots" · board.makeOne "Make this shot" · board.estimate.need "{n} shots need a new take"
board.batch.progress "Making shots · {kind} {done} of {total} · about {time} left" · board.batch.kind.stills "stills" · board.batch.kind.clips "clips" · board.batch.kind.audio "audio" · board.batch.cancel "Cancel batch"
board.addShot "Add shot" · board.empty "No shots yet. Write a brief and press Write shots, or add a shot."
board.shot.status.idle "Not made yet" · board.shot.status.queued "Queued · {position}" · board.shot.status.running "Making…" · board.shot.status.done "Done · {seconds} s" · board.shot.status.failed "Failed" · board.shot.status.stale "Stale · after shot {n}" · board.shot.status.missing "Take file missing" · board.shot.status.locked "Locked"
board.shot.staleTitle "Shot {n} got a new take; this shot starts from its last frame." · board.shot.rerun "Re-run" · board.shot.keep "Keep"
board.shot.redo "Redo" · board.shot.redo.still "Still only" · board.shot.redo.clip "Clip only" · board.shot.redo.both "Still and clip" · board.shot.newTake "New take"
board.shot.lock "Lock this shot" · board.shot.unlock "Unlock this shot" · board.shot.lockedNote "Locked shots are skipped by Make shots."
board.shot.transition "Transition" · board.shot.transition.cut "Cut" · board.shot.transition.crossfade "Crossfade {seconds} s"
board.shot.menu.edit "Edit prompts" · board.shot.menu.import "Import footage" · board.shot.menu.addSound "Add sound to this clip" · board.shot.menu.up "Move up" · board.shot.menu.down "Move down" · board.shot.menu.duplicate "Duplicate" · board.shot.menu.delete "Delete" · board.shot.menu.openImage "Open in Image Studio" · board.shot.menu.openVideo "Open in Video Studio"
board.shot.moved "Shot {n} moved to position {position}"
board.shot.relink "Relink" · board.shot.failed.oom "The clip did not render: out of memory. Try a shorter shot."
board.takes.label "Takes" · board.takes.active "Active take" · board.takes.pick "Use this take" · board.takes.delete "Delete take"
board.editor.title "Shot {n}" · board.editor.stillPrompt "Still prompt" · board.editor.motionPrompt "Motion prompt" · board.editor.startFrom "Start from" · board.editor.startFrom.generate "A new still" · board.editor.startFrom.upload "My image" · board.editor.startFrom.previous "Last frame of the previous shot" · board.editor.endImage "End image" · board.editor.length "Length" · board.editor.nativeSound "Shot sound level" · board.editor.backToShot "Back to shot {n}" · board.editor.saveAsTake "Save as take of shot {n}"
board.conflict "Another window changed this project. Reload to continue." · board.reload "Reload"

mixer.title "{project} · Mixer" · mixer.subtitle "{seconds} s · {aspect} · preview {width} × {height}"
mixer.lane.picture "Picture" · mixer.lane.native "Shot sound" · mixer.lane.voice "Voice" · mixer.lane.sfx "Effects" · mixer.lane.music "Music"
mixer.transport.play "Play" · mixer.transport.pause "Pause" · mixer.transport.stop "Stop" · mixer.transport.loop "Loop" · mixer.transport.zoomIn "Zoom in" · mixer.transport.zoomOut "Zoom out"
mixer.addFromLibrary "Add from library" · mixer.importAudio "Import audio" · mixer.clip.reading "Reading…" · mixer.clip.missing "Missing" · mixer.clip.ariaLabel "{lane}, {name}, from {start} to {end} seconds, {gain} dB"
mixer.inspector.title "Selected clip" · mixer.inspector.gain "Gain" · mixer.inspector.fadeIn "Fade in" · mixer.inspector.fadeOut "Fade out" · mixer.inspector.trim "Trim" · mixer.inspector.trimIn "in" · mixer.inspector.trimOut "out" · mixer.inspector.duck "Duck under voice" · mixer.inspector.duckTitle "Lowers the music while someone speaks (sidechain compression)." · mixer.inspector.loop "Loop to fill" · mixer.inspector.anchor "Anchor" · mixer.inspector.anchorShot "shot {n} + {offset} s" · mixer.inspector.remove "Remove from lane"
mixer.inspector.nativeLevel "Shot sound level for shot {n}"
mixer.preview.rendering "Rendering preview…" · mixer.transitionsNote "Transitions shorten the film by their length; the ruler shows the real total."
mixer.captions.label "Captions" · mixer.captions.burnIn "Burn in" · mixer.captions.srt "SRT file" · mixer.captions.source "From the Voice lane (Whisper)."
mixer.export.title "Export" · mixer.export.preset.landscape "16:9 1080p" · mixer.export.preset.portrait "9:16 1080p" · mixer.export.preset.square "1:1 1080p" · mixer.export.preset.custom "Custom" · mixer.export.reframeNote "Reframed at export; shots are not regenerated."
mixer.export.button "Export MP4" · mixer.export.exporting "Exporting…" · mixer.export.done "Exported · {size} · {duration}" · mixer.export.reveal "Reveal file" · mixer.export.open "Open"
mixer.export.phase.normalise "Normalise shots" · mixer.export.phase.join "Join" · mixer.export.phase.mix "Mix sound" · mixer.export.phase.encode "Encode"
mixer.export.failed "Export failed while {phase}. Nothing was written." · mixer.export.keepAwake "The computer stays awake until the export finishes." · mixer.export.notify "You will be notified when it is done."
mixer.empty "Nothing to mix yet. Make shots on the Board first." · mixer.openBoard "Open Board"
aiLabel.title "AI label" · aiLabel.metadata "Mark the file as AI-generated (metadata)" · aiLabel.metadataTitle "A machine-readable tag in the MP4 and audio metadata. On by default; a Settings switch turns it off for private use." · aiLabel.endCard "Add a visible 'AI-generated' end card" · aiLabel.forced "This project uses a cloned voice, so the AI tag is always written." · aiLabel.settingsLink "AI label settings"
voiceClone.consent "I have the right to clone this voice." · voiceClone.consentNote "Consent is recorded with the sample. Cloned speech is always tagged as machine generated."
```

---

## H. De-cluttering proposals for the existing Image and Video Studios

Not to be built now. Ranked by value, then risk. Each reuses a pattern from B to E so the Studios converge on the one anatomy.

| # | Change | Value | Risk | Reuses |
|---|---|---|---|---|
| 1 | **Run estimate strip above Generate** replacing the capacity line, the resource line inside the safety callout and the "will load" hint; the danger acknowledgement moves into the strip | High: cost before the click, in one place | Low: presentation only; `assess*Safety` unchanged | `RunEstimateStrip` |
| 2 | **Expert disclosure** holding Sampler, Diffusion cache + threshold, CFG decay, Preview VAE, FP8, NF4, STG, Fast preview, Steps, Guidance, with a summary line of non-default values; the sentence-long checkbox descriptions move into the tooltips | High: removes about 10 controls and most of the words from the default view | Low: no state or logic change; tests that query by label still pass once the disclosure is open | `Disclosure` |
| 3 | **One status row** in the header (model · engine · Installed / Loaded · fit badge) replacing the hero badges, the stats row and the "Generate disabled" hint; the runtime banner shows only when an action is needed | High: one place to read readiness | Medium: the banners carry install flows; keep them, gate their visibility | `FitBadge`, `LicenceChip` |
| 4 | **Size as one decision**: aspect pills (5, as video) + a size segmented S / M / L (0.5 / 1 / 2 MP for image; 480p / 720p / 1080p for video) with the exact W × H as a caption; Width, Height numbers (and Frames, FPS for video) move under Options; video gets Length pills in seconds (2 s / 4 s / 6 s / 8 s) that set Frames from the model's fps rule | High: 15 controls become 8 | Medium: `sizePresets.ts` tests and the frames-snap rule need care | preset pills, `Disclosure` |
| 5 | **Apply the recommendation on first model pick** with one line "Set up for this machine · Undo", and move `RecommendedSettingsPanel` into Expert | Medium: a first-time user gets working settings without reading | Medium: behaviour change; must not re-apply on every render; owner decision (I.3) | `RecommendedSettingsPanel` |
| 6 | **Options disclosure** for negative prompt, reference strength, end image, images per run, seed | Medium | Low | `Disclosure` |
| 7 | **Result card shows measured vs estimated** ("Done in 38 s, estimated 40 s") and feeds the perf table | Medium: estimates improve | Low | output cards |
| 8 | **Header actions**: Find models and Installed collapse to one "Models" button (the Models page has the filter); Model Card, Load into memory and Unload go into a `⋯` menu | Low: 4 fewer buttons | Low | icon button |
| 9 | **Fit badge in the model select** option label ("FLUX.1-dev · Tight") | Low | Low | `fitLabel` |

Order to build when the time comes: 2, 1, 6 (one PR, presentation only), then 3, then 4, then 5 after I.3.

---

## I. Open design questions for the owner

| # | Question | Recommendation |
|---|---|---|
| 1 | Sidebar: accordion (one group open) by default, with a Settings switch to keep all open? | Yes. It is the only change that keeps eight groups inside 800 px without hiding tabs. |
| 2 | Add the two section labels "Make" and "Run" to the sidebar? | Yes, as plain eyebrows; no new behaviour. Drop them if the accordion alone reads clearly in a screenshot. |
| 3 | Apply the machine recommendation automatically on first model pick (with Undo), in Music now and in Image / Video later (H.5)? | Yes for Music from day one (no existing habits to break); Image / Video after H.1 to H.3 land. |
| 4 | Sound effects: a mode inside Music Studio (segmented, as drawn) or its own tab "Sounds"? | Mode. One screen, one anatomy; the model and licence differ, the form barely does. |
| 5 | AI label defaults in the export panel: metadata on, end card off, per D-34? | Yes. Remember the end-card choice per project; force metadata on for cloned-voice projects and say so. |
| 6 | Mixer first version: four fixed audio lanes (Shot sound, Voice, Effects, Music) or free "add lane"? | Fixed. Free lanes make every mix rule (ducking, loudness, stems) conditional; add later behind P6. |
| 7 | Time estimates: ship with the static table from the model cards and switch to per-machine measurement after the first run, or wait for measured numbers before showing a time at all? | Show the estimate with the `Estimated` chip from day one and replace it with `Measured` after the first run per model and preset. An honest estimate beats a blank. |
| 8 | Narrow layouts: at 760 px, is the Mixer usable with a drawer inspector, or should the Mixer require 900 px and show a "widen the window" note below that? | Drawer at 760. The Board and Studios are fine at 760; only the Mixer is tight, and a note that blocks work is worse than a cramped but working timeline. |

Decisions taken here go to `docs/hub/decisions.md` the same turn they are answered (working agreement).
