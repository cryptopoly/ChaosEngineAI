# Models page, fit badges, recommended settings, size limits

Owner brief, 2026-10-29. Branch `feat/models-fit`, stacked on `fix/hardware-gating-ddtree` (PR 119). Tracker rows: FU-129 (one Models page per area), FU-133 (fit badge and recommended settings, per-model pixel limits).

## What the owner asked for

1. **One "Models" page per area** (Chat, Images, Video) instead of Discover plus My Models. Filters replace the split: Installed / Available / Downloading, plus the existing family, size and platform filters. Old routes stay as redirects.
2. **"Best fit for your hardware" badge** on every model row in all three areas, plus one best-fit pick per family.
3. **Recommended settings for this hardware** in Image Studio and Video Studio (cache strategy, threshold, CFG decay, preview VAE, steps...), one click to apply, each with a reason.
4. **Raise the 2048 px per-side cap** on images (and video) to per-model limits that big machines can use.

Owner's rule of thumb: newer and faster is better. Editorial rule (CLAUDE.md): never name other desktop AI apps in code, comments, UI strings or docs.

## Release gate

The work ships in v0.9.9 only if all of this is true at the end; otherwise it ships as the next release and PR 119 goes out alone:
- full pytest, vitest, tsc and `npm run i18n:validate` pass;
- each of the three areas was opened in the running app and the old Discover / My Models routes redirect;
- no feature regresses (download, resume, delete, launch, "installed" state, search);
- fit and recommendation numbers say "measured" or "estimated" honestly;
- an independent reviewer found no unresolved correctness issue.

## Rules for every agent (read before touching anything)

- Work in `/Users/dan/ChaosEngineAI` on branch `feat/models-fit`. **No worktrees, no `git checkout`, `git stash`, `git reset`, `git commit`, `git push`.** The lead commits. Edit only the files you own (below); if you need a change elsewhere, write it in your progress file under "Requests".
- **Do not edit locale JSON** (`src/locales/*`), `CLAUDE.md`, `docs/hub/*`, `CHANGELOG.md` or `RELEASE_NOTES_v0.9.9.md`: several agents work at once and these files race. Use inline `defaultValue` strings with `t()` and list every new key (`namespace:key = English text`) in your progress file; the lead adds translations.
- Progress file: `docs/hub/progress/<your-stream>.md` (copy `_template.md`), updated at every checkpoint so a fresh agent can resume.
- Run only your own tests (`pytest tests/test_<yours>.py`, `npx vitest run <path>`, `npx tsc --noEmit`). Do not run the whole suite; the lead does.
- Python: standard patterns of the repo (FastAPI routers, `unittest.TestCase`), lazy heavy imports, `pathlib`, list-form subprocess. TypeScript: vitest, factory helpers.
- UI: monochrome SVG icons only, no emoji. Strings through `useTranslation` with `defaultValue`.
- **GPU and memory:** this is one M4 Max with 64 GB. Only the `settings-rec` agent runs model inference. Nobody else loads models or generates. Backend ports: the lead's shared backend is on 8876 (read-only catalog calls are fine); `settings-rec` uses 8890. Vite dev servers: `fit-ui` 5176, `ui-core` 5177, `ui-video` 5178, `ui-chat` 5179; start the backend you need with `CHAOSENGINE_ALLOWED_ORIGINS=http://localhost:<port>` and `CHAOSENGINE_EXTRAS_SITE_PACKAGES=/tmp/ce-no-extras`. Stop what you start.
- Honest reporting: say what you did not test (Windows, Linux, other hardware tiers).

## Shared contracts (code to these; do not change them without telling the lead)

`backend_service/helpers/hardware_class.py` exists: `hardware_class(system) -> HardwareClass(kind, bucket, label, total_gb, usable_gb)`. Its `usable_gb` for Apple is a heuristic (0.75 of RAM); the real Metal working-set limit on this Mac is 51.8 GB of 64, so the `fit-backend` agent verifies it once through `mx.metal.device_info()` in a subprocess (never import mlx in the backend process) and caches it.

**Fit**, attached to every variant in the chat (workspace), image (`/api/images/catalog`) and video (`/api/video/catalog`) payloads:

```json
"fit": {"status": "fits" | "tight" | "wont-fit" | "unknown",
        "needGb": 18.4, "usableGb": 51.8, "headroomPct": 64,
        "reason": "Weights 15.9 GB + 2.5 GB cache at 8K context", "basis": "measured" | "estimated"}
```
and on each family `"bestFitVariantId": "<variant id>" | null`. Thresholds: need <= 70 % of usable is `fits`, <= 92 % is `tight`, above is `wont-fit`. Best fit per family: newest release that `fits`, then the faster one (measured speed, else estimated), then the smaller. `basis` is `measured` only when the number comes from a run on this machine (the catalog's measured `runtimeFootprintGb`, speed history).

**Recommended settings**, attached to image and video variants:

```json
"recommendedSettings": {"hardwareBucket": "apple-64", "basis": "measured" | "estimated",
   "settings": {"cacheStrategy": "teacache", "cacheThreshold": 0.4, "cfgDecay": true,
                "previewVae": false, "steps": 28, "qualityPreset": "balanced"},
   "reasons": {"cacheStrategy": "1.8x faster on FLUX with no visible change"}}
```
Every key in `settings` is optional; the Studio applies only what is present and shows each reason. `basis` is `measured` only for settings timed on this machine.

**Size limits**, attached to image and video variants: `"maxSide": 4096` (largest allowed edge for this model), `"maxPixels": 16777216`, `"nativeSide": 1024`. The Studio inputs, the size guard (`helpers/generation_limits.py`) and the memory gate use these; the global 2048 cap goes away.

## Streams

| Stream | Agent | Owns | Depends on |
|---|---|---|---|
| `map` | Explore | `docs/hub/progress/map-models-pages.md` (read-only research) | none |
| `fit-backend` | sonnet | `backend_service/helpers/fit.py`, `hardware_class.py` refinements, fit fields in the three catalog payload builders, `tests/test_model_fit.py` | none |
| `limits` | sonnet | `backend_service/helpers/generation_limits.py`, `maxSide`/`maxPixels` catalog fields, memory-gate integration, `scripts/measure-image-ceiling.py`, `tests/test_size_limits.py` | none |
| `settings-rec` | opus | `backend_service/helpers/recommendations.py`, catalog `recommendedSettings` data, measurement scripts and results, the "Recommended for this Mac" panel in `ImageStudioTab.tsx` / `VideoStudioTab.tsx` (and the Studio size inputs reading `maxSide`), `tests/test_recommendations.py` | `hardware_class.py`, `limits` field names |
| `ui-core` | opus | `src/components/models/*` (shared Models page shell, card, filter bar, FitBadge), Image area port, `App.tsx` routing and sidebar, redirects | `map`, fit contract |
| `ui-video` | sonnet | Video area port | `ui-core` shell |
| `ui-chat` | sonnet | Chat area port (`OnlineModelsTab` + `MyModelsTab`) | `ui-core` shell |
| `review` | reviewer | read-only diff review at the end | everything |

The lead (main thread) owns commits, locale translations, `CLAUDE.md` / hub / release-notes text, LTX-2.5 validation and the final full test run.
