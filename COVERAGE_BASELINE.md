# Coverage baseline — v0.7.6 → v0.8.0 refactor

Captured 2026-05-09 against `feature/refactor-n-audit @ a53bd5d`.

This file is the floor: any phase PR that lowers a percentage here without a written justification fails CI.

## Python

```
pytest tests/ --cov=backend_service --cov=cache_compression --cov=dflash
```

| Metric | Value |
|---|---|
| Tests passed | 1,302 (1,284 pre-existing + 18 new contract tests) |
| Tests skipped | 1 |
| Line coverage | **61.7%** (11,819 / 19,157) |
| Lines missing | 7,338 |
| Wall time | ~107s |

### Lowest-coverage modules (>50 lines)

| % | Lines | Path | Notes |
|---|---|---|---|
| 0.0 | 308 | `backend_service/ddtree.py` | DFlash-MLX adapter; only exercised on Apple Silicon w/ live model |
| 0.0 | 132 | `cache_compression/_teacache_patches/flux.py` | Vendored FLUX forward — only loaded when TeaCache strategy active on FLUX |
| 0.0 | 87 | `backend_service/task_datasets.py` | Fine-tuning dataset prep, fine-tune flow not wired |
| 0.0 | 61 | `backend_service/helpers/remote_providers.py` | OpenAI/Anthropic remote fallback — runtime only |
| 10.9 | 64 | `cache_compression/_teacache_patches/hunyuan_video.py` | Same pattern — vendored forward only loaded for HunyuanVideo+TeaCache |
| 11.3 | 62 | `cache_compression/_teacache_patches/cogvideox.py` | Same pattern |
| 12.7 | 55 | `cache_compression/_teacache_patches/ltx_video.py` | Same pattern |
| 22.4 | 245 | `backend_service/routes/storage.py` | Move-job + size-walk paths; needs disk-fixture tests |
| 22.8 | 127 | `backend_service/vllm_engine.py` | CUDA-only, dev box can't exercise |
| 26.6 | 109 | `backend_service/helpers/finetuning.py` | Fine-tune flow not wired (TODO in finetuning.py) |
| 39.3 | 178 | `backend_service/routes/images.py` | Heavy paths covered by integration; download/delete branches gap |
| 43.5 | 872 | `backend_service/image_runtime.py` | Most coverage in pipeline-loader; engine-specific branches gap |
| 43.6 | 1,719 | `backend_service/inference.py` | RuntimeController happy paths covered; vLLM/turbo-binary branches gap |

### Highest-coverage modules (>50 lines)

| % | Lines | Path |
|---|---|---|
| 100.0 | 256 | `backend_service/models/__init__.py` |
| 100.0 | 91 | `backend_service/progress.py` |
| 98.9 | 91 | `cache_compression/teacache.py` |
| 98.5 | 67 | `backend_service/helpers/chat_template.py` |
| 97.9 | 96 | `backend_service/mlx_video_wan_convert.py` |
| 96.1 | 51 | `backend_service/helpers/cache.py` |
| 93.6 | 94 | `backend_service/catalog/capabilities.py` |
| 93.3 | 60 | `backend_service/runtime_paths.py` |

## TypeScript / React

```
vitest run --coverage
```

| Metric | Value |
|---|---|
| Test files | 28 (`src/`-scoped after vitest config fix) |
| Tests passed | 335 |
| Statement coverage | **58.7%** (1,249 / 2,126) of imported source |
| Branch coverage | 54.0% (1,291 / 2,392) |
| Function coverage | 48.3% (260 / 538) |
| Line coverage | 60.2% (1,102 / 1,830) of imported source |
| Wall time | ~700ms (post-scope-fix) |

**Caveat:** `vitest --coverage` only instruments files that get imported. Most `src/components/` and `src/features/` files never reach a test, so the percentages above understate the gap. Cross-referenced: only **28 of 80 components+features (35%)** have a sibling `.test.tsx`.

### Untested top-level features (no sibling test file)

- `src/features/htmlchallenge/` — 2,535 LOC (HtmlChallengeTab + supporting components)
- `src/features/video/VideoStudioTab.tsx` — 1,796 LOC
- `src/features/images/ImageStudioTab.tsx` — 1,178 LOC
- `src/features/compare/CompareView.tsx` — 766 LOC
- `src/features/settings/SettingsPanel.tsx` — covered (existing `SettingsPanel.test.tsx`)
- 40 of 42 `src/components/` files

### Lowest-coverage imported files (>50 lines)

| % lines | Path |
|---|---|
| 0 | `src/utils/keyboard.ts` |
| 0 | `src/utils/runtime.ts` |
| 0 | `src/components/RichMarkdown.tsx` |
| 7.7 | `src/components/CodeBlock.tsx` |
| 18.5 | `src/components/ReasoningPanel.tsx` |
| 33.2 | `src/api.ts` (1,430 LOC, 96+ exports) |
| 36.0 | `src/features/chat/runtimeDetails.ts` |
| 47.6 | `src/features/models/MyModelsTab.tsx` |

## tsc

```
npx tsc --noEmit
```

Clean — zero errors.

## Coverage gate

CI fails any PR that drops Python line coverage below **60%** or TypeScript imported-line coverage below **58%** without an attached justification commit message line `coverage-justification: <reason>`.

Phase-by-phase targets:

| Phase | Python % | TS lines % | TS files w/ tests % |
|---|---|---|---|
| 0 (this baseline) | 61.7 | 60.2 | 35 |
| Phase 0 PR target | ≥62 | ≥61 | ≥45 |
| Phase 6 target (final) | ≥70 | ≥70 | ≥60 |

## How to re-capture

```bash
# Python
.venv/bin/python -m pytest tests/ \
  --cov=backend_service --cov=cache_compression --cov=dflash \
  --cov-report=term --cov-report=json:coverage.json

# TypeScript
npm test -- --run --coverage
```
