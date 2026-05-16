# Coding guidelines

These rules came out of the v0.7.6 → v0.8.0 refactor + audit. Apply them to
every PR that touches a backend module > 500 LOC, a hook > 400 LOC, or any
file that mutates worker subprocess / file-system / network state. Skip on
trivial typo fixes, doc-only edits, and one-line bug patches.

The full canonical version lives in
[`CLAUDE.md`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/CLAUDE.md);
this page is a human-readable mirror of the key points.

## Editorial rules

ChaosEngineAI is a standalone product. **Do not reference external desktop
AI apps** in code, comments, UI strings, docs, or commit messages. This
includes — but is not limited to — names of competing desktop AI runners,
chat apps, or model managers that share underlying weights or workflows.

Allowed exceptions:

- **Model names from upstream providers** (e.g. *"Stable Diffusion 3.5
  Medium"*, *"FLUX.1-schnell"*, *"Wan 2.1"*) — these are model identifiers
  shipped by Stability AI / Black Forest Labs / Alibaba, not apps.
- **Hugging Face organization namespaces** (e.g. `lmstudio-community/...`,
  `mlx-community/...`) — these are repo namespaces on HF, not promotion
  of any app.
- **Open-source dependencies we vendor or shell out to** (e.g.
  `stable-diffusion.cpp`, `llama.cpp`, `mlx-video`) — these are named
  libraries we ship as runtime components.

When describing reference defaults or upstream behaviour, name the **model
author** (e.g. "Lightricks reference defaults", "Wan-AI model card") rather
than third-party tools that expose them.

## Performance

- **Lazy-import heavy deps.** `torch`, `diffusers`, `mlx`, `mlx_lm`,
  `mlx_vlm`, `transformers`, `nunchaku`, `bitsandbytes`, `huggingface_hub`,
  `gguf` are all multi-second imports. Put them inside the function that
  needs them, not at module top, unless the file is *only* loaded when
  inference is about to run. Backend startup target:
  `python -X importtime backend_service.app` < 2 s.
- **Process isolation for memory hogs.** Models > 1 GB stay in subprocess
  workers (MLX worker, sd-cli, longlive engine). Never load them in the
  FastAPI parent — a stuck pipeline takes the whole backend down with it.
- **Always release before reload.** Before swapping models, call the
  engine's `unload_model()` (or equivalent) so the OS reclaims the RAM
  before the next snapshot is mapped in. Two 47 GB workers = a 96 GB RAM
  exhaustion bug, not a feature.
- **No re-render thunder in React.** New object literals in `useMemo` /
  `useState` initialisers without a stable dep array are silent
  performance killers. Run the React Profiler on any tab > 200 LOC of
  state before shipping.
- **Profile before optimizing.** Don't rewrite a hot path on intuition —
  capture a number first (`scripts/perf-baseline.py`, the React Profiler,
  `python -X importtime`), then validate the win against
  `PERF_BASELINE.md`'s ±5% gate.

## Security

- **Treat user-controlled paths as hostile.** Anything that comes from a
  request body, a settings file, an env var, or a Hub catalog entry must
  go through `pathlib.Path` + `.resolve()` + a parent-prefix check before
  being passed to `open()` / `subprocess.run` / `shutil.copy`. Never
  `os.path.join` a user string into a system path.
- **List-form subprocess only.** `subprocess.run([bin, *args])` — never
  the shell-string form. No `shell=True`. Quote nothing — let
  `subprocess` do the escaping.
- **No secrets in source.** No HF tokens, no API keys, no bearer tokens,
  no signed URLs in `*.py`, `*.ts`, `*.rs`, `*.toml`, or `*.md`. Use the
  Settings store + secure-storage at runtime; CI builds get keys from
  GitHub secrets, not commits.
- **Validate at the boundary, trust internally.** Pydantic models on the
  FastAPI request edge + `serde` on the Tauri IPC edge + Zod on the
  frontend fetch wrapper. Once the value is past the boundary, internal
  helpers don't need defensive `isinstance` re-checks.
- **GGUF / safetensors are user data.** They can be malicious archives on
  a snapshot a user pasted in. Always load with `local_files_only=True`
  when probing; surface gated / 404 errors as user-readable messages.

## Modularisation

- **File-size soft caps.** Backend modules > 600 LOC, hooks > 400 LOC,
  components > 500 LOC, Rust modules > 800 LOC are a refactor signal —
  not an automatic block, but a prompt for the next change to extract
  before adding. The v0.8.0 pattern: pull a coherent subset into a
  sibling module, leave thin wrappers in the original site, re-export so
  test mock paths and existing imports don't break.
- **Single-purpose modules.** A file's docstring should fit in one
  paragraph. If you can't summarise what it does without "and also...",
  split it. Bundle by *responsibility*, not by *type*.
- **Re-exports preserve call sites.** When extracting from a module that
  has external callers, re-export the moved symbol from the original
  module path. Tests that patch `module._private` keep patching; imports
  in other packages keep working; the diff stays surgical.
- **No premature abstraction.** Three similar lines is fine. Don't create
  a `BaseEngine` / `Strategy` / `Plugin` interface for two callers — wait
  until there are five. Half-finished abstractions cost more than copies.
- **Cross-platform from the first line.** `pathlib.Path` (Python),
  `PathBuf` (Rust), `path.posix` vs `path.win32` (Node). Never hardcode
  `/tmp`, `~/.cache`, or `\\` — use the platform-aware primitive.

## When to refactor vs ship

- **Bug fix** → ship the surgical patch, leave the surrounding module
  alone.
- **Feature add** → if the target file is already over the soft cap, do
  an extract pass before adding. Otherwise add inline.
- **Refactor pass** → bundle multiple extracts in a single PR with a
  clear phase number (see `REFACTOR_PLAN.md` for the v0.8.0 template).

## See also

- [Adding a feature](adding-a-feature.md) — the E2E coverage gate.
- [Pre-build check](../testing/pre-build-check.md) — what blocks a release.
- The canonical [`CLAUDE.md`](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/CLAUDE.md)
  — full architecture overview, build checklist, and follow-ups tracker.
