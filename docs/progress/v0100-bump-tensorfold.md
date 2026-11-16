# v0.10.0 S-R2: TensorFold native pin 1.2.8 -> 1.2.9

Branch `v0100/bump-tensorfold`, worktree `/Users/dan/ChaosEngineAI-v0100-bump-tensorfold`, from `origin/release/v0.10.0`. Pushed, not merged.

## What 1.2.9 changed (for us)

Sources: the v1.2.9 release notes, `git diff v1.2.8 v1.2.9` of `ashhart/TensorFold` (127 commits, cloned read-only into the scratchpad), and the macOS arm64 1.2.9 archive run from the scratchpad (the installed binary under `~/.chaosengine` was not touched).

- Licence: `LICENSE` unchanged, Apache-2.0.
- Archives: same names (`tensorfold-1.2.9-{linux-x86_64,linux-aarch64,macos-arm64}.tar.gz` + `.sha256`, now also a `.sha256.sig`), still no Windows archive. macOS sha256 `00600160...bbbc` matches its `.sha256` file.
- `capabilities --json` (macOS, M4 Max): families identical to 1.2.8 (same 21 model types, same formats). One serve flag added, `--config` (read serve flags from a TOML file); none removed. New top-level key `token_stream_versions` (ignored by the parser).
- CUDA families: `zig/src/native/cuda_families.zig` unchanged. CUDA now has a `kernelSupport` query and the release server refuses a GPU it has no kernels for, by name, at startup; our gate already reads the binary's own verdict.
- No source change to Gemma, LFM3 or Flint files: Gemma 4 26B-A4B (`native_refuses`), LFM3-VL and Flint (refuses thinking off) keep their status. D-77 rows (`deepseek_v4`, `prism_hadamard_qwen35`) still not listed; they stay on the 0.6.6 Python install.
- Behaviour changes that touch us: `--reasoning-effort` no longer changes a thinking-off prompt (the engine sends `reasoning_effort` only when thinking is on, and never passes the flag), video to a non-video model is a 400 not a 500. New Ollama-style `/api/*` routes are unused.
- The binary prints a stderr warning when Homebrew's `libturbojpeg` is not 3.1.4.1 (JPEG images refused until it is). `capabilities --json` stdout is unaffected.

## Done

- `NATIVE_VERSION = "1.2.9"`, comments in `_tensorfold_native.py`, `routes/setup/tensorfold_native.py`, `_tensorfold.py` (re-checked at 1.2.9; no row added or changed).
- Fixture `tests/fixtures/tensorfold_caps_macos_1_2_9.json` is the verbatim macOS 1.2.9 output (replaces the 1.2.8 one). The assembled CUDA fixture stays `..._cuda_1_2_8.json` (no CUDA dump possible here; its families did not change).
- Tests: fixture name, asset name, the `unlisted_families` version now comes from the fixture.
- CLAUDE.md TensorFold 1.x row. THIRD_PARTY_NOTICES.md names no version, so unchanged.
- `check-upstreams.py` with the scratch binary: TensorFold `ok 1.2.9`, no unlisted family.

## Tests

- `pytest tests/test_tensorfold_registry.py tests/test_tensorfold_routing.py tests/test_tensorfold_setup.py tests/test_tensorfold_engine_integration.py tests/test_tensorfold_native.py tests/test_tensorfold_family_report.py tests/test_check_upstreams.py`: 233 passed, 2 skipped.

## Real serve (M4 Max)

The scratch 1.2.9 binary run directly with the engine's serve flags (`--no-update-check --snapshot-dir none --context 4096 --parallel 1`), one model at a time, one non-streaming chat turn with thinking off, stopped after:

- LFM3-1.7B MLX 4-bit: ready in 3.5 s, reply "Paris", finish `stop`.
- LFM2.5-24B-A2B MLX 4-bit: ready in 13.6 s, reply "Paris", finish `stop`.
- LFM3-VL-2B MLX 4-bit: refused at start, "LFM3-VL inference requires the stacked vision contribution" (as in 1.2.7 and 1.2.8; row stays not offered).

Not run through the app's engine or `RuntimeController`; the serve argv matches `build_native_serve_command` without a drafter.

## Not tested

- CUDA (Linux, Windows 4090): no CUDA binary run; the CUDA fixture is still assembled. Windows has no archive, so the source-build adopt path is unchanged and unrun.
- Linux archives: names and checksum files checked in the release listing only.
- The install route end to end (download, sha256, unpack into `~/.chaosengine`): not run, to leave the installed binary alone.
- No full backend or E2E run; no `native=False` model loaded.
