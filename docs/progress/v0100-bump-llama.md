# v0.10.0: bump llama.cpp b12387 -> b12441 (2026-11-16)

Branch `v0100/bump-llama` (off `release/v0.10.0`), commit 865cea3e (pin).

## Done
- Release notes b12388..b12441 read: no flag removed or renamed. `llama-server --help` of b12441 vs b12387 differs only in a docs path in the TTS help text. Cache types (f32, f16, bf16, f8_e4m3, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1), `--spec-type` list (incl. draft-dflash, draft-dflash2, draft-mtp, ngram-mod), `--load-mode`, `--model-draft`, `--spec-draft-n-max` unchanged. Notable upstream: Inkling 2, MiniMax-M3.1, DFlash2 NoPE handling (b12416), server slot scheduler refactor (b12411), streamed-response connection reuse fix (b12404), LFM3 Metal fusion workaround (b12407).
- Asset names identical to b12387 (CUDA 13 archives still `13.5`). New Windows CUDA llama digests from the release `digest` fields; the tagless cudart zips are unchanged.
- Pin moved in `scripts/update-llama-cpp.sh`, `scripts/stage-runtime.mjs`, `backend_service/helpers/llama_cuda.py` (tag + `_PINNED_SHA256`), `scripts/linux-gpu-check.sh`, three tests naming the tag, CLAUDE.md upstream row.

## Mac install
- `scripts/update-llama-cpp.sh` builds from a `../llama.cpp` checkout that does not exist on this Mac, so installed the official prebuilt `llama-b12441-bin-macos-arm64.tar.gz` (what `stage-runtime.mjs` ships); sha256 60cd8f23... matches the release digest.
- Backups: `~/.chaosengine/bin/llama-server.bak-20261116` (b12387 binary) and `~/.chaosengine/bin/llama-b12387.bak-20261116/` (b12387 binary + all its dylibs, needed because the binary is dylib-linked). Roll back: copy that folder's contents over `~/.chaosengine/bin`. Same dylib file set in both builds. Homebrew untouched; `llama-server-turbo` still runs.
- Installed `llama-server --version`: build 12441 (164ab018f).

## Measured (M-series Mac, Metal, -ngl 99, -fa on, 300 tokens, temp 0, seed 42; machine shared with other agents, load average 7-26, so single runs swing +-30 %)
- Nemotron-3.5-Bolt-E4B Q4_K_M, 7 interleaved passes of 3 runs: per-pass means old/new 40.1/32.9, 36.6/36.0, 33.3/32.6, 33.6/34.0, 25.8/29.3, 23.8/23.2, 23.0/31.8. No consistent difference; pass 1 favoured old, pass 7 new.
- Qwen3-0.6B Q4_K_M, 3 passes: old 98.3/70.5/74.7, new 96.1/102.8/83.0.
- DFlash (app args from `_speculative_args`: `--model-draft <Bolt DFlash BF16> --spec-type draft-dflash --spec-draft-n-max 15`): both builds accept 202 of 671 drafted tokens, identical text; new 52-59 tok/s, old 41-53 tok/s (under load).
- Installed b12441 with `-ctk q8_0 -ctv q8_0` on Qwen3-0.6B: chat completion OK, ~53 tok/s.
- App pre-validation (`_llama_server_cache_types`, `_llama_server_supports`) against the installed binary: all types and flags the app uses are found; `--no-mmap` absent as on b12387 (`--load-mode` path used).

## Tests
- pytest test_inference, test_cache_strategies, test_stage_runtime_scripts, test_check_upstreams, test_llama_cuda, test_llama_cuda_windows, test_llama_cuda_prune, test_linux_arm64_build: 239 run, 0 failed, 5 skipped.

## Not done / not tested
- Windows CUDA and Linux CUDA installs of b12441 (digests only checked against the release metadata).
- Chat through the full app engine path (`LlamaCppEngine` load via the backend); direct llama-server with the app's flags only.
- MTP (`draft-mtp`): no GGUF with MTP heads on disk.
- A quiet-machine tok/s comparison.

---

# v0.10.0: bump llama.cpp b12387 and Playwright 1.65.1

Branch `v0100/bump-llama`, commits a9139306 (llama.cpp) and ba6aa7bc (Playwright).

## Done
- llama.cpp b12317 -> b12387 (newest release with macos-arm64 and win-cuda assets). Asset names unchanged. New Windows CUDA digests in `_PINNED_SHA256` taken from the release asset `digest` fields (12.4 x64, 13.5 x64, 13.5 arm64); cudart zips are tagless and unchanged. Also bumped `linux-gpu-check.sh` and three tests that name the tag. CLAUDE.md row updated.
- Playwright 1.65.0 -> 1.65.1 exact in `e2e/package.json` and its lockfile; chromium headless shell installed. CLAUDE.md row updated. `docs/research/playwright-security.md` left alone (dated research record naming 1.65.0 as latest on 2026-11-07).
- pytest tests/test_llama_cuda.py tests/test_llama_cuda_windows.py tests/test_linux_arm64_build.py: pass (some skips).

## Mac checks (after the gate was DONE)
- tests/test_inference.py: pass (70 tests). Run on the code, not on a new binary.
- Playwright 1.65.1, `npx playwright test 01-smoke --workers=1`: 3 passed.

## llama-server b12387 installed on the Mac (owner approved the download)
- Prebuilt `llama-b12387-bin-macos-arm64.tar.gz` (official release), sha256 matched the release digest (a654e602...).
- The new build is dylib-linked (`@loader_path`), the old one was a single file. Installed `llama-server` plus the `libggml*`/`libllama*`/`libmtmd*` dylibs into `~/.chaosengine/bin`. Old binary kept as `llama-server.bak-20261114`; no dylibs existed there before. To roll back: copy the .bak over `llama-server` and delete the `*.dylib` files in that folder. `llama-server-turbo` still runs.
- Direct run, Llama-3.2-1B Q4_K_M (Metal, -ngl 99): `/v1/chat/completions` returned a reply, version `build 12387 (a956a852c)`. Not run through the app's engine path.

## Not done
- Windows CUDA install on b12387 (4090 item).
- Chat through the app's llama.cpp engine path (direct server only).
