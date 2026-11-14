# v0.10.0: bump llama.cpp b12387 and Playwright 1.65.1

Branch `v0100/bump-llama`, commits a9139306 (llama.cpp) and ba6aa7bc (Playwright).

## Done
- llama.cpp b12317 -> b12387 (newest release with macos-arm64 and win-cuda assets). Asset names unchanged. New Windows CUDA digests in `_PINNED_SHA256` taken from the release asset `digest` fields (12.4 x64, 13.5 x64, 13.5 arm64); cudart zips are tagless and unchanged. Also bumped `linux-gpu-check.sh` and three tests that name the tag. CLAUDE.md row updated.
- Playwright 1.65.0 -> 1.65.1 exact in `e2e/package.json` and its lockfile; chromium headless shell installed. CLAUDE.md row updated. `docs/research/playwright-security.md` left alone (dated research record naming 1.65.0 as latest on 2026-11-07).
- pytest tests/test_llama_cuda.py tests/test_llama_cuda_windows.py tests/test_linux_arm64_build.py: pass (some skips).

## Not done (blocked: release gate not DONE, /tmp/claude-501/r1/summary2.txt absent)
- Rebuild/install of llama-server into `~/.chaosengine/bin` (back up first as `llama-server.bak-20261114`), GGUF chat on the new binary, tests/test_inference.py.
- Playwright specs (01-smoke etc.) with --workers=1.
- Windows CUDA install on b12387 (4090 item).
