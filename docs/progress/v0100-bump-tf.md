# v0.10.0: TensorFold native pin 1.2.7 -> 1.2.8 (FU-168 follow-up)

Branch `v0100/bump-tf`, worktree `/Users/dan/ChaosEngineAI-v0100-bump-tf`. Not pushed, not merged.

## What 1.2.8 changed (for us)

Source: `ashhart/TensorFold` cloned read-only into the scratchpad (`v1.2.7..v1.2.8`, 401 commits), the release notes, and the macOS arm64 1.2.8 archive run from the scratchpad (the installed 1.2.7 under `~/.chaosengine` was not touched).

- Archives: same names (`tensorfold-1.2.8-{linux-x86_64,linux-aarch64,macos-arm64}.tar.gz` + `.sha256`), still no Windows archive. New extra asset `tensorfold-engine-1.2.8-darwin-arm64-metal.tar.gz`, not used. macOS sha256 `7b7977d7...0df2` matches its `.sha256` file. The app pins no digest; it reads the `.sha256` file.
- `capabilities --json`: same `families` / `serve.flags` shape; new top-level keys `decisions`, `env`, `optional_libraries`, `profiles` (the parser ignores them).
- Metal families: one new, `flint` (`mlx-q4g64`). Nothing removed.
- Serve flags: added `--max-connections`, `--max-connections-per-address`, `--trusted-proxy`; none removed; every flag the engine passes is still listed.
- CUDA families (from `zig/src/native/cuda_families.zig`): new `qwen3` (embedding / reranker, bf16) and `dots_ocr` (bf16, text only).
- Gemma 4 26B-A4B: still refused (`GemmaMoeNotServed` on `enable_moe_block`); `native_refuses` stays.
- D-77 rows: `deepseek_v4` and `prism_hadamard_qwen35` still not listed. 1.2.8's experimental Metal DeepSeek is `deepseek_v41`, FP8 only, behind `TF_EXPERIMENTAL`, and not listed even with it set. All three stay on the 0.6.6 Python install.
- LFM3-VL: same refusal. LFM2.5 `info` memory floor unchanged (14.65 GiB).

## Done

- `NATIVE_VERSION = "1.2.8"` (`backend_service/inference/_tensorfold_native.py`), comment in `routes/setup/tensorfold_native.py`.
- `_tensorfold.py`: new rows `flint` (listed, not offered, see below), `qwen3_embed_rerank` and `dots_ocr` (CUDA only, listed, not offered); comments re-checked at 1.2.8.
- Fixtures: `tests/fixtures/tensorfold_caps_macos_1_2_8.json` is the verbatim macOS 1.2.8 output (M4 Max); `tensorfold_caps_cuda_1_2_8.json` is the old assembled CUDA fixture plus `qwen3`, `dots_ocr` and the three flags.
- Tests updated (fixture names, version strings, the served-type list, Flint and the two CUDA rows listed and not offered).
- `scripts/check-upstreams.py` hardcodes nothing; with the scratch 1.2.8 binary it reports TensorFold current and no unlisted family.
- CLAUDE.md row, FU-168 row, hub rebuilt.

## Why Flint is not offered

Flint always reasons and the server answers 400 to `chat_template_kwargs.enable_thinking: false`. The engine (`_tensorfold_wire.build_payload`) sends that whenever thinking is off, including every non-streaming turn. Offering it needs: leave the flag out for always-reasoning families, then `tested_repos=("RecursiveSI/flint-int4",)`, `exclusive=True`, a catalog row, an E2E family entry, and a real load (22.6 GB; upstream checked M5 only).

## Tests

See the final report for the run.

## Real loads (M4 Max, after the release gate's DONE)

In process (`RuntimeController`, worktree code) with `CHAOSENGINE_TENSORFOLD_NATIVE` pointing at the scratch 1.2.8 binary, speculative decoding on, one load at a time, unloaded after:

- LFM3-1.7B MLX 4-bit: engine `tensorfold-native`, loaded in 6.2 s, correct reply, stop, 248 tok/s.
- LFM2.5-24B-A2B MLX 4-bit: engine `tensorfold-native`, loaded in 8.2 s, correct reply, stop, 187 tok/s.

## Not tested

- No full backend or E2E run against 1.2.8; no `native=False` model loaded (none on disk).
- The 1.2.8 install route end to end (download, sha256, unpack into `~/.chaosengine`): not run, to leave the installed 1.2.7 alone. The archive layout and digest were checked by hand.
- CUDA / Windows (4090 items): the CUDA fixture is still assembled, not a verbatim dump; the new CUDA families are unrun.
- Flint never loaded (not on disk, not offered).
