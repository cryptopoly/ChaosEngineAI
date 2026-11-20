# Security review of `test/rtx4090` (2026-11-20)

Scope: everything on `test/rtx4090` since `origin/release/v0.10.0` (`git diff --stat origin/release/v0.10.0...HEAD`,
394 files, about 99 backend and script files), read against the CLAUDE.md Security rules: user-controlled paths
(`Path` + `resolve()` + parent-prefix check), list-form subprocess only, no secrets in source, validation at the
boundary, GGUF and safetensors as user data, `local_files_only` when probing.

Method: read the code of each surface, then try to break it with a unit test, a live check or a throwaway backend
(`HOME` and `USERPROFILE` on `E:\tmp\secrev-home`, port 18931, its own API token; no GPU used, the Hugging Face token
never read or printed).

## Findings

| # | Surface | Verdict | Where |
|---|---|---|---|
| 1 | WSL vLLM bridge command line | **BUG, high: command injection. Fixed.** | `backend_service/inference/vllm_wsl_engine.py` `_build_wsl_command` |
| 2 | `make_lipsync_shot` agent tool, `audio` argument | **BUG, medium: any WAV or FLAC on the disk can be read by an agent. Fixed.** | `backend_service/mcp_bridge/lipsync_tool.py` |
| 3 | Lip-sync ffmpeg commands | **BUG, low (hardening): inputs not held to local files. Fixed.** | `backend_service/lipsync_runtime.py` `prepare_audio_command`, `mux_command` |
| 4 | POST /api/video/generate `audioPath`, `audioStartSeconds` | OK | `routes/video_lipsync.py`, `lipsync_runtime.resolve_audio` |
| 5 | Storyboard `vocalClip`, `set_vocal_clip` | OK | `storyboard/ops.py`, `storyboard_tools.py`, `generators.py` |
| 6 | `/api/music/weights/copy` routes | OK | `music_weight_copy.py`, `routes/music_library.py` |
| 7 | Music download resume | OK | `hf_resume_script.py`, `helpers/hf_cache_paths.prune_stale_partials` |
| 8 | Music, sound and voice installers | OK, residual risk (no hashes) | `music_env.py`, `music_sound_env.py`, `voice_env_asr.py`, `voice_env.py` |
| 9 | Speech-to-text GPU add-on, `asr-site`, MOSS-Image pipeline | OK | `voice_env_asr.py`, `voice_worker_asr.py`, `image_runtime/moss_image*.py`, `flux3_modular.py` |
| 10 | Downloads: allow patterns, companions, Nunchaku base repo | OK | `state/downloads.py`, `helpers/companions.py`, `routes/models.py`, `routes/images.py` |
| 11 | Setup pip installs, `_INSTALLABLE_PIP_PACKAGES` | OK (unchanged on this branch) | `routes/setup/` |
| 12 | sd.cpp, Stable Audio, ACE-Step, lip-sync, ASR worker command lines | OK | `sdcpp_*_runtime.py`, `music_worker*.py`, `lipsync_*.py`, `voice_client.py` |
| 13 | `scripts/build-sdcpp.ps1` | OK | NVIDIA archives sha256-checked |
| 14 | `licence_gate` | OK | text only on this branch; no new way to set the tick |
| 15 | Auth on new routes, CORS | OK | global middleware in `app.py` |
| 16 | HF token in logs or responses | OK | nothing new logs or returns a token |
| 17 | Windows job object, stdin hand-off in workers | OK | `windows_job.py`, `warm_worker_base.py`, `music_worker.py` |

### 1. WSL vLLM bridge: command injection (high, fixed)

The bridge started vLLM with `["wsl", "--", "bash", "-c", inner]`. `wsl --` does not exec bash: it hands the command
line to the distro's default shell, which parses it again with the inner string in double quotes. Inside double
quotes `$(...)`, backticks and `$VAR` expand, so the `shlex.quote` single quotes around `--model` and
`--served-model-name` protected nothing. Shown live on this PC (Ubuntu 26.04 under WSL):

```
wsl -- bash -c "echo 'x$(echo INJECTED_OUTER)y`echo BT`'"   ->  xINJECTED_OUTERyBT
wsl --exec bash -c "echo 'x$(echo INJECTED_OUTER)y`echo BT`'" ->  x$(echo INJECTED_OUTER)y`echo BT`
```

Reach: the model ref (sent as `--served-model-name`, new on this branch), the local path and the runtime target
(`--model`, older) all come from POST /api/models/load and /api/compare, so any caller holding the app token (the
UI, the CLI, a remote connection) could run a command in WSL, which can read and write the whole Windows profile
through `/mnt/c`. A model folder whose name holds `$` or a backtick broke the load even without intent. Agent-bridge
clients cannot load models, so they could not reach it.

Fix: `wsl --exec bash -c inner` (bash runs directly, one parse). Tests: `WslShellInjectionTests` in
`tests/test_vllm_wsl_engine.py` (the argv shape, and a live WSL check behind `CHAOSENGINE_TEST_WSL=1`, which passed
here); the old shape test now expects `--exec`.

Side finding, not security: under `wsl --` the `export PATH="~/.chaosengine/vllm-venv/bin:$PATH"` prefix was lost
in the double parse; under `--exec` it is kept but with a literal `~`, which bash's own lookup does not expand. The
`python` path itself is unquoted and expands. FU-533.

### 2. `make_lipsync_shot` reads any audio file on the disk (medium, fixed)

The tool passed `audio` straight to `lipsync_runtime.resolve_audio`, which accepts any existing `.wav` or `.flac`
path. The finished clip carries that audio as its sound (AAC 192k) and the job answer gives a `download`, so an agent
could copy any WAV or FLAC on the machine (a voice memo, a private recording). The Storyboard already refuses this
for agents (`_agent_source`: an `upload:` handle or a file inside the output folders, "or a later export would hand
it a copy of any media file on the machine"); `storyboard_shot` `vocal` goes through it, the new tool did not.

Fix: `audio` goes through `_agent_source` first; the schema text says so. Test:
`test_the_tool_refuses_audio_outside_the_output_folders` in `tests/test_lipsync.py` (and the existing tool test now
puts its WAV in a patched output folder). The HTTP route keeps any local path, as the other UI routes do.

Not changed: `start_image` in this tool, like `generate_video`'s and `generate_image`'s inputs, still takes any PNG,
JPEG or WebP path (`resolve_input_image`). That is older behaviour and the still becomes frame 0 of the clip. FU-530.

### 3. Lip-sync ffmpeg: no protocol whitelist (low, fixed)

ffmpeg chooses the demuxer from the content, not the name. Every Storyboard ffmpeg input runs with
`-protocol_whitelist file,pipe` (`storyboard/ffmpeg_bin.PROTOCOLS`); the two lip-sync commands did not. With the
bundled ffmpeg 7.1 a playlist named `.wav` is refused anyway ("Not detecting m3u8/hls with non standard extension",
tried here), but `find_ffmpeg(allow_path=True)` may pick an older system ffmpeg. Fix: both commands carry the
whitelist. Test: `test_ffmpeg_reads_local_files_only`. Arguments are list-form; the input path is resolved and
absolute, so it cannot start with `-`.

### 4 to 17: what was checked

- **`audioPath` on POST /api/video/generate**: `resolve_source` resolves strictly, requires a file of kind audio,
  0 < size <= the media cap; `resolve_audio` then requires `.wav` or `.flac`. The route answers 400 for a bad path
  before any model work. `audioStartSeconds` is `ge=0, le=3600` in Pydantic, `audioPath` `max_length=1024`. The
  work files live in a `TemporaryDirectory`; the saved clip goes through `_save_video_artifact` (the gallery's own
  naming). Throwaway backend: the route answers 401 without the token.
- **Storyboard vocal**: the agent path uses `_cast_source` (upload handle, output file, or a `media/` path the op
  keeps inside the project); the op copies the file into `media/<shot>/vocal-<stamp>` and refuses a non-WAV/FLAC
  suffix; `generators.py` reads it back through `media.inside(folder, ...)`; `vocalClip` is a `RelPath`. Bundles
  list it like `endImage`.
- **Weight copy**: source and target come from the catalog and the Hub cache (`mirror_variants`), never from the
  request (the POST body is ignored). The source is `resolve()`d (HF cache symlinks to blobs); writes go through
  `<name>.partial` and `os.replace`, size-checked; free space is checked with a margin of max(1 GiB, 5 %) before it
  starts. The three routes answer 401 without the token (throwaway backend).
- **Download resume** (`hf_resume_script.py`): runs inside the download subprocess on paths the library computes
  (`<blob>.incomplete`), `glob.escape`d; keeps the largest partial, cuts 32 MiB, range-requests the rest and checks
  SHA-256 when the blob name is one (every LFS or Xet file, which are the only files over the 32 MiB floor; smaller
  files restart from zero). `prune_stale_partials` deletes only `*.incomplete` under `<cache>/models--<repo>/blobs`;
  the repo id passes `_HF_REPO_PATTERN` and `/` becomes `--`, so `..` stays inside one folder name (tried: `../x` is
  passed to the Hub, which refuses it; `../../..` is 400).
- **Installers**: list-form `pip` commands with exact `==` pins from `uv export`; `torch==...+cu128` only exists on
  the PyTorch index. The ACE-Step and Stable Audio 3 engines install from GitHub archives pinned by full commit SHA,
  with `--no-deps`; pip unpacks them (no own extraction code, so no zip-slip surface here). No `--require-hashes`, and
  `--extra-index-url` lets pip pick the highest version across both indexes; with exact pins that only matters if
  the PyTorch index ever served a same-version package with other content. Residual risk, FU-531.
- **ASR add-on**: Transformers 5.23 goes to `<voice venv>/asr-site` with `--target --no-deps`; only the ASR worker
  gets `PYTHONPATH=<asr-site>`, set by the backend (not from a request). Every `from_pretrained` in
  `voice_worker_asr.py` has `local_files_only=True` and no `trust_remote_code`. No `trust_remote_code=True`,
  `torch.load` or `pickle.load` anywhere in `backend_service/`.
- **MOSS-Image**: `moss_image_pipeline.py` imports only torch, diffusers and transformers; no file, network or process
  calls; loads with `local_files_only=True` from the catalog snapshot. FLUX.3 modular loads `ModularPipeline` with
  `local_files_only=True` (custom blocks would still need `trust_remote_code`).
- **Downloads**: allow patterns for the GPU dictation rows and the Nunchaku base repo come from the catalog by repo id;
  deleting a Nunchaku row deletes the base repo only when it holds no transformer weights and is not downloading.
- **Setup**: `_INSTALLABLE_PIP_PACKAGES` and the setup pip routes are not changed on this branch;
  `ensure_extras_on_sys_path` is unchanged. The WSL setup steps (`_run_wsl_step`) still use `wsl --`, but every
  command there is a constant (no request data); moving them to `--exec` is in FU-533.
- **Worker command lines**: all list-form, no `shell=True` added on the branch. sd.cpp, Stable Audio 3, ACE-Step CUDA
  and the lip-sync worker get paths from the backend (catalog snapshots, temp folders), not from requests. The
  ltx-2-mlx `a2v` command passes the prompt as its own argv item.
- **build-sdcpp.ps1**: each NVIDIA redistributable archive is checked against the sha256 in NVIDIA's
  `redistrib_<version>.json` (same HTTPS host, so it guards against corruption, not a compromised host); unpacked
  with `tar.exe` (bsdtar refuses `..` and absolute names by default). Not called from the backend; no execution
  policy changes.
- **Licence gate**: only the notice text, a live licence link and `key_for_model_id` changed. The agent bridge
  checks `require_variant` and has no way to set the tick.
- **Auth and CORS**: every new route sits under `/api/` and goes through `require_api_auth` (token compared with
  `secrets.compare_digest`); `EXEMPT_AUTH_PATHS`, `AGENT_FILE_PATHS` and the CORS setup are unchanged. The Plugins
  routes were removed. `make_lipsync_shot` needs the `generate` scope and counts against the per-token job cap.
- **Tokens**: no new code logs, stores or returns a Hugging Face token; the throwaway backend's output held no `hf_`
  string.
- **Windows job object**: creates an unnamed kill-on-close job for the backend's own process; no handle is shared.

## Residual risks (FU rows)

- FU-530: agent `start_image` and `images` inputs (`generate_image`, `generate_video`, `make_lipsync_shot`) take any
  PNG, JPEG or WebP path; the Storyboard holds agents to uploads and outputs.
- FU-531: installers have no hash pinning (`--require-hashes`, or a sha256 for the two GitHub archives).
- FU-532: the WSL bridge always passes `--trust-remote-code`, so loading any Hub repo through it runs that repo's
  code inside WSL (older behaviour; the native vLLM lane should be checked too).
- FU-533: WSL bridge `PATH` prefix never took effect, and the setup steps still use `wsl --`.
- FU-534: the weight-copy notice and some music refusals carry local folder paths, which reach agents as tool
  errors (the bridge otherwise answers without local paths).

## Tests run

- `tests/test_vllm_wsl_engine.py` (with `CHAOSENGINE_TEST_WSL=1`, live check included), `tests/test_lipsync.py`,
  `tests/test_mcp_bridge.py`, `tests/test_storyboard_tools.py`, `tests/test_music_env.py`: all pass, `HOME` and
  `USERPROFILE` on `E:\tmp\secrev-home`.
- Throwaway backend on port 18931: 401 without the token for `/api/music/weights/copy`, `/copy/status` and
  `/api/video/generate` (also with a foreign `Origin`); 200 with it; `/api/models/download` with `../../..` is 400.

Not tested: a real lip-sync run, a real vLLM start through the bridge after the `--exec` change (needs the GPU),
the Mac `a2v` path.
