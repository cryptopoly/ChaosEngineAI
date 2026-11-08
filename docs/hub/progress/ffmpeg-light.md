# ffmpeg light path (D-49): progress

Branch `feat/ffmpeg-light` (from `feat/home-studio`). Plan: `docs/plans/ffmpeg-download.md`. D-49: no own ffmpeg build, no hosting.

## Shipped

- One resolver `backend_service/storyboard/ffmpeg_bin.py`: `CHAOSENGINE_FFMPEG` -> a file chosen in Setup
  (`<dir>/custom`) -> app download (`~/.chaosengine/bin/ffmpeg/<ver>/`, `current` pointer) -> imageio-ffmpeg ->
  `PATH` only when the caller asks (`output_metadata` tagging). `FfmpegMissing` (a RuntimeError) carries
  `action = "install-ffmpeg"`; storyboard routes answer 409 `{code, message}`. `media.ffmpeg_exe`,
  `output_metadata._ffmpeg_exe` and the `video_runtime/pipeline_helpers.py` fallback writer use it.
  ltx-2-mlx is untouched (own venv, own libx264 ffmpeg).
- Encoder chooser `backend_service/storyboard/encoders.py` + `ffmpeg_bin.h264_encoder()`: libx264 when listed,
  else h264_videotoolbox (mac) / h264_nvenc, h264_amf, h264_qsv (win) / h264_nvenc, h264_qsv (linux) /
  libopenh264; hardware gets a 1-frame trial encode (cached per binary). `CHAOSENGINE_VIDEO_ENCODER` overrides.
  VAAPI and h264_mf are not offered. `assemble._codec_args(plan)` uses it.
- Optional pinned BtbN LGPL download (Windows and Linux), `backend_service/ffmpeg_fetch/`: pins verified on
  2026-11-07 against release `autobuild-2026-10-31-12-59` (asset list and `checksums.sha256`); SHA-256 checked
  before extract, only `bin/ffmpeg*` members, no links or `..`, validation runs only after the hash check
  (LGPL, no `--enable-gpl`, Storyboard filters), atomic rename plus `current` pointer, `.partial-*` cleanup.
  Switch: `pins.PINS_VERIFIED`. macOS has no download row (imageio-ffmpeg there).
- Routes `backend_service/routes/setup/ffmpeg.py`: GET `/api/setup/ffmpeg`, POST `/install`, GET `/install/status`,
  POST `/install/cancel`, POST `/custom` (an ffmpeg file, or the pinned archive fetched elsewhere), DELETE.
- UI `src/features/videotools/`: first-use prompt in the Mixer (one Install button, failure text, offline file
  field), Setup row in Settings > Accelerators. The install button uses `installPipPackage("imageio-ffmpeg")`.
- `THIRD_PARTY_NOTICES.md` FFmpeg entry rewritten. E2E `scripts/e2e_phases/phase_18.py`.

## Not done / notes

- Music export does not call ffmpeg today (tracks are written as FLAC/WAV by the worker), so no prompt there.
- The archive bytes of the pinned files were not re-hashed here; the pins come from BtbN's checksum file.
- Real Windows/Linux download and run: not tested. `scripts/audit-catalog-repos.py` HEAD check for pins: not added.
- Removing imageio-ffmpeg from the Setup list (plan step 9) is not part of this branch.
