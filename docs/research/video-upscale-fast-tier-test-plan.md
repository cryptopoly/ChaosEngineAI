# Video upscale, fast tier: the real-GPU test on the RTX 4090

For the owner's session, when the GPU is free. The backend code (branch of 2026-11-25, `backend_service/video_upscale/`) has only been tested with a pixel-doubling fake: no Real-ESRGAN model has been loaded, no weight file downloaded, nothing timed. This run replaces the catalog's estimated memory numbers, pins the weight hashes and decides whether the fast tier ships. Background: `docs/research/ai-video-upscaler-feasibility-2026-11.md` (section 5).

## 0. Before

1. Check out the branch, then the unit tests (no GPU, a few minutes):
   `.venv\Scripts\python.exe -m pytest tests -q -k "upscale or storyboard or catalog or setup_routes or fit"`
2. Start the backend from this branch as usual (port 8876). Nothing else may use the GPU during the timed runs.
3. Make a scratch folder outside the repo, for example `%USERPROFILE%\upscale-test`, and in PowerShell:

```powershell
$FF = "E:\ChaosEngineAI\.venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"   # has libx264
$T  = "$env:USERPROFILE\upscale-test"; New-Item -ItemType Directory -Force $T | Out-Null; Set-Location $T
```

## 1. Clips

- `hd_ref.mp4`: 10 s of real 1080p footage with sound (a Video Gallery clip, or the owner's camera). Synthetic fallback, worse for judging a GAN:
  `& $FF -f lavfi -i "mandelbrot=s=1920x1080:r=25" -f lavfi -i "sine=f=440" -t 10 -c:v libx264 -crf 12 -pix_fmt yuv420p -c:a aac hd_ref.mp4`
- `lr_360p.mp4`, the round-trip source (bicubic down, compressed like a web upload, sound copied):
  `& $FF -i hd_ref.mp4 -vf "scale=640:360:flags=bicubic" -c:v libx264 -b:v 500k -pix_fmt yuv420p -c:a copy lr_360p.mp4`
- `tape_576i.mp4`, a tape stand-in for the deinterlace path (interlaced, noisy):
  `& $FF -i hd_ref.mp4 -vf "scale=720:576,noise=alls=18:allf=t,interlace" -c:v libx264 -flags +ildct+ilme -b:v 1200k -c:a copy tape_576i.mp4`
  (`interlace` and `noise` are only used to make the test clip; the app does not use them.)
- `lanczos_1080.mp4`, the baseline every model must beat on looks:
  `& $FF -i lr_360p.mp4 -vf "scale=1920:1080:flags=lanczos" -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a copy lanczos_1080.mp4`

## 2. Install the add-on (downloads about 72 MB, on approval)

```powershell
$tok = (Invoke-RestMethod http://127.0.0.1:8876/api/auth/session).apiToken
$H = @{ Authorization = "Bearer $tok" }
$api = "http://127.0.0.1:8876"
Invoke-RestMethod -Method Post "$api/api/setup/install-video-upscale" -Headers $H -ContentType application/json `
  -Body '{"models":["realesr-general-x4v3","realesrgan-x4plus"]}'
do { Start-Sleep 3; $s = Invoke-RestMethod "$api/api/setup/video-upscale/status" -Headers $H } while ($s.job.running)
$s.job; $s.models | Format-Table id, installed, sha256Seen
```

Expected: `job.phase` = `done`, both models `installed`. Then pin the hashes: check each `sha256Seen` against
`certutil -hashfile "$env:USERPROFILE\.chaosengine\addons\video-upscale\weights\<file>" SHA256`, put them in
`backend_service/catalog/video_upscale_models.py` (`sha256`, drop `sha256Todo`) and confirm
`.venv\Scripts\python.exe scripts\audit-catalog-repos.py` no longer reports `UNPINNED FILE`.

## 3. Timed runs

Run each job alone. In a second PowerShell window, log the whole card's memory every 0.5 s for the length of the job:
`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > vram-<name>.log`
(stop it with Ctrl+C when the job is done; peak = largest value minus the idle value before the start).

```powershell
function Up($body, $name) {
  $job = Invoke-RestMethod -Method Post "$api/api/video/upscale" -Headers $H -ContentType application/json -Body $body
  $t0 = Get-Date
  do { Start-Sleep 2; $j = Invoke-RestMethod "$api/api/video/upscale/jobs/$($job.id)" -Headers $H
       "{0} {1}/{2} {3} fps" -f $j.state, $j.framesDone, $j.framesTotal, $j.framesPerSecond } while ($j.state -in "queued","running")
  $sec = ((Get-Date) - $t0).TotalSeconds
  "$name : $($j.state) $($j.framesDone) frames in $([math]::Round($sec,1)) s = $([math]::Round($j.framesDone/$sec,2)) fps end to end; $($j.error)"
  $j
}
$in = (Resolve-Path lr_360p.mp4).Path -replace '\\','/'
$a = Up "{`"inputPath`":`"$in`",`"modelId`":`"realesr-general-x4v3`",`"width`":1920,`"height`":1080}" "general-x4v3 360p->1080p"
$b = Up "{`"inputPath`":`"$in`",`"modelId`":`"realesrgan-x4plus`",`"width`":1920,`"height`":1080}" "x4plus 360p->1080p"
$c = Up "{`"inputPath`":`"$in`",`"modelId`":`"realesr-general-x4v3`",`"scale`":4}" "general-x4v3 360p->1440p (4x, no lanczos)"
```

Outputs land in `<video outputs>\upscaled\lr_360p-<model>-<W>x<H>.mp4` (the job's `output`). Copy them into `$T`.
Also time `hd_ref.mp4` at `"width":3840,"height":2160` with the general row (1080p input: the big-frame case, untiled).

Expected (ESTIMATES from the feasibility report, to be replaced): general-x4v3 20 to 40 fps from 360p/480p input;
x4plus 1 to 3 fps; 1080p to 2160p with general-x4v3 8 to 15 fps. Peak VRAM: general under 2 GB, x4plus (512-pixel
tiles) under 5 GB. End-to-end fps also includes decode, the 2x lanczos step and libx264 at the output size.

## 4. Quality: the round trip

For each output (and `lanczos_1080.mp4` as the floor):

```powershell
& $FF -i <out>.mp4 -i hd_ref.mp4 -lavfi "[0:v][1:v]psnr=stats_file=psnr-<name>.log" -f null - 2>&1 | Select-String "PSNR"
& $FF -i <out>.mp4 -i hd_ref.mp4 -lavfi "[0:v][1:v]ssim=stats_file=ssim-<name>.log" -f null - 2>&1 | Select-String "SSIM"
```

GAN rows often score lower PSNR than lanczos while looking better; read the numbers beside a look at frames side by side
(`& $FF -i <out>.mp4 -vf "select=eq(n\,100)" -frames:v 1 f100-<name>.png`). LPIPS (`lpips`, BSD-2, scratch venv only) is optional.

## 5. Function checks

- **Sound:** `& $FF -i <out>.mp4` lists the same audio stream as the source (copied, not re-encoded); play the first and last seconds: in sync.
- **Frame count and rate:** `& $FF -i <out>.mp4 -map 0:v -f null -` reports the same frame count as the source; the rate is the source's.
- **Cancel:** start the x4plus job, `Invoke-RestMethod -Method Post "$api/api/video/upscale/jobs/<id>/cancel" -Headers $H` after a minute: state `cancelled`, `resumable` true, no output file.
- **Resume:** `.../jobs/<id>/resume`: `resumedSegments` > 0 and the finished file has every frame. Repeat once with a backend restart between cancel and resume (state reads back `cancelled`, or `interrupted` if the backend was killed mid-job).
- **Queue:** start a video generation in the Video Studio, then an upscale: the upscale waits (`queued`) and starts after it.
- **Deinterlace:** `tape_576i.mp4` with `"deinterlace":true,"scale":2`: no combing on motion. First check `& $FF -hide_banner -filters | Select-String bwdif` on the app's pinned LGPL ffmpeg too (the job refuses with a message if it is missing).
- **Size cap:** a 1080p input at `"scale":4` is refused with "would be 7680x4320; the limit is 4096".
- **Storyboard:** a 720p project with captions burnt in, export with `upscale: "1440p"` and `upscaleMethod: "fast"` (no UI for the method yet: set it with the agent tool, `storyboard_project` action `edit` with `upscale_method: "fast"`); then `2160p`. Captions and cards sharp, film 2560x1440 / 3840x2160, sound in sync. A second export reuses the upscaled shots (fast; no new worker activity). Without the add-on the export is refused naming Setup.
- **10-minute clip:** memory flat over the run (`vram.log` and the worker's RSS in Task Manager).

## 6. Fill in

| Model | Input | Output | Frames | Seconds | fps end to end | Peak VRAM GB | PSNR dB | SSIM | Looks better than lanczos (y/n) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| lanczos (ffmpeg) | 640x360 | 1920x1080 | | | | n/a | | | n/a | |
| realesr-general-x4v3 | 640x360 | 1920x1080 | | | | | | | | |
| realesrgan-x4plus | 640x360 | 1920x1080 | | | | | | | | |
| realesr-general-x4v3 | 640x360 | 2560x1440 | | | | | n/a | n/a | | |
| realesr-general-x4v3 | 1920x1080 | 3840x2160 | | | | | n/a | n/a | | |
| realesr-general-x4v3 + deinterlace | 720x576i | 1440x1152 | | | | | n/a | n/a | | |

After the run: put the measured peaks in the catalog rows (`runtimeFootprintCudaGb`, `footprintBasis: "measured"`, `footprintMeasuredOn: "cuda"`), adjust `batch`/`tile` if the numbers say so, and close the matching items of FU-617.
