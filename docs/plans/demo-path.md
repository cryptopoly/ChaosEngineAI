# Podcast demo path (v0.10.0, Mac)

Status: **draft, rehearsal pending.** Stream F of the overnight review (`docs/plans/overnight-e2e-review.md`). Progress and evidence: `docs/progress/review-f.md`.

A five to eight minute live path on the M4 Max (64 GB) that shows ChaosEngineAI at its best. Every step has the exact clicks, what to say, the time it took in rehearsal and a fallback.

## Before going on air (10 minutes, off camera)

Do these in order. None of them is part of the show.

1. **Screen.** Share one window, not the desktop. Window size 1920x1080 with **Settings > Appearance > UI scale 125 %**, or 1440x900 at 100 %. At 1280x720 keep 90 % and collapse the Chats list (the `<` button next to New thread): the thread area is too short otherwise.
2. **Sidebar.** Settings > Appearance > turn on **Keep all sidebar groups open**. With it off, opening one group closes the other and every item below moves, so a click lands on the wrong row.
3. **Close other apps** that hold memory, quit any running model. Activity Monitor memory pressure green. The demo peaks at about 35 GB (chat model plus the video model).
4. **Warm the disk cache**: load the chat model once and unload it, run one Video Studio clip, speak one line in Voice Studio. A cold first load after a reboot is two to three times slower.
5. **Nothing personal on screen.** Do not open Settings > Data Directory, Server > Connect your app (its code samples print the API key in clear text), Logs, or Diagnostics on air. The paths there start with your home folder.
6. **Partial download.** Chat > Models shows "Downloading 1" from a paused partial download (Nemotron 3.5 Bolt E4B OptiQ, 0.1 of 17.8 GB). Cancel it or leave Chat > Models off the path.
7. Turn notifications off (Focus mode).

## The path

| # | Step | Target time |
|---|---|---|
| 1 | Launch and first screen | 0:30 |
| 2 | Pick a model already on disk and load it | 0:45 |
| 3 | Chat: a fast answer, then a tool call | 1:00 |
| 4 | Studio chat: a short film clip from one sentence | 2:00 (render runs while you talk) |
| 5 | Voice: speak a line | 0:45 |
| 6 | Server page | 0:45 |
| 7 | Wow: the sample film, then Keep going | 1:30 |

Details, timings and fallbacks: filled in from the rehearsals below.
