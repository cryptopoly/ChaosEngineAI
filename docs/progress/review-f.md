# Review stream F: podcast demo path

Branch `review/f` from `b31c4ea5`. Brief: `docs/plans/overnight-e2e-review.md` (stream F). Output: `docs/plans/demo-path.md`.

## Setup

- Backend on 127.0.0.1:8905, Vite on 127.0.0.1:8925, throwaway `HOME` in the session scratchpad (`review-f-home`).
- Real model library linked, not copied: `~/.cache/huggingface` and the runtime venvs and binaries under `~/.chaosengine` are symlinks. Data files (settings, chats, voices, storyboards) start empty.
- `HF_HUB_OFFLINE=1` so nothing downloads by accident.

## Log (newest last)

- 07:38 worktree created, backend and Vite up (health 200). PASS
- 07:40 first launch (clean data dir, cleared webview storage): terms dialog at about 2 s, I agree takes about 3 s to close (settings save), Dashboard. PASS
- 07:41 FAIL fixed: Chat model picker gave `Qwen3-0.6B-GGUF` the size (39.8B) and badges of the Flint catalogue row (format and quant alone counted as a match), then 27.0B from Qwen3.8-27B (same family, other scale). Commit 5c8d206d, vitest in `src/utils/__tests__/library.test.ts`.
- 07:41 FAIL fixed: empty-chat "Load Model" went to the 131-row Chat Models catalogue; it now opens the Select Model picker (same commit). Placeholder copy "then hit CHAT" replaced.
- 07:42 FAIL fixed: Escape did not close Select Model (checked in the browser after the fix; no DOM test environment in vitest).
- 07:44 Server page: "Connect your app" is open and its code samples print the API key in clear text next to a masked key field; it also names a desktop app in "OLLAMA-COMPATIBLE APPS". Server files belong to stream E: reported, not edited.
- 07:45 The app has no light theme (dark only, ignores the OS setting). UI scale tops out at 100 %: added 110 % and 125 % (a0d0fc5f, `src/hooks/__tests__/useUiScale.test.ts`). 125 % at 1280x720 leaves Chat no room for messages; fine at 1920x1080.
- 07:46 Sidebar accordion moves rows under the pointer when a group opens; demo prep turns on "Keep all sidebar groups open".
- 07:48 vitest 1817 pass, i18n OK. Waiting for the model lock (stream A holds it).
- 07:51-08:04 R1 (1280x720, 90 %, LFM2.5 24B-A2B, 8K, lock held). Load under 18 s; answer 2.7 s at 114 tok/s. FAIL tools: 4 calculator cards, 2 failed ("raw", "unreadable"); fixed the calculator side (48604a24). FAIL Studio: first plan asked for frames < 8 (refused), Edit by chat gave 3 s, Approve then refused "LTX-2.5 is not an installed video model"; fixed (933bcfb4); second Approve rendered in 1 min 48 s. Voice 12 to 16 s, result below the fold (fixed, f83f2960). Server: API key in clear text in Connect your app and in Test. Keep going: model looped on list_models, stopped at 90 s. Chat thread did not follow tool cards (fixed later, see 08:20).
- 08:05-08:11 R2 (1920x1080, 125 %, Qwen3.8 27B 4-bit, 32K, fresh data dir). Terms 1 s, agree 1.3 s, load 8.6 s, answer 2.3 s at 29.6 tok/s, calculator prompt 6.9 s with one call. Studio plan at 34 s (15 s prompt processing), render 3 min 36 s at 1280x704 (model chose 720p). FAIL: Speak started during the render waited 3 min with a still bar and no word (fixed 7c17bc37: `GET /api/voice/speak/queue` and a waiting line). Sample film opens in about 2 s and plays. Connect your app collapse is not remembered (stream E's file). Lock released 08:11 after unloading and stopping my voice worker.
- 08:14 the chat thread now follows any growth while it is following (MutationObserver), commit 204abda0. vitest 1817 pass, tsc clean.
- 08:16 demo-path.md written from R1 and R2; R3 (the final script) waits for the model lock.
