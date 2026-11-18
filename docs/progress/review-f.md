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
