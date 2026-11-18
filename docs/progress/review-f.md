# Review stream F: podcast demo path

Branch `review/f` from `b31c4ea5`. Brief: `docs/plans/overnight-e2e-review.md` (stream F). Output: `docs/plans/demo-path.md`.

## Setup

- Backend on 127.0.0.1:8905, Vite on 127.0.0.1:8925, throwaway `HOME` in the session scratchpad (`review-f-home`).
- Real model library linked, not copied: `~/.cache/huggingface` and the runtime venvs and binaries under `~/.chaosengine` are symlinks. Data files (settings, chats, voices, storyboards) start empty.
- `HF_HUB_OFFLINE=1` so nothing downloads by accident.

## Log (newest last)

- 07:38 worktree created, backend and Vite up (health 200). PASS
