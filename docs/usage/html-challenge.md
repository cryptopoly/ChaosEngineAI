# HTML Challenge

HTML Challenge is a structured Compare mode for HTML / web-page generation.
You pick 2 to 4 model slots, each with its own thinking-mode, reasoning
effort, and seed, then issue one prompt — every slot streams its HTML reply
in parallel with a sandboxed live preview rendered below the raw text.

## Use cases

- "Does this 27B model handle a 'build me a calculator app in a single
  file' prompt better than this 14B?"
- "Does flipping `reasoning_effort: high` on Qwen3.6 produce cleaner
  Tailwind than `low`?"
- "Does this MTPLX-optimized variant produce *identical* HTML to the base
  model, just faster?"

## Configuring slots

Each slot is an independent runtime profile:

- Model ref (anything in your library)
- Thinking mode (`off` / `auto`)
- Reasoning effort (`low` / `medium` / `high`)
- Seed (per-slot; lets you re-run a slot deterministically)

The shared prompt panel sits above. Hit Run and every slot streams in
parallel.

## Live preview + validation

Each slot has two views:

- **Code view** — the raw HTML stream as the model emits it.
- **Preview view** — a sandboxed `<iframe>` that renders the partial HTML
  as it arrives. Script errors, blank renders, and CSS issues surface
  inside the preview pane.

When the stream finishes, the validator classifies the result:

| Status | Meaning |
|---|---|
| `valid` | Parses cleanly + renders without console errors. |
| `partial` | Cuts off mid-document (the model stopped before closing tags). |
| `script-error` | Renders, but the sandbox console reported a JS exception. |
| `blank-render` | HTML present but the body renders empty (broken layout). |
| `no-html` | The model didn't emit HTML — usually a refusal or wrong format. |

The validator fires both from the iframe sandbox (post-render) and from a
server-side HTML parse, so script crashes and blank-renders surface
immediately, not just structural issues.

## Per-slot retry + repair

Failed slots get two recovery actions:

- **Retry** — re-runs the slot with the same settings (good for transient
  failures or to roll a different seed).
- **Continue / Repair** — sends the partial output back to the model with a
  "continue from where you left off" or "fix the broken HTML you produced"
  prompt, depending on the validator's verdict. The repair pass uses the
  same slot's runtime profile.

## Persistent history

Every run is saved with a title, the prompt, every slot's manifest, every
slot's HTML on disk, and the validator's verdict. Re-open earlier runs from
the history rail, delete runs you don't want, or open a slot's HTML in your
file manager / system editor with one click.

The full history is served from `GET /api/chat/html-challenges` — see
[API reference](../reference/api.md) for the endpoints.

## CLI

```bash
# List existing runs
./scripts/chaosengine-cli challenges-list

# Inspect a specific run
./scripts/chaosengine-cli challenges-get <challenge_id>

# Fetch a slot's HTML
./scripts/chaosengine-cli challenges-file <challenge_id> <slot_id>

# Retry a failed slot
./scripts/chaosengine-cli challenges-retry <challenge_id> <slot_id>

# Re-validate a slot
./scripts/chaosengine-cli challenges-validate <challenge_id> <slot_id> \
    --status valid
```

A `--smoke` E2E pass against HTML Challenge runs in Phase 3 of the suite — see
[E2E testing](../testing/e2e-testing.md) for the contract.
