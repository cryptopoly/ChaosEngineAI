# Chat

The Chat tab is the most-used surface in the app. It's a multi-thread
conversation workspace bound to whatever model is currently loaded in the
warm pool.

## Threads

Threads live in the left rail. Pin the ones you come back to often, drag to
reorder, and double-click a title to rename. Every thread persists across
launches and remembers the exact runtime profile that produced it:

- Model ref + engine (MLX / GGUF / vLLM / remote)
- Cache strategy + bits + fp16-layer count
- Context length
- Sampling presets (`temperature`, `top_p`, `top_k`, `min_p`,
  `repeat_penalty`, `seed`, `mirostat`, `reasoning_effort`)
- Speculative decoding state (off / DFlash / DDTree / MTPLX)

The runtime profile chip on the thread header is your shortcut to **Reload
with these settings** — useful when you've drifted to a different model and
want to come back.

## Composing

The composer at the bottom of the thread accepts:

- **Plain text.** Standard chat.
- **Documents.** Drag-and-drop `.pdf`, `.txt`, `.md`, `.docx` (when
  unstructured / docx2txt is installed). Each attachment is chunked and
  retrieved per turn; citations render inline in the assistant response.
- **Images.** Vision-capable models surface an Attach Image button; the
  composer auto-gates based on the model's declared capabilities.
- **Sampler overrides.** Per-turn `temperature`, `top_p`, `min_p`, `seed`,
  `reasoning_effort`, or a JSON schema for constrained output, all without
  touching the launch profile.
- **Cache strategy override.** The KV strategy chip in the composer lets
  you flip between `native` / `turboquant` / `triattention` per turn for
  models that support multiple strategies without a reload.
- **Voice.** The microphone button dictates into the message box, a speaker
  button on every reply reads it aloud, and **Call** starts a hands-free voice
  conversation. See [Voice](voice.md).

## Turn-level actions

Every assistant message exposes:

- **Retry.** Re-runs the same prompt against the active runtime; useful
  when a stream is interrupted or quality is poor.
- **Fork.** Spawns a sibling thread starting from that turn. Great for
  exploring alternate continuations without losing context.
- **Variant.** Streams a sibling response in-place, side-by-side, against
  whatever model you pick. The two variants share the prompt and history
  but each one renders its own metrics.
- **Swap model.** One-turn override — temporarily route the next response
  through a different model, then revert. The thread's resident runtime
  profile stays untouched.
- **Delve.** Critic pass — sends the assistant message back through the
  same model with a critique prompt and renders the critique inline.

## Tool calls

When the loaded model is tool-capable and tools are enabled, the composer
exposes:

- **Built-in tools:** `web_search`, `calculator`, `code_executor`,
  `file_reader`.
- **MCP servers:** any local Model Context Protocol stdio server registered
  in Settings is automatically callable.

Tool results render in dedicated cards above the assistant text — tables,
code blocks, markdown, or inline image previews depending on the tool's
returned shape.

### Studio tools (D-60)

The **Studio** switch next to **Tools** (off by default, remembered) lets the
chat model use the studios: `generate_image`, `generate_video`, `speak` (voice,
style, pitch), `make_music`, `make_sound`, the `storyboard_*` tools,
`music_library`, `list_models`, `list_voices`, `system_status` and the job
tools (`wait_for_job`, `job_status`, `list_jobs`, `cancel_job`). They are the
agent bridge's own tools, run inside the app with fixed permissions: no
downloads, no settings changes, no cloned voices, no file exports, and no
project or track deletes (do those in the app). Removing a shot, an audio clip
or a caption is allowed and can be undone (`storyboard_project` undo). Input
images (reference images, start and end frames) must be files the app made or
`upload:` handles, never any file on the disk. The job tools only see and act
on the jobs this chat started. The Agent access switch does not need to be
on. A short hint tells the model to state a plan, put every render of the plan
in one reply, and wait for each job.

**Nothing renders without your click.** When the model calls a tool that
renders (image, video, speech, music, sound, Storyboard generate or render,
caption transcription), Chat shows a **Plan to approve** card instead: what
will be made, the prompt (the whole prompt behind **more**), model, size, and
every argument that changes the result or its cost: input images by file name,
frames, steps, takes (music and sounds), and for Storyboard how many shots will
be made and whether locked shots are included. The estimated time and memory
come from the catalog where it has them (a video time is marked "at the
recommended size" when another size or length is asked for), plus a memory line such as "Qwen3.8-27B stays loaded:
about 30 GB free" and the busy-machine advice when memory is tight. The chat
model is not unloaded.

- **Approve all** (or **Approve** for one item) runs the listed calls.
- **Approve this** runs that one item and declines the rest (each row's
  button names its item for screen readers).
- **Deny** declines them all; the model is told nothing was made.
- **Edit by chat** declines them all, puts the cursor in the message box, and
  the model asks what to change.
- **Always allow renders in this chat** (off by default) skips the card for
  later renders of this chat, on later turns too. While it is on, a
  **Renders auto-approved** chip next to the Studio switch has a **Turn off**
  button. It ends on its own after 2 hours, when the app restarts, when the
  chat is deleted, or when the Studio switch is turned off.

Without "Always allow", an approval covers the listed calls of that turn only.
Reading tools (`system_status`, `list_models`, `list_jobs`, a Storyboard `get`)
and edits run without a card. A card left unanswered for 30 minutes, a stopped
turn, or a closed window counts as declined. A run that falls back to the
placeholder engine (no real model ran) ends as a failed job with the reason.

Each started job gets a card in the thread: queued (with its place in the
queue) or running (with the job's own progress), a **Cancel** button, and when
done the image, audio player or video with **Open in Library** (Storyboard
jobs: **Open in Project**). Jobs share the heavy-job queue with the studios
and agents, one at a time.

## Reasoning traces

For models that support thinking mode (gpt-oss, Qwen3.6, DeepSeek R1), an
opt-in **Show reasoning** toggle exposes the raw thinking tokens as a
collapsible block above the visible answer. The default-off setting strips
them cleanly; the trace is still captured for replay.

## Metrics

The per-turn host strip on every assistant message surfaces:

- **tok/s** for prompt + generation
- **TTFT** (time-to-first-token)
- **CPU / GPU / RAM / temperature** during the turn
- **Engine + binary** that handled the turn (and which fallback fired, if any)

These are pulled straight from the `runtimeNote` and `metrics` blocks in the
`/api/chat/generate` response — useful when comparing strategies.

## Programmatic alternative

The same surface is reachable from the CLI:

```bash
./scripts/chaosengine-cli prompt "Explain how MTPLX works" \
    --max-tokens 512 --stream --metrics
```

See [CLI recipes](../cli/recipes.md) for batch / scripted chat workflows.
