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
