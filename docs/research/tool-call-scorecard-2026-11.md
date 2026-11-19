# Tool-call scorecard, November 2026 (FU-230, D-90)

How well a chat model writes the Studio tool calls Chat offers it. The plan is in
[keep-going-and-tools-v2.md](../plans/keep-going-and-tools-v2.md) ("Measurement plan"). This page holds the results;
add a section per run.

## The script

`scripts/tool-call-scorecard.py` runs **tier 1**: twelve prompts, each with one right answer (an enum, a required
argument, numbers in range, a nested object, a call with no arguments, two calls in one reply). Each goes to a
running backend's `/v1/chat/completions` with the 20 Studio tools exactly as Chat offers them
(`tools.studio.chat_schema`) and the Studio hint as the system prompt. Nothing runs; replies are only scored, with the
bridge's own `validate`. Calls come back as the app reads them: llama-server's parser on GGUF, the agent loop's
reader on MLX.

Per reply it records: a call came back, the right tool, valid arguments, the right values, and a reply that looks like
a call the app could not read. Results are counts out of runs x prompts. With five runs a difference of one or two
is noise.

```
.venv/bin/python scripts/tool-call-scorecard.py --port 8876 --model REF=PATH   # repeat --model; or none for the loaded model
```

Defaults are the app's chat settings: temperature 0.7, reply limit 4,096, context 8,192, five runs. Output: a CSV of
every reply and a Markdown table in `~/.chaosengine/test-results/`. Tests: `tests/test_tool_call_scorecard.py`.

Not built yet: tier 2 (a short Studio task in a real chat) and tier 3 (the demo film).

**What "read" means here.** The script sees the calls the server returned. On the llama.cpp path there is no repair
step (llama-server parses with the model's own template). On the MLX path the server reads text calls with the agent
loop's reader, which mends a call that lacks only its closing braces, so "call came back" there includes mended calls;
a reply whose text still holds a call-like fragment counts as unreadable.

**A fix this needed.** Before 2026-11-18 `/v1/chat/completions` returned no `tool_calls` at all: llama-server's
structured calls and the MLX path's text calls were both dropped, so every client that sent `tools` got an empty
reply. The non-streamed reply now carries them (`state/openai_compat._reply_tool_calls`). Streaming still drops them
(FU-258).

## Run 1: Qwen3.8 27B and Qwen3-0.6B, GGUF, on the RTX 4090 (2026-11-18)

llama.cpp b12441 CUDA 13.5 build installed through the app, Windows 11, RTX 4090 (24 GB). Five runs per prompt,
Studio hint on, temperature 0.7, context 8,192, reply limit 4,096 (the app's defaults).

- `lmstudio-community/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-Q4_K_M.gguf` (15.7 GB, the catalogue row for 24 GB cards;
  downloaded for this run through the app). This is the GGUF side of claim C7; the MLX side runs on the Mac.
- `lmstudio-community/Qwen3-0.6B-GGUF`, `Qwen3-0.6B-Q8_0.gguf`: the floor of the scale.

| Model | Replies | Call came back | Right tool | Valid arguments | Right values | Looked up first | Unreadable | Median s | Prompt tokens |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8 27B Q4_K_M | 60 | 60 of 60 | 50 of 60 | 60 of 60 | **50 of 60** | 10 of 60 | 0 of 60 | 2.06 | 7,261 |
| Qwen3-0.6B Q8_0 | 60 | 48 of 60 | 40 of 60 | 48 of 60 | **37 of 60** | 0 of 60 | 0 of 60 | 0.17 | 7,141 |

"Looked up first": the reply holds only read-only look-ups (`list_models`, `list_voices`, `system_status` and the like)
where the prompt asked for an action. The Studio hint tells the model to do exactly that ("call list_models first when
you need a model id"), so it is a first step and not a wrong tool; tier 2, with real tool answers, will show whether the
action follows. A reply with the right call and an extra look-up beside it counts as right; an extra action does not.

| Prompt | Tests | Qwen3.8 27B | Qwen3-0.6B | What went wrong |
|---|---|---|---|---|
| kind-video | enum | 5 of 5 | 3 of 5 | 0.6B: prose, naming models it made up or took from other tools' text (a sound model as a video model) |
| kind-voice | enum | 5 of 5 | 3 of 5 | 0.6B: "I don't have access to a list of speech models", no call |
| image-prompt | required argument | 4 of 5 | 5 of 5 | 27B: once `list_models(image)` first |
| video-size | numbers in range | 2 of 5 | 5 of 5 | 27B: three times `list_models(video)` first, after writing the plan |
| speak-speed | number in range | 0 of 5 | 5 of 5 | 27B: `list_voices` first, every time |
| status | no arguments | 5 of 5 | 5 of 5 | |
| jobs-failed | enum | 5 of 5 | 0 of 5 | 0.6B: said it could not list failed jobs; no call |
| project-create | enums and a name | 5 of 5 | 4 of 5 | 0.6B: name put in `description` |
| shot-fields | nested object | 5 of 5 | 3 of 5 | 0.6B: a `path` instead of `fields`, or `fields` with only a title |
| music-flags | boolean and numbers | 5 of 5 | 0 of 5 | 0.6B: `storyboard_song attach` with placeholder ids, or `make_sound` with the example caption from that tool's own description |
| two-calls | two calls in one reply | 4 of 5 | 2 of 5 | 27B: once `list_voices` first; 0.6B: one image call with the voice line in its prompt |
| wait-job | required id | 5 of 5 | 2 of 5 | 0.6B: said it would wait; no call |

What this says:

- **Qwen3.8 27B on GGUF never wrote a wrong call.** All 60 replies were calls, all valid, and all 10 misses were
  look-ups the hint asks for. On this path llama-server builds a grammar from the tool schemas, so arguments always fit
  the types and enums; the remaining question is the order of steps, which tier 2 measures. This is the GGUF half of C7.
- **The 0.6B model is not for tools**, as expected: it answers in prose where a call is needed, and picks the wrong tool
  for music every time. Every call it did make was valid (48 of 48), again the grammar.
- **The `fields` object of `storyboard_shot` lists no keys** (F3), so a model guesses them; the 27B guessed right every
  time, the 0.6B did not.
- **The tool list costs about 7,200 prompt tokens on this path.** llama-server puts the 20 tools as JSON schema into
  Qwen's own template (the plan measured that form at 23,540 characters, F2). With the app's defaults (8,192 context,
  4,096 reply limit) that leaves about 1,000 tokens for history and the reply. FU-231's staged list (core tools always,
  Storyboard tools on demand) and its history budget matter on GGUF as much as on MLX.
- Speed: a call took about 2 s on the 27B (66 completion tokens on average) and 0.2 s on the 0.6B.

A first pass the same day scored a look-up as a wrong tool and took only the first call of a single-call prompt; the
numbers above are from the corrected scoring (`tool-call-scorecard-20261118-153057.csv`).

## Still to run

- The plan's four MLX models on the Mac: Qwen3.8 27B 4-bit, LFM2.5 24B-A2B 4-bit, Qwen3 8B, Qwen3 4B; then C7
  compares the MLX Qwen3.8 27B with the GGUF row above.
- Tiers 2 and 3.
