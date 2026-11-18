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

## Run 1: Qwen3-0.6B GGUF on the RTX 4090 (2026-11-18)

`lmstudio-community/Qwen3-0.6B-GGUF`, `Qwen3-0.6B-Q8_0.gguf`, llama.cpp b12441 CUDA 13.5 build installed through the
app, Windows 11, RTX 4090. Five runs, Studio hint on, temperature 0.7. This is the floor of the scale, not one of the
plan's four models (those are MLX builds on the Mac; the PC has no Qwen3.8 27B GGUF weights).

| Model | Replies | Call came back | Right tool | Valid arguments | Right values | Unreadable | Median s | Prompt tokens |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B Q8_0 (GGUF) | 60 | 46 of 60 | 36 of 60 | 46 of 60 | 33 of 60 | 0 of 60 | 0.16 | 7,141 |

| Prompt | Tests | Right values | What it did when wrong |
|---|---|---|---|
| kind-video | enum | 1 of 5 | answered in prose, naming models it made up or took from other tools' text (a sound model as a video model) |
| kind-voice | enum | 5 of 5 | |
| image-prompt | required argument | 5 of 5 | |
| video-size | numbers in range | 5 of 5 | |
| speak-speed | number in range | 5 of 5 | |
| status | no arguments | 5 of 5 | |
| jobs-failed | enum | 0 of 5 | said it could not, and asked for job ids; no call |
| project-create | enums and a name | 4 of 5 | once: `aspect` 1:1 beside `size` 16:9, and 1440p |
| shot-fields | nested object | 3 of 5 | `fields` without any picture text (start mode, seconds, seed) and an invented shot id |
| music-flags | boolean and numbers | 0 of 5 | called `storyboard_song` `attach` with placeholder ids, every time |
| two-calls | two calls in one reply | 0 of 5 | one image call with the voice line in its prompt |
| wait-job | required id | 0 of 5 | said it would wait; no call |

Every call that came back was valid against its schema (46 of 46). llama-server builds a grammar from the tool schemas,
so on this path the arguments fit the types and enums and the faults are the choice of tool and the values, not the
format. This bears on claim C7 and on option B (grammar decoding) for MLX. The shot-fields misses fit F3: the
`fields` object of `storyboard_shot` lists no keys, so the model guesses them.

**The tool list costs 7,141 prompt tokens on this path.** llama-server puts the 20 tools as JSON schema into Qwen's own
template (the plan measured that form at 23,540 characters, F2). With the app's defaults (8,192 context, 4,096 reply
limit) that leaves about 1,000 tokens for history and the reply. FU-231's staged list (core tools always, Storyboard
tools on demand) and its history budget matter on GGUF as much as on MLX.

## Still to run

- The plan's four MLX models on the Mac: Qwen3.8 27B 4-bit, LFM2.5 24B-A2B 4-bit, Qwen3 8B, Qwen3 4B.
- Qwen3.8 27B as GGUF for claim C7 (`unsloth/Qwen3.8-27B-GGUF`, about 17 GB at Q4_K_M; not on the PC, not downloaded).
- Tiers 2 and 3.
