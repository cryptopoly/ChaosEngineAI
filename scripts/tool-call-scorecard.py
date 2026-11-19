#!/usr/bin/env python3
"""Tool-call scorecard, tier 1 (FU-230, D-90): how well a chat model writes Studio tool calls.

Twelve prompts, each with one right answer (an enum, a required argument, numbers in range, a nested object, a
call with no arguments, two calls in one reply), are sent to a running backend's OpenAI-compatible endpoint with
the Studio tools exactly as Chat offers them (``tools.studio.chat_schema``) and the Studio hint as the system
prompt. Nothing runs: the reply is only scored. Calls come back as the app reads them (llama-server's own parser on
GGUF, the agent loop's reader on MLX) and are checked with the bridge's own ``validate``.

Per model it counts, out of runs x prompts: a call came back, the right tool, valid arguments, the right values, and
replies that look like a call the app could not read. Counts, never bare percentages: with five runs a difference
of one or two is noise. Output: a CSV of every reply and a Markdown table in ``~/.chaosengine/test-results/``.

    .venv/bin/python scripts/tool-call-scorecard.py --port 8876 --model lmstudio-community/Qwen3-0.6B-GGUF=/path/x.gguf
    .venv/bin/python scripts/tool-call-scorecard.py          # the model that is loaded now

Tiers 2 (a short Studio task) and 3 (the demo film) are not built yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Check = Callable[[dict[str, Any]], bool]


def _has(*texts: str) -> Check:
    """Any of ``texts`` appears in the arguments (lower case)."""
    return lambda args: any(text in json.dumps(args).lower() for text in texts)


def _eq(**want: Any) -> Check:
    return lambda args: all(args.get(key) == value for key, value in want.items())


def _all(*checks: Check) -> Check:
    return lambda args: all(check(args) for check in checks)


@dataclass(frozen=True)
class Expect:
    tools: tuple[str, ...]  # any of these is the right tool
    check: Check = lambda args: True


@dataclass(frozen=True)
class Prompt:
    id: str
    tests: str
    text: str
    expect: tuple[Expect, ...]  # every one must be met (two calls in one reply = two)


def _shot_fields(args: dict[str, Any]) -> bool:
    fields = args.get("fields")
    return isinstance(fields, dict) and "lighthouse" in json.dumps(fields).lower() and fields.get("seconds") == 4


PROMPTS: tuple[Prompt, ...] = (
    Prompt("kind-video", "enum", "Which video models are installed here? Check before you plan anything.",
           (Expect(("list_models",), _eq(kind="video")),)),
    Prompt("kind-voice", "enum", "Before you write the voiceover, find out which speech models I have.",
           (Expect(("list_models", "list_voices"), lambda a: a.get("kind", "voice") == "voice"),)),
    Prompt("image-prompt", "required argument", "Make one image: a red fox asleep in fresh snow, soft morning light.",
           (Expect(("generate_image",), _has("fox")),)),
    Prompt("video-size", "numbers in range",
           "Make a 3 second video clip of waves breaking on black rocks at 768x512.",
           (Expect(("generate_video",), _all(_eq(width=768, height=512), lambda a: a.get("seconds") == 3
                                              or isinstance(a.get("frames"), int))),)),
    Prompt("speak-speed", "number in range", "Say 'Welcome to the flower shop' in a calm voice, a little slower than normal.",
           (Expect(("speak",), _all(_has("welcome"), lambda a: isinstance(a.get("speed"), (int, float))
                                    and 0.5 <= a["speed"] < 1.0)),)),
    Prompt("status", "no arguments", "Is anything running on this computer right now? Check the status.",
           (Expect(("system_status", "list_jobs")),)),
    Prompt("jobs-failed", "enum", "Show me the jobs that failed.",
           (Expect(("list_jobs",), _eq(state="failed")),)),
    Prompt("project-create", "enums and a name",
           "Start a new film project called Harbour Lights, 16:9 at 1080p, 24 fps. Do not add shots yet.",
           (Expect(("storyboard_project",), _all(_eq(action="create", name="Harbour Lights", resolution="1080p", fps=24),
                                                lambda a: "16:9" in (a.get("aspect"), a.get("shape"), a.get("size")))),)),
    Prompt("shot-fields", "nested object",
           "In project p-123 at revision 4, add one shot: a lighthouse at dusk, 4 seconds long.",
           (Expect(("storyboard_shot",), _all(_eq(action="add", project_id="p-123", revision=4), _shot_fields)),)),
    Prompt("music-flags", "boolean and numbers", "Make a 30 second instrumental lo-fi beat at 80 bpm.",
           (Expect(("make_music",), _all(_eq(instrumental=True, seconds=30, bpm=80), _has("lo-fi", "lofi", "lo fi"))),)),
    Prompt("two-calls", "two calls in one reply",
           "Make an image of a lighthouse at night, and a voice line that says 'Safe harbour'. Both at once.",
           (Expect(("generate_image",), _has("lighthouse")), Expect(("speak",), _has("safe harbour")))),
    Prompt("wait-job", "required id", "Wait for job job-42 to finish, then tell me.",
           (Expect(("wait_for_job",), _eq(job_id="job-42")),)),
)

# Text a reply holds when the model tried to call a tool that the app could not read.
_CALL_MARKS = ("<tool_call>", "<function=", "<|tool_call_start|>", '"arguments"', '"name":')


def studio_tools() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    """Chat's Studio tools in OpenAI form, their schemas by name, and the Studio hint."""
    from backend_service.tools.studio import STUDIO_HINT, STUDIO_TOOL_NAMES, _bridge, chat_schema

    by_name = {tool["name"]: tool for tool in _bridge()[0]}
    schemas = {name: chat_schema(by_name[name]) for name in STUDIO_TOOL_NAMES}
    tools = [{"type": "function", "function": {"name": name, "description": by_name[name]["description"],
                                                "parameters": schemas[name]}} for name in STUDIO_TOOL_NAMES]
    return tools, schemas, STUDIO_HINT


@dataclass
class Score:
    called: bool = False
    unreadable: bool = False
    right_tool: bool = False
    valid: bool = False
    right_values: bool = False
    looked_up: bool = False  # only read-only look-ups where an action was asked: what the Studio hint tells it to do
    problems: list[str] = field(default_factory=list)


# Read-only tools the Studio hint tells the model to call before acting ("call list_models first when you need a
# model id"). A reply of only these, where the prompt asks for an action, is a first step, not a wrong tool.
LOOKUPS = frozenset({"system_status", "list_models", "list_voices", "list_jobs", "job_status", "music_library"})


def score_reply(prompt: Prompt, message: dict[str, Any], schemas: dict[str, dict[str, Any]]) -> Score:
    """Score one reply (an OpenAI ``message``) against the prompt's expectations."""
    from backend_service.mcp_bridge.tools import validate

    score = Score()
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        try:
            args = json.loads(function.get("arguments") or "{}")
        except (TypeError, json.JSONDecodeError):
            args = None
        calls.append((str(function.get("name") or ""), args))
    content = str(message.get("content") or "")
    score.called = bool(calls)
    score.unreadable = not calls and any(mark in content for mark in _CALL_MARKS)
    if not calls:
        return score
    score.valid = True
    for name, args in calls:
        problem = "arguments are not JSON" if args is None else (
            f"unknown tool {name}" if name not in schemas else validate(schemas[name], args))
        if problem:
            score.valid = False
            score.problems.append(f"{name}: {problem}")
    names = [name for name, _args in calls]
    # Every expected tool is called; anything else in the reply is a read-only look-up, never another action.
    wanted = {tool for want in prompt.expect for tool in want.tools}
    score.right_tool = (all(any(name in want.tools for name in names) for want in prompt.expect)
                        and all(name in wanted or name in LOOKUPS for name in names))
    score.right_values = score.right_tool and score.valid and all(
        any(name in want.tools and isinstance(args, dict) and want.check(args) for name, args in calls)
        for want in prompt.expect)
    asked_action = any(not set(want.tools) <= LOOKUPS for want in prompt.expect)
    score.looked_up = not score.right_tool and asked_action and all(name in LOOKUPS for name in names)
    return score


class Backend:
    def __init__(self, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}"
        token = self._json("GET", "/api/auth/session", auth=False).get("apiToken")
        self.headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}

    def _json(self, method: str, path: str, body: Any = None, auth: bool = True, timeout: float = 60.0) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = self.headers if auth else {}
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8") or "{}")

    def loaded(self) -> str | None:
        runtime = self._json("GET", "/api/workspace", timeout=120).get("runtime") or {}
        model = runtime.get("loadedModel") or {}
        return model.get("ref") or model.get("name")

    def load(self, ref: str, path: str | None, context: int) -> None:
        body = {"modelRef": ref, "canonicalRepo": ref, "path": path, "contextTokens": context,
                "source": "library" if path else "catalog"}
        self._json("POST", "/api/models/load", body, timeout=1800)

    def unload(self) -> None:
        self._json("POST", "/api/models/unload", {}, timeout=300)

    def chat(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        return self._json("POST", "/v1/chat/completions", body, timeout=timeout)


def run_model(backend: Backend, ref: str, args: argparse.Namespace, tools: list[dict[str, Any]],
              schemas: dict[str, dict[str, Any]], hint: str, prompts: tuple[Prompt, ...]) -> list[dict[str, Any]]:
    rows = []
    for run in range(1, args.runs + 1):
        for prompt in prompts:
            messages = ([{"role": "system", "content": hint}] if args.system == "studio" else []) + [
                {"role": "user", "content": prompt.text}]
            body = {"model": ref, "messages": messages, "tools": tools, "temperature": args.temperature,
                    "max_tokens": args.max_tokens}
            started = time.perf_counter()
            try:
                reply = backend.chat(body, timeout=args.timeout)
                error = ""
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reply, error = {}, f"{type(exc).__name__}: {exc}"[:300]
            seconds = round(time.perf_counter() - started, 2)
            choice = (reply.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            score = score_reply(prompt, message, schemas)
            usage = reply.get("usage") or {}
            rows.append({
                "model": ref, "run": run, "prompt": prompt.id, "tests": prompt.tests,
                "called": int(score.called), "right_tool": int(score.right_tool), "valid": int(score.valid),
                "right_values": int(score.right_values), "unreadable": int(score.unreadable),
                "looked_up": int(score.looked_up), "finish": choice.get("finish_reason") or "", "seconds": seconds,
                "prompt_tokens": usage.get("prompt_tokens") or 0, "completion_tokens": usage.get("completion_tokens") or 0,
                "calls": json.dumps([[c.get("function", {}).get("name"), c.get("function", {}).get("arguments")]
                                     for c in message.get("tool_calls") or []])[:600],
                "content": str(message.get("content") or "")[:300], "problems": "; ".join(score.problems)[:300],
                "error": error,
            })
            print(f"[{ref}] run {run} {prompt.id}: called={score.called} tool={score.right_tool} "
                  f"valid={score.valid} values={score.right_values} {seconds}s", flush=True)
    return rows


def summary(rows: list[dict[str, Any]]) -> str:
    """The Markdown table: one line per model, counts out of every reply."""
    lines = ["| Model | Replies | Call came back | Right tool | Valid arguments | Right values | Looked up first | "
             "Unreadable | Median s | Prompt tokens |", "|---|---|---|---|---|---|---|---|---|---|"]
    for model in dict.fromkeys(row["model"] for row in rows):
        mine = [row for row in rows if row["model"] == model]
        n = len(mine)
        seconds = sorted(row["seconds"] for row in mine)
        median = seconds[n // 2] if seconds else 0
        lines.append(f"| `{model}` | {n} | " + " | ".join(
            f"{sum(row.get(key, 0) for row in mine)} of {n}"
            for key in ("called", "right_tool", "valid", "right_values", "looked_up", "unreadable"))
                     + f" | {median} | {max((row['prompt_tokens'] for row in mine), default=0)} |")
    lines += ["", "| Model | Prompt | Tests | Right values | Valid | Problems seen |", "|---|---|---|---|---|---|"]
    for model in dict.fromkeys(row["model"] for row in rows):
        for prompt in dict.fromkeys(row["prompt"] for row in rows if row["model"] == model):
            mine = [row for row in rows if row["model"] == model and row["prompt"] == prompt]
            seen = "; ".join(sorted({row["problems"] for row in mine if row["problems"]}))[:200] or "-"
            lines.append(f"| `{model}` | {prompt} | {mine[0]['tests']} | {sum(r['right_values'] for r in mine)} of "
                         f"{len(mine)} | {sum(r['valid'] for r in mine)} of {len(mine)} | {seen} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=int(os.environ.get("CHAOSENGINE_PORT", "8876")))
    parser.add_argument("--model", action="append", default=[], metavar="REF[=PATH]",
                        help="load this model (repeat for several); without it the loaded model is scored")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7, help="the app's chat default")
    parser.add_argument("--max-tokens", type=int, default=4096, help="the app's chat default reply limit")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--system", choices=("studio", "none"), default="studio")
    parser.add_argument("--prompts", default="", help="comma-separated prompt ids (default: all twelve)")
    parser.add_argument("--keep-loaded", action="store_true", help="do not unload after each model")
    parser.add_argument("--out-dir", default=str(Path.home() / ".chaosengine" / "test-results"))
    args = parser.parse_args(argv)

    prompts = tuple(p for p in PROMPTS if not args.prompts or p.id in args.prompts.split(","))
    tools, schemas, hint = studio_tools()
    backend = Backend(args.port)
    rows: list[dict[str, Any]] = []
    targets = [spec.partition("=")[::2] for spec in args.model] or [(backend.loaded() or "", "")]
    for ref, path in targets:
        if not ref:
            print("No model is loaded; pass --model REF[=PATH].", file=sys.stderr)
            return 2
        if args.model:
            print(f"loading {ref} ...", flush=True)
            backend.load(ref, path or None, args.context)
        try:
            rows += run_model(backend, ref, args, tools, schemas, hint, prompts)
        finally:
            if args.model and not args.keep_loaded:
                backend.unload()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with (out / f"tool-call-scorecard-{stamp}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = summary(rows)
    (out / f"tool-call-scorecard-{stamp}.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"written: {out / f'tool-call-scorecard-{stamp}'}.{{csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
