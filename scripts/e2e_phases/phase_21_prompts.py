"""Phase 213 (Prompts in Chat), from the overnight whole-app review, stream C; D-87 built.

The template store behind Chat's Prompts panel (list, create with variables, update, search, the refusals,
delete), and one real-model check: a template's text sent as ``systemPrompt`` (the panel's "Use as system
prompt") reaches the chat model (the prompt gets longer by about the template's length, and the reply follows
it when the model is able to). The Plugins feature was removed (D-87): its routes must answer 404.

Every template made here has a unique name and is deleted at the end. The real-model
check loads the smallest MLX text model on disk and unloads it, and skips with the reason when there is none.

Called from ``scripts/e2e_test_suite.py`` under key 213 (review group 21, stream C; the other review streams
have 211, 212 and 214).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

PHASE_KEY = 213
NAME = "Phase 213: Prompts in Chat"
MARKER = "PINEAPPLE"


def bind(suite: Any):
    """The suite's ``PHASES`` entry: a named function of ``cap``, so the run log prints its name."""

    def phase_21_prompts(cap: Any):
        return run(cap, suite)

    return phase_21_prompts


def _folder_bytes(path: str, limit: int = 4000) -> int:
    """Size of a model folder (following the Hub cache's links), to pick the smallest one."""
    total = 0
    try:
        for index, item in enumerate(Path(path).rglob("*")):
            if index > limit:
                break
            if item.is_file():
                total += item.stat().st_size
    except OSError:
        return 1 << 62
    return total or 1 << 62


def run(cap: Any, suite: Any):
    phase = suite.PhaseResult(phase=PHASE_KEY, name=NAME)
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 21 prompts", "skip", reason="backend not reachable"))
        return phase

    def api(method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        code, raw = suite._voice_http(method, path, body=body, content_type="application/json" if body else None, timeout=timeout)
        try:
            return code, json.loads(raw.decode("utf-8")) if raw.strip() else None
        except ValueError:
            return code, raw

    tag = uuid.uuid4().hex[:8]
    name = f"e2e-template-{tag}"
    state: dict[str, Any] = {"made": []}

    # ---- Prompts -------------------------------------------------------------------------------------

    def prompts_list():
        code, data = api("GET", "/api/prompts")
        if code == 404:
            return "skip", "prompt routes not present in this build", {}
        if code != 200 or not isinstance(data, dict) or not isinstance(data.get("templates"), list):
            return "fail", f"HTTP {code}: {str(data)[:160]}", {}
        if data.get("count") != len(data["templates"]):
            return "fail", f"count {data.get('count')} is not the number of templates {len(data['templates'])}", {}
        for row in data["templates"]:
            missing = [key for key in ("id", "name", "systemPrompt", "tags", "category") if key not in row]
            if missing:
                return "fail", f"template {row.get('id')} lacks {missing}", {}
        state["prompts"] = True
        return "pass", "", {"templates": data["count"]}

    def prompts_create():
        if not state.get("prompts"):
            return "skip", "prompt routes not present", {}
        code, data = api("POST", "/api/prompts", {
            "name": name,
            "systemPrompt": "Answer only in {{language}}. Mention {{ animal }} once.",
            "category": "E2E",
            "tags": ["e2e", tag],
            "variables": [
                {"name": "language", "type": "string", "default": "French", "description": "Reply language"},
                {"name": "animal", "type": "colour", "default": "quokka"},  # unknown type becomes string
                {"name": "language", "type": "string"},  # duplicate name is dropped
                {"type": "string"},  # no name is dropped
            ],
            "presetSamplers": {"topP": 0.9},
            "presetModelRef": "any/model",
        })
        template = (data or {}).get("template") if isinstance(data, dict) else None
        if code != 200 or not template or not data.get("created"):
            return "fail", f"HTTP {code}: {str(data)[:200]}", {}
        state["made"].append(template["id"])
        state["id"] = template["id"]
        variables = [(v["name"], v["type"]) for v in template.get("variables", [])]
        if variables != [("language", "string"), ("animal", "string")]:
            return "fail", f"variables were stored as {variables}", {}
        if not isinstance(template.get("createdAt"), (int, float)):
            return "fail", f"createdAt is {template.get('createdAt')!r}, expected epoch seconds", {}
        return "pass", "", {"id": template["id"]}

    def prompts_update_and_search():
        if not state.get("id"):
            return "skip", "no template was made", {}
        code, data = api("POST", "/api/prompts", {
            "id": state["id"], "name": f"{name} v2", "systemPrompt": f"Reply with the code word {tag}zebra.",
            "category": "E2E", "tags": ["e2e", tag],
        })
        template = (data or {}).get("template") if isinstance(data, dict) else None
        if code != 200 or not template or data.get("created") is not False or template["id"] != state["id"]:
            return "fail", f"update answered HTTP {code}: {str(data)[:200]}", {}
        if template["updatedAt"] < template["createdAt"] or template["name"] != f"{name} v2":
            return "fail", "the update did not change the name or the updated time", {}
        found = {}
        for label, query in (("text", f"query={tag}zebra"), ("tag", f"tag={tag}"), ("category", f"category=e2e&query={tag}")):
            _, listed = api("GET", f"/api/prompts?{query}")
            ids = [t["id"] for t in (listed or {}).get("templates", [])]
            found[label] = ids == [state["id"]]
        if not all(found.values()):
            return "fail", f"search did not find exactly the template: {found}", {}
        _, none = api("GET", f"/api/prompts?query=no-such-text-{tag}")
        if (none or {}).get("count") != 0:
            return "fail", "a query that matches nothing returned templates", {}
        return "pass", "", found

    def prompts_refusals():
        if not state.get("prompts"):
            return "skip", "prompt routes not present", {}
        answers = {
            "empty name": api("POST", "/api/prompts", {"name": "", "systemPrompt": "x"})[0],
            "no name": api("POST", "/api/prompts", {"systemPrompt": "x"})[0],
            "prompt over 16000 characters": api("POST", "/api/prompts", {"name": f"{name}-long", "systemPrompt": "x" * 16001})[0],
            "tags not a list": api("POST", "/api/prompts", {"name": f"{name}-tags", "tags": "one"})[0],
            "a variable that is not an object": api("POST", "/api/prompts", {"name": f"{name}-var", "variables": ["x"]})[0],
            "delete unknown": api("DELETE", f"/api/prompts/no-such-template-{tag}")[0],
        }
        wanted = {"empty name": 422, "no name": 422, "prompt over 16000 characters": 422, "tags not a list": 422,
                  "a variable that is not an object": 422, "delete unknown": 404}
        if answers != wanted:
            return "fail", f"got {answers}, wanted {wanted}", {}
        return "pass", "", answers

    def prompts_delete():
        if not state.get("id"):
            return "skip", "no template was made", {}
        code, data = api("DELETE", f"/api/prompts/{state['id']}")
        again, _ = api("DELETE", f"/api/prompts/{state['id']}")
        _, listed = api("GET", "/api/prompts")
        still = any(t["id"] == state["id"] for t in (listed or {}).get("templates", []))
        if code != 200 or not (data or {}).get("deleted") or again != 404 or still:
            return "fail", f"delete HTTP {code}, second delete HTTP {again}, still listed: {still}", {}
        state["made"].remove(state.pop("id"))
        return "pass", "", {}

    def prompt_reaches_the_model():
        if not state.get("prompts"):
            return "skip", "prompt routes not present", {}
        if not getattr(cap, "mlx_usable", False) or not cap.local_mlx_models:
            return "skip", "no MLX text model on disk", {}
        ref, path = min(cap.local_mlx_models, key=lambda pair: _folder_bytes(pair[1]))
        system = f"You are a test fixture. Whatever the user says, reply with exactly one word: {MARKER}"
        code, made = api("POST", "/api/prompts", {"name": f"{name}-real", "systemPrompt": system, "category": "E2E"})
        if code != 200:
            return "fail", f"could not save the template: HTTP {code}", {}
        state["made"].append(made["template"]["id"])
        rc, _loaded, err = suite._cli_json(
            "load", ref, "--backend", "mlx", "--cache-strategy", "native", "--context", "4096",
            "--path", path, "--timeout", "900", timeout=960.0)
        if rc != 0:
            return "fail", f"load {ref} rc={rc}: {(err or '')[:200]}", {}
        try:
            replies: dict[str, tuple[int, str]] = {}
            # "plain" carries a two-word system prompt too: a chat template with a default system prompt (Qwen2.5:
            # "You are Qwen, created by Alibaba Cloud...") swaps it for ours, so no system prompt at all would make
            # the template look like it added nothing.
            for label, extra in (("plain", {"systemPrompt": "Be brief."}),
                                 ("template", {"systemPrompt": made["template"]["systemPrompt"]})):
                body = {"sessionId": f"e2e-p21-{tag}-{label}", "prompt": "What is the capital of France?",
                        "modelRef": ref, "backend": "mlx", "cacheStrategy": "native", "maxTokens": 48,
                        "temperature": 0.0, "thinkingMode": "off", **extra}
                code, gen = api("POST", "/api/chat/generate", body, timeout=300.0)
                if code != 200 or not isinstance(gen, dict):
                    return "fail", f"generate ({label}) HTTP {code}: {str(gen)[:200]}", {}
                assistant = gen.get("assistant") or {}
                metrics = assistant.get("metrics") or {}
                messages = (gen.get("session") or {}).get("messages") or [{}]
                text = str(assistant.get("text") or messages[-1].get("text") or "")
                replies[label] = (int(metrics.get("promptTokens") or 0), text)
                api("DELETE", f"/api/chat/sessions/e2e-p21-{tag}-{label}")
        finally:
            suite._cli("unload", timeout=60.0)
        plain_tokens, plain_text = replies["plain"]
        template_tokens, template_text = replies["template"]
        detail = {"model": ref, "promptTokens": {"plain": plain_tokens, "template": template_tokens},
                  "followed": MARKER in template_text.upper(), "reply": template_text[:80]}
        # The hard check does not depend on how obedient a small model is: the prompt the model saw is
        # longer by about the template's length (at least 10 tokens for this 20-word template against "Be brief.").
        if template_tokens - plain_tokens < 10:
            return "fail", f"the prompt did not grow with the template: {plain_tokens} then {template_tokens} tokens", detail
        if MARKER in plain_text.upper():
            return "fail", "the marker word is in the reply that had no template", detail
        return "pass", "" if detail["followed"] else "the prompt reached the model; this model did not follow it word for word", detail

    # ---- Plugins (removed, D-87) -------------------------------------------------------------------

    def plugins_gone():
        answers = {"GET /api/plugins": api("GET", "/api/plugins")[0]}
        for action in ("enable", "disable"):
            answers[f"POST /api/plugins/tool.calculator/{action}"] = api("POST", f"/api/plugins/tool.calculator/{action}")[0]
        if set(answers.values()) != {404}:
            return "fail", f"the removed Plugins routes still answer: {answers}", {}
        return "pass", "", answers

    def token_needed():
        # With auth on (the default), the prompt routes refuse a call that carries no token.
        answers = {}
        anonymous = getattr(suite, "_anonymous_http", None)  # the in-process test supplies its own client
        for path in ("/api/prompts",):
            if anonymous is not None:
                answers[path] = anonymous(path)
                continue
            try:
                with urllib.request.urlopen(f"http://{suite._HOST}:{suite._PORT}{path}", timeout=10.0) as resp:
                    answers[path] = resp.status
            except urllib.error.HTTPError as exc:
                answers[path] = exc.code
            except OSError as exc:
                return "fail", f"{path}: {exc}", {}
        if set(answers.values()) == {200}:
            return "skip", "this backend runs with auth off", answers
        if set(answers.values()) != {401}:
            return "fail", f"expected 401 without a token, got {answers}", {}
        return "pass", "", answers

    try:
        for label, fn in [
            ("GET /api/prompts lists templates with their fields", prompts_list),
            ("a template with variables and presets saves, with bad variable rows dropped", prompts_create),
            ("an update keeps the id, and search finds it by text, tag and category", prompts_update_and_search),
            ("a missing name, an oversized prompt, bad tags, a bad variable and an unknown delete are refused", prompts_refusals),
            ("delete removes the template and a second delete is 404", prompts_delete),
            ("the removed Plugins routes answer 404", plugins_gone),
            ("the prompt routes need the API token", token_needed),
            ("a template sent as the system prompt reaches a real chat model", prompt_reaches_the_model),
        ]:
            phase.checks.append(suite._check(label, fn))
    finally:
        for template_id in list(state["made"]):
            api("DELETE", f"/api/prompts/{template_id}")
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase
