"""Phase 17: Agent bridge v2 (annotations, scopes, list_jobs, queue view, the download refusal path).

Called from ``scripts/e2e_test_suite.py`` as ``run(cap, suite)`` where ``suite`` is that module (it owns the
result types and HTTP helpers). Needs no model: nothing here starts a generating job.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def run(cap: Any, suite: Any):
    phase = suite.PhaseResult(phase=17, name="Agent bridge v2")
    if not cap.backend_reachable:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 17", "skip", reason="backend not reachable"))
        return phase
    base = f"http://{suite._HOST}:{suite._PORT}"

    def api(method: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        code, raw = suite._voice_http(method, path, body=body, content_type="application/json" if body else None, timeout=60.0)
        try:
            return code, json.loads(raw.decode("utf-8")) if raw.strip() else None
        except ValueError:
            return code, raw

    def mcp(method: str, params: dict[str, Any] | None = None, token: str | None = None) -> tuple[int, dict[str, Any]]:
        """One JSON-RPC call. With ``token`` it is sent as that agent token instead of the app's own."""
        message = {"jsonrpc": "2.0", "id": 1, "method": method, **({"params": params} if params is not None else {})}
        if token is None:
            code, body = api("POST", "/mcp", message)
            return code, body if isinstance(body, dict) else {}
        request = urllib.request.Request(f"{base}/mcp", data=json.dumps(message).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, {}

    def tool(name: str, arguments: dict[str, Any] | None = None, token: str | None = None) -> tuple[bool, Any]:
        code, body = mcp("tools/call", {"name": name, "arguments": arguments or {}}, token)
        if code != 200:
            raise RuntimeError(f"{name}: HTTP {code}")
        result = body["result"]
        text = result["content"][0]["text"]
        try:
            return bool(result["isError"]), json.loads(text)
        except ValueError:
            return bool(result["isError"]), text

    code, before = api("GET", "/api/agent/access")
    if code == 404 or not isinstance(before, dict) or "scopes" not in before:
        phase.status = "skip"
        phase.checks.append(suite.CheckResult("phase 17", "skip", reason="agent bridge v2 not present in this build"))
        return phase
    saved = {"enabled": bool(before.get("enabled")), "scopes": before.get("scopes") or [], "readToken": bool(before.get("readToken"))}
    state: dict[str, Any] = {}

    def annotations():
        api("POST", "/api/agent/access", {"enabled": True})
        tools = mcp("tools/list")[1]["result"]["tools"]
        hints = ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")
        bad = [t["name"] for t in tools if not all(isinstance(t.get("annotations", {}).get(h), bool) for h in hints)]
        names = {t["name"] for t in tools}
        if bad or "upload_file" not in names or "list_jobs" not in names or "upload_image" in names:
            return "fail", f"annotations missing on {bad}; upload_file/list_jobs present: {sorted(names & {'upload_file', 'list_jobs'})}", {}
        return "pass", "", {"tools": len(tools)}

    def default_token_scopes():
        scopes = set(api("GET", "/api/agent/access")[1]["scopes"])
        if not {"read", "generate", "files"} <= scopes:
            return "fail", f"scopes are {sorted(scopes)}", {}
        return "pass", "", {"scopes": sorted(scopes)}

    def read_only_token():
        made = api("POST", "/api/agent/access", {"readToken": True})[1]
        state["read_token"] = made.get("readToken")
        if not state["read_token"]:
            return "fail", "no read-only token was made", {}
        token = state["read_token"]
        is_error, _ = tool("system_status", token=token)
        if is_error:
            return "fail", "the read-only token could not read system_status", {}
        for name, args in (("generate_image", {"prompt": "x"}), ("speak", {"text": "hi"}),
                           ("upload_file", {"data_base64": base64.b64encode(PNG).decode()})):
            is_error, text = tool(name, args, token=token)
            if not is_error or "scope" not in str(text):
                return "fail", f"{name} with a read-only token was not refused for scope: {text!r}", {}
        listed = {t["name"] for t in mcp("tools/list", token=token)[1]["result"]["tools"]}
        if "generate_image" in listed or "system_status" not in listed:
            return "fail", f"read-only tools/list shows {sorted(listed)}", {}
        request = urllib.request.Request(f"{base}/api/agent/uploads", data=PNG, method="POST",
                                         headers={"Authorization": f"Bearer {token}"})
        try:
            urllib.request.urlopen(request, timeout=30)
            return "fail", "a read-only token could upload over HTTP", {}
        except urllib.error.HTTPError as exc:
            if exc.code != 403:
                return "fail", f"upload with a read-only token answered HTTP {exc.code}, wanted 403", {}
        return "pass", "", {}

    def jobs_and_queue():
        is_error, listing = tool("list_jobs", {"limit": 5})
        if is_error or "jobs" not in listing:
            return "fail", f"list_jobs: {listing!r}", {}
        is_error, status = tool("system_status")
        queue = status.get("queue") if isinstance(status, dict) else None
        if is_error or not isinstance(queue, dict) or "jobs" not in queue or "estimated" not in str(queue.get("estimateNote", "")):
            return "fail", f"system_status queue view: {queue!r}", {}
        is_error, text = tool("job_status", {"job_id": "job-0"})
        if not is_error or "list_jobs" not in str(text):
            return "fail", f"unknown job message: {text!r}", {}
        return "pass", "", {"recentJobs": len(listing["jobs"]), "queued": len(queue["jobs"])}

    def old_upload_name():
        is_error, out = tool("upload_image", {"data_base64": base64.b64encode(PNG).decode()})
        if is_error or not str(out.get("handle", "")).startswith("upload:"):
            return "fail", f"upload_image alias: {out!r}", {}
        return "pass", "", {}

    def download_refusal():
        # The confirmed model download (A5) is not built yet: no tool for it, and no download scope unless ticked.
        scopes = api("GET", "/api/agent/access")[1]["scopes"]
        if "download" in scopes and "download" not in saved["scopes"]:
            return "fail", "the default token has the download scope", {}
        is_error, text = tool("request_model_download", {"model": "x"})
        if not is_error or "Unknown tool" not in str(text):
            return "fail", f"request_model_download should not exist yet: {text!r}", {}
        return "pass", "", {"note": "no download tool and no download scope by default"}

    try:
        for name, fn in [
            ("every tool has the four annotations; upload_file replaces upload_image", annotations),
            ("main token has read, generate, files", default_token_scopes),
            ("read-only token reads, is refused everything else", read_only_token),
            ("list_jobs, queue view in system_status, unknown job hint", jobs_and_queue),
            ("old upload_image name still works", old_upload_name),
            ("no download tool or scope by default", download_refusal),
        ]:
            phase.checks.append(suite._check(name, fn))
    finally:
        if not saved["readToken"]:
            api("POST", "/api/agent/access", {"readToken": False})
        api("POST", "/api/agent/access", {"enabled": saved["enabled"]})
    fails = [c for c in phase.checks if c.status == "fail"]
    phase.status = "fail" if fails else ("skip" if all(c.status == "skip" for c in phase.checks) else "pass")
    return phase
