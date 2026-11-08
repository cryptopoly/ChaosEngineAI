"""Approve-the-plan gate for Chat's studio tools (D-60).

The first time the chat model calls a tool that renders (image, video, speech, music, sound, Storyboard
generate or render, caption transcription) the call is not run. The agent loop hands the reply's tool calls
to ``StudioGate.review``, which emits a plan (``studio_plan`` event: what will be made, model, size,
estimated time and memory) and waits until the user decides in the app (``POST
/api/chat/studio/plans/{id}/decision``). Approval is only that click, never chat text. Declined calls get a
plain "declined" result for the model. A decision covers the listed calls of this turn only; "Always allow
renders in this chat" (off by default) lets later render calls of the turn run, and the app sends it on
later turns of the same chat. Read-only and edit tools never reach the gate.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

DECLINED_TEXT = (
    "Declined: the user did not approve this call, so nothing was made. Do not call it again; "
    "ask the user what to change."
)
EDIT_TEXT = (
    "Declined: the user wants to change the plan before anything is made, so nothing was made. "
    "Stop calling tools and ask what to change."
)
NO_ANSWER_TEXT = "Declined: the plan was not answered in the app, so nothing was made."

WAIT_TIMEOUT_SECONDS = 30 * 60
KEEPALIVE_SECONDS = 10.0
_KINDS = {
    "generate_image": "image",
    "generate_video": "video",
    "speak": "speech",
    "make_music": "music",
    "make_sound": "sound",
    "storyboard_generate": "storyboard",
    "storyboard_render": "film",
    "storyboard_captions": "captions",
}


def call_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = (tool_call.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def call_name(tool_call: dict[str, Any]) -> str:
    return str((tool_call.get("function") or {}).get("name") or "")


@dataclass
class PendingPlan:
    id: str
    session_id: str | None
    items: list[dict[str, Any]]
    memory: dict[str, Any] | None
    event: threading.Event = field(default_factory=threading.Event)
    decisions: dict[str, str] = field(default_factory=dict)
    edit: bool = False
    always_allow: bool = False

    def payload(self) -> dict[str, Any]:
        return {"planId": self.id, "sessionId": self.session_id, "items": self.items, "memory": self.memory,
                "status": "pending"}


_LOCK = threading.Lock()
_PLANS: dict[str, PendingPlan] = {}


def pending(plan_id: str) -> PendingPlan | None:
    with _LOCK:
        return _PLANS.get(plan_id)


def decide(plan_id: str, decisions: dict[str, str], edit: bool = False, always_allow: bool = False) -> PendingPlan | None:
    """Record the user's click. Unlisted calls are declined. None when the plan is unknown or already decided."""
    with _LOCK:
        plan = _PLANS.get(plan_id)
        if plan is None or plan.event.is_set():
            return None
        listed = {item["callId"] for item in plan.items}
        plan.decisions = {cid: ("approve" if decisions.get(cid) == "approve" and not edit else "deny") for cid in listed}
        plan.edit = bool(edit)
        plan.always_allow = bool(always_allow) and not edit
        plan.event.set()
        return plan


def _short(text: Any, limit: int = 160) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _model_row(state: Any, kind: str, model: str | None) -> dict[str, Any] | None:
    from backend_service.mcp_bridge import models as bridge_models

    rows = bridge_models.installed_models(state, kind)
    if model:
        return next((r for r in rows if str(r.get("id")) == str(model)), None)
    return bridge_models.default_model(state, kind)


def _file_names(args: dict[str, Any]) -> list[str]:
    """Input images a call names (shown by file name on the card)."""
    names: list[str] = []
    for key in ("reference_images", "start_image", "end_image"):
        raw = args.get(key)
        for value in raw if isinstance(raw, list) else ([raw] if raw else []):
            text = str(value)
            names.append(text if text.startswith("upload:") else text.replace("\\", "/").rsplit("/", 1)[-1])
    return names


def _storyboard_shots(args: dict[str, Any], render: bool) -> int | None:
    """How many shots a storyboard_generate will work on (or a render will join); None when unreadable."""
    try:
        from backend_service.routes.storyboard import get_store
        from backend_service.storyboard.batch import needs_work

        project = get_store().load(str(args.get("project_id")))
    except Exception:  # noqa: BLE001 - the plan shows without the count
        return None
    if render:
        return len(project.shots)
    if args.get("shot_ids"):
        return len(list(args["shot_ids"]))
    return sum(1 for s in project.shots if needs_work(s, str(args.get("what") or "both")) and (args.get("force") or not s.locked))


def plan_item(tool_call: dict[str, Any], state: Any | None) -> dict[str, Any]:
    """One row of the plan card: what will be made and every argument that changes it. Never raises.

    Estimates come from the catalog rows the bridge already reads; ``estimateAtRecommended`` marks a video
    time measured at the recommended size when the call asks for another size or a frame count.
    """
    name, args = call_name(tool_call), call_args(tool_call)
    prompt = args.get("prompt") or args.get("text") or args.get("caption") or ""
    item: dict[str, Any] = {
        "callId": str(tool_call.get("id") or ""),
        "tool": name,
        "kind": _KINDS.get(name, name),
        "prompt": _short(prompt),
        "promptFull": _short(prompt, 4000) if len(str(prompt)) > 160 else None,
        "model": args.get("model") or args.get("modelId") or args.get("voice"),
        "size": None,
        "estimatedSeconds": None,
        "peakGb": None,
        "inputs": _file_names(args),
    }
    for key in ("variations", "frames", "steps", "fps", "lyrics"):
        if key in args:
            item[key] = _short(args[key], 200) if key == "lyrics" else args[key]
    if args.get("instrumental"):
        item["instrumental"] = True
    if name.startswith("storyboard_"):
        item["project"] = args.get("project_id")
        if args.get("what"):
            item["what"] = args["what"]
        if args.get("force"):
            item["force"] = True
        if args.get("preview"):
            item["preview"] = True
        if name in ("storyboard_generate", "storyboard_render"):
            item["shots"] = _storyboard_shots(args, render=name == "storyboard_render")
    if args.get("width") and args.get("height"):
        item["size"] = f"{args['width']}x{args['height']}"
    if args.get("seconds"):
        item["size"] = f"{args['seconds']} s" if not item["size"] else f"{item['size']}, {args['seconds']} s"
    if state is None:
        return item
    try:
        catalog_kind = {"image": "image", "video": "video", "music": "music"}.get(item["kind"])
        if catalog_kind:
            row = _model_row(state, catalog_kind, args.get("model"))
            if row:
                item["model"] = row.get("name") or row.get("id")
                item["peakGb"] = row.get("peakGb") or row.get("sizeGb")
                if catalog_kind == "music":
                    per_10s = row.get("secondsPer10sAudio")
                    if per_10s:
                        takes = int(args.get("variations", 1) or 1)
                        item["estimatedSeconds"] = round(float(args.get("seconds", 30)) / 10 * float(per_10s) * takes)
                else:
                    item["estimatedSeconds"] = row.get("estimatedSeconds")
                    recommended = str(row.get("recommendedResolution") or "")
                    if not item["size"] and recommended:
                        item["size"] = recommended
                    asked = f"{args.get('width')}x{args.get('height')}" if args.get("width") and args.get("height") else recommended
                    if item["estimatedSeconds"] and (asked != recommended or "frames" in args or "seconds" in args):
                        item["estimateAtRecommended"] = True
    except Exception:  # noqa: BLE001 - an estimate is a nicety; the plan shows without it
        pass
    return item


def memory_note(items: list[dict[str, Any]], state: Any | None) -> dict[str, Any] | None:
    """Free memory and the loaded chat model (the card says '<model> stays loaded: about N GB free'), plus the
    busy-machine advice for the heaviest item."""
    try:
        import psutil

        from backend_service.helpers.busy_advice import DEFAULT_JOB_PEAK_GB, busy_advice, running_jobs_now
        from backend_service.helpers.memory_gate import available_memory_gb

        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024 ** 3)
        free_gb = available_memory_gb(memory.total, memory.available)
        kinds = {"speech": "voice", "sound": "music", "film": "video", "storyboard": "video", "captions": "voice"}
        peaks = [
            float(i["peakGb"]) if isinstance(i.get("peakGb"), (int, float)) and i["peakGb"] > 0
            else DEFAULT_JOB_PEAK_GB.get(kinds.get(i["kind"], i["kind"]), 8.0)
            for i in items
        ]
        advice = busy_advice(running_jobs_now(), max(peaks) if peaks else None, total_gb, free_gb)
        loaded = getattr(getattr(state, "runtime", None), "loaded_model", None) if state is not None else None
        chat_model = getattr(loaded, "name", None)
        return {
            "chatModel": chat_model,
            "freeGb": round(free_gb, 1),
            "totalGb": round(total_gb, 1),
            "peakGb": round(max(peaks), 1) if peaks else None,
            "level": advice["level"],
            "advice": advice["message"] or None,
        }
    except Exception:  # noqa: BLE001 - never block a plan on a memory read
        return None


class StudioGate:
    """Holds render calls of one chat turn until the user decides in the app."""

    def __init__(
        self,
        session_id: str | None,
        *,
        auto_approve: bool = False,
        is_cancelled: Callable[[], bool] = lambda: False,
        state: Any | None = None,
        wait_timeout: float = WAIT_TIMEOUT_SECONDS,
        poll_seconds: float = 0.25,
        keepalive_seconds: float = KEEPALIVE_SECONDS,
    ) -> None:
        self.session_id = session_id
        self.auto_approve = auto_approve
        self.is_cancelled = is_cancelled
        self.state = state
        self.wait_timeout = wait_timeout
        self.poll_seconds = poll_seconds
        self.keepalive_seconds = keepalive_seconds
        self.resolved: list[dict[str, Any]] = []

    def review(self, tool_calls: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Generator: yields plan events while waiting; returns ``{call_id: declined text}`` for calls not to run."""
        from backend_service.tools.studio import needs_approval

        render = [tc for tc in tool_calls if needs_approval(call_name(tc), call_args(tc))]
        if not render or self.auto_approve:
            return {}
        items = [plan_item(tc, self.state) for tc in render]
        plan = PendingPlan(id=f"plan-{uuid.uuid4().hex[:10]}", session_id=self.session_id, items=items,
                           memory=memory_note(items, self.state))
        with _LOCK:
            _PLANS[plan.id] = plan
        answered = True
        try:
            yield {"studio_plan": plan.payload()}
            started = last_ping = time.monotonic()
            while not plan.event.wait(self.poll_seconds):
                now = time.monotonic()
                if self.is_cancelled() or now - started > self.wait_timeout:
                    answered = False
                    break
                if now - last_ping >= self.keepalive_seconds:
                    last_ping = now
                    yield {"keepalive": True}
        finally:
            # Also on a closed stream (window reload): the plan is gone, and a late click gets a 404.
            with _LOCK:
                _PLANS.pop(plan.id, None)
                plan.event.set()
        denials: dict[str, str] = {}
        for item in items:
            if not answered:
                denials[item["callId"]] = NO_ANSWER_TEXT
            elif plan.decisions.get(item["callId"]) != "approve":
                denials[item["callId"]] = EDIT_TEXT if plan.edit else DECLINED_TEXT
        if answered and plan.always_allow:
            self.auto_approve = True
        status = "unanswered" if not answered else ("edit" if plan.edit else "decided")
        resolved = {
            **plan.payload(),
            "status": status,
            "decisions": {i["callId"]: ("deny" if i["callId"] in denials else "approve") for i in items},
            "alwaysAllow": bool(answered and plan.always_allow),
        }
        self.resolved.append(resolved)
        yield {"studio_plan_resolved": resolved}
        return denials
