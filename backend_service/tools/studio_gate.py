"""Approve-the-plan gate for Chat's studio tools (D-60).

The first time the chat model calls a tool that renders (image, video, speech, music, sound, Storyboard
generate or render, caption transcription) the call is not run. The agent loop hands the reply's tool calls
to ``StudioGate.review``, which emits a plan (``studio_plan`` event: what will be made, model, size,
estimated time and memory) and waits until the user decides in the app (``POST
/api/chat/studio/plans/{id}/decision``). Approval is only that click, never chat text. Declined calls get a
plain "declined" result for the model. A decision covers the listed calls of this turn only; "Always allow
renders in this chat" (off by default) lets later render calls of the turn run, and the app sends it on
later turns of the same chat. Read-only and edit tools never reach the gate.

The turn does not end with its stream (S-J): after a window reload ``pending_for_session`` gives the waiting
plans back to the app, which shows the card again; the click answers the same waiting turn.
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
    tier: str | None = None  # the quality the user picked on the card (FU-241)
    keep_tier: bool = False  # "use for the rest of this chat"

    def payload(self) -> dict[str, Any]:
        return {"planId": self.id, "sessionId": self.session_id, "items": self.items, "memory": self.memory,
                "status": "pending"}


_LOCK = threading.Lock()
_PLANS: dict[str, PendingPlan] = {}


def pending(plan_id: str) -> PendingPlan | None:
    with _LOCK:
        return _PLANS.get(plan_id)


def pending_for_session(session_id: str) -> list[dict[str, Any]]:
    """The plans of a chat that still wait for a click, oldest first: a reloaded window shows them again (S-J)."""
    with _LOCK:
        return [p.payload() for p in _PLANS.values() if p.session_id == session_id and not p.event.is_set()]


# Chats whose studio turn is running (the turn goes on when its stream closes, S-J): session id -> turns.
_TURNS: dict[str, int] = {}


def turn_started(session_id: str) -> None:
    with _LOCK:
        _TURNS[session_id] = _TURNS.get(session_id, 0) + 1


def turn_ended(session_id: str) -> None:
    with _LOCK:
        left = _TURNS.get(session_id, 0) - 1
        if left > 0:
            _TURNS[session_id] = left
        else:
            _TURNS.pop(session_id, None)


def turn_running(session_id: str) -> bool:
    with _LOCK:
        return _TURNS.get(session_id, 0) > 0


def decide(plan_id: str, decisions: dict[str, str], edit: bool = False, always_allow: bool = False,
           tier: str | None = None, keep_tier: bool = False) -> PendingPlan | None:
    """Record the user's click. Unlisted calls are declined. None when the plan is unknown or already decided."""
    with _LOCK:
        plan = _PLANS.get(plan_id)
        if plan is None or plan.event.is_set():
            return None
        listed = {item["callId"] for item in plan.items}
        plan.decisions = {cid: ("approve" if decisions.get(cid) == "approve" and not edit else "deny") for cid in listed}
        plan.edit = bool(edit)
        plan.always_allow = bool(always_allow) and not edit
        plan.tier = tier if tier in ("draft", "standard", "final") and not edit else None
        plan.keep_tier = bool(keep_tier) and plan.tier is not None
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


MAX_ESTIMATE_SHOTS = 24


def storyboard_estimate(args: dict[str, Any], state: Any | None) -> dict[str, Any] | None:
    """S7: time per shot (still and clip) and in total for a ``storyboard_generate``, from the same estimate the
    Storyboard run bar shows (``storyboard.estimate``: this project's measured takes, else the catalogue). None
    when the project cannot be read; a missing model is named in ``missing``, never guessed."""
    from types import SimpleNamespace

    from backend_service.routes.storyboard import _draft_sizes, _draft_variant, _final_model, _project_variant, get_store
    from backend_service.storyboard.batch import needs_work
    from backend_service.storyboard.estimate import estimate, plan_steps

    store = get_store()
    project = store.load(str(args.get("project_id")))
    folder = store.folder(project.id)
    what = str(args.get("what") or "both")
    shot_ids = [str(i) for i in args["shot_ids"]] if args.get("shot_ids") else None
    if shot_ids is None and args.get("force"):  # a forced batch also runs locked shots; the estimate skips them
        shot_ids = [s.id for s in project.shots if needs_work(s, what)]
    steps = plan_steps(project, what, shot_ids)
    if not steps:
        return None
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(chaosengine=state)))
    missing: list[str] = []
    image = video = None
    draft = bool(args.get("draft")) and what != "final"
    if any(kind == "still" for _, kind in steps):
        # The final pass prices the quality model; drafts and the normal run price the project's model.
        model = _final_model(request, project) if what == "final" else project.defaults.imageModel
        image, why = _project_variant(request, "image", model)
        missing += [why] if why else []
        if draft:
            # Drafts run on the fastest installed still model, as the batch does (S-N: the card priced the
            # project's slow model, "about 114 min" for stills that take seconds).
            image = _draft_variant(request, project, image)
    if any(kind == "clip" for _, kind in steps):
        video, why = _project_variant(request, "video", project.defaults.videoModel)
        missing += [why] if why else []
    hardware = None
    try:
        from backend_service.helpers.hardware_class import host_hardware_class

        hardware = host_hardware_class()
    except Exception:  # noqa: BLE001 - the estimate reads it again or goes without memory
        pass

    sizes = _draft_sizes(project, image, video) if draft else None

    def run(kind: str, ids: list[str] | None) -> dict[str, Any]:
        return estimate(project, folder, what=kind, shot_ids=ids, image_variant=image, video_variant=video,
                        missing=missing, hardware=hardware, render_pass="draft" if draft else "final", sizes=sizes)

    total = run(what, shot_ids)
    rows: list[dict[str, Any]] = []
    order: list[str] = []
    kinds: dict[str, set[str]] = {}
    for shot, kind in steps:
        if shot.id not in kinds:
            order.append(shot.id)
            kinds[shot.id] = set()
        kinds[shot.id].add(kind)
    titles = {s.id: s.title for s in project.shots}
    for shot_id in order[:MAX_ESTIMATE_SHOTS]:
        row: dict[str, Any] = {"id": shot_id, "title": titles.get(shot_id) or ""}
        for kind in ("still", "clip"):
            if kind in kinds[shot_id]:
                row[kind] = run(kind, [shot_id])["seconds"]
        rows.append(row)
    return {
        "seconds": total["seconds"],
        "basis": total["basis"],
        "peakGb": total["peakGb"],
        "totalGb": total["totalGb"],
        "missing": missing,
        "imageModel": total["imageModel"],
        "videoModel": total["videoModel"],
        "shots": rows,
        "moreShots": max(0, len(order) - MAX_ESTIMATE_SHOTS),
    }


def plan_item(tool_call: dict[str, Any], state: Any | None, tier: str | None = None) -> dict[str, Any]:
    """One row of the plan card: what will be made and every argument that changes it. Never raises.

    Estimates come from the catalog rows the bridge already reads; ``estimateAtRecommended`` marks a video
    time measured at the recommended size when the call asks for another size or a frame count.
    """
    name, args = call_name(tool_call), call_args(tool_call)
    asked_args = args  # before the tier: the card prices every tier from what the model asked
    quality: list[str] = []
    if tier and state is not None:
        from backend_service.tools.studio_tiers import apply_tier, variant_for

        try:
            args, quality = apply_tier(tier, name, args, variant_for(name, args.get("model"), state))
        except Exception:  # noqa: BLE001 - the card shows the call as asked
            quality = []
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
    if tier and name in ("generate_image", "generate_video"):
        item["tier"] = tier
        if quality:
            item["qualityNotes"] = quality
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
        if name == "storyboard_generate":
            try:
                board = storyboard_estimate(args, state)
            except Exception:  # noqa: BLE001 - the card shows without the times
                board = None
            if board is not None:
                item["estimate"] = board
                item["peakGb"] = board["peakGb"]
                models = [m for m in (board["imageModel"], board["videoModel"]) if m]
                if models and not item["model"]:
                    item["model"] = " + ".join(models)
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
                    _price_settings(item, catalog_kind, row.get("id"), args)
                    if tier and name in ("generate_image", "generate_video"):
                        item["tierOptions"] = _tier_options(name, catalog_kind, row.get("id"), asked_args, state)
    except Exception:  # noqa: BLE001 - an estimate is a nicety; the plan shows without it
        pass
    return item


def _tier_options(name: str, kind: str, model_id: Any, args: dict[str, Any], state: Any) -> dict[str, Any]:
    """Size and time of this call at each tier, for the card's three buttons (FU-241)."""
    from backend_service.tools.studio_tiers import TIERS, apply_tier, variant_for

    variant = variant_for(name, model_id, state)
    options: dict[str, Any] = {}
    for tier in TIERS:
        scratch: dict[str, Any] = {}
        _price_settings(scratch, kind, model_id, apply_tier(tier, name, args, variant)[0])
        options[tier] = {"size": scratch.get("size"), "estimatedSeconds": scratch.get("estimatedSeconds")}
    return options


def _price_settings(item: dict[str, Any], kind: str, model_id: Any, args: dict[str, Any]) -> None:
    """Fill the card with the size, frames, fps and steps that will really run, and a time for exactly those
    settings (FU-240), the way ``generate_image`` and ``generate_video`` resolve them. A model the catalog cannot
    price keeps the catalogue figure and its "at the recommended size" note."""
    from backend_service.helpers.generation_limits import align_video_frames, parse_resolution
    from backend_service.helpers.time_estimate import variant_seconds
    from backend_service.mcp_bridge.tools import _recommended

    if kind == "image":
        from backend_service.helpers.images import _find_image_variant as find
    else:
        from backend_service.helpers.video import _find_video_variant as find
    variant = find(str(model_id))
    if variant is None:
        return
    size = parse_resolution(variant.get("recommendedResolution"))
    width = int(args.get("width") or (size[0] if size else 0))
    height = int(args.get("height") or (size[1] if size else 0))
    steps = args.get("steps", _recommended(variant).get("steps"))
    frames = 1
    if kind == "video":
        fps = int(args.get("fps", 24))
        if "frames" in args:
            frames = int(args["frames"])
        else:
            frames = round(float(args.get("seconds") or variant.get("defaultDurationSeconds") or 4) * fps)
        frames, _note = align_video_frames(str(variant.get("repo") or ""), max(8, min(257, frames)))
        item["frames"], item["fps"] = frames, fps
        if not size and not (width and height):
            return
    if width and height:
        item["size"] = f"{width}x{height}"
    if steps is not None:
        item["steps"] = steps
    seconds = variant_seconds(variant, kind, width, height, frames, steps) if width and height else None
    if seconds:
        item["estimatedSeconds"] = round(seconds)
        item.pop("estimateAtRecommended", None)


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


def _edited_earlier(tool_calls: list[dict[str, Any]], call: dict[str, Any], args: dict[str, Any]) -> bool:
    """An earlier call of the same reply names the same project: it runs first and moves the revision, so the
    revision is checked when the call runs, not before the card."""
    pid = args.get("project_id")
    if not pid:
        return False
    for tc in tool_calls:
        if tc is call:
            return False
        if call_args(tc).get("project_id") == pid:
            return True
    return False


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
        auto_until: float | None = None,
        tier: Any = "standard",
    ) -> None:
        self.session_id = session_id
        # The chat's quality tier (FU-241): the card shows a single render as it will run at this tier.
        self.tier = tier
        self.auto_approve = auto_approve
        # Keep going's run grant (FU-191): auto-approval ends at this epoch time, even mid-turn. None: no end.
        self.auto_until = auto_until
        self.is_cancelled = is_cancelled
        self.state = state
        self.wait_timeout = wait_timeout
        self.poll_seconds = poll_seconds
        self.keepalive_seconds = keepalive_seconds
        self.resolved: list[dict[str, Any]] = []

    def _auto_now(self) -> bool:
        return self.auto_approve and (self.auto_until is None or time.time() < self.auto_until)

    def review(self, tool_calls: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Generator: yields plan events while waiting; returns ``{call_id: answer}`` for calls not to run, where
        the answer is the declined text or a ready result (``StructuredToolOutput``).

        Before a card (S1, FU-179): a render call that would be refused (bad argument, stale revision, nothing to
        generate) gets that refusal at once, and a repeat of a storyboard render that is still running gets that
        job; neither shows on a card. Identical storyboard renders in one reply make one row; the copies follow
        its decision and, once it runs, find its job.
        """
        from backend_service.tools.studio import needs_approval, ready_answer, render_signature

        render = [tc for tc in tool_calls if needs_approval(call_name(tc), call_args(tc))]
        if not render:
            return {}
        answers: dict[str, Any] = {}
        planned: list[dict[str, Any]] = []
        leaders: dict[tuple[Any, ...], str] = {}
        follows: dict[str, str] = {}
        for tc in render:
            name, args = call_name(tc), call_args(tc)
            ready = ready_answer(name, args, check_revision=not _edited_earlier(tool_calls, tc, args))
            if ready is not None:
                answers[str(tc.get("id"))] = ready
                continue
            signature = render_signature(name, args)
            if signature is not None and signature in leaders:
                follows[str(tc.get("id"))] = leaders[signature]
                continue
            if signature is not None:
                leaders[signature] = str(tc.get("id"))
            planned.append(tc)
        if not planned or self._auto_now():
            return answers
        from backend_service.tools.studio_tiers import current

        items = [plan_item(tc, self.state, current(self.tier)) for tc in planned]
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
            # Also when the turn itself is cancelled (Stop) or closed: the plan is gone, and a late click gets a 404.
            # A window reload does not close it: the turn runs on (S-J) and the reloaded window lists the plan.
            with _LOCK:
                _PLANS.pop(plan.id, None)
                plan.event.set()
        denials: dict[str, str] = {}
        for item in items:
            if not answered:
                denials[item["callId"]] = NO_ANSWER_TEXT
            elif plan.decisions.get(item["callId"]) != "approve":
                denials[item["callId"]] = EDIT_TEXT if plan.edit else DECLINED_TEXT
        if answered and plan.tier:
            from backend_service.tools.studio_tiers import TierHolder

            if isinstance(self.tier, TierHolder):
                self.tier.value = plan.tier
            else:
                self.tier = plan.tier
            if plan.keep_tier and self.session_id and self.state is not None:
                chat_tiers = getattr(self.state, "chat_tiers", None)
                if chat_tiers is None:
                    chat_tiers = self.state.chat_tiers = {}
                chat_tiers[self.session_id] = plan.tier
        if answered and plan.always_allow:
            self.auto_approve = True
            self.auto_until = None  # the user's own click on this card: the rest of the turn, as before
        status = "unanswered" if not answered else ("edit" if plan.edit else "decided")
        resolved = {
            **plan.payload(),
            "status": status,
            "decisions": {i["callId"]: ("deny" if i["callId"] in denials else "approve") for i in items},
            "alwaysAllow": bool(answered and plan.always_allow),
            "tier": plan.tier if answered else None,
            "keepTier": bool(answered and plan.keep_tier),
        }
        self.resolved.append(resolved)
        yield {"studio_plan_resolved": resolved}
        for copy_id, leader in follows.items():
            if leader in denials:
                denials[copy_id] = denials[leader]
        return {**answers, **denials}
