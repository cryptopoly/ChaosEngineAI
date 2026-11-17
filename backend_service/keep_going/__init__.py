"""Keep going for Studio chats (FU-191): an opt-in, per-chat run the app drives turn after turn.

The controller (``controller.KeepGoing``) decides after every Studio turn whether to stop with a plain reason
or send the next nudge, a user-role message the app writes from the saved project. ``for_state`` returns the
app's one controller; the chat path calls ``on_user_turn``, ``on_turn_finished`` and ``on_chat_cancel``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from backend_service.keep_going.controller import KeepGoing
from backend_service.keep_going.nudge import TurnOutcome

RUNS_FILE = "keep_going_runs.json"
_LOCK = threading.Lock()


def for_state(state: Any) -> KeepGoing:
    """The controller for this app state, created on first use. Its runs file sits beside the chat sessions in
    the app data folder; a run that was active when the app stopped loads as stopped, with Resume."""
    controller = getattr(state, "_keep_going", None)
    if controller is not None:
        return controller
    with _LOCK:
        controller = getattr(state, "_keep_going", None)
        if controller is None:
            from backend_service.keep_going.hooks import AppHooks

            from backend_service.keep_going.gpu_swap import AppSwap

            path = Path(state._chat_sessions_path).parent / RUNS_FILE
            controller = KeepGoing(path, AppHooks(state), swap=AppSwap(state))
            state._keep_going = controller
    return controller


def on_user_turn(state: Any, session_id: str | None, request_fields: dict[str, Any], prompt: str,
                 app_request: Any = None) -> float | None:
    if not session_id:
        return None
    return for_state(state).on_user_turn(session_id, request_fields, prompt, app_request)


def on_turn_finished(state: Any, outcome: TurnOutcome) -> None:
    try:
        for_state(state).on_turn_finished(outcome)
    except Exception as exc:  # noqa: BLE001 - a controller bug must never break the chat turn that ended
        try:
            state.add_log("chat", "error", f"Keep going failed after a turn: {type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            pass


def defer_render(state: Any, session_id: str, name: str, args: dict[str, Any]) -> Any:
    """An approved render in a run's turn: kept for after the turn (the answer for the model) or None to run now."""
    from backend_service.keep_going.gpu_swap import DEFERRED_TEXT
    from backend_service.tools import StructuredToolOutput

    try:
        kept = for_state(state).defer_render(session_id, name, args)
    except Exception as exc:  # noqa: BLE001 - a swap bug never blocks the render: it runs as before
        state.add_log("chat", "warning", f"Keep going could not plan a GPU swap: {exc}")
        return None
    if not kept:
        return None
    return StructuredToolOutput(text=DEFERRED_TEXT, render_as="studio",
                                data={"tool": name, "deferred": True, "next": DEFERRED_TEXT})


def on_chat_cancel(state: Any, session_id: str) -> None:
    for_state(state).on_chat_cancel(session_id)


__all__ = ["KeepGoing", "TurnOutcome", "for_state", "on_chat_cancel", "on_turn_finished", "on_user_turn"]
