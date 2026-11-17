"""The controller's view of the running app (FU-188): starting a nudge turn through the normal chat path, and
reading turns, plan cards, chat jobs, the project and the thread. Tests replace this with fakes."""

from __future__ import annotations

from typing import Any

from backend_service.keep_going.run import Run, TurnStartError

# A nudge is about this long; the history check reserves room for it like the chat turn does.
_NUDGE_CHARS = 1200


class AppHooks:
    def __init__(self, state: Any) -> None:
        self.state = state

    def start_turn(self, run: Run, prompt: str, auto_until: float | None) -> None:
        """Send ``prompt`` as the next user message of the chat, as its own Studio turn on a background thread
        (the same ``generate_stream`` the composer uses; a window that reloads sees it like any detached turn)."""
        from fastapi import HTTPException

        from backend_service.models import GenerateRequest

        if run.abort.is_set():
            raise TurnStartError("user_message", "")
        if run.app_request is None:
            raise TurnStartError("turn_failed", "the run has no request context; press Resume")
        fields = {**run.template, "prompt": prompt, "sessionId": run.session_id, "studioTools": True,
                  "studioAutoApprove": auto_until is not None}
        try:
            request = GenerateRequest(**fields)
        except Exception as exc:  # noqa: BLE001 - a saved template the model no longer accepts
            raise TurnStartError("turn_failed", f"the saved chat settings are not valid any more: {exc}") from exc
        turn = {"runId": run.id, "nudge": run.nudges, "cap": run.cap, "abort": run.abort, "autoUntil": auto_until}
        try:
            self.state.generate_stream(request, app_request=run.app_request, keep_going=turn, background=True)
        except HTTPException as exc:
            detail = exc.detail.get("message") if isinstance(exc.detail, dict) else str(exc.detail)
            code = "model_unavailable" if exc.status_code == 409 else "turn_failed"
            raise TurnStartError(code, str(detail)) from exc
        except RuntimeError as exc:  # the model would not load again
            raise TurnStartError("model_unavailable", str(exc)) from exc

    def turn_running(self, session_id: str) -> bool:
        from backend_service.tools.studio_gate import turn_running

        return turn_running(session_id)

    def pending_plans(self, session_id: str) -> int:
        from backend_service.tools.studio_gate import pending_for_session

        return len(pending_for_session(session_id))

    def chat_jobs(self) -> list[Any]:
        from backend_service.mcp_bridge.jobs import MAX_KEPT, STORE
        from backend_service.tools.studio import CHAT_OWNER

        return [j for j in STORE.list(None, MAX_KEPT) if j.owner == CHAT_OWNER]

    def _session(self, session_id: str) -> dict[str, Any] | None:
        return next((s for s in self.state.chat_sessions if s.get("id") == session_id), None)

    def goal_in_history(self, run: Run) -> bool:
        """Whether the next turn's history window still holds the brief (else the nudge repeats it)."""
        from backend_service.state._helpers import _build_history_with_reasoning, _history_token_budget
        from backend_service.tools.studio import STUDIO_HINT

        with self.state._lock:
            session = self._session(run.session_id)
            messages = list(session["messages"]) if session else []
            loaded = self.state.runtime.loaded_model
            context = (run.template.get("contextTokens") or getattr(loaded, "contextTokens", None)
                       or self.state._launch_preferences().get("contextTokens") or 8192)
        budget = _history_token_budget(
            context_tokens=int(context), max_tokens=int(run.template.get("maxTokens") or 4096),
            system_prompt=f"{run.template.get('systemPrompt') or ''}\n\n{STUDIO_HINT}", prompt="x" * _NUDGE_CHARS)
        history = _build_history_with_reasoning(messages, preserve_reasoning=False, token_budget=budget)
        goal = run.goal.strip()
        return any(str(m.get("text") or "").strip() == goal for m in history if m.get("role") == "user")

    def last_user_text(self, session_id: str) -> str | None:
        with self.state._lock:
            session = self._session(session_id)
            for message in reversed(session["messages"] if session else []):
                if message.get("role") == "user":
                    return str(message.get("text") or "")
        return None

    def message_count(self, session_id: str) -> int:
        with self.state._lock:
            session = self._session(session_id)
            return len(session["messages"]) if session else 0

    def cancel_turn(self, session_id: str) -> None:
        self.state.request_cancel_chat(session_id)

    def log(self, level: str, text: str) -> None:
        try:
            self.state.add_log("chat", level, text)
        except Exception:  # noqa: BLE001 - logging never breaks a run
            pass
