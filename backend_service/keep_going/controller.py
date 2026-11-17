"""The Keep going run controller (FU-188): the app, not the model, decides whether a Studio chat goes on.

A run belongs to one chat. Turned on, it is ``armed``: the user's next message is the brief (the goal) and
starts it. After each Studio turn of a ``running`` run the controller reads how the turn ended and the saved
project, then either stops with a plain reason or sends the next nudge: a user-role message the app writes,
through the normal chat path. While a chat job still runs the run is ``waiting`` and nudges once it ends; it
never nudges while a turn, a wait or a plan card is in flight.

Stop reasons (``STOP_CODES``) are shown to the user and logged. The render grant (renders auto-approved for
this run, up to 12 hours) lives only in memory: an app restart drops it with the run's turn, and Resume asks
again. Runs are saved as JSON beside the settings, written atomically; a run that was active when the app
stopped comes back stopped with ``app_restart`` and a Resume button.

Locking: ``self._lock`` guards the run table and every run field. It is never held while a turn starts or the
chat state is read (the chat path calls in here without its own lock, so neither lock waits on the other).
A turn is started outside the lock with an abort event the chat turn checks, so a user message that arrives
while a nudge starts still wins.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from backend_service.keep_going.nudge import (
    TurnOutcome,
    claims_done,
    error_signature,
    finished_job_lines,
    nudge_text,
    plan_outcome,
    project_id_from,
    unfixed_errors,
)
from backend_service.keep_going.adapters import View, adapter_for
from backend_service.keep_going.run import (  # noqa: F401 - re-exported for callers and tests
    ACTIVE,
    CONTEXT_FULL_SHARE,
    DEFAULT_CAP,
    DEFAULT_GRANT_HOURS,
    FILE_VERSION,
    MAX_CAP,
    MAX_GRANT_HOURS,
    MIN_CAP,
    NO_PROGRESS_LIMIT,
    REPEATED_ERROR_LIMIT,
    RESUMABLE,
    STOP_CODES,
    TURN_FAILURE_LIMIT,
    Grant,
    Hooks,
    Run,
    TurnStartError,
    _FINISHED,
    _context_error,
    _finished_ids,
    _gpu_mode,
    _grant,
    _template,
    clamp_cap,
    clamp_hours,
    reason_text,
)


class KeepGoing:
    def __init__(
        self,
        path: Path | None,
        hooks: Hooks,
        *,
        clock: Callable[[], float] = time.time,
        threaded: bool = True,
        poll_seconds: float = 2.0,
        adapters: Callable[[str], Any] = adapter_for,
    ) -> None:
        self.adapters = adapters
        self.path = path
        self.hooks = hooks
        self.clock = clock
        self.threaded = threaded
        self.poll_seconds = poll_seconds
        self._lock = threading.RLock()
        self._runs: dict[str, Run] = {}
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        runs = data.get("runs") if isinstance(data, dict) else None
        changed = False
        for sid, record in (runs or {}).items() if isinstance(runs, dict) else ():
            try:
                run = Run.from_record(record)
            except (TypeError, ValueError):
                continue
            if run.state in ACTIVE:
                # Nothing of a run survives a restart but its record: it waits for Resume (never restarts itself).
                run.state = "stopped"
                run.stop_code = "app_restart"
                run.stop_reason = reason_text("app_restart")
                run.stopped_at = self.clock()
                changed = True
            self._runs[str(sid)] = run
        if changed:
            self._save()

    def _save(self) -> None:
        if self.path is None:
            return
        from backend_service.storyboard.store import atomic_write

        payload = {"version": FILE_VERSION, "runs": {sid: run.record() for sid, run in self._runs.items()}}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(self.path, json.dumps(payload, indent=1, default=str).encode("utf-8"))
        except OSError as exc:
            self.hooks.log("warning", f"Keep going: could not save runs: {exc}")

    # -- reading -----------------------------------------------------------

    def get(self, session_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(session_id)

    def status(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(session_id)
            if run is None:
                return None
            view = run.record()
            grant = run.grant
            dispatching = run.dispatching
        now = self.clock()
        turn = self.hooks.turn_running(session_id)
        pending = self.hooks.pending_plans(session_id) if turn else 0
        if view["state"] == "running":
            phase = ("awaiting_approval" if pending else "turn") if turn else ("nudging" if dispatching else "turn")
        elif view["state"] == "waiting":
            phase = "waiting_job"
        else:
            phase = view["state"]
        return {
            "runId": view["id"],
            "sessionId": session_id,
            "state": view["state"],
            "phase": phase,
            "nudges": view["nudges"],
            "cap": view["cap"],
            "goal": (view["goal"] or "")[:200],
            "projectId": view["project_id"],
            "startedAt": view["started_at"],
            "updatedAt": view["updated_at"],
            "noProgress": view["no_progress"],
            "stopCode": view["stop_code"],
            "stopReason": view["stop_reason"],
            "stopDetail": view["stop_detail"],
            "stoppedAt": view["stopped_at"],
            "resumable": view["state"] == "stopped" and view["stop_code"] in RESUMABLE and bool(view["template"]),
            "grant": None if grant is None else {
                "mode": grant.mode,
                "until": grant.until,
                "secondsLeft": max(0, int(grant.until - now)) if grant.until else None,
            },
            "gpuMode": view["gpu_mode"],
            "turnRunning": turn,
            "pendingPlans": pending,
            "messageCount": self.hooks.message_count(session_id),
        }

    # -- user actions ------------------------------------------------------

    def start(self, session_id: str, *, cap: Any = DEFAULT_CAP, grant_mode: str = "manual",
              hours: Any = DEFAULT_GRANT_HOURS, app_request: Any = None, gpu_mode: str = "auto") -> Run:
        """Turn Keep going on for a chat. Idempotent: an active run is returned unchanged (a reloaded page or a
        double click never starts a second one)."""
        turn = self.hooks.turn_running(session_id)
        last_user = self.hooks.last_user_text(session_id) if turn else None
        jobs = self.hooks.chat_jobs()
        with self._lock:
            run = self._runs.get(session_id)
            if run is not None and run.state in ACTIVE:
                return run
            now = self.clock()
            run = Run(id=f"kg-{uuid.uuid4().hex[:10]}", session_id=session_id, cap=clamp_cap(cap),
                      started_at=now, updated_at=now, gpu_mode=_gpu_mode(gpu_mode))
            run.grant = _grant(grant_mode, hours, now)
            run.app_request = app_request
            run.seen_jobs = _finished_ids(jobs)
            if turn and last_user:
                # Turned on while the brief's own turn runs: that message is the goal, its end is the first check.
                run.state = "running"
                run.goal = last_user
                run.turn_pending = True
            self._runs[session_id] = run
            self._save()
        self.hooks.log("info", f"Keep going on for chat {session_id} ({run.id}, cap {run.cap}, "
                               f"renders {'auto-approved' if run.grant and run.grant.mode == 'auto' else 'manual'}).")
        return run

    def stop(self, session_id: str, code: str = "user_stopped", detail: str | None = None,
             *, cancel_turn: bool = True) -> Run | None:
        with self._lock:
            run = self._runs.get(session_id)
            if run is None or run.state not in ACTIVE:
                return run
            turn_pending = run.turn_pending
            self._stop_locked(run, code, detail)
        if cancel_turn and turn_pending and self.hooks.turn_running(session_id):
            self.hooks.cancel_turn(session_id)
        return run

    def resume(self, session_id: str, *, grant_mode: str = "manual", hours: Any = DEFAULT_GRANT_HOURS,
               cap: Any = None, app_request: Any = None) -> Run:
        """Pick a stopped run up again: counters start over, the grant is asked again, and the next nudge
        rebuilds the context from the brief plus the saved project."""
        with self._lock:
            run = self._runs.get(session_id)
            if run is None:
                raise LookupError("No Keep going run for this chat.")
            if run.state in ACTIVE:
                return run
            if run.stop_code not in RESUMABLE or not run.template:
                raise ValueError("This run cannot be resumed. Turn Keep going on again with a new message.")
            now = self.clock()
            run.state = "running"
            run.cap = clamp_cap(cap) if cap is not None else run.cap
            run.nudges = 0
            run.no_progress = 0
            run.error_streaks = {}
            run.turn_failures = 0
            run.stop_code = run.stop_reason = run.stop_detail = None
            run.stopped_at = None
            run.resumes += 1
            run.updated_at = now
            run.grant = _grant(grant_mode, hours, now)
            run.app_request = app_request or run.app_request
            run.abort = threading.Event()
            run.turn_pending = False
            run.nudge_why = "resume"
            run.snapshot = None
            self._save()
        self.hooks.log("info", f"Keep going resumed for chat {session_id} ({run.id}).")
        self._schedule(session_id)
        return run

    def forget(self, session_id: str) -> None:
        """The chat was deleted: stop its run and drop the record."""
        self.stop(session_id, "user_stopped")
        with self._lock:
            if self._runs.pop(session_id, None) is not None:
                self._save()

    # -- chat path hooks ---------------------------------------------------

    def on_user_turn(self, session_id: str, request_fields: dict[str, Any], prompt: str,
                     app_request: Any = None) -> float | None:
        """A message the user sent. An armed run takes it as the brief and starts; an active run stops (the
        user's message wins). Returns the time until which this turn's renders are auto-approved, or None."""
        jobs = self.hooks.chat_jobs()
        with self._lock:
            run = self._runs.get(session_id)
            if run is None or run.state not in ACTIVE:
                return None
            now = self.clock()
            if run.state == "armed":
                if not request_fields.get("studioTools"):
                    return None  # only a Studio turn can be the brief; the run stays armed
                run.seen_jobs = _finished_ids(jobs)
                run.state = "running"
                run.goal = prompt
                run.template = _template(request_fields)
                run.app_request = app_request
                run.turn_pending = True
                run.updated_at = now
                self._save()
                self.hooks.log("info", f"Keep going started in chat {session_id}: the brief is this message.")
                return run.grant.until if run.grant and run.grant.live(now) else None
            # A nudge turn that is starting or running ends through its abort event (set by the stop), not the
            # chat's cancel flag: the user's own turn clears that flag when it starts.
            self._stop_locked(run, "user_message", None)
        return None

    def on_chat_cancel(self, session_id: str) -> None:
        """Stop in the chat: the run ends too (the turn is already being cancelled)."""
        self.stop(session_id, "user_stopped", cancel_turn=False)

    def on_turn_finished(self, outcome: TurnOutcome) -> None:
        """A Studio turn of this chat ended (saved or failed). Decide: stop with a reason, wait, or nudge."""
        sid = outcome.session_id
        run_id = outcome.extra.get("runId")
        with self._lock:
            run = self._runs.get(sid)
            if run is None or run.state != "running" or not run.turn_pending:
                return
            if run_id is not None and run_id != run.id:
                return  # a turn of an older run
            run.turn_pending = False
            project_id = project_id_from(outcome) or run.project_id
        # Read the work outside the lock (disk); decide inside it.
        view = self.adapters(run.adapter).read(goal=run.goal, project_id=project_id, outcome=outcome,
                                                last_reply=outcome.text)
        jobs = self.hooks.chat_jobs()
        with self._lock:
            if run.state != "running":
                return
            run.project_id = view.project_id or project_id
            run.last_reply = outcome.text[-1500:]
            run.updated_at = self.clock()
            decision = self._decide(run, outcome, view, jobs)
            if decision is not None:
                code, detail = decision
                self._stop_locked(run, code, detail)
                return
            self._save()
        self._schedule(sid)

    # -- the decision ------------------------------------------------------

    def _decide(self, run: Run, outcome: TurnOutcome, view: View,
                jobs: list[Any]) -> tuple[str, str | None] | None:
        """Called with the lock held. A stop (code, detail), or None to go on (``run.nudge_why`` says why)."""
        if outcome.cancelled:
            return "user_stopped", None
        if outcome.error:
            if outcome.extra.get("modelMissing"):
                return "model_unavailable", outcome.error
            if _context_error(outcome.error):
                return "context_full", outcome.error
            run.turn_failures += 1
            if run.turn_failures >= TURN_FAILURE_LIMIT:
                return "turn_failed", outcome.error
            run.last_error = ["chat", outcome.error]
            run.nudge_why = "error"
            return None
        run.turn_failures = 0
        plan = plan_outcome(outcome)
        if plan is not None:
            return {"declined": "plan_declined", "edit": "plan_edit", "unanswered": "plan_unanswered"}[plan], None
        if outcome.context_tokens and outcome.prompt_tokens >= CONTEXT_FULL_SHARE * outcome.context_tokens:
            return "context_full", f"{outcome.prompt_tokens} of {outcome.context_tokens} tokens"
        # Errors the model left unfixed: one nudge with the text; the same one three times stops the run.
        errors = unfixed_errors(outcome)
        counts: dict[str, int] = {}
        for tool, text in errors:
            sig = error_signature(tool, text)
            counts[sig] = counts.get(sig, 0) + 1
        run.error_streaks = {sig: run.error_streaks.get(sig, 0) + n for sig, n in counts.items()}
        for sig, streak in run.error_streaks.items():
            if streak >= REPEATED_ERROR_LIMIT:
                return "repeated_error", sig.split(":", 1)[1] if ":" in sig else sig
        run.last_error = list(errors[-1]) if errors else None
        # Progress since the last nudge: a project edit, a take, audio, a render, a finished chat job.
        snapshot = {"work": view.signature, "jobs": _finished_ids(jobs)}
        if run.nudges > 0 and run.snapshot is not None:
            if snapshot == run.snapshot:
                run.no_progress += 1
                if run.no_progress >= NO_PROGRESS_LIMIT:
                    return "no_progress", None
            else:
                run.no_progress = 0
        run.snapshot = snapshot
        running_job = any(getattr(j, "state", "") in ("queued", "running") for j in jobs)
        if view.complete and not running_job and not errors:
            return view.complete, None
        if run.nudges >= run.cap:
            return "cap_reached", None
        if run.grant is not None and run.grant.mode == "auto" and not run.grant.live(self.clock()):
            return "approval_expired", None
        if errors:
            run.nudge_why = "error"
        elif outcome.max_iterations:
            run.nudge_why = "cap"
        elif claims_done(outcome.text):
            run.nudge_why = "claimed"
        else:
            run.nudge_why = "stopped"
        return None

    def _stop_locked(self, run: Run, code: str, detail: str | None) -> None:
        now = self.clock()
        run.state = "stopped"
        run.stop_code = code
        run.stop_detail = (detail or "")[:600] or None
        run.stop_reason = reason_text(code, cap=run.cap, detail=detail)
        run.stopped_at = run.updated_at = now
        run.grant = None  # the grant ends the moment the run stops
        run.turn_pending = False
        run.abort.set()
        self._save()
        self.hooks.log("info" if code in ("complete", "complete_no_project", "user_stopped", "user_message")
                       else "warning", f"Keep going stopped in chat {run.session_id}: {run.stop_reason}")

    # -- nudging -----------------------------------------------------------

    def _start_failed(self, run: Run, code: str, detail: str) -> None:
        with self._lock:
            run.nudges = max(0, run.nudges - 1)  # nothing was sent
            run.turn_pending = False
            if run.state in ACTIVE:  # a user message may have stopped it meanwhile; keep that reason
                self._stop_locked(run, code, detail)

    def _schedule(self, session_id: str) -> None:
        if self.threaded:
            threading.Thread(target=self._dispatch, args=(session_id,), name=f"keep-going-{session_id}",
                             daemon=True).start()
        else:
            self._dispatch(session_id)

    def _blocked(self, session_id: str) -> str | None:
        """Why a nudge must wait now: a chat turn, a plan card or a chat job is in flight."""
        if self.hooks.turn_running(session_id):
            return "turn"
        if self.hooks.pending_plans(session_id):
            return "plan"
        if any(getattr(j, "state", "") in ("queued", "running") for j in self.hooks.chat_jobs()):
            return "job"
        return None

    def _dispatch(self, session_id: str) -> None:
        blocked = self._blocked(session_id)
        with self._lock:
            run = self._runs.get(session_id)
            if run is None or run.state not in ("running", "waiting") or run.turn_pending or run.dispatching:
                return
            if blocked:
                if run.state != "waiting":
                    run.state = "waiting"
                    self._save()
                start_watch = not run.waiting_thread and self.threaded
                if start_watch:
                    run.waiting_thread = True
            else:
                start_watch = False
        if blocked:
            if start_watch:
                threading.Thread(target=self._watch, args=(session_id,), name=f"keep-going-wait-{session_id}",
                                 daemon=True).start()
            return
        self._send_nudge(session_id)

    def poll_waiting(self) -> None:
        """One pass over waiting runs (the watcher thread's body; tests call it directly)."""
        with self._lock:
            waiting = [sid for sid, r in self._runs.items() if r.state == "waiting"]
        for sid in waiting:
            if self._blocked(sid) is None:
                self._send_nudge(sid)

    def _watch(self, session_id: str) -> None:
        try:
            while True:
                time.sleep(self.poll_seconds)
                with self._lock:
                    run = self._runs.get(session_id)
                    if run is None or run.state != "waiting":
                        return
                if self._blocked(session_id) is None:
                    self._send_nudge(session_id)
                    return
        finally:
            with self._lock:
                run = self._runs.get(session_id)
                if run is not None:
                    run.waiting_thread = False

    def _send_nudge(self, session_id: str) -> None:
        with self._lock:
            run = self._runs.get(session_id)
            if run is None or run.state not in ("running", "waiting") or run.turn_pending or run.dispatching:
                return
            now = self.clock()
            if run.grant is not None and run.grant.mode == "auto" and not run.grant.live(now):
                self._stop_locked(run, "approval_expired", None)
                return
            if run.nudges >= run.cap:
                self._stop_locked(run, "cap_reached", None)
                return
            run.dispatching = True
            run.state = "running"
            project_id, goal, why = run.project_id, run.goal, run.nudge_why
            last_error = tuple(run.last_error) if run.last_error else None
            seen = set(run.seen_jobs)
        # Gather outside the lock: the project on disk, finished jobs, whether the brief is still in the history.
        try:
            view = self.adapters(run.adapter).read(goal=goal, project_id=project_id, outcome=None,
                                                    last_reply=run.last_reply)
            jobs = self.hooks.chat_jobs()
            include_goal = why == "resume" or not self.hooks.goal_in_history(run)
        except Exception as exc:  # noqa: BLE001 - a failed read must not leave the run stuck in "dispatching"
            with self._lock:
                run.dispatching = False
                self._stop_locked(run, "turn_failed", f"could not read the run state: {exc}")
            return
        finished = [j for j in jobs if getattr(j, "state", "") in _FINISHED and str(getattr(j, "id", "")) not in seen]
        with self._lock:
            if run.state != "running" or self._runs.get(session_id) is not run:
                run.dispatching = False
                return
            run.nudges += 1
            number = run.nudges
            run.seen_jobs = sorted(seen | set(_finished_ids(jobs)))[-200:]
            if run.snapshot is None:  # after a Resume: progress counts from here
                run.snapshot = {"work": view.signature, "jobs": _finished_ids(jobs)}
            prompt = nudge_text(number=number, cap=run.cap, goal=goal, include_goal=include_goal,
                                summary=view.summary, next_step=view.next_step,
                                last_error=last_error, finished_jobs=finished_job_lines(finished), why=why)
            run.turn_pending = True
            run.abort = threading.Event()
            auto_until = run.grant.until if run.grant and run.grant.live(self.clock()) else None
            run.updated_at = self.clock()
            self._save()
        self.hooks.log("info", f"Keep going nudge {number} of {run.cap} in chat {session_id} ({why}).")
        try:
            self.hooks.start_turn(run, prompt, auto_until)
        except TurnStartError as exc:
            self._start_failed(run, exc.code, exc.detail)
        except Exception as exc:  # noqa: BLE001 - never loop on a turn that cannot start
            self._start_failed(run, "turn_failed", f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                run.dispatching = False
