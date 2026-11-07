"""One queue for heavy generation work: image, video and voice file jobs, whoever asked.

The Studios and the agent bridge both take a slot here before they touch a model, so two heavy
jobs never run side by side (six images, five clips and browser renders at once crashed a Mac).
Slots are granted first come, first served. A waiting job can be cancelled; its caller then gets
``GenerationCancelled``.
"""

from __future__ import annotations

import itertools
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from backend_service.progress import GenerationCancelled


@dataclass
class _Ticket:
    id: int
    kind: str
    source: str
    label: str
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    cancelled: bool = False
    batch: str | None = None  # a storyboard batch id: one Cancel for all of its tickets

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "label": self.label,
            "queuedAt": self.queued_at,
            "startedAt": self.started_at,
            "batch": self.batch,
        }


class HeavyJobQueue:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._waiting: list[_Ticket] = []
        self._running: _Ticket | None = None
        self._ids = itertools.count(1)
        self._start_listeners: list[Callable[[str], Any]] = []

    def add_start_listener(self, listener: Callable[[str], Any]) -> None:
        """Call ``listener(kind)`` when a job starts (warm workers of other kinds then free their memory)."""
        self._start_listeners.append(listener)

    @contextmanager
    def slot(self, kind: str, source: str = "ui", label: str = "", batch: str | None = None) -> Iterator[_Ticket]:
        """Block until this job is first in line and nothing else runs; release on exit."""
        ticket = self.enqueue(kind, source, label, batch)
        try:
            self.wait_turn(ticket)
            yield ticket
        finally:
            self.release(ticket)

    def enqueue(self, kind: str, source: str = "ui", label: str = "", batch: str | None = None) -> _Ticket:
        with self._cond:
            ticket = _Ticket(next(self._ids), kind, source, label, batch=batch)
            self._waiting.append(ticket)
            return ticket

    def wait_turn(self, ticket: _Ticket) -> None:
        with self._cond:
            while True:
                if ticket.cancelled:
                    raise GenerationCancelled("cancelled while waiting in the queue")
                if self._running is None and self._waiting and self._waiting[0] is ticket:
                    self._waiting.pop(0)
                    ticket.started_at = time.time()
                    self._running = ticket
                    break
                self._cond.wait(timeout=1.0)
        for listener in list(self._start_listeners):  # outside the lock: a listener may stop a process
            try:
                listener(ticket.kind)
            except Exception:  # noqa: BLE001 - a listener never blocks the job
                pass

    def release(self, ticket: _Ticket) -> None:
        with self._cond:
            if self._running is ticket:
                self._running = None
            elif ticket in self._waiting:
                self._waiting.remove(ticket)
            self._cond.notify_all()

    def cancel_waiting(self, kind: str | None = None, ticket_id: int | None = None, source: str | None = None) -> int:
        """Cancel waiting jobs (all of a kind, one id, or one source). A running job is not touched."""
        count = 0
        with self._cond:
            for ticket in list(self._waiting):
                if ticket_id is not None and ticket.id != ticket_id:
                    continue
                if kind is not None and ticket.kind != kind:
                    continue
                if source is not None and ticket.source != source:
                    continue
                ticket.cancelled = True
                self._waiting.remove(ticket)
                count += 1
            self._cond.notify_all()
        return count

    def running_source(self, kind: str) -> str | None:
        """Who owns the running job of this kind ("ui", "agent" or "storyboard"), or None when none of that kind runs."""
        with self._cond:
            running = self._running
            return running.source if running is not None and running.kind == kind else None

    def cancel_studio(self, kind: str, stop_running: Callable[[], bool]) -> bool:
        """What a Studio's Cancel button does: stop its own running job and drop its own waiting jobs.

        An agent's job (running or waiting) is left alone; an agent cancels its own with ``cancel_job``.
        """
        stopped = stop_running() if self.running_source(kind) in (None, "ui") else False
        return stopped or self.cancel_waiting(kind, source="ui") > 0

    def cancel_batch(self, batch_id: str, stop_running: Callable[[str], bool] | None = None) -> int:
        """Cancel every waiting ticket of one batch and, if its ticket runs, ask ``stop_running(kind)`` to stop it.

        Returns how many tickets were hit. Nothing outside the batch is touched: ``stop_running`` is called
        with the queue lock held, so the batch's ticket cannot be released and another job (a Studio job on
        the same global progress tracker) cannot start between the ownership check and the stop. It must
        be quick and must not call back into this queue from another thread it waits on (setting a cancel
        flag is all it does).
        """
        count = 0
        with self._cond:
            for ticket in list(self._waiting):
                if ticket.batch == batch_id:
                    ticket.cancelled = True
                    self._waiting.remove(ticket)
                    count += 1
            running = self._running if self._running is not None and self._running.batch == batch_id else None
            if running is not None and stop_running is not None and stop_running(running.kind):
                count += 1
            self._cond.notify_all()
        return count

    def position(self, ticket: _Ticket) -> int:
        """0 when running, 1.. for waiting (1 = next), -1 when finished or cancelled."""
        with self._cond:
            if self._running is ticket:
                return 0
            if ticket in self._waiting:
                return self._waiting.index(ticket) + 1
            return -1

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            return {
                "running": self._running.describe() if self._running else None,
                "waiting": [t.describe() for t in self._waiting],
            }

    def queue_state(self, kind: str) -> dict[str, Any]:
        """What the Studio of this kind shows: is its request waiting, where, and on whom."""
        with self._cond:
            for index, ticket in enumerate(self._waiting):
                if ticket.kind == kind and ticket.source == "ui":
                    ahead = ([self._running] if self._running else []) + self._waiting[:index]
                    return {
                        "waiting": True,
                        "position": index + 1,
                        "agentAhead": any(t.source == "agent" for t in ahead),
                    }
        return {"waiting": False, "position": 0, "agentAhead": False}

    def agent_ahead(self) -> bool:
        """True when an agent job runs or waits; the Studio then shows "waiting for an agent job"."""
        with self._cond:
            jobs = ([self._running] if self._running else []) + self._waiting
            return any(t.source == "agent" for t in jobs)


HEAVY_JOBS = HeavyJobQueue()
