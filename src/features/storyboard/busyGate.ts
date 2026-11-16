/**
 * When a project change makes the tab "busy" (S-N). The busy flag disables buttons so a command is not sent twice.
 * A field that saves on blur starts its save on the mousedown that leaves it: when busy turned on at once, the
 * button under the pointer became disabled between mousedown and click and the click was lost (the Length chips,
 * the Plan suggestion's Use link, Time the lyrics, and any other button next to a field).
 *
 * The cause is fixed here, once: a change that starts while a pointer is held down (a blur-save made by that very
 * press) shows busy only after the pointer is released and its click has been delivered. Every other change (a
 * click's own command, a key) shows busy at once, so a double click still cannot send a command twice. The click
 * that lands is safe: the op queue runs it after the save, against the saved revision.
 */
export interface BusyGate {
  begin: () => void;
  end: () => void;
  dispose: () => void;
}

type Listen = (type: string, handler: () => void) => () => void;

const windowListen: Listen = (type, handler) => {
  if (typeof window === "undefined") return () => undefined;
  window.addEventListener(type, handler, true);
  return () => window.removeEventListener(type, handler, true);
};

export function createBusyGate(
  set: (busy: boolean) => void,
  listen: Listen = windowListen,
  later: (run: () => void) => void = (run) => void setTimeout(run, 0),
): BusyGate {
  let count = 0;
  let shown = false;
  let pointerDown = false;
  let pending = false;
  const show = () => {
    pending = false;
    if (count > 0 && !shown) {
      shown = true;
      set(true);
    }
  };
  const release = () => {
    pointerDown = false;
    // The click is dispatched in the same task as pointerup: show busy after it.
    if (pending) later(show);
  };
  const stops = [
    listen("pointerdown", () => void (pointerDown = true)),
    listen("pointerup", release),
    listen("pointercancel", release),
  ];
  return {
    begin() {
      count += 1;
      if (count > 1 || shown) return;
      if (pointerDown) pending = true;
      else show();
    },
    end() {
      count = Math.max(0, count - 1);
      if (count > 0) return;
      pending = false;
      if (shown) {
        shown = false;
        set(false);
      }
    },
    dispose() {
      for (const stop of stops) stop();
    },
  };
}
