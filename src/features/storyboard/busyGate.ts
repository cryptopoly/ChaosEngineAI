/**
 * When a project change makes the tab "busy" (S-N). The busy flag disables buttons, and a field that saves on blur
 * starts its save on the mousedown that leaves it: if busy turned on at once, the button under the pointer became
 * disabled between mousedown and click and the click was lost (the Length chips, the Plan suggestion's Use link,
 * Time the lyrics). Busy now turns on only once a change has run for `BUSY_GRACE_MS`, so a click that follows a
 * blur-save lands, and a quick save never flashes the controls off. Clicks in that window are safe: the op queue
 * runs them after the save, against its revision.
 */
export const BUSY_GRACE_MS = 400;

export interface BusyGate {
  begin: () => void;
  end: () => void;
  dispose: () => void;
}

type Timers = {
  setTimeout: (run: () => void, ms: number) => unknown;
  clearTimeout: (handle: unknown) => void;
};

const realTimers: Timers = {
  setTimeout: (run, ms) => globalThis.setTimeout(run, ms),
  clearTimeout: (handle) => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
};

export function createBusyGate(set: (busy: boolean) => void, graceMs = BUSY_GRACE_MS, timers: Timers = realTimers): BusyGate {
  let count = 0;
  let timer: unknown = null;
  let shown = false;
  const clear = () => {
    if (timer !== null) timers.clearTimeout(timer);
    timer = null;
  };
  return {
    begin() {
      count += 1;
      if (count === 1 && !shown) {
        timer = timers.setTimeout(() => {
          timer = null;
          shown = true;
          set(true);
        }, graceMs);
      }
    },
    end() {
      count = Math.max(0, count - 1);
      if (count > 0) return;
      clear();
      if (shown) {
        shown = false;
        set(false);
      }
    },
    dispose: clear,
  };
}
