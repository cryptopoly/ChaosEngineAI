import { describe, expect, it, vi } from "vitest";
import { createBusyGate } from "../busyGate";

/** A fake window: the gate's pointer listeners, fired by hand; `later` runs when flushed. */
function harness() {
  const handlers = new Map<string, () => void>();
  const queued: Array<() => void> = [];
  const set = vi.fn();
  const gate = createBusyGate(
    set,
    (type, handler) => {
      handlers.set(type, handler);
      return () => handlers.delete(type);
    },
    (run) => void queued.push(run),
  );
  const fire = (type: string) => handlers.get(type)?.();
  const flush = () => queued.splice(0).forEach((run) => run());
  return { gate, set, fire, flush, handlers };
}

describe("busy gate (S-N: a click right after a blur-save is not lost)", () => {
  it("shows busy at once for a change no pointer press started (a click's own command, a key)", () => {
    const { gate, set } = harness();
    gate.begin();
    expect(set).toHaveBeenLastCalledWith(true);
    gate.end();
    expect(set).toHaveBeenLastCalledWith(false);
  });

  it("holds busy back while the press that blurred the field is down, and shows it after its click", () => {
    const { gate, set, fire, flush } = harness();
    fire("pointerdown");
    gate.begin(); // the field's blur-save
    expect(set).not.toHaveBeenCalled(); // the button under the pointer stays enabled for the click
    fire("pointerup");
    expect(set).not.toHaveBeenCalled(); // the click is delivered in this task
    flush();
    expect(set).toHaveBeenLastCalledWith(true);
    gate.end();
    expect(set).toHaveBeenLastCalledWith(false);
  });

  it("never shows busy when the save ends before the press is released", () => {
    const { gate, set, fire, flush } = harness();
    fire("pointerdown");
    gate.begin();
    gate.end();
    fire("pointerup");
    flush();
    expect(set).not.toHaveBeenCalled();
  });

  it("counts overlapping changes and removes its listeners on dispose", () => {
    const { gate, set, handlers } = harness();
    gate.begin();
    gate.begin();
    gate.end();
    expect(set).toHaveBeenCalledTimes(1);
    gate.end();
    expect(set).toHaveBeenLastCalledWith(false);
    gate.dispose();
    expect(handlers.size).toBe(0);
  });
});
