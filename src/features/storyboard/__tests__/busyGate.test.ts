import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BUSY_GRACE_MS, createBusyGate } from "../busyGate";

describe("busy gate (S-N: a click after a blur-save is not lost)", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("does not turn busy on for a quick change, so the button under the pointer stays enabled", () => {
    const set = vi.fn();
    const gate = createBusyGate(set);
    gate.begin();
    vi.advanceTimersByTime(BUSY_GRACE_MS - 1);
    gate.end();
    vi.advanceTimersByTime(BUSY_GRACE_MS * 2);
    expect(set).not.toHaveBeenCalled();
  });

  it("turns busy on after the grace and off when the last change ends", () => {
    const set = vi.fn();
    const gate = createBusyGate(set);
    gate.begin();
    gate.begin();
    vi.advanceTimersByTime(BUSY_GRACE_MS);
    expect(set).toHaveBeenLastCalledWith(true);
    gate.end();
    expect(set).toHaveBeenCalledTimes(1);
    gate.end();
    expect(set).toHaveBeenLastCalledWith(false);
  });

  it("drops a pending timer on dispose", () => {
    const set = vi.fn();
    const gate = createBusyGate(set);
    gate.begin();
    gate.dispose();
    vi.advanceTimersByTime(BUSY_GRACE_MS * 2);
    expect(set).not.toHaveBeenCalled();
  });
});
