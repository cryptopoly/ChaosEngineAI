import { describe, expect, it } from "vitest";
import { NO_ACK, ackKey, dangerBlocked, isAcked } from "../studioGating";

describe("danger gate", () => {
  it("blocks Generate at danger level until the box is ticked, and never blocks a lower level", () => {
    expect(dangerBlocked("danger", false)).toBe(true);
    expect(dangerBlocked("danger", true)).toBe(false);
    expect(dangerBlocked("caution", false)).toBe(false);
    expect(dangerBlocked("safe", false)).toBe(false);
  });

  it("starts unticked", () => {
    expect(isAcked(NO_ACK, ackKey(["m", 832, 480, 33]))).toBe(false);
  });

  it("counts the tick only for the settings it was given for", () => {
    const key = ackKey(["wan", 832, 480, 33]);
    const ticked = { key, checked: true };
    expect(isAcked(ticked, key)).toBe(true);
    // model, width, height or frames changed: the box is unticked again
    expect(isAcked(ticked, ackKey(["wan-14b", 832, 480, 33]))).toBe(false);
    expect(isAcked(ticked, ackKey(["wan", 1280, 480, 33]))).toBe(false);
    expect(isAcked(ticked, ackKey(["wan", 832, 720, 33]))).toBe(false);
    expect(isAcked(ticked, ackKey(["wan", 832, 480, 161]))).toBe(false);
    // unticking stays unticked
    expect(isAcked({ key, checked: false }, key)).toBe(false);
  });

  it("keys do not collide across missing parts", () => {
    expect(ackKey([undefined, 1, 2])).not.toBe(ackKey([1, undefined, 2]));
  });
});
