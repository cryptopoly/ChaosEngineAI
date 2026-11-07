import { describe, expect, it } from "vitest";
import { trapTabTarget } from "../useDialogFocus";

describe("trapTabTarget", () => {
  it("wraps Tab from the last control to the first and Shift+Tab the other way", () => {
    expect(trapTabTarget(3, 2, false)).toBe(0);
    expect(trapTabTarget(3, 0, true)).toBe(2);
  });

  it("lets the browser move focus inside the dialog", () => {
    expect(trapTabTarget(3, 1, false)).toBeNull();
    expect(trapTabTarget(3, 1, true)).toBeNull();
    expect(trapTabTarget(3, 0, false)).toBeNull();
  });

  it("pulls focus in when it is outside the dialog", () => {
    expect(trapTabTarget(3, -1, false)).toBe(0);
    expect(trapTabTarget(3, -1, true)).toBe(2);
  });

  it("does nothing for a dialog with no controls", () => {
    expect(trapTabTarget(0, -1, false)).toBeNull();
  });

  it("keeps a single control as the only stop", () => {
    expect(trapTabTarget(1, 0, false)).toBe(0);
    expect(trapTabTarget(1, 0, true)).toBe(0);
  });
});
