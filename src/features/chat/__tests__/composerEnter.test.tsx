import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));
vi.mock("../../voice/ComposerMicButton", () => ({ ComposerMicButton: () => null }));

import { canSendNow, composerEnterKey, composerPlaceholderKind } from "../ChatComposer";

// FU-193: Enter follows the Send button's busy rule.
const enter = { key: "Enter", shiftKey: false, isComposing: false };
const shiftEnter = { key: "Enter", shiftKey: true, isComposing: false };
const composingEnter = { key: "Enter", shiftKey: false, isComposing: true };

describe("canSendNow", () => {
  it("is true only with a model loaded, not loading, and no turn running anywhere", () => {
    expect(canSendNow("m", false, null)).toBe(true);
    expect(canSendNow("m", false, "s1")).toBe(false); // this chat's own turn, a detached turn or a Keep going nudge
    expect(canSendNow("m", false, "other")).toBe(false); // the button is a Stop while any chat answers
    expect(canSendNow(undefined, false, null)).toBe(false);
    expect(canSendNow("m", true, null)).toBe(false);
  });
});

describe("composerEnterKey", () => {
  it("sends on Enter when idle and swallows it when busy", () => {
    expect(composerEnterKey(enter, canSendNow("m", false, null))).toBe("send");
    expect(composerEnterKey(enter, canSendNow("m", false, "s1"))).toBe("swallow");
  });

  it("leaves Shift+Enter to insert a newline, busy or idle", () => {
    expect(composerEnterKey(shiftEnter, true)).toBe("default");
    expect(composerEnterKey(shiftEnter, false)).toBe("default");
  });

  it("never sends while an IME composition is open", () => {
    expect(composerEnterKey(composingEnter, true)).toBe("default");
    expect(composerEnterKey(composingEnter, false)).toBe("default");
  });

  it("ignores other keys", () => {
    expect(composerEnterKey({ key: "a", shiftKey: false, isComposing: false }, true)).toBe("default");
  });
});

// Review D, L-18: right after a load the composer said "Type a message" while Send was still off.
describe("composerPlaceholderKind", () => {
  it("says the model is loading until the load finishes, then invites a message", () => {
    expect(composerPlaceholderKind("m", true)).toBe("loading");
    expect(composerPlaceholderKind(undefined, true)).toBe("loading");
    expect(composerPlaceholderKind("m", false)).toBe("ready");
    expect(composerPlaceholderKind(null, false)).toBe("noModel");
  });
});
