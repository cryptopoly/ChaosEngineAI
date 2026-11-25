import { describe, expect, it } from "vitest";
import type { StoryboardFailure, StoryboardFailureCode } from "../../../types";
import { FAILURE_ACTIONS, FAILURE_CODES, failureActionOf, failureButton, failureSentence } from "../failureText";
import type { TFn } from "../qualityText";

const t: TFn = (_key, options) =>
  String(options.defaultValue).replace(/\{(\w+)\}/g, (_m, name: string) => String(options[name] ?? ""));

const failure = (code: string, action: string, extra: Partial<StoryboardFailure> = {}) =>
  ({ shotId: "s_1", kind: "still", code, message: "backend sentence", action, error: "raw", ...extra }) as StoryboardFailure;

describe("failure sentences", () => {
  it("has its own plain sentence for every code the backend sends", () => {
    for (const code of FAILURE_CODES) {
      const text = failureSentence(t, failure(code, "retry"));
      expect(text.length).toBeGreaterThan(10);
      expect(text).not.toContain("backend sentence");
      expect(text).not.toMatch(/\{\w+\}/);
    }
  });

  it("says what happened, not 'edit the prompts'", () => {
    expect(failureSentence(t, failure("out-of-memory", "smaller-size"))).toBe(
      "There was not enough free memory to make this shot at this size.",
    );
    expect(failureSentence(t, failure("disk-full", "free-disk"))).toContain("Free some space");
    expect(failureSentence(t, null)).toBe("The shot did not render.");
    for (const code of FAILURE_CODES) expect(failureSentence(t, failure(code, "retry"))).not.toContain("edit the prompts");
  });

  it("uses the backend's sentence for a code this app does not know", () => {
    expect(failureSentence(t, failure("new-code" as StoryboardFailureCode, "retry"))).toBe("backend sentence");
    expect(failureSentence(t, failure("new-code" as StoryboardFailureCode, "retry", { message: "" }))).toBe("The shot did not render.");
  });
});

describe("the one failure button", () => {
  const can = { canHandle: () => true };

  it("maps each action code to its button", () => {
    expect(failureButton(t, failure("out-of-memory", "smaller-size"), can)).toMatchObject({
      action: "smaller-size",
      label: "Make it at draft size",
    });
    expect(failureButton(t, failure("out-of-memory", "smaller-size"), { ...can, draftSize: { width: 640, height: 384 } }).label).toBe(
      "Make it at draft size (640 x 384)",
    );
    expect(failureButton(t, failure("model-missing", "open-models", { kind: "still" }), can).label).toBe("Open Image Models");
    expect(failureButton(t, failure("lipsync-model", "open-models", { kind: "clip" }), can).label).toBe("Open Video Models");
    expect(failureButton(t, failure("video-tools-missing", "install-video-tools"), can)).toMatchObject({
      action: "install-video-tools",
      label: "Install video tools",
    });
    expect(failureButton(t, failure("no-prompt", "edit-shot"), can)).toMatchObject({ action: "edit-shot", label: "Edit the prompts" });
    expect(failureButton(t, failure("cancelled", "retry"), can)).toMatchObject({ action: "retry", label: "Retry" });
  });

  it("has no button of its own for a full disk: the sentence says what to free, Retry follows", () => {
    expect(failureButton(t, failure("disk-full", "free-disk"), can)).toMatchObject({ action: "retry", label: "Retry" });
  });

  it("falls back to Retry when the screen cannot do the action, or the action is new", () => {
    expect(failureButton(t, failure("model-missing", "open-models")).action).toBe("retry");
    expect(failureButton(t, failure("out-of-memory", "smaller-size"), { canHandle: () => false }).action).toBe("retry");
    expect(failureActionOf(failure("x", "do-something-new"))).toBe("retry");
    expect(failureButton(t, null).action).toBe("retry");
  });

  it("knows every action the backend lists", () => {
    expect([...FAILURE_ACTIONS].sort()).toEqual(
      ["edit-shot", "free-disk", "install-video-tools", "open-models", "retry", "smaller-size"],
    );
  });
});
