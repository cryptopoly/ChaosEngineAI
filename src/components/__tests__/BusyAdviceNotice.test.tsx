import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { BusyAdviceNotice } from "../BusyAdviceNotice";
import { isVisibleAdvice, needsConfirmation, pickChatAdvice } from "../busyAdvice";
import type { BusyAdvice, ChatMessage } from "../../types";

function advice(level: BusyAdvice["level"], message = "A video job is running."): BusyAdvice {
  return { level, message, jobs: [], combinedPeakGb: 10, totalGb: 64, availableGb: 30, footprintGb: 0 };
}
const assistant = (busyAdvice?: BusyAdvice): ChatMessage => ({ role: "assistant", text: "", busyAdvice });

describe("busy advice helpers", () => {
  it("shows only non-ok advice with a message", () => {
    expect(isVisibleAdvice(advice("slower"))).toBe(true);
    expect(isVisibleAdvice(advice("ok", ""))).toBe(false);
    expect(isVisibleAdvice(advice("risky", ""))).toBe(false);
    expect(isVisibleAdvice(null)).toBe(false);
  });

  it("picks the open turn's advice only while generating", () => {
    const messages: ChatMessage[] = [{ role: "user", text: "hi" }, assistant(advice("slower"))];
    expect(pickChatAdvice("c1", messages, true)?.key).toBe("c1:1");
    expect(pickChatAdvice("c1", messages, false)).toBeNull();
    expect(pickChatAdvice("c1", [{ role: "user", text: "hi" }], true)).toBeNull();
    expect(pickChatAdvice("c1", [assistant(advice("ok", ""))], true)).toBeNull();
    expect(pickChatAdvice("c1", [], true)).toBeNull();
  });

  it("offers confirmation only when risky", () => {
    expect(needsConfirmation(advice("risky"))).toBe(true);
    expect(needsConfirmation(advice("slower"))).toBe(false);
    expect(needsConfirmation(null)).toBe(false);
  });
});

describe("BusyAdviceNotice", () => {
  it("renders the message as a status for slower, with a dismiss button when given a handler", () => {
    const html = renderToStaticMarkup(<BusyAdviceNotice advice={advice("slower")} onDismiss={() => {}} />);
    expect(html).toContain("A video job is running.");
    expect(html).toContain('role="status"');
    expect(html).toContain("Dismiss");
  });

  it("renders risky as an alert and omits Dismiss without a handler", () => {
    const html = renderToStaticMarkup(<BusyAdviceNotice advice={advice("risky", "Estimated peak 60 GB.")} />);
    expect(html).toContain('role="alert"');
    expect(html).toContain("Estimated peak 60 GB.");
    expect(html).not.toContain("Dismiss");
  });
});
