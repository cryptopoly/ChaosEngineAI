import { describe, expect, it } from "vitest";
import { PROMPT_EVAL_TIMEOUT_MS, promptEvalTimeoutMs } from "../promptEvalWatchdog";

describe("promptEvalTimeoutMs", () => {
  it("guards a plain streamed turn", () => {
    expect(promptEvalTimeoutMs({})).toBe(PROMPT_EVAL_TIMEOUT_MS);
  });

  it("does not cancel a tool turn, whose first round is not streamed (S-P: empty reply at 60 s)", () => {
    expect(promptEvalTimeoutMs({ studioTools: true })).toBeNull();
    expect(promptEvalTimeoutMs({ enableTools: true })).toBeNull();
    expect(promptEvalTimeoutMs({ skills: true })).toBeNull();
  });
});
