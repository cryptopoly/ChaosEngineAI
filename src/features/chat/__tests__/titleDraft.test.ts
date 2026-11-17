import { describe, expect, it } from "vitest";

import { followTitle } from "../optimisticTurns";

describe("followTitle", () => {
  it("moves an untouched name field to the title the first message gave the chat", () => {
    expect(followTitle("New chat", "New chat", "Plan a three-shot film")).toBe("Plan a three-shot film");
  });

  it("keeps a name the person is typing", () => {
    expect(followTitle("My film", "New chat", "Plan a three-shot film")).toBe("My film");
  });
});
