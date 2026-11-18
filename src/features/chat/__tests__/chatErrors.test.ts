import { describe, expect, it } from "vitest";
import { chatErrorText } from "../chatErrors";

describe("chatErrorText", () => {
  it("says the backend is not answering for each browser's network failure", () => {
    for (const raw of ["Failed to fetch", "Load failed", "NetworkError when attempting to fetch resource."]) {
      expect(chatErrorText(raw)).toMatch(/^The backend is not answering, so the message was not sent\./);
    }
  });

  it("keeps any other error as it is", () => {
    expect(chatErrorText("Streaming timed out")).toBe("Chat error: Streaming timed out");
    expect(chatErrorText("Context window exceeded")).toBe("Chat error: Context window exceeded");
  });
});
