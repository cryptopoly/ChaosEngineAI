import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { speakQueueLineKey } from "../speakQueue";

describe("speakQueueLineKey", () => {
  it("says nothing unless the request waits", () => {
    expect(speakQueueLineKey(null)).toBeNull();
    expect(speakQueueLineKey({ waiting: false, position: 0, agentAhead: false, runningKind: null })).toBeNull();
  });

  it("names the job ahead", () => {
    expect(speakQueueLineKey({ waiting: true, position: 1, agentAhead: false, runningKind: "video" })).toBe("video");
    expect(speakQueueLineKey({ waiting: true, position: 1, agentAhead: false, runningKind: "music" })).toBe("music");
    expect(speakQueueLineKey({ waiting: true, position: 2, agentAhead: true, runningKind: "voice" })).toBe("other");
    expect(speakQueueLineKey({ waiting: true, position: 1, agentAhead: false, runningKind: null })).toBe("other");
  });
});
