import { describe, expect, it } from "vitest";

import { sessionPreview, sortSessions, titleFromPrompt, uniqueSessionTitle, upsertSession } from "../chat";
import type { ChatSession } from "../../types";

function makeSession(overrides: Partial<ChatSession> & { id: string }): ChatSession {
  return {
    title: "Test",
    updatedAt: new Date().toISOString(),
    model: "test-model",
    cacheLabel: "none",
    messages: [],
    ...overrides,
  };
}

describe("sortSessions()", () => {
  it("places pinned sessions before unpinned", () => {
    const sessions = [
      makeSession({ id: "a", pinned: false }),
      makeSession({ id: "b", pinned: true }),
      makeSession({ id: "c", pinned: false }),
    ];
    const sorted = sortSessions(sessions);
    expect(sorted[0].id).toBe("b");
  });

  it("preserves relative order among same-pin-status sessions", () => {
    const sessions = [
      makeSession({ id: "x", pinned: true }),
      makeSession({ id: "y", pinned: true }),
    ];
    const sorted = sortSessions(sessions);
    expect(sorted[0].id).toBe("x");
    expect(sorted[1].id).toBe("y");
  });

  it("does not mutate the original array", () => {
    const sessions = [
      makeSession({ id: "a", pinned: false }),
      makeSession({ id: "b", pinned: true }),
    ];
    const sorted = sortSessions(sessions);
    expect(sorted).not.toBe(sessions);
    expect(sessions[0].id).toBe("a");
  });

  it("handles empty array", () => {
    expect(sortSessions([])).toEqual([]);
  });

  it("treats undefined pinned as unpinned", () => {
    const sessions = [
      makeSession({ id: "a" }),
      makeSession({ id: "b", pinned: true }),
    ];
    const sorted = sortSessions(sessions);
    expect(sorted[0].id).toBe("b");
  });
});

describe("titleFromPrompt()", () => {
  it("takes the first four words of a prompt", () => {
    expect(titleFromPrompt("Tell me about the weather today")).toBe("Tell me about the");
  });

  it("returns full text if fewer than four words", () => {
    expect(titleFromPrompt("Hello world")).toBe("Hello world");
  });

  it("trims leading/trailing whitespace", () => {
    expect(titleFromPrompt("  spaced out  ")).toBe("spaced out");
  });

  it("collapses multiple spaces between words", () => {
    expect(titleFromPrompt("a   b   c   d   e")).toBe("a b c d");
  });

  it("returns 'New chat' for empty/whitespace-only input", () => {
    expect(titleFromPrompt("")).toBe("New chat");
    expect(titleFromPrompt("   ")).toBe("New chat");
  });

  it("adds a numeric suffix when the generated title already exists", () => {
    const sessions = [
      makeSession({ id: "a", title: "Explain how cache compression" }),
      makeSession({ id: "b", title: "Explain how cache compression (2)" }),
    ];
    expect(titleFromPrompt("Explain how cache compression helps long contexts.", sessions)).toBe(
      "Explain how cache compression (3)",
    );
  });
});

describe("uniqueSessionTitle()", () => {
  it("returns the base title when it is unused", () => {
    expect(uniqueSessionTitle([makeSession({ id: "a", title: "Other chat" })], "Fresh title")).toBe("Fresh title");
  });

  it("skips the excluded session when checking suffixes", () => {
    const sessions = [
      makeSession({ id: "a", title: "Repeat me" }),
      makeSession({ id: "b", title: "Repeat me (2)" }),
    ];
    expect(uniqueSessionTitle(sessions, "Repeat me", "b")).toBe("Repeat me (2)");
  });
});

describe("upsertSession()", () => {
  it("prepends a new session", () => {
    const existing = [makeSession({ id: "a" })];
    const newSession = makeSession({ id: "b" });
    const result = upsertSession(existing, newSession);
    expect(result.length).toBe(2);
    expect(result[0].id).toBe("b");
    expect(result[1].id).toBe("a");
  });

  it("moves an existing session to the front when updated", () => {
    const sessions = [
      makeSession({ id: "a", title: "First" }),
      makeSession({ id: "b", title: "Second" }),
    ];
    const updated = makeSession({ id: "b", title: "Updated" });
    const result = upsertSession(sessions, updated);
    expect(result.length).toBe(2);
    expect(result[0].id).toBe("b");
    expect(result[0].title).toBe("Updated");
    expect(result[1].id).toBe("a");
  });

  it("handles empty sessions list", () => {
    const session = makeSession({ id: "x" });
    const result = upsertSession([], session);
    expect(result).toEqual([session]);
  });
});

describe("sessionPreview()", () => {
  it("returns text of the last message", () => {
    const session = makeSession({
      id: "a",
      messages: [
        { role: "user", text: "Hello" } as any,
        { role: "assistant", text: "Hi there" } as any,
      ],
    });
    expect(sessionPreview(session)).toBe("Hi there");
  });

  it("returns fallback for empty messages", () => {
    const session = makeSession({ id: "a", messages: [] });
    expect(sessionPreview(session)).toBe("No messages yet");
  });
});
