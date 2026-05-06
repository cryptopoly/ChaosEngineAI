import { describe, it, expect } from "vitest";
import { filterSessions } from "../sessionSearch";
import type { ChatSession } from "../../../types";

function s(id: string, title: string, messages: { role: "user" | "assistant"; text: string; reasoning?: string }[] = []): ChatSession {
  return {
    id,
    title,
    updatedAt: "2026-05-01",
    cacheLabel: "f16",
    model: "Test/Model",
    messages: messages.map((m) => ({ ...m })),
  };
}

describe("filterSessions", () => {
  const sessions = [
    s("a", "Refactor auth flow", [{ role: "user", text: "How do I handle JWT expiry?" }]),
    s("b", "Holiday plans", [{ role: "user", text: "Suggest beaches in Portugal" }]),
    s("c", "Debug session", [{ role: "assistant", text: "Stack trace shows null deref", reasoning: "Looking at the call site of getUser..." }]),
  ];

  it("returns all sessions for empty query", () => {
    expect(filterSessions(sessions, "")).toHaveLength(3);
    expect(filterSessions(sessions, "   ")).toHaveLength(3);
  });

  it("matches by title (case-insensitive)", () => {
    const result = filterSessions(sessions, "REFACTOR");
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("a");
  });

  it("matches by message body", () => {
    const result = filterSessions(sessions, "portugal");
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("b");
  });

  it("matches by reasoning trace", () => {
    const result = filterSessions(sessions, "getuser");
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("c");
  });

  it("returns empty when nothing matches", () => {
    expect(filterSessions(sessions, "nonexistent-string-xyz")).toEqual([]);
  });
});
