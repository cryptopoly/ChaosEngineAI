import { describe, it, expect } from "vitest";
import { buildMarkdown, buildJson, buildTxt, buildExportContent } from "../exportThread";
import type { ChatSession } from "../../../types";

function makeSession(overrides: Partial<ChatSession> = {}): ChatSession {
  return {
    id: "s1",
    title: "Test Thread",
    updatedAt: "2026-05-01T08:00:00Z",
    cacheLabel: "f16",
    model: "Test/Model",
    messages: [
      { role: "user", text: "What is 2+2?" },
      {
        role: "assistant",
        text: "The answer is 4.",
        reasoning: "Adding two and two yields four.",
        reasoningDone: true,
      },
    ],
    ...overrides,
  };
}

describe("exportThread", () => {
  it("builds markdown with title, model, and messages", () => {
    const md = buildMarkdown(makeSession());
    expect(md).toContain("# Test Thread");
    expect(md).toContain("**Model:** Test/Model");
    expect(md).toContain("## User");
    expect(md).toContain("What is 2+2?");
    expect(md).toContain("## Assistant");
    expect(md).toContain("The answer is 4.");
    expect(md).toContain("<details><summary>Reasoning</summary>");
    expect(md).toContain("Adding two and two yields four.");
  });

  it("builds JSON with exportedAt + full session payload", () => {
    const raw = buildJson(makeSession());
    const parsed = JSON.parse(raw);
    expect(parsed.exportedAt).toBeDefined();
    expect(parsed.session.title).toBe("Test Thread");
    expect(parsed.session.messages).toHaveLength(2);
  });

  it("builds plain text with role headers", () => {
    const txt = buildTxt(makeSession());
    expect(txt).toContain("Test Thread");
    expect(txt).toContain("--- USER ---");
    expect(txt).toContain("--- ASSISTANT ---");
    expect(txt).toContain("[reasoning]");
    expect(txt).toContain("[/reasoning]");
  });

  it("derives a safe filename per format", () => {
    const session = makeSession({ title: "What/are\\sandwich:cookies?" });
    expect(buildExportContent(session, "md").filename).toBe("What_are_sandwich_cookies_.md");
    expect(buildExportContent(session, "json").filename).toMatch(/\.json$/);
    expect(buildExportContent(session, "txt").filename).toMatch(/\.txt$/);
  });

  it("falls back to 'chat' when title is empty", () => {
    const session = makeSession({ title: "" });
    expect(buildExportContent(session, "md").filename).toBe("chat.md");
  });

  it("renders citations with doc name + page when present", () => {
    const session = makeSession({
      messages: [
        {
          role: "assistant",
          text: "See doc.",
          citations: [
            { docId: "d1", docName: "spec.pdf", chunkIndex: 3, page: 5, preview: "..." },
            { docId: "d2", docName: "notes.md", chunkIndex: 1, preview: "..." },
          ],
        },
      ],
    });
    const md = buildMarkdown(session);
    expect(md).toContain("- spec.pdf p.5 (chunk 3)");
    expect(md).toContain("- notes.md (chunk 1)");
  });
});
