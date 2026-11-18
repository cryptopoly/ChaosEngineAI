import { describe, expect, it } from "vitest";
import { formatTemplateTime, jsonBoxProblem, matchesTemplateSearch } from "../promptLogic";

describe("formatTemplateTime", () => {
  it("turns the backend's epoch seconds into a date, not a raw number", () => {
    const text = formatTemplateTime(1794987548.93451, "en-GB");
    expect(text).not.toContain("1794987548");
    expect(text).toMatch(/2026/);
  });

  it("accepts milliseconds, numeric strings and ISO strings", () => {
    expect(formatTemplateTime(1794987548934, "en-GB")).toMatch(/2026/);
    expect(formatTemplateTime("1794987548.9", "en-GB")).toMatch(/2026/);
    expect(formatTemplateTime("2026-11-18T08:00:00Z", "en-GB")).toMatch(/2026/);
  });

  it("gives an empty string for nothing usable", () => {
    expect(formatTemplateTime(undefined)).toBe("");
    expect(formatTemplateTime("")).toBe("");
    expect(formatTemplateTime("not a date")).toBe("");
    expect(formatTemplateTime(0)).toBe("");
  });
});

describe("matchesTemplateSearch", () => {
  const template = { name: "Translator", category: "Language", tags: ["idiom"], systemPrompt: "Preserve cultural nuance." };

  it("matches name, category, tag and the prompt text", () => {
    expect(matchesTemplateSearch(template, "transl")).toBe(true);
    expect(matchesTemplateSearch(template, "LANGUAGE")).toBe(true);
    expect(matchesTemplateSearch(template, "idiom")).toBe(true);
    expect(matchesTemplateSearch(template, "nuance")).toBe(true);
  });

  it("matches everything on an empty query and nothing on a miss", () => {
    expect(matchesTemplateSearch(template, "  ")).toBe(true);
    expect(matchesTemplateSearch(template, "sql")).toBe(false);
  });
});

describe("jsonBoxProblem", () => {
  it("accepts an empty box and the expected shape", () => {
    expect(jsonBoxProblem("  ", "array")).toBe(false);
    expect(jsonBoxProblem('[{"name": "topic"}]', "array")).toBe(false);
    expect(jsonBoxProblem('{"topP": 0.9}', "object")).toBe(false);
  });

  it("reports text that would have been dropped on save", () => {
    expect(jsonBoxProblem('[{"name": "topic"}', "array")).toBe(true);
    expect(jsonBoxProblem('{"name": "topic"}', "array")).toBe(true);
    expect(jsonBoxProblem("[1, 2]", "object")).toBe(true);
    expect(jsonBoxProblem("null", "object")).toBe(true);
    expect(jsonBoxProblem("topP: 0.9", "object")).toBe(true);
  });
});
