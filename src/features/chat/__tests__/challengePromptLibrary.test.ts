import { describe, it, expect } from "vitest";
import {
  CHALLENGE_PROMPTS,
  CHALLENGE_PROMPT_CATEGORIES,
  challengePromptCountByCategory,
  filterChallengePrompts,
} from "../html_challenge/challengePromptLibrary";

describe("challenge prompt library", () => {
  it("holds 32 prompts split evenly across 4 categories", () => {
    expect(CHALLENGE_PROMPTS).toHaveLength(32);
    expect(CHALLENGE_PROMPT_CATEGORIES).toHaveLength(4);
    const counts = challengePromptCountByCategory();
    expect(counts).toEqual({
      games: 8,
      simulations: 8,
      "tech-demos": 8,
      "creative-tools": 8,
    });
  });

  it("has unique ids and non-empty title/summary/prompt for every entry", () => {
    const ids = new Set<string>();
    for (const entry of CHALLENGE_PROMPTS) {
      expect(entry.id).toBeTruthy();
      expect(ids.has(entry.id)).toBe(false);
      ids.add(entry.id);
      expect(entry.title.trim().length).toBeGreaterThan(0);
      expect(entry.summary.trim().length).toBeGreaterThan(0);
      // Full prompts should be substantially longer than the card summary.
      expect(entry.prompt.trim().length).toBeGreaterThan(80);
    }
    expect(ids.size).toBe(32);
  });

  it("only assigns prompts to known category ids", () => {
    const known = new Set(CHALLENGE_PROMPT_CATEGORIES.map((c) => c.id));
    for (const entry of CHALLENGE_PROMPTS) {
      expect(known.has(entry.category)).toBe(true);
    }
  });
});

describe("filterChallengePrompts", () => {
  it("returns the whole library for ('all', empty query)", () => {
    expect(filterChallengePrompts("all", "")).toHaveLength(32);
    expect(filterChallengePrompts("all", "   ")).toHaveLength(32);
  });

  it("filters by category", () => {
    const games = filterChallengePrompts("games", "");
    expect(games).toHaveLength(8);
    expect(games.every((entry) => entry.category === "games")).toBe(true);
  });

  it("matches title case-insensitively", () => {
    const result = filterChallengePrompts("all", "TETRIS");
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("tetris");
  });

  it("matches on mechanic keywords in the prompt body, not just the title", () => {
    // "pheromone" only appears in the ant colony prompt/summary, not its title.
    const result = filterChallengePrompts("all", "pheromone");
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("ant-colony");

    // "FFT" lives in the spectrum analyzer prompt body.
    const fft = filterChallengePrompts("all", "fft");
    expect(fft.map((e) => e.id)).toContain("spectrum-analyzer");
  });

  it("combines category + query filters", () => {
    // "ball" appears across several game prompts; scope to games only.
    const result = filterChallengePrompts("games", "paddle");
    expect(result.length).toBeGreaterThan(0);
    expect(result.every((entry) => entry.category === "games")).toBe(true);
    expect(result.map((e) => e.id)).toContain("pong");
  });

  it("returns empty for a query that matches nothing", () => {
    expect(filterChallengePrompts("all", "zzzznomatch")).toHaveLength(0);
  });
});
