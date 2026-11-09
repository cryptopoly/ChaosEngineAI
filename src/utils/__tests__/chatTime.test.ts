import { describe, expect, it } from "vitest";
import {
  formatClock,
  formatDuration,
  formatFullTime,
  isValidTime,
  liveElapsedSeconds,
  progressFacts,
} from "../chatTime";

// Built from local parts so the expected text does not depend on the machine's time zone.
const at = (y: number, mo: number, d: number, h: number, mi: number, s = 0) => new Date(y, mo - 1, d, h, mi, s).getTime();

describe("formatClock", () => {
  const now = at(2026, 11, 9, 12, 0);

  it("shows only the short time for today", () => {
    expect(formatClock(at(2026, 11, 9, 7, 31), "en-GB", now)).toBe("07:31");
  });

  it("adds the date when the message is not from today", () => {
    expect(formatClock(at(2026, 11, 8, 23, 5), "en-GB", now)).toBe("8 Nov, 23:05");
  });

  it("adds the year for another year", () => {
    expect(formatClock(at(2025, 12, 31, 9, 0), "en-GB", now)).toContain("2025");
  });

  it("follows the app locale", () => {
    const german = formatClock(at(2026, 11, 8, 23, 5), "de", now);
    expect(german).toContain("23:05");
    expect(german).toContain("Nov");
    expect(german).not.toBe(formatClock(at(2026, 11, 8, 23, 5), "en-GB", now));
  });

  it("falls back to the default locale for a bad tag", () => {
    expect(formatClock(at(2026, 11, 9, 7, 31), "not a locale!!", now)).toMatch(/7:31|07:31/);
  });
});

describe("formatFullTime", () => {
  it("has the weekday, the date and the seconds", () => {
    const text = formatFullTime(at(2026, 11, 9, 7, 31, 42), "en-GB");
    expect(text).toContain("Monday");
    expect(text).toContain("2026");
    expect(text).toContain("07:31:42");
  });
});

describe("formatDuration", () => {
  it("keeps two units at most", () => {
    expect(formatDuration(12, "en")).toBe("12 sec");
    expect(formatDuration(252, "en")).toBe("4 min 12 sec");
    expect(formatDuration(600, "en")).toBe("10 min");
    expect(formatDuration(3900, "en")).toBe("1 hr 5 min");
  });

  it("rounds and never goes negative", () => {
    expect(formatDuration(11.6, "en")).toBe("12 sec");
    expect(formatDuration(-5, "en")).toBe("0 sec");
  });
});

describe("liveElapsedSeconds", () => {
  it("ticks from the start time while running", () => {
    expect(liveElapsedSeconds({ startedAt: 1_000_000, running: true, elapsed: 0 }, 1_252_000)).toBe(252);
  });

  it("uses the saved elapsed time once finished and nothing for an old call without one", () => {
    expect(liveElapsedSeconds({ startedAt: 1_000_000, elapsed: 3.2 }, 9_999_999)).toBe(3.2);
    expect(liveElapsedSeconds({ elapsed: 0 }, 9_999_999)).toBeNull();
    expect(liveElapsedSeconds({ running: true, elapsed: 0 }, 9_999_999)).toBeNull();
  });
});

describe("isValidTime", () => {
  it("rejects what old sessions carry", () => {
    expect(isValidTime(undefined)).toBe(false);
    expect(isValidTime(null)).toBe(false);
    expect(isValidTime(0)).toBe(false);
    expect(isValidTime("2026-11-09")).toBe(false);
    expect(isValidTime(1_762_000_000_000)).toBe(true);
  });
});

describe("progressFacts", () => {
  it("passes an estimate through only with a basis the app knows", () => {
    expect(progressFacts({ phase: "clip 2 of 4", step: 2, totalSteps: 4, etaSeconds: 600, etaBasis: "clips" })).toMatchObject({
      etaSeconds: 600,
      etaBasis: "clips",
    });
    expect(progressFacts({ etaSeconds: 600, etaBasis: "made-up" as never }).etaBasis).toBeNull();
  });

  it("has no estimate without one", () => {
    expect(progressFacts({ elapsedSeconds: 30 }).etaSeconds).toBeNull();
    expect(progressFacts(null).etaSeconds).toBeNull();
    expect(progressFacts({ etaSeconds: -3 }).etaSeconds).toBeNull();
  });
});
