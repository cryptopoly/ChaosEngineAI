import { describe, expect, it } from "vitest";
import type { StoryboardCutPlan } from "../../../../types";
import { clock, cutPlanGroups, lyricsToLines, relabel, rulerParts, sameLines, shotsWithTakes } from "../songLogic";

const sections = [
  { start: 10, end: 30, label: "A", energy: 0.8 },
  { start: 0, end: 10, label: "intro", energy: 0.2 },
  { start: 30, end: 45, label: "outro" },
];

describe("song helpers", () => {
  it("writes clock times", () => {
    expect(clock(185)).toBe("3:05");
    expect(clock(4.2)).toBe("0:04");
    expect(clock(-3)).toBe("0:00");
  });

  it("turns sections into ruler parts in order, as shares of the song", () => {
    const parts = rulerParts(sections, 40);
    expect(parts.map((p) => p.label)).toEqual(["intro", "A", "outro"]);
    expect(parts[0]).toMatchObject({ left: 0, width: 25, energy: 0.2, index: 1 });
    expect(parts[2]).toMatchObject({ start: 30, end: 40, width: 25, energy: null });
    expect(rulerParts(sections, 0)).toEqual([]);
  });

  it("relabels one section, or says nothing changed", () => {
    expect(relabel(sections, 0, "  chorus  ")?.[0].label).toBe("chorus");
    expect(relabel(sections, 0, "A")).toBeNull();
    expect(relabel(sections, 0, "   ")).toBeNull();
    expect(relabel(sections, 9, "x")).toBeNull();
    expect(relabel(sections, 0, "x".repeat(60))?.[0].label).toHaveLength(40);
  });

  it("reads lyrics as lines", () => {
    expect(lyricsToLines("first  line\n\n  second\r\n")).toEqual(["first line", "second"]);
    expect(sameLines(["a"], ["a"])).toBe(true);
    expect(sameLines(["a"], ["b"])).toBe(false);
  });

  it("groups a cut plan's shots under their sections", () => {
    const shot = (title: string) => ({ title, stillPrompt: "", motionPrompt: "", seconds: 3, trimOut: 3 });
    const plan = {
      shots: [shot("intro 1"), shot("A 1"), shot("A 2"), shot("extra")],
      sections: [
        { start: 0, end: 4, label: "intro", shots: 1 },
        { start: 4, end: 10, label: "A", shots: 2 },
      ],
      totalSeconds: 12,
    } as unknown as StoryboardCutPlan;
    const groups = cutPlanGroups(plan);
    expect(groups.map((g) => [g.label, g.shots.map((s) => s.number)])).toEqual([
      ["intro", [1]],
      ["A", [2, 3, 4]],
    ]);
  });

  it("counts shots with takes", () => {
    expect(shotsWithTakes([{ takes: [] }, { takes: [1] }, { takes: [1, 2] }])).toBe(2);
  });
});
