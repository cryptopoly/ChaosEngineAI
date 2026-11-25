import { describe, expect, it } from "vitest";
import type { StoryboardLipSyncCoverage, StoryboardLipSyncRow, StoryboardProject } from "../../../../types";
import {
  LIP_SYNC_RANGE,
  coverageInUse,
  cutOptions,
  fixAction,
  fixLabel,
  initialSingAnswer,
  reasonText,
  rowFix,
  stateLabel,
  waitingShotIds,
} from "../coverageLogic";

function row(extra: Partial<StoryboardLipSyncRow>): StoryboardLipSyncRow {
  return {
    shotId: "s1", number: 1, songStart: 0, songEnd: 3, seconds: 3, state: "synced", code: null, message: "",
    fix: null, minSeconds: 2, maxSeconds: 7.6, ...extra,
  };
}

const fill = (text: { defaultValue: string; values?: Record<string, string | number> }) =>
  text.defaultValue.replace(/\{(\w+)\}/g, (_, k: string) => String(text.values?.[k] ?? `{${k}}`));

describe("fix code to action", () => {
  it("maps each fix code to an existing action", () => {
    expect(fixAction("lengthen", "a")).toEqual({ kind: "open-length", shotId: "a" });
    expect(fixAction("shorten", "a")).toEqual({ kind: "open-length", shotId: "a" });
    expect(fixAction("unlock", "a")).toEqual({ kind: "unlock", shotId: "a" });
    expect(fixAction("real-speed", "a")).toEqual({ kind: "real-speed", shotId: "a" });
    expect(fixAction("assign-vocals", "a")).toEqual({ kind: "assign-vocals" });
    expect(fixAction("make-lipsync", "a")).toEqual({ kind: "make-lipsync", shotIds: ["a"] });
  });

  it("puts a button only on skipped and waiting rows", () => {
    expect(rowFix(row({ state: "skipped", code: "locked", fix: "unlock" }))).toBe("unlock");
    expect(rowFix(row({ state: "skipped", code: "no-singing", fix: null }))).toBeNull();
    expect(rowFix(row({ state: "waiting", fix: "make-lipsync" }))).toBe("make-lipsync");
    expect(rowFix(row({ state: "unassigned", code: "not-assigned", fix: "assign-vocals" }))).toBeNull();
    expect(rowFix(row({ state: "synced" }))).toBeNull();
    expect(rowFix(row({ state: "skipped", code: "too-short", fix: "lengthen" }), false)).toBeNull();
    expect(rowFix(row({ state: "skipped", code: "locked", fix: "unlock" }), false)).toBe("unlock");
  });

  it("labels fixes with the row's own limits", () => {
    expect(fill(fixLabel("lengthen", row({})))).toBe("Lengthen to 2 s or more");
    expect(fill(fixLabel("shorten", row({ maxSeconds: 7.6 })))).toBe("Shorten to 7.6 s or less");
  });
});

describe("plain text for states and reasons", () => {
  it("has a sentence for every reason code and state", () => {
    const codes = ["other-speed", "locked", "outside-song", "too-short", "too-long", "no-singing", "not-assigned"] as const;
    for (const code of codes) expect(reasonText(code, row({})).defaultValue.length).toBeGreaterThan(5);
    for (const state of ["synced", "waiting", "skipped", "unassigned"] as const) expect(stateLabel(state).defaultValue).toBeTruthy();
    expect(fill(reasonText("too-short", row({ seconds: 1.64 })))).toBe("Too short: 1.6 s, a lip-sync shot needs at least 2 s.");
  });
});

describe("hint text", () => {
  it("says 2 s and up, with silent room, and 3 to 8 s in the clip", () => {
    expect(fill(LIP_SYNC_RANGE)).toBe(
      "A shot of 2 s or more can lip-sync. Its clip lasts 3 to 8 s: a shot shorter than 3 s gets silent room to fill it.",
    );
  });
});

describe("flow choice", () => {
  const project = (extra: Partial<StoryboardProject>) => ({ shots: [], song: null, ...extra }) as unknown as StoryboardProject;

  it("asks for singing shots of 3 to 5 s on Yes and the ordinary cut otherwise", () => {
    expect(cutOptions("yes")).toEqual({ minShotSeconds: 3, maxShotSeconds: 5 });
    expect(cutOptions("no")).toEqual({});
    expect(cutOptions(null)).toEqual({});
  });

  it("starts on Yes only when the project already used lip sync", () => {
    expect(initialSingAnswer(project({}))).toBeNull();
    expect(initialSingAnswer(project({ shots: [{ vocalClip: "a.wav" }] } as never))).toBe("yes");
    expect(initialSingAnswer(project({ song: { coverage: { source: "song", at: 1, shots: [] } } } as never))).toBe("yes");
  });

  it("keeps the table on show once lip sync was used", () => {
    const cov = (source: "stem" | null, sings: number) =>
      ({ source, at: null, counts: { synced: 0, waiting: 0, skipped: 0, unassigned: 0, total: 0, sings }, shots: [] }) as StoryboardLipSyncCoverage;
    expect(coverageInUse(null)).toBe(false);
    expect(coverageInUse(cov(null, 0))).toBe(false);
    expect(coverageInUse(cov("stem", 0))).toBe(true);
    expect(coverageInUse(cov(null, 2))).toBe(true);
    const mixed = { ...cov("stem", 1), shots: [row({ state: "waiting" }), row({ shotId: "s2" })] };
    expect(waitingShotIds(mixed)).toEqual(["s1"]);
  });
});
