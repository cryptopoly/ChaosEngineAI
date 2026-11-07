import { describe, expect, it } from "vitest";
import { StoryboardApiError, StoryboardRevisionConflict } from "../../../api/storyboard";
import type { StoryboardBatchRecord, StoryboardProject, StoryboardShot, StoryboardTake } from "../../../types";
import {
  batchProgress,
  classifyFailure,
  draftFromShot,
  drawerEscapeAction,
  effectiveShotStatus,
  importKindOf,
  isBareKey,
  liveShotStates,
  mapWithLimit,
  missingShotIds,
  moveItem,
  nextTakeIndex,
  reorderByDrop,
  reorderByStep,
  rovingTakeIndex,
  shotCardClassName,
  shotNeedsWork,
  shotPatchFromDraft,
  sortByRecent,
  staleAfterShot,
  thumbnailOf,
  toggleReference,
  transitionWith,
  trapTabTarget,
  unlockedShotIds,
} from "../storyboardLogic";

function take(id: string, extra: Partial<StoryboardTake> = {}): StoryboardTake {
  return { id, still: null, clip: null, hashes: {}, settings: {}, measuredSeconds: null, createdAt: 0, origin: "upload", basis: null, ...extra };
}

function shot(id: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id, order: 0, title: "", status: "idle", stillPrompt: "", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}

function batch(results: StoryboardBatchRecord["results"], state: StoryboardBatchRecord["state"] = "running"): StoryboardBatchRecord {
  return { id: "b_1", what: "both", force: false, state, createdAt: 0, finishedAt: null, results };
}

describe("shot card state", () => {
  it("maps each status to a state class and adds locked on top", () => {
    for (const status of ["idle", "queued", "running", "done", "failed", "stale", "missing"] as const) {
      expect(shotCardClassName(status, false)).toBe(`shot-card shot-card--${status}`);
    }
    expect(shotCardClassName("done", true)).toBe("shot-card shot-card--done shot-card--locked");
  });

  it("lets the live batch state win over the stored status", () => {
    const s = shot("s1", { status: "done" });
    expect(effectiveShotStatus(s, "running")).toBe("running");
    expect(effectiveShotStatus(s, undefined)).toBe("done");
  });

  it("reads live states from an unfinished batch only, running over queued", () => {
    const live = liveShotStates(
      batch([
        { shotId: "a", kind: "still", state: "done", takeId: null, error: null },
        { shotId: "a", kind: "clip", state: "running", takeId: null, error: null },
        { shotId: "b", kind: "still", state: "queued", takeId: null, error: null },
        { shotId: "b", kind: "clip", state: "queued", takeId: null, error: null },
      ]),
    );
    expect(live.get("a")).toBe("running");
    expect(live.get("b")).toBe("queued");
    expect(liveShotStates(batch([{ shotId: "a", kind: "still", state: "queued", takeId: null, error: null }], "done")).size).toBe(0);
    expect(liveShotStates(null).size).toBe(0);
  });

  it("names the shot a stale shot continues from only when it starts from the previous shot", () => {
    const shots = [shot("a"), shot("b", { status: "stale", startFrom: "previous-shot" }), shot("c", { status: "stale" })];
    expect(staleAfterShot(shots, 1)).toBe(0);
    expect(staleAfterShot(shots, 2)).toBeNull();
    expect(staleAfterShot(shots, 0)).toBeNull();
  });

  it("shows the still of the active take, else its clip", () => {
    const both = shot("a", { takes: [take("t1", { still: "media/a/t1/still.png", clip: "media/a/t1/clip.mp4" })], activeTake: "t1" });
    const clipOnly = shot("b", { takes: [take("t1", { clip: "media/b/t1/clip.mp4" })], activeTake: "t1" });
    expect(thumbnailOf(both)).toEqual({ kind: "still", path: "media/a/t1/still.png" });
    expect(thumbnailOf(clipOnly)).toEqual({ kind: "clip", path: "media/b/t1/clip.mp4" });
    expect(thumbnailOf(shot("c"))).toBeNull();
  });
});

describe("takes strip keyboard model", () => {
  it("moves with arrows, Home and End, clamped at the ends", () => {
    expect(nextTakeIndex(1, 4, "ArrowRight")).toBe(2);
    expect(nextTakeIndex(3, 4, "ArrowRight")).toBe(3);
    expect(nextTakeIndex(1, 4, "ArrowLeft")).toBe(0);
    expect(nextTakeIndex(0, 4, "ArrowLeft")).toBe(0);
    expect(nextTakeIndex(2, 4, "Home")).toBe(0);
    expect(nextTakeIndex(0, 4, "End")).toBe(3);
  });

  it("ignores other keys and an empty strip", () => {
    expect(nextTakeIndex(0, 3, "a")).toBeNull();
    expect(nextTakeIndex(0, 0, "ArrowRight")).toBeNull();
  });

  it("makes exactly one take tabbable: the focused one, else the active one, else the first", () => {
    const ids = ["t1", "t2", "t3"];
    expect(rovingTakeIndex(ids, "t2", null)).toBe(1);
    expect(rovingTakeIndex(ids, "t2", 2)).toBe(2);
    expect(rovingTakeIndex(ids, null, null)).toBe(0);
    expect(rovingTakeIndex(ids, "gone", null)).toBe(0);
  });
});

describe("batch progress", () => {
  const r = (shotId: string, kind: "still" | "clip", state: StoryboardBatchRecord["results"][number]["state"]) =>
    ({ shotId, kind, state, takeId: null, error: null });

  it("counts stills first, with the shot being made", () => {
    const progress = batchProgress(
      batch([
        r("a", "still", "done"), r("b", "still", "running"), r("c", "still", "queued"),
        r("a", "clip", "queued"), r("b", "clip", "queued"), r("c", "clip", "queued"),
      ]),
    );
    expect(progress).toEqual({ kind: "stills", done: 1, total: 3, shotNumber: 2, shotCount: 3, thenClips: true });
  });

  it("moves on to clips once every still is finished", () => {
    const progress = batchProgress(
      batch([
        r("a", "still", "done"), r("b", "still", "failed"),
        r("a", "clip", "done"), r("b", "clip", "running"),
      ]),
    );
    expect(progress).toMatchObject({ kind: "clips", done: 1, total: 2, shotNumber: 2 });
  });

  it("returns null for an empty batch", () => {
    expect(batchProgress(batch([]))).toBeNull();
  });
});

describe("which shots a batch makes", () => {
  const withTake = (id: string, extra: Partial<StoryboardShot> = {}) =>
    shot(id, { takes: [take("t1", { clip: `media/${id}/t1/clip.mp4` })], activeTake: "t1", status: "done", ...extra });

  it("makes missing: unlocked shots without an active take that a batch can make", () => {
    const shots = [
      shot("a"),
      withTake("b"),
      shot("c", { locked: true }),
      shot("d", { startFrom: "upload" }),
      shot("e", { status: "missing" }),
    ];
    expect(missingShotIds(shots)).toEqual(["a"]);
  });

  it("makes all unlocked: every unlocked shot a batch can work on, taken or not", () => {
    const shots = [shot("a"), withTake("b"), shot("c", { locked: true }), shot("d", { startFrom: "upload" })];
    expect(unlockedShotIds(shots)).toEqual(["a", "b"]);
  });

  it("matches the backend rule for what needs work", () => {
    expect(shotNeedsWork(shot("a"))).toBe(true);
    expect(shotNeedsWork(withTake("b"))).toBe(false);
    expect(shotNeedsWork(withTake("c", { redo: true }))).toBe(true);
    expect(shotNeedsWork(withTake("d", { status: "stale" }))).toBe(true);
    expect(shotNeedsWork(shot("e", { startFrom: "upload" }))).toBe(false);
  });
});

describe("reorder", () => {
  it("moves an item and clamps the target", () => {
    expect(moveItem(["a", "b", "c"], 0, 2)).toEqual(["b", "c", "a"]);
    expect(moveItem(["a", "b", "c"], 2, -4)).toEqual(["c", "a", "b"]);
    const same = ["a", "b"];
    expect(moveItem(same, 1, 1)).toBe(same);
  });

  it("steps one place with the buttons and refuses to leave the list", () => {
    expect(reorderByStep(["a", "b", "c"], "b", -1)).toEqual(["b", "a", "c"]);
    expect(reorderByStep(["a", "b", "c"], "b", 1)).toEqual(["a", "c", "b"]);
    expect(reorderByStep(["a", "b", "c"], "a", -1)).toBeNull();
    expect(reorderByStep(["a", "b", "c"], "c", 1)).toBeNull();
    expect(reorderByStep(["a"], "zz", 1)).toBeNull();
  });

  it("drops a dragged shot onto another and ignores a drop on itself", () => {
    expect(reorderByDrop(["a", "b", "c"], "a", "c")).toEqual(["b", "c", "a"]);
    expect(reorderByDrop(["a", "b", "c"], "c", "a")).toEqual(["c", "a", "b"]);
    expect(reorderByDrop(["a", "b"], "a", "a")).toBeNull();
    expect(reorderByDrop(["a", "b"], "a", "x")).toBeNull();
  });

  it("always keeps every id exactly once", () => {
    const order = reorderByDrop(["a", "b", "c", "d"], "b", "d") as string[];
    expect([...order].sort()).toEqual(["a", "b", "c", "d"]);
  });
});

describe("transitions", () => {
  it("a cut has no length; switching to crossfade starts at half a second", () => {
    expect(transitionWith({ type: "crossfade", seconds: 1 }, { type: "cut" })).toEqual({ type: "cut", seconds: 0 });
    expect(transitionWith({ type: "cut", seconds: 0 }, { type: "crossfade" })).toEqual({ type: "crossfade", seconds: 0.5 });
  });

  it("clamps the crossfade length to 0.1 to 5 s", () => {
    expect(transitionWith({ type: "crossfade", seconds: 1 }, { seconds: 99 }).seconds).toBe(5);
    expect(transitionWith({ type: "crossfade", seconds: 1 }, { seconds: 0 }).seconds).toBe(0.1);
    expect(transitionWith({ type: "crossfade", seconds: 1 }, { seconds: Number.NaN }).seconds).toBe(0.5);
  });
});

describe("shot editor draft", () => {
  it("sends only what changed", () => {
    const s = shot("a", { stillPrompt: "a lighthouse", seconds: 4, seed: 12 });
    const draft = { ...draftFromShot(s), stillPrompt: "a lighthouse at dusk", seed: "" };
    expect(shotPatchFromDraft(s, draft)).toEqual({ stillPrompt: "a lighthouse at dusk", seed: null });
    expect(shotPatchFromDraft(s, draftFromShot(s))).toEqual({});
  });

  it("reads numbers and clears an empty trim-out", () => {
    const s = shot("a", { trimOut: 3 });
    const draft = { ...draftFromShot(s), seconds: "6.5", trimOut: "", speed: "2" };
    expect(shotPatchFromDraft(s, draft)).toEqual({ seconds: 6.5, trimOut: null, speed: 2 });
  });

  it("toggles references and keeps the limit of eight", () => {
    expect(toggleReference(["r1"], "r2")).toEqual(["r1", "r2"]);
    expect(toggleReference(["r1", "r2"], "r1")).toEqual(["r2"]);
    const full = ["1", "2", "3", "4", "5", "6", "7", "8"];
    expect(toggleReference(full, "9")).toBe(full);
  });

  it("picks the import argument from the file extension", () => {
    expect(importKindOf("/x/a.PNG")).toBe("still");
    expect(importKindOf("/x/a.mp4")).toBe("clip");
    expect(importKindOf("/x/a.txt")).toBeNull();
  });
});

describe("failed requests", () => {
  it("adopts the fresh project on a revision conflict", () => {
    const project = { id: "abc", revision: 9 } as unknown as StoryboardProject;
    expect(classifyFailure(new StoryboardRevisionConflict("stale", project))).toEqual({ kind: "conflict", project });
  });

  it("reports anything else as an error message", () => {
    expect(classifyFailure(new StoryboardApiError("Unknown shot", 400))).toEqual({ kind: "error", message: "Unknown shot" });
    expect(classifyFailure("boom")).toEqual({ kind: "error", message: "boom" });
  });
});

describe("project ordering", () => {
  it("sorts the most recently changed first", () => {
    expect(sortByRecent([{ recency: 1, n: "a" }, { recency: 9, n: "b" }, { recency: 5, n: "c" }]).map((x) => x.n)).toEqual(["b", "c", "a"]);
  });
});

describe("shot editor keys", () => {
  it("Escape asks before it throws away edits and does nothing mid-composition", () => {
    const base = { key: "Escape", isComposing: false, dirty: false, confirming: false };
    expect(drawerEscapeAction(base)).toBe("close");
    expect(drawerEscapeAction({ ...base, dirty: true })).toBe("ask");
    expect(drawerEscapeAction({ ...base, dirty: true, isComposing: true })).toBe("none");
    expect(drawerEscapeAction({ ...base, dirty: true, confirming: true })).toBe("none");
    expect(drawerEscapeAction({ ...base, key: "a" })).toBe("none");
  });

  it("keeps Tab inside the drawer", () => {
    expect(trapTabTarget(5, 4, false)).toBe(0);
    expect(trapTabTarget(5, 0, true)).toBe(4);
    expect(trapTabTarget(5, 2, false)).toBeNull();
    expect(trapTabTarget(5, -1, false)).toBe(0);
    expect(trapTabTarget(5, -1, true)).toBe(4);
    expect(trapTabTarget(0, -1, false)).toBeNull();
  });

  it("ignores card shortcuts that carry Cmd, Ctrl or Alt", () => {
    expect(isBareKey({ metaKey: false, ctrlKey: false, altKey: false })).toBe(true);
    expect(isBareKey({ metaKey: true, ctrlKey: false, altKey: false })).toBe(false);
    expect(isBareKey({ metaKey: false, ctrlKey: true, altKey: false })).toBe(false);
    expect(isBareKey({ metaKey: false, ctrlKey: false, altKey: true })).toBe(false);
  });
});

describe("mapWithLimit", () => {
  it("never runs more than the limit at once and keeps the order", async () => {
    let running = 0;
    let peak = 0;
    const out = await mapWithLimit([1, 2, 3, 4, 5, 6, 7], 3, async (n) => {
      running += 1;
      peak = Math.max(peak, running);
      await new Promise((resolve) => setTimeout(resolve, 2));
      running -= 1;
      return n * 10;
    });
    expect(out).toEqual([10, 20, 30, 40, 50, 60, 70]);
    expect(peak).toBe(3);
    expect(await mapWithLimit([], 3, async (n) => n)).toEqual([]);
  });
});
