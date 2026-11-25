import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  StoryboardGuideAction,
  StoryboardProject,
  StoryboardShot,
  StoryboardStage,
  StoryboardStageId,
  StoryboardStageStatus,
  StoryboardStages,
} from "../../../../types";
import { isMusicVideo, isProjectTab, markMusicVideo, MUSIC_VIDEO_KEY, PROJECT_TABS, visibleTabs } from "../../projectRoute";
import { PROJECT_TEMPLATES, templateById, templateShotArgs } from "../../projectTemplates";
import { batchArgs, currentStage, destinationOf, focusSongSection, opensSongSection, railStages, shownStage, visibleStages } from "../stageLogic";

/** Every action code the backend sends (`StoryboardGuideAction`). */
const ACTIONS: StoryboardGuideAction[] = [
  "attach-song", "open-cast", "cut-plan", "plan-shots", "open-storyboard", "watch-batch", "resume-batch", "retry-failed",
  "make-drafts", "make-missing", "make-final", "choose-final", "make-clips", "remake-stale", "assign-vocals",
  "make-lipsync", "open-coverage", "open-mixer", "open-check", "export", "real-speed", "unlock", "lengthen", "shorten",
];

function stage(id: StoryboardStageId, status: StoryboardStageStatus, extra: Partial<StoryboardStage> = {}): StoryboardStage {
  return { id, applies: true, optional: false, status, message: `${id} sentence`, action: null, ...extra };
}

/** A film with no song: the backend marks Song and Lip sync as not applying. */
function filmStages(current: StoryboardStageId | null, overrides: Partial<Record<StoryboardStageId, Partial<StoryboardStage>>> = {}): StoryboardStages {
  const base: StoryboardStage[] = [
    stage("song", "not-started", { applies: false, optional: true, action: "attach-song" }),
    stage("look", "not-started", { optional: true, action: "open-cast" }),
    stage("shots", "done"),
    stage("drafts", "not-started", { action: "make-drafts" }),
    stage("final", "not-started", { action: "make-final" }),
    stage("lipsync", "not-started", { applies: false, optional: true }),
    stage("sound", "not-started", { optional: true, action: "open-mixer" }),
    stage("check", "not-started", { action: "open-check" }),
    stage("export", "not-started", { action: "export" }),
  ];
  return { current, stages: base.map((s) => ({ ...s, ...(overrides[s.id] ?? {}) })) };
}

function shot(id: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id, order: 0, title: "", status: "idle", stillPrompt: "p", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}

function made(id: string, pass: "draft" | "final", extra: Partial<StoryboardShot> = {}): StoryboardShot {
  const take = { id: `${id}-t`, still: "s.png", clip: "c.mp4", hashes: {}, settings: {}, measuredSeconds: 4, createdAt: 0, origin: "generate" as const, basis: null, renderPass: pass };
  return shot(id, { status: "done", takes: [take] as unknown as StoryboardShot["takes"], activeTake: take.id, ...extra });
}

const quality = (staleShots: string[] = []) => ({ quality: { staleShots } as never });

describe("action code to destination", () => {
  it("sends every action code somewhere, and never to Overview", () => {
    for (const action of ACTIONS) {
      const to = destinationOf(action);
      expect(to.tab, action).not.toBe("overview");
      expect(isProjectTab(to.tab)).toBe(true);
    }
  });

  it("opens Plan for the song, the cut, the plan and every lip-sync code", () => {
    for (const action of ["attach-song", "cut-plan", "plan-shots", "assign-vocals", "make-lipsync", "open-coverage", "real-speed", "unlock", "lengthen", "shorten"] as const) {
      expect(destinationOf(action)).toEqual({ kind: "tab", tab: "plan" });
    }
  });

  it("opens Cast, the Storyboard, the Mixer and Export for the matching codes", () => {
    expect(destinationOf("open-cast")).toEqual({ kind: "tab", tab: "cast" });
    expect(destinationOf("open-storyboard")).toEqual({ kind: "tab", tab: "storyboard" });
    expect(destinationOf("watch-batch")).toEqual({ kind: "tab", tab: "storyboard" });
    expect(destinationOf("choose-final")).toEqual({ kind: "tab", tab: "storyboard" });
    expect(destinationOf("open-mixer")).toEqual({ kind: "tab", tab: "mixer" });
    expect(destinationOf("open-check")).toEqual({ kind: "tab", tab: "export" });
    expect(destinationOf("export")).toEqual({ kind: "tab", tab: "export" });
  });

  it("starts the run bar's batches, or resumes the paused one, then shows the Storyboard", () => {
    expect(destinationOf("resume-batch")).toEqual({ kind: "resume", tab: "storyboard" });
    expect(destinationOf("make-drafts")).toEqual({ kind: "batch", batch: "drafts", tab: "storyboard" });
    expect(destinationOf("make-missing")).toEqual({ kind: "batch", batch: "missing", tab: "storyboard" });
    expect(destinationOf("make-clips")).toEqual({ kind: "batch", batch: "missing", tab: "storyboard" });
    expect(destinationOf("make-final")).toEqual({ kind: "batch", batch: "final", tab: "storyboard" });
    expect(destinationOf("remake-stale")).toEqual({ kind: "batch", batch: "stale", tab: "storyboard" });
    expect(destinationOf("retry-failed")).toEqual({ kind: "batch", batch: "failed", tab: "storyboard" });
  });
});

describe("batch arguments", () => {
  const project = (shots: StoryboardShot[]) => ({ shots }) as Pick<StoryboardProject, "shots">;

  it("makes draft stills of the missing shots, like Make drafts", () => {
    expect(batchArgs("drafts", project([shot("a"), made("b", "final")]), null)).toEqual({ what: "still", shotIds: ["a"], force: false, draft: true });
  });

  it("makes the missing shots, or nothing when none is missing", () => {
    expect(batchArgs("missing", project([shot("a"), shot("b", { locked: true })]), null)).toEqual({ what: "both", shotIds: ["a"], force: false, draft: false });
    expect(batchArgs("missing", project([made("a", "final")]), null)).toBeNull();
  });

  it("runs the final pass on the kept drafts, else makes the missing shots at full size", () => {
    expect(batchArgs("final", project([made("a", "draft", { kept: true }), made("b", "draft")]), null)).toEqual({ what: "final", shotIds: ["a"], force: false, draft: false });
    expect(batchArgs("final", project([shot("a"), shot("b")]), null)).toEqual({ what: "both", shotIds: ["a", "b"], force: false, draft: false });
  });

  it("remakes stale shots (status or format), leaving locked ones out", () => {
    const shots = [made("a", "final", { status: "stale" }), made("b", "final"), made("c", "final", { status: "stale", locked: true })];
    expect(batchArgs("stale", project(shots), quality(["b"]))).toEqual({ what: "both", shotIds: ["a", "b"], force: false, draft: false });
  });

  it("tries the failed shots again, like the card's Try again", () => {
    expect(batchArgs("failed", project([shot("a", { status: "failed" }), shot("b")]), null)?.shotIds).toEqual(["a"]);
    expect(batchArgs("failed", project([shot("b")]), null)).toBeNull();
  });
});

describe("which stages show", () => {
  it("hides Song and Lip sync for a film without a song", () => {
    const ids = visibleStages(filmStages("drafts").stages).map((s) => s.id);
    expect(ids).toEqual(["look", "shots", "drafts", "final", "sound", "check", "export"]);
  });

  it("shows Song and Lip sync once a song is attached", () => {
    const block = filmStages("drafts", { song: { applies: true, status: "done" }, lipsync: { applies: true } });
    expect(visibleStages(block.stages)).toHaveLength(9);
  });

  it("shows Song and Lip sync for a music video start before the song is attached, and leads with Song", () => {
    const block = filmStages("drafts");
    expect(visibleStages(block.stages, { musicVideo: true }).map((s) => s.id)).toContain("song");
    expect(currentStage(block, { musicVideo: true })).toBe("song");
    expect(currentStage(block)).toBe("drafts");
    const attached = filmStages("drafts", { song: { applies: true, status: "done" } });
    expect(currentStage(attached, { musicVideo: true })).toBe("drafts");
  });
});

describe("how each stage reads", () => {
  it("marks done, current, needs a look, under way, skipped and to do", () => {
    const block = filmStages("final", {
      drafts: { status: "in-progress" },
      final: { status: "needs-attention" },
      check: { status: "needs-attention" },
    });
    const looks = Object.fromEntries(railStages(block).map((r) => [r.stage.id, r.look]));
    expect(looks).toEqual({
      look: "skipped", // optional, never started, and the work moved past it
      shots: "done",
      drafts: "progress",
      final: "attention", // current, and it needs a look

      sound: "todo", // optional but still ahead
      check: "attention",
      export: "todo",
    });
    expect(railStages(block).filter((r) => r.current).map((r) => r.stage.id)).toEqual(["final"]);
    expect(railStages(block).map((r) => r.number)).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it("marks nothing current when every stage is done, and shows the last stage", () => {
    const all = filmStages(null, Object.fromEntries(["look", "drafts", "final", "sound", "check", "export"].map((id) => [id, { status: "done" }])));
    const rail = railStages(all);
    expect(rail.every((r) => r.look === "done")).toBe(true);
    expect(shownStage(rail, null)?.stage.id).toBe("export");
  });

  it("shows the picked stage, else the current one", () => {
    const rail = railStages(filmStages("drafts"));
    expect(shownStage(rail, null)?.stage.id).toBe("drafts");
    expect(shownStage(rail, "sound")?.stage.id).toBe("sound");
    expect(shownStage(rail, "song")?.stage.id).toBe("drafts"); // hidden: falls back to the current one
  });
});

describe("Board behind More (audit change 12)", () => {
  it("keeps Board out of the row until a card is placed, but shows it while open", () => {
    expect(visibleTabs({ board: { pos: {} } }, "overview")).not.toContain("freestyle");
    expect(visibleTabs(null, "overview")).toEqual(["overview", "plan", "cast", "storyboard", "mixer", "export"]);
    expect(visibleTabs({ board: { pos: { s1: { x: 0, y: 0 } } } }, "overview")).toContain("freestyle");
    expect(visibleTabs(null, "freestyle")).toContain("freestyle");
    expect(PROJECT_TABS).toContain("freestyle"); // remembered routes still open it
  });
});

describe("Music video start (audit change 9)", () => {
  function memoryStorage(): Storage {
    const data = new Map<string, string>();
    return {
      get length() {
        return data.size;
      },
      clear: () => data.clear(),
      getItem: (key) => data.get(key) ?? null,
      key: (index) => Array.from(data.keys())[index] ?? null,
      removeItem: (key) => void data.delete(key),
      setItem: (key, value) => void data.set(key, String(value)),
    };
  }
  beforeEach(() => vi.stubGlobal("window", Object.assign(new EventTarget(), { localStorage: memoryStorage() })));
  afterEach(() => vi.unstubAllGlobals());

  it("is a template with no shots of its own, offered right after Blank", () => {
    const music = templateById("musicVideo");
    expect(PROJECT_TEMPLATES[1].id).toBe("musicVideo");
    expect(music.musicVideo).toBe(true);
    expect(templateShotArgs(music, (s) => ({ title: s.title, stillPrompt: s.stillPrompt }))).toEqual([]);
  });

  it("remembers which projects started as a music video, and survives bad storage", () => {
    expect(isMusicVideo("p1")).toBe(false);
    markMusicVideo("p1");
    markMusicVideo("p1");
    expect(isMusicVideo("p1")).toBe(true);
    expect(isMusicVideo("p2")).toBe(false);
    expect(JSON.parse(window.localStorage.getItem(MUSIC_VIDEO_KEY) ?? "[]")).toEqual(["p1"]);
    window.localStorage.setItem(MUSIC_VIDEO_KEY, "{bad");
    expect(isMusicVideo("p1")).toBe(false);
    expect(isMusicVideo(null)).toBe(false);
  });
});

describe("the Song section (audit item: Add the song lands at the top of Plan)", () => {
  it("names the song actions whose work is in the Song section", () => {
    const song = ACTIONS.filter(opensSongSection);
    expect(song).toEqual(["attach-song", "cut-plan", "assign-vocals", "make-lipsync", "open-coverage"]);
    for (const action of song) expect(destinationOf(action)).toEqual({ kind: "tab", tab: "plan" });
  });

  it("focuses and scrolls to the Song heading once the Plan tab has drawn it", () => {
    const frames: FrameRequestCallback[] = [];
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => frames.push(cb));
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
    try {
      const heading = { focus: vi.fn(), scrollIntoView: vi.fn() };
      let drawn = false;
      const doc = { getElementById: (id: string) => (drawn && id === "song-panel-title" ? heading : null) } as unknown as Document;
      focusSongSection(doc);
      frames.shift()?.(0); // the tab has not rendered yet: it looks again
      expect(frames).toHaveLength(1);
      expect(heading.focus).not.toHaveBeenCalled();
      drawn = true;
      frames.shift()?.(0);
      expect(heading.focus).toHaveBeenCalled();
      expect(heading.scrollIntoView).toHaveBeenCalled();
      expect(frames).toHaveLength(0);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
