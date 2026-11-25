import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type {
  StoryboardBatchRecord,
  StoryboardGuide,
  StoryboardProject,
  StoryboardShot,
  StoryboardStage,
  StoryboardStageId,
  StoryboardStageStatus,
} from "../../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));
let estimateNow: unknown = null;
vi.mock("../../useProjectEstimate", () => ({ useProjectEstimate: () => estimateNow, ESTIMATE_DEBOUNCE_MS: 400 }));

import { StageRail } from "../StageRail";

function stage(id: StoryboardStageId, status: StoryboardStageStatus, extra: Partial<StoryboardStage> = {}): StoryboardStage {
  return { id, applies: true, optional: false, status, message: `The ${id} sentence.`, action: null, ...extra };
}

function guideWith(current: StoryboardStageId | null, overrides: Partial<Record<StoryboardStageId, Partial<StoryboardStage>>> = {}): StoryboardGuide {
  const stages = [
    stage("song", "not-started", { applies: false, optional: true, action: "attach-song", message: "Add the song this film is cut to." }),
    stage("look", "not-started", { optional: true, action: "open-cast" }),
    stage("shots", "not-started", { action: "plan-shots", message: "Write the shots." }),
    stage("drafts", "not-started", { message: "Write the shots first." }),
    stage("final", "not-started", { action: "make-final" }),
    stage("lipsync", "not-started", { applies: false, optional: true }),
    stage("sound", "not-started", { optional: true, action: "open-mixer" }),
    stage("check", "not-started", { action: "open-check" }),
    stage("export", "not-started", { action: "export" }),
  ].map((s) => ({ ...s, ...(overrides[s.id] ?? {}) }));
  return {
    stages: { current, stages },
    quality: { staleShots: [], draftShots: [] },
    coverage: null,
    failures: [],
    readiness: { ready: true, stops: [], worthFixing: [], forInfo: [] },
    cast: {},
  } as unknown as StoryboardGuide;
}

function shot(id: string): StoryboardShot {
  return {
    id, order: 0, title: "", status: "idle", stillPrompt: "p", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
  };
}

const project = { id: "p1", revision: 2, shots: [shot("a"), shot("b")] } as unknown as StoryboardProject;

function render(guide: StoryboardGuide, extra: { musicVideo?: boolean; batch?: StoryboardBatchRecord | null; busy?: boolean; initialStage?: StoryboardStageId } = {}) {
  const story = { busy: extra.busy ?? false, batch: extra.batch ?? null, startBatch: vi.fn(), resumeBatch: vi.fn() };
  return renderToStaticMarkup(
    <StageRail guide={guide} project={project} story={story} musicVideo={extra.musicVideo} onTab={vi.fn()} initialStage={extra.initialStage ?? null} />,
  );
}

/** The panel under the rail: its sentence and its one button. */
function panel(html: string): string {
  return html.slice(html.indexOf("stage-rail__panel"));
}

/** The name of the stage marked current. */
function currentName(html: string): string | null {
  const item = html.split("<li>").find((part) => part.includes('aria-current="step"'));
  return item?.match(/stage-rail__name">([^<]+)</)?.[1] ?? null;
}

/** How the named stage reads (its `stage-rail__step--` class). */
function lookOf(html: string, name: string): string | null {
  const item = html.split("<li>").find((part) => part.includes(`stage-rail__name">${name}<`));
  return item?.match(/stage-rail__step--(\w+)/)?.[1] ?? null;
}

describe("stage rail for a film without a song", () => {
  it("lists seven stages, Song and Lip sync hidden, with the current one marked", () => {
    const html = render(guideWith("shots"));
    expect(html.match(/<li>/g)).toHaveLength(7);
    expect(html).not.toContain(">Song<");
    expect(html).not.toContain(">Lip sync<");
    expect(currentName(html)).toBe("Shots");
    expect(panel(html)).toContain("Write the shots.");
    expect(panel(html)).toMatch(/class="primary-button stage-rail__button">Plan the shots</);
    expect(panel(html).match(/<button/g)).toHaveLength(1);
  });
});

describe("stage rail in each status", () => {
  it("not started: the sentence and its button", () => {
    const html = render(guideWith("final", { shots: { status: "done" }, drafts: { status: "done" }, final: { message: "Make the shots at 1280 x 720." } }));
    expect(html).toContain("stage-rail__step--current");
    expect(panel(html)).toContain("Make the shots at 1280 x 720.");
    expect(panel(html)).toContain(">Make them full size<");
  });

  it("in progress: a running batch offers Watch the progress", () => {
    const running = { id: "b1", state: "running", results: [] } as unknown as StoryboardBatchRecord;
    const html = render(guideWith("drafts", { shots: { status: "done" }, drafts: { status: "in-progress", action: "watch-batch", message: "The shots are being made." } }), { batch: running });
    expect(panel(html)).toContain("The shots are being made.");
    expect(panel(html)).toMatch(/<button[^>]*>Watch the progress</);
    expect(panel(html)).not.toContain("disabled");
  });

  it("in progress while paused: Resume", () => {
    const paused = { id: "b1", state: "paused", results: [] } as unknown as StoryboardBatchRecord;
    const html = render(guideWith("drafts", { drafts: { status: "in-progress", action: "resume-batch" } }), { batch: paused });
    expect(panel(html)).toMatch(/<button[^>]*>Resume</);
    expect(panel(html)).not.toContain("disabled");
  });

  it("needs attention: marked on the rail, with its fix", () => {
    const html = render(guideWith("drafts", { drafts: { status: "needs-attention", action: "retry-failed", message: "2 shots did not render." } }), { initialStage: "drafts" });
    expect(panel(html)).toContain("stage-rail__panel--attention");
    expect(currentName(html)).toBe("Drafts");
    expect(panel(html)).toContain("2 shots did not render.");
    expect(panel(html)).toContain(">Try them again<");
    const elsewhere = render(guideWith("final", { drafts: { status: "needs-attention", action: "retry-failed" } }));
    expect(elsewhere).toContain("stage-rail__step--attention");
    expect(elsewhere).toContain("Needs a look");
  });

  it("done: a tick, and no button when the stage has nothing to press", () => {
    const html = render(guideWith("final", { shots: { status: "done", action: null, message: "The shots are written." } }), { initialStage: "shots" });
    expect(html).toContain("stage-rail__step--done");
    expect(panel(html)).toContain("The shots are written.");
    expect(panel(html)).not.toContain("<button");
  });

  it("skipped: an optional stage the work moved past", () => {
    const html = render(guideWith("final", { shots: { status: "done" }, drafts: { status: "done" } }));
    expect(lookOf(html, "Look")).toBe("skipped");
    expect(html).toContain("Skipped");
  });

  it("waits with a batch button while a change is saving or another batch runs", () => {
    estimateNow = null;
    const drafts = guideWith("drafts", { drafts: { action: "make-drafts", message: "Make quick drafts to choose the shots you like." } });
    expect(panel(render(drafts))).not.toContain("disabled");
    expect(panel(render(drafts, { busy: true }))).toContain("disabled");
    const running = { id: "b1", state: "running", results: [] } as unknown as StoryboardBatchRecord;
    expect(panel(render(drafts, { batch: running }))).toContain("disabled");
    // Opening a tab never waits.
    expect(panel(render(guideWith("shots"), { busy: true }))).not.toContain("disabled");
  });

  it("keeps Make off and says why when no model is installed (FU-474)", () => {
    estimateNow = { missing: ["No image model is installed."] };
    const html = render(guideWith("drafts", { drafts: { action: "make-drafts" } }));
    expect(panel(html)).toContain("No image model is installed.");
    expect(panel(html)).toContain("disabled");
    estimateNow = null;
  });
});

describe("stage rail for a music video", () => {
  it("shows the Song stage first and leads with it before a song is attached", () => {
    const html = render(guideWith("shots"), { musicVideo: true });
    expect(html.match(/<li>/g)).toHaveLength(9);
    expect(currentName(html)).toBe("Song");
    expect(panel(html)).toContain("Add the song this film is cut to.");
    expect(panel(html)).toContain(">Add the song<");
  });

  it("moves on once the song is attached", () => {
    const html = render(guideWith("shots", { song: { applies: true, status: "done", action: null }, lipsync: { applies: true } }), { musicVideo: true });
    expect(currentName(html)).toBe("Shots");
  });
});
