import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { StoryboardProject, StoryboardShot, StoryboardTimeline } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { OverviewTab } from "../OverviewTab";
import type { UseStoryboardResult } from "../useStoryboard";

function shot(id: string): StoryboardShot {
  return {
    id, order: 0, title: `Shot ${id}`, status: "idle", stillPrompt: "a street", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
  };
}

const project = {
  id: "p1", name: "Film", revision: 3, shots: ["a", "b", "c", "d"].map(shot), audio: [], captions: [], batches: [],
  references: { characters: [], places: [], styles: [] },
  defaults: { exportPreset: "16:9", width: 1280, height: 720, fps: 24, imageModel: "slow", videoModel: null },
} as unknown as StoryboardProject;
const timeline = { shots: [], audio: [], total: 16, problems: [] } as unknown as StoryboardTimeline;

function render(p: StoryboardProject) {
  const story = { batch: null, batchProgress: null, busy: false, op: vi.fn(), startBatch: vi.fn(), cancelBatch: vi.fn() } as unknown as UseStoryboardResult;
  const deps = { imageGallery: [], videoGallery: [], imageModels: [], videoModels: [] };
  return renderToStaticMarkup(<OverviewTab story={story} project={p} timeline={timeline} deps={deps} onTab={vi.fn()} onOpenShot={vi.fn()} />);
}

describe("Overview after the stage rail (audit change 1)", () => {
  it("leaves the next step to the stage rail: no next-step card and no checklist", () => {
    const html = render(project);
    expect(html).not.toContain("Next step");
    expect(html).not.toContain("overview-checklist");
    expect(html).not.toContain("Make 4 missing");
  });

  it("still shows the shots at a glance", () => {
    const html = render(project);
    expect(html).toContain("4 shots");
    expect(html.match(/contact-sheet__item /g)).toHaveLength(4);
  });

  it("shows no shot sheet for an empty project", () => {
    expect(render({ ...project, shots: [] } as StoryboardProject)).not.toContain("contact-sheet");
  });
});
