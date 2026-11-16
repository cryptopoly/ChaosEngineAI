import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { StoryboardBatchRecord, StoryboardProject, StoryboardShot, StoryboardTimeline } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

// The Overview reads the same estimate as the run bar; here it says a still takes 10 minutes.
const slowEstimate = { what: "both", shots: 4, stills: 4, clips: 4, seconds: 2600, stillSeconds: 2400, peakGb: 12, totalGb: 64, diskMb: 10, basis: "estimated", missing: [] };
let estimateNow: unknown = null;
vi.mock("../useProjectEstimate", () => ({ useProjectEstimate: () => estimateNow, ESTIMATE_DEBOUNCE_MS: 400 }));

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
const timeline = { clips: [], seconds: 16 } as unknown as StoryboardTimeline;

function render(batch: StoryboardBatchRecord | null) {
  const story = { batch, batchProgress: null, busy: false, op: vi.fn(), startBatch: vi.fn(), cancelBatch: vi.fn(), resumeBatch: vi.fn() } as unknown as UseStoryboardResult;
  const deps = { imageGallery: [], videoGallery: [], imageModels: [], videoModels: [] };
  return renderToStaticMarkup(
    <OverviewTab story={story} project={project} timeline={timeline} deps={deps} onTab={vi.fn()} onOpenShot={vi.fn()} />,
  );
}

const paused = {
  id: "b1", state: "paused", pauseReason: "restart", what: "both", createdAt: 1, finishedAt: null,
  results: [
    { shotId: "a", kind: "still", state: "done" },
    { shotId: "b", kind: "still", state: "queued" },
    { shotId: "c", kind: "still", state: "queued" },
  ],
} as unknown as StoryboardBatchRecord;

describe("Overview next-step card (S-N)", () => {
  it("shows a paused batch with Resume primary and no Make button", () => {
    estimateNow = slowEstimate;
    const html = render(paused);
    expect(html).toContain("Paused when the app closed; Resume carries on.");
    expect(html).toContain("1 of 3 steps done, 2 left");
    expect(html).toMatch(/class="primary-button">Resume</);
    expect(html).not.toContain("Make 4 missing");
    expect(html).not.toContain("Make drafts");
  });

  it("makes Make drafts primary on a fresh film with a slow still model", () => {
    estimateNow = slowEstimate;
    const html = render(null);
    expect(html).toMatch(/class="primary-button">Make drafts</);
    expect(html).toMatch(/class="secondary-button">Make 4 missing</);
    expect(html.match(/class="primary-button"/g)?.length).toBe(1);
  });

  it("keeps Make missing primary when the estimate is fast or unknown", () => {
    estimateNow = { ...slowEstimate, stillSeconds: 40 };
    expect(render(null)).toMatch(/class="primary-button">Make 4 missing</);
    estimateNow = null;
    const html = render(null);
    expect(html).toMatch(/class="primary-button">Make 4 missing</);
    expect(html).not.toContain("Make drafts");
  });
});
