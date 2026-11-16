import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { StoryboardBatchRecord, StoryboardCutPlan, StoryboardProject } from "../../../../types";
import { BatchBar } from "../../BatchBar";
import { PausedBatchBar, pausedCounts, pausedProgressOf } from "../../PausedBatchBar";
import { CutPlanReview } from "../../song/CutPlanReview";
import { SongRuler } from "../../song/SongRuler";
import { BoardToolbar } from "../BoardToolbar";

const noop = () => undefined;

function batch(extra: Partial<StoryboardBatchRecord> = {}): StoryboardBatchRecord {
  return {
    id: "b1",
    what: "both",
    force: false,
    state: "paused",
    createdAt: 1,
    finishedAt: null,
    pauseReason: "restart",
    results: [
      { shotId: "s1", kind: "still", state: "done", takeId: "t1", error: null },
      { shotId: "s2", kind: "still", state: "queued", takeId: null, error: null },
      { shotId: "s1", kind: "clip", state: "queued", takeId: "t1", error: null },
      { shotId: "s3", kind: "clip", state: "skipped", takeId: null, error: null },
    ],
    ...extra,
  };
}

describe("pause and resume in the batch bar", () => {
  it("says a restart paused it, with steps done and left, time left, Resume and Cancel", () => {
    const progress = { paused: true as const, pauseReason: "restart" as const, totalSteps: 12, step: 5, remainingSteps: 7, remaining: { still: 2, clip: 5 }, etaSeconds: 840 };
    const html = renderToStaticMarkup(<PausedBatchBar batch={batch()} progress={progress} onResume={noop} onCancel={noop} />);
    expect(html).toContain("Paused when the app closed; Resume carries on.");
    expect(html).toContain("5 of 12 steps done, 7 left");
    expect(html).toContain("about 14 min left");
    expect(html).toMatch(/class="primary-button"[^>]*>Resume</);
    expect(html).toContain("Cancel batch");
  });

  it("counts from the record before the bar data comes, and says when the user paused", () => {
    expect(pausedCounts(batch(), null)).toEqual({ done: 1, left: 2, total: 3 });
    expect(pausedProgressOf({ step: 1 } as never)).toBeNull();
    const html = renderToStaticMarkup(<PausedBatchBar batch={batch({ pauseReason: "user" })} onResume={noop} onCancel={noop} />);
    expect(html).toContain("Resume carries on with the next one");
    expect(html).not.toContain("about");
  });

  it("offers Pause while a batch runs, and says it is pausing after the step", () => {
    const running = batch({ state: "running" });
    expect(renderToStaticMarkup(<BatchBar batch={running} onCancel={noop} onPause={noop} />)).toContain(">Pause</button>");
    expect(renderToStaticMarkup(<BatchBar batch={running} onCancel={noop} onPause={noop} pausing />)).toContain("Pausing after this step");
    expect(renderToStaticMarkup(<BatchBar batch={running} onCancel={noop} />)).not.toContain("Pause");
  });
});

describe("song ruler and cut plan", () => {
  const sections = [
    { start: 0, end: 8, label: "intro", energy: 0.2 },
    { start: 8, end: 30, label: "A", energy: 0.9 },
  ];

  it("draws each section with its label; labels are buttons when they can be renamed", () => {
    const html = renderToStaticMarkup(<SongRuler seconds={30} sections={sections} onRename={noop} />);
    expect(html).toContain('aria-label="Song sections"');
    expect(html).toMatch(/<button[^>]*song-ruler__label[^>]*>intro<\/button>/);
    expect(html).toContain("Section A, 0:08 to 0:30");
    const plain = renderToStaticMarkup(<SongRuler seconds={30} sections={sections} compact />);
    expect(plain).not.toContain("<button");
  });

  it("reviews a cut plan by section before it replaces the shots", () => {
    const plan = {
      shots: [
        { title: "intro 1", stillPrompt: "", motionPrompt: "", seconds: 4, trimOut: 4 },
        { title: "A 1", stillPrompt: "a road at night", motionPrompt: "", seconds: 3.5, trimOut: 3.5 },
      ],
      mode: "replace",
      totalSeconds: 30,
      clipSeconds: 4,
      clipSecondsSource: "model",
      videoModel: "LTX-2.5 distilled",
      sections: [
        { ...sections[0], shots: 1 },
        { ...sections[1], shots: 1 },
      ],
      note: "Cuts fall on beats.",
      rough: true,
    } as StoryboardCutPlan;
    const project = { shots: [{ id: "s1", takes: [{ id: "t" }], locked: false }] } as unknown as StoryboardProject;
    const html = renderToStaticMarkup(<CutPlanReview plan={plan} project={project} busy={false} onUse={noop} onDiscard={noop} />);
    expect(html).toContain("Cut plan: 2 shots over 0:30");
    expect(html).toContain("the clip length of LTX-2.5 distilled");
    expect(html).toContain("a road at night");
    expect(html).toContain("Use these 2 shots");
    expect(html).toContain("Undo brings the old shots back");
  });
});

describe("board toolbar", () => {
  const base = { zoom: 0.75, multi: false, selectedCount: 0, selectedShots: 0, canOrder: true, busy: false, onZoomIn: noop, onZoomOut: noop, onFit: noop, onMulti: noop, onClear: noop, onOrder: noop, onAddFiles: noop };

  it("orders all shots with no selection, or the selected ones", () => {
    expect(renderToStaticMarkup(<BoardToolbar {...base} />)).toContain("Order all shots");
    const html = renderToStaticMarkup(<BoardToolbar {...base} selectedCount={3} selectedShots={3} />);
    expect(html).toContain("Order these 3");
    expect(html).toContain("3 cards selected");
    expect(html).toContain("75%");
    expect(html).toContain('aria-pressed="false"');
  });
});
