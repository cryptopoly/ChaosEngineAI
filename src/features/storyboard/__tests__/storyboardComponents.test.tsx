import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { StoryboardBatchRecord, StoryboardShot, StoryboardTake } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { BatchBar, batchProgressText } from "../BatchBar";
import { lastFailure } from "../BoardTab";
import { ShotCard, type ShotCardProps } from "../ShotCard";
import { TakesStrip } from "../TakesStrip";

function take(id: string): StoryboardTake {
  return { id, still: null, clip: null, hashes: {}, settings: {}, measuredSeconds: 4.04, createdAt: 0, origin: "upload", basis: null };
}

function shot(extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id: "s_1", order: 0, title: "Arrival", status: "idle", stillPrompt: "", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}

const noop = () => undefined;
function card(overrides: Partial<ShotCardProps> = {}) {
  const props: ShotCardProps = {
    projectId: "abc", shot: shot(), index: 1, count: 3, status: "idle", staleAfter: null, kept: false, failure: null,
    tabbable: true, dragOver: false, disabled: false,
    onRename: noop, onPickTake: noop, onToggleLock: noop, onToggleRedo: noop, onTransition: noop, onMove: noop,
    onEdit: noop, onKeep: noop, onRerun: noop, onRetry: noop, onRemove: noop, onDragStart: noop, onDragEnd: noop,
    onDropOn: noop, onDragOverCard: noop,
    ...overrides,
  };
  return renderToStaticMarkup(<ShotCard {...props} />);
}

describe("ShotCard", () => {
  it("carries the state class and the status sentence for each state", () => {
    expect(card({ status: "idle" })).toContain("shot-card--idle");
    expect(card({ status: "idle" })).toContain("Not made yet");
    expect(card({ status: "queued", queuePosition: 3 })).toContain("Queued · 3");
    expect(card({ status: "running" })).toContain("Making…");
    expect(card({ status: "missing" })).toContain("Take file missing");
    expect(card({ status: "missing" })).toContain("Relink");
  });

  it("shows the measured length of the active take, not the planned one", () => {
    const html = card({ status: "done", shot: shot({ status: "done", takes: [take("t1")], activeTake: "t1" }) });
    expect(html).toContain("shot-card--done");
    expect(html).toContain("Done · 4.04 s");
  });

  it("offers Re-run and Keep on a stale shot until Keep is chosen", () => {
    const stale = shot({ status: "stale", startFrom: "previous-shot" });
    const html = card({ status: "stale", shot: stale, staleAfter: 0 });
    expect(html).toContain("Stale · after shot 1");
    expect(html).toContain("Re-run");
    expect(html).toContain("Keep");
    expect(card({ status: "stale", shot: stale, staleAfter: 0, kept: true })).not.toContain("Re-run");
  });

  it("marks a locked shot with aria-pressed and a Locked badge", () => {
    const html = card({ shot: shot({ locked: true }) });
    expect(html).toContain("shot-card--locked");
    expect(html).toContain('aria-pressed="true"');
    expect(html).toContain("Unlock this shot");
  });

  it("shows technical details and Retry on a failed shot", () => {
    const html = card({ status: "failed", failure: "out of memory" });
    expect(html).toContain("shot-card--failed");
    expect(html).toContain("Technical details");
    expect(html).toContain("out of memory");
    expect(html).toContain("Retry");
  });

  it("has no transition picker on the first shot and disables Move up there", () => {
    const first = card({ index: 0 });
    expect(first).not.toContain("transition-picker");
    expect(first).toMatch(/disabled=""[^>]*aria-label="Move up"|aria-label="Move up"[^>]*disabled=""/);
    expect(card({ index: 1 })).toContain("transition-picker");
  });

  it("keeps only one card in the tab order", () => {
    expect(card({ tabbable: true })).toContain('tabindex="0"');
    expect(card({ tabbable: false })).toContain('tabindex="-1"');
  });
});

describe("TakesStrip", () => {
  it("is a radio group with the active take checked and only it tabbable", () => {
    const html = renderToStaticMarkup(
      <TakesStrip projectId="abc" takes={[take("t1"), take("t2"), take("t3")]} activeId="t2" onPick={noop} />,
    );
    expect(html).toContain('role="radiogroup"');
    expect(html.match(/role="radio"/g)).toHaveLength(3);
    expect(html.match(/aria-checked="true"/g)).toHaveLength(1);
    expect(html.match(/tabindex="0"/g)).toHaveLength(1);
    expect(html.match(/tabindex="-1"/g)).toHaveLength(2);
    expect(html).toContain("takes-strip__take--active");
  });

  it("renders nothing without takes", () => {
    expect(renderToStaticMarkup(<TakesStrip projectId="abc" takes={[]} activeId={null} onPick={noop} />)).toBe("");
  });
});

describe("BatchBar", () => {
  const record = (results: StoryboardBatchRecord["results"]): StoryboardBatchRecord => ({
    id: "b_1", what: "both", force: false, state: "running", createdAt: 0, finishedAt: null, results,
  });
  const r = (shotId: string, kind: "still" | "clip", state: StoryboardBatchRecord["results"][number]["state"]) =>
    ({ shotId, kind, state, takeId: null, error: null });

  it("reads 'Shot 3 of 8 · stills, then clips' while stills come first", () => {
    const results = [];
    for (let i = 1; i <= 8; i++) results.push(r(`s${i}`, "still", i < 3 ? "done" : i === 3 ? "running" : "queued"));
    for (let i = 1; i <= 8; i++) results.push(r(`s${i}`, "clip", "queued"));
    const html = renderToStaticMarkup(<BatchBar batch={record(results)} onCancel={noop} />);
    expect(html).toContain("Shot 3 of 8 · stills, then clips");
    expect(html).toContain("Cancel batch");
    expect(html).toContain("2 of 8 done");
  });

  it("drops 'then clips' once clips are being made", () => {
    const t = (key: string, options: Record<string, unknown>) => String(options.defaultValue)
      .replace("{n}", String(options.n)).replace("{total}", String(options.total)).replace("{kind}", String(options.kind));
    expect(
      batchProgressText(t, { kind: "clips", done: 1, total: 2, shotNumber: 2, shotCount: 2, thenClips: true }),
    ).toBe("Shot 2 of 2 · clips");
  });
});

describe("lastFailure", () => {
  it("returns the error the last batch recorded for a shot", () => {
    const batches: StoryboardBatchRecord[] = [
      {
        id: "b_1", what: "both", force: false, state: "failed", createdAt: 0, finishedAt: 1,
        results: [{ shotId: "s_1", kind: "clip", state: "failed", takeId: null, error: "out of memory" }],
      },
    ];
    expect(lastFailure(batches, "s_1")).toBe("out of memory");
    expect(lastFailure(batches, "s_2")).toBeNull();
    expect(lastFailure([], "s_1")).toBeNull();
  });
});
