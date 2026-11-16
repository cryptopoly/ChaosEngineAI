import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { StoryboardShot, StoryboardTake } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { ShotCard, type ShotCardProps } from "../ShotCard";
import { TakeCompare } from "../TakeCompare";
import { TakesStrip } from "../TakesStrip";
import { PROJECT_TEMPLATES, templateById, templateShotArgs } from "../projectTemplates";

const S1 = "a".repeat(64);
const S2 = "b".repeat(64);
const C1 = "c".repeat(64);
const C2 = "d".repeat(64);

function take(id: string, extra: Partial<StoryboardTake> = {}): StoryboardTake {
  return {
    id, still: `media/s/${id}/still.png`, clip: `media/s/${id}/clip.mp4`, hashes: { still: S1, clip: C1 },
    settings: {}, measuredSeconds: 2, createdAt: 0, origin: "batch", basis: "x", ...extra,
  };
}

function shot(extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id: "s_1", order: 0, title: "Dawn", status: "done", stillPrompt: "A lighthouse", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 2, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}

const noop = () => undefined;

describe("TakeCompare", () => {
  const takes = [
    take("t1", { createdAt: 1, settings: { clipSeed: 11, clipModel: "LTX-2.5" } }),
    take("t2", { createdAt: 2, renderPass: "draft", hashes: { still: S1, clip: C2 }, settings: { clipSeed: 22, clipModel: "LTX-2.5" } }),
  ];
  const html = renderToStaticMarkup(
    <TakeCompare projectId="p" shot={shot({ takes, activeTake: "t1" })} number={1} onPick={noop} onClose={noop} />,
  );

  it("opens on the active take and the newest other one, with their passes", () => {
    expect(html).toContain("Compare takes · Shot 1");
    expect(html).toContain("take-compare__pane take-compare__pane--active");
    expect(html).toContain(">Final</span>");
    expect(html).toContain(">Draft</span>");
    expect(html).toContain("Take 2 · Draft");
    expect(html).toContain("Take 1 · Final · in use");
  });

  it("offers Use this one only on the take not in use", () => {
    expect(html.match(/Use this one/g)?.length).toBe(1);
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>In use<\/button>/);
  });

  it("names how the takes relate and marks the setting that differs", () => {
    expect(html).toContain("Same still, a different clip.");
    expect(html).toMatch(/take-compare__differs"><th scope="row">Clip seed/);
    expect(html).not.toMatch(/take-compare__differs"><th scope="row">Clip model/);
    expect(html).toContain("Play both");
    expect(html).toContain("Clips");
  });
});

describe("TakeCompare with a still-only draft", () => {
  it("says No clip for the draft's clip facts, not Not recorded", () => {
    const takes = [
      take("t1", { createdAt: 1, settings: { clipSeed: 11, clipModel: "LTX-2.5" } }),
      take("t2", { createdAt: 2, renderPass: "draft", clip: null, hashes: { still: S2 }, measuredSeconds: 0 }),
    ];
    const out = renderToStaticMarkup(
      <TakeCompare projectId="p" shot={shot({ takes, activeTake: "t1" })} number={1} onPick={noop} onClose={noop} />,
    );
    expect(out).toContain("No clip");
    expect(out).not.toContain("Not recorded");
  });
});

describe("TakesStrip compare button", () => {
  it("shows Compare from two takes up when asked for", () => {
    const two = renderToStaticMarkup(<TakesStrip projectId="p" takes={[take("t1"), take("t2")]} activeId="t1" onPick={noop} onCompare={noop} />);
    expect(two).toContain(">Compare</button>");
    const one = renderToStaticMarkup(<TakesStrip projectId="p" takes={[take("t1")]} activeId="t1" onPick={noop} onCompare={noop} />);
    expect(one).not.toContain("Compare");
  });
});

describe("ShotCard after Redo still only", () => {
  function card(overrides: Partial<ShotCardProps> = {}) {
    const props: ShotCardProps = {
      projectId: "abc", shot: shot(), index: 0, count: 1, status: "done", staleAfter: null, kept: false, failure: null,
      tabbable: true, dragOver: false, disabled: false,
      onRename: noop, onPickTake: noop, onToggleLock: noop, onToggleRedo: noop, onTransition: noop, onMove: noop,
      onEdit: noop, onKeep: noop, onRerun: noop, onRetry: noop, onRemove: noop, onDragStart: noop, onDragEnd: noop,
      onDropOn: noop, onDragOverCard: noop, onRedoClip: noop, onRedoStill: noop,
      ...overrides,
    };
    return renderToStaticMarkup(<ShotCard {...props} />);
  }

  it("says the clip is behind the new still and offers Redo clip instead of Re-run", () => {
    const outdated = take("t2", { hashes: { still: S2, clip: C1 }, settings: { clipStill: S1 } });
    const html = card({ status: "stale", shot: shot({ status: "stale", takes: [take("t1"), outdated], activeTake: "t2" }) });
    expect(html).toContain("New still · clip not remade yet");
    expect(html).toContain(">Redo clip</button>");
    expect(html).not.toContain("Re-run");
  });

  it("keeps Re-run for a shot that is stale for another reason", () => {
    const html = card({ status: "stale", shot: shot({ status: "stale", takes: [take("t1")], activeTake: "t1" }) });
    expect(html).toContain("Re-run");
    expect(html).not.toContain(">Redo clip</button>");
  });
});

describe("Sample film template", () => {
  it("is offered last and adds no shots of its own (the bundle has them)", () => {
    expect(PROJECT_TEMPLATES[PROJECT_TEMPLATES.length - 1].id).toBe("sample");
    const sample = templateById("sample");
    expect(sample.sample).toBe(true);
    expect(sample.shots).toHaveLength(4);
    expect(templateShotArgs(sample, (s) => ({ title: s.title, stillPrompt: s.stillPrompt }))).toEqual([]);
  });
});
