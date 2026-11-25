import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { StoryboardProject, StoryboardQuality, StoryboardShot, StoryboardTimeline } from "../../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { RoughCutButton } from "../RoughCutButton";
import { RoughCutPlayer } from "../RoughCutPlayer";
import { RoughCutStage } from "../RoughCutStage";
import { buildRoughCut } from "../roughCutLogic";
import type { UseRoughCutResult } from "../useRoughCut";

function shot(id: string, title: string, take: Partial<StoryboardShot["takes"][number]> | null): StoryboardShot {
  return {
    id, order: 0, title, status: "done", stillPrompt: "", motionPrompt: "", referenceImages: [], startFrom: "generate",
    endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null, speed: 1,
    takes: take ? [{ id: "t", still: null, clip: null, hashes: {}, settings: {}, measuredSeconds: 4, createdAt: 0, origin: "batch", basis: null, ...take }] : [],
    activeTake: take ? "t" : null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
  };
}

const project: StoryboardProject = {
  schemaVersion: 1, revision: 2, id: "p1", name: "Tea film", createdAt: 0,
  defaults: { width: 1280, height: 720, fps: 24, exportPreset: "16:9" } as StoryboardProject["defaults"],
  references: { characters: [], places: [] } as unknown as StoryboardProject["references"],
  shots: [
    shot("a", "Kettle", { clip: "media/a/clip.mp4", still: "media/a/still.png" }),
    shot("b", "", { clip: "media/b/clip.mp4", renderPass: "draft" }),
    shot("c", "Cup", { still: "media/c/still.png" }),
    shot("d", "Sign", null),
  ],
  audio: [], export: {} as StoryboardProject["export"], captions: [], batches: [], undoStack: [], redoStack: [],
};

const span = (id: string, start: number, ready: boolean) => ({
  id, start, end: start + 4, duration: 4, source_in: 0, source_out: 4, transition_type: "cut", transition_seconds: 0, measured: true, ready,
});
const timeline: StoryboardTimeline = {
  total: 16, shots: [span("a", 0, true), span("b", 4, true), span("c", 8, false), span("d", 12, false)], audio: [], problems: [],
};

describe("RoughCutPlayer", () => {
  const html = renderToStaticMarkup(<RoughCutPlayer project={project} timeline={timeline} onClose={() => undefined} />);

  it("is a labelled modal dialog with a slider over the whole film", () => {
    expect(html).toContain('role="dialog"');
    expect(html).toContain('aria-modal="true"');
    expect(html).toContain('aria-labelledby="rough-cut-title"');
    expect(html).toContain('role="slider"');
    expect(html).toContain('aria-valuemax="16"');
  });

  it("marks each shot by what it will play", () => {
    for (const state of ["clip", "draft", "still", "none"]) expect(html).toContain(`rough-cut__marker--${state}`);
    expect(html).toContain("Shot 1 · Kettle · Clip");
    expect(html).toContain("Shot 2 · Draft take");
    expect(html).toContain("No clip yet · still held for 4 s");
    expect(html).toContain("No picture yet · 4 s");
  });

  it("says what the export does differently and shows loading first", () => {
    expect(html).toContain("2 shots have no clip yet");
    expect(html).toContain("1 shot plays a draft take.");
    expect(html).toContain("Loading the film: 0 of 3 files");
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*aria-label="Play"/);
  });

  it("offers captions only when there are some", () => {
    expect(html).toMatch(/<button[^>]*rough-cut__cc[^>]*disabled=""/);
  });
});

describe("RoughCutButton", () => {
  it("is off for a film with nothing in it", () => {
    const empty = renderToStaticMarkup(<RoughCutButton project={{ ...project, shots: [] }} timeline={{ ...timeline, total: 0, shots: [] }} />);
    expect(empty).toContain("disabled");
    expect(empty).toContain("Add a shot to play the film.");
    const ready = renderToStaticMarkup(<RoughCutButton project={project} timeline={timeline} />);
    expect(ready).toContain("Play film");
    expect(ready).not.toContain("disabled");
  });
});

describe("Draft size on the rough cut (audit change 2)", () => {
  const sized: StoryboardProject = {
    ...project,
    shots: project.shots.map((s) =>
      s.id === "b" ? { ...s, takes: [{ ...s.takes[0], settings: { clipWidth: 640, clipHeight: 384, stillWidth: 640, stillHeight: 384 } }] } : s,
    ),
  };
  const target = { width: 1280, height: 720, fps: 24 };

  it("names the size in the draft marker, read from the take's settings", () => {
    const html = renderToStaticMarkup(<RoughCutPlayer project={sized} timeline={timeline} onClose={() => undefined} />);
    expect(html).toContain("Shot 2 · Draft take · 640 x 384");
    expect(html).toContain("Drafts are smaller than the film. Choose them for final");
  });

  it("prefers the guide's sizes", () => {
    const quality: StoryboardQuality = {
      target, draftSize: { width: 640, height: 384 }, draftShots: ["b"], staleShots: [],
      takes: { b: { t: { pass: "draft", draft: true, still: null, clip: { width: 704, height: 416 }, stillStale: false, clipStale: false } } },
    };
    const html = renderToStaticMarkup(<RoughCutPlayer project={sized} timeline={timeline} quality={quality} onClose={() => undefined} />);
    expect(html).toContain("Shot 2 · Draft take · 704 x 416");
  });

  it("puts the size on the draft chip over the picture, with a plain explanation", () => {
    const cut = buildRoughCut(sized, timeline);
    const player = {
      timeS: 4, ready: true, urlOf: () => null, registerLayer: () => undefined, load: { done: 0, total: 0, failed: [] },
    } as unknown as UseRoughCutResult;
    const stage = (current: (typeof cut.pieces)[number]) =>
      renderToStaticMarkup(
        <RoughCutStage cut={cut} player={player} width={1280} height={720} caption={null} current={current}
          shotLabel={(p) => `Shot ${p.index + 1}`} target={target} />,
      );
    const html = stage(cut.pieces[1]);
    expect(html).toContain(">Draft 640 x 384</span>");
    expect(html).toContain("The film is 1280 x 720");
    expect(stage(cut.pieces[0])).not.toContain("rough-cut__chip--draft");
  });
});
