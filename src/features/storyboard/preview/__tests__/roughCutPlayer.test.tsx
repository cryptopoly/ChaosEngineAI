import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { StoryboardProject, StoryboardShot, StoryboardTimeline } from "../../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { RoughCutButton } from "../RoughCutButton";
import { RoughCutPlayer } from "../RoughCutPlayer";

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
