import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { StoryboardProject, StoryboardShot } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { CaptionsPanel } from "../mixer/CaptionsPanel";
import { PlanBrief } from "../PlanBrief";
import { PlanShotRow } from "../PlanShotRow";
import { ReferenceCard } from "../ReferenceCard";
import { SongRuler } from "../song/SongRuler";
import { createUnsavedTextStore, trackSave, UnsavedTextContext, type UnsavedTextStore } from "../unsavedText";
import { UnsavedTextNotice } from "../useUnsavedText";
import type { UseStoryboardResult } from "../useStoryboard";

const noop = () => undefined;
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

function shot(id: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id, order: 0, title: "Steam", status: "idle", stillPrompt: "A teapot", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}

function project(extra: Partial<StoryboardProject> = {}): StoryboardProject {
  return {
    schemaVersion: 3, revision: 4, id: "0123456789abcdef", name: "Tea shop", createdAt: 0,
    defaults: { width: 1280, height: 720, fps: 24, imageModel: null, videoModel: null, voice: null, exportPreset: "16:9", resolution: "720p" },
    references: { characters: [], places: [], styles: [] }, shots: [], audio: [],
    export: { aiLabel: { embedTag: true }, loudnessLufs: -16, captionsSrt: false, captionsBurn: false, stems: false, upscale: "none" },
    captions: [], batches: [], undoStack: [], redoStack: [], ...extra,
  } as StoryboardProject;
}

function story(p: StoryboardProject): UseStoryboardResult {
  return {
    projectId: p.id, project: p, timeline: null, loading: false, error: null, notFound: false, notice: null,
    dismissNotice: noop, busy: false, batch: null, reload: async () => undefined, op: vi.fn(async () => true),
    opFrom: vi.fn(async () => true), undo: async () => undefined, redo: async () => undefined,
    startBatch: async () => true, cancelBatch: async () => undefined,
  } as unknown as UseStoryboardResult;
}

function within(store: UnsavedTextStore, node: React.ReactNode) {
  return renderToStaticMarkup(<UnsavedTextContext.Provider value={store}>{node}</UnsavedTextContext.Provider>);
}

describe("unsaved text store (S-O)", () => {
  it("keeps a refused save with its reason and clears it when a later save lands", async () => {
    const store = createUnsavedTextStore();
    const seen: number[] = [];
    store.subscribe(() => seen.push(1));

    store.lastFailure = "conflict";
    trackSave(store, "plan:brief", "My words", Promise.resolve(false));
    await flush();
    expect(store.get("plan:brief")).toEqual({ text: "My words", kind: "conflict" });

    store.lastFailure = "failed";
    trackSave(store, "plan:brief", "My words", Promise.resolve(false));
    await flush();
    expect(store.get("plan:brief")).toEqual({ text: "My words", kind: "failed" });

    trackSave(store, "plan:brief", "My words", Promise.resolve(true));
    await flush();
    expect(store.get("plan:brief")).toBeNull();
    expect(seen.length).toBe(3);
  });

  it("takes a rejected save as not answered, and a caller without a promise as saved", async () => {
    const store = createUnsavedTextStore();
    trackSave(store, "k", "typed", Promise.reject(new Error("offline")));
    await flush();
    expect(store.get("k")).toEqual({ text: "typed", kind: "failed" });
    trackSave(store, "k", "typed", undefined);
    expect(store.get("k")).toBeNull();
  });

  it("keeps each field apart", async () => {
    const store = createUnsavedTextStore();
    store.lastFailure = "conflict";
    trackSave(store, "a", "one", Promise.resolve(false));
    trackSave(store, "b", "two", Promise.resolve(true));
    await flush();
    expect(store.get("a")?.text).toBe("one");
    expect(store.get("b")).toBeNull();
  });
});

describe("Your text was not saved", () => {
  it("offers Put it back after a conflict and Retry when the backend did not answer, with the text", () => {
    const conflict = renderToStaticMarkup(<UnsavedTextNotice entry={{ text: "A calm tea shop", kind: "conflict" }} onPutBack={noop} onDismiss={noop} />);
    expect(conflict).toContain("Your text was not saved");
    expect(conflict).toContain("It changed elsewhere first; the field shows the latest.");
    expect(conflict).toContain("A calm tea shop");
    expect(conflict).toContain(">Put it back</button>");
    expect(conflict).toContain(">Keep the latest</button>");
    const failed = renderToStaticMarkup(<UnsavedTextNotice entry={{ text: "Line one", kind: "failed" }} onPutBack={noop} onDismiss={noop} />);
    expect(failed).toContain("The backend did not answer.");
    expect(failed).toContain(">Retry</button>");
    expect(failed).toContain(">Discard my text</button>");
    expect(failed).not.toContain("Put it back");
  });

  it("shows under the brief, the shot row's fields, a cast card, a caption and a section label when the project kept text", () => {
    const store = createUnsavedTextStore();
    const p = project({ shots: [shot("s1")] });
    store.set("plan:brief", { text: "My own brief", kind: "conflict" });
    store.set("plan:s1:stillPrompt", { text: "My picture prompt", kind: "failed" });
    store.set("ref:c1:description", { text: "Red coat", kind: "conflict" });
    store.set("caption:cap1", { text: "Hello there", kind: "failed" });
    store.set("song:section:1", { text: "Chorus", kind: "conflict" });

    const brief = within(store, (
      <PlanBrief story={story(p)} project={p} imageGallery={[]} canPropose proposing={false} elapsed={0} modelName={null} onPropose={noop} onOpenCast={noop} />
    ));
    expect(brief).toContain("My own brief");
    expect(brief).toContain(">Put it back</button>");
    // The notice sits outside the label, so the brief's name stays "Brief".
    expect(brief.indexOf("unsaved-text")).toBeGreaterThan(brief.indexOf("</label>"));

    const row = within(store, <PlanShotRow story={story(p)} projectId={p.id} shot={p.shots[0]} number={1} canAsk={false} onModelMissing={noop} />);
    expect(row).toContain("My picture prompt");
    expect(row).toContain(">Retry</button>");

    const card = within(store, (
      <ReferenceCard
        projectId={p.id}
        reference={{ id: "c1", kind: "character", name: "Mara", description: "Blue coat", images: [] } as never}
        usedBy={0}
        disabled={false}
        onEdit={() => Promise.resolve(true)}
        onRemove={noop}
        onRemovePicture={noop}
        onOpenPicker={noop}
        onFiles={noop}
      />
    ));
    expect(card).toContain("Red coat");

    const ruler = within(store, (
      <SongRuler seconds={60} sections={[{ start: 0, end: 30, label: "A", energy: 0.4 }, { start: 30, end: 60, label: "B", energy: 0.8 }] as never} onRename={() => Promise.resolve(true)} />
    ));
    expect(ruler).toContain("Chorus");

    const timeline = { duration: 4, shots: [], audio: [], captions: [{ id: "cap1", text: "Hi", start: 0, end: 2 }] };
    const captions = within(store, (
      <CaptionsPanel
        project={project({ captions: [{ id: "cap1", text: "Hi", start: 0, end: 2 }] } as never)}
        timeline={timeline as never}
        state={{ busy: false, notes: [], error: null, file: null }}
        disabled={false}
        playheadS={0}
        revealLabel="Show"
        onGenerate={noop}
        onAdd={noop}
        onEditText={() => Promise.resolve(true)}
        onRemove={noop}
        onExportSrt={noop}
        onReveal={noop}
      />
    ));
    expect(captions).toContain("Hello there");
  });

  it("shows nothing when every save landed", () => {
    const p = project({ shots: [shot("s1")] });
    const html = within(createUnsavedTextStore(), <PlanShotRow story={story(p)} projectId={p.id} shot={p.shots[0]} number={1} canAsk={false} onModelMissing={noop} />);
    expect(html).not.toContain("unsaved-text");
  });
});
