import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { StoryboardProject, StoryboardShot } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { BrandKitPanel } from "../BrandKitPanel";
import { PlanDirection } from "../PlanDirection";
import { PlanModelBar } from "../PlanModelBar";
import { PlanProposal } from "../PlanProposal";
import { PlanShotRow } from "../PlanShotRow";
import { PlanTab } from "../PlanTab";
import { ReferenceCard } from "../ReferenceCard";
import type { UseStoryboardResult } from "../useStoryboard";

const noop = () => undefined;

function shot(id: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id, order: 0, title: "", status: "idle", stillPrompt: "", motionPrompt: "", referenceImages: [],
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

const proposal = {
  model: "Qwen3.8-27B", seconds: 41.2, sawPictures: false, note: "A slow morning ritual.",
  shots: [
    { title: "Steam", stillPrompt: "A teapot on a counter", motionPrompt: "steam rises", seconds: 6 },
    { title: "Pour", stillPrompt: "Tea poured into a cup", motionPrompt: "", seconds: 8 },
  ],
};

afterEach(() => vi.unstubAllGlobals());

describe("Plan tab parts (S8)", () => {
  it("offers to load a chat model when none is loaded", () => {
    const html = renderToStaticMarkup(
      <PlanModelBar chatModel={null} known choices={[{ key: "a", label: "Qwen3.8-27B", detail: "MLX 4-bit" }]} loading={false} onLoad={noop} />,
    );
    expect(html).toContain("No chat model is loaded.");
    expect(html).toContain("Qwen3.8-27B · MLX 4-bit");
    expect(html).toMatch(/class="primary-button">Load</);
    const none = renderToStaticMarkup(<PlanModelBar chatModel={null} known choices={[]} loading={false} onLoad={noop} />);
    expect(none).toContain("No chat model is installed yet.");
    const checking = renderToStaticMarkup(<PlanModelBar chatModel={null} known={false} choices={[]} loading={false} />);
    expect(checking).toContain("Checking the chat model");
  });

  it("names the loaded model and whether it reads pictures", () => {
    const html = renderToStaticMarkup(
      <PlanModelBar chatModel={{ name: "Gemma 4", ref: "g", vision: true }} known choices={[]} loading={false} />,
    );
    expect(html).toContain("Planning with Gemma 4");
    expect(html).toContain("reads your pictures");
  });

  it("shows a proposal with its length against the aim and one Use button on an empty project", () => {
    const html = renderToStaticMarkup(
      <PlanProposal proposal={proposal} project={project()} target={15} busy={false} onUse={noop} onAgain={noop} onDiscard={noop} />,
    );
    expect(html).toContain("2 shots · 14 s");
    expect(html).toContain("About 15 s, as asked");
    expect(html).toContain("Use these 2 shots");
    expect(html).not.toContain("Replace the shots");
    expect(html).toContain("Proposed by Qwen3.8-27B in 41 s");
    expect(html).toContain("A slow morning ritual.");
  });

  it("offers Add after and Replace on a project with shots, and holds Replace while a shot is locked", () => {
    const html = renderToStaticMarkup(
      <PlanProposal proposal={proposal} project={project({ shots: [shot("a"), shot("b", { locked: true })] })} target={60} busy={false} onUse={noop} onAgain={noop} onDiscard={noop} />,
    );
    expect(html).toContain("Add after the 2 shots");
    expect(html).toMatch(/disabled=""[^>]*>Replace the shots/);
    expect(html).toContain("1 shot is locked, so Replace is off.");
    expect(html).toContain("Shorter than the 60 s asked");
  });

  it("edits a shot in place and says how the style went in", () => {
    const p = project();
    const s = shot("a", {
      title: "Steam", stillPrompt: "A teapot", motionPrompt: "steam rises", seconds: 6,
      takes: [{ id: "t", still: "media/a/t/still.png", clip: null, hashes: {}, settings: { style: "words" }, measuredSeconds: null, createdAt: 0, origin: "batch", basis: null }],
      activeTake: "t",
    });
    const html = renderToStaticMarkup(<PlanShotRow story={story(p)} projectId={p.id} shot={s} number={3} canAsk onModelMissing={noop} />);
    expect(html).toContain('value="Steam"');
    expect(html).toContain('aria-label="Picture prompt of shot 3"');
    expect(html).toContain("Style as words");
    expect(html).toContain("Rework with the model");
    expect(html).toContain('aria-expanded="false"');
  });

  it("shows the direction chat log and an empty hint", () => {
    const empty = renderToStaticMarkup(<PlanDirection story={story(project())} project={project()} canAsk onModelMissing={noop} />);
    expect(empty).toContain("make it feel like early morning");
    const p = project({ planChat: [{ id: "m1", role: "user", text: "Warmer please", at: 1 }, { id: "m2", role: "assistant", text: "Warmer light in every shot.", at: 2 }] });
    const html = renderToStaticMarkup(<PlanDirection story={story(p)} project={p} canAsk onModelMissing={noop} />);
    expect(html).toContain("plan-chat__line--user");
    expect(html).toContain("Warmer light in every shot.");
    expect(html).toContain(">Clear<");
  });

  it("renders the whole tab while the status loads", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => undefined)));
    const p = project({ plan: { brief: "A tea shop advert", filmSeconds: 60 }, shots: [shot("a", { title: "Steam" })] });
    const html = renderToStaticMarkup(<PlanTab story={story(p)} project={p} imageGallery={[]} chat={{}} onTab={noop} />);
    expect(html).toContain("Checking the chat model");
    expect(html).toContain("A tea shop advert");
    expect(html).toMatch(/aria-pressed="true"[^>]*>1 min</);
    expect(html).toContain("Propose new shots");
    expect(html).toContain("Direction");
    expect(html).toContain("Make them in Storyboard");
  });
});

describe("style and brand kit (S15)", () => {
  it("shows a style's colours with a way to take one out", () => {
    const reference = { id: "r1", name: "Look", description: "", images: [], palette: ["#0a3d1a", "#f0e6c8"] };
    const html = renderToStaticMarkup(
      <ReferenceCard projectId="0123456789abcdef" reference={reference} usedBy={0} disabled={false} onEdit={noop} onRemove={noop}
        onRemovePicture={noop} onOpenPicker={noop} onFiles={noop} palette={reference.palette} onRemoveColour={noop}
        descriptionPlaceholder="The look in words: mood, materials, light" />,
    );
    expect(html).toContain("Remove colour #0a3d1a");
    expect(html).toContain("The look in words");
    expect(html).not.toContain("Not used by a shot yet");
  });

  it("renders the brand kit with the style use, logo, caption look and preview", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => undefined)));
    const p = project({
      references: { characters: [], places: [], styles: [{ id: "r1", name: "Look", description: "", images: [], palette: ["#0a3d1a"] }] },
      brand: { logo: null, logoCorner: "bottom-right", captions: { font: "Avenir", colour: "#0a3d1a", position: "top" }, styleUse: "words" },
    });
    const html = renderToStaticMarkup(<BrandKitPanel story={story(p)} project={p} imageGallery={[]} />);
    expect(html).toContain("How shots use the style");
    expect(html).toMatch(/checked=""[^>]*value="words"|value="words"[^>]*checked=""/);
    expect(html).toContain("Add logo");
    expect(html).toContain('<option value="Avenir" selected="">Avenir</option>');
    expect(html).toMatch(/aria-pressed="true"[^>]*>Top</);
    expect(html).toContain("brand-kit__frame-caption--top");
    expect(html).toContain("Fresh leaves, poured slowly.");
    expect(html).toContain("Add a style picture to see the caption on it.");
  });
});
