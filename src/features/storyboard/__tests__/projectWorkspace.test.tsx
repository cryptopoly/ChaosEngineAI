import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resolveTabRequest } from "../../../constants/tabRedirects";
import type { ImageOutputArtifact, StoryboardProject, StoryboardReference, StoryboardShot, VoiceEntry } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { sizeFor } from "../formatLogic";
import {
  canAddPicture,
  galleryChoices,
  isPictureFile,
  isSmallPicture,
  MAX_PICTURES,
  nameFromFile,
  picturesFrom,
  shotStillChoices,
  usedByCount,
} from "../pickerLogic";
import {
  LEGACY_OPEN_KEY,
  ROUTE_EVENT,
  ROUTE_KEY,
  applyStoryboardNavigation,
  openProjectAt,
  parseRoute,
  readRoute,
  routeWith,
} from "../projectRoute";
import { filterAndSort, isBundleName, nextTabIndex, readListSort, writeListSort } from "../projectShellLogic";
import { ProjectShellHeader } from "../ProjectShellHeader";
import { PROJECT_TEMPLATES, templateById, templateShotArgs } from "../projectTemplates";
import { ReferenceCard } from "../ReferenceCard";
import { canPlaySample, groupVoices, voiceLabel } from "../voicePickerLogic";

function memoryStorage(): Storage {
  const data = new Map<string, string>();
  return {
    get length() {
      return data.size;
    },
    clear: () => data.clear(),
    getItem: (key) => data.get(key) ?? null,
    key: (index) => Array.from(data.keys())[index] ?? null,
    removeItem: (key) => void data.delete(key),
    setItem: (key, value) => void data.set(key, String(value)),
  };
}

/** A browser-like window for the route code: local storage and events. */
function fakeWindow() {
  return Object.assign(new EventTarget(), { localStorage: memoryStorage() });
}

function shot(id: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id, order: 0, title: "", status: "idle", stillPrompt: "", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}

function take(id: string, still: string | null) {
  return { id, still, clip: null, hashes: {}, settings: {}, measuredSeconds: 4, createdAt: 0, origin: "upload" as const, basis: null };
}

describe("old Storyboard tab ids (D-65)", () => {
  it("land on the one Projects entry: the list, or the last project's Storyboard or Mixer tab", () => {
    expect(resolveTabRequest("story-projects")).toEqual({ tab: "storyboard", storyboard: { projectId: null } });
    expect(resolveTabRequest("story-board")).toEqual({ tab: "storyboard", storyboard: { tab: "storyboard" } });
    expect(resolveTabRequest("story-mixer")).toEqual({ tab: "storyboard", storyboard: { tab: "mixer" } });
    expect(resolveTabRequest({ tab: "story-board" as never, storyboard: { projectId: "p1", tab: "cast" } })).toEqual({
      tab: "storyboard",
      storyboard: { projectId: "p1", tab: "cast" },
    });
    expect(resolveTabRequest("storyboard")).toEqual({ tab: "storyboard" });
  });
});

describe("project route storage", () => {
  beforeEach(() => vi.stubGlobal("window", fakeWindow()));
  afterEach(() => vi.unstubAllGlobals());

  it("parses safely: bad JSON, unknown tabs and shapes fall back to the list", () => {
    expect(parseRoute(null)).toEqual({ projectId: null, tab: "overview", lastTabs: {} });
    expect(parseRoute("{not json")).toEqual({ projectId: null, tab: "overview", lastTabs: {} });
    expect(parseRoute(JSON.stringify({ projectId: "p1", tab: "board", lastTabs: { p1: "mixer", p2: "nope" } }))).toEqual({
      projectId: "p1",
      tab: "mixer",
      lastTabs: { p1: "mixer" },
    });
  });

  it("migrates the old open-project key once, onto the Storyboard tab", () => {
    window.localStorage.setItem(LEGACY_OPEN_KEY, "abc123");
    expect(readRoute()).toEqual({ projectId: "abc123", tab: "storyboard", lastTabs: { abc123: "storyboard" } });
    expect(window.localStorage.getItem(LEGACY_OPEN_KEY)).toBeNull();
    expect(JSON.parse(window.localStorage.getItem(ROUTE_KEY) ?? "{}").projectId).toBe("abc123");
  });

  it("remembers the last tab per project and reopens on it", () => {
    let route = routeWith(parseRoute(null), "p1", "cast");
    route = routeWith(route, "p2");
    expect(route).toMatchObject({ projectId: "p2", tab: "overview" });
    route = routeWith(route, "p1");
    expect(route).toMatchObject({ projectId: "p1", tab: "cast" });
    expect(routeWith(route, null)).toMatchObject({ projectId: null, lastTabs: { p1: "cast", p2: "overview" } });
  });

  it("writes and announces navigations; a tab-only request keeps the open project or the list", () => {
    const heard = vi.fn();
    window.addEventListener(ROUTE_EVENT, heard);
    applyStoryboardNavigation({ tab: "mixer" });
    expect(readRoute().projectId).toBeNull(); // no project was open: the list
    openProjectAt("p9", "export");
    applyStoryboardNavigation({ tab: "storyboard" });
    expect(readRoute()).toMatchObject({ projectId: "p9", tab: "storyboard" });
    applyStoryboardNavigation({ projectId: null });
    expect(readRoute().projectId).toBeNull();
    expect(heard).toHaveBeenCalledTimes(4);
    window.removeEventListener(ROUTE_EVENT, heard);
  });
});

describe("project list and tab keys", () => {
  const cards = [
    { name: "Lighthouse keeper", recency: 30, createdAt: 1, seconds: 32 },
    { name: "Coffee shop ad", recency: 50, createdAt: 3, seconds: 15 },
    { name: "Café night", recency: 10, createdAt: 2, seconds: 60 },
  ];

  it("searches by every word of the name, ignoring case and accents", () => {
    expect(filterAndSort(cards, "cafe", "edited").map((c) => c.name)).toEqual(["Café night"]);
    expect(filterAndSort(cards, "SHOP ad", "edited").map((c) => c.name)).toEqual(["Coffee shop ad"]);
    expect(filterAndSort(cards, "nothing", "edited")).toEqual([]);
  });

  it("sorts by last edited, name, created and length", () => {
    expect(filterAndSort(cards, "", "edited").map((c) => c.name)).toEqual(["Coffee shop ad", "Lighthouse keeper", "Café night"]);
    expect(filterAndSort(cards, "", "name").map((c) => c.name)).toEqual(["Café night", "Coffee shop ad", "Lighthouse keeper"]);
    expect(filterAndSort(cards, "", "created").map((c) => c.name)).toEqual(["Coffee shop ad", "Café night", "Lighthouse keeper"]);
    expect(filterAndSort(cards, "", "length").map((c) => c.name)).toEqual(["Café night", "Lighthouse keeper", "Coffee shop ad"]);
  });

  it("remembers the sort, and only valid ones", () => {
    vi.stubGlobal("window", fakeWindow());
    expect(readListSort()).toBe("edited");
    writeListSort("length");
    expect(readListSort()).toBe("length");
    window.localStorage.setItem("chaos.storyboard.listSort", "bogus");
    expect(readListSort()).toBe("edited");
    vi.unstubAllGlobals();
  });

  it("moves between tabs with arrows, Home and End, wrapping round", () => {
    expect(nextTabIndex(0, 5, "ArrowRight")).toBe(1);
    expect(nextTabIndex(4, 5, "ArrowRight")).toBe(0);
    expect(nextTabIndex(0, 5, "ArrowLeft")).toBe(4);
    expect(nextTabIndex(2, 5, "Home")).toBe(0);
    expect(nextTabIndex(2, 5, "End")).toBe(4);
    expect(nextTabIndex(2, 5, "Enter")).toBeNull();
  });

  it("knows a bundle by its name", () => {
    expect(isBundleName("Film.chaosfilm")).toBe(true);
    expect(isBundleName("film.ZIP")).toBe(true);
    expect(isBundleName("film.mp4")).toBe(false);
  });
});

describe("picture picker sources", () => {
  const gallery = [
    { artifactId: "a1", prompt: "red lighthouse", modelName: "FLUX", previewUrl: "blob:1", imagePath: "/out/a1.png", projectId: null },
    { artifactId: "a2", prompt: "keeper portrait", modelName: "FLUX", previewUrl: "blob:2", imagePath: "/out/a2.png", projectId: "p1" },
    { artifactId: "a3", prompt: "no file", modelName: "FLUX", previewUrl: "blob:3", imagePath: null, projectId: null },
  ] as unknown as ImageOutputArtifact[];

  it("offers saved images with a file, this project's first, and searches the prompt", () => {
    expect(galleryChoices(gallery, "", "p1").map((c) => c.id)).toEqual(["a2", "a1"]);
    expect(galleryChoices(gallery, "lighthouse", "p1").map((c) => c.path)).toEqual(["/out/a1.png"]);
  });

  it("offers every shot still, the active take first, labelled by shot and take", () => {
    const project = {
      shots: [
        shot("s1", { takes: [take("t1", "media/shots/s1/t1/still.png"), take("t2", "media/shots/s1/t2/still.png")], activeTake: "t2" }),
        shot("s2", { takes: [take("t3", null)] }),
        shot("s3", { takes: [take("t4", "media/shots/s3/t4/still.png")], activeTake: "t4" }),
      ],
    };
    expect(shotStillChoices(project).map((c) => [c.shotNumber, c.takeNumber, c.active])).toEqual([
      [1, 2, true],
      [1, 1, false],
      [3, 1, true],
    ]);
  });

  it("takes PNG, JPEG and WebP files, up to the room left", () => {
    const file = (name: string, type: string) => ({ name, type }) as File;
    expect(isPictureFile(file("a.png", "image/png"))).toBe(true);
    expect(isPictureFile(file("a.gif", "image/gif"))).toBe(false);
    expect(isPictureFile(file("a.webp", ""))).toBe(true);
    const files = [file("1.png", "image/png"), file("x.txt", "text/plain"), file("2.jpg", "image/jpeg"), file("3.webp", "image/webp")];
    expect(picturesFrom(files, 2).map((f) => f.name)).toEqual(["1.png", "2.jpg"]);
    expect(picturesFrom(files, 0)).toEqual([]);
  });

  it("holds four pictures, notes a small one, names a dropped file and counts users", () => {
    const ref = (n: number) => ({ images: Array.from({ length: n }, (_, i) => ({ path: `media/refs/${i}.png`, hash: null })) });
    expect(MAX_PICTURES).toBe(4);
    expect(canAddPicture(ref(3))).toBe(true);
    expect(canAddPicture(ref(4))).toBe(false);
    expect(isSmallPicture(400, 900)).toBe(true);
    expect(isSmallPicture(512, 512)).toBe(false);
    expect(nameFromFile("old_man-front.PNG")).toBe("old man front");
    expect(usedByCount({ shots: [shot("a", { referenceImages: ["r1"] }), shot("b")] }, "r1")).toBe(1);
  });
});

describe("voice picker", () => {
  const voice = (id: string, kind: VoiceEntry["kind"], extra: Partial<VoiceEntry> = {}): VoiceEntry => ({
    id, name: id, kind, language: "en-US", model: "kokoro", models: ["kokoro"],
    capabilities: { speak: true, style: false, clone: false, design: false }, available: true, ...extra,
  });
  const voices = [voice("Zed", "clone"), voice("Bella", "builtin"), voice("Anna", "builtin"), voice("Nora", "design"), voice("Mo", "preset", { available: false, needs: "Needs the Qwen3-TTS model" })];

  it("groups built-in, preset, designed and cloned voices in that order, sorted, with a search", () => {
    expect(groupVoices(voices, "").map((g) => [g.kind, g.voices.map((v) => v.id)])).toEqual([
      ["builtin", ["Anna", "Bella"]],
      ["preset", ["Mo"]],
      ["design", ["Nora"]],
      ["clone", ["Zed"]],
    ]);
    expect(groupVoices(voices, "bel").map((g) => g.voices.map((v) => v.id))).toEqual([["Bella"]]);
  });

  it("plays a sample only for an available, uncloned voice; labels the stored id", () => {
    expect(canPlaySample(voices[1])).toBe(true);
    expect(canPlaySample(voices[0])).toBe(false);
    expect(canPlaySample(voices[4])).toBe(false);
    expect(voiceLabel("Bella", voices)).toBe("Bella");
    expect(voiceLabel("gone", voices)).toBe("gone");
    expect(voiceLabel(null, voices)).toBeNull();
  });
});

describe("new project templates", () => {
  it("starts with Blank, and every template has a valid format and shots", () => {
    expect(PROJECT_TEMPLATES[0].id).toBe("blank");
    expect(PROJECT_TEMPLATES[0].shots).toEqual([]);
    for (const template of PROJECT_TEMPLATES) {
      expect(sizeFor(template.aspect, template.resolution).width).toBeGreaterThan(0);
      expect([24, 25, 30, 48, 50, 60]).toContain(template.fps);
      for (const s of template.shots) expect(s.seconds).toBeGreaterThanOrEqual(0.5);
    }
  });

  it("vertical social is 9:16 at 1080p, and the scene template needs no character", () => {
    const vertical = templateById("vertical");
    expect([vertical.aspect, vertical.resolution]).toEqual(["9:16", "1080p"]);
    expect(sizeFor(vertical.aspect, vertical.resolution)).toEqual({ width: 1080, height: 1920 });
    const broll = templateById("broll");
    expect(broll.shots.length).toBeGreaterThan(0);
    expect(templateById("missing").id).toBe("blank");
    const args = templateShotArgs(broll, (s) => ({ title: s.title.toUpperCase(), stillPrompt: s.stillPrompt }));
    expect(args[0]).toEqual({ title: broll.shots[0].title.toUpperCase(), stillPrompt: broll.shots[0].stillPrompt, seconds: broll.shots[0].seconds });
  });
});

describe("project shell header and cast card", () => {
  const project = {
    id: "p1", name: "Lighthouse", revision: 3, undoStack: [{}], redoStack: [],
    defaults: { exportPreset: "9:16", resolution: "2160p", fps: 30 },
    references: { characters: [{ id: "r1" }], places: [] },
    shots: [shot("s1", { status: "stale" })],
  } as unknown as StoryboardProject;

  it("renders five tabs as a tab list with the active one selected and counts on Cast and Storyboard", () => {
    const html = renderToStaticMarkup(
      <ProjectShellHeader name="Lighthouse" project={project} tab="cast" busy={false} conflict
        onDismissConflict={() => undefined} onBack={() => undefined} onRename={() => undefined}
        onUndo={() => undefined} onRedo={() => undefined} onTab={() => undefined} />,
    );
    expect(html).toContain('role="tablist"');
    expect(html.match(/role="tab"/g)).toHaveLength(5);
    expect(html).toMatch(/aria-selected="true"[^>]*>Cast/);
    expect(html).toContain("1 stale");
    expect(html).toContain("9:16");
    expect(html).toContain("4K");
    expect(html).toContain("30 fps");
    expect(html).toContain("Changed elsewhere; showing the latest");
    expect(html).toContain("Projects");
  });

  it("shows up to four pictures and an Add picture slot only while there is room", () => {
    const ref = (n: number): StoryboardReference => ({
      id: "r1", name: "Mia", description: "", images: Array.from({ length: n }, (_, i) => ({ path: `media/refs/${i}.png`, hash: null })),
    });
    const props = { projectId: "p1", usedBy: 2, disabled: false, onEdit: () => undefined, onRemove: () => undefined,
      onRemovePicture: () => undefined, onOpenPicker: () => undefined, onFiles: () => undefined };
    const two = renderToStaticMarkup(<ReferenceCard {...props} reference={ref(2)} />);
    expect(two).toContain("Add a picture of Mia");
    expect(two).toContain("2 of 4");
    expect(two.match(/Remove picture/g)?.length).toBe(4); // label and title of two buttons
    expect(two).toContain("Used by 2 shots");
    const full = renderToStaticMarkup(<ReferenceCard {...props} reference={ref(4)} />);
    expect(full).not.toContain("Add a picture of Mia");
  });
});

describe("shell states and the picker", () => {
  it("handles offline, not found, error and loading once for every tab", async () => {
    const { shellState } = await import("../ProjectShell");
    const base = { backendOnline: true, notFound: false, loaded: false, error: null };
    expect(shellState({ ...base, backendOnline: false })).toBe("offline");
    expect(shellState({ ...base, notFound: true })).toBe("notFound"); // a stored project that was deleted
    expect(shellState({ ...base, error: "no route" })).toBe("error");
    expect(shellState(base)).toBe("loading");
    expect(shellState({ ...base, loaded: true, error: "a later op failed" })).toBe("ready");
  });

  it("offers this computer, the gallery and shot stills, with a paste box and Paste from clipboard", async () => {
    const { ReferenceImagePicker } = await import("../ReferenceImagePicker");
    const project = { id: "p1", shots: [shot("s1", { takes: [take("t1", "media/shots/s1/t1/still.png")], activeTake: "t1" })] } as unknown as StoryboardProject;
    const props = { title: "Add a picture of Mia", project, imageGallery: [], room: 4, onPick: async () => true, onClose: () => undefined };
    const html = renderToStaticMarkup(<ReferenceImagePicker {...props} />);
    expect(html).toContain('role="dialog"');
    expect(html.match(/role="tab"/g)).toHaveLength(3);
    expect(html).toMatch(/aria-selected="true"[^>]*>From a shot/); // a project with stills opens on them
    expect(html).toContain("Shot 1 · take 1");
    expect(html).toContain("Paste a path instead");
    const empty = { ...project, shots: [] } as unknown as StoryboardProject;
    const first = renderToStaticMarkup(<ReferenceImagePicker {...props} project={empty} />);
    expect(first).toMatch(/aria-selected="true"[^>]*>This computer/);
    expect(first).toContain('aria-label="Paste a picture here"');
    expect(first).toContain("Paste from clipboard");
  });
});
