import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { MusicInstallJob, MusicLibraryItem, MusicModelFamily, MusicModelVariant, MusicRuntimeStatus } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { buildMusicEntries, measuredSpeed, musicEntryMatchesQuery } from "../../../components/models/adapters/music";
import type { UseMusicState } from "../../../hooks/useMusicState";
import { INITIAL_GENERATION, classifyMusicError, defaultForm, generationReducer } from "../musicLogic";
import { MusicLibraryTab } from "../MusicLibraryTab";
import { MusicModelsPage } from "../MusicModelsPage";
import { MusicPackBanner } from "../MusicPackBanner";
import { MusicStudioTab } from "../MusicStudioTab";
import { MusicOptionsBody, expertSummary, optionsSummary } from "../MusicStudioOptions";

function variant(over: Partial<MusicModelVariant> = {}): MusicModelVariant {
  return {
    id: "ACE-Step/Ace-Step1.5",
    familyId: "ace-step-1.5",
    name: "ACE-Step 1.5 turbo",
    provider: "ACE-Step",
    repo: "ACE-Step/Ace-Step1.5",
    link: "https://huggingface.co/ACE-Step/Ace-Step1.5",
    engine: "acestep",
    engineModel: "acestep-v15-turbo",
    makes: ["songs", "instrumental"],
    platforms: ["apple-silicon", "cuda"],
    sizeGb: 10.09,
    sizeNote: "Includes the planner LM.",
    maxSeconds: 600,
    minSeconds: 10,
    licence: {
      id: "mit",
      name: "MIT",
      commercial: "yes",
      gated: false,
      summaryLines: ["Weights are MIT-licensed.", "You are responsible for rights in lyrics you type."],
    },
    estimates: { basis: "measured", secondsPer10sAudio: 1, peakGb: 22.9 },
    defaultSettings: { seconds: 60 },
    availableLocally: true,
    missingRequires: [],
    note: "Default music model.",
    fit: { status: "fits", needGb: 22.9, usableGb: 48, headroomPct: 52, reason: "Peak memory 22.9 GB while generating", basis: "estimated" },
    ...over,
  };
}

function family(variants: MusicModelVariant[]): MusicModelFamily {
  return { id: "ace-step-1.5", name: "ACE-Step 1.5", provider: "ACE Studio", defaultVariantId: variants[0].id, bestFitVariantId: variants[0].id, variants };
}

const IDLE_JOB: MusicInstallJob = {
  phase: "idle", phaseLabel: "", phaseIndex: 0, phaseTotal: 0, percent: 0, message: "", error: null, log: [], startedAt: 0, finishedAt: 0, running: false,
};

function runtime(over: Partial<MusicRuntimeStatus> = {}): MusicRuntimeStatus {
  return {
    lane: "mlx", unavailableReason: null, installed: true, engine: "acestep", workerRunning: false, loadedModel: null,
    idleUnloadMinutes: 10, embedTag: true, libraryDir: "/data/music/library", settings: { idleUnloadMinutes: 10 }, install: IDLE_JOB,
    ...over,
  };
}

function item(over: Partial<MusicLibraryItem> = {}): MusicLibraryItem {
  return {
    id: "1700000000_aaaaaaaa", kind: "track", title: "Lo-fi piano", caption: "warm lo-fi hip hop", lyrics: "", tags: ["lo-fi"],
    createdAt: 1_700_000_000, updatedAt: 1_700_000_000, audio: ".flac", durationS: 30, seed: 8142, instrumental: true, modelName: "ACE-Step 1.5 turbo", sizeBytes: 6_240_000,
    ...over,
  };
}

const noop = () => undefined;
const asyncNoop = async () => undefined;

function fakeMusic(over: Record<string, unknown> = {}): UseMusicState {
  const v = variant();
  const base = {
    families: [family([v])],
    variants: [v],
    runtime: runtime(),
    downloads: {},
    installJob: IDLE_JOB,
    catalogLoaded: true,
    selectedModelId: null,
    setSelectedModelId: noop,
    selectedVariant: v,
    installedVariants: [v],
    hasInstalledModel: true,
    refreshMusicData: asyncNoop,
    refreshRuntime: asyncNoop,
    refreshCatalog: asyncNoop,
    startInstall: asyncNoop,
    cancelInstall: asyncNoop,
    setEmbedTag: asyncNoop,
    setIdleMinutes: asyncNoop,
    startDownload: asyncNoop,
    pauseDownload: asyncNoop,
    deleteDownload: asyncNoop,
    library: {
      items: [] as MusicLibraryItem[], diskBytes: 0, loaded: true, refreshLibrary: asyncNoop,
      renameItem: asyncNoop, setTags: asyncNoop, deleteItems: async () => [] as string[],
    },
    studio: {
      form: { ...defaultForm(v), caption: "lo-fi piano" },
      setForm: noop,
      replaceForm: noop,
      estimate: null,
      generation: INITIAL_GENERATION,
      running: false,
      progress: null,
      generate: asyncNoop,
      cancel: asyncNoop,
      dismissError: noop,
      forgetResults: noop,
    },
    openStudio: noop,
    makeAnotherLike: noop,
    readyGroups: new Set(),
    ...over,
  };
  return base as unknown as UseMusicState;
}

describe("music model entries", () => {
  it("maps a catalog row to a Models page entry", () => {
    const [entry] = buildMusicEntries([family([variant()])], { loadedEngineModel: "acestep-v15-turbo" });
    expect(entry).toMatchObject({
      area: "music",
      id: "ACE-Step/Ace-Step1.5",
      name: "ACE-Step 1.5 turbo",
      downloadKey: "ACE-Step/Ace-Step1.5",
      availableLocally: true,
      loaded: true,
      gated: false,
      license: "MIT",
      sizeLabel: "10.1 GB",
      bestFit: true,
      hostCompatible: true,
    });
    expect(entry.fit?.status).toBe("fits");
    expect(entry.memory?.label).toBe("~23 GB peak");
  });

  it("is not installed while a required model is missing, and not loaded when another is", () => {
    const [entry] = buildMusicEntries([family([variant({ missingRequires: ["x"] })])], { loadedEngineModel: "other" });
    expect(entry.availableLocally).toBe(false);
    expect(entry.hasLocalData).toBe(true);
    expect(entry.loaded).toBe(false);
  });

  it("reads the measured speed and searches name, licence and what it makes", () => {
    expect(measuredSpeed(variant())).toEqual({ measured: true, rate: 10 });
    expect(measuredSpeed(variant({ estimates: { basis: "estimated" } }))).toEqual({ measured: false, rate: null });
    const [entry] = buildMusicEntries([family([variant()])]);
    expect(musicEntryMatchesQuery(entry, "mit")).toBe(true);
    expect(musicEntryMatchesQuery(entry, "vocals")).toBe(true);
    expect(musicEntryMatchesQuery(entry, "flux")).toBe(false);
    expect(musicEntryMatchesQuery(entry, "")).toBe(true);
  });
});

describe("MusicPackBanner", () => {
  const banner = (rt: MusicRuntimeStatus | null, job: MusicInstallJob | null) =>
    renderToStaticMarkup(<MusicPackBanner music={{ runtime: rt, installJob: job, startInstall: asyncNoop, cancelInstall: asyncNoop } as never} />);

  it("offers the install while the pack is missing, with its size", () => {
    const html = banner(runtime({ installed: false }), IDLE_JOB);
    expect(html).toContain("Install the music pack");
    expect(html).toContain("About 1.6 GB");
  });

  it("shows progress, Cancel and the log while installing", () => {
    const html = banner(runtime({ installed: false }), { ...IDLE_JOB, running: true, percent: 40, phaseLabel: "Installing ACE-Step", log: ["$ pip install"] });
    expect(html).toContain("Cancel");
    expect(html).toContain("Installing ACE-Step");
    expect(html).toContain("$ pip install");
    expect(html).toContain('value="40"');
  });

  it("says it in plain words when the install failed, with technical details", () => {
    const html = banner(runtime({ installed: false }), { ...IDLE_JOB, error: "pip exited with 1", log: ["ERROR: no wheel"] });
    expect(html).toContain("The music pack could not be installed.");
    expect(html).toContain("Technical details");
    expect(html).toContain("pip exited with 1");
    expect(html).toContain("Retry");
  });

  it("is gone once the pack is installed", () => {
    expect(banner(runtime(), IDLE_JOB)).toBe("");
    expect(banner(null, null)).toBe("");
  });
});

describe("MusicModelsPage", () => {
  const render = (music: UseMusicState) =>
    renderToStaticMarkup(<MusicModelsPage music={music} backendOnline availability="all" onAvailabilityChange={noop} onOpenExternalUrl={noop} />);

  it("shows the card fields: name, makes, licence chip, size, fit and status", () => {
    const html = render(fakeMusic());
    expect(html).toContain("ACE-Step 1.5 turbo");
    expect(html).toContain("Best fit");
    expect(html).toContain("Songs with vocals");
    expect(html).toContain("MIT · commercial");
    expect(html).toContain("10.1 GB");
    expect(html).toContain("Fits");
    expect(html).toContain("Installed");
    expect(html).toContain("Open in Studio");
    expect(html).toContain("Apple Silicon");
  });

  it("says Music needs an Apple Silicon Mac and shows the backend's reason, and hides the install banner", () => {
    const html = render(fakeMusic({ runtime: runtime({ installed: false, unavailableReason: "Music runs on Apple Silicon (MLX) in this version." }) }));
    expect(html).toContain("Music needs an Apple Silicon Mac for now.");
    expect(html).toContain("Music runs on Apple Silicon (MLX) in this version.");
    expect(html).not.toContain("Install the music pack");
  });

  it("shows the pack banner and the first-run line on a fresh machine", () => {
    const fresh = variant({ availableLocally: false, estimates: { basis: "measured", secondsPer10sAudio: 1, peakGb: 22.9 } });
    const html = render(fakeMusic({ runtime: runtime({ installed: false }), families: [family([fresh])], variants: [fresh], selectedVariant: fresh, installedVariants: [], hasInstalledModel: false }));
    expect(html).toContain("Install the music pack");
    expect(html).toContain("Pick a music model to start.");
    expect(html).toContain("Download ACE-Step 1.5 turbo");
  });
});

describe("MusicStudioTab", () => {
  const render = (music: UseMusicState, backendOnline = true) =>
    renderToStaticMarkup(<MusicStudioTab music={music} backendOnline={backendOnline} onTabChange={noop} />);

  it("says it is waiting when the backend is offline, with no form", () => {
    const html = render(fakeMusic(), false);
    expect(html).toContain("Waiting for the backend…");
    expect(html).not.toContain("Make track");
  });

  it("points to Music Models when no model is installed, with the size", () => {
    const v = variant({ availableLocally: false });
    const html = render(fakeMusic({ variants: [v], selectedVariant: v, installedVariants: [], hasInstalledModel: false }));
    expect(html).toContain("No music model installed yet.");
    expect(html).toContain("Open Music Models");
    expect(html).toContain("about 10.1 GB");
    expect(html).not.toContain("Describe the music");
  });

  it("points to the pack when it is not installed, and shows the backend text on a machine that cannot run music", () => {
    expect(render(fakeMusic({ runtime: runtime({ installed: false }) }))).toContain("The music pack is not installed yet.");
    expect(render(fakeMusic({ runtime: runtime({ unavailableReason: "Music runs on Apple Silicon (MLX)." }) }))).toContain("Music runs on Apple Silicon (MLX).");
  });

  it("shows the simple view: mode, description, lyrics, length, estimate and one action", () => {
    const html = render(fakeMusic());
    expect(html).toContain("Describe the music");
    expect(html).toContain("Lyrics");
    expect(html).toContain("You are responsible for rights in lyrics you type.");
    for (const pill of ["15 s", "30 s", "1 min", "2 min", "Custom"]) expect(html).toContain(pill);
    expect(html).toContain("Song");
    expect(html).toContain("Instrumental");
    expect(html).toContain("Make track");
    expect(html).toContain("Time: unknown until the first run");
    expect(html).toContain("11 / 512");
    // The expert and options disclosures are there, collapsed, with a value summary.
    expect(html).toContain("BPM auto · Key auto · 1 variation · Random seed");
    expect(html).toContain("Model defaults");
    // No faked quality control.
    expect(html).not.toContain("Quality");
  });

  it("hides the lyrics for an instrumental and asks for lyrics for a song", () => {
    const v = variant();
    const instrumental = fakeMusic({ studio: { ...fakeMusic().studio, form: { ...defaultForm(v), mode: "instrumental", caption: "piano" } } });
    expect(render(instrumental)).not.toContain("[verse] ... [chorus] ...");
    expect(render(fakeMusic())).toContain("Add lyrics, or switch to Instrumental.");
  });

  it("warns, without blocking, when the description names a well-known artist", () => {
    const v = variant();
    const html = render(fakeMusic({ studio: { ...fakeMusic().studio, form: { ...defaultForm(v), mode: "instrumental", caption: "like Daft Punk" } } }));
    expect(html).toContain("Naming an artist can produce a copy of their style.");
    expect(html).toMatch(/<button[^>]*class="primary-button"(?![^>]*disabled)[^>]*>Make track/);
  });

  it("renders the estimate from the backend with the basis chip", () => {
    const estimate = {
      estimate: { model: "m", seconds: 35, coldStartSeconds: 25, warm: false, peakGb: 22.9, idleGb: 15, writesMb: 6, modelDiskGb: 10, basis: "measured" },
      fit: null, totalGb: 64, queue: { running: null, waiting: [] },
    };
    const html = render(fakeMusic({ studio: { ...fakeMusic().studio, estimate } }));
    expect(html).toContain("About 35 s");
    expect(html).toContain("Peak 22.9 GB of 64 GB");
    expect(html).toContain("Writes 6 MB");
    expect(html).toContain("Measured");
  });

  it("shows the progress with phases, Cancel and the queue line while running", () => {
    const v = variant();
    const form = { ...defaultForm(v), caption: "piano", mode: "instrumental" as const };
    const generation = generationReducer(INITIAL_GENERATION, { type: "start", at: Date.now(), form });
    const progress = { kind: "music", active: true, phase: "compose", message: "", step: 3, totalSteps: 8, startedAt: 0, updatedAt: 0, elapsedSeconds: 4, runLabel: "60 s · ACE-Step", queue: { waiting: false, position: 0, agentAhead: false } };
    const html = render(fakeMusic({ studio: { ...fakeMusic().studio, form, generation, running: true, progress } }));
    for (const phase of ["Load model", "Encode", "Compose", "Decode audio", "Save"]) expect(html).toContain(phase);
    expect(html).toContain("step 3 / 8");
    expect(html).toContain("live-progress--audio");
    expect(html).toContain("Cancel");
    expect(html).toContain("Making…");
    const queued = render(fakeMusic({ studio: { ...fakeMusic().studio, form, generation, running: true, progress: { ...progress, active: false, queue: { waiting: true, position: 2, agentAhead: false } } } }));
    expect(queued).toContain("Waiting for another job to finish (position 2)");
  });

  it("shows an error with its fix button and technical details", () => {
    const v = variant();
    const form = { ...defaultForm(v), caption: "piano", mode: "instrumental" as const, seconds: 600 };
    let generation = generationReducer(INITIAL_GENERATION, { type: "start", at: 1, form });
    generation = generationReducer(generation, { type: "failed", error: classifyMusicError("Not enough free memory: 23 GB needed.", 503) });
    const html = render(fakeMusic({ studio: { ...fakeMusic().studio, form, generation } }));
    expect(html).toContain("There is not enough free memory for this track. Try a shorter length.");
    expect(html).toContain("Not enough free memory: 23 GB needed.");
    expect(html).toContain("Use a shorter length");
    expect(html).toContain("Technical details");
    expect(html).toContain("HTTP 503");
  });

  it("shows the result card with the measured line and the actions, and the empty line before the first run", () => {
    expect(render(fakeMusic())).toContain("No music yet. Describe a track and press Make track.");
    const v = variant();
    const form = { ...defaultForm(v), caption: "piano", mode: "instrumental" as const };
    let generation = generationReducer(INITIAL_GENERATION, { type: "start", at: 1, form });
    generation = generationReducer(generation, { type: "succeeded", results: [{ item: item(), jobSeconds: 5, seed: 8142 }] });
    const html = render(fakeMusic({ studio: { ...fakeMusic().studio, form, generation } }));
    expect(html).toContain("Lo-fi piano");
    expect(html).toContain("Done in 5 s · 6.2 MB · seed 8142");
    for (const label of ["Open in Library", "Again", "Vary"]) expect(html).toContain(label);
  });
});

describe("Studio disclosures", () => {
  const t = (key: string, options?: Record<string, unknown>) => {
    const text = String(options?.defaultValue ?? key);
    return text.replace(/\{(\w+)(?:, plural, one \{# variation\} other \{# variations\})?\}/g, (_m, name) => String(options?.[name] ?? ""));
  };

  it("summarises the non-default option values", () => {
    const base = defaultForm(variant());
    expect(optionsSummary(t as never, { ...base, bpm: 85, key: "A minor", seed: 7 })).toContain("BPM 85");
    expect(optionsSummary(t as never, { ...base, bpm: 85, key: "A minor", seed: 7 })).toContain("Seed 7");
  });

  it("summarises Expert as Model defaults until something differs", () => {
    expect(expertSummary(t as never, runtime(), variant(), 1)).toBe("Model defaults");
    expect(expertSummary(t as never, runtime({ embedTag: false }), variant(), 1)).toContain("AI tag off");
  });

  it("offers variations 1, 2 and 4 with the 'adds the same time again' note", () => {
    const html = renderToStaticMarkup(<MusicOptionsBody form={defaultForm(variant())} onChange={noop} estimateSeconds={null} />);
    expect(html).toContain("Each variation adds the same time again.");
    expect(html).not.toContain(">3<");
    for (const n of ["1", "2", "4"]) expect(html).toContain(`<span class="preset-pill-label">${n}</span>`);
  });
});

describe("MusicLibraryTab", () => {
  const render = (music: UseMusicState, backendOnline = true) =>
    renderToStaticMarkup(<MusicLibraryTab music={music} backendOnline={backendOnline} onTabChange={noop} />);

  it("shows the empty state with a button to the Studio", () => {
    const html = render(fakeMusic());
    expect(html).toContain("No music yet. Describe a track and press Make track.");
    expect(html).toContain("Open Studio");
  });

  it("shows loading and offline lines", () => {
    expect(render(fakeMusic({ library: { ...fakeMusic().library, loaded: false } }))).toContain("Loading your library…");
    expect(render(fakeMusic(), false)).toContain("Waiting for the backend…");
  });

  it("shows a card with title, tags, prompt, meta and the actions", () => {
    const music = fakeMusic({ library: { ...fakeMusic().library, items: [item()], diskBytes: 6_240_000 } });
    const html = render(music);
    expect(html).toContain("Lo-fi piano");
    expect(html).toContain("1 track");
    expect(html).toContain("0 sounds");
    expect(html).toContain("6 MB on disk");
    expect(html).toContain("warm lo-fi hip hop");
    expect(html).toContain("ACE-Step 1.5 turbo");
    expect(html).toContain("seed 8142");
    expect(html).toContain("lo-fi");
    for (const label of ["Make another like this", "Rename", "Delete", "Add tag"]) expect(html).toContain(label);
    expect(html).toContain("Newest first");
    expect(html).toContain("Longest first");
    expect(html).toContain('draggable="true"');
  });
});
