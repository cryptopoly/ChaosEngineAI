import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { MusicInstallJob, MusicLibraryItem, MusicModelFamily, MusicModelVariant, MusicRuntimeStatus } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { UseMusicState } from "../../../hooks/useMusicState";
import { MusicModelsPage } from "../MusicModelsPage";
import { MusicStudioTab } from "../MusicStudioTab";
import { SoundPackNotice } from "../SoundPackNotice";
import { SoundsStudio } from "../SoundsStudio";
import {
  INITIAL_SOUND_RUN,
  SOUND_CAPTION_MAX,
  buildSoundPayload,
  defaultSoundForm,
  soundBlocker,
  soundFormFromItem,
  soundRunReducer,
  validateSoundForm,
  varySoundForm,
  againSoundForm,
  type SoundForm,
} from "../soundsLogic";
import { rememberLicenceAccepted } from "../musicLicence";

const IDLE_JOB: MusicInstallJob = {
  phase: "idle", phaseLabel: "", phaseIndex: 0, phaseTotal: 0, percent: 0, message: "", error: null, log: [], startedAt: 0, finishedAt: 0, running: false,
};

function soundVariant(over: Partial<MusicModelVariant> = {}): MusicModelVariant {
  return {
    id: "stabilityai/stable-audio-open-small",
    kind: "sound",
    pack: "sound",
    familyId: "stable-audio-open",
    name: "Stable Audio Open Small",
    provider: "Stability AI",
    repo: "stabilityai/stable-audio-open-small",
    link: "https://huggingface.co/stabilityai/stable-audio-open-small",
    engine: "stableaudio",
    engineModel: "stable-audio-open-small",
    makes: ["sfx", "loops"],
    platforms: ["apple-silicon"],
    sizeGb: 2.57,
    maxSeconds: 11,
    minSeconds: 1,
    licence: {
      id: "stability-community",
      name: "Stability AI Community Licence",
      commercial: "limited",
      gated: true,
      attribution: "Powered by Stability AI",
      revenueCapUsd: 1_000_000,
      summaryLines: ["Free while you earn under US$1 million a year.", "Show Powered by Stability AI where you share it."],
      url: "https://huggingface.co/stabilityai/stable-audio-open-small/blob/main/LICENSE",
    },
    estimates: { basis: "estimated", secondsPer10sAudio: 5, peakGb: 6 },
    defaultSettings: { seconds: 5, loop: false },
    availableLocally: false,
    missingRequires: [],
    packInstalled: false,
    note: "Opt-in sound effects model.",
    ...over,
  };
}

function musicVariant(): MusicModelVariant {
  return soundVariant({
    id: "ACE-Step/Ace-Step1.5", kind: "music", pack: "music", familyId: "ace-step-1.5", name: "ACE-Step 1.5 turbo", repo: "ACE-Step/Ace-Step1.5",
    engine: "acestep", engineModel: "acestep-v15-turbo", makes: ["songs", "instrumental"], sizeGb: 10.09, maxSeconds: 600, minSeconds: 10,
    licence: { id: "mit", name: "MIT", commercial: "yes", gated: false, summaryLines: ["MIT."] }, availableLocally: true, defaultSettings: { seconds: 60 },
  });
}

function runtime(over: Partial<MusicRuntimeStatus> = {}): MusicRuntimeStatus {
  return {
    lane: "mlx", unavailableReason: null, installed: true, engine: "acestep", workerRunning: false, loadedModel: null,
    idleUnloadMinutes: 10, embedTag: true, libraryDir: "/data/music/library", settings: { idleUnloadMinutes: 10 }, install: IDLE_JOB,
    sound: { lane: "mps", unavailableReason: null, installed: false, engine: "stableaudio", workerRunning: false, loadedModel: null, install: IDLE_JOB },
    ...over,
  };
}

function families(sound: MusicModelVariant): MusicModelFamily[] {
  return [
    { id: "ace-step-1.5", name: "ACE-Step 1.5", provider: "ACE Studio", defaultVariantId: "ACE-Step/Ace-Step1.5", variants: [musicVariant()] },
    { id: "stable-audio-open", kind: "sound", name: "Stable Audio Open", provider: "Stability AI", defaultVariantId: sound.id, variants: [sound] },
  ];
}

const noop = () => undefined;
const asyncNoop = async () => undefined;

function fakeMusic(sound: MusicModelVariant, over: Record<string, unknown> = {}, soundsOver: Record<string, unknown> = {}): UseMusicState {
  const music = musicVariant();
  const form: SoundForm = { ...defaultSoundForm(sound), caption: "door creak" };
  const base = {
    families: families(sound),
    variants: [music],
    runtime: runtime(),
    downloads: {},
    installJob: IDLE_JOB,
    soundInstallJob: IDLE_JOB,
    soundVariants: [sound],
    soundVariant: sound,
    startSoundPackInstall: asyncNoop,
    cancelSoundPackInstall: asyncNoop,
    catalogLoaded: true,
    selectedVariant: music,
    installedVariants: [music],
    hasInstalledModel: true,
    studioMode: "sounds",
    setStudioMode: noop,
    startDownload: asyncNoop,
    pauseDownload: asyncNoop,
    deleteDownload: asyncNoop,
    startInstall: asyncNoop,
    cancelInstall: asyncNoop,
    openStudio: noop,
    library: { items: [], diskBytes: 0, loaded: true, refreshLibrary: asyncNoop, renameItem: asyncNoop, setTags: asyncNoop, deleteItems: async () => [] },
    studio: {
      form: { mode: "song", caption: "", lyrics: "", seconds: 60, customLength: false, bpm: null, key: null, variations: 1, seed: null },
      setForm: noop, replaceForm: noop, estimate: null,
      generation: { phase: "idle", startedAt: null, cancelling: false, error: null, results: [], lastForm: null },
      running: false, progress: null, generate: asyncNoop, cancel: asyncNoop, dismissError: noop, forgetResults: noop,
    },
    sounds: {
      form, setForm: noop, replaceForm: noop, estimate: null, run: INITIAL_SOUND_RUN, running: false, progress: null,
      generate: asyncNoop, cancel: asyncNoop, dismissError: noop, forgetResults: noop, ...soundsOver,
    },
    ...over,
  };
  return base as unknown as UseMusicState;
}

function renderStudio(music: UseMusicState, online = true) {
  return renderToStaticMarkup(<MusicStudioTab music={music} backendOnline={online} onTabChange={noop} />);
}

const readySound = () => soundVariant({ availableLocally: true, packInstalled: true, textEncoderReady: true, weightsSource: "checkpoints" });
const readyRuntime = () =>
  runtime({ sound: { lane: "mps", unavailableReason: null, installed: true, engine: "stableaudio", workerRunning: false, loadedModel: null, install: IDLE_JOB } });

function acceptLicence() {
  const store = new Map<string, string>();
  vi.stubGlobal("window", { localStorage: { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v) } });
  rememberLicenceAccepted("stabilityai/stable-audio-open-small", "stability-community");
}

describe("sounds form", () => {
  it("starts from the model defaults with a random seed", () => {
    expect(defaultSoundForm(soundVariant())).toEqual({ caption: "", seconds: 5, loop: false, variations: 1, seed: null });
    expect(defaultSoundForm(null).seconds).toBe(5);
    expect(defaultSoundForm(soundVariant({ defaultSettings: { seconds: 60 } })).seconds).toBe(5);
  });

  it("validates the 1 to 11 s range, the description and the seed", () => {
    const ok = { ...defaultSoundForm(), caption: "rain" };
    expect(validateSoundForm(ok)).toEqual([]);
    expect(validateSoundForm({ ...ok, caption: "  " })).toEqual(["captionEmpty"]);
    expect(validateSoundForm({ ...ok, caption: "x".repeat(SOUND_CAPTION_MAX + 1) })).toEqual(["captionTooLong"]);
    expect(validateSoundForm({ ...ok, seconds: 12 })).toEqual(["secondsRange"]);
    expect(validateSoundForm({ ...ok, seconds: 0.5 })).toEqual(["secondsRange"]);
    expect(validateSoundForm({ ...ok, seed: -1 })).toEqual(["seedRange"]);
  });

  it("builds the request without a seed when it is random, and with the loop flag", () => {
    const form: SoundForm = { caption: " door ", seconds: 4, loop: true, variations: 2, seed: null };
    expect(buildSoundPayload(form, "m")).toEqual({ modelId: "m", caption: "door", seconds: 4, loop: true, variations: 2 });
    expect(buildSoundPayload({ ...form, seed: 7 }, "m").seed).toBe(7);
  });

  it("remakes a library sound, with Again and Vary seeds", () => {
    const item = { id: "1_a", kind: "sound", title: "t", caption: "rain", lyrics: "", tags: [], createdAt: 0, updatedAt: 0, audio: ".flac", durationS: 3.4, seed: 9, loop: true } as MusicLibraryItem;
    const form = soundFormFromItem(item);
    expect(form).toMatchObject({ caption: "rain", seconds: 3, loop: true, seed: 9 });
    expect(againSoundForm(form).seed).toBeNull();
    expect(varySoundForm(form, 9).seed).toBe(10);
    expect(soundFormFromItem({ ...item, durationS: 40 }).seconds).toBe(11);
  });
});

describe("sound readiness", () => {
  const base = { catalogLoaded: true, variant: soundVariant(), packInstalled: false, unavailableReason: null, licenceAccepted: false };

  it("names the next step in the order a person fixes it", () => {
    expect(soundBlocker({ ...base, catalogLoaded: false })).toBe("loading");
    expect(soundBlocker({ ...base, unavailableReason: "Apple Silicon only" })).toBe("unsupported");
    expect(soundBlocker(base)).toBe("licence");
    expect(soundBlocker({ ...base, licenceAccepted: true })).toBe("download");
    const downloaded = soundVariant({ availableLocally: true });
    expect(soundBlocker({ ...base, variant: downloaded, licenceAccepted: true })).toBe("pack");
    expect(soundBlocker({ ...base, variant: downloaded, licenceAccepted: true, packInstalled: true })).toBeNull();
  });

  it("does not ask for the licence again once the weights are on disk", () => {
    expect(soundBlocker({ ...base, variant: soundVariant({ availableLocally: true }), packInstalled: true })).toBeNull();
  });
});

describe("sound run state", () => {
  const form = defaultSoundForm();
  const result = { item: { id: "1_a" } as MusicLibraryItem, jobSeconds: 3, seed: 1 };

  it("runs, finishes, keeps newest first and forgets", () => {
    let state = soundRunReducer(INITIAL_SOUND_RUN, { type: "start", at: 5, form });
    expect(state.running).toBe(true);
    expect(soundRunReducer(state, { type: "start", at: 6, form })).toBe(state);
    state = soundRunReducer(state, { type: "succeeded", results: [result] });
    expect(state.running).toBe(false);
    expect(state.results).toHaveLength(1);
    expect(soundRunReducer(state, { type: "forget", ids: ["1_a"] }).results).toHaveLength(0);
  });

  it("turns a cancel into the quiet kind and a memory refusal into the memory kind", () => {
    let state = soundRunReducer(INITIAL_SOUND_RUN, { type: "start", at: 1, form });
    state = soundRunReducer(state, { type: "failed", message: "cancelled", status: 409 });
    expect(state.error?.kind).toBe("cancelled");
    state = soundRunReducer(state, { type: "start", at: 2, form });
    state = soundRunReducer(state, { type: "failed", message: "needs about 6 GB of free memory", status: 503 });
    expect(state.error?.kind).toBe("oom");
    expect(soundRunReducer(state, { type: "dismiss" }).error).toBeNull();
  });
});

describe("Music Studio with Sounds", () => {
  it("shows the Music | Sounds switch only when the catalog has a sound row", () => {
    const html = renderStudio(fakeMusic(readySound(), { runtime: readyRuntime() }));
    expect(html).toContain("music-studio__studio-modes");
    expect(html).toContain(">Music<");
    expect(html).toContain(">Sounds<");
    expect(html).toContain("Powered by Stability AI");
    const musicOnly = renderStudio(fakeMusic(readySound(), { soundVariant: null, studioMode: "music" }));
    expect(musicOnly).not.toContain("music-studio__studio-modes");
  });

  it("keeps the music mode as it was when Music is chosen", () => {
    const html = renderStudio(fakeMusic(readySound(), { studioMode: "music" }));
    expect(html).toContain("Make track");
    expect(html).not.toContain("Make sound");
    expect(html).toContain("ACE-Step 1.5 turbo");
  });

  it("a ready Sounds mode shows the prompt, 1 to 11 s pills, Loop and one action", () => {
    acceptLicence();
    const html = renderStudio(fakeMusic(readySound(), { runtime: readyRuntime() }));
    expect(html).toContain("Describe the sound");
    for (const n of [2, 4, 6, 8, 11]) expect(html).toContain(`${n} s`);
    expect(html).toContain("Make it a seamless loop");
    expect(html).toContain("Make sound");
    expect(html).not.toContain("Make track");
    expect(html).not.toContain("Lyrics");
    expect(html).toContain("No sounds yet");
  });

  it("says what is missing, one sentence and one button, in order", () => {
    vi.stubGlobal("window", { localStorage: { getItem: () => null, setItem: noop } });
    let html = renderStudio(fakeMusic(soundVariant()));
    expect(html).toContain("accept its licence first");
    expect(html).toContain("Open Music Models");
    expect(html).not.toContain("Make sound");
    acceptLicence();
    html = renderStudio(fakeMusic(soundVariant()));
    expect(html).toContain("Accept the licence on Hugging Face first");
    html = renderStudio(fakeMusic(soundVariant({ availableLocally: true })));
    expect(html).toContain("small pack");
  });

  it("is quiet offline and when the machine cannot run sounds", () => {
    expect(renderStudio(fakeMusic(readySound()), false)).toContain("Waiting for the backend");
    const html = renderStudio(fakeMusic(soundVariant(), { runtime: runtime({ sound: { ...readyRuntime().sound!, unavailableReason: "Sound effects run on Apple Silicon in this version." } }) }));
    expect(html).toContain("run on Apple Silicon");
    expect(html).not.toContain("Open Music Models");
  });

  it("shows results with Loop, Again, Vary and the storyboard menu hook", () => {
    acceptLicence();
    const item = { id: "1700000000_aaaaaaaa", kind: "sound", title: "door creak", caption: "door", lyrics: "", tags: [], createdAt: 1, updatedAt: 1, audio: ".flac", durationS: 4, seed: 11, loop: true, sizeBytes: 700_000 } as MusicLibraryItem;
    const music = fakeMusic(readySound(), { runtime: readyRuntime() }, { run: { ...INITIAL_SOUND_RUN, results: [{ item, jobSeconds: 3, seed: 11 }] } });
    const html = renderToStaticMarkup(<SoundsStudio music={music} backendOnline onTabChange={noop} header={<h2>Header</h2>} />);
    expect(html).toContain("door creak");
    expect(html).toContain("Loop · 0:04");
    expect(html).toContain("seed 11");
    expect(html).toContain(">Again<");
    expect(html).toContain(">Vary<");
    expect(html).toContain("Open in Library");
  });

  it("shows the running state with Cancel", () => {
    acceptLicence();
    const music = fakeMusic(readySound(), { runtime: readyRuntime() }, { running: true, run: { ...INITIAL_SOUND_RUN, running: true, startedAt: 1 } });
    const html = renderStudio(music);
    expect(html).toContain("Making your sound");
    expect(html).toContain("Cancel");
  });
});

describe("Music Models with the Sounds row", () => {
  const page = (music: UseMusicState) =>
    renderToStaticMarkup(
      <MusicModelsPage music={music} backendOnline availability="all" onAvailabilityChange={noop} onOpenExternalUrl={noop} />,
    );

  it("lists the gated sound row with its licence chip and 'Accept licence' before anything else", () => {
    vi.stubGlobal("window", { localStorage: { getItem: () => null, setItem: noop } });
    const html = page(fakeMusic(soundVariant()));
    expect(html).toContain("Stable Audio Open Small");
    expect(html).toContain("Stability AI Community Licence");
    expect(html).toContain("Gated");
    expect(html).toContain("Needs licence");
    expect(html).toContain("Accept licence");
    expect(html).not.toContain("Install the sounds pack");
  });

  it("after the licence is accepted the row offers the weights and the separate sounds pack", () => {
    acceptLicence();
    const html = page(fakeMusic(soundVariant()));
    expect(html).not.toContain("Accept licence");
    expect(html).toContain("Install the sounds pack");
    expect(html).toContain("About 1 GB");
  });

  it("the sounds pack notice is gone once installed and shows progress while installing", () => {
    const installed = { ...runtime(), sound: readyRuntime().sound };
    expect(renderToStaticMarkup(<SoundPackNotice music={{ runtime: installed, soundInstallJob: IDLE_JOB, startSoundPackInstall: asyncNoop, cancelSoundPackInstall: asyncNoop }} />)).toBe("");
    const running = { ...IDLE_JOB, running: true, percent: 40, phaseLabel: "Installing sound packages", log: ["$ pip install"] };
    const html = renderToStaticMarkup(<SoundPackNotice music={{ runtime: runtime(), soundInstallJob: running, startSoundPackInstall: asyncNoop, cancelSoundPackInstall: asyncNoop }} />);
    expect(html).toContain("Installing sound packages");
    expect(html).toContain("40%");
    expect(html).toContain("Cancel");
  });

  it("the music first-run line ignores an installed sound row", () => {
    const notInstalledMusic = { ...musicVariant(), availableLocally: false };
    const all = families(soundVariant({ availableLocally: true }));
    all[0] = { ...all[0], variants: [notInstalledMusic] };
    const music = fakeMusic(soundVariant(), { installedVariants: [], hasInstalledModel: false, selectedVariant: notInstalledMusic, families: all });
    expect(page(music)).toContain("Pick a music model to start");
  });
});
