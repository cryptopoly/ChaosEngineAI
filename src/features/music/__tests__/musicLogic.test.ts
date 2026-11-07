import { describe, expect, it } from "vitest";
import type { MusicEstimateResponse, MusicLibraryItem, MusicModelFamily, MusicModelVariant, MusicResult } from "../../../types";
import { findNamedArtist } from "../artistNames";
import { peaksFromChannel } from "../musicAudio";
import {
  CAPTION_MAX,
  INITIAL_GENERATION,
  LYRICS_MAX,
  againForm,
  allTags,
  audioPerSecond,
  buildPayload,
  classifyMusicError,
  clampSeconds,
  clock,
  defaultForm,
  diskLabel,
  filterLibrary,
  formFromItem,
  generationReducer,
  isQueued,
  itemFilePath,
  libraryCounts,
  megabytes,
  pickDefaultVariant,
  queueForStrip,
  shorterSeconds,
  stripFromEstimate,
  validateForm,
  varyForm,
  withTag,
  type MusicForm,
} from "../musicLogic";

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
    platforms: ["apple-silicon"],
    sizeGb: 10.09,
    maxSeconds: 600,
    minSeconds: 10,
    licence: { id: "mit", name: "MIT", commercial: "yes", gated: false },
    estimates: { basis: "measured", secondsPer10sAudio: 1 },
    defaultSettings: { seconds: 60 },
    availableLocally: true,
    missingRequires: [],
    ...over,
  };
}

function item(over: Partial<MusicLibraryItem> = {}): MusicLibraryItem {
  return {
    id: "1700000000_aaaaaaaa",
    kind: "track",
    title: "Lo-fi piano",
    caption: "warm lo-fi hip hop",
    lyrics: "",
    tags: [],
    createdAt: 100,
    updatedAt: 100,
    audio: ".flac",
    durationS: 30,
    seed: 8142,
    instrumental: true,
    ...over,
  };
}

function form(over: Partial<MusicForm> = {}): MusicForm {
  return { ...defaultForm(variant()), caption: "lo-fi piano", lyrics: "[verse] la la", ...over };
}

describe("form limits", () => {
  it("accepts a complete song and an instrumental without lyrics", () => {
    expect(validateForm(form())).toEqual([]);
    expect(validateForm(form({ mode: "instrumental", lyrics: "" }))).toEqual([]);
  });

  it("asks for a description, and for lyrics unless Instrumental", () => {
    expect(validateForm(form({ caption: "  " }))).toEqual(["captionEmpty"]);
    expect(validateForm(form({ lyrics: "" }))).toEqual(["lyricsEmpty"]);
  });

  it("enforces the caption (512) and lyrics (4096) limits", () => {
    expect(validateForm(form({ caption: "a".repeat(CAPTION_MAX) }))).toEqual([]);
    expect(validateForm(form({ caption: "a".repeat(CAPTION_MAX + 1) }))).toEqual(["captionTooLong"]);
    expect(validateForm(form({ lyrics: "a".repeat(LYRICS_MAX) }))).toEqual([]);
    expect(validateForm(form({ lyrics: "a".repeat(LYRICS_MAX + 1) }))).toEqual(["lyricsTooLong"]);
    // Lyrics do not count in Instrumental mode.
    expect(validateForm(form({ mode: "instrumental", lyrics: "a".repeat(LYRICS_MAX + 1) }))).toEqual([]);
  });

  it("keeps the length in 10 to 600 s", () => {
    expect(validateForm(form({ seconds: 10 }))).toEqual([]);
    expect(validateForm(form({ seconds: 600 }))).toEqual([]);
    expect(validateForm(form({ seconds: 9 }))).toEqual(["secondsRange"]);
    expect(validateForm(form({ seconds: 601 }))).toEqual(["secondsRange"]);
    expect(validateForm(form({ seconds: Number.NaN }))).toEqual(["secondsRange"]);
    expect(clampSeconds(5)).toBe(10);
    expect(clampSeconds(9000)).toBe(600);
    expect(clampSeconds(61.4)).toBe(61);
    expect(clampSeconds(5, { minSeconds: 15, maxSeconds: 120 })).toBe(15);
  });

  it("allows 1, 2 and 4 variations only", () => {
    expect(validateForm(form({ variations: 4 }))).toEqual([]);
    expect(validateForm(form({ variations: 3 as never }))).toEqual(["variations"]);
  });

  it("checks BPM (30 to 300) and the seed range", () => {
    expect(validateForm(form({ bpm: 120 }))).toEqual([]);
    expect(validateForm(form({ bpm: 29 }))).toEqual(["bpmRange"]);
    expect(validateForm(form({ bpm: 301 }))).toEqual(["bpmRange"]);
    expect(validateForm(form({ seed: -1 }))).toEqual(["seedRange"]);
    expect(validateForm(form({ seed: 0 }))).toEqual([]);
  });

  it("starts from the model's defaults (60 s is the 1 min pill)", () => {
    const start = defaultForm(variant());
    expect(start).toMatchObject({ mode: "song", seconds: 60, customLength: false, variations: 1, seed: null, bpm: null, key: null });
    expect(defaultForm(variant({ defaultSettings: { seconds: 45 } })).customLength).toBe(true);
    expect(defaultForm(null).seconds).toBe(60);
  });
});

describe("request payload", () => {
  it("sends lyrics for a song and leaves optional fields out", () => {
    expect(buildPayload(form(), "m")).toEqual({
      modelId: "m",
      caption: "lo-fi piano",
      instrumental: false,
      seconds: 60,
      variations: 1,
      lyrics: "[verse] la la",
    });
  });

  it("sends no lyrics for an instrumental and carries bpm, key and seed when set", () => {
    const payload = buildPayload(form({ mode: "instrumental", bpm: 85, key: "A minor", seed: 7, variations: 2 }), "m");
    expect(payload).toMatchObject({ instrumental: true, bpm: 85, key: "A minor", seed: 7, variations: 2 });
    expect("lyrics" in payload).toBe(false);
  });

  it("Again drops the seed, Vary adds one, Make another keeps the item's seed", () => {
    const base = form({ seed: 8142 });
    expect(againForm(base).seed).toBeNull();
    expect(varyForm(base, 8142).seed).toBe(8143);
    expect(varyForm(base, 2 ** 31 - 1).seed).toBe(2 ** 31 - 1);
    const like = formFromItem(item({ bpm: 90, key: "C major", lyrics: "words", instrumental: false }));
    expect(like).toMatchObject({ mode: "song", caption: "warm lo-fi hip hop", seconds: 30, seed: 8142, bpm: 90, key: "C major" });
    expect(formFromItem(item({ durationS: 45 })).customLength).toBe(true);
  });
});

function estimateResponse(over: Partial<MusicEstimateResponse["estimate"]> = {}, queue?: MusicEstimateResponse["queue"], total = 64): MusicEstimateResponse {
  return {
    estimate: {
      model: "m",
      seconds: 35,
      coldStartSeconds: 25,
      warm: false,
      peakGb: 22.9,
      idleGb: 15.2,
      writesMb: 6,
      modelDiskGb: 10,
      basis: "measured",
      ...over,
    },
    fit: null,
    totalGb: total,
    queue: queue ?? { running: null, waiting: [] },
  };
}

describe("estimate strip mapping", () => {
  it("maps time, peak of total, disk and basis", () => {
    expect(stripFromEstimate(estimateResponse())).toMatchObject({
      seconds: 35,
      peakGb: 22.9,
      totalGb: 64,
      diskMb: 6,
      basis: "measured",
      risk: "safe",
      queue: null,
    });
  });

  it("colours the memory by share of total (70 and 92 per cent)", () => {
    expect(stripFromEstimate(estimateResponse({ peakGb: 47 }, undefined, 64)).risk).toBe("caution");
    expect(stripFromEstimate(estimateResponse({ peakGb: 47 }, undefined, 48)).risk).toBe("danger");
    expect(stripFromEstimate(estimateResponse({ peakGb: 10 }, undefined, 64)).risk).toBe("safe");
  });

  it("keeps unknown values null so the strip says 'unknown until the first run'", () => {
    const strip = stripFromEstimate(estimateResponse({ seconds: 0, peakGb: null, writesMb: null }));
    expect(strip).toMatchObject({ seconds: null, peakGb: null, diskMb: null, basis: null });
    expect(stripFromEstimate(null)).toMatchObject({ seconds: null, peakGb: null, totalGb: null, basis: null, queue: null });
  });

  it("labels a model-card estimate as Estimated", () => {
    expect(stripFromEstimate(estimateResponse({ basis: "estimated" })).basis).toBe("estimated");
  });

  it("shows the queue line when another job runs or waits, or when this request waits", () => {
    const ticket = { id: "1", kind: "image", source: "ui", label: "x" };
    expect(queueForStrip({ running: ticket, waiting: [] }, null)).toEqual({ what: "", secondsLeft: null, position: 1 });
    expect(queueForStrip({ running: ticket, waiting: [ticket] }, null)?.position).toBe(2);
    expect(queueForStrip({ running: null, waiting: [] }, null)).toBeNull();
    expect(queueForStrip(null, { waiting: true, position: 3, agentAhead: false })?.position).toBe(3);
    expect(stripFromEstimate(estimateResponse(), { waiting: true, position: 2, agentAhead: false }).queue?.position).toBe(2);
  });
});

describe("progress text", () => {
  it("is queued only while the request waits behind another job", () => {
    const base = { kind: "music", active: false, phase: "idle", message: "", step: 0, totalSteps: 0, startedAt: 0, updatedAt: 0, elapsedSeconds: 0, runLabel: null } as const;
    expect(isQueued({ ...base, queue: { waiting: true, position: 1, agentAhead: false } } as never)).toBe(true);
    expect(isQueued({ ...base, queue: { waiting: false, position: 0, agentAhead: false } } as never)).toBe(false);
    expect(isQueued(null)).toBe(false);
  });

  it("formats clocks and sizes", () => {
    expect(clock(30)).toBe("0:30");
    expect(clock(125)).toBe("2:05");
    expect(clock(null)).toBe("0:00");
    expect(megabytes(11_400_000)).toBe("11 MB");
    expect(megabytes(6_240_000)).toBe("6.2 MB");
    expect(megabytes(0)).toBeNull();
    expect(diskLabel(86_000_000)).toBe("86 MB");
    expect(diskLabel(2_300_000_000)).toBe("2.3 GB");
  });
});

describe("errors in plain words", () => {
  it("sorts failures by what the user can do", () => {
    expect(classifyMusicError("cancelled", 409).kind).toBe("cancelled");
    expect(classifyMusicError("Not enough free memory: 23 GB needed, 9 GB free. Unload the image model.", 503).kind).toBe("oom");
    expect(classifyMusicError("The music pack is not installed. Install it from Music > Models.", 409).kind).toBe("notReady");
    expect(classifyMusicError("Add lyrics, or switch Instrumental on.", 422).kind).toBe("invalid");
    expect(classifyMusicError("Failed to fetch", null).kind).toBe("offline");
    expect(classifyMusicError("Request to /api/music/generate timed out or was stopped", 0).kind).toBe("offline");
    expect(classifyMusicError("boom", 500).kind).toBe("generic");
  });

  it("keeps the backend sentence and puts the raw text under technical details", () => {
    const info = classifyMusicError("Music generation failed: worker exited", 503);
    expect(info.detail).toBe("Music generation failed: worker exited");
    expect(info.technical).toBe("HTTP 503 · Music generation failed: worker exited");
    expect(classifyMusicError("cancelled", 409).detail).toBe("");
  });

  it("offers the next shorter length", () => {
    expect(shorterSeconds(600)).toBe(120);
    expect(shorterSeconds(120)).toBe(60);
    expect(shorterSeconds(61)).toBe(60);
    expect(shorterSeconds(15)).toBe(10);
    expect(shorterSeconds(10)).toBe(10);
  });
});

function result(id: string): MusicResult {
  return { item: item({ id }), jobSeconds: 5, seed: 1 };
}

describe("generation lifecycle", () => {
  const start = generationReducer(INITIAL_GENERATION, { type: "start", at: 10, form: form() });

  it("goes idle to running, and ignores a second start while running", () => {
    expect(start).toMatchObject({ phase: "running", startedAt: 10, cancelling: false, error: null });
    expect(generationReducer(start, { type: "start", at: 99, form: form() })).toBe(start);
  });

  it("ends in done with the newest result first, capped for the session", () => {
    const first = generationReducer(start, { type: "succeeded", results: [result("a")] });
    expect(first).toMatchObject({ phase: "done", startedAt: null });
    const second = generationReducer(generationReducer(first, { type: "start", at: 20, form: form() }), {
      type: "succeeded",
      results: [result("b"), result("c")],
    });
    expect(second.results.map((entry) => entry.item.id)).toEqual(["b", "c", "a"]);
    let many = INITIAL_GENERATION;
    for (let i = 0; i < 20; i += 1) {
      many = generationReducer(generationReducer(many, { type: "start", at: i, form: form() }), { type: "succeeded", results: [result(`r${i}`)] });
    }
    expect(many.results).toHaveLength(12);
    expect(many.results[0].item.id).toBe("r19");
  });

  it("keeps the error until dismissed and never drops results on failure", () => {
    const done = generationReducer(start, { type: "succeeded", results: [result("a")] });
    const failed = generationReducer(generationReducer(done, { type: "start", at: 30, form: form() }), {
      type: "failed",
      error: classifyMusicError("boom", 500),
    });
    expect(failed).toMatchObject({ phase: "error" });
    expect(failed.error?.kind).toBe("generic");
    expect(failed.results).toHaveLength(1);
    expect(generationReducer(failed, { type: "dismiss" })).toMatchObject({ phase: "idle", error: null });
  });

  it("marks a cancel request only while running, and does not dismiss a running state", () => {
    expect(generationReducer(start, { type: "cancelRequested" }).cancelling).toBe(true);
    expect(generationReducer(INITIAL_GENERATION, { type: "cancelRequested" })).toBe(INITIAL_GENERATION);
    expect(generationReducer(start, { type: "dismiss" })).toBe(start);
  });

  it("ignores results or errors that arrive when nothing is running", () => {
    expect(generationReducer(INITIAL_GENERATION, { type: "succeeded", results: [result("a")] })).toBe(INITIAL_GENERATION);
    expect(generationReducer(INITIAL_GENERATION, { type: "failed", error: classifyMusicError("x", 500) })).toBe(INITIAL_GENERATION);
  });

  it("forgets deleted tracks from the session list", () => {
    const done = generationReducer(start, { type: "succeeded", results: [result("a"), result("b")] });
    expect(generationReducer(done, { type: "forget", ids: ["a"] }).results.map((entry) => entry.item.id)).toEqual(["b"]);
  });
});

describe("library filter and sort", () => {
  const items = [
    item({ id: "1700000001_00000001", title: "Rain", caption: "soft rain on a roof", tags: ["ambient"], createdAt: 10, durationS: 120 }),
    item({ id: "1700000002_00000002", title: "Lo-fi piano", caption: "warm lo-fi hip hop", tags: ["lo-fi", "piano"], createdAt: 20, durationS: 30 }),
    item({ id: "1700000003_00000003", title: "Drive", caption: "synthwave night drive", tags: ["Lo-Fi"], createdAt: 30, durationS: 60 }),
  ];
  const base = { query: "", tag: "", sort: "newest" as const };

  it("sorts newest, oldest and longest", () => {
    expect(filterLibrary(items, base).map((i) => i.title)).toEqual(["Drive", "Lo-fi piano", "Rain"]);
    expect(filterLibrary(items, { ...base, sort: "oldest" }).map((i) => i.title)).toEqual(["Rain", "Lo-fi piano", "Drive"]);
    expect(filterLibrary(items, { ...base, sort: "longest" }).map((i) => i.title)).toEqual(["Rain", "Drive", "Lo-fi piano"]);
  });

  it("filters by text over title, prompt and tags, and by tag without regard to case", () => {
    expect(filterLibrary(items, { ...base, query: "rain" }).map((i) => i.title)).toEqual(["Rain"]);
    expect(filterLibrary(items, { ...base, query: "synthwave" }).map((i) => i.title)).toEqual(["Drive"]);
    expect(filterLibrary(items, { ...base, query: "piano" }).map((i) => i.title)).toEqual(["Lo-fi piano"]);
    expect(filterLibrary(items, { ...base, tag: "lo-fi" }).map((i) => i.title)).toEqual(["Drive", "Lo-fi piano"]);
    expect(filterLibrary(items, { ...base, tag: "lo-fi", query: "drive" }).map((i) => i.title)).toEqual(["Drive"]);
    expect(filterLibrary(items, { ...base, query: "nothing like this" })).toEqual([]);
  });

  it("lists tags once, adds a tag the way the backend cleans it, and counts", () => {
    expect(allTags(items)).toEqual(["ambient", "lo-fi", "piano"]);
    expect(withTag(["lo-fi"], "  Lo-Fi  ")).toEqual(["lo-fi"]);
    expect(withTag(["lo-fi"], "  new   tag ")).toEqual(["lo-fi", "new tag"]);
    expect(withTag(["a"], "   ")).toEqual(["a"]);
    expect(withTag([], "x".repeat(60))[0]).toHaveLength(40);
    expect(libraryCounts(items)).toEqual({ tracks: 3, sounds: 0 });
    expect(libraryCounts([...items, item({ kind: "sound" })])).toEqual({ tracks: 3, sounds: 1 });
  });

  it("builds the audio file path for Send to storyboard", () => {
    expect(itemFilePath("/home/x/music/library", item())).toBe("/home/x/music/library/1700000000_aaaaaaaa.flac");
    expect(itemFilePath("C:\\Users\\x\\library\\", item({ audio: ".wav" }))).toBe("C:\\Users\\x\\library\\1700000000_aaaaaaaa.wav");
    expect(itemFilePath(null, item())).toBeUndefined();
  });
});

describe("model choice and speed", () => {
  const family = (variants: MusicModelVariant[], best?: string): MusicModelFamily => ({
    id: "f",
    name: "F",
    provider: "p",
    defaultVariantId: variants[0].id,
    bestFitVariantId: best ?? null,
    variants,
  });

  it("prefers the best-fit installed model, else the family default", () => {
    const turbo = variant({ id: "turbo", availableLocally: true });
    const xl = variant({ id: "xl", availableLocally: true });
    expect(pickDefaultVariant([family([turbo, xl], "xl")])?.id).toBe("xl");
    expect(pickDefaultVariant([family([turbo, variant({ id: "xl", availableLocally: false })], "xl")])?.id).toBe("turbo");
    expect(pickDefaultVariant([family([variant({ id: "a", availableLocally: false }), variant({ id: "b", availableLocally: false })])])?.id).toBe("a");
    expect(pickDefaultVariant([])).toBeNull();
  });

  it("does not count a model whose required weights are missing", () => {
    const xl = variant({ id: "xl", availableLocally: true, missingRequires: ["turbo"] });
    expect(pickDefaultVariant([family([xl])])?.id).toBe("xl");
  });

  it("states a measured speed as audio seconds per second, and nothing when not measured", () => {
    expect(audioPerSecond(variant({ estimates: { basis: "measured", secondsPer10sAudio: 1 } }))).toBe(10);
    expect(audioPerSecond(variant({ estimates: { basis: "measured", secondsPer10sAudio: 2 } }))).toBe(5);
    expect(audioPerSecond(variant({ estimates: { basis: "estimated", secondsPer10sAudio: 1 } }))).toBeNull();
    expect(audioPerSecond(variant({ estimates: { basis: "measured" } }))).toBeNull();
  });
});

describe("artist-name heuristic", () => {
  it("finds a well-known name as a whole word, in any case", () => {
    expect(findNamedArtist("a song like Taylor Swift")).toBe("taylor swift");
    expect(findNamedArtist("DAFT PUNK style house")).toBe("daft punk");
    expect(findNamedArtist("mellow, hans zimmer strings")).toBe("hans zimmer");
  });

  it("does not trigger on everyday words that are also artist names", () => {
    expect(findNamedArtist("queen of the night, prince of thieves")).toBeNull();
    expect(findNamedArtist("an oasis in the desert")).toBeNull();
    expect(findNamedArtist("warm lo-fi hip hop, vinyl crackle")).toBeNull();
    expect(findNamedArtist("")).toBeNull();
  });

  it("does not match inside a longer word", () => {
    expect(findNamedArtist("my abbatoir scene")).toBeNull();
    expect(findNamedArtist("the beatles-like harmonies")).toBe("the beatles");
  });
});

describe("waveform peaks", () => {
  it("reduces samples to buckets and scales the loudest to 1", () => {
    const samples = new Float32Array([0, 0.1, -0.2, 0.05, 0.4, -0.1, 0, 0]);
    const peaks = peaksFromChannel(samples, 4);
    expect(peaks).toHaveLength(4);
    expect(Math.max(...peaks)).toBe(1);
    expect(peaks[2]).toBe(1);
  });

  it("gives no bars for silence-free empty input and zeros for silence", () => {
    expect(peaksFromChannel(new Float32Array(0), 10)).toEqual([]);
    expect(peaksFromChannel(new Float32Array(8), 4)).toEqual([0, 0, 0, 0]);
    expect(peaksFromChannel(new Float32Array([0.5, 0.5]), 10)).toHaveLength(2);
  });
});
