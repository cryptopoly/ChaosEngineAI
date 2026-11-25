import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type {
  StoryboardLipSyncCoverage,
  StoryboardLipSyncRow,
  StoryboardProject,
  StoryboardShot,
  StoryboardSong,
} from "../../../../types";
import { ShotVocalFields } from "../../ShotVocalFields";
import type { UseStoryboardResult } from "../../useStoryboard";
import { CutPlanReview } from "../CutPlanReview";
import { LipSyncCoverage } from "../LipSyncCoverage";
import { SingQuestion } from "../SingQuestion";
import { SongPanel } from "../SongPanel";
import { SongVocals } from "../SongVocals";

const noop = () => undefined;

function row(extra: Partial<StoryboardLipSyncRow>): StoryboardLipSyncRow {
  return {
    shotId: "s1", number: 1, songStart: 0, songEnd: 3, seconds: 3, state: "synced", code: null, message: "",
    fix: null, minSeconds: 2, maxSeconds: 7.6, ...extra,
  };
}

function coverage(rows: StoryboardLipSyncRow[], source: "stem" | "song" | null = "stem"): StoryboardLipSyncCoverage {
  const counts = { synced: 0, waiting: 0, skipped: 0, unassigned: 0 };
  for (const r of rows) counts[r.state] += 1;
  return { source, at: source ? 1 : null, counts: { ...counts, total: rows.length, sings: counts.synced + counts.waiting }, shots: rows };
}

function shot(id: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id, order: 0, title: "", status: "idle", stillPrompt: "", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}

function song(extra: Partial<StoryboardSong> = {}): StoryboardSong {
  return {
    path: "song.wav", hash: null, title: "Harbour", seconds: 30, bpm: 96, beats: [],
    sections: [{ start: 0, end: 30, label: "A", energy: 0.5 }] as StoryboardSong["sections"],
    lyrics: [], clipId: "a1", libraryId: null, rough: false, note: "", analysedAt: 0, ...extra,
  };
}

function project(extra: Partial<StoryboardProject> = {}): StoryboardProject {
  return {
    schemaVersion: 3, revision: 4, id: "0123456789abcdef", name: "Night Drive", createdAt: 0,
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
    startBatch: vi.fn(async () => true), cancelBatch: async () => undefined, call: vi.fn(async () => true),
  } as unknown as UseStoryboardResult;
}

const table = (cov: StoryboardLipSyncCoverage, extra: { batchRunning?: boolean; canOpenShot?: boolean } = {}) =>
  renderToStaticMarkup(
    <LipSyncCoverage coverage={cov} shots={[shot("s1", { title: "Mirror" })]} busy={false} batchRunning={extra.batchRunning ?? false} canOpenShot={extra.canOpenShot} onFix={noop} />,
  );

describe("Lip-sync coverage table (audit change 3)", () => {
  it("shows a synced shot with its counts, song time and the vocal's source, and no button", () => {
    const html = table(coverage([row({ songStart: 0, songEnd: 3.2, seconds: 3.2 })]));
    expect(html).toContain("Lip sync: 1 of 1 shot sings");
    expect(html).toContain("Vocal from the stem");
    expect(html).toContain("1 synced · 0 waiting for a clip · 0 not lip-synced · 0 without a vocal");
    expect(html).toContain("0:00 – 0:03");
    expect(html).toContain("3.2 s");
    expect(html).toContain("Synced");
    expect(html).toContain("Mirror");
    expect(html).not.toContain("<button");
  });

  it("offers Make its clip on a waiting shot and one button for all waiting shots, held while a batch runs", () => {
    const cov = coverage([row({ state: "waiting", fix: "make-lipsync" }), row({ shotId: "s2", number: 2, state: "waiting", fix: "make-lipsync" })]);
    const html = table(cov);
    expect(html).toContain("Waiting for its clip");
    expect(html).toContain("Make its clip");
    expect(html).toContain("Make 2 lip-sync clips");
    const held = table(cov, { batchRunning: true });
    expect(held).toContain("Waits for the shots being made now.");
    expect(held).toMatch(/<button[^>]*disabled=""[^>]*>Make 2 lip-sync clips<\/button>/);
  });

  it("names why a skipped shot is not lip-synced and offers its one fix", () => {
    const html = table(
      coverage([
        row({ state: "skipped", code: "too-short", fix: "lengthen", seconds: 1.6, songEnd: 1.6 }),
        row({ shotId: "s2", number: 2, state: "skipped", code: "too-long", fix: "shorten", seconds: 9 }),
        row({ shotId: "s3", number: 3, state: "skipped", code: "locked", fix: "unlock" }),
        row({ shotId: "s4", number: 4, state: "skipped", code: "other-speed", fix: "real-speed" }),
        row({ shotId: "s5", number: 5, state: "skipped", code: "no-singing", fix: null }),
      ]),
    );
    expect(html).toContain("Too short: 1.6 s, a lip-sync shot needs at least 2 s.");
    expect(html).toContain("Lengthen to 2 s or more");
    expect(html).toContain("Too long: 9 s, a lip-sync shot here lasts at most 7.6 s.");
    expect(html).toContain("Shorten to 7.6 s or less");
    expect(html).toContain("The shot is locked.");
    expect(html).toContain(">Unlock</button>");
    expect(html).toContain("Play at real speed");
    expect(html).toContain("No singing here; it plays as an ordinary clip.");
    expect((html.match(/<button/g) ?? []).length).toBe(4);
  });

  it("hides Lengthen and Shorten when no shot editor can open", () => {
    const html = table(coverage([row({ state: "skipped", code: "too-short", fix: "lengthen", seconds: 1.6 })]), { canOpenShot: false });
    expect(html).toContain("Too short");
    expect(html).not.toContain("Lengthen");
  });

  it("shows a shot without a vocal as an ordinary clip, with no row button", () => {
    const html = table(coverage([row({ state: "unassigned", code: "not-assigned", fix: "assign-vocals" })], null));
    expect(html).toContain("Lip sync: 0 of 1 shot sings");
    expect(html).toContain("No vocal");
    expect(html).toContain("No vocal; it plays as an ordinary clip.");
    expect(html).not.toContain("Vocal from");
    expect(html).not.toContain("<button");
  });
});

describe("Lip-sync section", () => {
  const vocals = (cov: StoryboardLipSyncCoverage | null) =>
    renderToStaticMarkup(
      <SongVocals coverage={cov} shots={[]} result={null} problem={null} disabled={false} working={false} batchRunning={false} onAssign={noop} onFix={noop} />,
    );

  it("states the real length rule and offers the vocal run", () => {
    const html = vocals(null);
    expect(html).toContain("Lip-sync the song");
    expect(html).toContain("0.4 s before the shot");
    expect(html).toContain("A shot of 2 s or more can lip-sync. Its clip lasts 3 to 8 s: a shot shorter than 3 s gets silent room to fill it.");
    expect(html).not.toContain("3 to 5 s");
    expect(html).toContain("Give each shot its vocal");
  });

  it("warns that running again gives every shot its vocal back", () => {
    expect(vocals(coverage([row({})]))).toContain("including shots you took the vocal off");
    expect(vocals(coverage([row({ state: "unassigned", code: "not-assigned", fix: "assign-vocals" })], null))).toContain("1 shot has no vocal yet.");
  });
});

describe("Does someone sing on screen? (audit change 8)", () => {
  const ask = (answer: "yes" | "no" | null, hasShots = false) =>
    renderToStaticMarkup(
      <SingQuestion answer={answer} onAnswer={noop} stem={null} onStem={noop} disabled={false} working={null} hasShots={hasShots} onCut={noop} onVocalsOnly={noop} />,
    );

  it("asks first and holds the one cut button until answered", () => {
    const html = ask(null);
    expect(html).toContain("Does someone sing on screen?");
    expect(html).toContain("Yes, lip-sync the singing");
    expect(html).toContain("No, pictures only");
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Cut the song into shots<\/button>/);
    expect(html).toContain("Choose an answer first.");
    expect(html).not.toContain("Vocal stem");
  });

  it("Yes shows the stem field, the length rule and that not every shot has to sing", () => {
    const html = ask("yes", true);
    expect(html).toContain("Cuts the song into shots of 3 to 5 s, then gives each shot its part of the vocal.");
    expect(html).toContain("Vocal stem (optional, WAV or FLAC)");
    expect(html).toContain("A shot of 2 s or more can lip-sync.");
    expect(html).toContain("Not every shot has to sing");
    expect(html).toContain("Keep my shots and give each its vocal");
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Cut the song into shots/);
  });

  it("No cuts to the beat with no lip sync and no vocal buttons", () => {
    const html = ask("no", true);
    expect(html).toContain("Cuts to the beat, with no lip sync.");
    expect(html).not.toContain("Vocal stem");
    expect(html).not.toContain("give each its vocal");
  });

  it("the cut review says a singing cut also gives each shot its vocal", () => {
    const plan = { shots: [{ title: "a", stillPrompt: "", motionPrompt: "", seconds: 4, trimOut: 4 }], sections: [], totalSeconds: 4, clipSeconds: 5, clipSecondsSource: "default", videoModel: null, note: "" };
    const p = project();
    const review = (singing: boolean) =>
      renderToStaticMarkup(<CutPlanReview plan={plan as never} project={p} busy={false} singing={singing} onUse={noop} onDiscard={noop} />);
    expect(review(true)).toContain("Use this shot and give it its vocal");
    expect(review(false)).toContain("Use this shot");
    expect(review(false)).not.toContain("vocal");
  });
});

describe("Song panel: one flow instead of two cut buttons", () => {
  it("has no separate Cut for singing button, and asks the question once the song is in", () => {
    const p = project({ song: song() });
    const html = renderToStaticMarkup(<SongPanel story={story(p)} project={p} onOpenShot={noop} />);
    expect(html).toContain("Does someone sing on screen?");
    expect(html).not.toContain("Cut for singing");
    // One button in the question and, as the song is longer than the empty film, one beside "Time the lyrics".
    expect((html.match(/>Cut the song into shots</g) ?? []).length).toBe(2);
    expect(html).not.toContain("Lip-sync the song"); // not asked yet, never used
  });

  it("starts on Yes and shows the lip-sync section when the project already used lip sync", () => {
    const p = project({ song: song(), shots: [shot("s1", { vocalClip: "audio/s1.wav" } as Partial<StoryboardShot>)] });
    const html = renderToStaticMarkup(<SongPanel story={story(p)} project={p} onOpenShot={noop} />);
    expect(html).toMatch(/<input[^>]*checked=""[^>]*value="yes"/);
    expect(html).toContain("Lip-sync the song");
  });
});

describe("Shot editor vocal hint", () => {
  it("says the real range, not 3 to 8 s for the shot, and that a shot need not sing", () => {
    const p = project();
    const html = renderToStaticMarkup(<ShotVocalFields project={p} shot={shot("s1")} disabled={false} op={vi.fn(async () => true) as never} />);
    expect(html).toContain("A shot of 2 s or more can lip-sync.");
    expect(html).not.toContain("lasts 3 to 8 s and keeps");
    expect(html).toContain("A shot does not have to sing");
  });
});
