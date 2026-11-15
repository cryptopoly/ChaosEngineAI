import { describe, expect, it } from "vitest";
import type { StoryboardAudioClip, StoryboardProject, StoryboardShot, StoryboardTimeline } from "../../../../types";
import {
  DUCK_GAIN,
  buildRoughCut,
  captionAt,
  cutMarkers,
  dbToGain,
  duckWindows,
  envelopeAt,
  envelopePoints,
  mediaFiles,
  pieceIndexAt,
  previewKeyAction,
  shotJump,
  soundStart,
  sourceTimeAt,
  videoCorrection,
  visibleAt,
  type CutSound,
} from "../roughCutLogic";

function shot(id: string, extra: Partial<StoryboardShot> = {}, take: Partial<StoryboardShot["takes"][number]> | null = {}): StoryboardShot {
  return {
    id, order: 0, title: `Title ${id}`, status: "done", stillPrompt: "", motionPrompt: "", referenceImages: [], startFrom: "generate",
    endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null, speed: 1,
    takes: take === null ? [] : [{ id: "t", still: `media/${id}/still.png`, clip: `media/${id}/clip.mp4`, hashes: {}, settings: {}, measuredSeconds: 4, createdAt: 0, origin: "batch", basis: null, ...take }],
    activeTake: take === null ? null : "t", transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false, ...extra,
  };
}

function audio(id: string, extra: Partial<StoryboardAudioClip> = {}): StoryboardAudioClip {
  return {
    id, kind: "music", source: { path: `media/audio/${id}.wav`, hash: null, measuredSeconds: 10, clonedVoice: false, label: "" },
    anchor: null, start: 0, trimIn: 0, trimOut: null, gainDb: 0, fadeIn: 0, fadeOut: 0, duck: false, loop: false, status: "ok", ...extra,
  };
}

function project(shots: StoryboardShot[], clips: StoryboardAudioClip[] = [], captions: StoryboardProject["captions"] = []): StoryboardProject {
  return {
    schemaVersion: 1, revision: 3, id: "p", name: "Film", createdAt: 0,
    defaults: {} as StoryboardProject["defaults"], references: { characters: [], places: [] } as unknown as StoryboardProject["references"],
    shots, audio: clips, export: {} as StoryboardProject["export"], captions, batches: [], undoStack: [], redoStack: [],
  };
}

// Three shots: a 4 s clip, a 3 s clip coming in on a 1 s crossfade (sped up 2x from source 1 s), and a shot with only a still.
const s1 = shot("s1", { nativeSoundDb: -6 });
const s2 = shot("s2", { speed: 2, transition: { type: "crossfade", seconds: 1 } }, { renderPass: "draft" });
const s3 = shot("s3", {}, { clip: null });
const timeline: StoryboardTimeline = {
  total: 9,
  shots: [
    { id: "s1", start: 0, end: 4, duration: 4, source_in: 0, source_out: 4, transition_type: "cut", transition_seconds: 0, measured: true, ready: true },
    { id: "s2", start: 3, end: 6, duration: 3, source_in: 1, source_out: 7, transition_type: "crossfade", transition_seconds: 1, measured: true, ready: true },
    { id: "s3", start: 6, end: 9, duration: 3, source_in: 0, source_out: 3, transition_type: "cut", transition_seconds: 0, measured: false, ready: false },
  ],
  audio: [
    { id: "v1", start: 3.5, duration: 2, trim_in: 0, end: 5.5 },
    { id: "m1", start: 0, duration: 9, trim_in: 1, end: 9 },
  ],
  problems: [],
};
const clips = [
  audio("v1", { kind: "voice", gainDb: 6, anchor: { shotId: "s2", offset: 0.5 }, start: null }),
  audio("m1", { gainDb: -12, fadeIn: 2, fadeOut: 3, duck: true, loop: true, trimIn: 1, source: { path: "media/audio/m1.wav", hash: null, measuredSeconds: 4, clonedVoice: false, label: "" } }),
];
const cut = buildRoughCut(project([s1, s2, s3], clips, [{ id: "c1", clipId: "v1", start: 0.2, end: 1.5, text: "Tea time" }]), timeline);

describe("buildRoughCut", () => {
  it("keeps every shot in timeline order with what it will show", () => {
    expect(cut.pieces.map((p) => [p.shotId, p.media.kind, p.draft, p.fadeIn])).toEqual([
      ["s1", "clip", false, 0],
      ["s2", "clip", true, 1],
      ["s3", "still", false, 0],
    ]);
    expect([cut.missing, cut.drafts, cut.total]).toEqual([1, 1, 9]);
  });

  it("a shot with no take at all is a card", () => {
    const bare = buildRoughCut(project([shot("x", {}, null)]), { ...timeline, total: 4, shots: [{ ...timeline.shots[0], id: "x", ready: false }], audio: [] });
    expect(bare.pieces[0].media).toEqual({ kind: "none" });
    expect(bare.pieces[0].draft).toBe(false);
  });

  it("gives each clip its own sound at the shot's gain, crossfaded with its neighbours", () => {
    const natives = cut.sounds.filter((s) => s.kind === "native");
    expect(natives.map((s) => [s.id, s.start, s.duration, s.offset, s.rate, s.fadeIn, s.fadeOut])).toEqual([
      ["native:s1", 0, 4, 0, 1, 0, 1],
      ["native:s2", 3, 3, 1, 2, 1, 0],
    ]);
    expect(natives[0].gain).toBeCloseTo(0.501, 3);
  });

  it("places audio clips where the timeline does, ducks only non-voice sounds", () => {
    const voice = cut.sounds.find((s) => s.id === "v1") as CutSound;
    const music = cut.sounds.find((s) => s.id === "m1") as CutSound;
    expect([voice.start, voice.duration, voice.duck, voice.gain]).toEqual([3.5, 2, false, dbToGain(6)]);
    expect([music.offset, music.loop, music.duck, music.fadeIn, music.fadeOut]).toEqual([1, true, true, 2, 3]);
  });

  it("drops ducking when there is no voice, like the export", () => {
    const noVoice = buildRoughCut(project([s1], [clips[1]]), { ...timeline, audio: [timeline.audio[1]] });
    expect(noVoice.sounds.find((s) => s.id === "m1")?.duck).toBe(false);
  });

  it("puts captions on film time through their clip", () => {
    expect(cut.captions).toEqual([{ start: 3.7, end: 5, text: "Tea time" }]);
    expect(captionAt(cut, 4)).toBe("Tea time");
    expect(captionAt(cut, 5.2)).toBeNull();
  });

  it("lists the files to load once each", () => {
    expect(mediaFiles(cut)).toEqual({
      videos: ["media/s1/clip.mp4", "media/s2/clip.mp4"],
      stills: ["media/s3/still.png"],
      audio: ["media/audio/v1.wav", "media/audio/m1.wav"],
    });
  });
});

describe("picture timing", () => {
  it("shows one shot on a cut and two during a crossfade", () => {
    expect(visibleAt(cut, 1).map((l) => [l.piece.shotId, l.opacity])).toEqual([["s1", 1]]);
    expect(visibleAt(cut, 3.25).map((l) => [l.piece.shotId, l.opacity])).toEqual([["s1", 1], ["s2", 0.25]]);
    expect(visibleAt(cut, 4).map((l) => l.piece.shotId)).toEqual(["s2"]);
    expect(visibleAt(cut, 9).map((l) => l.piece.shotId)).toEqual(["s3"]); // the last frame holds at the end
  });

  it("reads the right source second, with speed", () => {
    const [p1, p2] = cut.pieces;
    expect(sourceTimeAt(p1, 2.5)).toBe(2.5);
    expect(sourceTimeAt(p2, 4)).toBe(3); // in point 1 s + 1 s of film at 2x
    expect(sourceTimeAt(p2, 99)).toBe(7);
  });

  it("names the newest shot that has started", () => {
    expect(pieceIndexAt(cut, 0)).toBe(0);
    expect(pieceIndexAt(cut, 3.1)).toBe(1);
    expect(pieceIndexAt(cut, 8)).toBe(2);
  });
});

describe("sound timing", () => {
  const voice = cut.sounds.find((s) => s.id === "v1") as CutSound;
  const music = cut.sounds.find((s) => s.id === "m1") as CutSound;

  it("starts a later sound after a delay, a running one part way in, and skips a finished one", () => {
    expect(soundStart(voice, 0)).toEqual({ delay: 3.5, offset: 0, length: 2 });
    expect(soundStart(voice, 4)).toEqual({ delay: 0, offset: 0.5, length: 1.5 });
    expect(soundStart(voice, 5.5)).toBeNull();
  });

  it("plays a sped-up shot's sound at its rate", () => {
    const native = cut.sounds.find((s) => s.id === "native:s2") as CutSound;
    expect(soundStart(native, 4)).toEqual({ delay: 0, offset: 3, length: 2 });
  });

  it("wraps a looping sound inside its loop region", () => {
    // Loop region is 1..4 s of the file (3 s long). At film 7 s it has played 7 s: 1 + (7 mod 3) = 2.
    expect(soundStart(music, 7, 4)).toEqual({ delay: 0, offset: 2, length: 2 });
  });

  it("follows fades in and out", () => {
    expect(envelopeAt(music, 1)).toBeCloseTo(0.5);
    expect(envelopeAt(music, 4)).toBe(1);
    expect(envelopeAt(music, 7.5)).toBeCloseTo(0.5);
    expect(envelopePoints(music, 1)).toEqual([
      { t: 1, value: 0.5 },
      { t: 2, value: 1 },
      { t: 6, value: 1 },
      { t: 9, value: 0 },
    ]);
    expect(envelopePoints(voice, 0)).toEqual([{ t: 3.5, value: 1 }, { t: 5.5, value: 1 }]); // no fade: full level until it stops
  });

  it("ducks under merged voice lines", () => {
    const two = { ...cut, sounds: [...cut.sounds, { ...voice, id: "v2", start: 5, duration: 2 }, { ...voice, id: "v3", start: 8, duration: 0.5 }] };
    expect(duckWindows(two)).toEqual([[3.5, 7], [8, 8.5]]);
    expect(DUCK_GAIN).toBeLessThan(0.5);
  });
});

describe("controls", () => {
  it("maps the keys", () => {
    expect(previewKeyAction(" ", false)).toEqual({ type: "toggle-play" });
    expect(previewKeyAction("k", false)).toEqual({ type: "toggle-play" });
    expect(previewKeyAction("j", false)).toEqual({ type: "seek", deltaS: -5 });
    expect(previewKeyAction("ArrowRight", true)).toEqual({ type: "seek", deltaS: 10 });
    expect(previewKeyAction("End", false)).toEqual({ type: "seek-to", where: "end" });
    expect(previewKeyAction("PageDown", false)).toEqual({ type: "shot", direction: 1 });
    expect(previewKeyAction("c", false)).toEqual({ type: "captions" });
    expect(previewKeyAction("x", false)).toBeNull();
  });

  it("steps between shots", () => {
    expect(shotJump(cut, 1, 1)).toBe(3);
    expect(shotJump(cut, 8, 1)).toBe(9);
    expect(shotJump(cut, 5, -1)).toBe(3); // well into shot 2: back to its start
    expect(shotJump(cut, 3.2, -1)).toBe(0); // just started: the shot before
    expect(shotJump(cut, 0.2, -1)).toBe(0);
  });

  it("marks shots on the scrubber", () => {
    expect(cutMarkers(cut).map((m) => [m.shotId, m.state, Math.round(m.left), Math.round(m.width)])).toEqual([
      ["s1", "clip", 0, 44],
      ["s2", "draft", 33, 33],
      ["s3", "still", 67, 33],
    ]);
  });

  it("keeps video on the clock by seeking only when it is clearly off", () => {
    expect(videoCorrection(1.0, 1.05, 1)).toEqual({ rate: 1, seek: false });
    expect(videoCorrection(1.0, 1.2, 2)).toEqual({ rate: 2, seek: true });
  });
});
