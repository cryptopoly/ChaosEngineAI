import { describe, expect, it } from "vitest";
import type { StoryboardAudioClip, StoryboardProject, StoryboardShot, StoryboardTimeline } from "../../../../types";
import {
  DEFAULT_ZOOM_INDEX,
  GRID_S,
  ZOOM_LEVELS,
  anchorPatchFor,
  bucketsFor,
  buildLanes,
  clampNumber,
  clampTrimIn,
  clampTrimOut,
  clampZoom,
  clipSpan,
  dropToAnchor,
  duckRegionsFor,
  fitZoomIndex,
  formatTimecode,
  laneOfKind,
  levelFraction,
  nudgeAmount,
  peaksForBlock,
  pxPerSecond,
  pxToTime,
  rulerTicks,
  seekFromPointer,
  shiftPosition,
  snapEdges,
  snapRange,
  snapTime,
  tickStep,
  timeToPx,
  trimAtPlayhead,
  trimEndPatch,
  trimStartPatch,
  voiceIntervals,
} from "../timelineGeometry";

const spans = [
  { id: "s1", start: 0, end: 4 },
  { id: "s2", start: 3.5, end: 8 }, // a 0.5 s crossfade into shot 2
  { id: "s3", start: 8, end: 12 },
];

function clip(extra: Partial<StoryboardAudioClip> = {}): StoryboardAudioClip {
  return {
    id: "a1", kind: "music",
    source: { path: "media/audio/a1.wav", hash: "h", measuredSeconds: 10, clonedVoice: false, label: "" },
    anchor: null, start: 2, trimIn: 0, trimOut: null, gainDb: 0, fadeIn: 0, fadeOut: 0, duck: false, loop: false, status: "ok",
    ...extra,
  };
}

describe("time and pixels", () => {
  it("converts both ways at every zoom level", () => {
    for (const pps of ZOOM_LEVELS) {
      expect(timeToPx(2.5, pps)).toBeCloseTo(2.5 * pps);
      expect(pxToTime(timeToPx(7.3, pps), pps)).toBeCloseTo(7.3);
    }
    expect(pxToTime(100, 0)).toBe(0);
  });

  it("clamps the zoom index and maps it to pixels per second", () => {
    expect(clampZoom(-3)).toBe(0);
    expect(clampZoom(99)).toBe(ZOOM_LEVELS.length - 1);
    expect(pxPerSecond(DEFAULT_ZOOM_INDEX)).toBe(ZOOM_LEVELS[DEFAULT_ZOOM_INDEX]);
    expect(pxPerSecond(-1)).toBe(ZOOM_LEVELS[0]);
  });

  it("picks the deepest zoom where the whole film fits", () => {
    const fit = fitZoomIndex(20, 900);
    expect(20 * ZOOM_LEVELS[fit]).toBeLessThanOrEqual(900);
    if (fit + 1 < ZOOM_LEVELS.length) expect(20 * ZOOM_LEVELS[fit + 1]).toBeGreaterThan(900);
    expect(fitZoomIndex(0, 900)).toBe(DEFAULT_ZOOM_INDEX);
    expect(fitZoomIndex(20, 0)).toBe(DEFAULT_ZOOM_INDEX);
  });

  it("maps a pointer position on the track to a time inside the film", () => {
    expect(seekFromPointer(220, 120, 40, 30)).toBe(2.5);
    expect(seekFromPointer(50, 120, 40, 30)).toBe(0);
    expect(seekFromPointer(5000, 120, 40, 30)).toBe(30);
  });
});

describe("ruler", () => {
  it("labels every 1 s up close and every 5, 10 or 30 s further out", () => {
    expect(tickStep(128)).toBeLessThanOrEqual(1);
    expect(tickStep(40)).toBe(5);
    expect(tickStep(12)).toBe(10);
    expect(tickStep(8)).toBe(10);
    for (const pps of ZOOM_LEVELS) expect(tickStep(pps) * pps).toBeGreaterThanOrEqual(64);
  });

  it("puts a major tick every step and four small ones between", () => {
    const ticks = rulerTicks(12, 40);
    expect(ticks[0]).toEqual({ timeS: 0, major: true });
    expect(ticks.filter((t) => t.major).map((t) => t.timeS)).toEqual([0, 5, 10]);
    expect(ticks[1]).toEqual({ timeS: 1, major: false });
    expect(ticks[ticks.length - 1]?.timeS).toBeLessThanOrEqual(12);
  });

  it("formats time as mm:ss with tenths when asked", () => {
    expect(formatTimecode(12.45)).toBe("00:12.4");
    expect(formatTimecode(75, false)).toBe("01:15");
    expect(formatTimecode(-3)).toBe("00:00.0");
    expect(formatTimecode(32.4)).toBe("00:32.4"); // 32.4 * 10 is 323.99999999999994 in floats
    expect(formatTimecode(19.7085)).toBe("00:19.7");
  });
});

describe("snapping", () => {
  it("collects the film start, shot edges and the playhead", () => {
    expect(snapEdges(spans, 5)).toEqual([0, 3.5, 4, 5, 8, 12]);
  });

  it("snaps to an edge within 8 px, else to the 0.1 s grid", () => {
    const edges = snapEdges(spans, 5);
    expect(snapTime(4.1, 40, edges)).toBe(4); // 0.1 s = 4 px at 40 px/s
    expect(snapTime(4.5, 40, edges)).toBe(4.5);
    expect(snapTime(6.04, 40, edges)).toBe(6);
    expect(GRID_S).toBe(0.1);
  });

  it("snaps a clip by whichever end is closer to an edge, keeping its length", () => {
    const edges = snapEdges(spans, 99);
    expect(snapRange(1.9, 2.0, 40, edges)).toBe(2); // the end (3.9) is next to 4, so the start moves by 0.1
    expect(snapRange(5.3, 1, 40, edges)).toBe(5.3);
    expect(snapRange(7.95, 1, 40, edges)).toBe(8);
  });
});

describe("drop to anchor", () => {
  it("anchors to the shot under the drop with an offset", () => {
    expect(dropToAnchor(1.5, spans, false)).toEqual({ anchor: { shotId: "s1", offset: 1.5 } });
    expect(dropToAnchor(9, spans, false)).toEqual({ anchor: { shotId: "s3", offset: 1 } });
  });

  it("takes the shot coming in inside a crossfade", () => {
    expect(dropToAnchor(3.75, spans, false)).toEqual({ anchor: { shotId: "s2", offset: 0.25 } });
  });

  it("falls to the nearest shot outside the picture", () => {
    expect(dropToAnchor(14, spans, false)).toEqual({ anchor: { shotId: "s3", offset: 6 } });
    expect(dropToAnchor(-2, spans, false)).toEqual({ anchor: { shotId: "s1", offset: 0 } });
  });

  it("holding Alt gives an absolute start, as does a film with no shots", () => {
    expect(dropToAnchor(9.234, spans, true)).toEqual({ start: 9.23 });
    expect(dropToAnchor(-1, spans, true)).toEqual({ start: 0 });
    expect(dropToAnchor(3, [], false)).toEqual({ start: 3 });
  });

  it("works out the position when the user picks what a clip hangs on", () => {
    expect(anchorPatchFor("s3", 9.5, spans)).toEqual({ anchor: { shotId: "s3", offset: 1.5 } });
    expect(anchorPatchFor(null, 9.5, spans)).toEqual({ start: 9.5 });
    expect(anchorPatchFor("gone", 2, spans)).toEqual({ start: 2 });
  });
});

describe("keyboard nudge", () => {
  it("moves 0.1 s, or 1 s with Shift", () => {
    expect(nudgeAmount(false)).toBe(0.1);
    expect(nudgeAmount(true)).toBe(1);
  });

  it("changes the offset of an anchored clip and the start of an absolute one", () => {
    expect(shiftPosition(clip({ start: null, anchor: { shotId: "s2", offset: 0.5 } }), 0.1)).toEqual({ anchor: { shotId: "s2", offset: 0.6 } });
    expect(shiftPosition(clip({ start: 2 }), -1)).toEqual({ start: 1 });
  });

  it("never goes before the film start or past the limits", () => {
    expect(shiftPosition(clip({ start: 0.05 }), -1)).toEqual({ start: 0 });
    expect(shiftPosition(clip({ start: null, anchor: { shotId: "s1", offset: -59.95 } }), -1)).toEqual({ anchor: { shotId: "s1", offset: -60 } });
  });

  it("does not drift when many small nudges add up", () => {
    let start = 1;
    for (let i = 0; i < 10; i += 1) start = (shiftPosition(clip({ start }), 0.1) as { start: number }).start;
    expect(start).toBe(2);
  });
});

describe("trim clamping", () => {
  it("keeps the in point inside the clip and at least 0.1 s before the out point", () => {
    const c = clip({ trimIn: 1, trimOut: 4 });
    expect(clampTrimIn(c, -5)).toBe(0);
    expect(clampTrimIn(c, 3.95)).toBe(3.9);
    expect(clampTrimIn(c, 2)).toBe(2);
  });

  it("keeps the out point after the in point and inside the source", () => {
    const c = clip({ trimIn: 2, trimOut: 6 });
    expect(clampTrimOut(c, 1)).toBe(2.1);
    expect(clampTrimOut(c, 99)).toBe(10);
    expect(clampTrimOut(c, 7.5)).toBe(7.5);
  });

  it("dragging the start edge moves the in point and the position by the same amount", () => {
    expect(trimStartPatch(clip({ start: 2 }), 1)).toEqual({ trimIn: 1, start: 3 });
    expect(trimStartPatch(clip({ start: null, anchor: { shotId: "s1", offset: 0.5 } }), 0.5)).toEqual({
      trimIn: 0.5,
      anchor: { shotId: "s1", offset: 1 },
    });
  });

  it("only moves by what the clamp allowed", () => {
    // 0.5 s before the start of the source is not possible: nothing moves
    expect(trimStartPatch(clip({ start: 2, trimIn: 0 }), -0.5)).toEqual({ trimIn: 0, start: 2 });
    const tight = clip({ start: 2, trimIn: 0, trimOut: 0.3 });
    expect(trimStartPatch(tight, 5)).toEqual({ trimIn: 0.2, start: 2.2 });
  });

  it("dragging the end edge only changes the out point", () => {
    expect(trimEndPatch(clip({ trimOut: 6 }), -2)).toEqual({ trimOut: 4 });
    expect(trimEndPatch(clip({ trimOut: null }), 5)).toEqual({ trimOut: 10 });
    expect(trimEndPatch(clip({ trimIn: 3, trimOut: 6 }), -9)).toEqual({ trimOut: 3.1 });
  });

  it("an unmeasured source has no upper limit except the model's", () => {
    const open = clip({ source: { path: "media/audio/a1.wav", hash: null, measuredSeconds: null, clonedVoice: false, label: "" } });
    expect(clampTrimOut(open, 500)).toBe(500);
    expect(clampTrimOut(open, 9999)).toBe(3600);
  });

  it("[ and ] cut at the playhead and refuse a playhead outside the clip", () => {
    const span = { start: 2, duration: 4, end: 6, approximate: false };
    const c = clip({ start: 2, trimOut: null });
    expect(trimAtPlayhead(c, span, 3, "in")).toEqual({ trimIn: 1, start: 3 });
    expect(trimAtPlayhead(c, span, 4.5, "out")).toEqual({ trimOut: 2.5 });
    expect(trimAtPlayhead(c, span, 7, "in")).toBeNull();
    expect(trimAtPlayhead(c, span, 2, "out")).toBeNull();
  });

  it("parses a typed number into a range", () => {
    expect(clampNumber("12", 0, 10, 3)).toBe(10);
    expect(clampNumber("-1", 0, 10, 3)).toBe(0);
    expect(clampNumber("2,5", 0, 10, 3)).toBe(2.5);
    expect(clampNumber("abc", 0, 10, 3)).toBe(3);
    expect(clampNumber("", 0, 10, 3)).toBe(3);
  });
});

function timeline(): StoryboardTimeline {
  return {
    total: 12,
    shots: spans.map((s) => ({
      ...s, duration: s.end - s.start, source_in: 0, source_out: s.end - s.start, transition_type: "cut", transition_seconds: 0, measured: true, ready: true,
    })),
    audio: [{ id: "a1", start: 2, duration: 5, trim_in: 0, end: 7 }],
    problems: [],
  };
}

describe("where a clip sits", () => {
  it("uses the computed timeline when it knows the clip", () => {
    expect(clipSpan(clip(), timeline(), spans)).toEqual({ start: 2, duration: 5, end: 7, approximate: false });
  });

  it("works a missing clip out from its own fields", () => {
    const missing = clip({ id: "gone", status: "missing", start: null, anchor: { shotId: "s3", offset: 1 }, trimIn: 1, trimOut: 4 });
    expect(clipSpan(missing, timeline(), spans)).toEqual({ start: 9, duration: 3, end: 12, approximate: true });
    const unmeasured = clip({ id: "gone", start: 1, source: { path: "media/audio/x.wav", hash: null, measuredSeconds: null, clonedVoice: false, label: "" } });
    expect(clipSpan(unmeasured, timeline(), spans).duration).toBe(2);
  });

  it("maps clip kinds to lanes: imports share Effects", () => {
    expect(laneOfKind("voice")).toBe("voice");
    expect(laneOfKind("music")).toBe("music");
    expect(laneOfKind("sfx")).toBe("sfx");
    expect(laneOfKind("import")).toBe("sfx");
  });
});

describe("ducking shade", () => {
  it("merges voice clips and cuts them to the clip's own span", () => {
    const voice = voiceIntervals([
      { start: 1, end: 3, missing: false },
      { start: 2.5, end: 4, missing: false },
      { start: 9, end: 10, missing: true },
      { start: 6, end: 7, missing: false },
    ] as never);
    expect(voice).toEqual([{ from: 1, to: 4 }, { from: 6, to: 7 }]);
    expect(duckRegionsFor(2, 6.5, voice)).toEqual([{ from: 0, to: 2 }, { from: 4, to: 4.5 }]);
    expect(duckRegionsFor(8, 9, voice)).toEqual([]);
  });
});

function shot(id: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id, order: 0, title: id === "s1" ? "Arrival" : "", status: "done", stillPrompt: "", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null, speed: 1,
    takes: [{ id: "t", still: null, clip: `media/${id}/t/clip.mp4`, hashes: { clip: "hh" }, settings: {}, measuredSeconds: 4, createdAt: 0, origin: "upload", basis: null }],
    activeTake: "t", transition: { type: "cut", seconds: 0 }, nativeSoundDb: -6, redo: false, ...extra,
  };
}

describe("buildLanes", () => {
  const project = {
    shots: [shot("s1"), shot("s2"), shot("s3")],
    audio: [
      clip({ id: "v1", kind: "voice", start: 1 }),
      clip({ id: "v2", kind: "voice", start: 2.5 }),
      clip({ id: "m1", kind: "music", duck: true, start: 0 }),
      clip({ id: "x1", kind: "import", start: 5, status: "missing" }),
    ],
  } as unknown as StoryboardProject;
  const tl: StoryboardTimeline = {
    ...timeline(),
    audio: [
      { id: "v1", start: 1, duration: 2, trim_in: 0, end: 3 },
      { id: "v2", start: 2.5, duration: 2, trim_in: 0, end: 4.5 },
      { id: "m1", start: 0, duration: 10, trim_in: 0, end: 10 },
    ],
  };

  it("returns picture, shot sound, voice, effects and music in order", () => {
    const lanes = buildLanes(project, tl);
    expect(lanes.map((l) => l.id)).toEqual(["picture", "native", "voice", "sfx", "music"]);
    expect(lanes[0].clips.map((c) => [c.id, c.label, c.start])).toEqual([["s1", "Arrival", 0], ["s2", "Shot 2", 3.5], ["s3", "Shot 3", 8]]);
    expect(lanes[1].clips[0].sourcePath).toBe("media/s1/t/clip.mp4");
  });

  it("places audio from the timeline and marks the missing one", () => {
    const lanes = buildLanes(project, tl);
    expect(lanes[2].clips.map((c) => c.id)).toEqual(["v1", "v2"]);
    const gone = lanes[3].clips[0];
    expect(gone.missing).toBe(true);
    expect(gone.start).toBe(5);
    expect(gone.laneId).toBe("sfx");
  });

  it("shades the ducked music where voice plays", () => {
    const music = buildLanes(project, tl)[4].clips[0];
    expect(music.duckRegions).toEqual([{ from: 1, to: 4.5 }]);
  });
});

describe("waveform slice and buckets", () => {
  const peaks = Array.from({ length: 100 }, (_, i) => i / 100);

  it("shows the part of the source after the in point", () => {
    const slice = peaksForBlock(peaks, 10, 5, 2, false);
    expect(slice).toHaveLength(20);
    expect(slice[0]).toBe(0.5);
  });

  it("repeats the slice for a looping clip longer than its source", () => {
    const slice = peaksForBlock(peaks, 10, 8, 6, true);
    expect(slice).toHaveLength(60);
    expect(slice[20]).toBe(0.8);
    expect(peaksForBlock(peaks, 10, 8, 6, false)).toHaveLength(20);
  });

  it("is empty with nothing to show", () => {
    expect(peaksForBlock([], 10, 0, 2, false)).toEqual([]);
    expect(peaksForBlock(peaks, 0, 0, 2, false)).toEqual([]);
  });

  it("asks for about 20 buckets a second within the server's bounds", () => {
    expect(bucketsFor(10)).toBe(200);
    expect(bucketsFor(0.5)).toBe(32);
    expect(bucketsFor(1000)).toBe(2000);
    expect(bucketsFor(null)).toBe(400);
  });

  it("maps a gain in dB to a level bar fraction", () => {
    expect(levelFraction(-60)).toBe(0);
    expect(levelFraction(12)).toBe(1);
    expect(levelFraction(0)).toBeCloseTo(60 / 72);
    expect(levelFraction(-99)).toBe(0);
  });
});
