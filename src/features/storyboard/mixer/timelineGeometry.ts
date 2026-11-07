/**
 * Pure geometry and editing rules for the Mixer timeline: time <-> pixels at each zoom level, ruler
 * ticks, snapping, where an audio clip sits, what a drop / nudge / trim turns into (the `edit_audio`
 * fields), and how the lanes are built from the project. No React here so the rules are unit-tested
 * directly.
 */
import type {
  StoryboardAudioClip,
  StoryboardProject,
  StoryboardShot,
  StoryboardShotSpan,
  StoryboardTimeline,
} from "../../../types";

// ---- zoom ----
/** Pixels per second at each zoom level. */
export const ZOOM_LEVELS = [8, 16, 32, 64, 128, 256] as const;
export const DEFAULT_ZOOM_INDEX = 2;

export function clampZoom(index: number): number {
  return Math.max(0, Math.min(ZOOM_LEVELS.length - 1, Math.round(index)));
}

export function pxPerSecond(zoomIndex: number): number {
  return ZOOM_LEVELS[clampZoom(zoomIndex)];
}

/** The deepest zoom level at which the whole film still fits in `widthPx`. */
export function fitZoomIndex(totalS: number, widthPx: number): number {
  if (!(totalS > 0) || !(widthPx > 0)) return DEFAULT_ZOOM_INDEX;
  let best = 0;
  ZOOM_LEVELS.forEach((pps, index) => {
    if (totalS * pps <= widthPx) best = index;
  });
  return best;
}

export function timeToPx(timeS: number, pps: number): number {
  return timeS * pps;
}

export function pxToTime(px: number, pps: number): number {
  return pps > 0 ? px / pps : 0;
}

const round = (value: number, places: number): number => {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
};

// ---- ruler ----
const TICK_STEPS = [0.1, 0.5, 1, 5, 10, 30, 60, 300] as const;
/** Smallest distance between two labelled ticks. */
export const MIN_TICK_GAP_PX = 64;

/** The labelled tick interval for a zoom level: the first step that leaves at least MIN_TICK_GAP_PX. */
export function tickStep(pps: number): number {
  return TICK_STEPS.find((step) => step * pps >= MIN_TICK_GAP_PX) ?? TICK_STEPS[TICK_STEPS.length - 1];
}

export interface RulerTick {
  timeS: number;
  /** Labelled tick; the others are the small marks between them. */
  major: boolean;
}

/** Ticks from 0 to `durationS`: a major one every `tickStep`, four small ones between. */
export function rulerTicks(durationS: number, pps: number): RulerTick[] {
  const step = tickStep(pps);
  const minor = step / 5;
  const ticks: RulerTick[] = [];
  const count = Math.floor(Math.max(0, durationS) / minor + 1e-6);
  for (let i = 0; i <= count && i < 4000; i += 1) {
    ticks.push({ timeS: round(i * minor, 3), major: i % 5 === 0 });
  }
  return ticks;
}

/** `mm:ss` for whole-second ticks, `mm:ss.s` otherwise. */
export function formatTimecode(timeS: number, tenths = true): string {
  const total = Math.max(0, timeS);
  const whole = Math.floor(total);
  const minutes = Math.floor(whole / 60);
  const seconds = String(whole % 60).padStart(2, "0");
  const base = `${String(minutes).padStart(2, "0")}:${seconds}`;
  return tenths ? `${base}.${Math.floor((total - whole) * 10)}` : base;
}

export function tickLabel(timeS: number, step: number): string {
  return formatTimecode(timeS, step < 1);
}

// ---- snapping ----
export const GRID_S = 0.1;
export const SNAP_PX = 8;

/** Edges a dragged clip may snap to: the film start, every shot start and end, and the playhead. */
export function snapEdges(spans: Pick<StoryboardShotSpan, "start" | "end">[], playheadS: number): number[] {
  const edges = new Set<number>([0, round(playheadS, 3)]);
  for (const span of spans) {
    edges.add(round(span.start, 3));
    edges.add(round(span.end, 3));
  }
  return [...edges].sort((a, b) => a - b);
}

/** A time pulled to the nearest edge within SNAP_PX, else rounded to the 0.1 s grid. */
export function snapTime(timeS: number, pps: number, edges: number[]): number {
  const limit = SNAP_PX / Math.max(1, pps);
  let best: number | null = null;
  let bestDistance = limit;
  for (const edge of edges) {
    const distance = Math.abs(edge - timeS);
    if (distance <= bestDistance) {
      best = edge;
      bestDistance = distance;
    }
  }
  return best ?? round(Math.round(timeS / GRID_S) * GRID_S, 3);
}

/** Snap a clip by its start or its end, whichever is closer to an edge; keeps the length. */
export function snapRange(startS: number, durationS: number, pps: number, edges: number[]): number {
  const limit = SNAP_PX / Math.max(1, pps);
  let shift: number | null = null;
  for (const point of [startS, startS + durationS]) {
    for (const edge of edges) {
      const delta = edge - point;
      if (Math.abs(delta) <= limit && (shift === null || Math.abs(delta) < Math.abs(shift))) shift = delta;
    }
  }
  return shift === null ? round(Math.round(startS / GRID_S) * GRID_S, 3) : round(startS + shift, 3);
}

// ---- where a clip sits ----
export interface ClipSpan {
  start: number;
  duration: number;
  end: number;
  /** True for a clip the timeline left out (missing file): placed from its own fields. */
  approximate: boolean;
}

const FALLBACK_CLIP_S = 2;

/** Start and length of an audio clip: from the computed timeline, else worked out from the clip itself. */
export function clipSpan(
  clip: StoryboardAudioClip,
  timeline: StoryboardTimeline | null,
  spans: Pick<StoryboardShotSpan, "id" | "start">[],
): ClipSpan {
  const known = timeline?.audio.find((span) => span.id === clip.id);
  if (known) return { start: known.start, duration: known.duration, end: known.start + known.duration, approximate: false };
  const anchor = clip.anchor ? spans.find((span) => span.id === clip.anchor?.shotId) : null;
  const start = Math.max(0, clip.anchor ? (anchor?.start ?? 0) + clip.anchor.offset : (clip.start ?? 0));
  const length = clip.source.measuredSeconds ?? FALLBACK_CLIP_S;
  const out = clip.trimOut != null ? Math.min(clip.trimOut, length) : length;
  const duration = Math.max(0.1, out - clip.trimIn);
  return { start, duration, end: start + duration, approximate: true };
}

// ---- drop, nudge, trim ----
export type PositionPatch = { anchor: { shotId: string; offset: number } } | { start: number };

/**
 * A drop position as an `edit_audio` position: anchored to the shot under it (the later one inside a
 * crossfade) with an offset, or to the nearest shot when it falls outside the picture. With `alt`, or
 * with no shots at all, it is an absolute start.
 */
export function dropToAnchor(
  startS: number,
  spans: Pick<StoryboardShotSpan, "id" | "start" | "end">[],
  alt: boolean,
): PositionPatch {
  const at = Math.max(0, startS);
  if (alt || spans.length === 0) return { start: round(at, 2) };
  let under: (typeof spans)[number] | null = null;
  for (const span of spans) if (at >= span.start && at < span.end) under = span;
  if (!under) {
    let distance = Infinity;
    for (const span of spans) {
      const d = at < span.start ? span.start - at : at - span.end;
      if (d < distance) {
        distance = d;
        under = span;
      }
    }
  }
  const shot = under ?? spans[0];
  return { anchor: { shotId: shot.id, offset: round(at - shot.start, 2) } };
}

export const NUDGE_S = 0.1;
export const NUDGE_BIG_S = 1;

export function nudgeAmount(shift: boolean): number {
  return shift ? NUDGE_BIG_S : NUDGE_S;
}

const MAX_OFFSET = 3600;
const MIN_OFFSET = -60;

/** Move a clip by `deltaS`, keeping what it is anchored to (an anchored clip changes its offset). */
export function shiftPosition(clip: StoryboardAudioClip, deltaS: number): PositionPatch {
  if (clip.anchor) {
    const offset = Math.max(MIN_OFFSET, Math.min(MAX_OFFSET, round(clip.anchor.offset + deltaS, 3)));
    return { anchor: { shotId: clip.anchor.shotId, offset } };
  }
  return { start: Math.max(0, Math.min(MAX_OFFSET, round((clip.start ?? 0) + deltaS, 3))) };
}

export const MIN_CLIP_S = 0.1;

/** The longest a clip can be trimmed out to: its source length, else the model's limit. */
function sourceLength(clip: StoryboardAudioClip): number {
  return clip.source.measuredSeconds ?? MAX_OFFSET;
}

/** Where a clip's out point is, in source seconds. */
export function outPoint(clip: StoryboardAudioClip): number {
  const length = sourceLength(clip);
  return clip.trimOut != null ? Math.min(clip.trimOut, length) : length;
}

export function clampTrimIn(clip: StoryboardAudioClip, value: number): number {
  return round(Math.max(0, Math.min(outPoint(clip) - MIN_CLIP_S, value)), 3);
}

export function clampTrimOut(clip: StoryboardAudioClip, value: number): number {
  return round(Math.max(clip.trimIn + MIN_CLIP_S, Math.min(sourceLength(clip), value)), 3);
}

/**
 * Dragging the left edge by `deltaS` (negative = earlier): the in point moves with it and the clip's
 * position moves the same amount, so the sound under the edge stays where it was in the film.
 */
export function trimStartPatch(clip: StoryboardAudioClip, deltaS: number): { trimIn: number } & PositionPatch {
  const trimIn = clampTrimIn(clip, clip.trimIn + deltaS);
  const applied = trimIn - clip.trimIn;
  return { trimIn, ...shiftPosition(clip, applied) };
}

/** Dragging the right edge by `deltaS`: only the out point changes. */
export function trimEndPatch(clip: StoryboardAudioClip, deltaS: number): { trimOut: number } {
  return { trimOut: clampTrimOut(clip, outPoint(clip) + deltaS) };
}

/** `[` and `]`: cut the clip at the playhead. Null when the playhead is not inside the clip. */
export function trimAtPlayhead(
  clip: StoryboardAudioClip,
  span: ClipSpan,
  playheadS: number,
  side: "in" | "out",
): ({ trimIn: number } & PositionPatch) | { trimOut: number } | null {
  if (playheadS <= span.start || playheadS >= span.end) return null;
  const into = playheadS - span.start;
  return side === "in" ? trimStartPatch(clip, into) : { trimOut: clampTrimOut(clip, clip.trimIn + into) };
}

// ---- lanes ----
export type LaneId = "picture" | "native" | "voice" | "sfx" | "music";
export const AUDIO_LANES = ["voice", "sfx", "music"] as const;
export type AudioLaneId = (typeof AUDIO_LANES)[number];

/** Which lane a clip kind sits on; imported sound shares the Effects lane. */
export function laneOfKind(kind: StoryboardAudioClip["kind"]): AudioLaneId {
  if (kind === "voice") return "voice";
  if (kind === "music") return "music";
  return "sfx";
}

/** The clip kind `add_audio` gets for a lane. */
export function kindOfLane(lane: AudioLaneId): StoryboardAudioClip["kind"] {
  return lane;
}

export interface ClipModel {
  id: string;
  laneId: LaneId;
  label: string;
  start: number;
  duration: number;
  end: number;
  missing: boolean;
  /** "audio" for a lane clip, "shot" for a picture or shot-sound block. */
  type: "audio" | "shot";
  clip?: StoryboardAudioClip;
  shot?: StoryboardShot;
  span?: StoryboardShotSpan;
  /** Project-relative file whose sound the waveform shows, and a key that changes when the file does. */
  sourcePath: string | null;
  sourceKey: string | null;
  /** Source seconds the block shows: from `trimIn`, `shown` long, repeated when looping. */
  trimIn: number;
  sourceLength: number | null;
  /** Intervals (seconds from the clip's start) where the clip is ducked under voice. */
  duckRegions: { from: number; to: number }[];
}

export interface LaneModel {
  id: LaneId;
  clips: ClipModel[];
}

/** Spans where voice plays, merged, for the ducking shading. */
export function voiceIntervals(models: ClipModel[]): { from: number; to: number }[] {
  const sorted = models.filter((m) => !m.missing).map((m) => ({ from: m.start, to: m.end })).sort((a, b) => a.from - b.from);
  const merged: { from: number; to: number }[] = [];
  for (const interval of sorted) {
    const last = merged[merged.length - 1];
    if (last && interval.from <= last.to) last.to = Math.max(last.to, interval.to);
    else merged.push({ ...interval });
  }
  return merged;
}

/** The part of each voice interval that falls inside a clip, relative to the clip's start. */
export function duckRegionsFor(
  clipStart: number,
  clipEnd: number,
  voice: { from: number; to: number }[],
): { from: number; to: number }[] {
  const regions: { from: number; to: number }[] = [];
  for (const interval of voice) {
    const from = Math.max(clipStart, interval.from);
    const to = Math.min(clipEnd, interval.to);
    if (to > from) regions.push({ from: round(from - clipStart, 3), to: round(to - clipStart, 3) });
  }
  return regions;
}

export function shotLabel(shot: StoryboardShot, index: number, untitled: string): string {
  return shot.title.trim() || untitled.replace("{n}", String(index + 1));
}

export function audioLabel(clip: StoryboardAudioClip): string {
  const label = clip.source.label.trim();
  if (label) return label;
  const name = clip.source.path.split("/").pop() ?? clip.id;
  return name.replace(/\.[^.]+$/, "");
}

/** The five lanes of the Mixer from the project and its computed timeline. */
export function buildLanes(project: StoryboardProject, timeline: StoryboardTimeline | null, untitled = "Shot {n}"): LaneModel[] {
  const spans = timeline?.shots ?? [];
  const picture: ClipModel[] = [];
  const native: ClipModel[] = [];
  project.shots.forEach((shot, index) => {
    const span = spans.find((s) => s.id === shot.id);
    if (!span) return;
    const take = shot.takes.find((t) => t.id === shot.activeTake) ?? null;
    const base = {
      label: shotLabel(shot, index, untitled),
      start: span.start,
      duration: span.duration,
      end: span.end,
      missing: shot.status === "missing",
      type: "shot" as const,
      shot,
      span,
      duckRegions: [],
    };
    picture.push({ ...base, id: shot.id, laneId: "picture", sourcePath: null, sourceKey: null, trimIn: span.source_in, sourceLength: null });
    native.push({
      ...base,
      id: shot.id,
      laneId: "native",
      sourcePath: take?.clip ?? null,
      sourceKey: take?.hashes.clip ?? take?.clip ?? null,
      trimIn: span.source_in,
      sourceLength: take?.measuredSeconds ?? null,
    });
  });
  const audio: Record<AudioLaneId, ClipModel[]> = { voice: [], sfx: [], music: [] };
  for (const clip of project.audio) {
    const span = clipSpan(clip, timeline, spans);
    const lane = laneOfKind(clip.kind);
    audio[lane].push({
      id: clip.id,
      laneId: lane,
      label: audioLabel(clip),
      start: span.start,
      duration: span.duration,
      end: span.end,
      missing: clip.status === "missing",
      type: "audio",
      clip,
      sourcePath: clip.source.path,
      sourceKey: clip.source.hash ?? clip.source.path,
      trimIn: clip.trimIn,
      sourceLength: clip.source.measuredSeconds,
      duckRegions: [],
    });
  }
  const voice = voiceIntervals(audio.voice);
  for (const lane of ["sfx", "music"] as const) {
    for (const model of audio[lane]) {
      if (model.clip?.duck) model.duckRegions = duckRegionsFor(model.start, model.end, voice);
    }
  }
  return [
    { id: "picture", clips: picture },
    { id: "native", clips: native },
    { id: "voice", clips: audio.voice },
    { id: "sfx", clips: audio.sfx },
    { id: "music", clips: audio.music },
  ];
}

// ---- waveform slice ----
/**
 * The peaks a block shows. `peaks` covers the whole source (`sourceLength` s); the block shows
 * `shownS` seconds from `trimIn`, repeated when it is longer than what is left of the source (loop).
 */
export function peaksForBlock(
  peaks: number[],
  sourceLength: number,
  trimIn: number,
  shownS: number,
  loop: boolean,
): number[] {
  if (peaks.length === 0 || !(sourceLength > 0) || !(shownS > 0)) return [];
  const perSecond = peaks.length / sourceLength;
  const from = Math.max(0, Math.min(peaks.length - 1, Math.floor(trimIn * perSecond)));
  const available = Math.max(1, peaks.length - from);
  const wanted = Math.max(1, Math.round(shownS * perSecond));
  const slice = peaks.slice(from, from + Math.min(wanted, available));
  if (!loop || wanted <= available) return slice;
  const out: number[] = [];
  while (out.length < wanted) out.push(...slice);
  return out.slice(0, wanted);
}

/** Peak buckets to ask for: about 20 a second, within the server's bounds. */
export function bucketsFor(lengthS: number | null): number {
  const seconds = lengthS && lengthS > 0 ? lengthS : 20;
  return Math.max(32, Math.min(2000, Math.round(seconds * 20)));
}

// ---- level bar ----
const LEVEL_FLOOR_DB = -60;
const LEVEL_CEIL_DB = 12;

/** 0 to 1 for a gain in dB, for the level bar. */
export function levelFraction(db: number): number {
  return Math.max(0, Math.min(1, (db - LEVEL_FLOOR_DB) / (LEVEL_CEIL_DB - LEVEL_FLOOR_DB)));
}

/** Where a click at `clientX` lands on the track, as a time in seconds. */
export function seekFromPointer(clientX: number, trackLeft: number, pps: number, durationS: number): number {
  return Math.max(0, Math.min(durationS, round(pxToTime(clientX - trackLeft, pps), 3)));
}

/**
 * The position fields when the user picks what a clip hangs on: a shot (the clip stays where it is,
 * the offset is worked out from the shot's start) or nothing (an absolute start at its current time).
 */
export function anchorPatchFor(
  shotId: string | null,
  clipStartS: number,
  spans: Pick<StoryboardShotSpan, "id" | "start">[],
): PositionPatch {
  const span = shotId ? spans.find((s) => s.id === shotId) : null;
  if (!span) return { start: round(Math.max(0, clipStartS), 2) };
  return { anchor: { shotId: span.id, offset: round(Math.max(MIN_OFFSET, Math.min(MAX_OFFSET, clipStartS - span.start)), 2) } };
}

/** Clamp a typed number into a range; the old value when the text is not a number. */
export function clampNumber(text: string, min: number, max: number, fallback: number): number {
  const value = Number(text.trim().replace(",", "."));
  if (text.trim() === "" || !Number.isFinite(value)) return fallback;
  return round(Math.max(min, Math.min(max, value)), 3);
}
