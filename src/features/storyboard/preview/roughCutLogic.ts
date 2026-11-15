/**
 * The rough cut (S9): the film as the project defines it today, played in the app from the media files
 * that already exist, with no render. Pure functions only: what is on screen at a time, which source
 * second each clip shows, when each sound starts and how loud it is, and the keys. The player hook
 * (`useRoughCut`) schedules these with Web Audio and moves the video elements to match.
 *
 * Timing follows the project's timeline (the same one the Mixer draws): every shot keeps its place,
 * a shot with no clip yet holds its still (or a card) for its planned length, a crossfade overlaps
 * the shot before it, and sound anchored to a shot starts with that shot. The export leaves shots
 * with no clip out, so a film with such shots is shorter there; the player says so.
 */
import type { StoryboardProject, StoryboardTimeline } from "../../../types";
import { captionRows } from "../mixer/mixerLogic";

export type PieceMedia =
  | { kind: "clip"; clip: string; still: string | null }
  | { kind: "still"; still: string }
  | { kind: "none" };

/** One shot on the rough cut. Times are film seconds. */
export interface CutPiece {
  shotId: string;
  index: number;
  title: string;
  start: number;
  end: number;
  duration: number;
  /** Source second the clip starts at (its in point). */
  sourceIn: number;
  speed: number;
  /** Length of the crossfade into this shot; 0 for a cut. */
  fadeIn: number;
  media: PieceMedia;
  /** The active take is a draft (S6): it plays, with a label. */
  draft: boolean;
}

export type SoundKind = "music" | "sfx" | "voice" | "import" | "native";

/** One sound to schedule: an audio clip, or a shot's own sound. */
export interface CutSound {
  id: string;
  kind: SoundKind;
  /** Project-relative file. */
  path: string;
  start: number;
  duration: number;
  /** Source second heard at `start`. */
  offset: number;
  /** Playback rate (a shot's speed; 1 for audio clips). */
  rate: number;
  /** Linear gain. */
  gain: number;
  fadeIn: number;
  fadeOut: number;
  /** Dipped while a voice line plays. */
  duck: boolean;
  /** Repeats from `offset` to the end of the file until `duration` is used up. */
  loop: boolean;
}

export interface CutCaption {
  start: number;
  end: number;
  text: string;
}

export interface RoughCut {
  total: number;
  pieces: CutPiece[];
  sounds: CutSound[];
  captions: CutCaption[];
  /** Shots with no clip yet (they hold a still or a card). */
  missing: number;
  /** Shots whose active take is a draft. */
  drafts: number;
}

/** How far a ducked sound drops under a voice line (about -12 dB, close to the export's compressor). */
export const DUCK_GAIN = 0.25;

export function dbToGain(db: number): number {
  if (!Number.isFinite(db)) return 0;
  return Math.pow(10, db / 20);
}

export function buildRoughCut(project: StoryboardProject, timeline: StoryboardTimeline): RoughCut {
  const shots = new Map(project.shots.map((shot, index) => [shot.id, { shot, index }]));
  const pieces: CutPiece[] = [];
  const sounds: CutSound[] = [];
  for (const span of timeline.shots) {
    const found = shots.get(span.id);
    if (!found) continue;
    const { shot, index } = found;
    const take = shot.takes.find((t) => t.id === shot.activeTake) ?? null;
    let media: PieceMedia = { kind: "none" };
    if (take?.clip && span.ready) media = { kind: "clip", clip: take.clip, still: take.still };
    else if (take?.still) media = { kind: "still", still: take.still };
    pieces.push({
      shotId: shot.id,
      index,
      title: shot.title,
      start: span.start,
      end: span.end,
      duration: span.duration,
      sourceIn: span.source_in,
      speed: shot.speed || 1,
      fadeIn: span.transition_type === "crossfade" ? span.transition_seconds : 0,
      media,
      draft: media.kind !== "none" && take?.renderPass === "draft",
    });
  }
  pieces.forEach((piece, i) => {
    if (piece.media.kind !== "clip") return;
    const shot = shots.get(piece.shotId)?.shot;
    sounds.push({
      id: `native:${piece.shotId}`,
      kind: "native",
      path: piece.media.clip,
      start: piece.start,
      duration: piece.duration,
      offset: piece.sourceIn,
      rate: piece.speed,
      gain: dbToGain(shot?.nativeSoundDb ?? 0),
      fadeIn: piece.fadeIn,
      fadeOut: pieces[i + 1]?.fadeIn ?? 0,
      duck: false,
      loop: false,
    });
  });
  const clips = new Map(project.audio.map((clip) => [clip.id, clip]));
  for (const span of timeline.audio) {
    const clip = clips.get(span.id);
    if (!clip || clip.status === "missing" || span.duration <= 0) continue;
    sounds.push({
      id: clip.id,
      kind: clip.kind,
      path: clip.source.path,
      start: span.start,
      duration: span.duration,
      offset: span.trim_in,
      rate: 1,
      gain: dbToGain(clip.gainDb),
      fadeIn: Math.min(clip.fadeIn, span.duration),
      fadeOut: Math.min(clip.fadeOut, span.duration),
      duck: clip.duck && clip.kind !== "voice",
      loop: clip.loop,
    });
  }
  // The export ducks only when there is a voice to duck under.
  if (!sounds.some((s) => s.kind === "voice")) sounds.forEach((s) => (s.duck = false));
  const captions = captionRows(project, timeline)
    .filter((row) => row.start !== null && row.end !== null && (row.end as number) > (row.start as number))
    .map((row) => ({ start: row.start as number, end: row.end as number, text: row.text }));
  return {
    total: timeline.total,
    pieces,
    sounds,
    captions,
    missing: pieces.filter((p) => p.media.kind !== "clip").length,
    drafts: pieces.filter((p) => p.draft).length,
  };
}

export interface VisibleLayer {
  piece: CutPiece;
  /** 0..1; an incoming shot fades up over the outgoing one during a crossfade. */
  opacity: number;
}

/** The shots on screen at `t`, back to front. */
export function visibleAt(cut: RoughCut, t: number): VisibleLayer[] {
  const last = cut.pieces.length - 1;
  const out: VisibleLayer[] = [];
  cut.pieces.forEach((piece, i) => {
    const inside = t >= piece.start && (t < piece.end || (i === last && t <= piece.end));
    if (!inside) return;
    const opacity = piece.fadeIn > 0 && t < piece.start + piece.fadeIn ? (t - piece.start) / piece.fadeIn : 1;
    out.push({ piece, opacity: Math.max(0, Math.min(1, opacity)) });
  });
  return out;
}

/** The shot that "owns" time `t` (the newest one that has started): its label and marker are shown. */
export function pieceIndexAt(cut: RoughCut, t: number): number {
  let found = 0;
  cut.pieces.forEach((piece, i) => {
    if (piece.start <= t + 1e-6) found = i;
  });
  return found;
}

/** The source second a shot's clip shows at film time `t`. */
export function sourceTimeAt(piece: CutPiece, t: number): number {
  const into = Math.max(0, Math.min(piece.duration, t - piece.start));
  return piece.sourceIn + into * piece.speed;
}

export interface SoundStart {
  /** Film seconds from the play position until the sound starts (0 when it is already playing). */
  delay: number;
  /** Source second to start from. */
  offset: number;
  /** Film seconds it plays for. */
  length: number;
}

/**
 * When and where a sound starts if the film plays from `from`, or null when it is over by then.
 * `sourceSeconds` (the decoded file's length) wraps the start of a looping sound.
 */
export function soundStart(sound: CutSound, from: number, sourceSeconds: number | null = null): SoundStart | null {
  const end = sound.start + sound.duration;
  if (from >= end - 1e-4) return null;
  const into = Math.max(0, from - sound.start);
  let offset = sound.offset + into * sound.rate;
  if (sound.loop && sourceSeconds && sourceSeconds > sound.offset) {
    const span = sourceSeconds - sound.offset;
    offset = sound.offset + ((offset - sound.offset) % span);
  }
  return { delay: Math.max(0, sound.start - from), offset, length: end - Math.max(sound.start, from) };
}

/** The fade envelope (0..1, before gain and ducking) of a sound at film time `t`. */
export function envelopeAt(sound: CutSound, t: number): number {
  const end = sound.start + sound.duration;
  if (t < sound.start || t > end) return 0;
  let value = 1;
  if (sound.fadeIn > 0 && t < sound.start + sound.fadeIn) value = Math.min(value, (t - sound.start) / sound.fadeIn);
  if (sound.fadeOut > 0 && t > end - sound.fadeOut) value = Math.min(value, (end - t) / sound.fadeOut);
  return Math.max(0, Math.min(1, value));
}

/** Points to ramp the envelope through, linearly, when playing from `from` (film seconds, ascending). */
export function envelopePoints(sound: CutSound, from: number): Array<{ t: number; value: number }> {
  const end = sound.start + sound.duration;
  const begin = Math.max(from, sound.start);
  if (begin >= end) return [];
  const keys = [sound.start + sound.fadeIn, end - sound.fadeOut, end].filter((k) => k > begin + 1e-6);
  const unique = [...new Set(keys.map((k) => Math.round(k * 1e6) / 1e6))].sort((a, b) => a - b);
  return [{ t: begin, value: envelopeAt(sound, begin) }, ...unique.map((k) => ({ t: k, value: envelopeAt(sound, k) }))];
}

/** Time ranges where a voice line plays (merged), for ducking the sounds marked `duck`. */
export function duckWindows(cut: RoughCut): Array<[number, number]> {
  const voice = cut.sounds
    .filter((s) => s.kind === "voice")
    .map((s): [number, number] => [s.start, s.start + s.duration])
    .sort((a, b) => a[0] - b[0]);
  const out: Array<[number, number]> = [];
  for (const [a, b] of voice) {
    const prev = out[out.length - 1];
    if (prev && a <= prev[1]) prev[1] = Math.max(prev[1], b);
    else out.push([a, b]);
  }
  return out;
}

/** The caption text on screen at `t` (overlapping cues are stacked), or null. */
export function captionAt(cut: RoughCut, t: number): string | null {
  const on = cut.captions.filter((c) => t >= c.start && t < c.end).map((c) => c.text);
  return on.length ? on.join("\n") : null;
}

/** Where "previous shot" / "next shot" goes from `t`. Previous restarts the current shot unless it just began. */
export function shotJump(cut: RoughCut, t: number, direction: 1 | -1): number {
  if (!cut.pieces.length) return 0;
  const index = pieceIndexAt(cut, t);
  if (direction === 1) return index + 1 < cut.pieces.length ? cut.pieces[index + 1].start : cut.total;
  const current = cut.pieces[index];
  if (t - current.start > 0.5 || index === 0) return current.start;
  return cut.pieces[index - 1].start;
}

export type PreviewKeyAction =
  | { type: "toggle-play" }
  | { type: "seek"; deltaS: number }
  | { type: "seek-to"; where: "start" | "end" }
  | { type: "shot"; direction: 1 | -1 }
  | { type: "captions" };

/** The player's keys, the Mixer's set plus shot steps: Space/K, J/L, arrows (Shift x10), Home/End, PageUp/PageDown, C. */
export function previewKeyAction(key: string, shift: boolean): PreviewKeyAction | null {
  switch (key) {
    case " ":
    case "k":
    case "K":
      return { type: "toggle-play" };
    case "j":
    case "J":
      return { type: "seek", deltaS: -5 };
    case "l":
    case "L":
      return { type: "seek", deltaS: 5 };
    case "ArrowLeft":
      return { type: "seek", deltaS: shift ? -10 : -1 };
    case "ArrowRight":
      return { type: "seek", deltaS: shift ? 10 : 1 };
    case "Home":
      return { type: "seek-to", where: "start" };
    case "End":
      return { type: "seek-to", where: "end" };
    case "PageUp":
    case ",":
      return { type: "shot", direction: -1 };
    case "PageDown":
    case ".":
      return { type: "shot", direction: 1 };
    case "c":
    case "C":
      return { type: "captions" };
    default:
      return null;
  }
}

export type MarkerState = "clip" | "draft" | "still" | "none";

export interface CutMarker {
  shotId: string;
  index: number;
  start: number;
  /** Percent of the film. */
  left: number;
  width: number;
  state: MarkerState;
}

/** The scrubber's shot segments, each marked by what it will show. */
export function cutMarkers(cut: RoughCut): CutMarker[] {
  const total = cut.total > 0 ? cut.total : 1;
  return cut.pieces.map((piece) => ({
    shotId: piece.shotId,
    index: piece.index,
    start: piece.start,
    left: (piece.start / total) * 100,
    width: (Math.max(0, piece.end - piece.start) / total) * 100,
    state: piece.draft ? "draft" : piece.media.kind === "clip" ? "clip" : piece.media.kind === "still" ? "still" : "none",
  }));
}

/** The project files the player needs: clips (picture and their own sound), stills, and the audio clips. */
export function mediaFiles(cut: RoughCut): { videos: string[]; stills: string[]; audio: string[] } {
  const videos = new Set<string>();
  const stills = new Set<string>();
  for (const piece of cut.pieces) {
    if (piece.media.kind === "clip") videos.add(piece.media.clip);
    if (piece.media.kind === "still") stills.add(piece.media.still);
  }
  const audio = new Set(cut.sounds.filter((s) => s.kind !== "native").map((s) => s.path));
  return { videos: [...videos], stills: [...stills], audio: [...audio] };
}

/**
 * How a playing video element follows the film clock: it plays at its shot's speed and is only moved
 * (a seek) when it is more than a frame or two off. Rate nudges were tried and dropped: each rate change
 * stalls the element for a moment in the browser, which made the drift worse, not better.
 */
export function videoCorrection(currentS: number, wantedS: number, speed: number): { rate: number; seek: boolean } {
  return { rate: speed, seek: Math.abs(currentS - wantedS) > 0.1 };
}
