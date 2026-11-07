/**
 * Pure rules for the Mixer screen: export presets and the pixels they give, the AI-label options
 * (D-34), the loudness field, render state, the keyboard map, and the arguments for adding sound.
 * No React here so the rules are unit-tested directly.
 */
import type {
  StoryboardDefaults,
  StoryboardExportPreset,
  StoryboardProject,
  StoryboardRenderJob,
  StoryboardShotSpan,
  StoryboardTimeline,
  VoiceLibrarySummary,
} from "../../../types";
import type { AudioLibraryChoice } from "../../../components/InputAudioPicker";
import { dropToAnchor, shiftPosition, type AudioLaneId, type PositionPatch } from "./timelineGeometry";

// ---- export presets ----
export const EXPORT_PRESET_ORDER: StoryboardExportPreset[] = ["16:9", "9:16", "1:1"];
const PREVIEW_SHORT_SIDE = 360;

function even(value: number): number {
  return Math.round(value / 2) * 2;
}

/**
 * The frame size an export (or preview) gets: a mirror of `output_size` in timeline.py. The preset
 * reframes the project's long side; it never changes the shots' own size, so nothing is regenerated.
 */
export function exportSize(
  defaults: Pick<StoryboardDefaults, "width" | "height">,
  preset: StoryboardExportPreset,
  preview = false,
): { width: number; height: number } {
  const longSide = Math.max(defaults.width, defaults.height);
  const shortSide = Math.min(defaults.width, defaults.height);
  let width: number;
  let height: number;
  if (preset === "16:9") {
    width = longSide;
    height = (longSide * 9) / 16;
  } else if (preset === "9:16") {
    width = (longSide * 9) / 16;
    height = longSide;
  } else {
    width = shortSide;
    height = shortSide;
  }
  if (preview) {
    const scale = PREVIEW_SHORT_SIDE / Math.min(width, height);
    if (scale < 1) {
      width *= scale;
      height *= scale;
    }
  }
  return { width: even(width), height: even(height) };
}

/** The `edit_project` arguments for picking a preset. Only the export preset changes. */
export function presetPatch(preset: StoryboardExportPreset): { defaults: { exportPreset: StoryboardExportPreset } } {
  return { defaults: { exportPreset: preset } };
}

// ---- AI label (D-34) ----
export interface AiLabelState {
  /** What the switch shows: the tag is written. */
  metadata: boolean;
  /** A cloned voice forces the tag on; the switch is locked. */
  forced: boolean;
}

export function hasClonedVoice(project: Pick<StoryboardProject, "audio">): boolean {
  return project.audio.some((clip) => clip.source.clonedVoice);
}

export function aiLabelState(project: Pick<StoryboardProject, "audio" | "export">): AiLabelState {
  const forced = hasClonedVoice(project);
  return { metadata: forced || project.export.aiLabel.embedTag, forced };
}

/** The `edit_project` arguments for one AI-label change; null when it would change nothing (a locked tag). */
export function aiLabelPatch(
  project: Pick<StoryboardProject, "audio" | "export">,
  change: { metadata?: boolean },
): { export: { aiLabel: { embedTag?: boolean } } } | null {
  const aiLabel: { embedTag?: boolean } = {};
  if (change.metadata !== undefined && !hasClonedVoice(project)) aiLabel.embedTag = change.metadata;
  return Object.keys(aiLabel).length > 0 ? { export: { aiLabel } } : null;
}

// ---- loudness ----
export const LOUDNESS_MIN = -40;
export const LOUDNESS_MAX = -5;
export const LOUDNESS_SUGGESTED = -16;

/** The typed loudness target: a number in range, `null` for "off" (empty), or `undefined` when it is not a usable number. */
export function parseLoudness(text: string): number | null | undefined {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  const value = Number(trimmed.replace(",", "."));
  if (!Number.isFinite(value)) return undefined;
  return Math.round(Math.max(LOUDNESS_MIN, Math.min(LOUDNESS_MAX, value)) * 10) / 10;
}

// ---- render state ----
export type RenderPhaseId = "normalise" | "join" | "mix" | "encode";
export const RENDER_PHASES: RenderPhaseId[] = ["normalise", "join", "mix", "encode"];

const STAGE_TO_PHASE: Record<string, RenderPhaseId> = {
  shots: "normalise",
  card: "normalise",
  join: "join",
  measure: "mix",
  mix: "encode",
};

/** Which of the four phases the assembler's stage name belongs to (the first one before any stage is reported). */
export function renderPhase(message: string | undefined): RenderPhaseId {
  return STAGE_TO_PHASE[(message ?? "").trim().toLowerCase()] ?? "normalise";
}

export type RenderUiState = "idle" | "running" | "done" | "failed" | "cancelled";

export function renderUiState(job: StoryboardRenderJob | null, starting = false): RenderUiState {
  if (starting) return "running";
  return job ? job.state : "idle";
}

export function isRenderRunning(job: StoryboardRenderJob | null): boolean {
  return !!job && job.state === "running";
}

/** A whole percent for the progress line. */
export function renderPercent(job: StoryboardRenderJob | null): number {
  return job ? Math.max(0, Math.min(100, Math.round(job.progress * 100))) : 0;
}

/** True when the project has changed since the preview was rendered. Unknown (no record) is not stale. */
export function previewIsStale(renderedRevision: number | null, currentRevision: number): boolean {
  return renderedRevision !== null && renderedRevision !== currentRevision;
}

// ---- which revision the preview shows ----
type RevisionStore = Pick<Storage, "getItem" | "setItem">;

const renderedKey = (projectId: string) => `chaosengine.storyboard.previewRevision.${projectId}`;

function sessionStore(): RevisionStore | null {
  try {
    return typeof sessionStorage === "undefined" ? null : sessionStorage;
  } catch {
    return null;
  }
}

/** The project revision the last finished preview was rendered from, remembered for the session. */
export function readRenderedRevision(projectId: string, store: RevisionStore | null = sessionStore()): number | null {
  try {
    const raw = store?.getItem(renderedKey(projectId));
    if (raw == null) return null;
    const value = Number(raw);
    return Number.isInteger(value) ? value : null;
  } catch {
    return null;
  }
}

export function writeRenderedRevision(projectId: string, revision: number, store: RevisionStore | null = sessionStore()): void {
  try {
    store?.setItem(renderedKey(projectId), String(revision));
  } catch {
    // storage is only a convenience
  }
}

// ---- playhead readout ----
export const PLAYHEAD_READOUT_MS = 100;

/** The numeric playhead is redrawn at most ten times a second while the film plays. */
export function readoutDue(lastPublishedMs: number, nowMs: number): boolean {
  return nowMs - lastPublishedMs >= PLAYHEAD_READOUT_MS;
}

export function readyShotCount(timeline: StoryboardTimeline | null): number {
  return timeline ? timeline.shots.filter((s) => s.ready).length : 0;
}

export function leftOutShotCount(timeline: StoryboardTimeline | null): number {
  return timeline ? timeline.shots.filter((s) => !s.ready).length : 0;
}

/** A film can be rendered when at least one shot has a playable clip. */
export function canRender(timeline: StoryboardTimeline | null): boolean {
  return readyShotCount(timeline) > 0;
}

export function formatBytes(bytes: number): string {
  if (!(bytes >= 0)) return "";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${Math.round((bytes / (1024 * 1024)) * 10) / 10} MB`;
  return `${Math.round((bytes / 1024 ** 3) * 100) / 100} GB`;
}

export function fileNameOf(rel: string): string {
  return rel.split("/").pop() ?? rel;
}

/** Best-effort absolute path of a project file, for "Copy path" (the backend does not report it). */
export function projectFilePath(dataDirectory: string | undefined, projectId: string, rel: string): string {
  const root = (dataDirectory || "~/.chaosengine").replace(/[\\/]+$/, "");
  return `${root}/storyboards/${projectId}/${rel}`;
}

// ---- keyboard ----
export type TargetKind = "control" | "clip" | "other";

export type MixerKeyAction =
  | { type: "toggle-play" }
  | { type: "seek"; deltaS: number }
  | { type: "seek-to"; where: "start" | "end" }
  | { type: "nudge-clip"; deltaS: number }
  | { type: "zoom"; direction: 1 | -1 }
  | { type: "remove-clip" }
  | { type: "trim"; side: "in" | "out" }
  | { type: "undo" }
  | { type: "redo" }
  | { type: "deselect" };

export interface MixerKeyInput {
  key: string;
  shift: boolean;
  alt: boolean;
  mod: boolean;
  /** What has focus: a form control or button (keys are theirs), a clip, or anything else on the screen. */
  target: TargetKind;
  /** The audio clip selected on the timeline, if any. */
  selectedClipId: string | null;
  /** The timeline block that has focus (`data-clip-id`), if one does. */
  focusedClipId: string | null;
  fps: number;
}

export const SEEK_BIG_S = 10;
export const SEEK_JKL_S = 5;

/** What a key does on the Mixer, or null when it is not ours. */
export function mixerKeyAction(input: MixerKeyInput): MixerKeyAction | null {
  const { key, shift, alt, mod, target, selectedClipId, focusedClipId, fps } = input;
  if (target === "control") return null;
  const clipSelected = selectedClipId !== null;
  // Nudge and Delete belong to the focused block, so they act only when that block is the selected clip.
  const onSelectedClip = target === "clip" && clipSelected && focusedClipId === selectedClipId;
  const lower = key.length === 1 ? key.toLowerCase() : key;
  if (mod) {
    if (lower === "z") return { type: shift ? "redo" : "undo" };
    if (lower === "y") return { type: "redo" };
    return null;
  }
  switch (lower) {
    case " ":
    case "k":
      return { type: "toggle-play" };
    case "j":
      return { type: "seek", deltaS: -SEEK_JKL_S };
    case "l":
      return { type: "seek", deltaS: SEEK_JKL_S };
    case "ArrowLeft":
    case "ArrowRight": {
      const sign = lower === "ArrowLeft" ? -1 : 1;
      if (onSelectedClip) return { type: "nudge-clip", deltaS: sign * (shift ? 1 : 0.1) };
      const step = alt ? 1 / Math.max(1, fps) : shift ? SEEK_BIG_S : 1;
      return { type: "seek", deltaS: sign * step };
    }
    case "Home":
      return { type: "seek-to", where: "start" };
    case "End":
      return { type: "seek-to", where: "end" };
    case "+":
    case "=":
      return { type: "zoom", direction: 1 };
    case "-":
    case "_":
      return { type: "zoom", direction: -1 };
    case "Delete":
    case "Backspace":
      return onSelectedClip ? { type: "remove-clip" } : null;
    case "[":
      return clipSelected ? { type: "trim", side: "in" } : null;
    case "]":
      return clipSelected ? { type: "trim", side: "out" } : null;
    case "Escape":
      return { type: "deselect" };
    default:
      return null;
  }
}

/** The playhead after a seek, kept inside the film. */
export function seekedTime(current: number, deltaS: number, durationS: number): number {
  return Math.max(0, Math.min(durationS, Math.round((current + deltaS) * 1000) / 1000));
}

// ---- nudging ----
export interface PendingNudge {
  clipId: string;
  /** Seconds, summed over the key presses since the last send. */
  delta: number;
}

/** Add one key press to the pending nudge; a press on another clip hands back the old one to send first. */
export function accumulateNudge(
  pending: PendingNudge | null,
  clipId: string,
  delta: number,
): { pending: PendingNudge; flush: PendingNudge | null } {
  if (pending && pending.clipId === clipId) return { pending: { clipId, delta: pending.delta + delta }, flush: null };
  return { pending: { clipId, delta }, flush: pending };
}

/**
 * The `edit_audio` arguments for a pending nudge, worked out from the project as it is when the request
 * runs, so two nudges sent back to back add up instead of both starting from the same old position.
 * Null when the clip is gone.
 */
export function nudgeArgs(project: Pick<StoryboardProject, "audio">, pending: PendingNudge) {
  const clip = project.audio.find((c) => c.id === pending.clipId);
  return clip ? { clipId: clip.id, ...shiftPosition(clip, pending.delta) } : null;
}

// ---- adding sound ----
/** Last resort only: the data folder can be moved, so a pick asks the server for the real path first. */
export const VOICE_LIBRARY_DIR = "~/.chaosengine/voice/library";

/** Voice Library speech clips that have audio, as picker choices. The path is the file in the library folder. */
export function voiceLibraryChoices(items: VoiceLibrarySummary[]): AudioLibraryChoice[] {
  return items
    .filter((item) => item.kind === "speech" && item.audio)
    .map((item) => ({ id: item.id, title: item.title || item.id, path: `${VOICE_LIBRARY_DIR}/${item.id}${item.audio}` }));
}

/** The file to add for a library pick: the absolute path the server reports, else the default folder guess. */
export function resolveLibraryPath(audioPath: string | null | undefined, fallback: string): string {
  return audioPath && audioPath.trim() ? audioPath : fallback;
}

export interface AddAudioArgs {
  kind: "voice" | "sfx" | "music";
  path: string;
  clonedVoice?: boolean;
  label?: string;
  anchor?: { shotId: string; offset: number };
  start?: number;
}

/** `add_audio` arguments for a lane: placed at the playhead, anchored to the shot under it. */
export function addAudioArgs(
  lane: AudioLaneId,
  picked: { path: string; name?: string },
  playheadS: number,
  spans: Pick<StoryboardShotSpan, "id" | "start" | "end">[],
  clonedVoice = false,
): AddAudioArgs {
  const position: PositionPatch = dropToAnchor(playheadS, spans, false);
  const args: AddAudioArgs = { kind: lane, path: picked.path.trim(), ...position };
  const name = picked.name ?? picked.path.trim().split(/[\\/]/).pop() ?? "";
  const label = name.replace(/\.[^.]+$/, "").replace(/[\u0000-\u001f\u007f]/g, "").slice(0, 200);
  if (label) args.label = label;
  if (lane === "voice" && clonedVoice) args.clonedVoice = true;
  return args;
}
