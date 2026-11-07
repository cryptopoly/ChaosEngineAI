/**
 * Pure helpers for the Storyboard Projects and Board screens: shot card state and class names, the
 * takes strip keyboard model, batch progress, reorder, which shots a batch should make, and project
 * summaries. No React here so the rules are unit-tested directly.
 */
import { StoryboardRevisionConflict } from "../../api/storyboard";
import type {
  StoryboardBatchRecord,
  StoryboardExportPreset,
  StoryboardProject,
  StoryboardShot,
  StoryboardShotStatus,
  StoryboardTimeline,
} from "../../types";

const STORAGE_KEY = "chaos.storyboard.openProject";

export function readOpenProjectId(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY) || null;
  } catch {
    return null;
  }
}

export function writeOpenProjectId(id: string | null): void {
  try {
    if (id) window.localStorage.setItem(STORAGE_KEY, id);
    else window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // storage unavailable: the open project is simply not remembered
  }
}

// ---- failed requests ----
export type FailureOutcome =
  | { kind: "conflict"; project: StoryboardProject }
  | { kind: "error"; message: string };

/**
 * What the screen does with a failed change: a stale revision adopts the project the server sent and
 * shows the short "Changed elsewhere" notice; anything else is an error message in the server's words.
 */
export function classifyFailure(err: unknown): FailureOutcome {
  if (err instanceof StoryboardRevisionConflict) return { kind: "conflict", project: err.project };
  return { kind: "error", message: err instanceof Error ? err.message : String(err) };
}

// ---- export presets ----
export const EXPORT_PRESETS: Record<StoryboardExportPreset, { width: number; height: number }> = {
  "16:9": { width: 1280, height: 720 },
  "9:16": { width: 720, height: 1280 },
  "1:1": { width: 1024, height: 1024 },
};

export const FPS_CHOICES = [24, 30] as const;

/** The `defaults` object for a new project from the dialog's choices. */
export function newProjectDefaults(preset: StoryboardExportPreset, fps: number): Record<string, unknown> {
  return { ...EXPORT_PRESETS[preset], fps, exportPreset: preset };
}

// ---- shot card state ----
export type ShotLiveState = "queued" | "running";

/** Live batch state per shot id: the furthest-along unfinished result wins ("running" over "queued"). */
export function liveShotStates(batch: StoryboardBatchRecord | null): Map<string, ShotLiveState> {
  const live = new Map<string, ShotLiveState>();
  if (!batch || (batch.state !== "queued" && batch.state !== "running")) return live;
  for (const result of batch.results) {
    if (result.state === "running") live.set(result.shotId, "running");
    else if (result.state === "queued" && live.get(result.shotId) !== "running") live.set(result.shotId, "queued");
  }
  return live;
}

/** What the card shows: the batch's live state wins over the stored status. */
export function effectiveShotStatus(shot: StoryboardShot, live: ShotLiveState | undefined): StoryboardShotStatus {
  return live ?? shot.status;
}

/** `shot-card shot-card--<status>` plus `shot-card--locked` for a locked shot. */
export function shotCardClassName(status: StoryboardShotStatus, locked: boolean): string {
  return `shot-card shot-card--${status}${locked ? " shot-card--locked" : ""}`;
}

/** Zero-based position of the shot a stale shot continues from, or null. */
export function staleAfterShot(shots: StoryboardShot[], index: number): number | null {
  const shot = shots[index];
  if (!shot || shot.status !== "stale") return null;
  if (shot.startFrom === "previous-shot" && index > 0) return index - 1;
  return null;
}

export function activeTakeOf(shot: StoryboardShot) {
  return shot.takes.find((take) => take.id === shot.activeTake) ?? null;
}

/** The file to show as the card thumbnail: the still when there is one, else the clip. */
export function thumbnailOf(shot: StoryboardShot): { kind: "still" | "clip"; path: string } | null {
  const take = activeTakeOf(shot);
  if (!take) return null;
  if (take.still) return { kind: "still", path: take.still };
  if (take.clip) return { kind: "clip", path: take.clip };
  return null;
}

// ---- takes strip keyboard model (roving tabindex) ----
export type TakesKey = "ArrowLeft" | "ArrowRight" | "ArrowUp" | "ArrowDown" | "Home" | "End";

/** The index to focus after a key, or null when the key is not handled. */
export function nextTakeIndex(current: number, count: number, key: string): number | null {
  if (count <= 0) return null;
  switch (key) {
    case "ArrowRight":
    case "ArrowDown":
      return Math.min(count - 1, current + 1);
    case "ArrowLeft":
    case "ArrowUp":
      return Math.max(0, current - 1);
    case "Home":
      return 0;
    case "End":
      return count - 1;
    default:
      return null;
  }
}

/** Which take is tabbable: the focused one, else the active one, else the first. */
export function rovingTakeIndex(takeIds: string[], activeId: string | null, focused: number | null): number {
  if (focused != null && focused >= 0 && focused < takeIds.length) return focused;
  const active = activeId ? takeIds.indexOf(activeId) : -1;
  return active >= 0 ? active : 0;
}

// ---- batches ----
export interface BatchProgress {
  kind: "stills" | "clips";
  done: number;
  total: number;
  /** One-based number of the shot being made, of `shotCount`. */
  shotNumber: number;
  shotCount: number;
  /** True when clips follow the stills. */
  thenClips: boolean;
}

const FINISHED = new Set(["done", "failed", "skipped", "cancelled"]);

export function isBatchRunning(batch: StoryboardBatchRecord | null | undefined): boolean {
  return !!batch && (batch.state === "queued" || batch.state === "running");
}

/** Progress of a batch for the sticky bar: stills first, then clips. Null for an empty batch. */
export function batchProgress(batch: StoryboardBatchRecord): BatchProgress | null {
  const stills = batch.results.filter((r) => r.kind === "still");
  const clips = batch.results.filter((r) => r.kind === "clip");
  if (stills.length === 0 && clips.length === 0) return null;
  const stillsOpen = stills.some((r) => !FINISHED.has(r.state));
  const kind: "stills" | "clips" = stills.length > 0 && (stillsOpen || clips.length === 0) ? "stills" : "clips";
  const group = kind === "stills" ? stills : clips;
  const done = group.filter((r) => FINISHED.has(r.state)).length;
  const shotCount = new Set(batch.results.map((r) => r.shotId)).size;
  const ids: string[] = [];
  for (const r of batch.results) if (!ids.includes(r.shotId)) ids.push(r.shotId);
  const current = group.find((r) => !FINISHED.has(r.state));
  const shotNumber = Math.min(shotCount, current ? ids.indexOf(current.shotId) + 1 : ids.indexOf(group[group.length - 1].shotId) + 1);
  return { kind, done, total: group.length, shotNumber, shotCount, thenClips: stills.length > 0 && clips.length > 0 };
}

export function batchFailures(batch: StoryboardBatchRecord | null): { shotId: string; error: string | null }[] {
  if (!batch) return [];
  return batch.results.filter((r) => r.state === "failed").map((r) => ({ shotId: r.shotId, error: r.error }));
}

/** A shot a batch would work on if the user did not name it. Mirrors `needs_work` in batch.py. */
export function shotNeedsWork(shot: StoryboardShot): boolean {
  if (shot.startFrom === "upload" || shot.status === "missing") return false;
  return shot.redo || activeTakeOf(shot) === null || ["idle", "stale", "failed"].includes(shot.status);
}

/** "Make missing": unlocked shots with no active take that a batch can make. */
export function missingShotIds(shots: StoryboardShot[]): string[] {
  return shots
    .filter((s) => !s.locked && s.startFrom !== "upload" && s.status !== "missing" && activeTakeOf(s) === null)
    .map((s) => s.id);
}

/** "Make all unlocked": every unlocked shot a batch can work on. */
export function unlockedShotIds(shots: StoryboardShot[]): string[] {
  return shots.filter((s) => !s.locked && s.startFrom !== "upload" && s.status !== "missing").map((s) => s.id);
}

// ---- reorder ----
/** Move the item at `from` to position `to` (both clamped); returns a new array. */
export function moveItem<T>(items: T[], from: number, to: number): T[] {
  if (from < 0 || from >= items.length) return items;
  const target = Math.max(0, Math.min(items.length - 1, to));
  if (target === from) return items;
  const next = items.slice();
  const [item] = next.splice(from, 1);
  next.splice(target, 0, item);
  return next;
}

/** New shot id order after moving `shotId` one step; null when it cannot move that way. */
export function reorderByStep(ids: string[], shotId: string, step: -1 | 1): string[] | null {
  const from = ids.indexOf(shotId);
  const to = from + step;
  if (from < 0 || to < 0 || to >= ids.length) return null;
  return moveItem(ids, from, to);
}

/** New order after dropping the dragged shot on the shot at `overId`; null when nothing changes. */
export function reorderByDrop(ids: string[], dragId: string, overId: string): string[] | null {
  const from = ids.indexOf(dragId);
  const to = ids.indexOf(overId);
  if (from < 0 || to < 0 || from === to) return null;
  return moveItem(ids, from, to);
}

// ---- transitions ----
export const DEFAULT_CROSSFADE_SECONDS = 0.5;

/** The transition after the user changes its type or its seconds. A cut has no length; a crossfade has 0.1 to 5 s. */
export function transitionWith(
  current: { type: "cut" | "crossfade"; seconds: number },
  patch: { type?: "cut" | "crossfade"; seconds?: number },
): { type: "cut" | "crossfade"; seconds: number } {
  const type = patch.type ?? current.type;
  if (type === "cut") return { type: "cut", seconds: 0 };
  const raw = patch.seconds ?? (current.seconds > 0 ? current.seconds : DEFAULT_CROSSFADE_SECONDS);
  const clamped = Math.min(5, Math.max(0.1, Number.isFinite(raw) ? raw : DEFAULT_CROSSFADE_SECONDS));
  return { type: "crossfade", seconds: Math.round(clamped * 10) / 10 };
}

// ---- shot editor draft ----
export interface ShotDraft {
  stillPrompt: string;
  motionPrompt: string;
  startFrom: StoryboardShot["startFrom"];
  /** Numbers are kept as typed so a half-typed value is not rewritten under the cursor. */
  seconds: string;
  seed: string;
  trimIn: string;
  trimOut: string;
  speed: string;
  nativeSoundDb: string;
  referenceImages: string[];
}

export function draftFromShot(shot: StoryboardShot): ShotDraft {
  return {
    stillPrompt: shot.stillPrompt,
    motionPrompt: shot.motionPrompt,
    startFrom: shot.startFrom,
    seconds: String(shot.seconds),
    seed: shot.seed == null ? "" : String(shot.seed),
    trimIn: String(shot.trimIn),
    trimOut: shot.trimOut == null ? "" : String(shot.trimOut),
    speed: String(shot.speed),
    nativeSoundDb: String(shot.nativeSoundDb),
    referenceImages: shot.referenceImages,
  };
}

function numberOr(text: string, fallback: number): number {
  const value = Number(text);
  return text.trim() !== "" && Number.isFinite(value) ? value : fallback;
}

/** Only the fields the user changed, ready for `edit_shot`. Empty seed or trim-out means "none". */
export function shotPatchFromDraft(shot: StoryboardShot, draft: ShotDraft): Record<string, unknown> {
  const patch: Record<string, unknown> = {};
  if (draft.stillPrompt !== shot.stillPrompt) patch.stillPrompt = draft.stillPrompt;
  if (draft.motionPrompt !== shot.motionPrompt) patch.motionPrompt = draft.motionPrompt;
  if (draft.startFrom !== shot.startFrom) patch.startFrom = draft.startFrom;
  const seconds = numberOr(draft.seconds, shot.seconds);
  if (seconds !== shot.seconds) patch.seconds = seconds;
  const seed = draft.seed.trim() === "" ? null : Math.trunc(numberOr(draft.seed, shot.seed ?? 0));
  if (seed !== shot.seed) patch.seed = seed;
  const trimIn = numberOr(draft.trimIn, shot.trimIn);
  if (trimIn !== shot.trimIn) patch.trimIn = trimIn;
  const trimOut = draft.trimOut.trim() === "" ? null : numberOr(draft.trimOut, shot.trimOut ?? 0);
  if (trimOut !== shot.trimOut) patch.trimOut = trimOut;
  const speed = numberOr(draft.speed, shot.speed);
  if (speed !== shot.speed) patch.speed = speed;
  const sound = numberOr(draft.nativeSoundDb, shot.nativeSoundDb);
  if (sound !== shot.nativeSoundDb) patch.nativeSoundDb = sound;
  const sameRefs =
    draft.referenceImages.length === shot.referenceImages.length &&
    draft.referenceImages.every((id, i) => id === shot.referenceImages[i]);
  if (!sameRefs) patch.referenceImages = draft.referenceImages;
  return patch;
}

/** Toggle one reference, keeping the model's limit of 8. */
export function toggleReference(ids: string[], refId: string): string[] {
  if (ids.includes(refId)) return ids.filter((id) => id !== refId);
  return ids.length >= 8 ? ids : [...ids, refId];
}

const IMAGE_EXT = /\.(png|jpe?g|webp)$/i;
const VIDEO_EXT = /\.(mp4|mov|m4v|webm)$/i;

/** Which `import_take` argument a file path belongs to, from its extension. */
export function importKindOf(path: string): "still" | "clip" | null {
  if (IMAGE_EXT.test(path)) return "still";
  if (VIDEO_EXT.test(path)) return "clip";
  return null;
}

// ---- summaries ----
export function staleCount(project: StoryboardProject): number {
  return project.shots.filter((s) => s.status === "stale").length;
}

export function filmSeconds(timeline: StoryboardTimeline | null | undefined): number {
  return timeline ? Math.round(timeline.total * 10) / 10 : 0;
}

/** The newest sign of activity in a project (epoch seconds): its creation, a take or a finished batch. */
export function projectRecency(project: StoryboardProject): number {
  let latest = project.createdAt;
  for (const shot of project.shots) for (const take of shot.takes) latest = Math.max(latest, take.createdAt);
  for (const batch of project.batches) latest = Math.max(latest, batch.finishedAt ?? batch.createdAt);
  return latest;
}

export function sortByRecent<T extends { recency: number }>(items: T[]): T[] {
  return items.slice().sort((a, b) => b.recency - a.recency);
}

/** A short relative time ("2 hours ago") in the given locale. */
export function relativeTime(epochSeconds: number, nowMs: number, locale?: string): string {
  const diff = Math.round(epochSeconds - nowMs / 1000);
  const fmt = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
  const abs = Math.abs(diff);
  if (abs < 60) return fmt.format(0, "second");
  if (abs < 3600) return fmt.format(Math.round(diff / 60), "minute");
  if (abs < 86400) return fmt.format(Math.round(diff / 3600), "hour");
  return fmt.format(Math.round(diff / 86400), "day");
}

// ---- keys and focus in the Board's drawers and cards ----

/** Plain keys only: a shortcut with Cmd, Ctrl or Alt held belongs to the app or the browser (Cmd+R reloads). */
export function isBareKey(event: { metaKey: boolean; ctrlKey: boolean; altKey: boolean }): boolean {
  return !event.metaKey && !event.ctrlKey && !event.altKey;
}

export type DrawerEscape = "close" | "ask" | "none";

/**
 * What Escape does in the shot editor: nothing inside an input-method composition (it cancels the
 * composition) or while the discard question is open, otherwise it closes, asking first when there are
 * unsaved edits.
 */
export function drawerEscapeAction(input: {
  key: string;
  isComposing: boolean;
  dirty: boolean;
  confirming: boolean;
}): DrawerEscape {
  if (input.key !== "Escape" || input.isComposing || input.confirming) return "none";
  return input.dirty ? "ask" : "close";
}

/**
 * Where Tab goes in a focus trap of `count` stops when focus is on stop `active` (-1 = the container
 * itself): the stop to move to when Tab would leave the trap, or null to let the browser move on.
 */
export function trapTabTarget(count: number, active: number, shift: boolean): number | null {
  if (count <= 0) return null;
  if (active < 0) return shift ? count - 1 : 0;
  if (shift && active === 0) return count - 1;
  if (!shift && active === count - 1) return 0;
  return null;
}

/** `fn` over `items` with at most `limit` running at once; results keep the order of `items`. */
export async function mapWithLimit<T, R>(items: readonly T[], limit: number, fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    for (;;) {
      const index = next;
      next += 1;
      if (index >= items.length) return;
      results[index] = await fn(items[index], index);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, worker));
  return results;
}
