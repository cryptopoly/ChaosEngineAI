/**
 * The Overview guide (D-65 4.4): one next step computed from the project, and a checklist that ticks as the work
 * happens. Cast is never required: its row is marked optional and the next step never asks for it. No React
 * here so the rules are unit-tested directly.
 */
import type { StoryboardProject } from "../../types";
import type { ProjectTabId } from "./projectRoute";
import { activeTakeOf, missingShotIds } from "./storyboardLogic";

export type NextStepId = "writeShots" | "makeMissing" | "makeClips" | "addSound" | "renderPreview" | "export" | "done";

export interface NextStep {
  id: NextStepId;
  tab: ProjectTabId;
  /** Shots the step is about (to make, or still without a clip). */
  count: number;
  /** Of those, shots that have a still and need only their clip. */
  stills?: number;
}

export interface GuideFacts {
  /** A preview was rendered and is not older than the project. */
  previewCurrent: boolean;
  /** At least one film was exported. */
  exported: boolean;
}

export type ChecklistId = "format" | "cast" | "shots" | "takes" | "sound" | "preview" | "export";

export interface ChecklistRow {
  id: ChecklistId;
  done: boolean;
  optional: boolean;
  tab: ProjectTabId;
}

/** Shots whose active take has a still but no clip yet. */
export function stillOnlyCount(project: Pick<StoryboardProject, "shots">): number {
  return project.shots.filter((shot) => {
    const take = activeTakeOf(shot);
    return Boolean(take && take.still && !take.clip);
  }).length;
}

export function clipCount(project: Pick<StoryboardProject, "shots">): number {
  return project.shots.filter((shot) => Boolean(activeTakeOf(shot)?.clip)).length;
}

export function nextStep(project: StoryboardProject, facts: GuideFacts): NextStep {
  if (project.shots.length === 0) return { id: "writeShots", tab: "storyboard", count: 0 };
  const missingIds = new Set(missingShotIds(project.shots));
  if (missingIds.size > 0) {
    const stills = project.shots.filter((shot) => missingIds.has(shot.id) && activeTakeOf(shot)?.still && !activeTakeOf(shot)?.clip).length;
    return { id: "makeMissing", tab: "storyboard", count: missingIds.size, stills };
  }
  const stills = stillOnlyCount(project);
  if (stills > 0) return { id: "makeClips", tab: "storyboard", count: stills };
  if (project.audio.length === 0 && !facts.previewCurrent) return { id: "addSound", tab: "mixer", count: 0 };
  if (!facts.previewCurrent) return { id: "renderPreview", tab: "mixer", count: 0 };
  if (!facts.exported) return { id: "export", tab: "export", count: 0 };
  return { id: "done", tab: "export", count: 0 };
}

export function checklist(project: StoryboardProject, facts: GuideFacts): ChecklistRow[] {
  const shots = project.shots.length;
  return [
    { id: "format", done: true, optional: false, tab: "overview" },
    { id: "cast", done: project.references.characters.length + project.references.places.length > 0, optional: true, tab: "cast" },
    { id: "shots", done: shots > 0, optional: false, tab: "storyboard" },
    { id: "takes", done: shots > 0 && clipCount(project) === shots, optional: false, tab: "storyboard" },
    { id: "sound", done: project.audio.length > 0, optional: true, tab: "mixer" },
    { id: "preview", done: facts.previewCurrent, optional: false, tab: "mixer" },
    { id: "export", done: facts.exported, optional: false, tab: "export" },
  ];
}

export type PreflightId = "format" | "upscale" | "clips" | "stale" | "loudness" | "captions" | "aiTag" | "cloned" | "preview";

export interface PreflightLine {
  id: PreflightId;
  ok: boolean;
  /** The numbers the line's text needs. */
  n: number;
  total: number;
  tab: ProjectTabId | null;
  /** Words the line shows as they are (a size, a format, a loudness). */
  detail?: string;
}

/** "Before you export": what the film will hold, what it leaves out, and what to know first. */
export function preflight(
  project: StoryboardProject,
  facts: {
    leftOut: number;
    previewStale: boolean;
    clonedVoice: boolean;
    /** "1920 × 1080 · 24 fps" as the export writes it. */
    format?: string;
    /** The upscaled size, when the export scales the film up. */
    upscaledTo?: string | null;
  },
): PreflightLine[] {
  const shots = project.shots.length;
  const stale = project.shots.filter((shot) => shot.status === "stale").length;
  const lines: PreflightLine[] = [];
  if (facts.format) lines.push({ id: "format", ok: true, n: 0, total: 0, tab: "overview", detail: facts.format });
  if (facts.upscaledTo) lines.push({ id: "upscale", ok: true, n: 0, total: 0, tab: null, detail: facts.upscaledTo });
  lines.push({ id: "clips", ok: facts.leftOut === 0 && shots > 0, n: shots - facts.leftOut, total: shots, tab: facts.leftOut > 0 ? "storyboard" : null });
  if (stale > 0) lines.push({ id: "stale", ok: false, n: stale, total: shots, tab: "storyboard" });
  const lufs = project.export?.loudnessLufs;
  lines.push({ id: "loudness", ok: true, n: lufs ?? 0, total: 0, tab: null, detail: lufs == null ? "" : String(lufs) });
  const kinds = [project.export?.captionsSrt ? "srt" : "", project.export?.captionsBurn ? "burn" : ""].filter(Boolean).join("+");
  lines.push({ id: "captions", ok: true, n: project.captions.length, total: 0, tab: "mixer", detail: kinds });
  lines.push({ id: "aiTag", ok: true, n: facts.clonedVoice || project.export?.aiLabel?.embedTag ? 1 : 0, total: 0, tab: null });
  if (facts.previewStale) lines.push({ id: "preview", ok: false, n: 0, total: 0, tab: "mixer" });
  if (facts.clonedVoice) lines.push({ id: "cloned", ok: true, n: 0, total: 0, tab: null });
  return lines;
}

export interface ActivityLine {
  id: string;
  state: "done" | "failed" | "cancelled" | "interrupted";
  shots: number;
  failed: number;
  at: number;
}

/** The last finished batches, newest first (at most `limit`), with how many shots each made or failed. */
export function recentActivity(project: Pick<StoryboardProject, "batches">, limit = 3): ActivityLine[] {
  return project.batches
    .filter((batch) => batch.state !== "queued" && batch.state !== "running")
    .slice(-limit)
    .reverse()
    .map((batch) => ({
      id: batch.id,
      state: batch.state as ActivityLine["state"],
      shots: new Set(batch.results.map((r) => r.shotId)).size,
      failed: new Set(batch.results.filter((r) => r.state === "failed").map((r) => r.shotId)).size,
      at: batch.finishedAt ?? batch.createdAt,
    }));
}

/** The checklist hides once every row that is not optional is ticked. */
export function checklistComplete(rows: ChecklistRow[]): boolean {
  return rows.every((row) => row.done || row.optional);
}
