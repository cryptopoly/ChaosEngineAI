/**
 * Overview helpers (D-65): shot counts, the recent activity list and the "Before you export" preflight (stream E
 * moves `preflight` to its own file). The next step and the checklist that lived here were replaced by the stage
 * rail (audit change 1, `guide/stageLogic.ts`). No React here so the rules are unit-tested directly.
 */
import type { StoryboardProject } from "../../types";
import type { ProjectTabId } from "./projectRoute";
import { activeTakeOf, draftShotIds } from "./storyboardLogic";

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

export type PreflightId = "format" | "upscale" | "clips" | "stale" | "drafts" | "loudness" | "captions" | "aiTag" | "cloned" | "preview";

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
    /** Captions that fall inside the film (`filmCaptionCount`); every caption of the project when not given. */
    captions?: number;
  },
): PreflightLine[] {
  const shots = project.shots.length;
  const stale = project.shots.filter((shot) => shot.status === "stale").length;
  const lines: PreflightLine[] = [];
  if (facts.format) lines.push({ id: "format", ok: true, n: 0, total: 0, tab: "overview", detail: facts.format });
  if (facts.upscaledTo) lines.push({ id: "upscale", ok: true, n: 0, total: 0, tab: null, detail: facts.upscaledTo });
  lines.push({ id: "clips", ok: facts.leftOut === 0 && shots > 0, n: shots - facts.leftOut, total: shots, tab: facts.leftOut > 0 ? "storyboard" : null });
  if (stale > 0) lines.push({ id: "stale", ok: false, n: stale, total: shots, tab: "storyboard" });
  const drafts = draftShotIds(project.shots).length;
  if (drafts > 0) lines.push({ id: "drafts", ok: false, n: drafts, total: shots, tab: "storyboard" });
  const lufs = project.export?.loudnessLufs;
  lines.push({ id: "loudness", ok: true, n: lufs ?? 0, total: 0, tab: null, detail: lufs == null ? "" : String(lufs) });
  const kinds = [project.export?.captionsSrt ? "srt" : "", project.export?.captionsBurn ? "burn" : ""].filter(Boolean).join("+");
  lines.push({ id: "captions", ok: true, n: facts.captions ?? project.captions.length, total: 0, tab: "mixer", detail: kinds });
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

/** The last finished batches, newest first (at most `limit`), with how many shots each made or failed. A paused
 *  batch (S-M1) is not finished: it can still resume, so it is left out. */
export function recentActivity(project: Pick<StoryboardProject, "batches">, limit = 3): ActivityLine[] {
  return project.batches
    .filter((batch) => batch.state !== "queued" && batch.state !== "running" && batch.state !== "paused")
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
