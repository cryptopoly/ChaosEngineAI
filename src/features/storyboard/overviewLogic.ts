/**
 * Overview helpers (D-65): shot counts and the recent activity list. The next step and the checklist that lived here
 * were replaced by the stage rail (audit change 1, `guide/stageLogic.ts`), and "Before you export" by the readiness
 * check (`exportReadiness.ts`). No React here so the rules are unit-tested directly.
 */
import type { StoryboardProject } from "../../types";
import { activeTakeOf } from "./storyboardLogic";

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
