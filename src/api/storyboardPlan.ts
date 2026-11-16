/**
 * Plan and style client (S8, S15): the loaded chat model and how the style reaches the still model, shots the
 * model proposes for a brief, a reworked prompt for one shot, a turn of the direction chat, and the installed
 * fonts for the caption look. Nothing here changes the project except the direction chat's log; proposals are
 * applied as ordinary ops (`apply_plan`, `edit_shot`, `plan_changes`) so Undo works.
 */
import type {
  StoryboardPlanChanges,
  StoryboardPlannedShot,
  StoryboardProjectPayload,
} from "../types";
import { request } from "./storyboard";

/** The model can take a while on a long brief: wait up to five minutes. */
export const PLAN_TIMEOUT_MS = 300000;

const base = (projectId: string) => `/api/storyboard/projects/${encodeURIComponent(projectId)}/plan`;

export type StoryboardStyleMode = "pictures" | "words" | "none";

export interface StoryboardPlanStatus {
  chatModel: { name: string; ref: string; vision: boolean } | null;
  style: {
    mode: StoryboardStyleMode;
    stillModel: string | null;
    pictureSlots: number;
    words: string;
    problem: string | null;
  };
}

/** What every model answer carries besides its content. */
export interface StoryboardModelFacts {
  model: string;
  seconds: number;
  sawPictures: boolean;
}

export interface StoryboardProposal extends StoryboardModelFacts {
  shots: StoryboardPlannedShot[];
  note: string;
}

export interface StoryboardRework extends StoryboardModelFacts {
  shot: StoryboardPlannedShot;
  why: string;
}

export interface StoryboardDirection extends StoryboardModelFacts, StoryboardProjectPayload {
  reply: string;
  changes: StoryboardPlanChanges;
}

export interface StoryboardChatTurn {
  role: "user" | "assistant";
  text: string;
}

export async function getStoryboardPlanStatus(projectId: string): Promise<StoryboardPlanStatus> {
  return await request("GET", `${base(projectId)}/status`);
}

export async function proposeStoryboardPlan(
  projectId: string,
  body: { brief?: string; filmSeconds?: number | null; shots?: number | null },
): Promise<StoryboardProposal> {
  const payload: Record<string, unknown> = { brief: body.brief ?? "" };
  if (body.filmSeconds) payload.filmSeconds = body.filmSeconds;
  if (body.shots) payload.shots = body.shots;
  return await request("POST", `${base(projectId)}/propose`, payload, PLAN_TIMEOUT_MS);
}

export async function reworkStoryboardShot(
  projectId: string,
  shotId: string,
  message: string,
  history: StoryboardChatTurn[],
): Promise<StoryboardRework> {
  return await request("POST", `${base(projectId)}/rework`, { shotId, message, history }, PLAN_TIMEOUT_MS);
}

export async function directStoryboardPlan(projectId: string, message: string): Promise<StoryboardDirection> {
  return await request("POST", `${base(projectId)}/direct`, { message }, PLAN_TIMEOUT_MS);
}

export async function getStoryboardFonts(): Promise<string[]> {
  return (await request<{ fonts: string[] }>("GET", "/api/storyboard/fonts")).fonts;
}
