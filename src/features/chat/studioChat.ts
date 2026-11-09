/**
 * D-60: studio tools in Chat. The remembered switch, the per-chat "Always allow renders" box, the plan
 * card's decision bodies and the small helpers the job cards use. Pure functions so vitest covers them.
 */
import { useEffect, useState } from "react";
import { decideStudioPlan } from "../../api";
import type { StudioJob, StudioPlan, TabId, ToolCallInfo } from "../../types";

const SWITCH_KEY = "chaosengine.chat.studioTools";
const AUTO_KEY_PREFIX = "chaosengine.chat.studioAutoApprove.";
/** Asks the app shell to switch tab (the job card's "Open in Library"). */
export const OPEN_TAB_EVENT = "chaosengine:open-tab";

function read(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    // storage blocked: the switch still works for this session
  }
}

export function readStudioToolsEnabled(): boolean {
  return read(SWITCH_KEY) === "1";
}

export function writeStudioToolsEnabled(enabled: boolean): void {
  write(SWITCH_KEY, enabled ? "1" : null);
  if (!enabled) clearAllStudioAutoApprove();
}

/**
 * "Always allow renders in this chat" lasts AUTO_APPROVE_HOURS at most and never survives an app restart:
 * it lives in sessionStorage (a new window starts empty) with the time it was turned on.
 */
export const AUTO_APPROVE_HOURS = 2;
/** Fired when the setting changes, so the composer chip follows a click on the plan card. */
export const AUTO_APPROVE_EVENT = "chaosengine:studio-auto-approve";

function session(): Storage | null {
  try {
    return window.sessionStorage ?? null;
  } catch {
    return null;
  }
}

function notifyAutoApprove(): void {
  try {
    window.dispatchEvent(new Event(AUTO_APPROVE_EVENT));
  } catch {
    // no window (tests): nothing listens
  }
}

export function readStudioAutoApprove(sessionId: string | null | undefined, now: number = Date.now()): boolean {
  if (!sessionId) return false;
  const store = session();
  let raw: string | null = null;
  try {
    raw = store?.getItem(AUTO_KEY_PREFIX + sessionId) ?? null;
  } catch {
    return false;
  }
  const since = Number(raw);
  if (!raw || !Number.isFinite(since)) return false;
  if (now - since > AUTO_APPROVE_HOURS * 3600_000) {
    writeStudioAutoApprove(sessionId, false);
    return false;
  }
  return true;
}

export function writeStudioAutoApprove(sessionId: string | null | undefined, allow: boolean, now: number = Date.now()): void {
  if (!sessionId) return;
  try {
    const store = session();
    if (allow) store?.setItem(AUTO_KEY_PREFIX + sessionId, String(now));
    else store?.removeItem(AUTO_KEY_PREFIX + sessionId);
  } catch {
    // storage blocked: the card simply keeps asking
  }
  notifyAutoApprove();
}

export function clearAllStudioAutoApprove(): void {
  const store = session();
  if (!store) return;
  try {
    const keys: string[] = [];
    for (let i = 0; i < store.length; i++) {
      const key = store.key(i);
      if (key?.startsWith(AUTO_KEY_PREFIX)) keys.push(key);
    }
    keys.forEach((key) => store.removeItem(key));
  } catch {
    // nothing stored
  }
  notifyAutoApprove();
}

/** Whether renders are auto-approved in this chat, following changes from the plan card and the chip. */
export function useStudioAutoApprove(sessionId: string | null | undefined): boolean {
  const [on, setOn] = useState(() => readStudioAutoApprove(sessionId));
  useEffect(() => {
    const update = () => setOn(readStudioAutoApprove(sessionId));
    update();
    window.addEventListener(AUTO_APPROVE_EVENT, update);
    const timer = window.setInterval(update, 60_000);  // the expiry shows without a click
    return () => {
      window.removeEventListener(AUTO_APPROVE_EVENT, update);
      window.clearInterval(timer);
    };
  }, [sessionId]);
  return on;
}

export type PlanAction = { type: "approveAll" } | { type: "approveOne"; callId: string } | { type: "deny" } | { type: "edit" };

/** The decision body for a plan card button. "Approve this" runs that call only and declines the rest. */
export function planDecision(plan: StudioPlan, action: PlanAction, alwaysAllow: boolean) {
  const decisions: Record<string, "approve" | "deny"> = {};
  for (const item of plan.items) {
    const approve = action.type === "approveAll" || (action.type === "approveOne" && action.callId === item.callId);
    decisions[item.callId] = approve ? "approve" : "deny";
  }
  const anyApproved = Object.values(decisions).includes("approve");
  return { decisions, edit: action.type === "edit", alwaysAllow: alwaysAllow && anyApproved };
}

/**
 * Send a plan card click to the backend, and remember "Always allow renders in this chat" once it is
 * accepted. Throws the backend's error (for example a plan that timed out).
 */
export async function submitPlanDecision(
  plan: StudioPlan,
  action: PlanAction,
  alwaysAllow: boolean,
  sessionId: string | null | undefined,
): Promise<ReturnType<typeof planDecision>> {
  const body = planDecision(plan, action, alwaysAllow);
  await decideStudioPlan(plan.planId, body);
  if (body.alwaysAllow) writeStudioAutoApprove(sessionId, true);
  return body;
}

/** Insert or replace a plan by id (the resolved event replaces the pending one). */
export function upsertPlan(plans: StudioPlan[] | undefined, plan: StudioPlan): StudioPlan[] {
  const list = plans ?? [];
  const index = list.findIndex((p) => p.planId === plan.planId);
  if (index < 0) return [...list, plan];
  const next = [...list];
  next[index] = plan;
  return next;
}

/**
 * Insert or replace a tool call by id (the running card, then its result, then the done payload). The result
 * keeps the start time the running card had when the result itself carries none.
 */
export function upsertToolCall(calls: ToolCallInfo[] | undefined, call: ToolCallInfo): ToolCallInfo[] {
  const list = calls ?? [];
  const index = list.findIndex((c) => c.id === call.id);
  if (index < 0) return [...list, call];
  const next = [...list];
  const startedAt = call.startedAt ?? list[index].startedAt;
  next[index] = startedAt == null ? call : { ...call, startedAt };
  return next;
}

/** A progress snapshot of a long wait lands on its running card (state, progress with elapsed and time left). */
export function applyToolProgress(
  calls: ToolCallInfo[] | undefined,
  progress: { id: string; data: Record<string, unknown> },
): ToolCallInfo[] {
  const list = calls ?? [];
  const index = list.findIndex((c) => c.id === progress.id);
  if (index < 0 || !list[index].running) return list;
  const next = [...list];
  next[index] = { ...list[index], renderAs: "studio", data: { ...(list[index].data ?? {}), ...progress.data } };
  return next;
}

/** A tool card, possibly standing for several polls of one job. */
export type DisplayToolCall = ToolCallInfo & { polls?: number };

/**
 * One card per job for repeated wait_for_job calls: the models poll the same job again and again, and the
 * thread showed N near-identical lines. The first call's position and start time stay, the newest answer
 * wins, and the elapsed times add up. Other tools pass through unchanged.
 */
export function mergeToolCalls(calls: ToolCallInfo[] | undefined): DisplayToolCall[] {
  const merged: DisplayToolCall[] = [];
  const byJob = new Map<string, number>();
  for (const call of calls ?? []) {
    const jobId = call.name === "wait_for_job" ? call.arguments?.job_id : null;
    if (typeof jobId !== "string" || !jobId) {
      merged.push(call);
      continue;
    }
    const at = byJob.get(jobId);
    if (at === undefined) {
      byJob.set(jobId, merged.length);
      merged.push({ ...call, polls: 1 });
      continue;
    }
    const first = merged[at];
    merged[at] = {
      ...call,
      id: first.id,
      startedAt: first.startedAt ?? call.startedAt ?? null,
      elapsed: (first.elapsed || 0) + (call.elapsed || 0),
      polls: (first.polls ?? 1) + 1,
    };
  }
  return merged;
}

/** Studio tools that start a job; their answer gets a live job card (wait_for_job answers do not). */
export const JOB_STARTING_TOOLS = new Set([
  "generate_image",
  "generate_video",
  "speak",
  "make_music",
  "make_sound",
  "storyboard_generate",
  "storyboard_render",
  "storyboard_captions",
]);

/** The job id a job-starting studio tool answer carries, or null. */
export function studioJobId(toolCall: ToolCallInfo): string | null {
  const data = toolCall.data ?? {};
  if (!JOB_STARTING_TOOLS.has(toolCall.name) || data.declined || data.error) return null;
  const id = data.jobId;
  return typeof id === "string" && id ? id : null;
}

export function isFinished(job: Pick<StudioJob, "state">): boolean {
  return ["done", "failed", "cancelled", "interrupted"].includes(job.state);
}

export type MediaKind = "image" | "video" | "audio" | "none";

export function mediaKind(job: Pick<StudioJob, "kind" | "output">): MediaKind {
  if (!job.output || (!job.output.path && !Array.isArray(job.output.tracks))) return "none";
  if (job.kind === "image") return "image";
  if (job.kind === "video") return "video";
  if (job.kind === "voice" || job.kind === "music") return "audio";
  return "none";
}

/** Where "Open in Library" goes for a job. Storyboard jobs open the Projects entry. */
export function libraryTab(job: Pick<StudioJob, "kind" | "output">, toolName?: string): TabId | null {
  if (toolName?.startsWith("storyboard_")) return "storyboard";
  switch (job.kind) {
    case "image":
      return "image-gallery";
    case "video":
      return "video-gallery";
    case "voice":
      return "voice-gallery";
    case "music":
      return "music-library";
    default:
      return null;
  }
}

/** Percent done from the job's own progress, or null when it reports none. */
export function jobPercent(job: Pick<StudioJob, "progress">): number | null {
  const step = job.progress?.step;
  const total = job.progress?.totalSteps;
  if (typeof step !== "number" || typeof total !== "number" || total <= 0) return null;
  return Math.max(0, Math.min(100, Math.round((step / total) * 100)));
}

export function formatSeconds(seconds: number | null | undefined): string | null {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds <= 0) return null;
  if (seconds < 90) return `${Math.round(seconds)} s`;
  return `${Math.round(seconds / 60)} min`;
}

export function openAppTab(tab: TabId): void {
  window.dispatchEvent(new CustomEvent(OPEN_TAB_EVENT, { detail: tab }));
}
