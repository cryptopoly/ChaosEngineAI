/**
 * Storyboard API client (``/api/storyboard/*``).
 *
 * Every call that changes a project sends the ``revision`` the caller last saw. A stale one answers
 * 409 with the current project; that is raised as a `StoryboardRevisionConflict` that carries the
 * fresh project so the caller can adopt it and tell the user. Other failures raise `StoryboardApiError`.
 * Project media needs the bearer token, so files are fetched as blob URLs (`fetchStoryboardFileBlobUrl`).
 */

import { apiFetch, readErrorDetail } from "./index";
import type {
  StoryboardBatchStatus,
  StoryboardBatchWhat,
  StoryboardListRow,
  StoryboardOpArgs,
  StoryboardOpName,
  StoryboardProject,
  StoryboardPeaks,
  StoryboardProjectPayload,
  StoryboardRenderJob,
} from "../types";

export class StoryboardApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "StoryboardApiError";
    this.status = status;
  }
}

/** The project changed since the caller loaded it; `project` is the current one. */
export class StoryboardRevisionConflict extends StoryboardApiError {
  readonly project: StoryboardProject;
  readonly revision: number;
  constructor(message: string, project: StoryboardProject) {
    super(message, 409);
    this.name = "StoryboardRevisionConflict";
    this.project = project;
    this.revision = project.revision;
  }
}

/** Reads the 409 body of a stale revision; null when it is some other 409 (a busy render, for example). */
export function parseRevisionConflict(status: number, body: unknown): StoryboardRevisionConflict | null {
  if (status !== 409 || !body || typeof body !== "object") return null;
  const detail = (body as { detail?: unknown }).detail;
  if (!detail || typeof detail !== "object") return null;
  const { message, project } = detail as { message?: unknown; project?: unknown };
  if (!project || typeof project !== "object" || typeof (project as StoryboardProject).revision !== "number") return null;
  return new StoryboardRevisionConflict(typeof message === "string" ? message : "Revision conflict", project as StoryboardProject);
}

async function request<T>(
  method: "GET" | "POST" | "DELETE",
  path: string,
  body?: object,
  timeoutMs = 60000,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await apiFetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    return await readResponse<T>(response);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new StoryboardApiError(`Request to ${path} timed out after ${Math.round(timeoutMs / 1000)}s`, 0);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function readResponse<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;
  let parsed: unknown = null;
  try {
    parsed = await response.clone().json();
  } catch {
    parsed = null;
  }
  const conflict = parseRevisionConflict(response.status, parsed);
  if (conflict) throw conflict;
  throw new StoryboardApiError(
    await readErrorDetail(response, `Request failed with status ${response.status}`),
    response.status,
  );
}

const base = (projectId: string) => `/api/storyboard/projects/${encodeURIComponent(projectId)}`;

// ---- projects ----
export async function listStoryboardProjects(): Promise<StoryboardListRow[]> {
  return (await request<{ projects: StoryboardListRow[] }>("GET", "/api/storyboard/projects")).projects;
}

export async function createStoryboardProject(
  name: string,
  defaults?: Record<string, unknown>,
): Promise<StoryboardProjectPayload> {
  return await request("POST", "/api/storyboard/projects", defaults ? { name, defaults } : { name });
}

export async function getStoryboardProject(projectId: string): Promise<StoryboardProjectPayload> {
  return await request("GET", base(projectId));
}

export async function deleteStoryboardProject(projectId: string, revision: number): Promise<{ deleted: string }> {
  return await request("DELETE", `${base(projectId)}?revision=${revision}`);
}

// ---- ops ----
export async function runStoryboardOp<K extends StoryboardOpName>(
  projectId: string,
  revision: number,
  op: K,
  args: StoryboardOpArgs[K],
): Promise<StoryboardProjectPayload> {
  return await request("POST", `${base(projectId)}/ops`, { revision, op, args }, 120000);
}

type OpWrapper<K extends StoryboardOpName> = (
  projectId: string,
  revision: number,
  args: StoryboardOpArgs[K],
) => Promise<StoryboardProjectPayload>;

function wrap<K extends StoryboardOpName>(op: K): OpWrapper<K> {
  return (projectId, revision, args) => runStoryboardOp(projectId, revision, op, args);
}

/** One typed wrapper per op name. */
export const storyboardOps = {
  editProject: wrap("edit_project"),
  addReference: wrap("add_reference"),
  editReference: wrap("edit_reference"),
  removeReference: wrap("remove_reference"),
  addShot: wrap("add_shot"),
  editShot: wrap("edit_shot"),
  removeShot: wrap("remove_shot"),
  reorder: wrap("reorder"),
  lock: wrap("lock"),
  pickTake: wrap("pick_take"),
  markRedo: wrap("mark_redo"),
  importTake: wrap("import_take"),
  relink: wrap("relink"),
  addAudio: wrap("add_audio"),
  editAudio: wrap("edit_audio"),
  removeAudio: wrap("remove_audio"),
  setCaptions: wrap("set_captions"),
  addCaption: wrap("add_caption"),
  editCaption: wrap("edit_caption"),
  removeCaption: wrap("remove_caption"),
};

export async function undoStoryboard(projectId: string, revision: number): Promise<StoryboardProjectPayload> {
  return await request("POST", `${base(projectId)}/undo`, { revision });
}

export async function redoStoryboard(projectId: string, revision: number): Promise<StoryboardProjectPayload> {
  return await request("POST", `${base(projectId)}/redo`, { revision });
}

// ---- batches ----
export async function startStoryboardBatch(
  projectId: string,
  options: { revision: number; what?: StoryboardBatchWhat; shotIds?: string[]; force?: boolean },
): Promise<StoryboardBatchStatus & { batchId: string }> {
  const { revision, what = "both", shotIds, force = false } = options;
  return await request("POST", `${base(projectId)}/batches`, { revision, what, shotIds: shotIds ?? null, force });
}

export async function getStoryboardBatch(projectId: string, batchId: string): Promise<StoryboardBatchStatus> {
  return await request("GET", `${base(projectId)}/batches/${encodeURIComponent(batchId)}`, undefined, 15000);
}

export async function cancelStoryboardBatch(projectId: string, batchId: string): Promise<{ cancelled: boolean }> {
  return await request("POST", `${base(projectId)}/batches/${encodeURIComponent(batchId)}/cancel`);
}

// ---- renders (used by the Mixer) ----
export async function startStoryboardRender(
  projectId: string,
  revision: number,
  preview = true,
): Promise<{ render: StoryboardRenderJob }> {
  return await request("POST", `${base(projectId)}/render`, { revision, preview, notify: !preview });
}

/** Transcribe the voice clips (or the named ones) into captions. Slow: it runs the speech model. */
export async function generateStoryboardCaptions(
  projectId: string,
  revision: number,
  options: { clipIds?: string[]; language?: string } = {},
): Promise<StoryboardProjectPayload & { notes: string[] }> {
  return await request("POST", `${base(projectId)}/captions/generate`, { revision, ...options }, 1800000);
}

/** Write the captions as an .srt in the project's exports folder. */
export async function exportStoryboardCaptions(projectId: string): Promise<{ file: string; cues: number }> {
  return await request("POST", `${base(projectId)}/captions/export`);
}

/** Write the cut as an OpenTimelineIO file for an editor. */
export async function exportStoryboardOtio(projectId: string): Promise<{ file: string; warnings: string[] }> {
  return await request("POST", `${base(projectId)}/otio`);
}

export async function getStoryboardRender(projectId: string): Promise<{ render: StoryboardRenderJob | null }> {
  return await request("GET", `${base(projectId)}/render`, undefined, 15000);
}

export async function cancelStoryboardRender(projectId: string): Promise<{ cancelled: boolean }> {
  return await request("POST", `${base(projectId)}/render/cancel`);
}

// ---- bundles ----
export async function exportStoryboardBundle(projectId: string): Promise<{ file: string; bytes: number }> {
  return await request("POST", `${base(projectId)}/bundle`, undefined, 600000);
}

/** Import a `.chaosfilm`: the raw bytes of a picked file, or `{ path }` for a file on this machine. */
export async function importStoryboardBundle(source: Blob | { path: string }): Promise<StoryboardProjectPayload> {
  if ("path" in source) return await request("POST", "/api/storyboard/import", { path: source.path }, 600000);
  const response = await apiFetch("/api/storyboard/import", {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: source,
  });
  return await readResponse<StoryboardProjectPayload>(response);
}

// ---- files ----
/** The API path of a file inside a project (``media/...`` or ``exports/...``). */
export function storyboardFilePath(projectId: string, rel: string): string {
  return `${base(projectId)}/files?path=${encodeURIComponent(rel)}`;
}

/**
 * Fetch a project file with the API token and wrap it in an object URL. The caller revokes it
 * (`URL.revokeObjectURL`) when it no longer needs it.
 */
export async function fetchStoryboardFileBlobUrl(projectId: string, rel: string): Promise<string> {
  const response = await apiFetch(storyboardFilePath(projectId, rel), { method: "GET" });
  if (!response.ok) {
    throw new StoryboardApiError(`Failed to load ${rel} (${response.status})`, response.status);
  }
  return URL.createObjectURL(await response.blob());
}

/** Peak buckets are bounded by the server (32 to 4000). */
export const PEAKS_MIN_BUCKETS = 32;
export const PEAKS_MAX_BUCKETS = 4000;

/** Waveform peaks of a project file (an audio clip, or a shot's clip for its own sound). */
export async function getStoryboardPeaks(projectId: string, rel: string, buckets = 400): Promise<StoryboardPeaks> {
  const count = Math.max(PEAKS_MIN_BUCKETS, Math.min(PEAKS_MAX_BUCKETS, Math.round(buckets)));
  return await request("GET", `${base(projectId)}/peaks?path=${encodeURIComponent(rel)}&buckets=${count}`, undefined, 120000);
}

/**
 * The size in bytes of a project file. A one-byte range request answers it when the response header is
 * readable (it is hidden from cross-origin pages unless the server exposes it); otherwise the file is read.
 */
export async function fetchStoryboardFileSize(projectId: string, rel: string): Promise<number> {
  const path = storyboardFilePath(projectId, rel);
  const probe = await apiFetch(path, { method: "GET", headers: { Range: "bytes=0-0" } });
  if (!probe.ok) throw new StoryboardApiError(`Failed to read ${rel} (${probe.status})`, probe.status);
  const total = /\/(\d+)$/.exec(probe.headers.get("Content-Range") ?? "");
  if (total) return Number(total[1]);
  if (probe.status !== 206) return (await probe.blob()).size;
  const full = await apiFetch(path, { method: "GET" });
  if (!full.ok) throw new StoryboardApiError(`Failed to read ${rel} (${full.status})`, full.status);
  return (await full.blob()).size;
}
