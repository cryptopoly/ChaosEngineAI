/**
 * Video upscale, fast tier (FU-617): the add-on install (``routes/setup/video_upscale.py``) and the jobs
 * (``routes/video_upscale.py``). Field names follow those routes; the job id is a hash of the request, so the
 * same request again returns the same job.
 */

import { apiFetch, fetchJson, postJson, readErrorDetail } from "./index";
import type { ModelFit } from "../types/models";

export type VideoUpscaleInstallPhase = "idle" | "packages" | "weights" | "check" | "done" | "error";

export interface VideoUpscaleInstallJob {
  phase: VideoUpscaleInstallPhase;
  error: string | null;
  running: boolean;
  models: string[];
  done: boolean;
  startedAt: number;
  finishedAt: number;
}

export interface VideoUpscaleAddonModel {
  id: string;
  name: string;
  label: string;
  installed: boolean;
  downloadMb: number;
  pinned: boolean;
  sha256Seen: string | null;
}

export interface VideoUpscaleAddonStatus {
  /** The packages and the default model's weights. */
  installed: boolean;
  packagesInstalled: boolean;
  folder: string;
  diskMb: number;
  packages: string[];
  models: VideoUpscaleAddonModel[];
  job: VideoUpscaleInstallJob;
}

export interface VideoUpscaleModel {
  id: string;
  name: string;
  /** "Fast" or "Fast, stronger clean-up". */
  label: string;
  author: string;
  scale: number;
  default: boolean;
  licence: import("../components/LicenceChip").ModelLicence;
  licenceNote: string;
  sizeBytes: number;
  speedNote?: string;
  notes?: string;
  footprintBasis?: string;
  fit?: ModelFit;
  installed: boolean;
}

export interface VideoUpscaleModels {
  models: VideoUpscaleModel[];
  addonInstalled: boolean;
  /** False when the video tools cannot deinterlace on this machine (the tab disables the option). */
  deinterlaceAvailable: boolean;
}

/** ``POST /api/video/upscale/probe``: what the backend reads from a file before a start. */
export interface VideoUpscaleProbe {
  /** The resolved path. */
  inputPath: string;
  width: number;
  height: number;
  fps: number;
  durationSeconds: number;
  frames: number;
  hasAudio: boolean;
  /** True or false when the field order is known, null when it could not be told. */
  interlaced: boolean | null;
}

export type VideoUpscaleState = "queued" | "running" | "done" | "failed" | "cancelled" | "interrupted";

export interface VideoUpscaleJob {
  id: string;
  state: VideoUpscaleState;
  input: string;
  output: string;
  modelId: string;
  width: number;
  height: number;
  framesDone: number;
  framesTotal: number;
  fraction: number;
  /** Measured while running, else null. */
  framesPerSecond: number | null;
  segment: number;
  segments: number;
  resumedSegments: number;
  error: string | null;
  notes: string[];
  startedAt: number;
  finishedAt: number | null;
  resumable: boolean;
}

/** One of ``scale`` or ``width`` + ``height``; neither means the backend's 2x. */
export interface VideoUpscaleRequest {
  inputPath: string;
  modelId: string;
  scale?: number;
  width?: number;
  height?: number;
  outputPath?: string;
  deinterlace?: boolean;
}

export async function getVideoUpscaleAddonStatus(): Promise<VideoUpscaleAddonStatus> {
  return await fetchJson<VideoUpscaleAddonStatus>("/api/setup/video-upscale/status", 10000);
}

export async function startVideoUpscaleInstall(models?: string[]): Promise<VideoUpscaleAddonStatus> {
  return await postJson<VideoUpscaleAddonStatus>("/api/setup/install-video-upscale", models ? { models } : {}, 15000);
}

export async function getVideoUpscaleModels(): Promise<VideoUpscaleModels> {
  return await fetchJson<VideoUpscaleModels>("/api/video/upscale/models", 15000);
}

export async function startVideoUpscale(request: VideoUpscaleRequest): Promise<VideoUpscaleJob> {
  return await postJson<VideoUpscaleJob>("/api/video/upscale", request, 60000);
}

export async function getVideoUpscaleJob(jobId: string): Promise<VideoUpscaleJob> {
  return await fetchJson<VideoUpscaleJob>(`/api/video/upscale/jobs/${encodeURIComponent(jobId)}`, 10000);
}

export async function cancelVideoUpscaleJob(jobId: string): Promise<{ cancelled: boolean }> {
  return await postJson<{ cancelled: boolean }>(`/api/video/upscale/jobs/${encodeURIComponent(jobId)}/cancel`, {}, 15000);
}

export async function resumeVideoUpscaleJob(jobId: string): Promise<VideoUpscaleJob> {
  return await postJson<VideoUpscaleJob>(`/api/video/upscale/jobs/${encodeURIComponent(jobId)}/resume`, {}, 60000);
}

export async function probeVideoUpscaleInput(inputPath: string): Promise<VideoUpscaleProbe> {
  return await postJson<VideoUpscaleProbe>("/api/video/upscale/probe", { inputPath }, 60000);
}

/** The 20 newest jobs, newest first. */
export async function listVideoUpscaleJobs(): Promise<VideoUpscaleJob[]> {
  const body = await fetchJson<{ jobs: VideoUpscaleJob[] }>("/api/video/upscale/jobs", 10000);
  return body.jobs;
}

/**
 * A media route of a finished job as a blob URL (the API token travels in a header, so a media element cannot
 * point at the route itself). ``null`` when the backend answers 404: it was restarted and removed the job's
 * work folder, or the film is gone. The caller revokes the URL.
 */
async function fetchJobMediaUrl(path: string): Promise<string | null> {
  const response = await apiFetch(path, { method: "GET" });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(await readErrorDetail(response, `Request failed with status ${response.status}`));
  return URL.createObjectURL(await response.blob());
}

export type VideoUpscaleFrameSide = "input" | "output";

/** The first frame of a job's input or finished output as a PNG blob URL, or ``null`` on 404. */
export async function fetchVideoUpscaleFrameUrl(jobId: string, which: VideoUpscaleFrameSide): Promise<string | null> {
  return await fetchJobMediaUrl(`/api/video/upscale/jobs/${encodeURIComponent(jobId)}/frame?which=${which}`);
}

/** The finished film as a blob URL a ``<video>`` can play, or ``null`` on 404. */
export async function fetchVideoUpscaleFileUrl(jobId: string): Promise<string | null> {
  return await fetchJobMediaUrl(`/api/video/upscale/jobs/${encodeURIComponent(jobId)}/file`);
}
