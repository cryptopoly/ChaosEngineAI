/**
 * Video upscale, fast tier (FU-617): the add-on install (``routes/setup/video_upscale.py``) and the jobs
 * (``routes/video_upscale.py``). Field names follow those routes; the job id is a hash of the request, so the
 * same request again returns the same job.
 */

import { fetchJson, postJson } from "./index";
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
  licence: string;
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
