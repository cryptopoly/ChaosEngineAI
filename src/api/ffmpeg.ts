/**
 * Video tools (ffmpeg) endpoints: status for the Setup row and the first-use prompt, the pinned LGPL
 * download (Windows and Linux), choosing a file, and removing the download. The main install runs through
 * the existing pip route (`installPipPackage("imageio-ffmpeg")`) on every system.
 */

import { deleteJson, fetchJson, postJson } from "./index";

export interface FfmpegJob {
  phase: string;
  percent: number;
  error: string | null;
  done: boolean;
  running: boolean;
  startedAt: number;
}

export interface FfmpegPin {
  url: string;
  sha256: string;
  size: number;
  version: string;
  licenceUrl: string;
  sourceUrl: string;
}

export interface FfmpegStatus {
  installed: boolean;
  path: string | null;
  /** env | custom | download | imageio | path */
  source: string | null;
  version: string | null;
  /** "LGPL", "GPL" or null when unknown. */
  licence: string | null;
  /** The H.264 encoder Storyboard will use (libx264, h264_videotoolbox, ...). */
  encoder: string | null;
  pin: FfmpegPin | null;
  canDownload: boolean;
  job: FfmpegJob;
  error?: string;
}

export async function getFfmpegStatus(): Promise<FfmpegStatus> {
  return await fetchJson<FfmpegStatus>("/api/setup/ffmpeg");
}

export async function startFfmpegDownload(): Promise<{ job: FfmpegJob }> {
  return await postJson("/api/setup/ffmpeg/install", { source: "download" }, 30000);
}

export async function getFfmpegJob(): Promise<{ job: FfmpegJob }> {
  return await fetchJson("/api/setup/ffmpeg/install/status");
}

export async function cancelFfmpegDownload(): Promise<{ job: FfmpegJob }> {
  return await postJson("/api/setup/ffmpeg/install/cancel", {}, 30000);
}

/** An ffmpeg program, or the pinned archive fetched elsewhere (the backend checks its SHA-256). */
export async function chooseFfmpegFile(path: string): Promise<{ job: FfmpegJob; version: string | null }> {
  return await postJson("/api/setup/ffmpeg/custom", { path }, 120000);
}

export async function removeFfmpegDownload(): Promise<FfmpegStatus> {
  return await deleteJson<FfmpegStatus>("/api/setup/ffmpeg", undefined, 60000);
}
