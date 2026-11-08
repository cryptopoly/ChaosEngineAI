/** Pure helpers for the video tools (ffmpeg) prompt and Setup row. */
import type { FfmpegJob, FfmpegStatus } from "../../api/ffmpeg";

export const INSTALL_FFMPEG_CODE = "install-ffmpeg";

/** True for an API error that says video tools are missing (the backend answers 409 with this code). */
export function isMissingVideoTools(err: unknown): boolean {
  return typeof err === "object" && err !== null && (err as { code?: unknown }).code === INSTALL_FFMPEG_CODE;
}

export type PromptPhase = "hidden" | "ask" | "installing" | "failed";

/** What the first-use prompt shows. Hidden while status is unknown or the tools are present. */
export function promptPhase(status: FfmpegStatus | null, busy: boolean, error: string | null, forced = false): PromptPhase {
  if (busy) return "installing";
  if (error) return "failed";
  if (forced) return "ask";
  if (!status || status.installed) return "hidden";
  return "ask";
}

export function formatMb(bytes: number): string {
  return `${Math.round(bytes / 1_000_000)}`;
}

/** The source of the ffmpeg in use, as a translation key suffix and an English default. */
export function sourceLabel(source: string | null): { key: string; fallback: string } {
  switch (source) {
    case "download":
      return { key: "download", fallback: "downloaded LGPL build" };
    case "imageio":
      return { key: "imageio", fallback: "imageio-ffmpeg package" };
    case "custom":
      return { key: "custom", fallback: "a file you chose (not verified)" };
    case "env":
      return { key: "env", fallback: "CHAOSENGINE_FFMPEG" };
    case "path":
      return { key: "path", fallback: "system PATH" };
    default:
      return { key: "unknown", fallback: "unknown" };
  }
}

export function canRemove(status: FfmpegStatus | null): boolean {
  return status?.source === "download" || status?.source === "custom";
}

export function jobRunning(job: FfmpegJob | null | undefined): boolean {
  return !!job && job.running;
}

export function jobPercent(job: FfmpegJob | null | undefined): number {
  return job ? Math.max(0, Math.min(100, Math.round(job.percent))) : 0;
}

/** Keep the tail of a pip log short enough for one line of the prompt. */
export function lastLine(output: string): string {
  const lines = output.split("\n").map((line) => line.trim()).filter(Boolean);
  const last = lines[lines.length - 1] ?? "";
  return last.length > 200 ? `${last.slice(0, 197)}...` : last;
}
