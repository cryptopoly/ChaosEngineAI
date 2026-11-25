/**
 * The video the Upscale tab opens with when another page sends one (the Video Gallery's "Upscale this"). The
 * sender calls ``requestUpscale`` and then moves the app to the "video-upscale" tab; the tab reads the request
 * when it mounts (``peekUpscaleRequest``) and clears it once taken (``clearUpscaleRequest``).
 */

export interface UpscaleSource {
  /** Absolute path on this machine (the backend reads the file). */
  path: string;
  /** A Video Gallery clip: its preview can be fetched (``fetchVideoOutputBlobUrl``). */
  artifactId?: string;
  /** The source size when known; the target sizes (1080p and up) need it. */
  width?: number;
  height?: number;
}

let pending: UpscaleSource | null = null;

export function requestUpscale(source: UpscaleSource): void {
  pending = { ...source };
}

export function peekUpscaleRequest(): UpscaleSource | null {
  return pending;
}

export function clearUpscaleRequest(): void {
  pending = null;
}

const LAST_JOB_KEY = "chaosengine.videoUpscale.lastJob";

/** The last job's id and source, so a restart can show it again and offer Resume. */
export interface RememberedJob {
  id: string;
  source: UpscaleSource;
}

export function readLastJob(): RememberedJob | null {
  try {
    const raw = window.localStorage.getItem(LAST_JOB_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<RememberedJob>;
    return typeof value?.id === "string" && typeof value.source?.path === "string" ? (value as RememberedJob) : null;
  } catch {
    return null;
  }
}

export function writeLastJob(job: RememberedJob | null): void {
  try {
    if (job) window.localStorage.setItem(LAST_JOB_KEY, JSON.stringify(job));
    else window.localStorage.removeItem(LAST_JOB_KEY);
  } catch {
    // not remembered: the card still shows until the page closes
  }
}
