/**
 * Video runtime install actions.
 *
 * Three async install flows pulled out of ``useVideoState``:
 *
 * * ``handleInstallVideoGpuRuntime`` — kicks off the backend GPU bundle
 *   install daemon and polls its status until ``done`` or 30 min timeout.
 * * ``handleInstallVideoOutputDeps`` — installs imageio + imageio-ffmpeg
 *   (or any caller-supplied package list) via the synchronous
 *   ``/api/setup/install-package`` endpoint.
 * * ``handleInstallLongLive`` — async LongLive install with the same
 *   poll-loop shape as the GPU bundle path.
 * * ``handleInstallMlxVideo`` — synchronous mlx-video pip install.
 *
 * Plus the two ``refresh*Status`` companions that ``useVideoState``
 * uses to refresh the chip-state probes after a config change.
 *
 * Extracted as part of the v0.8.0 Phase 2c-6 refactor.
 */

import {
  getGpuBundleStatus,
  getLongLiveInstallStatus,
  getLongLiveRuntime,
  getMlxVideoRuntime,
  getVideoRuntime,
  installPipPackage,
  startGpuBundleInstall,
  startLongLiveInstall,
} from "../../api";
import type {
  GpuBundleJobState,
  InstallResult,
  LongLiveJobState,
} from "../../api";
import { formatGpuBundleLabel, formatLongLiveLabel } from "../../hooks/installLabels";
import type { VideoRuntimeStatus } from "../../types";


type LongLiveStatus = Awaited<ReturnType<typeof getLongLiveRuntime>>;
type MlxVideoStatus = Awaited<ReturnType<typeof getMlxVideoRuntime>>;


export async function handleInstallVideoGpuRuntime(
  deps: {
    setVideoBusyLabel: (label: string | null) => void;
    setGpuBundleJob: (job: GpuBundleJobState) => void;
    setVideoRuntimeStatus: (status: VideoRuntimeStatus) => void;
    setError: (msg: string | null) => void;
  },
): Promise<InstallResult> {
  const { setVideoBusyLabel, setGpuBundleJob, setVideoRuntimeStatus, setError } = deps;
  setVideoBusyLabel("Starting GPU bundle install...");
  try {
    let job: GpuBundleJobState;
    try {
      job = await startGpuBundleInstall();
      setGpuBundleJob(job);
    } catch (err) {
      const message = `Failed to start GPU bundle install: ${err instanceof Error ? err.message : String(err)}`;
      setError(message);
      return { ok: false, output: message, capabilities: {} };
    }

    const POLL_MS = 1500;
    const MAX_WAIT_MS = 30 * 60_000;
    const deadline = Date.now() + MAX_WAIT_MS;
    while (!job.done && Date.now() < deadline) {
      setVideoBusyLabel(formatGpuBundleLabel(job));
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));
      try {
        job = await getGpuBundleStatus();
        setGpuBundleJob(job);
      } catch (err) {
        setVideoBusyLabel(
          `Install in progress (status fetch hiccup: ${err instanceof Error ? err.message : "unknown"})`,
        );
      }
    }

    try {
      const runtime = await getVideoRuntime();
      setVideoRuntimeStatus(runtime);
    } catch {
      // Stale status is fine — restart will refresh.
    }

    if (job.phase === "error" || job.error) {
      const rawMessage = job.error || job.message || "GPU bundle install failed.";
      const hint = job.targetDir
        ? ` See the install log below for per-step pip output. Target: ${job.targetDir}`
        : " See the install log below for per-step pip output.";
      const message = rawMessage + hint;
      setError(message);
      return { ok: false, output: message, capabilities: {} };
    }
    if (!job.done) {
      const message = "GPU bundle install did not finish within 30 minutes. See the install log below.";
      setError(message);
      return { ok: false, output: message, capabilities: {} };
    }

    setError(null);
    const output = job.requiresRestart
      ? `${job.message}\n\nRestart the backend to activate GPU acceleration.`
      : job.message;
    return { ok: true, output, capabilities: {} };
  } finally {
    setVideoBusyLabel(null);
  }
}


export async function handleInstallVideoOutputDeps(
  packages: readonly string[] | undefined,
  deps: {
    setVideoBusyLabel: (label: string | null) => void;
    setVideoRuntimeStatus: (status: VideoRuntimeStatus) => void;
    setError: (msg: string | null) => void;
  },
): Promise<InstallResult> {
  const { setVideoBusyLabel, setVideoRuntimeStatus, setError } = deps;
  const targetPackages =
    packages && packages.length > 0 ? Array.from(packages) : ["imageio", "imageio-ffmpeg"];
  const isMp4Only =
    targetPackages.length === 2
    && targetPackages.includes("imageio")
    && targetPackages.includes("imageio-ffmpeg");
  const friendlyLabel = isMp4Only
    ? "Installing mp4 encoder (imageio + imageio-ffmpeg)..."
    : `Installing video runtime packages (${targetPackages.join(", ")})...`;
  setVideoBusyLabel(friendlyLabel);
  const failures: string[] = [];
  let lastOutput = "";
  try {
    for (const pkg of targetPackages) {
      try {
        const result = await installPipPackage(pkg);
        lastOutput = result.output;
        if (!result.ok) {
          failures.push(`${pkg}: ${result.output.slice(0, 200)}`);
        }
      } catch (err) {
        failures.push(`${pkg}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    // Re-probe regardless — even a partial install can flip one flag.
    try {
      const runtime = await getVideoRuntime();
      setVideoRuntimeStatus(runtime);
    } catch {
      // keep the pre-install status if the probe itself fails
    }
    if (failures.length > 0) {
      const failureSummary = isMp4Only
        ? "mp4 encoder install failed"
        : "Video runtime package install failed";
      const message = `${failureSummary}:\n${failures.join("\n")}`;
      setError(message);
      return { ok: false, output: message, capabilities: {} };
    }
    setError(null);
    return { ok: true, output: lastOutput, capabilities: {} };
  } finally {
    setVideoBusyLabel(null);
  }
}


export async function refreshLongLiveStatus(
  setLongLiveStatus: (status: LongLiveStatus) => void,
): Promise<void> {
  try {
    const status = await getLongLiveRuntime();
    setLongLiveStatus(status);
  } catch {
    // Ignore — LongLive probe failures just mean we show a retry prompt.
  }
}


// Async install — kicks off a backend daemon thread, polls the status
// endpoint at ~1.5 Hz, surfaces phase progress through ``longLiveJob``
// (rendered by InstallLogPanel) and busy label through ``videoBusyLabel``.
// Replaces the synchronous ``installSystemPackage("longlive")`` path
// because the install routinely takes 10-20 minutes (~30 pip packages,
// optional flash-attn build, ~8 GB of HF weights) — well past the 600s
// timeout that route enforces.
export async function handleInstallLongLive(
  deps: {
    setInstallingLongLive: (busy: boolean) => void;
    setVideoBusyLabel: (label: string | null) => void;
    setLongLiveJob: (job: LongLiveJobState) => void;
    setLongLiveStatus: (status: LongLiveStatus) => void;
    setError: (msg: string | null) => void;
  },
): Promise<InstallResult> {
  const {
    setInstallingLongLive,
    setVideoBusyLabel,
    setLongLiveJob,
    setLongLiveStatus,
    setError,
  } = deps;
  setInstallingLongLive(true);
  setError(null);
  setVideoBusyLabel("Starting LongLive install...");
  try {
    let job: LongLiveJobState;
    try {
      job = await startLongLiveInstall();
      setLongLiveJob(job);
    } catch (err) {
      const message = `Failed to start LongLive install: ${err instanceof Error ? err.message : String(err)}`;
      setError(message);
      return { ok: false, output: message, capabilities: {} };
    }

    const POLL_MS = 1500;
    // Mirrors GPU bundle ceiling. The LongLive install is bounded by
    // HF download speed for ~8 GB; 30 minutes covers a 5 MB/s connection
    // with comfortable headroom for the pip+flash-attn build phase.
    const MAX_WAIT_MS = 30 * 60_000;
    const deadline = Date.now() + MAX_WAIT_MS;
    while (!job.done && Date.now() < deadline) {
      setVideoBusyLabel(formatLongLiveLabel(job));
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));
      try {
        job = await getLongLiveInstallStatus();
        setLongLiveJob(job);
      } catch (err) {
        setVideoBusyLabel(
          `LongLive install in progress (status fetch hiccup: ${err instanceof Error ? err.message : "unknown"})`,
        );
      }
    }

    // Refresh the runtime probe regardless of outcome — even a partial
    // install changes the install marker / repo state, and we want the
    // chip to reflect reality next time the user mounts the Studio.
    await refreshLongLiveStatus(setLongLiveStatus);

    if (job.phase === "error" || job.error) {
      const rawMessage = job.error || job.message || "LongLive install failed.";
      const hint = " See the install log below for per-phase output.";
      const message = rawMessage + hint;
      setError(message);
      return { ok: false, output: message, capabilities: {} };
    }
    if (!job.done) {
      const message = "LongLive install did not finish within 30 minutes. See the install log below.";
      setError(message);
      return { ok: false, output: message, capabilities: {} };
    }

    return { ok: true, output: job.message, capabilities: {} };
  } catch (err) {
    const message = err instanceof Error ? err.message : "LongLive install failed.";
    setError(message);
    return { ok: false, output: message, capabilities: {} };
  } finally {
    setInstallingLongLive(false);
    setVideoBusyLabel(null);
  }
}


export async function refreshMlxVideoStatus(
  setMlxVideoStatus: (status: MlxVideoStatus) => void,
): Promise<void> {
  try {
    const status = await getMlxVideoRuntime();
    setMlxVideoStatus(status);
  } catch {
    // Ignore — same rationale as LongLive: probe failure shows retry prompt.
  }
}


export async function handleInstallMlxVideo(
  deps: {
    setInstallingMlxVideo: (busy: boolean) => void;
    setMlxVideoStatus: (status: MlxVideoStatus) => void;
    setError: (msg: string | null) => void;
  },
): Promise<InstallResult> {
  const { setInstallingMlxVideo, setMlxVideoStatus, setError } = deps;
  setInstallingMlxVideo(true);
  setError(null);
  try {
    const result = await installPipPackage("mlx-video");
    if (!result.ok) {
      setError(`mlx-video install failed: ${result.output.slice(0, 300)}`);
    }
    await refreshMlxVideoStatus(setMlxVideoStatus);
    return result;
  } catch (err) {
    const message = err instanceof Error ? err.message : "mlx-video install failed.";
    setError(message);
    return { ok: false, output: message, capabilities: {} };
  } finally {
    setInstallingMlxVideo(false);
  }
}
