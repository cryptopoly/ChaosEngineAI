import type { GpuBundleJobState, LongLiveJobState } from "../api";


/**
 * Surface-ready label for the current GPU bundle install phase. Both
 * Studios poll the same job and render the same banner. The
 * ImageStudioTab / VideoStudioTab shows what's actually happening right
 * now (downloading torch vs. resolving deps vs. verifying CUDA), not a
 * generic spinner with no context.
 */
export function formatGpuBundleLabel(job: GpuBundleJobState): string {
  const phase = job.phase;
  if (phase === "preflight") return job.message || "Preparing GPU bundle install...";
  if (phase === "downloading") {
    const total = job.packageTotal || 1;
    const pct = Math.max(0, Math.min(100, Math.round(job.percent)));
    const current = job.packageCurrent || job.message || "package";
    return `Installing GPU bundle: ${current} (${job.packageIndex}/${total}, ${pct}%)`;
  }
  if (phase === "verifying") return "Verifying CUDA availability...";
  if (phase === "done") return job.message || "GPU bundle installed.";
  if (phase === "error") return job.error || job.message || "GPU bundle install failed.";
  return job.message || "Working...";
}


/**
 * Same shape as ``formatGpuBundleLabel`` but worded for the LongLive
 * install. The phases there (clone repo / build venv / pip / weights /
 * marker) need different copy than the GPU bundle's CUDA-walk vocab.
 */
export function formatLongLiveLabel(job: LongLiveJobState): string {
  const phase = job.phase;
  if (phase === "preflight") return job.message || "Preparing LongLive install...";
  if (phase === "downloading") {
    const total = job.packageTotal || 1;
    const pct = Math.max(0, Math.min(100, Math.round(job.percent)));
    const current = job.packageCurrent || job.message || "step";
    return `Installing LongLive: ${current} (${job.packageIndex}/${total}, ${pct}%)`;
  }
  if (phase === "done") return job.message || "LongLive installed.";
  if (phase === "error") return job.error || job.message || "LongLive install failed.";
  return job.message || "Working...";
}
