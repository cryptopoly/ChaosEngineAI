/**
 * Setup / install API endpoints.
 *
 * Covers everything in Settings → Setup that POSTs an install or
 * progress-poll: pip / system packages, CUDA torch install with index
 * walk, GPU bundle background job, LongLive (FU-003) + mlx-video Wan
 * (FU-025) async installs, llama-server-turbo update probe,
 * capability refresh, FU-022 prompt enhancer.
 *
 * Re-exported from ``./index`` so existing
 * ``import { startGpuBundleInstall } from "../api"`` paths keep
 * working.
 *
 * Extracted from ``api.ts`` as part of the v0.8.0 refactor.
 */

import { fetchJson, postJson } from "./index";

// ---------------------------------------------------------------------------
// Pip / system package install
// ---------------------------------------------------------------------------

export interface InstallResult {
  ok: boolean;
  output: string;
  capabilities: Record<string, unknown>;
}

export async function installPipPackage(packageName: string): Promise<InstallResult> {
  return await postJson<InstallResult>("/api/setup/install-package", { package: packageName }, 360000);
}

export async function installSystemPackage(packageName: string): Promise<InstallResult> {
  return await postJson<InstallResult>("/api/setup/install-system-package", { package: packageName }, 660000);
}

// ---------------------------------------------------------------------------
// CUDA torch install
// ---------------------------------------------------------------------------

export interface CudaTorchInstallAttempt {
  indexUrl: string;
  ok: boolean;
  output: string;
}

export interface CudaTorchInstallResult {
  ok: boolean;
  output: string;
  indexUrl: string | null;
  attempts: CudaTorchInstallAttempt[];
  requiresRestart: boolean;
  pythonExecutable: string;
  pythonVersion: string | null;
  noWheelForPython: boolean;
  targetDir?: string;
  capabilities: Record<string, unknown>;
}

export async function installCudaTorch(): Promise<CudaTorchInstallResult> {
  // 15 minute timeout — torch CUDA wheels are ~2.5 GB, and the endpoint
  // walks up to four CUDA indexes before giving up.
  return await postJson<CudaTorchInstallResult>("/api/setup/install-cuda-torch", {}, 900000);
}

// ---------------------------------------------------------------------------
// GPU bundle background install
// ---------------------------------------------------------------------------

export interface GpuBundlePackage {
  label: string;
  spec: string;
}

export interface GpuBundleInfo {
  targetDir: string | null;
  approxDownloadBytes: number;
  requiredFreeBytes: number;
  freeBytes: number | null;
  packages: GpuBundlePackage[];
}

export interface GpuBundleAttempt {
  indexUrl?: string;
  package?: string;
  phase?: string;
  ok: boolean;
  output: string;
}

export interface GpuBundleJobState {
  id: string;
  // Lifecycle: idle (no run yet) -> preflight -> downloading -> verifying -> done | error
  phase: "idle" | "preflight" | "downloading" | "verifying" | "done" | "error";
  message: string;
  packageCurrent: string | null;
  packageIndex: number;
  packageTotal: number;
  percent: number;
  targetDir: string | null;
  indexUrlUsed: string | null;
  pythonVersion: string | null;
  noWheelForPython: boolean;
  cudaVerified: boolean | null;
  requiresRestart: boolean;
  error: string | null;
  startedAt: number;
  finishedAt: number;
  attempts: GpuBundleAttempt[];
  done: boolean;
}

export async function fetchGpuBundleInfo(): Promise<GpuBundleInfo> {
  return await fetchJson<GpuBundleInfo>("/api/setup/gpu-bundle-info", 15000);
}

export async function startGpuBundleInstall(): Promise<GpuBundleJobState> {
  // Returns quickly — the install runs in a backend background thread.
  // Poll ``getGpuBundleStatus`` to follow progress.
  return await postJson<GpuBundleJobState>("/api/setup/install-gpu-bundle", {}, 15000);
}

export async function getGpuBundleStatus(): Promise<GpuBundleJobState> {
  return await fetchJson<GpuBundleJobState>("/api/setup/install-gpu-bundle/status", 10000);
}

// ---------------------------------------------------------------------------
// LongLive async install (FU-003)
// ---------------------------------------------------------------------------
//
// Same job pattern as the GPU bundle. The backend installer takes 10-20
// minutes (pip ~30 packages, optional flash-attn build, ~8 GB of HF
// weights) so we cannot run it through the synchronous
// ``/api/setup/install-system-package`` route.
//
// ``LongLiveJobState`` is shaped as a subset of ``GpuBundleJobState``
// so the same ``InstallLogPanel`` component can render either job.
// LongLive doesn't have a CUDA-index walk or wheel-availability check,
// so the LongLive-specific fields just default to neutral values.

export interface LongLiveAttempt {
  phase?: string;
  package?: string;
  // Always undefined for LongLive — the field exists in the type only
  // so the shared ``InstallLogPanel`` can read it on the discriminated
  // union without a per-job branch. Cheap to carry, expensive to fork
  // the panel just to drop one optional property.
  indexUrl?: string;
  ok: boolean;
  output: string;
}

export interface LongLiveJobState {
  id: string;
  phase: "idle" | "preflight" | "downloading" | "verifying" | "done" | "error";
  message: string;
  packageCurrent: string | null;
  packageIndex: number;
  packageTotal: number;
  percent: number;
  targetDir: string | null;
  error: string | null;
  startedAt: number;
  finishedAt: number;
  attempts: LongLiveAttempt[];
  done: boolean;
}

export async function startLongLiveInstall(): Promise<LongLiveJobState> {
  // Returns quickly — install runs in a backend daemon thread.
  // Poll ``getLongLiveInstallStatus`` to follow progress.
  return await postJson<LongLiveJobState>("/api/setup/install-longlive", {}, 15000);
}

export async function getLongLiveInstallStatus(): Promise<LongLiveJobState> {
  return await fetchJson<LongLiveJobState>("/api/setup/install-longlive/status", 10000);
}

// ---------------------------------------------------------------------------
// mlx-video Wan install (FU-025) — Apple Silicon only
// ---------------------------------------------------------------------------
//
// Same pattern as LongLive: kick off a background job (download raw HF
// weights → run mlx_video.models.wan_2.convert → verify), poll status,
// render attempts via InstallLogPanel.

export interface WanInstallAttempt {
  phase?: string;
  package?: string;
  /** Always undefined for Wan; carried for the shared InstallLogPanel union. */
  indexUrl?: string;
  ok: boolean;
  output: string;
}

export interface WanInstallJobState {
  id: string;
  phase: "idle" | "preflight" | "downloading" | "converting" | "verifying" | "done" | "error";
  message: string;
  repo: string | null;
  packageCurrent: string | null;
  packageIndex: number;
  packageTotal: number;
  percent: number;
  outputDir: string | null;
  error: string | null;
  startedAt: number;
  finishedAt: number;
  attempts: WanInstallAttempt[];
  done: boolean;
}

export interface WanConvertStatusFields {
  repo: string;
  converted: boolean;
  outputDir: string;
  hasTransformer: boolean;
  hasMoeExperts: boolean;
  hasVae: boolean;
  hasTextEncoder: boolean;
  note: string | null;
}

export interface WanInventoryItem {
  repo: string;
  approxRawSizeGb: number | null;
  converted: boolean;
  status: WanConvertStatusFields;
}

export interface WanInventory {
  items: WanInventoryItem[];
  convertRoot: string;
  rawRoot: string;
}

export async function startWanInstall(
  repo: string,
  options: {
    dtype?: "bfloat16" | "float16" | "float32";
    quantize?: boolean;
    bits?: 4 | 8;
    groupSize?: 32 | 64 | 128;
    cleanupRaw?: boolean;
  } = {},
): Promise<WanInstallJobState> {
  return await postJson<WanInstallJobState>(
    "/api/setup/install-mlx-video-wan",
    {
      repo,
      dtype: options.dtype ?? "bfloat16",
      quantize: options.quantize ?? false,
      bits: options.bits ?? 4,
      groupSize: options.groupSize ?? 64,
      cleanupRaw: options.cleanupRaw ?? false,
    },
    15000,
  );
}

export async function getWanInstallStatus(): Promise<WanInstallJobState> {
  return await fetchJson<WanInstallJobState>(
    "/api/setup/install-mlx-video-wan/status",
    10000,
  );
}

export async function getWanInventory(): Promise<WanInventory> {
  return await fetchJson<WanInventory>(
    "/api/setup/mlx-video-wan/inventory",
    10000,
  );
}

// ---------------------------------------------------------------------------
// MTPLX install (feature/mtplx) — isolated venv + forked mlx
// ---------------------------------------------------------------------------
//
// Same background-job shape as LongLiveJobState so the existing
// InstallLogPanel renders it without modification.

export interface MtplxAttempt {
  phase?: string;
  package?: string;
  indexUrl?: string;
  ok: boolean;
  output: string;
}

export interface MtplxJobState {
  id: string;
  phase: "idle" | "preflight" | "creating-venv" | "installing" | "verifying" | "done" | "error";
  message: string;
  packageCurrent: string | null;
  packageIndex: number;
  packageTotal: number;
  percent: number;
  targetDir: string | null;
  error: string | null;
  startedAt: number;
  finishedAt: number;
  attempts: MtplxAttempt[];
  done: boolean;
}

export interface MtplxStatus {
  installed: boolean;
  version: string | null;
  installedAt: string | null;
  venvPath: string | null;
}

export async function getMtplxStatus(): Promise<MtplxStatus> {
  return await fetchJson<MtplxStatus>("/api/setup/mtplx-status", 8000);
}

export async function startMtplxInstall(): Promise<MtplxJobState> {
  return await postJson<MtplxJobState>("/api/setup/install-mtplx", {}, 15000);
}

export async function getMtplxInstallStatus(): Promise<MtplxJobState> {
  return await fetchJson<MtplxJobState>("/api/setup/install-mtplx/status", 10000);
}

// ---------------------------------------------------------------------------
// llama-server-turbo update probe + capability refresh
// ---------------------------------------------------------------------------

export interface TurboUpdateInfo {
  installed: boolean;
  installedCommit: string | null;
  remoteCommit: string | null;
  updateAvailable: boolean;
  branch: string | null;
  buildDate: string | null;
}

export async function checkTurboUpdate(): Promise<TurboUpdateInfo> {
  return await fetchJson<TurboUpdateInfo>("/api/setup/turbo-update-check", 20000);
}

export async function refreshCapabilities(): Promise<Record<string, unknown>> {
  const result = await postJson<{ capabilities: Record<string, unknown> }>("/api/setup/refresh-capabilities");
  return result.capabilities;
}

// ---------------------------------------------------------------------------
// FU-022 LLM-based prompt enhancer
// ---------------------------------------------------------------------------

/**
 * FU-022: LLM-based prompt enhancer. Rewrites a short user prompt into
 * the structured format the requested image / video model was trained
 * on. Apple Silicon path uses mlx_lm with a small instruct model
 * (default mlx-community/Qwen2.5-0.5B-Instruct-4bit, ~700 MB). Other
 * platforms use the backend's deterministic template fallback.
 */
export interface PromptEnhanceResult {
  enhanced: string;
  note: string | null;
  modelUsed: string | null;
  family: string;
}

export async function enhancePromptViaLLM(payload: {
  prompt: string;
  repo: string;
  modelId?: string;
  maxTokens?: number;
}): Promise<PromptEnhanceResult> {
  // Long timeout: the first call materialises the model (~2-3s on
  // M-series cold cache), subsequent calls are sub-second. 30s is
  // enough headroom for first-call without waiting forever if the
  // model fails to load.
  const body = {
    prompt: payload.prompt,
    repo: payload.repo,
    modelId: payload.modelId ?? null,
    maxTokens: payload.maxTokens ?? 256,
  };
  return await postJson<PromptEnhanceResult>("/api/prompt/enhance", body, 30000);
}
