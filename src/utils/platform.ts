import type { SystemStats } from "../types";

/**
 * Host-platform classifiers (FU-056 follow-up).
 *
 * Centralised so UI gates can ask "is this host capable of running
 * X?" without sprinkling ``osSystem === "darwin"`` checks across every
 * Studio + Settings + RuntimeControls surface. Reads from
 * ``workspace.system`` which the backend already populates from
 * ``platform.system()`` + ``platform.machine()``.
 *
 * The three checks here are the ones that actually gate UI today:
 *
 *   - ``isAppleSiliconHost`` — Darwin + arm64. MLX, MLX-LM, MLX-VLM,
 *     mlx-video, mflux, MTPLX, dflash-mlx, turboquant-mlx-full all
 *     need Apple Silicon hardware (the MLX framework is closed to
 *     Metal-backed unified-memory devices). UI install prompts for
 *     any of these are noise + an install attempt would silently no-op
 *     on Windows / Linux / Intel Mac.
 *
 *   - ``isCudaHost`` — Windows or Linux (x86_64). vLLM, nunchaku,
 *     sageattention, dflash (CUDA package), triattention, kvpress,
 *     LongLive all need a CUDA-class GPU. macOS hosts can't reach
 *     these regardless of GPU brand (no CUDA drivers on macOS).
 *
 *   - ``isIntelMac`` — Darwin + x86_64. Rare today but worth gating
 *     separately because the user gets neither Apple-Silicon-only
 *     MLX nor (typically) CUDA, so the UI should be honest about
 *     the empty option set.
 *
 * All checks accept a missing / partial ``system`` so the early-paint
 * skeleton state never crashes a surface. ``undefined`` reads as
 * "don't show platform-specific affordances yet" — preferable to a
 * flash of inappropriate UI before the probe lands.
 */

type SystemLike = Pick<SystemStats, "platform" | "arch"> | null | undefined;

function normalize(value: string | undefined): string {
  return (value ?? "").trim().toLowerCase();
}

/** True when ``system`` describes a Darwin host on Apple Silicon
 * (``arm64`` / ``aarch64``). Negative otherwise — including when
 * ``system`` is undefined. */
export function isAppleSiliconHost(system: SystemLike): boolean {
  if (!system) return false;
  const platform = normalize(system.platform);
  const arch = normalize(system.arch);
  return platform === "darwin" && (arch === "arm64" || arch === "aarch64");
}

/** True when ``system`` describes a Windows or Linux x86_64 host.
 * Used to gate CUDA-only install affordances (vLLM, nunchaku, etc.). */
export function isCudaHost(system: SystemLike): boolean {
  if (!system) return false;
  const platform = normalize(system.platform);
  if (platform !== "windows" && platform !== "linux") return false;
  const arch = normalize(system.arch);
  return arch === "x86_64" || arch === "amd64";
}

/** True when ``system`` describes an Intel Mac (rare in 2026 but
 * still shipping — neither MLX nor CUDA applies). */
export function isIntelMac(system: SystemLike): boolean {
  if (!system) return false;
  const platform = normalize(system.platform);
  const arch = normalize(system.arch);
  return platform === "darwin" && (arch === "x86_64" || arch === "amd64");
}
