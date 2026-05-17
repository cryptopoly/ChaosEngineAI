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


// ---------------------------------------------------------------------------
// Catalog-variant platform gates (FU-056 follow-up)
//
// The Image / Video / Chat catalogs don't carry an explicit platform
// field — the routing info lives in ``runtime`` (image/video) and
// ``backend`` (chat). These helpers normalise that into a single
// ``"apple-silicon" | "cuda" | "any"`` discriminator so the tab
// filters + the AcceleratorsBoostPack can use one rule:
//
//     keep variant iff (gate === "any") || gate-matches-host
//
// "any" includes anything cross-platform — diffusers / llama-server /
// sd.cpp / GGUF — which is the vast majority of catalog rows.
// ---------------------------------------------------------------------------

export type PlatformGate = "apple-silicon" | "cuda" | "any";

interface VariantLikeImageOrVideo {
  runtime?: string | null;
  styleTags?: string[];
  repo?: string;
}

/** Classify an image / video variant by its runtime engine.
 *
 * Discriminators (in priority order):
 *   1. ``runtime`` includes ``mflux`` or ``mlx-video`` → Apple Silicon
 *      only (those engines literally don't exist on Win/Linux).
 *   2. ``runtime`` includes ``nunchaku`` → CUDA only (the SVDQuant
 *      wheels are CUDA-only; the diffusers fallback path is a
 *      separate variant in the catalog).
 *   3. ``styleTags`` carries ``apple-silicon`` / ``cuda`` — catalog
 *      curators flag this explicitly on rows where the discriminator
 *      isn't obvious from runtime alone.
 *   4. ``repo`` prefix ``prince-canuma/`` → the LTX-2 family, all
 *      MLX-native (no diffusers mirror today).
 *   5. Default ``"any"`` — diffusers / sd.cpp / GGUF / Wan-AI base
 *      rows run on every platform via the universal backends.
 */
export function imageOrVideoVariantPlatformGate(variant: VariantLikeImageOrVideo): PlatformGate {
  const runtime = normalize(variant.runtime ?? "");
  const tags = (variant.styleTags ?? []).map((t) => t.toLowerCase());
  const repo = (variant.repo ?? "").toLowerCase();

  if (runtime.includes("mflux") || runtime.includes("mlx-video") || runtime.includes("mlx native")) {
    return "apple-silicon";
  }
  if (repo.startsWith("prince-canuma/")) {
    return "apple-silicon";
  }
  if (tags.includes("apple-silicon")) {
    return "apple-silicon";
  }
  if (runtime.includes("nunchaku") || tags.includes("cuda")) {
    return "cuda";
  }
  return "any";
}

interface VariantLikeChat {
  backend?: string | null;
}

/** Classify a chat option by its inference backend.
 *
 *   - ``mlx`` → Apple Silicon only (the MLX framework has no
 *     Win/Linux build; both direct-launch ``mlx-community/*`` and
 *     convert-then-launch transformers variants share this gate).
 *   - ``vllm`` → CUDA host (no Windows wheels — the user installs
 *     into WSL on Windows, native on Linux).
 *   - everything else (``llama.cpp`` / ``gguf`` / ``transformers``
 *     for image-runtime-style HF loads / ``auto``) → ``"any"``.
 */
export function chatVariantPlatformGate(variant: VariantLikeChat): PlatformGate {
  const backend = normalize(variant.backend ?? "");
  if (backend === "mlx" || backend === "mlx-lm" || backend === "mtplx") {
    return "apple-silicon";
  }
  if (backend === "vllm") {
    return "cuda";
  }
  return "any";
}

/** Cross-cutting "should the UI show this gate's affordances?" check.
 *
 * Returns true when the variant runs cleanly on ``system``. The
 * Apple-Silicon gate accepts any Darwin arm64 host; the CUDA gate
 * accepts Windows + Linux x86_64; ``"any"`` always passes. Used by
 * Discover / Models tabs + the AcceleratorsBoostPack to filter
 * incompatible rows out entirely (per the FU-034 "hide unrecoverable
 * options" rule).
 *
 * Conservative on partial system info: when ``system`` is null /
 * undefined (early paint before probe lands) we return ``true`` so
 * the UI doesn't strip variants prematurely. The flash of slightly-
 * wrong-content is better than a flash of empty Discover.
 */
export function isVariantCompatibleWithHost(
  gate: PlatformGate,
  system: SystemLike,
): boolean {
  if (gate === "any") return true;
  if (!system) return true; // early-paint safety
  if (gate === "apple-silicon") return isAppleSiliconHost(system);
  if (gate === "cuda") return isCudaHost(system);
  return true;
}
