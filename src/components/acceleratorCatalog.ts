/**
 * Accelerator registry (FU-056 Phase 2).
 *
 * Source of truth for the six CUDA-side accelerators the in-app install
 * UX surfaces. Each entry pairs a stable ``id`` with the metadata that
 * downstream components need to render a "Recommended" badge, an
 * Install button, an "Installed ✓" pill, or a Boost Pack row:
 *
 *   - ``pipPackage`` — argument to ``POST /api/setup/install-package``.
 *     Must match a key in the backend's ``_INSTALLABLE_PIP_PACKAGES``
 *     allow-list ([backend_service/routes/setup/__init__.py]).
 *   - ``capabilityField`` / ``versionField`` — the ``NativeBackendStatus``
 *     keys to read for installed state + display version. Wired in
 *     FU-056 Phase 1 on the backend.
 *   - ``speedupClaim`` / ``appliesTo`` — copy for the "🚀 Nunchaku +3×
 *     available" pill. Marketing-honest: never claim more than the
 *     model card / upstream benchmark reports for the *typical* case
 *     a user will hit.
 *   - ``sizeOnDiskLabel`` — rough human-readable on-disk footprint
 *     (compressed download + extracted wheel). Sourced from the
 *     CLAUDE.md FU rows that registered each package.
 *   - ``installMode`` — ``"sync"`` for ~5 min installs that we can hold
 *     a single HTTP call open for; ``"async"`` for the >5 min builds
 *     (triattention compiles flash-attn from source; vLLM ships a
 *     ~2 GB wheel) that need the background-job + poll-status shape.
 *   - ``platformGate`` — when set, the affordance hides on platforms
 *     where the accelerator can't run at all (e.g. dflash-mlx on
 *     Windows, vLLM native on macOS). Diagnostic surfaces that show
 *     "everything" can override this to render a disabled row with
 *     an explanation.
 *
 * Adding a 7th accelerator is one entry here + one Phase 1 capability
 * flag + one row in ``_INSTALLABLE_PIP_PACKAGES``. No component edit.
 */

import type { NativeBackendStatus } from "../types/server";

export type AcceleratorId =
  | "nunchaku"
  | "sageattention"
  | "dflash-mlx"
  | "dflash-cuda"
  | "triattention"
  | "kvpress";

export type PlatformGate = "cuda" | "apple-silicon" | "any";

export interface AcceleratorMeta {
  id: AcceleratorId;
  /** Human-readable label shown in cards + Boost Pack rows. */
  label: string;
  /** Short noun phrase suitable for a pill: "4-bit FLUX/SD3" not "Adds 4-bit support". */
  shortLabel: string;
  /** Pip name as it appears in ``_INSTALLABLE_PIP_PACKAGES``. */
  pipPackage: string;
  /** Capability flag on ``NativeBackendStatus`` (FU-056 Phase 1). */
  capabilityField: keyof NativeBackendStatus;
  /** Version string field (may be ``null`` when installed without a __version__). */
  versionField: keyof NativeBackendStatus;
  /** One-line copy explaining the speedup. Used in the "🚀 X available" pill. */
  speedupClaim: string;
  /** Models / pipelines this accelerator applies to. Free-text — humans read this. */
  appliesTo: string;
  /** Rough on-disk footprint label, e.g. "~50 MB". */
  sizeOnDiskLabel: string;
  /** ``sync`` = one HTTP call held open; ``async`` = background job + status poll. */
  installMode: "sync" | "async";
  /** Platforms where this can actually run. Affordances hide on the wrong platform. */
  platformGate: PlatformGate;
  /** FU row in CLAUDE.md that registered or owns this accelerator. For provenance. */
  followUp: string;
  /** Optional doc link slug under ``docs/features/`` for a "Learn more" affordance. */
  docsSlug?: string;
}

export const ACCELERATOR_CATALOG: ReadonlyArray<AcceleratorMeta> = [
  {
    id: "nunchaku",
    label: "Nunchaku",
    shortLabel: "SVDQuant 4-bit",
    pipPackage: "nunchaku",
    capabilityField: "nunchakuAvailable",
    versionField: "nunchakuVersion",
    speedupClaim: "≈3× faster FLUX/SD3.5/Qwen-Image on CUDA",
    appliesTo: "FLUX.1, SD3.5, Qwen-Image, SANA, PixArt-Σ",
    sizeOnDiskLabel: "~50 MB",
    installMode: "sync",
    platformGate: "cuda",
    followUp: "FU-023",
  },
  {
    id: "sageattention",
    label: "SageAttention",
    shortLabel: "Fast attention DiT",
    pipPackage: "sageattention",
    capabilityField: "sageattentionAvailable",
    versionField: "sageattentionVersion",
    speedupClaim: "Stacks with FBCache for ~1.4× extra on DiT pipelines",
    appliesTo: "Any CUDA DiT image / video pipeline",
    sizeOnDiskLabel: "~30 MB",
    installMode: "sync",
    platformGate: "cuda",
    followUp: "FU-016",
  },
  {
    id: "dflash-mlx",
    label: "DFlash (MLX)",
    shortLabel: "Speculative decoding",
    pipPackage: "dflash-mlx",
    capabilityField: "dflashMlxAvailable",
    versionField: "dflashMlxVersion",
    speedupClaim: "≈1.5-2× tokens/sec on Qwen3.x and DeepSeek chat models",
    appliesTo: "Apple Silicon — any LLM with a registered draft model",
    sizeOnDiskLabel: "~80 MB",
    installMode: "sync",
    platformGate: "apple-silicon",
    followUp: "FU-031",
    docsSlug: "dflash",
  },
  {
    id: "dflash-cuda",
    label: "DFlash (CUDA)",
    shortLabel: "Speculative decoding",
    pipPackage: "dflash",
    capabilityField: "dflashCudaAvailable",
    versionField: "dflashCudaVersion",
    speedupClaim: "≈1.5-2× tokens/sec on Qwen3.x and DeepSeek chat models",
    appliesTo: "CUDA — any LLM with a registered draft model",
    sizeOnDiskLabel: "~80 MB",
    installMode: "sync",
    platformGate: "cuda",
    followUp: "FU-048",
    docsSlug: "dflash",
  },
  {
    id: "triattention",
    label: "TriAttention",
    shortLabel: "KV compressor + LongLive",
    // The full pip git+url is resolved server-side by the install-package
    // registry — the client only needs the package name as the registry
    // key. Keeps the catalog readable + avoids leaking the upstream pin
    // into the frontend bundle.
    pipPackage: "triattention",
    capabilityField: "triattentionAvailable",
    versionField: "triattentionVersion",
    speedupClaim: "Real-time long Wan video + 2-3× KV compression on long-context LLMs",
    appliesTo: "Wan 2.1 1.3B (LongLive), long-context chat models",
    sizeOnDiskLabel: "~2 GB (pulls vllm)",
    installMode: "async",
    platformGate: "cuda",
    followUp: "FU-003 / FU-002",
  },
  {
    id: "kvpress",
    label: "kvpress",
    shortLabel: "KV cache compression",
    pipPackage: "kvpress",
    capabilityField: "kvpressAvailable",
    versionField: "kvpressVersion",
    speedupClaim: "8-32× KV-cache compression on long-context CUDA inference",
    appliesTo: "CUDA — any HF transformer with KV cache",
    sizeOnDiskLabel: "~40 MB",
    installMode: "sync",
    platformGate: "cuda",
    followUp: "FU-027",
  },
];

/** Lookup an entry by id. Returns ``undefined`` for unknown ids so the
 * caller can render a "missing catalog row" diagnostic rather than
 * crashing — relevant for forward-compat when a backend probe lists a
 * new accelerator the frontend doesn't know about yet. */
export function getAccelerator(id: string): AcceleratorMeta | undefined {
  return ACCELERATOR_CATALOG.find((entry) => entry.id === id);
}

/** True when this accelerator's ``platformGate`` is satisfied by the
 * current ``NativeBackendStatus``. The caller can use this to hide
 * irrelevant cards (e.g. dflash-mlx on Windows) or to dim them with an
 * explanation tooltip. ``any`` always satisfies. */
export function isPlatformCompatible(
  meta: AcceleratorMeta,
  capabilities: Pick<NativeBackendStatus, "mlxAvailable">,
): boolean {
  switch (meta.platformGate) {
    case "any":
      return true;
    case "apple-silicon":
      // ``mlxAvailable`` is the strongest signal we have for "this is an
      // Apple Silicon box where MLX worker subprocesses can spawn".
      // ``platform.system() === "Darwin"`` would catch Intel Macs too, but
      // none of the MLX-side accelerators run on Intel anyway, so MLX
      // availability is the better gate.
      return Boolean(capabilities.mlxAvailable);
    case "cuda":
      // We don't have a single ``cudaAvailable`` capability flag today
      // (the vllm probe carries it implicitly). For Phase 2 we approximate
      // "this is a CUDA box" with "MLX is NOT available" — i.e. not an
      // Apple Silicon box. A more precise probe lands in Phase 8 alongside
      // the WSL bridge work, when we surface ``cudaAvailable`` explicitly.
      return !capabilities.mlxAvailable;
  }
}
