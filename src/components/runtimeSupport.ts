import type { SystemStats } from "../types";

const COMMUNITY_PREFIXES = ["mlx-community/", "lmstudio-community/", "thebloke/", "bartowski/"];
const QUANT_SUFFIXES = /[-_](?:bf16|fp16|f16|\d+bit|q\d(?:_[a-z0-9]+)*|gguf|mlx|instruct)$/i;

// FU-030 (2026-05-10): chaosengine + rotorquant slots dropped. Persisted
// session configs that still reference them coerce to ``turboquant`` via
// the backend's ``registry.resolve_legacy_id`` map; the same coercion is
// mirrored here so frontend filters work correctly when older session
// snapshots are rehydrated. Update both sides if the alias map changes.
export const LEGACY_STRATEGY_ALIASES: Record<string, string> = {
  chaosengine: "turboquant",
  rotorquant: "turboquant",
};

export function canonicalStrategyId(strategyId: string): string {
  return LEGACY_STRATEGY_ALIASES[strategyId] ?? strategyId;
}

export const STRATEGY_ENGINE_SUPPORT: Record<string, string[]> = {
  native: ["mlx", "gguf", "llama.cpp", "vllm", "auto"],
  triattention: ["vllm"],
  turboquant: ["mlx", "gguf", "llama.cpp", "vllm", "auto"],
};

export function isStrategyCompatible(strategyId: string, backend: string | null | undefined): boolean {
  if (!backend || backend === "auto") return true;
  const canonical = canonicalStrategyId(strategyId);
  const supported = STRATEGY_ENGINE_SUPPORT[canonical];
  if (!supported) return true;
  return supported.some((candidate) => backend.includes(candidate));
}

/** FU-056 Phase 5: pick the right pip package name for DFlash given
 * the active backend. Two distinct pip packages back the same feature:
 *
 *   - ``dflash-mlx`` — git+url to bstnxbt/dflash-mlx, Apple Silicon
 *     MLX backend.
 *   - ``dflash`` — PyPI ``dflash>=0.1.0``, CUDA / vLLM backend.
 *
 * The previous RuntimeControls install button hard-coded
 * ``"dflash-mlx"``, which silently installed the wrong package on
 * Windows / Linux CUDA boxes running vLLM. This helper picks the
 * right one based on the engine string. Falls back to the MLX
 * package for unknown backends — the install will fail loudly if
 * the host doesn't match, which is better than silent no-ops.
 */
export function dflashPackageFor(backend: string | null | undefined): "dflash-mlx" | "dflash" {
  if (backend && backend.toLowerCase().includes("vllm")) return "dflash";
  return "dflash-mlx";
}


export function strategyIncompatReason(strategyId: string, backend: string | null | undefined): string | null {
  if (!backend || backend === "auto" || isStrategyCompatible(strategyId, backend)) return null;
  const engineLabel = backend.includes("gguf") || backend.includes("llama") ? "llama.cpp" : backend;
  const canonical = canonicalStrategyId(strategyId);
  if (canonical === "triattention") return "TriAttention requires the vLLM backend (Linux + CUDA).";
  return `Not compatible with the ${engineLabel} backend.`;
}

function normalizeModelSupportKey(value: string): string {
  let normalized = value.trim().replace(/\\/g, "/");
  if (!normalized) return "";
  if (normalized.startsWith("/")) {
    normalized = normalized.split("/").pop() ?? normalized;
  }
  const lowered = normalized.toLowerCase();
  const communityPrefix = COMMUNITY_PREFIXES.find((prefix) => lowered.startsWith(prefix));
  if (communityPrefix) {
    normalized = normalized.slice(communityPrefix.length);
  }
  for (let i = 0; i < 3; i += 1) {
    const stripped = normalized.replace(QUANT_SUFFIXES, "");
    if (stripped === normalized) break;
    normalized = stripped;
  }
  return normalized;
}

export function candidateKeys(values: Array<string | null | undefined>): string[] {
  const out = new Set<string>();
  for (const value of values) {
    if (!value) continue;
    const normalized = normalizeModelSupportKey(value);
    if (!normalized) continue;
    out.add(normalized.toLowerCase());
    const lastSegment = normalized.split("/").pop();
    if (lastSegment) out.add(lastSegment.toLowerCase());
  }
  return [...out];
}

export function resolveDflashSupport({
  dflashInfo,
  selectedBackend,
  modelRef,
  canonicalRepo,
  modelName,
}: {
  dflashInfo?: SystemStats["dflash"];
  selectedBackend?: string | null;
  modelRef?: string | null;
  canonicalRepo?: string | null;
  modelName?: string | null;
}): {
  enabled: boolean;
  reason: string | null;
  matchedModel: string | null;
  modelSupported: boolean | null;
  ddtreeAvailable: boolean;
} {
  const backend = selectedBackend ?? null;
  const isGgufBackend = backend ? (backend.includes("gguf") || backend.includes("llama")) : false;
  const ddtreeAvailable = Boolean(dflashInfo?.ddtreeAvailable);
  const supportedModels = dflashInfo?.supportedModels ?? [];
  const candidates = candidateKeys([canonicalRepo, modelRef, modelName]);
  let matchedModel: string | null = null;

  if (isGgufBackend) {
    return {
      enabled: false,
      reason: "DFlash is not supported with llama.cpp models. Use an MLX or vLLM model.",
      matchedModel: null,
      modelSupported: false,
      ddtreeAvailable,
    };
  }

  if (supportedModels.length === 0 || candidates.length === 0) {
    if (!(dflashInfo?.available ?? false)) {
      return {
        enabled: false,
        reason: "Install dflash-mlx (Apple Silicon) or dflash (CUDA) to enable.",
        matchedModel: null,
        modelSupported: null,
        ddtreeAvailable,
      };
    }
    return {
      enabled: true,
      reason: null,
      matchedModel: null,
      modelSupported: null,
      ddtreeAvailable,
    };
  }

  for (const supportedModel of supportedModels) {
    const supportedKeys = candidateKeys([supportedModel]);
    if (supportedKeys.some((key) => candidates.includes(key))) {
      matchedModel = supportedModel;
      break;
    }
  }

  if (!matchedModel) {
    return {
      enabled: false,
      reason: "No DFlash draft exists for this model. Supported families: Qwen3/3.5/3.6, LLaMA 3.1, gpt-oss, Kimi.",
      matchedModel: null,
      modelSupported: false,
      ddtreeAvailable,
    };
  }

  if (!(dflashInfo?.available ?? false)) {
    return {
      enabled: false,
      reason: "Install dflash-mlx (Apple Silicon) or dflash (CUDA) to enable.",
      matchedModel,
      modelSupported: true,
      ddtreeAvailable,
    };
  }

  return {
    enabled: true,
    reason: null,
    matchedModel,
    modelSupported: true,
    ddtreeAvailable,
  };
}

export function sanitizeSpeculativeSelection({
  dflashInfo,
  selectedBackend,
  modelRef,
  canonicalRepo,
  modelName,
  speculativeDecoding,
  treeBudget,
}: {
  dflashInfo?: SystemStats["dflash"];
  selectedBackend?: string | null;
  modelRef?: string | null;
  canonicalRepo?: string | null;
  modelName?: string | null;
  speculativeDecoding: boolean;
  treeBudget: number;
}): {
  speculativeDecoding: boolean;
  treeBudget: number;
  support: ReturnType<typeof resolveDflashSupport>;
} {
  const support = resolveDflashSupport({
    dflashInfo,
    selectedBackend,
    modelRef,
    canonicalRepo,
    modelName,
  });
  if (!speculativeDecoding || support.enabled) {
    return {
      speculativeDecoding,
      treeBudget: speculativeDecoding ? treeBudget : 0,
      support,
    };
  }
  return {
    speculativeDecoding: false,
    treeBudget: 0,
    support,
  };
}
