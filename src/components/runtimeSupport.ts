import type { SystemStats } from "../types";

const COMMUNITY_PREFIXES = ["mlx-community/", "lmstudio-community/", "thebloke/", "bartowski/"];
const QUANT_SUFFIXES = /[-_](?:bf16|fp16|f16|\d+bit|q\d(?:_[a-z0-9]+)*|gguf|mlx|instruct)$/i;

export const STRATEGY_ENGINE_SUPPORT: Record<string, string[]> = {
  native: ["mlx", "gguf", "llama.cpp", "vllm", "auto"],
  triattention: ["vllm"],
  rotorquant: ["gguf", "llama.cpp", "vllm"],
  turboquant: ["mlx", "gguf", "llama.cpp", "vllm", "auto"],
  chaosengine: ["gguf", "llama.cpp", "vllm"],
};

export function isStrategyCompatible(strategyId: string, backend: string | null | undefined): boolean {
  if (!backend || backend === "auto") return true;
  const supported = STRATEGY_ENGINE_SUPPORT[strategyId];
  if (!supported) return true;
  return supported.some((candidate) => backend.includes(candidate));
}

export function strategyIncompatReason(strategyId: string, backend: string | null | undefined): string | null {
  if (!backend || backend === "auto" || isStrategyCompatible(strategyId, backend)) return null;
  const engineLabel = backend.includes("gguf") || backend.includes("llama") ? "llama.cpp" : backend;
  if (strategyId === "triattention") return "TriAttention requires the vLLM backend (Linux + CUDA).";
  if (strategyId === "rotorquant") return `RotorQuant requires llama.cpp or vLLM, not ${engineLabel}.`;
  if (strategyId === "chaosengine") return `ChaosEngine requires llama.cpp or vLLM, not ${engineLabel}.`;
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
  ddtreeAvailable: boolean;
} {
  const backend = selectedBackend ?? null;
  const isGgufBackend = backend ? (backend.includes("gguf") || backend.includes("llama")) : false;
  const ddtreeAvailable = Boolean(dflashInfo?.ddtreeAvailable);

  if (isGgufBackend) {
    return {
      enabled: false,
      reason: "DFlash is not supported with llama.cpp models. Use an MLX or vLLM model.",
      matchedModel: null,
      ddtreeAvailable,
    };
  }

  if (!(dflashInfo?.available ?? false)) {
    return {
      enabled: false,
      reason: "Install dflash-mlx (Apple Silicon) or dflash (CUDA) to enable.",
      matchedModel: null,
      ddtreeAvailable,
    };
  }

  const supportedModels = dflashInfo?.supportedModels ?? [];
  const candidates = candidateKeys([canonicalRepo, modelRef, modelName]);
  if (supportedModels.length === 0 || candidates.length === 0) {
    return {
      enabled: true,
      reason: null,
      matchedModel: null,
      ddtreeAvailable,
    };
  }

  for (const supportedModel of supportedModels) {
    const supportedKeys = candidateKeys([supportedModel]);
    if (supportedKeys.some((key) => candidates.includes(key))) {
      return {
        enabled: true,
        reason: null,
        matchedModel: supportedModel,
        ddtreeAvailable,
      };
    }
  }

  return {
    enabled: false,
    reason: "No compatible DFlash draft is registered for this model.",
    matchedModel: null,
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
