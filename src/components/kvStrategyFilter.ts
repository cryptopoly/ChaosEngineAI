import type { SystemStats } from "../types";

/**
 * Phase 3.2 hotfix: filter the cache-strategy popover to only show
 * strategies that are valid for the *currently loaded* model.
 *
 * Three filter layers:
 *
 * 1. Domain: drop strategies whose `appliesTo` doesn't include `"text"`
 *    (e.g. TeaCache is diffusion-only — it should never appear in the
 *    chat composer).
 *
 * 2. Engine compatibility: each engine has a different set of cache
 *    strategies it can actually run. Picking a strategy the engine
 *    can't run causes a hard "Chat error: Load failed" (the user
 *    reported this with TeaCache + Gemma-4 on MLX). We map engine →
 *    allowed strategy IDs based on the substrate.
 *
 * 3. Availability — the strategy itself reports `available: false`
 *    when the binary or pip dep is missing; we keep these in the list
 *    but the chip greys them out so the user can see the option exists.
 */

const ENGINE_TEXT_STRATEGIES: Record<string, string[]> = {
  // MLX worker: native f16 always works; turboquant has a dedicated
  // mlx pip path; triattention has an mlx_compressor (FU-002 in
  // CLAUDE.md flags upstream gaps but the strategy is registered).
  // RotorQuant + ChaosEngine are llama.cpp-only.
  mlx: ["native", "turboquant", "triattention"],
  mlx_worker: ["native", "turboquant", "triattention"],
  // llama.cpp: native + chaosengine on the standard binary; rotorquant
  // + turboquant on the turbo binary. TriAttention has no llama.cpp
  // hook (its forward patch targets transformers).
  llamacpp: ["native", "rotorquant", "turboquant", "chaosengine"],
  llama: ["native", "rotorquant", "turboquant", "chaosengine"],
  // vLLM (CUDA): triattention + native are the wired paths.
  vllm: ["native", "triattention"],
};

export function filterTextStrategies(
  strategies: SystemStats["availableCacheStrategies"] | undefined,
  engine: string | null | undefined,
): SystemStats["availableCacheStrategies"] {
  if (!strategies) return [];
  const engineLower = (engine ?? "").trim().toLowerCase();
  const allowList = engineLower ? ENGINE_TEXT_STRATEGIES[engineLower] : null;

  return strategies.filter((strategy) => {
    // Layer 1: domain — must apply to text inference.
    const appliesTo = strategy.appliesTo ?? ["text"];
    if (!appliesTo.includes("text")) return false;

    // Layer 2: engine compatibility — drop strategies the loaded
    // runtime can't actually run. When engine is unknown (no model
    // loaded yet), keep all text strategies so the user has options
    // post-load.
    if (allowList && !allowList.includes(strategy.id)) return false;

    return true;
  });
}
