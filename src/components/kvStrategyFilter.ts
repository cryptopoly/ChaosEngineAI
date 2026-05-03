import type { SystemStats } from "../types";
import { isStrategyCompatible } from "./runtimeSupport";

/**
 * Filter the in-chat KV cache strategy popover so it shows the same
 * "actually usable on this loaded model" set the launch-settings modal
 * shows under the Cache strategy section.
 *
 * Single source of truth = ``STRATEGY_ENGINE_SUPPORT`` in
 * ``runtimeSupport.ts``. The modal uses ``isStrategyCompatible`` to
 * mark cards N/A; we use the same predicate here to drop them
 * entirely from the popover (the chip is a quick override, not a
 * teaching surface — keeping a stale "RotorQuant 4-bit" entry in a
 * popover for an MLX-loaded model just adds noise).
 *
 * Three filter layers:
 *
 * 1. Domain: drop strategies whose ``appliesTo`` doesn't include
 *    ``"text"`` (e.g. TeaCache, FBCache — diffusion-only).
 * 2. Engine compatibility: drop strategies the loaded engine can't
 *    run, mirroring ``STRATEGY_ENGINE_SUPPORT``. When the engine is
 *    unknown (no model loaded yet, or the field arrived ``null``)
 *    keep every text strategy so the user has full options the moment
 *    a model loads.
 * 3. Availability: drop strategies whose backing pip / binary isn't
 *    installed in this venv. Mirrors the modal's "N/A" badge — except
 *    here we hide instead of grey-out to keep the popover compact.
 *    ``native`` always survives (no install dependency).
 */

// Substrates whose names appear inside the engine string and that
// ``STRATEGY_ENGINE_SUPPORT`` knows about. When the engine name doesn't
// contain any of these (e.g. ``"remote"``, ``"mock"``, ``"base"``,
// ``"made-up"``), we treat the engine as "unknown to this filter" and
// skip the layer-2 check rather than hiding every option — keeping the
// chip useful on stub / passthrough substrates the modal also doesn't
// gate.
const KNOWN_SUBSTRATE_TOKENS = ["mlx", "gguf", "llama.cpp", "llamacpp", "vllm", "auto"];

function isKnownSubstrate(engineKey: string): boolean {
  if (!engineKey) return false;
  const lowered = engineKey.toLowerCase();
  return KNOWN_SUBSTRATE_TOKENS.some((token) => lowered.includes(token));
}

export function filterTextStrategies(
  strategies: SystemStats["availableCacheStrategies"] | undefined,
  engine: string | null | undefined,
): SystemStats["availableCacheStrategies"] {
  if (!strategies) return [];
  const engineKey = (engine ?? "").trim();
  const knownSubstrate = isKnownSubstrate(engineKey);

  return strategies.filter((strategy) => {
    // Layer 1: domain.
    const appliesTo = strategy.appliesTo ?? ["text"];
    if (!appliesTo.includes("text")) return false;

    // Layer 2: engine compatibility — single source of truth shared
    // with the launch-settings modal so the two surfaces never drift.
    // ``native`` always survives because it has no substrate
    // requirement (it's the f16 fallback every engine speaks). Other
    // strategies are dropped on a known substrate where
    // ``isStrategyCompatible`` returns false. Unknown substrates
    // ("remote" / "mock" / "base" — values the modal never touches)
    // skip this layer so the chip stays useful in those modes.
    if (
      strategy.id !== "native"
      && knownSubstrate
      && !isStrategyCompatible(strategy.id, engineKey)
    ) {
      return false;
    }

    // Layer 3: availability. ``native`` is always usable; everything
    // else needs the backing package or binary present.
    if (strategy.id !== "native" && !strategy.available) return false;

    return true;
  });
}
