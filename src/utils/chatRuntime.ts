import type { ChatSession, LaunchPreferences, LoadedModel } from "../types";

export type ChatRuntimeProfile = Pick<
  LaunchPreferences,
  "cacheBits" | "fp16Layers" | "fusedAttention" | "cacheStrategy" | "fitModelInMemory" | "contextTokens" | "speculativeDecoding" | "treeBudget"
>;

export function resolveChatRuntimeProfile(
  _session: ChatSession | null | undefined,
  launchSettings: LaunchPreferences,
): ChatRuntimeProfile {
  // Launch settings (the global panel) always take precedence for runtime
  // parameters.  Previously, session-level overrides from creation time
  // could shadow the current panel values (e.g. a session created with
  // speculativeDecoding=false would ignore a later DFlash toggle).
  // Users control these via the launch settings panel and expect changes
  // to apply immediately to the active thread.
  return {
    cacheBits: launchSettings.cacheBits,
    fp16Layers: launchSettings.fp16Layers,
    fusedAttention: launchSettings.fusedAttention,
    cacheStrategy: launchSettings.cacheStrategy,
    fitModelInMemory: launchSettings.fitModelInMemory,
    contextTokens: launchSettings.contextTokens,
    speculativeDecoding: launchSettings.speculativeDecoding,
    treeBudget: launchSettings.treeBudget,
  };
}

export function loadedModelMatchesRuntimeProfile(
  loadedModel: LoadedModel | null | undefined,
  profile: ChatRuntimeProfile,
): boolean {
  if (!loadedModel) return false;
  return (
    loadedModel.cacheBits === profile.cacheBits &&
    loadedModel.fp16Layers === profile.fp16Layers &&
    loadedModel.fusedAttention === profile.fusedAttention &&
    loadedModel.cacheStrategy === profile.cacheStrategy &&
    loadedModel.fitModelInMemory === profile.fitModelInMemory &&
    loadedModel.contextTokens === profile.contextTokens &&
    (loadedModel.speculativeDecoding ?? false) === profile.speculativeDecoding &&
    (loadedModel.treeBudget ?? 0) === profile.treeBudget
  );
}
