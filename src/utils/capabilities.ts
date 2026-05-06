import type { ModelCapabilities } from "../types";

/**
 * Phase 2.11: frontend mirror of `backend/catalog/capabilities.py`.
 *
 * The backend resolves typed `ModelCapabilities` for the loaded model
 * (so the chat header can render runtime-aware badges). The picker
 * shows options that aren't loaded yet — we still want capability
 * badges so users know what each option supports before clicking
 * Load. This helper maps the catalog's free-form `capabilities: [...]`
 * string list onto the same typed shape.
 *
 * Catalog tags are conservative-by-design (omitted rather than
 * promised). Heuristic ref-name sniffing matches the backend so a
 * freshly-downloaded model without a catalog entry still gets sensible
 * defaults.
 */

const TAG_TO_FLAG: Record<string, keyof Omit<ModelCapabilities, "tags">> = {
  vision: "supportsVision",
  multimodal: "supportsVision",
  "tool-use": "supportsTools",
  tools: "supportsTools",
  "function-calling": "supportsTools",
  reasoning: "supportsReasoning",
  thinking: "supportsReasoning",
  coding: "supportsCoding",
  code: "supportsCoding",
  agents: "supportsAgents",
  agent: "supportsAgents",
  audio: "supportsAudio",
  video: "supportsVideo",
  multilingual: "supportsMultilingual",
};

export function emptyCapabilities(): ModelCapabilities {
  return {
    supportsVision: false,
    supportsTools: false,
    supportsReasoning: false,
    supportsCoding: false,
    supportsAgents: false,
    supportsAudio: false,
    supportsVideo: false,
    supportsMultilingual: false,
    tags: [],
  };
}

function heuristicTags(modelRef: string | null | undefined): string[] {
  if (!modelRef) return [];
  const lower = modelRef.toLowerCase();
  const out: string[] = [];
  if (
    ["-vl-", " vl ", "/vl-", "vision", "llava", "qwen-vl", "moondream"].some(
      (needle) => lower.includes(needle),
    )
  ) {
    out.push("vision");
  }
  if (
    ["coder", "/code-", "starcoder", "deepseek-coder", "code-llama"].some(
      (needle) => lower.includes(needle),
    )
  ) {
    out.push("coding");
  }
  if (
    ["r1", "reasoning", "think", "qwen3", "deepseek-r"].some((needle) =>
      lower.includes(needle),
    )
  ) {
    out.push("reasoning");
  }
  if (lower.includes("tool") || lower.includes("function")) {
    out.push("tool-use");
  }
  if (
    (lower.includes("instruct") || lower.includes("-it") || lower.includes("chat")) &&
    !out.includes("tool-use")
  ) {
    out.push("tool-use");
  }
  return out;
}

/**
 * Resolve typed capabilities from a model ref + optional catalog tags.
 *
 * - Catalog tags (when present) take precedence
 * - Otherwise heuristic ref-name sniffing fills in
 * - Result mirrors the backend resolver one-to-one
 */
export function resolveCapabilities(
  modelRef: string | null | undefined,
  catalogTags: string[] | null | undefined,
): ModelCapabilities {
  const raw = catalogTags?.length ? catalogTags : heuristicTags(modelRef);
  const caps = emptyCapabilities();
  const seen = new Set<string>();
  for (const tag of raw) {
    const normalised = tag.trim().toLowerCase();
    if (!normalised) continue;
    seen.add(normalised);
    const flag = TAG_TO_FLAG[normalised];
    if (flag) caps[flag] = true;
  }
  caps.tags = [...seen].sort();
  return caps;
}
