import type { SamplerOverrides } from "../../types";

/**
 * Phase 2.2: per-thread sampler override storage helpers.
 *
 * The SamplerPanel writes user-set overrides to localStorage keyed by
 * `chat.samplers.<sessionId>`. useChat reads from the same key when
 * assembling each stream payload so the override survives navigation
 * between threads and app restarts. Reads are best-effort — corrupt or
 * unparseable storage entries return an empty object so the backend's
 * defaults apply.
 */

const STORAGE_KEY_PREFIX = "chat.samplers.";

const NUMERIC_KEYS = [
  "topP",
  "topK",
  "minP",
  "repeatPenalty",
  "seed",
  "mirostatTau",
  "mirostatEta",
] as const;

function storageKey(sessionId: string): string {
  return `${STORAGE_KEY_PREFIX}${sessionId}`;
}

function sanitize(raw: unknown): SamplerOverrides {
  if (!raw || typeof raw !== "object") return {};
  const obj = raw as Record<string, unknown>;
  const result: SamplerOverrides = {};
  for (const key of NUMERIC_KEYS) {
    const value = obj[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      (result as Record<string, unknown>)[key] = value;
    }
  }
  if (obj.mirostatMode === 0 || obj.mirostatMode === 1 || obj.mirostatMode === 2) {
    result.mirostatMode = obj.mirostatMode;
  }
  // Phase 2.2: keep raw JSON-schema text round-trippable. We intentionally
  // don't validate-parse here so a half-typed schema persists across
  // remounts; the parse + validation happens at send time and on render.
  if (typeof obj.jsonSchemaText === "string" && obj.jsonSchemaText.length > 0) {
    result.jsonSchemaText = obj.jsonSchemaText;
  }
  return result;
}

/** Read the per-thread sampler overrides. Returns `{}` when nothing is stored. */
export function readSamplerOverrides(sessionId: string | null | undefined): SamplerOverrides {
  if (!sessionId || typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(storageKey(sessionId));
    if (!raw) return {};
    return sanitize(JSON.parse(raw));
  } catch {
    return {};
  }
}

/** Write per-thread sampler overrides. Pass an empty object to clear. */
export function writeSamplerOverrides(
  sessionId: string | null | undefined,
  overrides: SamplerOverrides,
): void {
  if (!sessionId || typeof window === "undefined") return;
  try {
    const cleaned = sanitize(overrides);
    if (Object.keys(cleaned).length === 0) {
      window.localStorage.removeItem(storageKey(sessionId));
    } else {
      window.localStorage.setItem(storageKey(sessionId), JSON.stringify(cleaned));
    }
  } catch {
    // localStorage unavailable; in-memory state still applies for the session
  }
}

/**
 * Project the override blob into the GeneratePayload field shape so
 * useChat can spread it directly into the request body. Returns only
 * fields that were actually set, matching the backend's "None means
 * use default" contract.
 */
export function samplerPayload(overrides: SamplerOverrides): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (overrides.topP != null) out.topP = overrides.topP;
  if (overrides.topK != null) out.topK = overrides.topK;
  if (overrides.minP != null) out.minP = overrides.minP;
  if (overrides.repeatPenalty != null) out.repeatPenalty = overrides.repeatPenalty;
  if (overrides.seed != null) out.seed = overrides.seed;
  if (overrides.mirostatMode != null) out.mirostatMode = overrides.mirostatMode;
  if (overrides.mirostatTau != null) out.mirostatTau = overrides.mirostatTau;
  if (overrides.mirostatEta != null) out.mirostatEta = overrides.mirostatEta;
  // Phase 2.2: parse raw schema text just-in-time. Mid-type / malformed
  // input drops out silently rather than 400-ing the request — the user
  // sees the in-panel error indicator while typing.
  const schemaText = overrides.jsonSchemaText;
  if (schemaText && typeof schemaText === "string" && schemaText.trim().length > 0) {
    try {
      const parsed = JSON.parse(schemaText);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        out.jsonSchema = parsed;
      }
    } catch {
      // Surface only via the panel UI; don't block the send.
    }
  }
  return out;
}
