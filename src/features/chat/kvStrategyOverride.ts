/**
 * Phase 3.2: per-thread KV strategy override storage.
 *
 * The composer's KV strategy chip writes a `{strategy, bits}` blob
 * to localStorage keyed by session id. useChat reads it when
 * assembling each stream payload — backend transparently reloads
 * the runtime when the requested cacheStrategy / cacheBits don't
 * match what's currently loaded.
 *
 * Pass `null` to clear and revert to the session's default profile.
 * Reads are best-effort — corrupt or unparseable storage entries
 * return null so the active runtime profile applies.
 */

export interface KvStrategyOverride {
  strategy: string;
  bits: number;
}

const STORAGE_KEY_PREFIX = "chat.kvStrategy.";

function storageKey(sessionId: string): string {
  return `${STORAGE_KEY_PREFIX}${sessionId}`;
}

export function readKvStrategyOverride(
  sessionId: string | null | undefined,
): KvStrategyOverride | null {
  if (!sessionId || typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(storageKey(sessionId));
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (
      parsed
      && typeof parsed === "object"
      && typeof parsed.strategy === "string"
      && parsed.strategy
      && typeof parsed.bits === "number"
      && Number.isFinite(parsed.bits)
    ) {
      return { strategy: parsed.strategy, bits: parsed.bits };
    }
    return null;
  } catch {
    return null;
  }
}

export function writeKvStrategyOverride(
  sessionId: string | null | undefined,
  value: KvStrategyOverride | null,
): void {
  if (!sessionId || typeof window === "undefined") return;
  try {
    if (value === null) {
      window.localStorage.removeItem(storageKey(sessionId));
    } else {
      window.localStorage.setItem(storageKey(sessionId), JSON.stringify(value));
    }
  } catch {
    // localStorage unavailable — in-memory state still applies for this render
  }
}
