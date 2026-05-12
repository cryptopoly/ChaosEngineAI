/**
 * Phase 1.12: per-thread reasoning effort storage helper.
 *
 * Stored alongside thinkingMode but separate so a session can independently
 * track "Off" vs Low/Medium/High effort. Returns ``undefined`` when no
 * level is stored, which lets the backend treat absence as "use whatever
 * the model defaults to".
 */

import type { ChatReasoningEffort } from "../../types";


const STORAGE_KEY_PREFIX = "chat.reasoningEffort.";

function storageKey(sessionId: string): string {
  return `${STORAGE_KEY_PREFIX}${sessionId}`;
}

export function readReasoningEffort(
  sessionId: string | null | undefined,
): ChatReasoningEffort | undefined {
  if (!sessionId || typeof window === "undefined") return undefined;
  try {
    const raw = window.localStorage.getItem(storageKey(sessionId));
    if (raw === "low" || raw === "medium" || raw === "high") return raw;
  } catch {
    // ignore
  }
  return undefined;
}
