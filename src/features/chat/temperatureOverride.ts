/**
 * Per-thread temperature override storage helper.
 *
 * Stored by ChatTab's TemperatureChip under ``chat.tempOverride.<sessionId>``;
 * read here when assembling the stream payload. Returns ``null`` when no
 * override is set, in which case the launch-settings default applies.
 */

const STORAGE_KEY_PREFIX = "chat.tempOverride.";

function storageKey(sessionId: string): string {
  return `${STORAGE_KEY_PREFIX}${sessionId}`;
}

export function readTemperatureOverride(sessionId: string | null | undefined): number | null {
  if (!sessionId || typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(storageKey(sessionId));
    if (raw == null) return null;
    const parsed = parseFloat(raw);
    return Number.isFinite(parsed) ? parsed : null;
  } catch {
    return null;
  }
}
