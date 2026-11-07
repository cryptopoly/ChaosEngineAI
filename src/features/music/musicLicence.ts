/**
 * "I have accepted" for a gated licence, remembered in this browser only (a per-viewer convenience: the
 * backend has no licence record yet, and no model in the catalog is gated today). A broken or missing
 * localStorage reads as "not accepted".
 */
const KEY = "chaos.music.licences";

function read(): Record<string, string> {
  try {
    const raw = window.localStorage.getItem(KEY);
    const parsed = raw ? (JSON.parse(raw) as unknown) : {};
    return parsed && typeof parsed === "object" ? (parsed as Record<string, string>) : {};
  } catch {
    return {};
  }
}

/** `tick` only exists so a caller re-reads after `rememberLicenceAccepted`. */
export function isLicenceAccepted(modelId: string, licenceId: string, _tick = 0): boolean {
  return Boolean(read()[`${modelId}::${licenceId}`]);
}

export function rememberLicenceAccepted(modelId: string, licenceId: string): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify({ ...read(), [`${modelId}::${licenceId}`]: new Date().toISOString() }));
  } catch {
    // The accept still lets this session continue; it just is not remembered.
  }
}
