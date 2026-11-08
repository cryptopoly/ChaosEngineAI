import { micErrorMessage } from "./audio";

// The only two URLs the app opens for this; src-tauri/capabilities/default.json allows exactly these.
const MAC_MIC_SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone";
const WINDOWS_MIC_SETTINGS_URL = "ms-settings:privacy-microphone";

/** The OS page for microphone permission, or null where none exists (Linux has no universal page). */
export function micSettingsUrl(platform: string): string | null {
  if (platform === "Darwin") return MAC_MIC_SETTINGS_URL;
  if (platform === "Windows") return WINDOWS_MIC_SETTINGS_URL;
  return null;
}

/** True when `message` is the "microphone access is blocked" text for this platform. */
export function isMicBlockedMessage(message: string | null | undefined, platform: string): boolean {
  if (!message) return false;
  return message === micErrorMessage(new DOMException("", "NotAllowedError"), platform);
}

/** The URL to offer next to this error, or null (not a blocked error, or no page for this OS). */
export function micSettingsUrlForError(message: string | null | undefined, platform: string): string | null {
  return isMicBlockedMessage(message, platform) ? micSettingsUrl(platform) : null;
}
