/**
 * The sentence the error banner shows when a chat turn fails. A request that never reached the backend
 * surfaces as the browser's own words ("Failed to fetch" in Chromium, "Load failed" in WebKit,
 * "NetworkError when attempting to fetch resource" in Firefox), which say nothing to a user.
 */
export function chatErrorText(detail: string): string {
  if (/failed to fetch|load failed|networkerror/i.test(detail)) {
    return "The backend is not answering, so the message was not sent. Send it again when the app has reconnected.";
  }
  return `Chat error: ${detail}`;
}
