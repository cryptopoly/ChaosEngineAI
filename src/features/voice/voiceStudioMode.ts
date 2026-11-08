/** Which Voice Studio mode opens: Speak the first time, then the last one used (kept in localStorage). */

export type StudioMode = "speak" | "transcribe" | "voices";

export const STUDIO_MODES: StudioMode[] = ["speak", "transcribe", "voices"];
export const DEFAULT_STUDIO_MODE: StudioMode = "speak";
export const STUDIO_MODE_KEY = "chaos.studio.voice-studio.mode";

function isMode(value: unknown): value is StudioMode {
  return typeof value === "string" && (STUDIO_MODES as string[]).includes(value);
}

export function readStudioMode(): StudioMode {
  try {
    const stored = window.localStorage.getItem(STUDIO_MODE_KEY);
    return isMode(stored) ? stored : DEFAULT_STUDIO_MODE;
  } catch {
    return DEFAULT_STUDIO_MODE;
  }
}

export function writeStudioMode(mode: StudioMode): void {
  try {
    window.localStorage.setItem(STUDIO_MODE_KEY, mode);
  } catch {
    // Storage blocked: the Studio opens on Speak next time.
  }
}

/** A request from another tab (Library, Models): a transcript opens Transcribe; otherwise the named mode. */
export interface VoiceStudioRequest {
  /** A Library transcript to open in the editor. */
  id?: string;
  mode?: StudioMode;
  /** A voice to select on Speak. */
  voice?: string;
  /** Text to put in the Speak box (the Library's Speak again). */
  text?: string;
  /** Changes on every request, so opening the same item twice works. */
  nonce: number;
}

export function requestedMode(request: VoiceStudioRequest): StudioMode | null {
  if (request.id) return "transcribe";
  return request.mode ?? null;
}
