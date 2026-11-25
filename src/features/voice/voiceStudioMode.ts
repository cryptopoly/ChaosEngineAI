/** Which Voice Studio task opens: Speak the first time, then the last one used (kept in localStorage). Shown in the
 * order Speak, Clone a voice, Transcribe (voice studio UX audit, change 3): tasks, not tools. */

/** The three Studio tasks. */
export type StudioTask = "speak" | "clone" | "transcribe";

/**
 * What a caller may ask for. ``voices`` is the old name of ``clone`` (the Voices screen before the tasks were
 * renamed); it is still accepted from callers and from storage and always means ``clone``.
 */
export type StudioMode = StudioTask | "voices";

export const STUDIO_MODES: StudioTask[] = ["speak", "clone", "transcribe"];
export const DEFAULT_STUDIO_MODE: StudioTask = "speak";
export const STUDIO_MODE_KEY = "chaos.studio.voice-studio.mode";

/** A mode name, old or new, as one of the three tasks; null for anything else. */
export function toStudioTask(value: unknown): StudioTask | null {
  if (value === "voices") return "clone";
  return typeof value === "string" && (STUDIO_MODES as string[]).includes(value) ? (value as StudioTask) : null;
}

export function readStudioMode(): StudioTask {
  try {
    return toStudioTask(window.localStorage.getItem(STUDIO_MODE_KEY)) ?? DEFAULT_STUDIO_MODE;
  } catch {
    return DEFAULT_STUDIO_MODE;
  }
}

export function writeStudioMode(mode: StudioTask): void {
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

export function requestedMode(request: VoiceStudioRequest): StudioTask | null {
  if (request.id) return "transcribe";
  return toStudioTask(request.mode);
}
