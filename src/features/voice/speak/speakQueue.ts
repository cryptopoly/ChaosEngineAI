/**
 * Speech shares the one heavy-job slot with video, image and music jobs. While a Speak request waits for it, the
 * screen asks the backend what it waits for and says so, instead of showing a still progress bar for minutes.
 */
import { useEffect, useState } from "react";
import { fetchJson } from "../../../api";

export interface SpeakQueueState {
  waiting: boolean;
  position: number;
  agentAhead: boolean;
  runningKind: string | null;
}

const POLL_MS = 2000;

/** The i18n key (under `voice.speak.queue`) for the line shown while waiting, or null when not waiting. */
export function speakQueueLineKey(state: SpeakQueueState | null): "video" | "image" | "music" | "other" | null {
  if (!state?.waiting) return null;
  const kind = state.runningKind;
  return kind === "video" || kind === "image" || kind === "music" ? kind : "other";
}

/** Polls `/api/voice/speak/queue` every 2 s while `active` (a Speak request is in flight). */
export function useSpeakQueue(active: boolean): SpeakQueueState | null {
  const [state, setState] = useState<SpeakQueueState | null>(null);
  useEffect(() => {
    if (!active) {
      setState(null);
      return;
    }
    let stopped = false;
    const poll = async () => {
      try {
        const next = await fetchJson<SpeakQueueState>("/api/voice/speak/queue", 5000);
        if (!stopped) setState(next);
      } catch {
        // An older backend has no queue route; the progress bar alone stays.
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), POLL_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [active]);
  return state;
}
