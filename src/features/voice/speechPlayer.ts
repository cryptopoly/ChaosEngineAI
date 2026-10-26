/**
 * Read-aloud player shared by the whole app: one voice speaks at a time.
 *
 * Sentences arrive from ``/api/voice/speak/stream`` as they are synthesized
 * and are scheduled back to back on one AudioContext, so speech starts after
 * the first sentence instead of the whole reply. Components subscribe with
 * ``useSpeechPlayer``.
 */

import { useSyncExternalStore } from "react";
import { speakStream, type SpokenSentence } from "../../api";

export type SpeechStatus = "idle" | "loading" | "playing" | "paused" | "error";

export interface SpeechState {
  status: SpeechStatus;
  /** Who started it (a message id, "studio", ...). */
  ownerId: string | null;
  sentence: string;
  index: number;
  total: number;
  error: string | null;
}

const IDLE: SpeechState = {
  status: "idle",
  ownerId: null,
  sentence: "",
  index: 0,
  total: 0,
  error: null,
};

let state: SpeechState = IDLE;
const listeners = new Set<() => void>();
let context: AudioContext | null = null;
let controller: AbortController | null = null;
let sources: AudioBufferSourceNode[] = [];
let nextStart = 0;
let generation = 0;

function set(patch: Partial<SpeechState>): void {
  state = { ...state, ...patch };
  listeners.forEach((listener) => listener());
}

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function teardown(): void {
  generation += 1;
  controller?.abort();
  controller = null;
  for (const source of sources) {
    source.onended = null;
    try {
      source.stop();
    } catch {
      /* not started */
    }
  }
  sources = [];
  void context?.close().catch(() => {});
  context = null;
}

export const speechPlayer = {
  getState(): SpeechState {
    return state;
  },

  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },

  stop(): void {
    teardown();
    set(IDLE);
  },

  togglePause(): void {
    if (!context) return;
    if (state.status === "playing") {
      void context.suspend();
      set({ status: "paused" });
    } else if (state.status === "paused") {
      void context.resume();
      set({ status: "playing" });
    }
  },

  /** Speak ``text``; a second call (any owner) replaces the first. */
  async play(
    ownerId: string,
    text: string,
    voice: string,
    speed: number,
  ): Promise<void> {
    teardown();
    const mine = generation;
    context = new AudioContext();
    controller = new AbortController();
    nextStart = context.currentTime + 0.05;
    let scheduled = 0;
    let streamDone = false;
    set({ ...IDLE, status: "loading", ownerId });

    const finishIfDone = () => {
      if (mine === generation && streamDone && sources.length === 0) {
        teardown();
        set(IDLE);
      }
    };

    // Decode and schedule strictly in order, even if a later sentence
    // decodes faster than an earlier one.
    let chain: Promise<void> = Promise.resolve();
    const schedule = async (sentence: SpokenSentence) => {
      if (mine !== generation || !context) return;
      const buffer = await context.decodeAudioData(
        base64ToArrayBuffer(sentence.audio),
      );
      if (mine !== generation || !context) return;
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      const startAt = Math.max(nextStart, context.currentTime + 0.02);
      source.start(startAt);
      nextStart = startAt + buffer.duration;
      sources.push(source);
      scheduled += 1;
      if (scheduled === 1)
        set({
          status: "playing",
          sentence: sentence.text,
          index: 0,
          total: sentence.total,
        });
      source.onended = () => {
        sources = sources.filter((s) => s !== source);
        if (mine !== generation) return;
        const next = sentence.index + 1;
        if (next < sentence.total) set({ index: next });
        finishIfDone();
      };
      // Show the sentence that is playing, not the one just decoded.
      const delayMs = Math.max(0, (startAt - context.currentTime) * 1000);
      window.setTimeout(() => {
        if (mine === generation && state.status !== "idle")
          set({ sentence: sentence.text, index: sentence.index });
      }, delayMs);
    };

    try {
      await speakStream(
        text,
        voice,
        speed,
        (sentence) => {
          chain = chain.then(() => schedule(sentence));
        },
        controller.signal,
      );
      await chain;
      streamDone = true;
      finishIfDone();
    } catch (error) {
      if (mine !== generation) return; // replaced or stopped
      teardown();
      set({
        ...IDLE,
        status: "error",
        ownerId,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  },
};

export function useSpeechPlayer(): SpeechState {
  return useSyncExternalStore(
    speechPlayer.subscribe,
    speechPlayer.getState,
    speechPlayer.getState,
  );
}
