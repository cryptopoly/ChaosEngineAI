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
  queue = null;
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
  async play(ownerId: string, text: string, voice: string, speed: number): Promise<void> {
    const mine = begin(ownerId);
    const scheduler = makeScheduler(mine);
    try {
      await speakStream(text, voice, speed, scheduler.add, controller!.signal);
      await scheduler.flushed();
      scheduler.inputDone();
    } catch (error) {
      fail(mine, ownerId, error);
    }
  },

  /**
   * Speak a reply that is still being written: ``enqueue`` each piece as it
   * becomes speakable and ``endQueue`` when the reply is complete. Pieces
   * are synthesized in order while earlier ones play.
   */
  startQueue(ownerId: string, voice: string, speed: number): void {
    const mine = begin(ownerId);
    const scheduler = makeScheduler(mine);
    let synth: Promise<void> = Promise.resolve();
    queue = {
      enqueue(text: string) {
        synth = synth.then(async () => {
          if (mine !== generation || !controller) return;
          try {
            await speakStream(text, voice, speed, scheduler.add, controller.signal);
          } catch (error) {
            // A piece with nothing speakable in it (only markup) is skipped.
            if (!(error instanceof Error && error.message.includes("Nothing to read aloud"))) throw error;
          }
        });
        synth.catch((error) => fail(mine, ownerId, error));
      },
      end() {
        void synth
          .then(() => scheduler.flushed())
          .then(() => scheduler.inputDone())
          .catch(() => {});
      },
    };
  },

  enqueue(text: string): void {
    queue?.enqueue(text);
  },

  endQueue(): void {
    queue?.end();
    queue = null;
  },
};

interface Queue {
  enqueue(text: string): void;
  end(): void;
}

let queue: Queue | null = null;

function begin(ownerId: string): number {
  teardown();
  context = new AudioContext();
  controller = new AbortController();
  nextStart = context.currentTime + 0.05;
  set({ ...IDLE, status: "loading", ownerId });
  return generation;
}

function fail(mine: number, ownerId: string, error: unknown): void {
  if (mine !== generation) return; // replaced or stopped
  teardown();
  set({ ...IDLE, status: "error", ownerId, error: error instanceof Error ? error.message : String(error) });
}

/** Decodes and schedules sentences strictly in order, even if a later one decodes faster. */
function makeScheduler(mine: number) {
  let chain: Promise<void> = Promise.resolve();
  let scheduled = 0;
  let done = false;

  const finishIfDone = () => {
    if (mine === generation && done && sources.length === 0) {
      teardown();
      set(IDLE);
    }
  };

  const schedule = async (sentence: SpokenSentence) => {
    if (mine !== generation || !context) return;
    const buffer = await context.decodeAudioData(base64ToArrayBuffer(sentence.audio));
    if (mine !== generation || !context) return;
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const startAt = Math.max(nextStart, context.currentTime + 0.02);
    source.start(startAt);
    nextStart = startAt + buffer.duration;
    sources.push(source);
    scheduled += 1;
    if (scheduled === 1) set({ status: "playing", sentence: sentence.text, index: 0, total: queue ? 0 : sentence.total });
    source.onended = () => {
      sources = sources.filter((s) => s !== source);
      if (mine !== generation) return;
      finishIfDone();
    };
    // Show the sentence that is playing, not the one just decoded.
    const delayMs = Math.max(0, (startAt - context.currentTime) * 1000);
    window.setTimeout(() => {
      if (mine === generation && state.status !== "idle" && state.status !== "error")
        set({ sentence: sentence.text, index: state.total ? sentence.index : 0 });
    }, delayMs);
  };

  return {
    add(sentence: SpokenSentence) {
      chain = chain.then(() => schedule(sentence));
    },
    flushed: () => chain,
    inputDone() {
      done = true;
      finishIfDone();
    },
  };
}

export function useSpeechPlayer(): SpeechState {
  return useSyncExternalStore(
    speechPlayer.subscribe,
    speechPlayer.getState,
    speechPlayer.getState,
  );
}
