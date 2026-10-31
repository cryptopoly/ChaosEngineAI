/**
 * Turn detection for conversation mode.
 *
 * Fed one 32 ms frame of 16 kHz audio at a time with its Silero speech
 * probability. A turn starts after ``minSpeechMs`` of speech and ends after
 * ``silenceMs`` below the lower threshold; the audio handed back includes
 * ``preRollMs`` from before the start so the first syllable is not cut.
 */

import { concatChunks } from "../audio";

export const FRAME_SAMPLES = 512;
export const FRAME_MS = 32;

export interface EndpointerOptions {
  /** Probability that counts as speech. */
  threshold: number;
  /** Speech continues until the probability drops below this. */
  negThreshold: number;
  minSpeechMs: number;
  silenceMs: number;
  preRollMs: number;
  /** A turn longer than this ends on its own. */
  maxTurnMs: number;
}

export const DEFAULT_ENDPOINTER: EndpointerOptions = {
  threshold: 0.5,
  negThreshold: 0.35,
  minSpeechMs: 160,
  silenceMs: 700,
  preRollMs: 320,
  maxTurnMs: 60_000,
};

export type EndpointEvent = { type: "start" } | { type: "end"; audio: Float32Array };

export class Endpointer {
  private options: EndpointerOptions;
  private speaking = false;
  private recent: Float32Array[] = [];
  private turn: Float32Array[] = [];
  private speechRun = 0;
  private silenceRun = 0;

  constructor(options: Partial<EndpointerOptions> = {}) {
    this.options = { ...DEFAULT_ENDPOINTER, ...options };
  }

  get inTurn(): boolean {
    return this.speaking;
  }

  update(options: Partial<EndpointerOptions>): void {
    this.options = { ...this.options, ...options };
  }

  reset(): void {
    this.speaking = false;
    this.recent = [];
    this.turn = [];
    this.speechRun = 0;
    this.silenceRun = 0;
  }

  push(frame: Float32Array, probability: number): EndpointEvent | null {
    const { threshold, negThreshold, minSpeechMs, silenceMs, preRollMs, maxTurnMs } = this.options;
    if (!this.speaking) {
      const minFrames = Math.max(1, Math.ceil(minSpeechMs / FRAME_MS));
      const keep = minFrames + Math.ceil(preRollMs / FRAME_MS);
      this.recent.push(frame);
      if (this.recent.length > keep) this.recent.shift();
      this.speechRun = probability >= threshold ? this.speechRun + 1 : 0;
      if (this.speechRun < minFrames) return null;
      this.speaking = true;
      this.turn = this.recent;
      this.recent = [];
      this.silenceRun = 0;
      return { type: "start" };
    }

    this.turn.push(frame);
    this.silenceRun = probability < negThreshold ? this.silenceRun + 1 : 0;
    const ended =
      this.silenceRun * FRAME_MS >= silenceMs || this.turn.length * FRAME_MS >= maxTurnMs;
    if (!ended) return null;
    const audio = concatChunks(this.turn);
    this.reset();
    return { type: "end", audio };
  }
}
