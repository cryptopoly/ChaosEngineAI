import { describe, expect, it } from "vitest";
import { StreamResampler } from "../audio";
import { Endpointer, FRAME_SAMPLES } from "./endpointer";
import { SpeechTextTracker } from "./speechText";

const frame = (value = 0) => new Float32Array(FRAME_SAMPLES).fill(value);

describe("Endpointer", () => {
  it("starts after the minimum speech and ends after the silence, keeping the pre-roll", () => {
    const ep = new Endpointer({ minSpeechMs: 64, silenceMs: 96, preRollMs: 64 });
    const events: string[] = [];
    let audio: Float32Array | null = null;
    // 3 quiet frames, 4 speech frames, 3 silent frames.
    const probs = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1];
    probs.forEach((p, i) => {
      const event = ep.push(frame(i), p);
      if (event) events.push(event.type);
      if (event?.type === "end") audio = event.audio;
    });
    expect(events).toEqual(["start", "end"]);
    // Pre-roll (2 frames) + 4 speech + 3 silence = 9 frames, starting at frame 1.
    expect(audio!.length).toBe(9 * FRAME_SAMPLES);
    expect(audio![0]).toBe(1);
    expect(ep.inTurn).toBe(false);
  });

  it("ignores a blip shorter than the minimum speech", () => {
    const ep = new Endpointer({ minSpeechMs: 96 });
    const events = [0.9, 0.9, 0.1, 0.9, 0.1].map((p) => ep.push(frame(), p)).filter(Boolean);
    expect(events).toEqual([]);
  });

  it("keeps the turn going through dips above the lower threshold", () => {
    const ep = new Endpointer({ minSpeechMs: 32, silenceMs: 64, negThreshold: 0.35 });
    const types = [0.9, 0.4, 0.4, 0.4, 0.2, 0.2].map((p) => ep.push(frame(), p)?.type ?? null);
    expect(types).toEqual(["start", null, null, null, null, "end"]);
  });

  it("ends a turn that runs past the maximum length", () => {
    const ep = new Endpointer({ minSpeechMs: 32, maxTurnMs: 96 });
    const types = [0.9, 0.9, 0.9].map((p) => ep.push(frame(), p)?.type ?? null);
    expect(types).toEqual(["start", null, "end"]);
  });
});

describe("SpeechTextTracker", () => {
  it("hands over whole sentences as the reply grows", () => {
    const t = new SpeechTextTracker();
    expect(t.next("Hello there")).toEqual([]);
    expect(t.next("Hello there. How are")).toEqual(["Hello there. "]);
    expect(t.next("Hello there. How are you? I am")).toEqual(["How are you? "]);
    expect(t.next("Hello there. How are you? I am fine", true)).toEqual(["I am fine"]);
    expect(t.next("Hello there. How are you? I am fine", true)).toEqual([]);
  });

  it("does not split numbers, abbreviations before lower case, or open code fences", () => {
    const t = new SpeechTextTracker();
    expect(t.next("Version 3.5 is out, e.g. today")).toEqual([]);
    expect(t.next("Try this:\n```\nprint(1). X = 2\n")).toEqual([]);
    const done = t.next("Try this:\n```\nprint(1). X = 2\n```\n\nDone. Next");
    expect(done).toEqual(["Try this:\n```\nprint(1). X = 2\n```\n\nDone. "]);
  });

  it("waits for a thinking block to close and handles CJK punctuation", () => {
    const t = new SpeechTextTracker();
    expect(t.next("<think>Plan. Then</think")).toEqual([]);
    expect(t.next("<think>Plan. Then</think>你好。再见")).toEqual(["<think>Plan. Then</think>你好。"]);
  });
});

describe("StreamResampler", () => {
  it("resamples 128-sample blocks without losing samples", () => {
    const r = new StreamResampler(48_000, 16_000);
    let total = 0;
    for (let i = 0; i < 375; i += 1) total += r.push(new Float32Array(128).fill(0.5)).length;
    // 48,000 samples in, 16,000 out (within one sample).
    expect(Math.abs(total - 16_000)).toBeLessThanOrEqual(1);
  });

  it("keeps a ramp continuous across block boundaries at 44.1 kHz", () => {
    const r = new StreamResampler(44_100, 16_000);
    const out: number[] = [];
    for (let block = 0; block < 10; block += 1) {
      const input = Float32Array.from({ length: 128 }, (_, i) => block * 128 + i);
      out.push(...r.push(input));
    }
    for (let i = 1; i < out.length; i += 1) {
      expect(out[i] - out[i - 1]).toBeCloseTo(44_100 / 16_000, 3);
    }
  });
});
