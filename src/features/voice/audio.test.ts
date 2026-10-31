import { describe, expect, it } from "vitest";
import { concatChunks, encodeWav, formatTimestamp, levelOf, micErrorMessage, resample, toSrt, toVtt } from "./audio";

describe("resample", () => {
  it("returns a copy at the same rate", () => {
    const input = new Float32Array([0.1, 0.2]);
    const out = resample(input, 16000, 16000);
    expect(Array.from(out)).toEqual(Array.from(input));
    expect(out).not.toBe(input);
  });

  it("shrinks 48 kHz to a third of the samples and keeps a ramp a ramp", () => {
    const input = new Float32Array(4800).map((_, i) => i / 4800);
    const out = resample(input, 48000, 16000);
    expect(out.length).toBe(1600);
    for (let i = 1; i < out.length; i += 1) expect(out[i]).toBeGreaterThan(out[i - 1]);
  });
});

describe("encodeWav", () => {
  it("writes a 16-bit mono header and clamps samples", async () => {
    const blob = encodeWav(new Float32Array([0, 1, -1, 2]), 16000);
    const view = new DataView(await blob.arrayBuffer());
    const text = (offset: number) => String.fromCharCode(...Array.from({ length: 4 }, (_, i) => view.getUint8(offset + i)));
    expect(text(0)).toBe("RIFF");
    expect(text(8)).toBe("WAVE");
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint32(24, true)).toBe(16000);
    expect(view.getUint16(34, true)).toBe(16); // bits
    expect(view.getUint32(40, true)).toBe(8); // 4 samples x 2 bytes
    expect(view.getInt16(46, true)).toBe(32767);
    expect(view.getInt16(48, true)).toBe(-32768);
    expect(view.getInt16(50, true)).toBe(32767); // 2.0 clamped
    expect(blob.type).toBe("audio/wav");
  });
});

describe("helpers", () => {
  it("concatenates chunks in order", () => {
    expect(Array.from(concatChunks([new Float32Array([1, 2]), new Float32Array([3])]))).toEqual([1, 2, 3]);
  });

  it("levels silence at zero and caps loud input at one", () => {
    expect(levelOf(new Float32Array(10))).toBe(0);
    expect(levelOf(new Float32Array(10).fill(1))).toBe(1);
  });

  it("formats subtitles", () => {
    const segments = [
      { start: 0, end: 1.5, text: "Hello." },
      { start: 61.25, end: 3725.004, text: "Bye." },
    ];
    expect(formatTimestamp(3725.004)).toBe("01:02:05,004");
    expect(toSrt(segments)).toBe("1\n00:00:00,000 --> 00:00:01,500\nHello.\n\n2\n00:01:01,250 --> 01:02:05,004\nBye.\n");
    expect(toVtt(segments).startsWith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello.\n")).toBe(true);
  });

  it("explains a blocked microphone per platform", () => {
    const denied = new DOMException("denied", "NotAllowedError");
    expect(micErrorMessage(denied, "Darwin")).toContain("System Settings");
    expect(micErrorMessage(denied, "Windows")).toContain("Privacy & security");
    expect(micErrorMessage(new DOMException("none", "NotFoundError"), "Linux")).toContain("No microphone");
  });
});
