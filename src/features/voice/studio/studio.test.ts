import { describe, expect, it } from "vitest";
import type { TranscriptSegment } from "../../../types";
import { formatTimestamp } from "../audio";
import { MAX_PART_CHARS, joinWavs, readWav, splitSpeech } from "./speech";
import {
  editSegment,
  exportTranscript,
  joinedText,
  segmentAt,
  speakerNamer,
  speakersIn,
  toPlainText,
  wordAt,
} from "./transcript";

const SEGMENTS: TranscriptSegment[] = [
  { start: 0, end: 1.5, text: "Hello.", speaker: 1, words: [{ start: 0.1, end: 0.6, text: "Hello." }] },
  { start: 1.6, end: 3, text: "How are you?", speaker: 1 },
  { start: 3.2, end: 4, text: "Fine.", speaker: 0 },
];
const name = speakerNamer({ "0": "Ana" }, (n) => `Speaker ${n + 1}`);
const ITEM = { title: "Call", language: "en", model: "m", durationS: 4, speakers: { "0": "Ana" }, text: "" };

function wav(samples: number[], rate = 24_000): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset: number, text: string) => [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => view.setInt16(44 + i * 2, s, true));
  return buffer;
}

describe("transcript helpers", () => {
  it("names speakers, falling back to a number, in order of appearance", () => {
    expect(speakersIn(SEGMENTS)).toEqual([1, 0]);
    expect([name(0), name(1)]).toEqual(["Ana", "Speaker 2"]);
  });

  it("joins consecutive lines by one speaker into one paragraph", () => {
    expect(toPlainText(SEGMENTS, name)).toBe("Speaker 2: Hello. How are you?\n\nAna: Fine.");
    expect(toPlainText([{ start: 0, end: 1, text: "Just text." }], name)).toBe("Just text.");
    expect(toPlainText([], name, "fallback")).toBe("fallback");
  });

  it("exports every format with speaker names", () => {
    expect(exportTranscript("srt", ITEM, SEGMENTS, name).text).toContain("00:00:03,200 --> 00:00:04,000\nAna: Fine.");
    expect(exportTranscript("vtt", ITEM, SEGMENTS, name).text).toContain("<v Speaker 2>Hello.");
    expect(exportTranscript("md", ITEM, SEGMENTS, name).text).toBe("# Call\n\n**Speaker 2** `0:00` Hello. How are you?\n\n**Ana** `0:03` Fine.\n");
    const json = JSON.parse(exportTranscript("json", ITEM, SEGMENTS, name).text);
    expect(json.speakers).toEqual({ "1": "Speaker 2", "0": "Ana" });
    expect(json.segments).toHaveLength(3);
    // No speaker labels: plain cues.
    expect(exportTranscript("srt", ITEM, [{ start: 0, end: 1, text: "Hi" }], name).text).toBe("1\n00:00:00,000 --> 00:00:01,000\nHi\n");
  });

  it("keeps a segment's times when its text is edited, and drops stale word timings", () => {
    const edited = editSegment(SEGMENTS[0], "  Hello   there. ");
    expect(edited).toEqual({ start: 0, end: 1.5, text: "Hello there.", speaker: 1 });
    expect(editSegment(SEGMENTS[0], "Hello.")).toBe(SEGMENTS[0]);
  });

  it("finds the line and word playing at a time", () => {
    expect([segmentAt(SEGMENTS, 0.2), segmentAt(SEGMENTS, 1.55), segmentAt(SEGMENTS, 3.5), segmentAt(SEGMENTS, 9)]).toEqual([0, 1, 2, -1]);
    expect([wordAt(SEGMENTS[0], 0.3), wordAt(SEGMENTS[0], 1.2)]).toEqual([0, -1]);
  });

  it("joins text without spaces for languages written without them", () => {
    const parts = [
      { start: 0, end: 1, text: "你好" },
      { start: 1, end: 2, text: "再见" },
    ];
    expect([joinedText(parts, "zh"), joinedText(parts, "en")]).toEqual(["你好再见", "你好 再见"]);
  });

  it("rounds timestamps to whole milliseconds without overflowing", () => {
    expect(formatTimestamp(1.9996)).toBe("00:00:02,000");
  });
});

describe("speech parts", () => {
  it("splits paragraphs and [pause] markers into parts with gaps", () => {
    expect(splitSpeech("One.\n\nTwo. [pause] Three.\n\n[PAUSE]\n\nFour.")).toEqual([
      { text: "One.", gapAfter: 0.6 },
      { text: "Two.", gapAfter: 1.2 },
      { text: "Three.", gapAfter: 1.8 },
      { text: "Four.", gapAfter: 0 },
    ]);
    expect(splitSpeech("  \n\n ")).toEqual([]);
  });

  it("cuts a long paragraph at sentence ends, and a huge sentence where it must", () => {
    const sentence = "word ".repeat(150).trim() + ". ";
    const parts = splitSpeech(sentence.repeat(12));
    expect(parts.length).toBeGreaterThan(1);
    expect(parts.every((p) => p.text.length <= MAX_PART_CHARS && p.text.endsWith("."))).toBe(true);
    const huge = splitSpeech("x".repeat(MAX_PART_CHARS * 2 + 5));
    expect(huge.map((p) => p.text.length)).toEqual([MAX_PART_CHARS, MAX_PART_CHARS, 5]);
  });

  it("joins WAVs with silence between them", async () => {
    const { blob, durationS } = joinWavs([wav([1, 2]), wav([3])], [0.0001, 0]);
    const joined = readWav(await blob.arrayBuffer());
    // 0.0001 s at 24 kHz rounds to 2 samples of silence.
    expect(Array.from(joined.samples)).toEqual([1, 2, 0, 0, 3]);
    expect(joined.sampleRate).toBe(24_000);
    expect(durationS).toBeCloseTo(5 / 24_000);
    expect(() => joinWavs([wav([1], 24_000), wav([1], 16_000)], [0, 0])).toThrow("sample rates");
    expect(() => readWav(new ArrayBuffer(12))).toThrow();
  });
});
