/**
 * Transcript helpers for the Studio editor and the Library: speaker names,
 * text and file exports, editing a segment, finding what plays at a time,
 * and the prompts that hand a transcript to chat.
 */

import type { VoiceLibraryItem, TranscriptSegment } from "../../../types";
import { formatClock, toSrt, toVtt, type SpeakerNamer } from "../audio";

export function speakerNamer(names: Record<string, string>, fallback: (speaker: number) => string): SpeakerNamer {
  return (speaker) => names[String(speaker)]?.trim() || fallback(speaker);
}

/** Speaker numbers in order of first appearance. */
export function speakersIn(segments: TranscriptSegment[]): number[] {
  const seen: number[] = [];
  for (const segment of segments) {
    if (segment.speaker != null && !seen.includes(segment.speaker)) seen.push(segment.speaker);
  }
  return seen;
}

interface Block {
  speaker: number | null;
  start: number;
  text: string;
}

/** Consecutive segments by the same speaker, joined. */
function blocks(segments: TranscriptSegment[]): Block[] {
  const out: Block[] = [];
  for (const segment of segments) {
    const speaker = segment.speaker ?? null;
    const last = out[out.length - 1];
    if (last && last.speaker === speaker) last.text = `${last.text} ${segment.text}`.trim();
    else out.push({ speaker, start: segment.start, text: segment.text });
  }
  return out;
}

/** Plain text: one paragraph per speaker turn ("Name: ..."), or the text as is without speakers. */
export function toPlainText(segments: TranscriptSegment[], name: SpeakerNamer, fallbackText = ""): string {
  if (segments.length === 0) return fallbackText;
  if (speakersIn(segments).length === 0) return segments.map((s) => s.text).join(" ").trim();
  return blocks(segments)
    .map((b) => (b.speaker == null ? b.text : `${name(b.speaker)}: ${b.text}`))
    .join("\n\n");
}

export function toMarkdown(title: string, segments: TranscriptSegment[], name: SpeakerNamer, fallbackText = ""): string {
  const heading = `# ${title}\n\n`;
  if (segments.length === 0) return `${heading}${fallbackText}\n`;
  const body = blocks(segments)
    .map((b) => {
      const who = b.speaker == null ? "" : `**${name(b.speaker)}** `;
      return `${who}\`${formatClock(b.start)}\` ${b.text}`;
    })
    .join("\n\n");
  return `${heading}${body}\n`;
}

export function toJson(item: Pick<VoiceLibraryItem, "title" | "language" | "model" | "durationS" | "speakers">, segments: TranscriptSegment[], name: SpeakerNamer): string {
  return JSON.stringify(
    {
      title: item.title,
      language: item.language ?? null,
      model: item.model ?? null,
      durationS: item.durationS ?? null,
      speakers: Object.fromEntries(speakersIn(segments).map((n) => [String(n), name(n)])),
      segments,
    },
    null,
    2,
  );
}

export type ExportFormat = "txt" | "md" | "srt" | "vtt" | "json";

export function exportTranscript(
  format: ExportFormat,
  item: Pick<VoiceLibraryItem, "title" | "language" | "model" | "durationS" | "speakers" | "text">,
  segments: TranscriptSegment[],
  name: SpeakerNamer,
): { text: string; type: string } {
  const named = speakersIn(segments).length > 0 ? name : undefined;
  switch (format) {
    case "md":
      return { text: toMarkdown(item.title, segments, name, item.text), type: "text/markdown" };
    case "srt":
      return { text: toSrt(segments, named), type: "application/x-subrip" };
    case "vtt":
      return { text: toVtt(segments, named), type: "text/vtt" };
    case "json":
      return { text: toJson(item, segments, name), type: "application/json" };
    default:
      return { text: toPlainText(segments, name, item.text), type: "text/plain" };
  }
}

const NO_SPACE_LANGUAGES = new Set(["zh", "ja", "ko", "th", "lo", "my"]);

/** The whole transcript as one text, without speaker names. */
export function joinedText(segments: TranscriptSegment[], language?: string | null): string {
  const separator = NO_SPACE_LANGUAGES.has((language ?? "").split("-")[0]) ? "" : " ";
  return segments.map((s) => s.text).join(separator).trim();
}

/** Replace a segment's text; its times stay, word timings go when the words changed. */
export function editSegment(segment: TranscriptSegment, text: string): TranscriptSegment {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean === segment.text) return segment;
  const { words: _words, ...rest } = segment;
  return { ...rest, text: clean };
}

/** Index of the segment playing at ``time`` (the last one started), or -1. */
export function segmentAt(segments: TranscriptSegment[], time: number): number {
  let found = -1;
  for (let i = 0; i < segments.length; i += 1) {
    if (segments[i].start <= time + 0.05) found = i;
    else break;
  }
  if (found >= 0 && time > segments[found].end + 1) return -1;
  return found;
}

export function wordAt(segment: TranscriptSegment, time: number): number {
  const words = segment.words ?? [];
  for (let i = 0; i < words.length; i += 1) {
    if (time >= words[i].start - 0.02 && time < words[i].end + 0.02) return i;
  }
  return -1;
}

export type HandOff = "summary" | "actions" | "notes";

export const HAND_OFF_PROMPTS: Record<HandOff, string> = {
  summary: "Summarise this transcript in a few short paragraphs.",
  actions: "List the action items in this transcript: who does what, and by when if it is said.",
  notes: "Write meeting notes for this transcript: attendees, topics discussed, decisions, and action items.",
};
