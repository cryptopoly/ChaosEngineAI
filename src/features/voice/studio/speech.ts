/**
 * Long-form speech for the Speak studio: the text is split into parts
 * (paragraphs, and ``[pause]`` markers), each part is synthesized on its own
 * so progress can be shown, and the WAVs are joined with a short silence.
 */

/** Silence between paragraphs, and for a ``[pause]`` marker. */
export const PARAGRAPH_GAP_S = 0.6;
export const PAUSE_S = 1.2;
/** Longer paragraphs are cut at sentence ends, so each request stays small. */
export const MAX_PART_CHARS = 4000;
const PAUSE_RE = /\[pause\]/gi;

export interface SpeechPart {
  /** Text to speak; empty for a pause. */
  text: string;
  /** Silence after this part, in seconds. */
  gapAfter: number;
}

/** Cut ``text`` at sentence ends into pieces of at most ``max`` characters. */
function cutLong(text: string, max: number): string[] {
  if (text.length <= max) return [text];
  const sentences = text.match(/[^.!?。！？]+[.!?。！？]*\s*/g) ?? [text];
  const out: string[] = [];
  let current = "";
  for (const sentence of sentences) {
    if (current && current.length + sentence.length > max) {
      out.push(current.trim());
      current = "";
    }
    current += sentence;
  }
  if (current.trim()) out.push(current.trim());
  // A single sentence longer than ``max`` is cut where it must be.
  return out.flatMap((piece) => (piece.length > max ? piece.match(new RegExp(`[\\s\\S]{1,${max}}`, "g")) ?? [] : [piece]));
}

export function splitSpeech(text: string): SpeechPart[] {
  const parts: SpeechPart[] = [];
  for (const paragraph of text.split(/\n\s*\n/)) {
    const pieces = paragraph.split(PAUSE_RE);
    pieces.forEach((piece, index) => {
      const clean = piece.trim();
      const pauseFollows = index < pieces.length - 1;
      if (clean) {
        const cut = cutLong(clean, MAX_PART_CHARS);
        cut.forEach((chunk, i) => parts.push({ text: chunk, gapAfter: i < cut.length - 1 ? 0 : pauseFollows ? PAUSE_S : PARAGRAPH_GAP_S }));
      } else if (pauseFollows && parts.length > 0) {
        const last = parts[parts.length - 1];
        last.gapAfter = Math.round((last.gapAfter + PAUSE_S) * 100) / 100;
      }
    });
  }
  if (parts.length > 0) parts[parts.length - 1].gapAfter = 0;
  return parts;
}

interface Pcm {
  sampleRate: number;
  samples: Int16Array;
}

/** Mono PCM16 samples from a WAV (as the voice worker writes them). */
export function readWav(buffer: ArrayBuffer): Pcm {
  const view = new DataView(buffer);
  const tag = (offset: number) => String.fromCharCode(...new Uint8Array(buffer, offset, 4));
  if (tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("Not a WAV file.");
  let sampleRate = 0;
  let offset = 12;
  while (offset + 8 <= buffer.byteLength) {
    const id = tag(offset);
    const size = view.getUint32(offset + 4, true);
    if (id === "fmt ") {
      const channels = view.getUint16(offset + 10, true);
      const bits = view.getUint16(offset + 22, true);
      if (channels !== 1 || bits !== 16) throw new Error("Only mono 16-bit WAV is supported.");
      sampleRate = view.getUint32(offset + 12, true);
    } else if (id === "data") {
      const length = Math.min(size, buffer.byteLength - offset - 8);
      const samples = new Int16Array(buffer.slice(offset + 8, offset + 8 + length - (length % 2)));
      return { sampleRate, samples };
    }
    offset += 8 + size + (size % 2);
  }
  throw new Error("The WAV file has no audio data.");
}

/** One WAV from several, with ``gaps[i]`` seconds of silence after part ``i``. */
export function joinWavs(parts: ArrayBuffer[], gaps: number[]): { blob: Blob; durationS: number } {
  const pcm = parts.map(readWav);
  const sampleRate = pcm[0]?.sampleRate ?? 24_000;
  if (pcm.some((p) => p.sampleRate !== sampleRate)) throw new Error("The parts have different sample rates.");
  const gapSamples = gaps.map((g) => Math.round(Math.max(0, g) * sampleRate));
  const total = pcm.reduce((sum, p, i) => sum + p.samples.length + (gapSamples[i] ?? 0), 0);
  const out = new Int16Array(total);
  let at = 0;
  pcm.forEach((p, i) => {
    out.set(p.samples, at);
    at += p.samples.length + (gapSamples[i] ?? 0);
  });
  const header = new DataView(new ArrayBuffer(44));
  const write = (offset: number, text: string) => [...text].forEach((c, i) => header.setUint8(offset + i, c.charCodeAt(0)));
  write(0, "RIFF");
  header.setUint32(4, 36 + out.byteLength, true);
  write(8, "WAVE");
  write(12, "fmt ");
  header.setUint32(16, 16, true);
  header.setUint16(20, 1, true);
  header.setUint16(22, 1, true);
  header.setUint32(24, sampleRate, true);
  header.setUint32(28, sampleRate * 2, true);
  header.setUint16(32, 2, true);
  header.setUint16(34, 16, true);
  write(36, "data");
  header.setUint32(40, out.byteLength, true);
  return { blob: new Blob([header.buffer, out.buffer], { type: "audio/wav" }), durationS: total / sampleRate };
}
