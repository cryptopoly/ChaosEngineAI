/**
 * Microphone capture and audio helpers for Voice.
 *
 * The microphone is recorded as raw samples (AudioWorklet, with a
 * ScriptProcessor fallback for older WebKitGTK builds), resampled to 16 kHz
 * mono and sent as a WAV file. Every webview the app runs in (WKWebView,
 * WebView2, WebKitGTK) supports that, and the backend never has to guess a
 * MediaRecorder container format.
 */

import type { TranscriptSegment } from "../../types";

export const STT_SAMPLE_RATE = 16_000;
/** A recording that would exceed this is stopped automatically. */
export const MAX_RECORDING_SECONDS = 15 * 60;

/** Linear-interpolation resample of mono samples to ``outputRate``. */
export function resample(input: Float32Array, inputRate: number, outputRate = STT_SAMPLE_RATE): Float32Array {
  if (inputRate === outputRate || input.length === 0) return input.slice();
  const ratio = inputRate / outputRate;
  const length = Math.max(1, Math.floor(input.length / ratio));
  const output = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const position = i * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const fraction = position - left;
    output[i] = input[left] * (1 - fraction) + input[right] * fraction;
  }
  return output;
}

/**
 * Linear-interpolation resampler for audio that arrives in small blocks.
 * It keeps the fractional read position and the last sample between
 * blocks, so 128-sample worklet blocks resample without drift or clicks.
 */
export class StreamResampler {
  private position = 0;
  private previous = 0;

  constructor(private readonly inputRate: number, private readonly outputRate = STT_SAMPLE_RATE) {}

  push(block: Float32Array): Float32Array {
    if (this.inputRate === this.outputRate) return block.slice();
    const step = this.inputRate / this.outputRate;
    // Index -1 is the last sample of the previous block.
    const at = (index: number) => (index < 0 ? this.previous : block[index]);
    const out: number[] = [];
    while (this.position <= block.length - 1) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      out.push(at(left) * (1 - fraction) + at(left + 1) * fraction);
      this.position += step;
    }
    if (block.length > 0) this.previous = block[block.length - 1];
    this.position -= block.length;
    return Float32Array.from(out);
  }
}

/** 16-bit PCM mono WAV. */
export function encodeWav(samples: Float32Array, sampleRate = STT_SAMPLE_RATE): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

export function concatChunks(chunks: Float32Array[]): Float32Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

/** RMS level of a block, scaled to 0..1 for a meter. */
export function levelOf(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) sum += samples[i] * samples[i];
  return Math.min(1, Math.sqrt(sum / samples.length) * 4);
}

// ---------------------------------------------------------------------------
// Microphone
// ---------------------------------------------------------------------------

export interface MicCapture {
  /** Stop recording and return a 16 kHz mono WAV. */
  stop(): Promise<Blob>;
  /** Stop recording and drop the audio. */
  cancel(): void;
  readonly startedAt: number;
}

export interface MicCaptureOptions {
  onLevel?: (level: number) => void;
  /** Called when the recording hits MAX_RECORDING_SECONDS. */
  onLimit?: () => void;
  deviceId?: string;
}

const WORKLET_SOURCE = `
class ChaosCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("chaos-capture", ChaosCapture);
`;

/** Explain a getUserMedia failure in words a user can act on. */
export function micErrorMessage(error: unknown, platform: string): string {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    if (platform === "Darwin") {
      return "Microphone access is blocked. Allow ChaosEngineAI in System Settings > Privacy & Security > Microphone, then try again.";
    }
    if (platform === "Windows") {
      return "Microphone access is blocked. Allow desktop apps to use the microphone in Settings > Privacy & security > Microphone, then try again.";
    }
    return "Microphone access is blocked. Check your desktop's privacy settings and that no other app holds the microphone.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No microphone was found. Connect one and try again.";
  }
  if (name === "NotReadableError") {
    return "The microphone is in use by another app or could not be opened.";
  }
  if (typeof navigator !== "undefined" && !navigator.mediaDevices?.getUserMedia) {
    return "This window cannot use the microphone (no media device support).";
  }
  return error instanceof Error ? error.message : "Could not open the microphone.";
}

interface MicGraph {
  sampleRate: number;
  teardown(): void;
}

/** Open the microphone and call ``onBlock`` with raw blocks at the device rate. */
async function openMic(onBlock: (block: Float32Array, sampleRate: number) => void, deviceId?: string): Promise<MicGraph> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  const context = new AudioContext();
  const source = context.createMediaStreamSource(stream);
  let finished = false;
  const guarded = (block: Float32Array) => {
    if (!finished) onBlock(block, context.sampleRate);
  };

  let node: AudioNode;
  let workletUrl: string | null = null;
  try {
    workletUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "application/javascript" }));
    await context.audioWorklet.addModule(workletUrl);
    const worklet = new AudioWorkletNode(context, "chaos-capture");
    worklet.port.onmessage = (event: MessageEvent<Float32Array>) => guarded(event.data);
    node = worklet;
  } catch {
    // Older WebKitGTK: no AudioWorklet.
    const processor = context.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (event) => guarded(new Float32Array(event.inputBuffer.getChannelData(0)));
    node = processor;
  }
  // A muted sink keeps the graph pulling audio without playing it back.
  const sink = context.createGain();
  sink.gain.value = 0;
  source.connect(node);
  node.connect(sink);
  sink.connect(context.destination);

  return {
    sampleRate: context.sampleRate,
    teardown() {
      finished = true;
      stream.getTracks().forEach((track) => track.stop());
      try {
        source.disconnect();
        node.disconnect();
      } catch {
        /* already disconnected */
      }
      void context.close().catch(() => {});
      if (workletUrl) URL.revokeObjectURL(workletUrl);
    },
  };
}

export async function startMicCapture(options: MicCaptureOptions = {}): Promise<MicCapture> {
  const chunks: Float32Array[] = [];
  let samples = 0;
  let maxSamples = Infinity;
  const startedAt = Date.now();
  const graph = await openMic((block) => {
    chunks.push(block);
    samples += block.length;
    options.onLevel?.(levelOf(block));
    if (samples >= maxSamples) options.onLimit?.();
  }, options.deviceId);
  maxSamples = MAX_RECORDING_SECONDS * graph.sampleRate;

  return {
    startedAt,
    async stop() {
      graph.teardown();
      return encodeWav(resample(concatChunks(chunks), graph.sampleRate));
    },
    cancel() {
      graph.teardown();
    },
  };
}

export interface MicStream {
  stop(): void;
}

/** Live microphone audio as 16 kHz mono blocks, for conversation mode. */
export async function startMicStream(options: {
  onAudio: (samples: Float32Array) => void;
  onLevel?: (level: number) => void;
  deviceId?: string;
}): Promise<MicStream> {
  let resampler: StreamResampler | null = null;
  const graph = await openMic((block, rate) => {
    resampler ??= new StreamResampler(rate);
    options.onLevel?.(levelOf(block));
    options.onAudio(resampler.push(block));
  }, options.deviceId);
  return { stop: () => graph.teardown() };
}

/** Float samples as base64 little-endian PCM16. */
export function pcm16Base64(samples: Float32Array): string {
  const bytes = new Uint8Array(samples.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
  }
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

// ---------------------------------------------------------------------------
// Transcript export
// ---------------------------------------------------------------------------

function pad(value: number, width = 2): string {
  return String(Math.floor(value)).padStart(width, "0");
}

export function formatTimestamp(seconds: number, separator: "," | "." = ","): string {
  const safe = Math.max(0, seconds);
  const ms = Math.round((safe % 1) * 1000);
  return `${pad(safe / 3600)}:${pad((safe % 3600) / 60)}:${pad(safe % 60)}${separator}${pad(ms, 3)}`;
}

export function toSrt(segments: TranscriptSegment[]): string {
  return segments
    .map((s, i) => `${i + 1}\n${formatTimestamp(s.start)} --> ${formatTimestamp(s.end)}\n${s.text}\n`)
    .join("\n");
}

export function toVtt(segments: TranscriptSegment[]): string {
  const cues = segments.map((s) => `${formatTimestamp(s.start, ".")} --> ${formatTimestamp(s.end, ".")}\n${s.text}\n`);
  return `WEBVTT\n\n${cues.join("\n")}`;
}

export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
