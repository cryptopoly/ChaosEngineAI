/**
 * Voice I/O API endpoints — STT transcription and TTS synthesis.
 *
 * Re-exported from ``./index`` alongside the other domain modules.
 */

import { apiFetch, fetchJson, postJson, readErrorDetail } from "./index";
import type {
  VoiceLibraryItem,
  VoiceLibraryKind,
  VoiceLibraryPatch,
  VoiceLibrarySummary,
  TranscriptSegment,
  VoiceInstallJob,
  VoiceRuntime,
  VoiceSettings,
  VoiceEntry,
  VoiceEntryKind,
  VoiceModelRow,
  SpeakBody,
  SpeakResult,
} from "../types";

export async function getVoiceRuntime(): Promise<VoiceRuntime> {
  return await fetchJson<VoiceRuntime>("/api/voice/runtime", 20000);
}

export interface TranscribeResult {
  text: string;
  language: string | null;
  segments: TranscriptSegment[];
  durationS: number | null;
  model: string;
  task?: "transcribe" | "translate";
  seconds: number | null;
  /** Set when a setting was changed or nothing was recognised, with the reason. */
  note?: string | null;
  /** Set when the request asked to save to the Library. */
  libraryId?: string;
}

export interface TranscribeOptions {
  model?: string;
  language?: string;
  signal?: AbortSignal;
  /** "translate" gives English text (Whisper models only). */
  task?: "transcribe" | "translate";
  words?: boolean;
  speakers?: boolean;
  /** 0 or unset: work out how many people speak. */
  speakerCount?: number;
  /** Keep the file and transcript in the Library. */
  save?: boolean;
}

export async function transcribeAudio(formData: FormData, signal?: AbortSignal): Promise<TranscribeResult> {
  const response = await apiFetch("/api/voice/transcribe", {
    method: "POST",
    body: formData,
    signal,
  });
  if (!response.ok) {
    throw new Error(await readErrorDetail(response, `Transcription failed with status ${response.status}`));
  }
  return (await response.json()) as TranscribeResult;
}

/** Transcribe one recorded or chosen file with the user's model and language. */
export async function transcribeBlob(
  blob: Blob,
  filename: string,
  options: TranscribeOptions = {},
): Promise<TranscribeResult> {
  const form = new FormData();
  form.append("audio", blob, filename);
  if (options.model) form.append("model", options.model);
  if (options.language) form.append("language", options.language);
  if (options.task) form.append("task", options.task);
  if (options.words) form.append("words", "true");
  if (options.speakers) form.append("speakers", "true");
  if (options.speakerCount) form.append("speakerCount", String(options.speakerCount));
  if (options.save) form.append("save", "true");
  return await transcribeAudio(form, options.signal);
}

export async function synthesizeSpeech(text: string, voice: string, speed: number): Promise<Blob> {
  const response = await apiFetch("/api/voice/synthesize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice, speed }),
  });
  if (!response.ok) {
    const errText = await response.text().catch(() => `Status ${response.status}`);
    throw new Error(errText || `Synthesis failed with status ${response.status}`);
  }
  return await response.blob();
}

export interface SpokenSentence {
  index: number;
  total: number;
  text: string;
  /** Base64 WAV. */
  audio: string;
  durationS: number;
  /** The voice that spoke it (the Kokoro voice after a fallback). */
  voice?: string;
}

/** Where read-aloud speaks: ``read`` (chat replies) or ``call`` (a voice call; cloned voices are refused). */
export type SpeechUse = "read" | "call";

export interface SpeakStreamOptions {
  use?: SpeechUse;
  /** A one-line note from the backend, e.g. the chosen voice could not speak and the Kokoro voice reads instead. */
  onNote?: (note: string) => void;
}

/**
 * Read text aloud with any listed voice (D-72): the backend cleans markdown, splits sentences for the voice's
 * model and sends each one as soon as it is synthesized. ``onSentence`` sees them in order.
 */
export async function speakStream(
  text: string,
  voice: string,
  speed: number,
  onSentence: (sentence: SpokenSentence) => void,
  signal?: AbortSignal,
  options: SpeakStreamOptions = {},
): Promise<void> {
  const use = options.use && options.use !== "read" ? { use: options.use } : {};
  const response = await apiFetch("/api/voice/speak/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice, speed, ...use }),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(await readErrorDetail(response, `Read aloud failed with status ${response.status}`));
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      newline = buffer.indexOf("\n");
      if (!line) continue;
      const event = JSON.parse(line) as Partial<SpokenSentence> & { error?: string; done?: boolean; note?: string };
      if (event.error) throw new Error(event.error);
      if (event.done) return;
      if (event.note) {
        options.onNote?.(event.note);
        continue;
      }
      onSentence(event as SpokenSentence);
    }
  }
}

/** Silero VAD speech probability per 32 ms frame of 16 kHz PCM16 (base64). */
export async function detectSpeech(session: string, pcm: string, reset = false): Promise<number[]> {
  const result = await postJson<{ probs: number[] }>("/api/voice/vad", { session, pcm, reset }, 30000);
  return result.probs;
}

export async function getVoiceSettings(): Promise<VoiceSettings> {
  return await fetchJson<VoiceSettings>("/api/voice/settings", 15000);
}

export async function saveVoiceSettings(patch: Partial<VoiceSettings>): Promise<VoiceSettings> {
  return await postJson<VoiceSettings>("/api/voice/settings", patch);
}

export async function startVoiceInstall(extras: string[] = []): Promise<VoiceInstallJob> {
  return await postJson<VoiceInstallJob>("/api/voice/install", { extras });
}

export async function getVoiceInstallStatus(): Promise<VoiceInstallJob> {
  return await fetchJson<VoiceInstallJob>("/api/voice/install/status", 15000);
}

export async function cancelVoiceInstall(): Promise<VoiceInstallJob> {
  return await postJson<VoiceInstallJob>("/api/voice/install/cancel", {});
}

export async function stopVoiceWorker(): Promise<void> {
  await postJson<{ stopped: boolean }>("/api/voice/worker/stop", {});
}

export async function getLibrary(query = "", kind: "" | VoiceLibraryKind = ""): Promise<VoiceLibrarySummary[]> {
  const params = new URLSearchParams();
  if (query) params.set("q", query);
  if (kind) params.set("kind", kind);
  const suffix = params.toString() ? `?${params}` : "";
  const result = await fetchJson<{ items: VoiceLibrarySummary[] }>(`/api/voice/library${suffix}`, 20000);
  return result.items;
}

export async function getLibraryItem(id: string): Promise<VoiceLibraryItem> {
  return await fetchJson<VoiceLibraryItem>(`/api/voice/library/${id}`, 20000);
}

export async function updateLibraryItem(id: string, patch: VoiceLibraryPatch): Promise<VoiceLibraryItem> {
  return await postJson<VoiceLibraryItem>(`/api/voice/library/${id}`, patch);
}

export async function deleteLibraryItems(ids: string[]): Promise<string[]> {
  const result = await postJson<{ deleted: string[] }>("/api/voice/library/delete", { ids });
  return result.deleted;
}

/** Save generated speech (a WAV) to the Library. */
export async function saveSpeech(
  audio: Blob,
  fields: { text: string; voice: string; speed: number; title?: string; voiceName?: string; voiceKind?: VoiceEntryKind },
): Promise<VoiceLibraryItem> {
  const form = new FormData();
  form.append("audio", audio, "speech.wav");
  form.append("text", fields.text);
  form.append("voice", fields.voice);
  form.append("speed", String(fields.speed));
  if (fields.title) form.append("title", fields.title);
  if (fields.voiceName) form.append("voiceName", fields.voiceName);
  if (fields.voiceKind) form.append("voiceKind", fields.voiceKind);
  const response = await apiFetch("/api/voice/library/speech", { method: "POST", body: form });
  if (!response.ok) {
    throw new Error(await readErrorDetail(response, `Saving failed with status ${response.status}`));
  }
  return (await response.json()) as VoiceLibraryItem;
}

/** An item's audio; ``mp3`` converts it first. */
export async function getLibraryAudio(id: string, format: "" | "mp3" = ""): Promise<Blob> {
  const response = await apiFetch(`/api/voice/library/${id}/audio${format ? `?format=${format}` : ""}`);
  if (!response.ok) {
    throw new Error(await readErrorDetail(response, `Loading audio failed with status ${response.status}`));
  }
  return await response.blob();
}

/** Convert audio (Speak studio output) to MP3. */
export async function encodeMp3(audio: Blob): Promise<Blob> {
  const form = new FormData();
  form.append("audio", audio, "speech.wav");
  form.append("format", "mp3");
  const response = await apiFetch("/api/voice/encode", { method: "POST", body: form });
  if (!response.ok) {
    throw new Error(await readErrorDetail(response, `MP3 export failed with status ${response.status}`));
  }
  return await response.blob();
}

/** Speak with any voice (built-in, preset, designed or cloned); the backend picks the model from the voice. */
export async function speakVoice(body: SpeakBody, signal?: AbortSignal): Promise<SpeakResult> {
  // Loading a model and speaking can take a few minutes the first time.
  return await postJson<SpeakResult>("/api/voice/speak", body, 900000, signal);
}

/** Stop the Speak screen's running speech job on the backend (the fetch abort alone leaves it running). */
export async function cancelSpeak(): Promise<void> {
  try {
    await postJson<{ cancelled: boolean }>("/api/voice/speak/cancel", {}, 15000);
  } catch {
    // The job may already be done; nothing to stop.
  }
}

export async function getVoices(): Promise<VoiceEntry[]> {
  return (await fetchJson<{ voices: VoiceEntry[] }>("/api/voice/voices", 20000)).voices;
}

export async function getVoiceModels(): Promise<{ models: VoiceModelRow[]; available: boolean }> {
  return await fetchJson<{ models: VoiceModelRow[]; available: boolean }>("/api/voice/models", 15000);
}
