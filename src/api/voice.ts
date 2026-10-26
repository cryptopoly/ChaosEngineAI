/**
 * Voice I/O API endpoints — STT transcription and TTS synthesis.
 *
 * Re-exported from ``./index`` alongside the other domain modules.
 */

import { apiFetch, deleteJson, fetchJson, postJson, readErrorDetail } from "./index";
import type { TranscriptSegment, VoiceInstallJob, VoiceRuntime, VoiceSettings } from "../types";

export async function getVoiceRuntime(): Promise<VoiceRuntime> {
  return await fetchJson<VoiceRuntime>("/api/voice/runtime", 20000);
}

export interface TranscribeResult {
  text: string;
  language: string | null;
  segments: TranscriptSegment[];
  durationS: number | null;
  model: string;
  seconds: number | null;
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
  options: { model?: string; language?: string; signal?: AbortSignal } = {},
): Promise<TranscribeResult> {
  const form = new FormData();
  form.append("audio", blob, filename);
  if (options.model) form.append("model", options.model);
  if (options.language) form.append("language", options.language);
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
}

/**
 * Read text aloud: the backend cleans markdown, splits sentences and sends
 * each one as soon as it is synthesized. ``onSentence`` sees them in order.
 */
export async function speakStream(
  text: string,
  voice: string,
  speed: number,
  onSentence: (sentence: SpokenSentence) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await apiFetch("/api/voice/speak/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice, speed }),
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
      const event = JSON.parse(line) as Partial<SpokenSentence> & { error?: string; done?: boolean };
      if (event.error) throw new Error(event.error);
      if (event.done) return;
      onSentence(event as SpokenSentence);
    }
  }
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

export interface VoiceGalleryItem {
  id: string;
  kind: "transcript" | "audio";
  createdAt: number;
  text: string;
  voice?: string;
}

export async function getVoiceGallery(): Promise<VoiceGalleryItem[]> {
  const result = await fetchJson<{ items: VoiceGalleryItem[] }>("/api/voice/gallery", 20000);
  return result.items;
}

export async function saveGalleryTranscript(text: string): Promise<VoiceGalleryItem> {
  return await postJson<VoiceGalleryItem>("/api/voice/gallery/transcript", { text });
}

async function blobToBase64(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  let binary = "";
  for (const byte of new Uint8Array(buffer)) binary += String.fromCharCode(byte);
  return btoa(binary);
}

export async function saveGalleryAudio(text: string, voice: string, audio: Blob): Promise<VoiceGalleryItem> {
  const audioBase64 = await blobToBase64(audio);
  return await postJson<VoiceGalleryItem>("/api/voice/gallery/audio", { text, voice, audioBase64 });
}

export async function deleteGalleryItem(id: string): Promise<{ deleted: boolean }> {
  return await deleteJson<{ deleted: boolean }>(`/api/voice/gallery/${id}`);
}

export async function getGalleryAudio(id: string): Promise<Blob> {
  const response = await apiFetch(`/api/voice/gallery/${id}/audio`);
  if (!response.ok) {
    throw new Error(`Failed to load audio (status ${response.status})`);
  }
  return await response.blob();
}
