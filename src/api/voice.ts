/**
 * Voice I/O API endpoints — STT transcription and TTS synthesis.
 *
 * Re-exported from ``./index`` alongside the other domain modules.
 */

import { apiFetch, deleteJson, fetchJson, postJson } from "./index";
import type { VoiceRuntime } from "../types";

export async function getVoiceRuntime(): Promise<VoiceRuntime> {
  return await fetchJson<VoiceRuntime>("/api/voice/runtime", 20000);
}

export interface TranscribeResult {
  text: string;
  duration_s: number;
}

export async function transcribeAudio(formData: FormData): Promise<TranscribeResult> {
  const response = await apiFetch("/api/voice/transcribe", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const text = await response.text().catch(() => `Status ${response.status}`);
    throw new Error(text || `Transcription failed with status ${response.status}`);
  }
  return (await response.json()) as TranscribeResult;
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

export interface KokoroDownloadStatus {
  state: "idle" | "downloading" | "completed" | "failed";
  progress: number;
  error: string | null;
  installed?: boolean;
}

/** Start the kokoro-onnx voice-file download (non-Apple TTS lane). */
export async function startKokoroDownload(): Promise<KokoroDownloadStatus> {
  return await postJson<KokoroDownloadStatus>("/api/voice/kokoro/download", {});
}

export async function getKokoroDownloadStatus(): Promise<KokoroDownloadStatus> {
  return await fetchJson<KokoroDownloadStatus>("/api/voice/kokoro/download/status", 15000);
}

export async function getGalleryAudio(id: string): Promise<Blob> {
  const response = await apiFetch(`/api/voice/gallery/${id}/audio`);
  if (!response.ok) {
    throw new Error(`Failed to load audio (status ${response.status})`);
  }
  return await response.blob();
}
