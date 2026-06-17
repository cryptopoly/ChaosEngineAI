/**
 * Voice I/O API endpoints — STT transcription and TTS synthesis.
 *
 * Re-exported from ``./index`` alongside the other domain modules.
 */

import { apiFetch, fetchJson } from "./index";
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
