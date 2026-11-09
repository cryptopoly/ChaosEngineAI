/** Voice cloning endpoints (``/api/voice/clone/*``). */

import { apiFetch, fetchJson, postJson, readErrorDetail } from "../../../api";
import type { VoiceLibraryItem } from "../../../types";

export interface CloneModel {
  key: string;
  name: string;
  summary: string;
  repo: string;
  revision: string;
  sizeGb: number;
  languages: string;
  default: boolean;
  licence: { id: string; name: string; url: string };
  supported: boolean;
  installed: boolean;
  download: { state: "idle" | "downloading" | "done" | "failed"; error: string | null; phase?: "weights" | "watermark" | null };
}

export interface CloneModels {
  models: CloneModel[];
  available: boolean;
  allowedHere: boolean;
}

export interface ClonedVoice {
  id: string;
  name: string;
  createdAt: number;
  durationS: number | null;
  referenceText: string;
  /** The permission record: who, on what basis, when, and the app version. */
  consent: { speakerName: string; basis: "own" | "permission"; consentedAt: string; appVersion: string };
}

export async function getCloneModels(): Promise<CloneModels> {
  return await fetchJson<CloneModels>("/api/voice/clone/models", 15000);
}

export async function downloadCloneModel(key: string): Promise<CloneModel["download"]> {
  return await postJson("/api/voice/clone/models/download", { key });
}

/** What deleting a speech model removes and keeps (companions another downloaded model still uses). */
export interface SpeechModelDeletePlan {
  key: string;
  name: string;
  remove: string[];
  kept: Array<{ repo: string; usedBy: string[] }>;
  /** The watermark component goes too (no other downloaded model can clone). */
  watermark: boolean;
  sizeGb: number;
}

export async function getSpeechModelDeletePlan(key: string): Promise<SpeechModelDeletePlan> {
  return await fetchJson<SpeechModelDeletePlan>(`/api/voice/clone/models/delete-plan?key=${encodeURIComponent(key)}`, 30000);
}

/** Delete a speech model's files (refused with 409 while a voice model is busy or a download runs). */
export async function deleteSpeechModel(key: string): Promise<{ freedGb: number }> {
  return await postJson("/api/voice/clone/models/delete", { key }, 120000);
}

export async function listClonedVoices(): Promise<ClonedVoice[]> {
  return (await fetchJson<{ voices: ClonedVoice[] }>("/api/voice/clone/voices", 15000)).voices;
}

export async function createClonedVoice(fields: {
  speakerName: string;
  basis: "own" | "permission";
  name: string;
  /** An uploaded file, or the microphone recording (WAV). */
  sample: Blob;
  sampleName: string;
  referenceText?: string;
}): Promise<ClonedVoice> {
  const form = new FormData();
  form.append("sample", fields.sample, fields.sampleName);
  form.append("speakerName", fields.speakerName);
  form.append("basis", fields.basis);
  form.append("name", fields.name);
  // The user ticked the permission box; the form cannot be saved otherwise.
  form.append("permissionConfirmed", "true");
  if (fields.referenceText) form.append("referenceText", fields.referenceText);
  const response = await apiFetch("/api/voice/clone/voices", { method: "POST", body: form });
  if (!response.ok) throw new Error(await readErrorDetail(response, `Saving the voice failed with status ${response.status}`));
  return (await response.json()) as ClonedVoice;
}

export async function deleteClonedVoice(id: string): Promise<void> {
  const response = await apiFetch(`/api/voice/clone/voices/${id}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await readErrorDetail(response, `Deleting failed with status ${response.status}`));
}

export async function speakCloned(voiceId: string, text: string, model: string): Promise<VoiceLibraryItem> {
  // Loading the model and speaking can take a few minutes the first time.
  return await postJson<VoiceLibraryItem>("/api/voice/clone/speak", { voiceId, text, model }, 900000);
}
