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
  download: { state: "idle" | "downloading" | "done" | "failed"; error: string | null };
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
  consent: { speakerName: string; basis: "own" | "permission"; consentedAt: string; appVersion: string; code: string };
}

export interface ConsentChallenge {
  id: string;
  code: string;
  sentence: string;
  speakerName: string;
  basis: "own" | "permission";
}

export async function getCloneModels(): Promise<CloneModels> {
  return await fetchJson<CloneModels>("/api/voice/clone/models", 15000);
}

export async function downloadCloneModel(key: string): Promise<CloneModel["download"]> {
  return await postJson("/api/voice/clone/models/download", { key });
}

export async function listClonedVoices(): Promise<ClonedVoice[]> {
  return (await fetchJson<{ voices: ClonedVoice[] }>("/api/voice/clone/voices", 15000)).voices;
}

export async function startConsent(speakerName: string, basis: "own" | "permission"): Promise<ConsentChallenge> {
  return await postJson<ConsentChallenge>("/api/voice/clone/consent", { speakerName, basis });
}

export async function createClonedVoice(fields: {
  challengeId: string;
  name: string;
  consent: Blob;
  reference?: File | null;
  referenceText?: string;
}): Promise<ClonedVoice> {
  const form = new FormData();
  form.append("consent", fields.consent, "consent.wav");
  form.append("challengeId", fields.challengeId);
  form.append("name", fields.name);
  // The user ticked the statement; the panel cannot get this far otherwise.
  form.append("statementAccepted", "true");
  if (fields.reference) form.append("reference", fields.reference, fields.reference.name);
  if (fields.referenceText) form.append("referenceText", fields.referenceText);
  const response = await apiFetch("/api/voice/clone/voices", { method: "POST", body: form });
  if (!response.ok) throw new Error(await readErrorDetail(response, `Saving the voice failed with status ${response.status}`));
  return (await response.json()) as ClonedVoice;
}

export async function deleteClonedVoice(id: string): Promise<void> {
  const response = await apiFetch(`/api/voice/clone/voices/${id}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await readErrorDetail(response, `Deleting failed with status ${response.status}`));
}

export async function speakCloned(voiceId: string, text: string, model: string, disclaimer: boolean): Promise<VoiceLibraryItem> {
  // Loading the model and speaking can take a few minutes the first time.
  return await postJson<VoiceLibraryItem>("/api/voice/clone/speak", { voiceId, text, model, disclaimer }, 900000);
}
