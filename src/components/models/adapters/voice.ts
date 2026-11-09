/**
 * Speech models -> Models page entries (Voice > Models, D-73 M3b).
 *
 * One row per catalog speech model from ``GET /api/voice/models``, plus Kokoro as a pinned first row: it has no
 * catalog row (it is part of the voice runtime and cannot be deleted on its own) but is a model a person compares
 * with the others, and the voice Chat falls back to. The row keeps the catalog object in ``source``; the page reads
 * capabilities, licence and measured cost from it. ``deleteTarget`` and ``downloadKey`` are the catalog key.
 */

import type { DownloadStatus } from "../../../api";
import type { VoiceCapabilities, VoiceFamily, VoiceModelRow } from "../../../types";
import { readFit, type Fit } from "../fit";
import type { ModelEntry } from "../modelEntry";

/** The pinned Kokoro row's id. */
export const KOKORO_ROW_ID = "kokoro";

/** What a speech model makes, as chips. */
export type VoiceMakes = "quick" | "natural" | "style" | "clone" | "design" | "tags" | "expression" | "languages";

export interface KokoroSource {
  kind: "kokoro";
  /** Quick voices this machine can offer, and in how many languages. */
  voices: number;
  languages: number;
  sizeGb: number | null;
  peakGb: number | null;
  installed: boolean;
}

export type VoiceModelSource = VoiceModelRow | KokoroSource;
export type VoiceModelEntry = ModelEntry<VoiceModelSource>;

/** Family display names (upstream model names) in the order the table lists them. */
export const VOICE_FAMILY_ORDER: Array<{ id: string; name: string }> = [
  { id: "kokoro", name: "Kokoro" },
  { id: "qwen3-tts", name: "Qwen3-TTS" },
  { id: "chatterbox", name: "Chatterbox" },
  { id: "voxcpm2", name: "VoxCPM2" },
  { id: "orpheus", name: "Orpheus" },
];

/** Chatterbox-Turbo and Chatterbox Multilingual share one family heading. */
export function voiceFamilyId(family: VoiceFamily | undefined): string {
  if (!family) return "qwen3-tts";
  return family === "chatterbox-turbo" ? "chatterbox" : family;
}

export function voiceFamilyRank(familyId: string | null | undefined): number {
  const index = VOICE_FAMILY_ORDER.findIndex((family) => family.id === familyId);
  return index < 0 ? VOICE_FAMILY_ORDER.length : index;
}

export function isKokoro(source: VoiceModelSource): source is KokoroSource {
  return (source as KokoroSource).kind === "kokoro";
}

/** The Makes chips of a catalog row, from its capabilities. */
export function voiceMakes(row: Pick<VoiceModelRow, "capabilities" | "kind">): VoiceMakes[] {
  const caps: VoiceCapabilities = row.capabilities;
  const makes: VoiceMakes[] = [];
  if (caps.presets || row.kind === "custom") makes.push("natural");
  if (caps.design) makes.push("design");
  if (caps.style && !caps.design) makes.push("style");
  if (caps.tags) makes.push("tags");
  if (caps.expression) makes.push("expression");
  if (caps.languageSelect) makes.push("languages");
  if (caps.clone) makes.push("clone");
  return makes;
}

function gbLabel(gb: number): string {
  return `${gb.toFixed(2)} GB`;
}

function memoryOf(peak: number | null | undefined, fit: Fit | null) {
  if (peak == null) return null;
  return { label: `~${peak >= 10 ? peak.toFixed(0) : peak.toFixed(1)} GB peak`, title: fit?.reason ?? null, peakGb: peak };
}

export function voiceRowToEntry(row: VoiceModelRow): VoiceModelEntry {
  const fit = readFit(row.fit);
  const familyId = voiceFamilyId(row.family);
  const size = row.downloadGb ?? row.sizeGb;
  return {
    id: row.key,
    area: "voice",
    name: row.name,
    provider: row.licence?.name ?? "",
    familyId,
    familyName: VOICE_FAMILY_ORDER.find((family) => family.id === familyId)?.name ?? null,
    note: row.summary || null,
    repo: row.repo,
    link: row.licence?.url ?? null,
    downloadKey: row.key,
    deleteTarget: row.key,
    availableLocally: row.installed,
    hasLocalData: row.installed,
    // Rows stay visible where they cannot run; the row says why and Download is off.
    hostCompatible: true,
    license: row.licence?.name ?? null,
    capabilities: voiceMakes(row),
    sizeGb: size,
    sizeLabel: gbLabel(size),
    sizeTitle: row.downloadGb && row.downloadGb !== row.sizeGb ? `${row.sizeGb} GB model + companions` : null,
    memory: memoryOf(row.estimates?.peakGb, fit),
    releaseSortKey: "",
    fit,
    shared: null,
    source: row,
  };
}

export function kokoroEntry(source: Omit<KokoroSource, "kind">, fit: unknown): VoiceModelEntry {
  const parsed = readFit(fit);
  return {
    id: KOKORO_ROW_ID,
    area: "voice",
    name: "Kokoro",
    provider: "Apache 2.0",
    familyId: "kokoro",
    familyName: "Kokoro",
    note: null,
    repo: "",
    downloadKey: KOKORO_ROW_ID,
    deleteTarget: KOKORO_ROW_ID,
    availableLocally: source.installed,
    hasLocalData: source.installed,
    // Kokoro comes with the runtime: never downloaded or deleted on its own.
    installedByRuntime: source.installed,
    hostCompatible: true,
    license: "Apache 2.0",
    capabilities: ["quick"],
    sizeGb: source.sizeGb,
    sizeLabel: source.sizeGb != null ? gbLabel(source.sizeGb) : null,
    memory: memoryOf(source.peakGb, parsed),
    releaseSortKey: "",
    fit: parsed,
    shared: null,
    source: { kind: "kokoro", ...source },
  };
}

/** The rows of the table: Kokoro first (when the runtime is installed), then every catalog row. */
export function buildVoiceEntries(models: VoiceModelRow[], kokoro: VoiceModelEntry | null): VoiceModelEntry[] {
  return [...(kokoro ? [kokoro] : []), ...models.map(voiceRowToEntry)];
}

/** A download state from the catalog row, in the shape the shared status ladder reads. */
export function voiceDownloadStatus(entry: VoiceModelEntry): DownloadStatus | null {
  const source = entry.source;
  if (isKokoro(source)) return null;
  const state = source.download.state;
  if (state === "downloading") {
    const { progress, downloadedGb, totalGb } = source.download;
    return { repo: source.repo, state: "downloading", progress: progress ?? 0, downloadedGb: downloadedGb ?? 0, totalGb: totalGb ?? entry.sizeGb ?? null, error: null };
  }
  if (state === "failed" && !source.installed) {
    return { repo: source.repo, state: "failed", progress: 0, downloadedGb: 0, totalGb: entry.sizeGb ?? null, error: source.download.error };
  }
  return null;
}

/** Search text: name, family, licence, summary and what it makes. */
export function voiceEntryMatchesQuery(entry: VoiceModelEntry, query: string, makesLabels: Record<VoiceMakes, string>): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  const source = entry.source;
  const haystack = [
    entry.name,
    entry.familyName,
    entry.license,
    entry.note,
    isKokoro(source) ? "quick voices kokoro" : source.languages,
    ...(entry.capabilities ?? []).map((kind) => makesLabels[kind as VoiceMakes] ?? kind),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(needle);
}
