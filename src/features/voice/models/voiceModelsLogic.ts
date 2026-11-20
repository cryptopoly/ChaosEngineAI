/** Pure rules for Voice > Models (D-73): delete confirms, family headings, sort options. */

import i18next from "i18next";
import { fitSortRank, statusSortRank, type ModelSortOption } from "../../../components/models";
import { isKokoro, voiceFamilyRank, type VoiceModelEntry } from "../../../components/models/adapters/voice";
import type { SttModel, VoiceEntry, VoiceModelRow, VoiceRuntime } from "../../../types";
import type { SpeechModelDeletePlan } from "../clone/cloneApi";
import { DESIGN_VOICE_ID, bestNaturalVoice } from "../speak/speakLogic";
import type { StudioMode } from "../voiceStudioMode";

/**
 * Where a model row's "Use in Speak" lands: a preset voice of that model in the UI language, the design voice for a
 * design model, or Voices (Browse voices) for a model that only clones.
 */
export function studioTargetFor(model: VoiceModelRow, voices: VoiceEntry[], uiLanguage: string): { mode: StudioMode; voice?: string } {
  const presets = voices.filter((v) => v.kind === "preset" && v.models.includes(model.key));
  const voice = bestNaturalVoice(presets, uiLanguage) ?? presets[0];
  if (voice) return { mode: "speak", voice: voice.id };
  if (model.kind === "design" || model.capabilities.design) return { mode: "speak", voice: DESIGN_VOICE_ID };
  if (model.capabilities.clone) return { mode: "voices" };
  return { mode: "speak" };
}

type T = (key: string, options?: Record<string, unknown>) => string;

const tStudio: T = (key, options) => i18next.t(`studio:${key}`, options) as string;

/**
 * A PC whose NVIDIA GPU could run the speech models, with the speech GPU add-on not installed (the backend lists the
 * add-on only where it can install). Until it is ticked, no speech model beyond Kokoro runs there.
 */
export function speechGpuAddonMissing(runtime: Pick<VoiceRuntime, "lane" | "extras"> | null | undefined): boolean {
  return runtime?.lane === "portable" && (runtime.extras ?? []).some((extra) => extra.id === "speech-gpu" && !extra.installed);
}

/** Where to get the speech GPU add-on, for a PC that lacks it. */
export function speechGpuAddonText(t: T = tStudio): string {
  return t("voice.speak.needsSpeechGpu", {
    defaultValue: "Needs the speech GPU add-on. Tick it under Voice > Models > Voice runtime > Add-ons.",
  });
}

/**
 * Why a row cannot run here: the GPU's memory or age (the backend sends the numbers), the missing speech GPU add-on
 * for a GPU row on a PC, else the Apple-only note.
 */
export function unsupportedModelText(
  row: Pick<VoiceModelRow, "unsupportedFacts" | "lanes">,
  runtime: Pick<VoiceRuntime, "lane" | "extras"> | null | undefined,
  t: T = tStudio,
): string {
  const facts = row.unsupportedFacts;
  if (facts?.kind === "memory") {
    return t("voice.speak.gpuTooSmall", {
      need: facts.need,
      have: facts.have,
      defaultValue: "Needs about {need} GB of GPU memory; this GPU has {have} GB.",
    });
  }
  if (facts?.kind === "compute") {
    return t("voice.speak.gpuTooOld", {
      need: facts.need,
      have: facts.have,
      defaultValue: "Needs an NVIDIA GPU of compute capability {need} or newer; this one is {have}.",
    });
  }
  if ((row.lanes ?? []).includes("cuda") && speechGpuAddonMissing(runtime)) return speechGpuAddonText(t);
  return t("voice.speak.appleOnly", { defaultValue: "Apple Silicon Macs only for now." });
}

/**
 * A speech model that only runs on an Apple Silicon Mac, seen on a PC (D-102 A): no GPU lane and not on disk. Such
 * rows are folded under one closed line instead of filling the table with rows that can only say no.
 */
export function macOnlyOnThisPc(row: Pick<VoiceModelRow, "lanes" | "installed">, lane: string | null | undefined): boolean {
  return lane === "portable" && Array.isArray(row.lanes) && !row.lanes.includes("cuda") && !row.installed;
}

/**
 * What a PC shows (D-102 A, P1): the models it can run, with "(GPU)" dropped from their names once the Mac twins are
 * folded away, and the voices that one of those models speaks (cloned and quick voices always stay). Elsewhere every
 * row and voice is kept and ``macOnly`` is empty.
 */
export function splitForLane(
  models: VoiceModelRow[],
  voices: VoiceEntry[],
  lane: string | null | undefined,
): { models: VoiceModelRow[]; voices: VoiceEntry[]; macOnly: VoiceModelRow[]; hiddenVoices: number } {
  const macOnly = models.filter((row) => macOnlyOnThisPc(row, lane));
  if (macOnly.length === 0) return { models, voices, macOnly, hiddenVoices: 0 };
  const hidden = new Set(macOnly.map((row) => row.key));
  const shown = models
    .filter((row) => !hidden.has(row.key))
    .map((row) => ({ ...row, name: row.name.replace(/\s*\(GPU\)$/, "") }));
  const keep = (voice: VoiceEntry) =>
    voice.kind === "clone" || voice.kind === "builtin" || voice.models.length === 0 || voice.models.some((key) => !hidden.has(key));
  const kept = voices.filter(keep);
  return { models: shown, voices: kept, macOnly, hiddenVoices: voices.length - kept.length };
}

/** The confirm before a speech model is deleted: what goes, its size, what stays because another model uses it. */
export function speechDeleteConfirmText(plan: SpeechModelDeletePlan, sizeGb: number | null | undefined, t: T = tStudio): string {
  const gb = plan.sizeGb > 0 ? plan.sizeGb : (sizeGb ?? 0);
  const lines = [
    t("voice.models.delete.confirmSpeech", {
      defaultValue: "Delete {name} ({gb} GB) from this computer? Download it again to use it. Cloned voices and Library items stay.",
      name: plan.name,
      gb: gb.toFixed(2),
    }),
  ];
  for (const kept of plan.kept) {
    lines.push(
      t("voice.models.delete.keptShared", {
        defaultValue: "Kept: {repo}, still used by {names}.",
        repo: kept.repo,
        names: kept.usedBy.join(", "),
      }),
    );
  }
  if (plan.watermark) {
    lines.push(
      t("voice.models.delete.watermarkGoes", {
        defaultValue: "The watermark component goes too: no other downloaded model clones. It comes back with the next cloning model.",
      }),
    );
  }
  return lines.join("\n\n");
}

/**
 * The GPU dictation row that will become the default once it is downloaded: the first GPU row (the backend lists the
 * default-to-be first), while it is not on disk and this GPU can run it. Undefined on a Mac or without the add-on.
 */
export function gpuDefaultPending(models: SttModel[]): SttModel | undefined {
  const first = models.find((m) => m.backend === "transformers-asr");
  return first && !first.installed && !first.gpuFacts ? first : undefined;
}

// i18n-extract: voice.models.licenceName.Apache-2_0
// i18n-extract: voice.models.licenceName.MIT
// i18n-extract: voice.models.licenceName.CC-BY-4_0
// i18n-extract: voice.models.licenceName.OpenMDW-1_1
const LICENCE_NAMES: Record<string, string> = {
  "Apache-2.0": "Apache 2.0",
  MIT: "MIT",
  "CC-BY-4.0": "Creative Commons BY 4.0 (credit the author)",
  "OpenMDW-1.1": "OpenMDW 1.1 (open model licence)",
};

/** A readable licence name for a dictation card ("OpenMDW-1.1" -> "OpenMDW 1.1 (open model licence)"); unknown ids as sent. */
export function licenceLabel(id: string | null | undefined, t: T = tStudio): string | null {
  if (!id) return null;
  return LICENCE_NAMES[id] ? t(`voice.models.licenceName.${id.replace(/\./g, "_")}`, { defaultValue: LICENCE_NAMES[id] }) : id;
}

/** Where a dictation model's licence is read: its Hugging Face page, which carries the licence text. */
export function licenceUrl(repo: string): string {
  return `https://huggingface.co/${repo}`;
}

// i18n-extract: voice.models.sttGuide.qwen3-asr-1_7b
// i18n-extract: voice.models.sttGuide.qwen3-asr-0_6b
// i18n-extract: voice.models.sttGuide.parakeet-tdt-v3
// i18n-extract: voice.models.sttGuide.nemotron-3_5-asr
// i18n-extract: voice.models.sttGuide.whisper-large-v3-turbo
// i18n-extract: voice.models.sttGuide.whisper-small
// i18n-extract: voice.models.sttGuide.whisper-base
// i18n-extract: voice.models.sttGuide.granite-speech-5_0-2b
// i18n-extract: voice.models.sttGuide.nemotron-3_5-transcribe-30b
/** One line per dictation model on what it is for (D-102 D, P2), by catalog key. */
export const STT_GUIDES: Record<string, string> = {
  "qwen3-asr-1.7b": "The pick for an NVIDIA GPU: 30 languages and 22 Chinese dialects, finds the language itself.",
  "qwen3-asr-0.6b": "A smaller Qwen3-ASR: less GPU memory and quicker, for the same languages.",
  "parakeet-tdt-v3": "Very fast for 25 European languages; finds the language itself and times each word on its own.",
  "nemotron-3.5-asr": "Small and quick, made for live speech in 40 languages; uses the Language setting or finds it.",
  "whisper-large-v3-turbo": "Works on any machine in 99 languages and times each word; does not translate.",
  "whisper-small": "Small and quick in 99 languages; one of the two that can translate speech into English.",
  "whisper-base": "The smallest: quick on any machine and can translate into English, but the least accurate.",
  "granite-speech-5.0-2b": "Six languages, English first; finds the language itself.",
  "nemotron-3.5-transcribe-30b": "The largest, 40 languages; for a GPU with about 80 GB of memory.",
};

export function sttGuideText(key: string | undefined, t: T = tStudio): string | null {
  if (!key || !STT_GUIDES[key]) return null;
  return t(`voice.models.sttGuide.${key.replace(/\./g, "_")}`, { defaultValue: STT_GUIDES[key] });
}

/**
 * The fit chip of a GPU dictation row that this GPU can run: the GPU memory it needs, saying whether the figure was
 * measured (an RTX 4090 run) or is an estimate from the weights. Null for rows without a figure (Whisper, the Mac).
 */
export function sttFitText(model: Pick<SttModel, "minVramGb" | "vramBasis" | "gpuFacts">, t: T = tStudio): { text: string; title: string } | null {
  if (model.minVramGb == null || model.gpuFacts) return null;
  const measured = model.vramBasis !== "estimate";
  return {
    text: measured
      ? t("voice.models.sttFit", { defaultValue: "Fits · {gb} GB GPU memory", gb: model.minVramGb })
      : t("voice.models.sttFitAbout", { defaultValue: "Fits · about {gb} GB GPU memory (estimate)", gb: model.minVramGb }),
    title: measured
      ? t("voice.models.sttFitMeasured", { defaultValue: "Measured on an RTX 4090 (24 GB), with about 1 GB of room added." })
      : t("voice.models.sttFitEstimate", { defaultValue: "An estimate from the size of the weights; not measured here." }),
  };
}

/** The confirm before a dictation model is deleted. */
export function sttDeleteConfirmText(name: string, sizeGb: number, t: T = tStudio): string {
  return t("voice.models.delete.confirmStt", {
    defaultValue: "Delete {name} ({gb} GB) from this computer? If it is the dictation model, it downloads again the next time you dictate.",
    name,
    gb: sizeGb.toFixed(2),
  });
}

/** The confirm before the voice runtime is removed. */
export function runtimeRemoveConfirmText(sizeGb: number | null, t: T = tStudio): string {
  return t("voice.runtime.removeConfirm", {
    defaultValue:
      "Remove the voice runtime{size}? Dictation, read-aloud, voice calls and Voice Studio stop working until it is installed again. Downloaded dictation and speech models, cloned voices and the Library stay.",
    size: sizeGb ? ` (${sizeGb.toFixed(1)} GB)` : "",
  });
}

/**
 * Asked row by row in the order shown: does this row open its family's heading (the first row of the family)? Keyed
 * by family, so asking again for the same row (a second render) gives the same answer.
 */
export function familyHeadingTracker(): (entry: Pick<VoiceModelEntry, "id" | "familyId">) => boolean {
  const first = new Map<string, string>();
  return (entry) => {
    const family = entry.familyId ?? "";
    if (!first.has(family)) first.set(family, entry.id);
    return first.get(family) === entry.id;
  };
}

/** A family with at least this many rows collapses to its shown rows (the mock-up's "Show 5 sizes"). */
export const COLLAPSE_FAMILY_FROM = 3;

/**
 * Rows a collapsed family keeps showing, by family id: every downloaded, downloading, just-cancelled or failed row, or
 * when none is, the preferred row (the recommended natural model), else the first row that can run here (on a PC the Apple-only rows come first in the catalog), then the first
 * GPU row (the speech GPU add-on is not in yet), then the first row. Families with fewer than COLLAPSE_FAMILY_FROM rows
 * are absent (never collapsed).
 */
export function collapsedFamilyRows(entries: VoiceModelEntry[], preferred?: string | null): Map<string, { total: number; shown: Set<string> }> {
  const byFamily = new Map<string, VoiceModelEntry[]>();
  for (const entry of entries) {
    if (isKokoro(entry.source)) continue;
    const key = entry.familyId ?? "";
    byFamily.set(key, [...(byFamily.get(key) ?? []), entry]);
  }
  const result = new Map<string, { total: number; shown: Set<string> }>();
  for (const [family, members] of byFamily) {
    if (members.length < COLLAPSE_FAMILY_FROM) continue;
    const keep = members.filter((m) => !isKokoro(m.source) && (m.source.installed || ["downloading", "cancelled", "failed"].includes(m.source.download.state)));
    const rows = members.map((m) => ({ id: m.id, row: m.source as VoiceModelRow }));
    // The recommended natural row (the install card's pick) when it is in this family, else the first that runs here.
    const first = rows.find((m) => m.id === preferred && m.row.supported) ?? rows.find((m) => m.row.supported) ?? rows.find((m) => (m.row.lanes ?? []).includes("cuda")) ?? rows[0];
    result.set(family, { total: members.length, shown: new Set(keep.length ? keep.map((m) => m.id) : [first.id]) });
  }
  return result;
}

export function voiceSortOptions(t: T, catalogOrder: string[]): ModelSortOption<VoiceModelEntry>[] {
  const order = (entry: VoiceModelEntry) => (isKokoro(entry.source) ? -1 : catalogOrder.indexOf(entry.id));
  return [
    {
      id: "family",
      label: t("voice.models.sort.family", { defaultValue: "By family" }),
      defaultDir: "asc",
      summary: () => t("voice.models.sort.familySummary", { defaultValue: "by family" }),
      compare: (a, b) => voiceFamilyRank(a.entry.familyId) - voiceFamilyRank(b.entry.familyId) || order(a.entry) - order(b.entry),
    },
    {
      id: "fit",
      label: t("voice.models.sort.fit", { defaultValue: "Best fit first" }),
      defaultDir: "asc",
      summary: () => t("voice.models.sort.fitSummary", { defaultValue: "best fit first" }),
      compare: (a, b) => fitSortRank(a.entry.fit) - fitSortRank(b.entry.fit) || (a.entry.sizeGb ?? 0) - (b.entry.sizeGb ?? 0),
    },
    {
      id: "size",
      label: t("voice.models.sort.size", { defaultValue: "Size" }),
      defaultDir: "asc",
      summary: (dir) =>
        dir === "asc"
          ? t("voice.models.sort.smallestFirst", { defaultValue: "smallest first" })
          : t("voice.models.sort.largestFirst", { defaultValue: "largest first" }),
      value: (row) => row.entry.sizeGb,
    },
    {
      id: "name",
      label: t("voice.models.sort.name", { defaultValue: "Name" }),
      defaultDir: "asc",
      summary: () => t("voice.models.sort.nameSummary", { defaultValue: "name A-Z" }),
      compare: (a, b) => a.entry.name.localeCompare(b.entry.name),
    },
    {
      id: "status",
      label: t("voice.models.sort.status", { defaultValue: "Status" }),
      defaultDir: "asc",
      summary: () => t("voice.models.sort.statusSummary", { defaultValue: "downloaded first" }),
      compare: (a, b) => statusSortRank(a.status) - statusSortRank(b.status),
    },
  ];
}
