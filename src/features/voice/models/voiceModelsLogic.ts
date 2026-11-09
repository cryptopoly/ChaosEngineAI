/** Pure rules for Voice > Models (D-73): delete confirms, family headings, sort options. */

import i18next from "i18next";
import { fitSortRank, statusSortRank, type ModelSortOption } from "../../../components/models";
import { isKokoro, voiceFamilyRank, type VoiceModelEntry } from "../../../components/models/adapters/voice";
import type { VoiceEntry, VoiceModelRow } from "../../../types";
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
