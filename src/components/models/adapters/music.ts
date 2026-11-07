/**
 * Music catalog -> Models page entries.
 *
 * One row per catalog variant. The row keeps the catalog variant in `source`; the card fields the Music
 * page adds (what it makes, licence, platforms, measured speed) are read from it by the page, not copied.
 * A row is "installed" when its weights are on disk and the weights it requires are too; the music pack
 * is a separate thing (the banner above the list), so a row can be installed while the pack is not.
 */

import type { MusicModelFamily, MusicModelVariant } from "../../../types";
import { audioPerSecond, isInstalled } from "../../../features/music/musicLogic";
import { readBestFitVariantId, readVariantFit } from "../fit";
import type { ModelEntry } from "../modelEntry";

export type MusicModelEntry = ModelEntry<MusicModelVariant>;

export interface MusicEntryContext {
  family?: MusicModelFamily | null;
  /** The `engineModel` of the model the music worker has in memory. */
  loadedEngineModel?: string | null;
}

function gbLabel(gb: number): string {
  return `${gb.toFixed(1)} GB`;
}

export function musicVariantToEntry(variant: MusicModelVariant, ctx: MusicEntryContext = {}): MusicModelEntry {
  const family = ctx.family ?? null;
  const installed = isInstalled(variant);
  const engineModel = variant.engineModel;
  const peak = variant.estimates?.peakGb ?? null;
  return {
    id: variant.id,
    area: "music",
    name: variant.name,
    provider: variant.provider,
    familyId: variant.familyId,
    familyName: variant.familyName ?? family?.name ?? null,
    note: variant.note || null,
    repo: variant.repo,
    link: variant.link,
    downloadKey: variant.repo,
    deleteTarget: variant.repo,
    availableLocally: installed,
    hasLocalData: Boolean(variant.availableLocally),
    loaded: Boolean(ctx.loadedEngineModel && engineModel && ctx.loadedEngineModel === engineModel),
    // Rows stay visible on a machine that cannot run music yet; the page says why and turns Download off.
    hostCompatible: true,
    gated: variant.licence?.gated ?? null,
    license: variant.licence?.name ?? null,
    runtime: variant.runtime,
    tasks: [...(variant.taskSupport ?? [])],
    styleTags: [],
    sizeGb: variant.sizeGb,
    sizeLabel: gbLabel(variant.sizeGb),
    sizeTitle: variant.sizeNote ?? null,
    memory:
      peak != null
        ? { label: `~${peak >= 10 ? peak.toFixed(0) : peak.toFixed(1)} GB peak`, title: variant.fit?.reason ?? null, peakGb: peak }
        : null,
    releaseSortKey: "",
    fit: readVariantFit(variant),
    bestFit: Boolean(family && readBestFitVariantId(family) === variant.id),
    shared: null,
    source: variant,
  };
}

export function buildMusicEntries(
  families: MusicModelFamily[],
  ctx: { loadedEngineModel?: string | null } = {},
): MusicModelEntry[] {
  return families.flatMap((family) =>
    family.variants.map((variant) =>
      musicVariantToEntry({ ...variant, familyName: variant.familyName ?? family.name }, { ...ctx, family }),
    ),
  );
}

/** The sentence under "Speed" in the row detail: measured rate or "not measured yet". */
export function measuredSpeed(variant: MusicModelVariant): { measured: boolean; rate: number | null } {
  const rate = audioPerSecond(variant);
  return { measured: rate !== null, rate };
}

/** Search text of a row: name, provider, licence and what it makes. */
export function musicEntryMatchesQuery(entry: MusicModelEntry, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  const variant = entry.source;
  const haystack = [
    entry.name,
    entry.provider,
    entry.familyName,
    variant.licence?.name,
    variant.makes.join(" "),
    variant.makes.map((kind) => ({ songs: "songs vocals", instrumental: "instrumental", sfx: "sound effects", loops: "loops" })[kind]).join(" "),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(needle);
}
