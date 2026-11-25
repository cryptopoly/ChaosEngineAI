/**
 * The voice runtime's download size: the one place these figures live (the Voice setup card, its presets and its
 * Choose parts labels all read them). Base: the speech packages, turn detection and speaker labels, about 2.3 GB on the
 * portable lane. The speech GPU add-on brought the speech packages to 8.0 GB on the RTX 4090 run (2026-11), so about
 * 5.7 GB of its own; the speech-to-text GPU add-on's figure is the label's estimate (not measured alone).
 */
export const RUNTIME_BASE_GB = 2.3;

export const ADDON_GB: Record<string, number> = {
  kokoro: 0.33,
  chinese: 0.06,
  "speech-gpu": 5.7,
  "stt-gpu": 2.9,
};

/** The runtime with these add-ons, rounded to one decimal (unknown ids count nothing). */
export function runtimeInstallGb(addons: Iterable<string>): number {
  let total = RUNTIME_BASE_GB;
  for (const id of addons) total += ADDON_GB[id] ?? 0;
  return Math.round(total * 10) / 10;
}

/** One add-on's size for its label: "330 MB" below 1 GB, else "5.7 GB". */
export function addonSizeText(id: string): string {
  const gb = ADDON_GB[id] ?? 0;
  return gb < 1 ? `${Math.round(gb * 1000)} MB` : `${gb.toFixed(1)} GB`;
}
