/** Pure rules for the Speak and Voices screens: which model runs, whether the style box is usable. */

import type { VoiceEntry, VoiceModelRow } from "../../../types";

export const DESIGN_VOICE_ID = "design";
export const MAX_SPEAK_CHARS = 200_000;
/** One request carries at most this much text (the backend limit). */
export const MAX_REQUEST_CHARS = 20_000;

/** Style chips: the label key and the instruction sent to the model (empty clears the box). */
export const STYLE_CHIPS: Array<{ id: string; instruction: string }> = [
  { id: "neutral", instruction: "" },
  { id: "angry", instruction: "Speak in an angry tone." },
  { id: "sad", instruction: "Speak in a sad, quiet tone." },
  { id: "excited", instruction: "Speak in an excited, energetic tone." },
  { id: "whisper", instruction: "Speak in a soft whisper." },
  { id: "calm", instruction: "Speak in a calm, relaxed tone." },
];

export type VoiceFilter = "all" | "mine" | "builtin";

/** What a voice costs to run, for the card chip, the cost line and the Models rows. Never a guess: an unmeasured model says so. */
export type CostLabel =
  | { kind: "kokoro" }
  | { kind: "measured"; downloadGb: number; peakGb: number; secondsPer10s: number; coldStartS: number | null; measuredOn: string }
  | { kind: "unmeasured"; downloadGb: number };

const oneDecimal = (value: number) => Math.round(value * 10) / 10;

/** The cost of ``model``; a null model is a quick (Kokoro) voice, part of the voice pack. */
export function costLabel(model: VoiceModelRow | null): CostLabel {
  if (!model) return { kind: "kokoro" };
  const est = model.estimates;
  if (est && est.basis === "measured" && est.peakGb != null && est.secondsPer10sAudio != null) {
    return {
      kind: "measured",
      downloadGb: model.sizeGb,
      peakGb: oneDecimal(est.peakGb),
      secondsPer10s: oneDecimal(est.secondsPer10sAudio),
      coldStartS: est.coldStartSeconds != null ? oneDecimal(est.coldStartSeconds) : null,
      measuredOn: est.measuredOn ?? "",
    };
  }
  return { kind: "unmeasured", downloadGb: model.sizeGb };
}

/**
 * The model that will run, mirroring the backend (``voice_speak._pick_custom``): an installed model first, then the
 * style-capable one, then the smaller. Typing in the style box never changes it, so there is no reload mid-session;
 * the 0.6B CustomVoice runs only when it is the only one installed (it is no quicker, FU-148).
 */
export function chooseModel(voice: VoiceEntry | null, models: VoiceModelRow[], preferredKey = ""): VoiceModelRow | null {
  if (!voice || voice.kind === "builtin") return null;
  const rows = voice.models.map((key) => models.find((m) => m.key === key)).filter((m): m is VoiceModelRow => Boolean(m));
  // A cloning model the person picked, when it is downloaded.
  const preferred = voice.kind === "clone" ? rows.find((m) => m.key === preferredKey && m.installed) : undefined;
  if (preferred) return preferred;
  const rank = (m: VoiceModelRow) => [m.capabilities.style ? 0 : 1, m.sizeGb] as const;
  const sort = (list: VoiceModelRow[]) =>
    [...list].sort((a, b) => {
      const [ax, as] = rank(a);
      const [bx, bs] = rank(b);
      return ax - bx || as - bs;
    });
  const installed = rows.filter((m) => m.installed);
  return sort(installed.length > 0 ? installed : rows)[0] ?? null;
}

/** The downloaded cloning models for a cloned voice; the Speak screen offers a choice when there are two or more. */
export function installedCloneModels(voice: VoiceEntry | null, models: VoiceModelRow[]): VoiceModelRow[] {
  if (!voice || voice.kind !== "clone") return [];
  return voice.models
    .map((key) => models.find((m) => m.key === key))
    .filter((m): m is VoiceModelRow => Boolean(m && m.installed));
}

export interface StyleState {
  /** Whether the style box can be edited. */
  enabled: boolean;
  /** Why it cannot, in a key the screen translates. */
  reason: "builtin" | "clone" | "smallModel" | "none";
  /** A model to download to unlock style (preset voices only). */
  unlockWith: VoiceModelRow | null;
}

/** Style is usable for a design voice (the box is its description) and for a preset when a style model is installed. */
export function styleState(voice: VoiceEntry | null, models: VoiceModelRow[]): StyleState {
  if (!voice) return { enabled: false, reason: "none", unlockWith: null };
  if (voice.kind === "builtin") return { enabled: false, reason: "builtin", unlockWith: null };
  if (voice.kind === "clone") return { enabled: false, reason: "clone", unlockWith: null };
  if (voice.kind === "design") return { enabled: true, reason: "none", unlockWith: null };
  const styled = voice.models.map((key) => models.find((m) => m.key === key)).filter((m): m is VoiceModelRow => Boolean(m && m.capabilities.style));
  if (styled.some((m) => m.installed)) return { enabled: true, reason: "none", unlockWith: null };
  return { enabled: false, reason: "smallModel", unlockWith: styled[0] ?? null };
}

export interface ModelNote {
  /** The model that runs, or null for a built-in voice (part of the voice pack). */
  model: VoiceModelRow | null;
  /** The model must be downloaded before Generate works. */
  needsDownload: boolean;
  downloading: boolean;
}

export function modelNote(voice: VoiceEntry | null, models: VoiceModelRow[], preferredKey = ""): ModelNote {
  const model = chooseModel(voice, models, preferredKey);
  return {
    model,
    needsDownload: Boolean(voice && voice.kind !== "builtin" && (!model || !model.installed)),
    downloading: model?.download.state === "downloading",
  };
}

/** Whether Generate can run. A design voice needs its description in the style box. */
export function canGenerate(voice: VoiceEntry | null, text: string, style: string, note: ModelNote): boolean {
  if (!voice || !text.trim() || note.needsDownload) return false;
  if (voice.kind === "design" && !style.trim()) return false;
  return true;
}

/**
 * Speed or pitch would need ffmpeg, which is missing (``needsFfmpeg`` from the voice list). Pitch is an ffmpeg step on
 * every voice; speed is one on every voice except the quick (Kokoro) ones, which change speed themselves.
 */
export function ffmpegBlocked(voice: VoiceEntry | null, speed: number, pitch: number): boolean {
  if (!voice?.needsFfmpeg) return false;
  return pitch !== 0 || (voice.kind !== "builtin" && Math.abs(speed - 1) > 1e-6);
}

/** Whether a request-sized piece of text is too long for one cloned-voice request. */
export function cloneTextTooLong(text: string): boolean {
  return text.length > MAX_REQUEST_CHARS;
}

/** The text of a cloned-voice request: one request, so a [pause] marker becomes a paragraph break. */
export function cloneRequestText(text: string): string {
  return text.replace(/\s*\[pause\]\s*/gi, "\n\n").trim();
}

export function matchesFilter(voice: VoiceEntry, filter: VoiceFilter): boolean {
  if (filter === "all") return true;
  if (filter === "builtin") return voice.kind === "builtin" || voice.kind === "preset";
  return voice.kind === "clone";
}

/** Voices shown in the picker: built-in ones that cannot be used here are left out, as before. */
export function pickerVoices(voices: VoiceEntry[]): VoiceEntry[] {
  return voices.filter((v) => v.kind !== "builtin" || v.available);
}

const GRADE_ORDER = ["A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F+", "F"];

/** Lower is better; an ungraded voice sorts after every graded one. */
export function gradeRank(grade: string | null | undefined): number {
  const index = grade ? GRADE_ORDER.indexOf(grade) : -1;
  return index < 0 ? GRADE_ORDER.length : index;
}

/** "en" from "en-US", "zh" from "zh-CN". */
function baseLanguage(code: string): string {
  return code.toLowerCase().split("-")[0] ?? "";
}

function sameLanguage(voice: VoiceEntry, uiLanguage: string): boolean {
  return baseLanguage(voice.language) === baseLanguage(uiLanguage);
}

/** The best natural voice that runs now: a preset in the UI language, else the first preset that runs. */
export function bestNaturalVoice(voices: VoiceEntry[], uiLanguage: string): VoiceEntry | null {
  const ready = voices.filter((v) => v.kind === "preset" && v.available);
  return ready.find((v) => sameLanguage(v, uiLanguage)) ?? ready[0] ?? null;
}

/** The top-graded quick (Kokoro) voice for the UI language; the list order breaks ties. */
export function topQuickVoice(voices: VoiceEntry[], uiLanguage: string): VoiceEntry | null {
  const quick = voices.filter((v) => v.kind === "builtin" && v.available);
  const pool = quick.filter((v) => sameLanguage(v, uiLanguage));
  const ranked = [...(pool.length > 0 ? pool : quick)].map((v, i) => ({ v, i }));
  ranked.sort((a, b) => gradeRank(a.v.grade) - gradeRank(b.v.grade) || a.i - b.i);
  return ranked[0]?.v ?? null;
}

/**
 * Which voice Speak starts on: the voice last used in Speak (``speakVoice``, apart from the chat voice); else the
 * best installed natural voice for the UI language; else the top-graded quick voice for the UI language; else the
 * first usable voice.
 */
export function defaultVoice(voices: VoiceEntry[], speakVoice: string, uiLanguage: string): string {
  const list = pickerVoices(voices);
  return (
    list.find((v) => v.id === speakVoice && speakVoice !== "")?.id ??
    bestNaturalVoice(list, uiLanguage)?.id ??
    topQuickVoice(list, uiLanguage)?.id ??
    list[0]?.id ??
    ""
  );
}

/** Quick voices graded at least this well make the shortlist (most others sound poor). */
const SHORTLIST_FLOOR = gradeRank("C");
const SHORTLIST_PER_GENDER = 3;
const SHORTLIST_FALLBACK = 6;

/**
 * The quick voices shown before "All voices": per language, the best graded (C or better), at most three of each
 * gender, best first. A language with no graded voice (Spanish, Portuguese) or only poor ones (Mandarin) shows its
 * first six; the expander shows the rest. American English gives Heart, Bella, Nicole, Fenrir, Michael, Puck.
 */
export function quickShortlist(voices: VoiceEntry[]): VoiceEntry[] {
  const ranked = voices.map((v, i) => ({ v, i })).sort((a, b) => gradeRank(a.v.grade) - gradeRank(b.v.grade) || a.i - b.i);
  const perGender: Record<string, number> = {};
  const picked = ranked
    .filter(({ v }) => gradeRank(v.grade) <= SHORTLIST_FLOOR)
    .filter(({ v }) => {
      const key = v.gender ?? "other";
      perGender[key] = (perGender[key] ?? 0) + 1;
      return perGender[key] <= SHORTLIST_PER_GENDER;
    })
    .map(({ v }) => v);
  return picked.length > 0 ? picked : voices.slice(0, SHORTLIST_FALLBACK);
}

/** The language label a quick voice is grouped under. */
export function quickLanguage(voice: VoiceEntry): string {
  return voice.languageLabel ?? voice.language;
}

export interface VoiceGroups {
  /** Best natural voice, top quick voice, newest cloned voice, design card: picked for this machine and UI language. */
  recommended: VoiceEntry[];
  /** Kokoro voices that run here, plus those an add-on unlocks (Mandarin), in list order. */
  quick: VoiceEntry[];
  clones: VoiceEntry[];
  design: VoiceEntry | null;
  /** Natural presets other than the recommended one. */
  more: VoiceEntry[];
}

/** The Speak picker's groups, in the order shown: Recommended, Quick, Cloned, Designed, More. */
export function groupVoices(voices: VoiceEntry[], uiLanguage: string, modelsAvailable: boolean): VoiceGroups {
  const quick = voices.filter((v) => v.kind === "builtin" && (v.available || Boolean(v.addon)));
  const clones = voices.filter((v) => v.kind === "clone");
  const design = voices.find((v) => v.kind === "design") ?? null;
  const presets = voices.filter((v) => v.kind === "preset");
  const natural = modelsAvailable ? bestNaturalVoice(presets, uiLanguage) : null;
  const topQuick = topQuickVoice(quick, uiLanguage);
  const recommended = [natural, topQuick, clones[0] ?? null, modelsAvailable ? design : null].filter((v): v is VoiceEntry => Boolean(v));
  return { recommended, quick, clones, design, more: presets.filter((v) => v.id !== natural?.id) };
}

/** The model the "Install a better voice" card offers: the style-capable CustomVoice model. */
export function recommendedNaturalModel(models: VoiceModelRow[]): VoiceModelRow | null {
  const custom = models.filter((m) => m.kind === "custom" && m.supported);
  return custom.find((m) => m.capabilities.style) ?? custom[0] ?? null;
}

/** The install card shows while no natural (CustomVoice) model is downloaded on a machine that can run one. */
export function showInstallCard(models: VoiceModelRow[], modelsAvailable: boolean): boolean {
  if (!modelsAvailable) return false;
  const custom = models.filter((m) => m.kind === "custom");
  return custom.length > 0 && !custom.some((m) => m.installed);
}
