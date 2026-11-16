/**
 * Pure rules for the Plan tab and the style and brand kit (S8, S15): the length choices, how a proposal's
 * length compares with the aim, what the direction chat's changes say in words, which fields a reworked shot
 * changes, where a brief picture goes, the brand kit with its defaults filled in for older projects, and the shot
 * count suggested from the brief's length (`shotSuggestion`).
 * No React here so the rules are unit-tested directly.
 */
import type {
  StoryboardBrand,
  StoryboardPlan,
  StoryboardPlanChanges,
  StoryboardPlannedShot,
  StoryboardProject,
  StoryboardReference,
  StoryboardShot,
} from "../../types";
import { MAX_PICTURES } from "./pickerLogic";

/** Film lengths offered next to the brief, in seconds. */
export const LENGTH_CHOICES = [15, 30, 60, 90, 120] as const;
/** Shot counts the model may be asked for; null lets it choose between 4 and 8. */
export const SHOT_COUNT_CHOICES = [4, 5, 6, 7, 8] as const;
/** The most shots one proposal writes (mirrors MAX_SHOTS in planner.py). */
export const MAX_PROPOSE_SHOTS = 8;
/** Style references a project keeps (mirrors References.styles in models.py). */
export const MAX_STYLES = 20;
/** Colours kept per style (mirrors MAX_PALETTE in models.py). */
export const MAX_PALETTE = 8;

export const DEFAULT_BRAND: StoryboardBrand = {
  logo: null,
  logoCorner: "bottom-right",
  captions: { font: null, colour: "#ffffff", position: "bottom" },
  styleUse: "auto",
};

export const DEFAULT_PLAN: StoryboardPlan = { brief: "", filmSeconds: null };

export function planOf(project: Pick<StoryboardProject, "plan">): StoryboardPlan {
  return { ...DEFAULT_PLAN, ...(project.plan ?? {}) };
}

export function brandOf(project: Pick<StoryboardProject, "brand">): StoryboardBrand {
  const brand = project.brand;
  if (!brand) return DEFAULT_BRAND;
  return { ...DEFAULT_BRAND, ...brand, captions: { ...DEFAULT_BRAND.captions, ...(brand.captions ?? {}) } };
}

export function stylesOf(project: Pick<StoryboardProject, "references">): StoryboardReference[] {
  return project.references.styles ?? [];
}

/** Every colour of every style, once each, in order. */
export function projectPalette(project: Pick<StoryboardProject, "references">, limit = 12): string[] {
  const out: string[] = [];
  for (const ref of stylesOf(project)) for (const colour of ref.palette ?? []) if (!out.includes(colour)) out.push(colour);
  return out.slice(0, limit);
}

/** The sum of the shots' seconds, to one decimal. */
export function totalSeconds(shots: readonly Pick<StoryboardPlannedShot, "seconds">[]): number {
  return Math.round(shots.reduce((sum, shot) => sum + (Number.isFinite(shot.seconds) ? shot.seconds : 0), 0) * 10) / 10;
}

export type LengthVerdict = "match" | "short" | "long" | "none";

/** Within 15 % of the aim is a match; no aim gives "none". */
export function lengthVerdict(total: number, target: number | null | undefined): LengthVerdict {
  if (!target) return "none";
  if (total < target * 0.85) return "short";
  if (total > target * 1.15) return "long";
  return "match";
}

/** "1 min", "1 min 30 s", "45 s". */
export function lengthLabel(seconds: number): { minutes: number; seconds: number } {
  const whole = Math.round(seconds);
  return { minutes: Math.floor(whole / 60), seconds: whole % 60 };
}

export function changeCount(changes: StoryboardPlanChanges | null | undefined): number {
  if (!changes) return 0;
  return changes.edits.length + changes.add.length + changes.remove.length;
}

export type ChangeRow =
  | { kind: "edit"; number: number; title: string; fields: string[] }
  | { kind: "add"; after: number; title: string }
  | { kind: "remove"; number: number; title: string };

/** The direction chat's changes as rows the user can read before applying them. Unknown shots are dropped. */
export function changeRows(changes: StoryboardPlanChanges, shots: readonly Pick<StoryboardShot, "id" | "title">[]): ChangeRow[] {
  const numberOf = (id: string | null) => (id === null ? 0 : shots.findIndex((s) => s.id === id) + 1);
  const rows: ChangeRow[] = [];
  for (const edit of changes.edits) {
    const number = numberOf(edit.shotId);
    if (number <= 0) continue;
    const fields = (["title", "stillPrompt", "motionPrompt", "seconds"] as const).filter((key) => edit[key] !== undefined);
    rows.push({ kind: "edit", number, title: shots[number - 1].title, fields });
  }
  for (const id of changes.remove) {
    const number = numberOf(id);
    if (number > 0) rows.push({ kind: "remove", number, title: shots[number - 1].title });
  }
  for (const added of changes.add) {
    const after = numberOf(added.after);
    if (after >= 0) rows.push({ kind: "add", after, title: added.title });
  }
  return rows;
}

export type ShotField = "title" | "stillPrompt" | "motionPrompt" | "seconds";

/** The fields a reworked shot changes, compared with the shot as it is. */
export function changedFields(shot: Pick<StoryboardShot, ShotField>, suggestion: Pick<StoryboardPlannedShot, ShotField>): ShotField[] {
  return (["title", "stillPrompt", "motionPrompt", "seconds"] as const).filter((key) => {
    if (key === "seconds") return Math.abs(shot.seconds - suggestion.seconds) > 0.01;
    return (shot[key] ?? "").trim() !== (suggestion[key] ?? "").trim() && (suggestion[key] ?? "").trim() !== "";
  });
}

/** `edit_shot` arguments for a reworked shot: only what changed. */
export function reworkEdit(shot: Pick<StoryboardShot, "id" | ShotField>, suggestion: Pick<StoryboardPlannedShot, ShotField>) {
  const args: { shotId: string; title?: string; stillPrompt?: string; motionPrompt?: string; seconds?: number } = { shotId: shot.id };
  for (const key of changedFields(shot, suggestion)) {
    if (key === "seconds") args.seconds = suggestion.seconds;
    else args[key] = suggestion[key];
  }
  return args;
}

/** The style a brief picture joins: the first with room, or null when a new style is needed. */
export function styleWithRoom(project: Pick<StoryboardProject, "references">): StoryboardReference | null {
  return stylesOf(project).find((ref) => ref.images.length < MAX_PICTURES) ?? null;
}

/** True when an apply would replace shots that already have pictures or clips. */
export function replaceLosesTakes(project: Pick<StoryboardProject, "shots">): number {
  return project.shots.filter((shot) => shot.takes.length > 0).length;
}

export function lockedCount(project: Pick<StoryboardProject, "shots">): number {
  return project.shots.filter((shot) => shot.locked).length;
}

/** `#rrggbb` only (what the backend takes). */
export function isHexColour(value: string): boolean {
  return /^#[0-9a-f]{6}$/.test(value);
}

/** Black or white text, whichever reads better on the colour (for swatch labels and the caption preview). */
export function inkOn(hex: string): "#000000" | "#ffffff" {
  if (!isHexColour(hex)) return "#ffffff";
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const luminance = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  return luminance > 0.4 ? "#000000" : "#ffffff";
}

/** What the backend's error code means for the Plan tab. */
export type PlanFailure = "noModel" | "busy" | "other";

export function planFailure(code: string | null | undefined): PlanFailure {
  if (code === "no-model") return "noModel";
  if (code === "busy") return "busy";
  return "other";
}

/** The style word on an active take (`settings.style`), when the batch recorded one. */
export function takeStyle(shot: Pick<StoryboardShot, "takes" | "activeTake">): "pictures" | "words" | null {
  const take = shot.takes.find((t) => t.id === shot.activeTake);
  const value = take?.settings?.style;
  return value === "pictures" || value === "words" ? value : null;
}

// ---- shot count and length suggestion (FU-188 item 5) ----
export type FilmKind = "ad" | "trailer" | "film";

/** Seconds per shot the suggestion assumes: a shot of 4 to 5 s reads well and fits every clip model. */
export const SUGGEST_SHOT_SECONDS = { short: 4, long: 5 } as const;
/** A usual length when the brief states none: an ad 30 s, a trailer 60 s, anything else 30 s. */
export const TYPICAL_SECONDS: Record<FilmKind, number> = { ad: 30, trailer: 60, film: 30 };

// Length and kind words in the shipping languages (S-N): a brief in German, Japanese or Russian gets the same
// suggestion as one in English. Digits work in any language; number words are English plus "one"/"a" (and a few
// tens) in the others, and Han numerals ("三十秒", "一分钟") in Chinese and Japanese. Not read: other number words
// spelled out ("fünfundvierzig Sekunden", Korean "삼십 초"); digits work there.
const NUMBER_WORDS: Record<string, number> = {
  one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  fifteen: 15, twenty: 20, thirty: 30, forty: 40, "forty-five": 45, fifty: 50, sixty: 60, ninety: 90,
  eine: 1, einer: 1, une: 1, un: 1, una: 1, um: 1, uma: 1, "одна": 1, "одну": 1, "один": 1,
  zwei: 2, deux: 2, dos: 2, dois: 2, duas: 2, "две": 2, "два": 2,
  "dreißig": 30, trente: 30, treinta: 30, trinta: 30, "тридцать": 30,
};
/** Seconds or minutes, by the unit's word (minute words are the ones listed here). */
const SECOND_UNITS = "seconds?|secs?|s|sekunden?|sek|secondes?|segundos?|seg|секунд\\p{L}*|сек";
const MINUTE_UNITS = "minutes?|mins?|minuten?|minutos?|минут\\p{L}*|мин";
const CJK_UNITS = "秒|分钟|分鐘|分間|分|초|분";
// Digits may follow a CJK letter ("一个1分钟"), not a Latin or Cyrillic one ("mp4s"); number words stand alone.
const NUMBER = `(?<![\\p{Script=Latin}\\p{Script=Cyrillic}\\p{N}.,])\\d+(?:[.,]\\d+)?|(?<![\\p{L}\\p{N}])(?:${Object.keys(NUMBER_WORDS).join("|")})`;
// Han numerals before a CJK unit ("三十秒", "一分钟", "九十秒"), up to 99.
const HAN_DIGITS: Record<string, number> = { 一: 1, 二: 2, 两: 2, 兩: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9 };
const HAN_NUMBER = "[一二两兩三四五六七八九十]{1,3}";
const LENGTH_RE = new RegExp(
  `(${NUMBER})(?:[\\s-]*(${SECOND_UNITS}|${MINUTE_UNITS})(?![\\p{L}])|\\s*(${CJK_UNITS}))|(${HAN_NUMBER})\\s*(${CJK_UNITS})`,
  "iu",
);

/** "三十" 30, "十五" 15, "九十" 90, "五" 5; NaN for anything else. */
function hanNumber(text: string): number {
  const [tens, ones] = text.includes("十") ? text.split("十") : ["", text];
  if (text.includes("十") && (tens.length > 1 || ones.length > 1)) return Number.NaN;
  if (!text.includes("十")) return text.length === 1 ? (HAN_DIGITS[text] ?? Number.NaN) : Number.NaN;
  const t = tens === "" ? 1 : HAN_DIGITS[tens];
  const o = ones === "" ? 0 : HAN_DIGITS[ones];
  return t === undefined || o === undefined ? Number.NaN : t * 10 + o;
}
const MINUTE_RE = new RegExp(`^(?:${MINUTE_UNITS}|分钟|分鐘|分間|分|분)$`, "iu");

/** The film length a brief states ("a 30 s ad", "a one-minute advert", "90 Sekunden", "30秒の広告"), in seconds, or null. */
export function briefSeconds(brief: string): number | null {
  const match = LENGTH_RE.exec(brief);
  if (!match) return null;
  if (match[4]) {
    const amount = hanNumber(match[4]);
    const seconds = MINUTE_RE.test(match[5]) ? amount * 60 : amount;
    return Number.isFinite(seconds) && seconds >= 2 && seconds <= 600 ? Math.round(seconds) : null;
  }
  const word = match[1].toLowerCase();
  const amount = NUMBER_WORDS[word] ?? Number(word.replace(",", "."));
  const unit = match[2] ?? match[3];
  const seconds = MINUTE_RE.test(unit) ? amount * 60 : amount;
  return Number.isFinite(seconds) && seconds >= 2 && seconds <= 600 ? Math.round(seconds) : null;
}

const AD_RE =
  /(?<![\p{L}])(ad|ads|advert|adverts|advertisement|commercial|promo|werbung|werbespot|spot|publicité|publicite|anuncio|comercial|anúncio|propaganda|реклам\p{L}*)(?![\p{L}])|広告|コマーシャル|广告|廣告|광고/iu;
const TRAILER_RE = /(?<![\p{L}])(trailer|bande-annonce|tráiler|трейлер\p{L}*)(?![\p{L}])|予告|トレーラー|预告|預告|예고편|트레일러/iu;

/** What kind of film a brief asks for, from a few words in any shipping language: an ad, a trailer, or another film. */
export function briefKind(brief: string): FilmKind {
  if (AD_RE.test(brief)) return "ad";
  if (TRAILER_RE.test(brief)) return "trailer";
  return "film";
}

export interface ShotSuggestion {
  seconds: number;
  /** Where the length came from: the brief's words, the Length choice, or the usual length for the kind. */
  source: "brief" | "target" | "typical";
  kind: FilmKind;
  low: number;
  high: number;
}

/**
 * A rule-of-thumb shot count for the Plan tab, labelled as a suggestion: a length stated in the brief wins, then the
 * project's Length choice, then the usual length for the kind of film; shots of 4 to 5 s give the range
 * (30 s is about 6 to 8 shots). Null while the brief is empty and no length is chosen.
 */
export function shotSuggestion(brief: string, target: number | null | undefined): ShotSuggestion | null {
  const text = brief.trim();
  const kind = briefKind(text);
  const stated = briefSeconds(text);
  let seconds: number;
  let source: ShotSuggestion["source"];
  if (stated != null) [seconds, source] = [stated, "brief"];
  else if (target) [seconds, source] = [target, "target"];
  else if (text) [seconds, source] = [TYPICAL_SECONDS[kind], "typical"];
  else return null;
  const low = Math.max(1, Math.ceil(seconds / SUGGEST_SHOT_SECONDS.long));
  const high = Math.max(low, Math.ceil(seconds / SUGGEST_SHOT_SECONDS.short));
  return { seconds, source, kind, low, high };
}
