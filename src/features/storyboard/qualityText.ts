/**
 * Quality made visible (Storyboard UX audit, change 2): what a take was made at, as a tag text and a plain
 * explanation, and the draft-or-final choice stated in sizes. The numbers come from `guide.quality`; for a take the
 * guide does not know (an older payload, a card rendered without it) they are read from the take's own settings the
 * same way `backend_service/storyboard/quality.py` reads them.
 */
import type { StoryboardEstimate } from "../../api/storyboard";
import type { StoryboardQuality, StoryboardSize, StoryboardTake, StoryboardTakeQuality } from "../../types";
import { formatDuration, formatGb } from "../../components/RunEstimateStrip";

export type TFn = (key: string, options: Record<string, unknown>) => string;

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? Math.round(value) : null;
}

function half(settings: StoryboardTake["settings"], kind: "still" | "clip", last: boolean): (StoryboardSize & { fps?: number }) | null {
  const width = num(settings[`${kind}Width`]) ?? (last ? num(settings.width) : null);
  const height = num(settings[`${kind}Height`]) ?? (last ? num(settings.height) : null);
  if (width == null || height == null) return null;
  const out: StoryboardSize & { fps?: number } = { width, height };
  if (kind === "clip") {
    const fps = num(settings.clipFps) ?? (last ? num(settings.fps) : null);
    if (fps != null) out.fps = fps;
  }
  return out;
}

/** What a take was made at: the guide's label when it has one, else read from the take's own settings. */
export function takeQualityOf(take: StoryboardTake, fromGuide?: StoryboardTakeQuality | null): StoryboardTakeQuality {
  if (fromGuide) return fromGuide;
  const settings = take.settings ?? {};
  return {
    pass: take.renderPass === "draft" ? "draft" : "final",
    draft: take.renderPass === "draft",
    still: take.still ? half(settings, "still", !take.clip) : null,
    clip: take.clip ? half(settings, "clip", true) : null,
    stillStale: false,
    clipStale: false,
  };
}

/** The size a take plays or shows at: its clip, else its still. Null when it was not recorded. */
export function takeSize(quality: StoryboardTakeQuality | null | undefined): StoryboardSize | null {
  const size = quality?.clip ?? quality?.still ?? null;
  return size ? { width: size.width, height: size.height } : null;
}

export function sizeText(size: StoryboardSize): string {
  return `${size.width} x ${size.height}`;
}

export interface DraftTag {
  /** "Draft 640 x 384", or "Draft" when the size was not recorded. */
  label: string;
  /** A plain explanation for the tooltip. */
  title: string;
  size: StoryboardSize | null;
}

/** The tag for a draft take, or null for a final one. `target` is the film's size. */
export function draftTag(t: TFn, quality: StoryboardTakeQuality | null | undefined, target?: StoryboardSize | null): DraftTag | null {
  if (!quality?.draft) return null;
  const size = takeSize(quality);
  const label = size
    ? t("board.quality.tag", { defaultValue: "Draft {size}", size: sizeText(size) })
    : t("board.quality.tagNoSize", { defaultValue: "Draft" });
  let title: string;
  if (size && target) {
    title = t("board.quality.titleSized", {
      defaultValue: "A draft, made at {size}, about half the film's size, so you can choose shots fast. The film is {film}: choose this shot for final to make it that size.",
      size: sizeText(size),
      film: sizeText(target),
    });
  } else if (size) {
    title = t("board.quality.titleDraftOnly", {
      defaultValue: "A draft, made at {size}, smaller than the film, so you can choose shots fast. Choose this shot for final to make it full size.",
      size: sizeText(size),
    });
  } else {
    title = t("board.quality.titleNoSize", {
      defaultValue: "A draft, made smaller than the film, so you can choose shots fast. Choose this shot for final to make it full size.",
    });
  }
  return { label, title, size };
}

export interface StartChoice {
  draft: { size: StoryboardSize; seconds: number | null } | null;
  final: { size: StoryboardSize; seconds: number | null; peakGb: number | null; totalGb: number | null } | null;
}

/**
 * The draft-or-final choice in sizes and time. The draft size is the one the estimate names (the model's real
 * size), else the guide's; the final size is the film's. Null when there is nothing to say.
 */
export function startChoice(
  quality: StoryboardQuality | null | undefined,
  estimate: StoryboardEstimate | null | undefined,
  withClips = false,
): StartChoice | null {
  const pass = estimate?.passes?.draft;
  const draftSize = pass && pass.width > 0 && pass.height > 0 ? { width: pass.width, height: pass.height } : (quality?.draftSize ?? null);
  const finalSize = quality?.target ? { width: quality.target.width, height: quality.target.height } : null;
  if (!draftSize && !finalSize) return null;
  const draftSeconds = pass?.seconds != null ? pass.seconds + (withClips ? (pass.clipSeconds ?? 0) : 0) : null;
  return {
    draft: draftSize ? { size: draftSize, seconds: draftSeconds } : null,
    final: finalSize
      ? { size: finalSize, seconds: estimate?.seconds ?? null, peakGb: estimate?.peakGb ?? null, totalGb: estimate?.totalGb ?? null }
      : null,
  };
}

export function draftChoiceText(t: TFn, choice: NonNullable<StartChoice["draft"]>): string {
  const head = t("runBar.choice.draft", { defaultValue: "Drafts first: {size}", size: sizeText(choice.size) });
  const time = choice.seconds != null ? ` · ${t("runBar.choice.about", { defaultValue: "about {time}", time: formatDuration(t, choice.seconds) })}` : "";
  const why = t("runBar.choice.draftWhy", { defaultValue: "half size, to choose the shots you like. You then make those full size." });
  return `${head}${time} · ${why}`;
}

export function finalChoiceText(t: TFn, choice: NonNullable<StartChoice["final"]>): string {
  const head = t("runBar.choice.final", { defaultValue: "Straight to final: {size}", size: sizeText(choice.size) });
  const time = choice.seconds != null ? ` · ${t("runBar.choice.about", { defaultValue: "about {time}", time: formatDuration(t, choice.seconds) })}` : "";
  const memory =
    choice.peakGb != null && choice.totalGb != null
      ? ` · ${t("runBar.choice.peak", { defaultValue: "peak {peak} of {total}", peak: formatGb(t, choice.peakGb), total: formatGb(t, choice.totalGb) })}`
      : "";
  return `${head}${time}${memory}`;
}
