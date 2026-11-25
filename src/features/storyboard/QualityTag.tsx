/** The "Draft 640 x 384" tag on a draft picture, clip or take (Storyboard UX audit, change 2). Final takes get none. */
import { useTranslation } from "react-i18next";
import type { StoryboardSize, StoryboardTakeQuality } from "../../types";
import { draftTag, type TFn } from "./qualityText";
import "./quality.css";

export interface QualityTagProps {
  quality: StoryboardTakeQuality | null | undefined;
  /** The film's size, named in the explanation. */
  target?: StoryboardSize | null;
  /** `corner`: over a picture; `inline`: beside text; `mini`: just "Draft" on a small thumbnail (size in the tooltip). */
  variant?: "corner" | "inline" | "mini";
}

export function QualityTag({ quality, target, variant = "corner" }: QualityTagProps) {
  const { t } = useTranslation("studio");
  const tag = draftTag(t as unknown as TFn, quality, target);
  if (!tag) return null;
  const text = variant === "mini" ? (t as unknown as TFn)("board.quality.tagNoSize", { defaultValue: "Draft" }) : tag.label;
  return (
    <span className={`quality-tag quality-tag--${variant}`} title={tag.title}>
      {text}
    </span>
  );
}
