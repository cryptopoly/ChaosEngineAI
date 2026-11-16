/**
 * The brand kit on the film (S15), as the rough cut shows it: the title and end cards and the corner logo.
 * The same choices as the export (`backend_service/storyboard/cards.py`), so the preview and the file agree:
 * the card text is the Export tab's text or the project name; the typeface is the brand caption font; the
 * background is the main colour of the first style with a palette (else black); the text colour is the brand
 * caption colour when it reads on that background (contrast 3:1 or more), else white or black. A card lasts
 * whole frames, like the rendered one.
 */
import type { StoryboardLogoCorner, StoryboardProject } from "../../../types";

export const MIN_CONTRAST = 3;

function channel(value: number): number {
  const c = value / 255;
  return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

export function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

export function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

const HEX = /^#[0-9a-fA-F]{6}$/;

export interface CardLook {
  background: string;
  colour: string;
  /** The brand caption font family, or null for the app's own type. */
  font: string | null;
}

export function cardLook(project: StoryboardProject): CardLook {
  let background = "#000000";
  for (const style of project.references.styles ?? []) {
    const first = style.palette?.[0];
    if (first && HEX.test(first)) {
      background = first.toUpperCase();
      break;
    }
  }
  const brand = project.brand;
  const wanted = brand?.captions?.colour && HEX.test(brand.captions.colour) ? brand.captions.colour.toUpperCase() : "#FFFFFF";
  let colour = wanted;
  if (contrast(background, wanted) < MIN_CONTRAST) {
    colour = contrast(background, "#FFFFFF") >= contrast(background, "#000000") ? "#FFFFFF" : "#000000";
  }
  return { background, colour, font: brand?.captions?.font ?? null };
}

export type CardWhich = "title" | "end";

export function cardText(project: StoryboardProject, which: CardWhich): string {
  const raw = which === "title" ? project.export.titleText : project.export.endText;
  return (raw ?? "").trim() || project.name.trim();
}

/** A card's length on the film: whole frames, as the export renders it. */
export function cardSeconds(project: StoryboardProject): number {
  const fps = project.defaults.fps || 24;
  const seconds = project.export.cardSeconds ?? 2;
  return Math.max(1, Math.round(seconds * fps)) / fps;
}

export interface BrandOnFilm {
  title: { text: string; seconds: number } | null;
  end: { text: string; seconds: number } | null;
  /** Project-relative logo file, when the logo is drawn. */
  logo: string | null;
  /** The logo on the cards (whenever the project has one, as the export does). */
  cardLogo: string | null;
  corner: StoryboardLogoCorner;
  look: CardLook;
}

/** What the export adds to the film from the brand kit, read from the project. */
export function brandOnFilm(project: StoryboardProject): BrandOnFilm {
  const settings = project.export;
  const logo = project.brand?.logo?.path ?? null;
  const seconds = cardSeconds(project);
  return {
    title: settings.titleCard ? { text: cardText(project, "title"), seconds } : null,
    end: settings.endCard ? { text: cardText(project, "end"), seconds } : null,
    logo: logo && settings.showLogo !== false ? logo : null,
    cardLogo: logo,
    corner: project.brand?.logoCorner ?? "bottom-right",
    look: cardLook(project),
  };
}

/** Seconds the cards add to the exported film (0 with no card). */
export function brandSeconds(project: StoryboardProject): number {
  const settings = project.export;
  return (settings.titleCard ? cardSeconds(project) : 0) + (settings.endCard ? cardSeconds(project) : 0);
}
