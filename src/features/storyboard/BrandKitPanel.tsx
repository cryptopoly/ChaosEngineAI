/**
 * The brand kit on the Cast tab (S15): how the style reaches the still model, an optional logo and its corner,
 * and the caption look (a font installed on this machine, a colour from the palette or any colour, a place on
 * the frame), with a live preview drawn on the first style picture. Every change is one `set_brand` op, so
 * Undo works. Pictures only: nothing is fetched from the web.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { getStoryboardFonts } from "../../api/storyboardPlan";
import type { ImageOutputArtifact, StoryboardCaptionLook, StoryboardLogoCorner, StoryboardOpArgs, StoryboardProject, StoryboardStyleUse } from "../../types";
import { PaletteSwatches } from "./PaletteSwatches";
import { brandOf, isHexColour, projectPalette, stylesOf } from "./planLogic";
import { useProjectFileUrl } from "./ProjectMedia";
import { ReferenceImagePicker } from "./ReferenceImagePicker";
import type { UseStoryboardResult } from "./useStoryboard";

export interface BrandKitPanelProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  imageGallery: ImageOutputArtifact[];
}

const CORNERS: StoryboardLogoCorner[] = ["top-left", "top-right", "bottom-left", "bottom-right"];
const POSITIONS: StoryboardCaptionLook["position"][] = ["top", "middle", "bottom"];
const STYLE_USES: StoryboardStyleUse[] = ["auto", "words", "off"];
const BASICS = ["#ffffff", "#000000"];

let fontCache: Promise<string[]> | null = null;

function useFonts(): { fonts: string[]; failed: boolean } {
  const [fonts, setFonts] = useState<string[]>([]);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;
    fontCache = fontCache ?? getStoryboardFonts();
    fontCache
      .then((list) => live && setFonts(list))
      .catch(() => {
        fontCache = null;
        if (live) setFailed(true);
      });
    return () => {
      live = false;
    };
  }, []);
  return { fonts, failed };
}

export function BrandKitPanel({ story, project, imageGallery }: BrandKitPanelProps) {
  const { t } = useTranslation("studio");
  const brand = brandOf(project);
  const look = brand.captions;
  const palette = projectPalette(project);
  const { fonts, failed } = useFonts();
  const [pickerOpen, setPickerOpen] = useState(false);
  const [custom, setCustom] = useState(look.colour);
  useEffect(() => setCustom(look.colour), [look.colour]);
  const backdropRef = stylesOf(project).find((r) => r.images.length > 0)?.images[0] ?? null;
  const backdrop = useProjectFileUrl(project.id, backdropRef?.path ?? null, backdropRef?.hash, { thumb: 512 });
  const logo = useProjectFileUrl(project.id, brand.logo?.path ?? null, brand.logo?.hash, { thumb: 256 });
  const set = (args: StoryboardOpArgs["set_brand"]) => void story.op("set_brand", args);

  const cornerLabel: Record<StoryboardLogoCorner, string> = {
    "top-left": t("board.style.corner.topLeft", { defaultValue: "Top left" }),
    "top-right": t("board.style.corner.topRight", { defaultValue: "Top right" }),
    "bottom-left": t("board.style.corner.bottomLeft", { defaultValue: "Bottom left" }),
    "bottom-right": t("board.style.corner.bottomRight", { defaultValue: "Bottom right" }),
  };
  const positionLabel: Record<StoryboardCaptionLook["position"], string> = {
    top: t("board.style.position.top", { defaultValue: "Top" }),
    middle: t("board.style.position.middle", { defaultValue: "Middle" }),
    bottom: t("board.style.position.bottom", { defaultValue: "Bottom" }),
  };
  const useLabel: Record<StoryboardStyleUse, { title: string; body: string }> = {
    auto: {
      title: t("board.style.use.auto", { defaultValue: "Automatic" }),
      body: t("board.style.use.autoBody", { defaultValue: "As reference pictures when the still model takes them, otherwise as words in the prompt." }),
    },
    words: {
      title: t("board.style.use.words", { defaultValue: "Always as words" }),
      body: t("board.style.use.wordsBody", { defaultValue: "The look and colours are written into each prompt; the pictures are not sent." }),
    },
    off: {
      title: t("board.style.use.off", { defaultValue: "Off" }),
      body: t("board.style.use.offBody", { defaultValue: "Keep the style here but leave the shots alone." }),
    },
  };
  const ratio = project.defaults.exportPreset.replace(":", " / ");

  return (
    <section className="brand-kit" aria-labelledby="brand-kit-title">
      <div className="cast-tab__group-head">
        <div>
          <h4 id="brand-kit-title">{t("board.style.kitTitle", { defaultValue: "Brand kit" })}</h4>
          <p className="muted-text">{t("board.style.kitSubtitle", { defaultValue: "Optional. A logo and a caption look, saved with the project for the export step." })}</p>
        </div>
      </div>

      <div className="brand-kit__grid">
        <div className="brand-kit__controls">
          <fieldset className="brand-kit__field">
            <legend>{t("board.style.useTitle", { defaultValue: "How shots use the style" })}</legend>
            <div className="brand-kit__radios">
              {STYLE_USES.map((value) => (
                <label key={value} className={`brand-kit__radio${brand.styleUse === value ? " is-active" : ""}`}>
                  <input type="radio" name={`style-use-${project.id}`} value={value} checked={brand.styleUse === value} disabled={story.busy} onChange={() => set({ styleUse: value })} />
                  <span>
                    <strong>{useLabel[value].title}</strong>
                    <span className="muted-text">{useLabel[value].body}</span>
                  </span>
                </label>
              ))}
            </div>
          </fieldset>

          <fieldset className="brand-kit__field">
            <legend>{t("board.style.logoTitle", { defaultValue: "Logo" })}</legend>
            <div className="brand-kit__logo-row">
              <span className="brand-kit__logo">
                {logo ? <img src={logo} alt={t("board.style.logoAlt", { defaultValue: "The project's logo" })} /> : <span className="muted-text">{t("board.style.noLogo", { defaultValue: "None" })}</span>}
              </span>
              <div className="brand-kit__logo-actions">
                <button type="button" className="secondary-button" disabled={story.busy} onClick={() => setPickerOpen(true)}>
                  {brand.logo ? t("board.style.changeLogo", { defaultValue: "Change logo" }) : t("board.style.addLogo", { defaultValue: "Add logo" })}
                </button>
                {brand.logo ? (
                  <button type="button" className="secondary-button plan-quiet-button" disabled={story.busy} onClick={() => set({ clearLogo: true })}>
                    {t("board.style.removeLogo", { defaultValue: "Remove" })}
                  </button>
                ) : null}
              </div>
            </div>
            <label className="field brand-kit__inline">
              <span>{t("board.style.cornerLabel", { defaultValue: "Corner" })}</span>
              <select className="text-input" value={brand.logoCorner} disabled={story.busy || !brand.logo} onChange={(event) => set({ logoCorner: event.target.value as StoryboardLogoCorner })}>
                {CORNERS.map((corner) => (
                  <option key={corner} value={corner}>{cornerLabel[corner]}</option>
                ))}
              </select>
            </label>
          </fieldset>

          <fieldset className="brand-kit__field">
            <legend>{t("board.style.captionTitle", { defaultValue: "Caption look" })}</legend>
            <label className="field">
              <span>{t("board.style.fontLabel", { defaultValue: "Font" })}</span>
              <select
                className="text-input"
                value={look.font ?? ""}
                disabled={story.busy}
                onChange={(event) => set({ captions: event.target.value ? { font: event.target.value } : { clearFont: true } })}
              >
                <option value="">{t("board.style.fontDefault", { defaultValue: "Default" })}</option>
                {look.font && !fonts.includes(look.font) ? <option value={look.font}>{look.font}</option> : null}
                {fonts.map((font) => (
                  <option key={font} value={font}>{font}</option>
                ))}
              </select>
              {failed ? <span className="muted-text">{t("board.style.fontsFailed", { defaultValue: "Could not list the fonts on this machine." })}</span> : null}
            </label>
            <div className="field">
              <span id={`caption-colour-${project.id}`}>{t("board.style.colourLabel", { defaultValue: "Colour" })}</span>
              <div className="brand-kit__colours" role="group" aria-labelledby={`caption-colour-${project.id}`}>
                <PaletteSwatches
                  colours={[...BASICS, ...palette.filter((c) => !BASICS.includes(c))]}
                  label={t("board.style.colourChoices", { defaultValue: "Caption colours" })}
                  selected={look.colour}
                  disabled={story.busy}
                  onPick={(colour) => set({ captions: { colour } })}
                />
                <label className="brand-kit__custom">
                  <span className="sr-only">{t("board.style.customColour", { defaultValue: "Any colour" })}</span>
                  <input
                    type="color"
                    value={custom}
                    disabled={story.busy}
                    onChange={(event) => setCustom(event.target.value)}
                    onBlur={() => isHexColour(custom.toLowerCase()) && custom.toLowerCase() !== look.colour && set({ captions: { colour: custom.toLowerCase() } })}
                  />
                </label>
              </div>
            </div>
            <div className="field">
              <span id={`caption-position-${project.id}`}>{t("board.style.positionLabel", { defaultValue: "Place" })}</span>
              <div className="plan-chips" role="group" aria-labelledby={`caption-position-${project.id}`}>
                {POSITIONS.map((position) => (
                  <button
                    key={position}
                    type="button"
                    className={`plan-chip${look.position === position ? " is-active" : ""}`}
                    aria-pressed={look.position === position}
                    disabled={story.busy}
                    onClick={() => set({ captions: { position } })}
                  >
                    {positionLabel[position]}
                  </button>
                ))}
              </div>
            </div>
          </fieldset>
        </div>

        <figure className="brand-kit__preview" aria-label={t("board.style.previewLabel", { defaultValue: "Preview of the caption and logo" })}>
          <div className="brand-kit__frame" style={{ aspectRatio: ratio, backgroundImage: backdrop ? `url(${backdrop})` : undefined }}>
            {logo ? <img className={`brand-kit__frame-logo brand-kit__frame-logo--${brand.logoCorner}`} src={logo} alt="" /> : null}
            <span
              className={`brand-kit__frame-caption brand-kit__frame-caption--${look.position}`}
              style={{ color: look.colour, fontFamily: look.font ? `"${look.font}", var(--font-latin)` : undefined }}
            >
              {t("board.style.previewText", { defaultValue: "Fresh leaves, poured slowly." })}
            </span>
          </div>
          <figcaption className="muted-text">
            {backdrop
              ? t("board.style.previewOnStyle", { defaultValue: "Shown on your first style picture." })
              : t("board.style.previewPlain", { defaultValue: "Add a style picture to see the caption on it." })}
          </figcaption>
        </figure>
      </div>

      {pickerOpen ? (
        <ReferenceImagePicker
          title={t("board.style.logoPicker", { defaultValue: "Choose a logo" })}
          project={project}
          imageGallery={imageGallery}
          room={1}
          onPick={(source) => story.op("set_brand", { logo: source })}
          onClose={() => setPickerOpen(false)}
        />
      ) : null}
    </section>
  );
}
