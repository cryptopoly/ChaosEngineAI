/**
 * The rough cut's picture (S9): one layer per shot (a muted clip, a held still, or a card for a shot with no
 * picture yet), the title and end cards and the corner logo of the brand kit (S15), the captions in the brand
 * look, and the loading bar. The player hook shows, fades and steps the layers; this only draws them. Used by
 * the "Play film" dialog and the Mixer's own transport.
 */
import type { CSSProperties } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardCaptionLook, StoryboardSize } from "../../../types";
import { draftTag, type TFn } from "../qualityText";
import { logoVisibleAt, type CutPiece, type RoughCut } from "./roughCutLogic";
import type { UseRoughCutResult } from "./useRoughCut";
import "./brand.css";

export interface RoughCutStageProps {
  cut: RoughCut;
  player: UseRoughCutResult;
  /** The frame shape: the export size. */
  width: number;
  height: number;
  /** The caption on screen now, or null. */
  caption: string | null;
  captionLook?: StoryboardCaptionLook | null;
  /** The shot chip and Draft badge over the picture. */
  current?: CutPiece;
  shotLabel: (piece: CutPiece) => string;
  className?: string;
  /** The film's size (`guide.quality.target`), named in the Draft tag's explanation. */
  target?: StoryboardSize | null;
}

/** Inline style for a family name the user picked: quoted, and only the characters a family name has. */
export function fontStack(family: string | null | undefined): string | undefined {
  const clean = (family ?? "").replace(/[^A-Za-z0-9 \-]/g, "").trim();
  return clean ? `"${clean}", var(--font-latin)` : undefined;
}

export function RoughCutStage({ cut, player, width, height, caption, captionLook, current, shotLabel, className, target }: RoughCutStageProps) {
  const { t } = useTranslation("studio");
  const { timeS, ready } = player;
  const brand = cut.brand;
  const short = Math.min(width, height);
  const stageStyle = { "--cut-w": width, "--cut-h": height } as CSSProperties; // stage shape: the export frame
  const margin = { x: `${(4 * short) / width}%`, y: `${(4 * short) / height}%` };
  // The Draft chip carries the size the take was made at, with a plain explanation as its tooltip.
  const draft = current?.draft ? draftTag(t as unknown as TFn, current.quality, target) : null;
  const draftTitle = draft?.title;
  const logoUrl = brand?.logo ? player.urlOf(brand.logo) : null;
  const cardLogoUrl = brand?.cardLogo ? player.urlOf(brand.cardLogo) : null;
  const corner = brand?.corner ?? "bottom-right";
  const logoStyle: CSSProperties = {
    [corner.startsWith("top") ? "top" : "bottom"]: margin.y,
    [corner.endsWith("left") ? "left" : "right"]: margin.x,
  };
  const position = captionLook?.position ?? "bottom";
  const captionStyle: CSSProperties = {
    color: captionLook?.colour,
    fontFamily: fontStack(captionLook?.font),
  };
  return (
    <div className={`rough-cut__stage${className ? ` ${className}` : ""}`} style={stageStyle}>
      {cut.pieces.map((piece) => {
        if (piece.media.kind === "card") {
          const look = brand?.look;
          return (
            <div
              key={piece.shotId}
              ref={(el) => player.registerLayer(piece.shotId, el)}
              className={`rough-cut__layer rough-cut__brand-card${cardLogoUrl && piece.media.text ? " rough-cut__brand-card--logo" : ""}`}
              style={{ background: look?.background, color: look?.colour, fontFamily: fontStack(look?.font) }}
              data-card={piece.media.which}
            >
              {cardLogoUrl ? <img className="rough-cut__brand-card-logo" src={cardLogoUrl} alt="" /> : null}
              {piece.media.text ? <span className="rough-cut__brand-card-text">{piece.media.text}</span> : null}
            </div>
          );
        }
        if (piece.media.kind === "clip") {
          const url = player.urlOf(piece.media.clip);
          return (
            <video
              key={piece.shotId}
              ref={(el) => player.registerLayer(piece.shotId, el)}
              className="rough-cut__layer"
              src={url ?? undefined}
              muted
              playsInline
              preload="auto"
              aria-hidden="true"
            />
          );
        }
        if (piece.media.kind === "still") {
          const url = player.urlOf(piece.media.still);
          return (
            <div key={piece.shotId} ref={(el) => player.registerLayer(piece.shotId, el)} className="rough-cut__layer rough-cut__layer--still">
              {url ? <img src={url} alt="" /> : null}
              <span className="rough-cut__hold">
                {t("board.preview.heldStill", { defaultValue: "No clip yet · still held for {seconds} s", seconds: Math.round(piece.duration * 10) / 10 })}
              </span>
            </div>
          );
        }
        return (
          <div key={piece.shotId} ref={(el) => player.registerLayer(piece.shotId, el)} className="rough-cut__layer rough-cut__layer--card">
            <strong>{shotLabel(piece)}</strong>
            <span>{t("board.preview.noPicture", { defaultValue: "No picture yet · {seconds} s", seconds: Math.round(piece.duration * 10) / 10 })}</span>
          </div>
        );
      })}
      {logoUrl && logoVisibleAt(cut, timeS) ? <img className="rough-cut__brand-logo" src={logoUrl} alt="" style={logoStyle} /> : null}
      {current ? (
        <div className="rough-cut__badges">
          <span className="rough-cut__chip">{shotLabel(current)}</span>
          {current.draft ? (
            <span className="rough-cut__chip rough-cut__chip--draft" title={draftTitle}>
              {draft?.label ?? t("board.preview.draft", { defaultValue: "Draft" })}
            </span>
          ) : null}
        </div>
      ) : null}
      {caption ? (
        <p className={`rough-cut__caption rough-cut__caption--${position}`} style={captionStyle} aria-live="off">
          {caption}
        </p>
      ) : null}
      {!ready ? (
        <div className="rough-cut__loading" role="status">
          <span>
            {t("board.preview.loading", { defaultValue: "Loading the film: {done} of {total} files", done: player.load.done, total: player.load.total })}
          </span>
          <span className="rough-cut__loading-bar" aria-hidden="true">
            <span style={{ width: `${player.load.total ? (player.load.done / player.load.total) * 100 : 0}%` }} />
          </span>
        </div>
      ) : null}
    </div>
  );
}
