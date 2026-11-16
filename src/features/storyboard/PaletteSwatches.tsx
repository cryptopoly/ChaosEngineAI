/**
 * A style's colours as a row of swatches (S15). Read-only, or, with `onRemove`, each swatch is a button that
 * takes that colour out of the palette; with `onPick`, a swatch chooses that colour (the caption colour).
 */
import { useTranslation } from "react-i18next";
import { CloseIcon } from "../../components/icons/ActionIcons";
import { inkOn } from "./planLogic";

export interface PaletteSwatchesProps {
  colours: string[];
  label: string;
  size?: "sm" | "md";
  selected?: string | null;
  disabled?: boolean;
  onRemove?: (index: number) => void;
  onPick?: (colour: string) => void;
}

export function PaletteSwatches({ colours, label, size = "md", selected, disabled, onRemove, onPick }: PaletteSwatchesProps) {
  const { t } = useTranslation("studio");
  if (colours.length === 0) return null;
  return (
    <ul className={`palette-swatches palette-swatches--${size}`} aria-label={label}>
      {colours.map((colour, index) => {
        const style = { background: colour, color: inkOn(colour) };
        if (onRemove) {
          return (
            <li key={`${colour}-${index}`}>
              <button
                type="button"
                className="palette-swatch palette-swatch--removable"
                style={style}
                disabled={disabled}
                aria-label={t("board.style.removeColour", { defaultValue: "Remove colour {colour}", colour })}
                title={t("board.style.removeColour", { defaultValue: "Remove colour {colour}", colour })}
                onClick={() => onRemove(index)}
              >
                <CloseIcon aria-hidden="true" />
              </button>
            </li>
          );
        }
        if (onPick) {
          return (
            <li key={`${colour}-${index}`}>
              <button
                type="button"
                className={`palette-swatch${selected === colour ? " is-selected" : ""}`}
                style={style}
                disabled={disabled}
                aria-pressed={selected === colour}
                aria-label={t("board.style.useColour", { defaultValue: "Use colour {colour}", colour })}
                title={colour}
                onClick={() => onPick(colour)}
              />
            </li>
          );
        }
        return (
          <li key={`${colour}-${index}`}>
            <span className="palette-swatch" style={style} title={colour} role="img" aria-label={colour} />
          </li>
        );
      })}
    </ul>
  );
}
