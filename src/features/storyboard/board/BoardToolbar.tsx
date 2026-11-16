/**
 * The freestyle board's toolbar (S-M2): zoom out, the zoom level, zoom in, Fit; "Select several" for touch (each tap
 * adds or removes a card); Add pictures from files; how many cards are selected with Clear; and "Order these", which
 * writes the film's shot order from where the cards sit (left to right, then top to bottom) with one `reorder` op.
 */
import { useRef } from "react";
import { useTranslation } from "react-i18next";
import { MinusIcon, PlusIcon } from "../../../components/icons/ActionIcons";

export interface BoardToolbarProps {
  zoom: number;
  multi: boolean;
  selectedCount: number;
  /** Shots among the selection; two or more order only those. */
  selectedShots: number;
  canOrder: boolean;
  busy: boolean;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFit: () => void;
  onMulti: (on: boolean) => void;
  onClear: () => void;
  onOrder: () => void;
  onAddFiles: (files: File[]) => void;
}

function FitIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
    </svg>
  );
}

export function BoardToolbar(props: BoardToolbarProps) {
  const { t } = useTranslation("studio");
  const files = useRef<HTMLInputElement>(null);
  const { zoom, multi, selectedCount, selectedShots, canOrder, busy } = props;
  return (
    <div className="board-toolbar" role="toolbar" aria-label={t("board.freestyle.toolbar", { defaultValue: "Board tools" })}>
      <div className="board-toolbar__group">
        <button type="button" className="secondary-button icon-button" aria-label={t("board.freestyle.zoomOut", { defaultValue: "Zoom out" })} title={t("board.freestyle.zoomOut", { defaultValue: "Zoom out" })} onClick={props.onZoomOut}>
          <MinusIcon aria-hidden="true" />
        </button>
        <span className="board-toolbar__zoom" aria-live="polite">{`${Math.round(zoom * 100)}%`}</span>
        <button type="button" className="secondary-button icon-button" aria-label={t("board.freestyle.zoomIn", { defaultValue: "Zoom in" })} title={t("board.freestyle.zoomIn", { defaultValue: "Zoom in" })} onClick={props.onZoomIn}>
          <PlusIcon aria-hidden="true" />
        </button>
        <button type="button" className="secondary-button" title={t("board.freestyle.fitTitle", { defaultValue: "Show every card" })} onClick={props.onFit}>
          <FitIcon /> {t("board.freestyle.fit", { defaultValue: "Fit" })}
        </button>
      </div>
      <div className="board-toolbar__group">
        <button type="button" className={`secondary-button${multi ? " is-on" : ""}`} aria-pressed={multi} onClick={() => props.onMulti(!multi)}>
          {t("board.freestyle.selectSeveral", { defaultValue: "Select several" })}
        </button>
        <button type="button" className="secondary-button" disabled={busy} onClick={() => files.current?.click()}>
          {t("board.freestyle.addPictures", { defaultValue: "Add pictures" })}
        </button>
        <input
          ref={files}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          multiple
          hidden
          onChange={(event) => {
            const chosen = Array.from(event.target.files ?? []);
            event.target.value = "";
            if (chosen.length > 0) props.onAddFiles(chosen);
          }}
        />
      </div>
      <div className="board-toolbar__group board-toolbar__group--end">
        {selectedCount > 0 ? (
          <>
            <span className="muted-text" role="status">
              {t("board.freestyle.selected", { defaultValue: "{n, plural, one {# card selected} other {# cards selected}}", n: selectedCount })}
            </span>
            <button type="button" className="secondary-button plan-quiet-button" onClick={props.onClear}>
              {t("board.freestyle.clear", { defaultValue: "Clear" })}
            </button>
          </>
        ) : null}
        <button
          type="button"
          className="primary-button"
          disabled={busy || !canOrder}
          title={t("board.freestyle.orderTitle", { defaultValue: "Sets the film's shot order from where the cards sit: left to right, then top to bottom. Undo puts it back." })}
          onClick={props.onOrder}
        >
          {selectedShots >= 2
            ? t("board.freestyle.orderThese", { defaultValue: "Order these {n}", n: selectedShots })
            : t("board.freestyle.orderAll", { defaultValue: "Order all shots" })}
        </button>
      </div>
    </div>
  );
}
