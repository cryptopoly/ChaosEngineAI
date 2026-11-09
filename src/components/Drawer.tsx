/**
 * A side drawer over the page: a backdrop, a titled panel from the right with a Close button. Escape and a click on
 * the backdrop close it; focus moves in when it opens, Tab stays inside, and focus returns to what had it before
 * (the control that opened it) when it closes. Full width on a phone. Shares the `.drawer` look with the shot editor.
 */
import { useId, useRef, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useDialogFocus } from "../hooks/useDialogFocus";

export interface DrawerProps {
  title: string;
  onClose: () => void;
  children: ReactNode;
  /** A line under the title. */
  subtitle?: ReactNode;
  className?: string;
}

/** Render it only while open: mounting runs the focus handling, unmounting returns focus. */
export function Drawer({ title, onClose, children, subtitle, className }: DrawerProps) {
  const { t } = useTranslation("common");
  const ref = useRef<HTMLElement | null>(null);
  const titleId = useId();
  useDialogFocus(ref, { onEscape: onClose });
  return (
    <div className="drawer-layer">
      <div className="drawer-backdrop" aria-hidden="true" onClick={onClose} />
      <aside ref={ref} className={`drawer drawer--wide${className ? ` ${className}` : ""}`} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <div className="drawer__head">
          <div>
            <h3 id={titleId}>{title}</h3>
            {subtitle ? <p className="muted-text drawer__subtitle">{subtitle}</p> : null}
          </div>
          <button type="button" className="secondary-button" onClick={onClose}>
            {t("actions.close", { defaultValue: "Close" })}
          </button>
        </div>
        <div className="drawer__body">{children}</div>
      </aside>
    </div>
  );
}
