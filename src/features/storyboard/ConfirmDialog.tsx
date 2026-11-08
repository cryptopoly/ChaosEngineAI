/** A confirm dialog on the shared modal classes. Escape or the overlay cancels. */
import { useRef, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useDialogFocus } from "../../hooks/useDialogFocus";

export interface ConfirmDialogProps {
  title: string;
  body: ReactNode;
  confirmLabel: string;
  danger?: boolean;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({ title, body, confirmLabel, danger, busy, onConfirm, onCancel }: ConfirmDialogProps) {
  const { t } = useTranslation("studio");
  const dialog = useRef<HTMLDivElement>(null);
  useDialogFocus(dialog, { onEscape: onCancel });

  return (
    <div className="modal-overlay" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <div ref={dialog} className="modal-content" role="alertdialog" aria-modal="true" aria-label={title}>
        <div className="modal-header">
          <h3>{title}</h3>
        </div>
        <div className="modal-body">{body}</div>
        <div className="modal-footer">
          <button type="button" className="secondary-button" onClick={onCancel} data-autofocus={danger ? "" : undefined}>
            {t("storyboard.cancel", { defaultValue: "Cancel" })}
          </button>
          <button
            type="button"
            className={danger ? "secondary-button danger-button" : "primary-button"}
            disabled={busy}
            onClick={onConfirm}
            data-autofocus={danger ? undefined : ""}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
