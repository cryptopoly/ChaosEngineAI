/** A confirm dialog on the shared modal classes. Escape or the overlay cancels. */
import { useEffect, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

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
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onCancel]);

  return (
    <div className="modal-overlay" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <div className="modal-content" role="alertdialog" aria-modal="true" aria-label={title}>
        <div className="modal-header">
          <h3>{title}</h3>
        </div>
        <div className="modal-body">{body}</div>
        <div className="modal-footer">
          <button type="button" className="secondary-button" onClick={onCancel}>
            {t("storyboard.cancel", { defaultValue: "Cancel" })}
          </button>
          <button
            type="button"
            className={danger ? "secondary-button danger-button" : "primary-button"}
            disabled={busy}
            onClick={onConfirm}
            autoFocus
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
