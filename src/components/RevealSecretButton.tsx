import { useTranslation } from "react-i18next";

interface RevealSecretButtonProps {
  revealed: boolean;
  onToggle: () => void;
  disabled?: boolean;
}

/** Reveal / Hide for a masked key in code samples. Copy buttons next to it copy the real key either way. */
export function RevealSecretButton({ revealed, onToggle, disabled }: RevealSecretButtonProps) {
  const { t } = useTranslation("common");
  return (
    <button
      className="secondary-button"
      type="button"
      aria-pressed={revealed}
      disabled={disabled}
      data-testid="reveal-secret"
      title={t("secretMask.note", { defaultValue: "The key is hidden on screen. Copy always copies the real key." })}
      onClick={onToggle}
    >
      {revealed ? t("serverTab.hide", { defaultValue: "Hide" }) : t("serverTab.reveal", { defaultValue: "Reveal" })}
    </button>
  );
}
