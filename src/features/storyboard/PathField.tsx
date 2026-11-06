/**
 * A file path on this machine: typed or chosen with the desktop file dialog. The backend takes a path
 * (never an upload), so a typed path works where there is no dialog. `onSubmit` fires on Enter or the button.
 */
import { useState } from "react";
import { useTranslation } from "react-i18next";

export interface PathFieldProps {
  label: string;
  buttonLabel: string;
  extensions: string[];
  disabled?: boolean;
  onSubmit: (path: string) => void;
}

export function PathField({ label, buttonLabel, extensions, disabled, onSubmit }: PathFieldProps) {
  const { t } = useTranslation("studio");
  const [value, setValue] = useState("");
  const [pickError, setPickError] = useState(false);

  async function choose() {
    setPickError(false);
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({ multiple: false, title: label, filters: [{ name: label, extensions }] });
      if (typeof selected === "string" && selected) setValue(selected);
    } catch {
      setPickError(true);
    }
  }

  function submit() {
    const path = value.trim();
    if (!path) return;
    onSubmit(path);
    setValue("");
  }

  return (
    <div className="path-field">
      <span className="eyebrow">{label}</span>
      <div className="path-field__row">
        <input
          type="text"
          value={value}
          placeholder={t("storyboard.pathPlaceholder", { defaultValue: "/path/to/file" })}
          aria-label={label}
          disabled={disabled}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              submit();
            }
          }}
        />
        <button type="button" className="secondary-button" disabled={disabled} onClick={() => void choose()}>
          {t("inputImage.chooseFile", { defaultValue: "Choose file" })}
        </button>
        <button type="button" className="primary-button" disabled={disabled || !value.trim()} onClick={submit}>
          {buttonLabel}
        </button>
      </div>
      {pickError ? (
        <p className="muted-text">
          {t("storyboard.pickUnavailable", { defaultValue: "The file picker is not available here. Type the path instead." })}
        </p>
      ) : null}
    </div>
  );
}
