/**
 * Accessibility switch (Settings > General): stronger borders on fields and buttons.
 *
 * The saved setting is `strongControlBorders` (on by default). App.tsx mirrors it onto
 * `<html data-strong-borders>`; this panel also sets the attribute when the switch is pressed so the
 * look changes at once, then saves through `updateSettings`.
 */

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { updateSettings } from "../../api";

/** Strong borders are on unless the page has been switched off. */
export function readStrongBorders(): boolean {
  return typeof document === "undefined" || document.documentElement.dataset.strongBorders !== "off";
}

export async function saveStrongBorders(on: boolean): Promise<void> {
  if (typeof document !== "undefined") {
    if (on) delete document.documentElement.dataset.strongBorders;
    else document.documentElement.dataset.strongBorders = "off";
  }
  await updateSettings({ strongControlBorders: on });
}

export function AccessibilityPanel() {
  const { t } = useTranslation("common");
  const [on, setOn] = useState(readStrongBorders);

  // The saved value reaches the page after the workspace loads (App.tsx), so follow the attribute.
  useEffect(() => {
    const sync = () => setOn(readStrongBorders());
    const observer = new MutationObserver(sync);
    sync();
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-strong-borders"] });
    return () => observer.disconnect();
  }, []);

  const toggle = () => {
    const next = !on;
    setOn(next);
    void saveStrongBorders(next);
  };

  return (
    <Panel title={t("accessibilityPanel.title", { defaultValue: "Accessibility" })}>
      <div className="control-stack">
        <div className="field-row" style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span className="field-label" id="strong-borders-label">
            {t("accessibilityPanel.strongBorders", { defaultValue: "Stronger borders on fields and buttons" })}
          </span>
          <button
            type="button"
            role="switch"
            aria-checked={on}
            aria-labelledby="strong-borders-label"
            className={on ? "segment active" : "segment"}
            data-testid="strong-borders-switch"
            onClick={toggle}
          >
            {on ? t("accessibilityPanel.on", { defaultValue: "On" }) : t("accessibilityPanel.off", { defaultValue: "Off" })}
          </button>
        </div>
        <p className="help-text">
          {t("accessibilityPanel.note", {
            defaultValue: "Outlines inputs and buttons with a higher-contrast line so they are easier to see. Applies immediately.",
          })}
        </p>
      </div>
    </Panel>
  );
}
