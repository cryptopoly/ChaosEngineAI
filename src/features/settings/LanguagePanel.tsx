/**
 * FU-042 — Language + clock-format settings panel.
 *
 * Renders a dropdown of supported locales (each shown in its own
 * endonym per i18n convention), a 12h / 24h / system clock override,
 * and a "Help improve translations" link that opens the relevant
 * locale's JSON tree on GitHub.
 *
 * On dropdown change we *both* (a) call ``changeLocale`` so i18next
 * re-renders the live UI immediately, and (b) mutate ``settingsDraft``
 * so the user's save click persists the choice through the existing
 * ``onSaveSettings`` flow.  Without (a) the user would have to save +
 * restart before seeing the new locale; without (b) the choice would
 * vanish on reload.
 */

import { useTranslation } from "react-i18next";
import { invoke } from "@tauri-apps/api/core";
import { Panel } from "../../components/Panel";
import { changeLocale, SUPPORTED_LOCALES, type SupportedLocale } from "../../i18n";
import enMeta from "../../locales/en/meta.json";
import type { SettingsDraft } from "../../types/chat";
import type { SetStateAction } from "react";

/**
 * Push the picked locale into the Tauri shell so native menu / tray /
 * updater dialog strings flip too.  Fails silently in browser dev mode
 * (vite serve without Tauri) — the React UI re-renders regardless and
 * the user just doesn't see native chrome update.
 */
async function syncTauriLocale(locale: string): Promise<void> {
  try {
    await invoke("set_app_locale", { locale });
  } catch {
    // Browser dev mode or older Tauri build without the command — fine.
  }
}

interface LocaleEntry {
  code: string;
  endonym: string;
  rtl: boolean;
}

const REPO_TRANSLATIONS_BASE =
  "https://github.com/anthropics/ChaosEngineAI/tree/main/src/locales";

export interface LanguagePanelProps {
  settingsDraft: SettingsDraft;
  onSettingsDraftChange: (action: SetStateAction<SettingsDraft>) => void;
  onCopyText: (text: string) => void;
}

export function LanguagePanel({
  settingsDraft,
  onSettingsDraftChange,
  onCopyText,
}: LanguagePanelProps) {
  const { t, i18n } = useTranslation("common");

  const locales: LocaleEntry[] = (enMeta as { supportedLocales: LocaleEntry[] }).supportedLocales;
  const selected = settingsDraft.locale || "system";
  const activeLocale = i18n.language;

  // Resolve target GitHub URL for the *currently-active* locale (not the
  // pending dropdown value) — the user sees something to fix in the live
  // language, then jumps to that locale's tree to edit + open a PR.
  const githubUrl = `${REPO_TRANSLATIONS_BASE}/${activeLocale || "en"}`;

  const handleLocaleChange = (next: string) => {
    onSettingsDraftChange((current) => ({ ...current, locale: next }));
    // Apply live so the UI flips immediately.  ``"system"`` maps to the
    // detected default (already what i18next has loaded) so the
    // ``changeLocale`` call is a no-op in that case — but we still call
    // it to surface the layout pass for the *current* navigator default.
    if (next === "system") {
      // Re-detect from navigator on the fly. Simplest impl: do nothing —
      // i18next stays on whatever it picked at init.  The persisted
      // ``"system"`` value means future cold-starts will re-detect; the
      // live session stays on the current pick.
      return;
    }
    if (SUPPORTED_LOCALES.includes(next as SupportedLocale)) {
      void changeLocale(next as SupportedLocale);
      void syncTauriLocale(next);
    }
  };

  const handleClockChange = (next: SettingsDraft["clockFormat"]) => {
    onSettingsDraftChange((current) => ({ ...current, clockFormat: next }));
  };

  return (
    <Panel
      title={t("languagePanel.title", { defaultValue: "Settings" })}
      subtitle={t("languagePanel.subtitle", {
        defaultValue:
          "Choose the language used across the app and how clock times are shown. New choices apply immediately and save with your other settings.",
      })}
    >
      <div className="control-stack">
        <label className="field-row" htmlFor="settings-language-select">
          <span className="field-label">{t("languagePanel.label", { defaultValue: "Language" })}</span>
          <select
            id="settings-language-select"
            className="text-input"
            value={selected}
            onChange={(event) => handleLocaleChange(event.target.value)}
          >
            <option value="system">{t("languagePanel.systemDefault", { defaultValue: "System default" })}</option>
            {locales.map((entry) => (
              <option key={entry.code} value={entry.code}>
                {entry.endonym}
              </option>
            ))}
          </select>
        </label>
        <p className="help-text">
          {t("languagePanel.helpText", {
            defaultValue:
              "Each language is shown in its own writing — choose the row that reads naturally to you. Missing translations fall back to English automatically; if you spot something off, the link below jumps straight to the JSON file on GitHub so you can suggest a fix.",
          })}
        </p>

        <div
          className="segmented"
          role="radiogroup"
          aria-label={t("languagePanel.clockFormatAria", { defaultValue: "Clock format" })}
        >
          {(["system", "12h", "24h"] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              role="radio"
              aria-checked={settingsDraft.clockFormat === mode}
              className={settingsDraft.clockFormat === mode ? "segment active" : "segment"}
              onClick={() => handleClockChange(mode)}
            >
              {mode === "system"
                ? t("languagePanel.clockSystem", { defaultValue: "System" })
                : mode === "12h"
                  ? t("languagePanel.clock12h", { defaultValue: "12-hour" })
                  : t("languagePanel.clock24h", { defaultValue: "24-hour" })}
            </button>
          ))}
        </div>
        <p className="help-text">
          {t("languagePanel.clockHint", {
            defaultValue:
              "System follows your language's default (12-hour for US English, 24-hour elsewhere). Pick a side if you'd rather override that.",
          })}
        </p>

        <div className="field-row">
          <span className="field-label">
            {t("languagePanel.helpImprove", { defaultValue: "Help improve translations" })}
          </span>
          <div className="row-gap">
            <a
              className="link-button"
              href={githubUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              {t("languagePanel.openOnGithub", {
                locale: activeLocale || "en",
                defaultValue: `Open ${activeLocale || "en"} on GitHub →`,
              })}
            </a>
            <button
              type="button"
              className="ghost-button"
              onClick={() => onCopyText(githubUrl)}
            >
              {t("actions.copy")}
            </button>
          </div>
        </div>
      </div>
    </Panel>
  );
}
