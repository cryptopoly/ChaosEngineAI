/**
 * FU-042 — first-launch locale banner.
 *
 * One-shot prompt for users whose OS / browser default isn't English.
 * Detection runs on mount; the banner appears only when:
 *
 *   1. The persisted ``settings.locale`` is the ``"system"`` sentinel
 *      (i.e. the user has never explicitly picked a language).
 *   2. The detected locale is a *supported* locale other than ``en``.
 *   3. The dismissal flag (``localStorage`` key
 *      ``chaosengine.localeBannerDismissed``) hasn't been set yet.
 *
 * The app has already started in the detected language by the time
 * the banner shows (``main.tsx`` boots i18n with the same
 * ``detectLocale()``), so the banner does not ask whether to switch to
 * it (review L-1: "Switch the app to Deutsch?" in German read as
 * nonsense). It states what happened, in that language, and offers the
 * way back:
 *
 *   "The app is in {endonym}, your system language."
 *   [Keep {endonym}]   [Use English]
 *
 * "Use English" is always written in English, whatever the app
 * language, so someone who cannot read the detected language can still
 * find it. Either choice marks the dismissal flag so the banner never
 * re-appears, and persists the pick via ``onPersistLocale`` (the
 * detected locale, or ``"en"``) so the detection does not fire again.
 */

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  changeLocale,
  detectLocale,
  SUPPORTED_LOCALES,
  type SupportedLocale,
} from "../i18n";
import enMeta from "../locales/en/meta.json";

const DISMISSED_KEY = "chaosengine.localeBannerDismissed";

interface LocaleEntry {
  code: string;
  endonym: string;
  rtl: boolean;
}

const SUPPORTED_META: LocaleEntry[] = (enMeta as { supportedLocales: LocaleEntry[] }).supportedLocales;

export interface FirstLaunchLocaleBannerProps {
  /** Current persisted locale from ``settings.locale`` (``"system"`` for fresh installs). */
  persistedLocale: string | null | undefined;
  /** Called with either the detected locale or ``"en"`` when the user makes a choice. */
  onPersistLocale: (locale: string) => void;
}

export function FirstLaunchLocaleBanner({
  persistedLocale,
  onPersistLocale,
}: FirstLaunchLocaleBannerProps) {
  const { t } = useTranslation("common");
  const [dismissed, setDismissed] = useState(false);
  const [detected, setDetected] = useState<SupportedLocale | null>(null);

  useEffect(() => {
    // ``persistedLocale`` must be the sentinel — any explicit choice
    // means the user already opted in (or out) on a previous launch.
    if (persistedLocale && persistedLocale !== "system") return;
    if (typeof window === "undefined") return;
    if (window.localStorage.getItem(DISMISSED_KEY) === "1") {
      setDismissed(true);
      return;
    }
    const target = detectLocale();
    if (target === "en") return;
    if (!SUPPORTED_LOCALES.includes(target)) return;
    setDetected(target);
  }, [persistedLocale]);

  if (dismissed || detected === null) return null;

  const targetMeta = SUPPORTED_META.find((entry) => entry.code === detected);
  const endonym = targetMeta?.endonym ?? detected;

  const finalise = (next: string) => {
    if (typeof window !== "undefined") {
      try {
        window.localStorage.setItem(DISMISSED_KEY, "1");
      } catch {
        // Ignore quota errors — the banner will simply re-appear on
        // the next cold start, which is a small annoyance and not a
        // correctness break.
      }
    }
    setDismissed(true);
    onPersistLocale(next);
  };

  const handleKeep = () => {
    void changeLocale(detected);
    finalise(detected);
  };

  const handleUseEnglish = () => {
    // The app already started in the detected language, so this has to switch now;
    // it used to stay in that language until the next settings poll came back (up to 5 s).
    void changeLocale("en");
    finalise("en");
  };

  return (
    <div
      className="locale-banner"
      role="dialog"
      aria-live="polite"
      aria-label={t("localeBanner.ariaLabel", { defaultValue: "Language preference" })}
    >
      <div className="locale-banner__copy">
        <strong>
          {t("localeBanner.inUse", { endonym, defaultValue: "The app is in {endonym}, your system language." })}
        </strong>
        <span>
          {t("localeBanner.changeHint", {
            defaultValue: "You can change this any time in Settings → Language.",
          })}
        </span>
      </div>
      <div className="locale-banner__actions">
        <button type="button" className="primary-button" onClick={handleKeep}>
          {t("localeBanner.keepCurrent", { endonym, defaultValue: "Keep {endonym}" })}
        </button>
        <button type="button" className="secondary-button" lang="en" onClick={handleUseEnglish}>
          {t("localeBanner.useEnglish", { lng: "en", defaultValue: "Use English" })}
        </button>
      </div>
    </div>
  );
}
