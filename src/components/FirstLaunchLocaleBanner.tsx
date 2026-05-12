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
 * Either choice (Switch / Keep English) marks the dismissal flag so
 * the banner never re-appears.  Picking "Switch" also calls
 * ``changeLocale`` (live) and persists via ``onPersistLocale`` so the
 * choice survives a restart.  Picking "Keep English" persists the
 * sentinel ``"en"`` so the user's explicit choice is recorded and the
 * detection logic doesn't fire again.
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

  const handleSwitch = () => {
    void changeLocale(detected);
    finalise(detected);
  };

  const handleKeepEnglish = () => {
    finalise("en");
  };

  return (
    <div
      className="locale-banner"
      role="dialog"
      aria-live="polite"
      aria-label="Language preference"
    >
      <div className="locale-banner__copy">
        <strong>{`Detected ${endonym}.`}</strong>
        <span>
          Switch the app to {endonym}? You can change this any time in Settings → Language.
        </span>
      </div>
      <div className="locale-banner__actions">
        <button type="button" className="primary-button" onClick={handleSwitch}>
          {`Switch to ${endonym}`}
        </button>
        <button type="button" className="ghost-button" onClick={handleKeepEnglish}>
          {t("status.success") /* placeholder to ensure ns loaded; replaced inline below */}
          {/* eslint-disable-next-line react/jsx-no-literals */}
          Keep English
        </button>
      </div>
    </div>
  );
}
