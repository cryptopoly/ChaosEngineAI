import { useCallback } from "react";
import { t, ti } from "../i18n/index";

/**
 * useI18n — React hook for accessing synchronous translations.
 *
 * Usage inside a component:
 *   const { t } = useI18n();
 *   return <button>{t("sidebar.brand.tagline")}</button>
 *
 * For interpolated strings:
 *   const { t } = useI18n();
 *   return <span>{t("models.installedCount", "{{count}} 个", { count: 5 })}</span>
 */
export function useI18n() {
  const translate = useCallback((key: string, fallback?: string) => t(key, fallback), []);
  const translateInterpolated = useCallback(
    (key: string, fallback: string, vars: Record<string, string | number>) =>
      ti(key, fallback, vars),
    []
  );
  return { t: translate, ti: translateInterpolated };
}
