// Synchronous i18n — translations loaded eagerly at module import time.
// Usage: const { t } = useI18n(); inside a React component.

import zhCN from "./zh-CN";

// Translation map is statically imported — no async delay
const translations: Record<string, string> = zhCN;

/**
 * Synchronous translate function.
 * Replaces the async t() with a simple map lookup.
 */
export function t(key: string, fallback?: string): string {
  return translations[key] ?? fallback ?? key;
}

/**
 * Interpolate variables into a translation string.
 * e.g. t("models.installedCount", "{{count}} 个", { count: 5 })
 */
export function ti(key: string, fallback: string, vars: Record<string, string | number>): string {
  let result = t(key, fallback);
  Object.keys(vars).forEach(function (k) {
    result = result.replace(new RegExp("\\{\\{" + k + "\\}\\}", "g"), String(vars[k]));
  });
  return result;
}
