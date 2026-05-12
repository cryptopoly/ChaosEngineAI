/**
 * Global vitest setup — initialises i18next synchronously with the
 * English bundles so components that call `useTranslation(...)` resolve
 * their `t()` calls to substituted English strings during tests.
 */
import { initI18n } from "./i18n";

await initI18n({ forceLocale: "en" });
