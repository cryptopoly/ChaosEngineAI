import { chromium, type FullConfig } from "@playwright/test";

/**
 * Load the app once before the tests. Vite transforms every module on the first request; on Windows that
 * first page load took over 30 s (the per-test timeout), so whichever test ran first failed. Playwright
 * starts the web servers before this runs.
 */
export default async function warmup(config: FullConfig): Promise<void> {
  const baseURL = config.projects[0]?.use?.baseURL;
  if (!baseURL) return;
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.goto(baseURL, { timeout: 180_000 });
    await page.locator("aside.sidebar").waitFor({ state: "visible", timeout: 180_000 });
  } finally {
    await browser.close();
  }
}
