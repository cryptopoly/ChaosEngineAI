import { test, expect } from "./fixtures";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { openApp, SHOTS_DIR } from "./helpers";

// FU-241 review of the Boost Pack on Windows (2026-11-19): Nunchaku and SageAttention are not pip packages, so
// their Install answers 400 with the manual steps; the rows showed only an "Install failed" badge and the steps
// were never visible. The row now shows the message (and the pip output when there is one) under the row.

test("Boost Pack: a failed install shows why, under its row", async ({ page }) => {
  await page.route("**/api/setup/install-package", async (route) => {
    const body = route.request().postDataJSON() as { package: string };
    if (body.package === "nunchaku") {
      await route.fulfill({
        status: 400,
        json: { detail: "Nunchaku (SVDQuant) is not installable from PyPI. Download the wheel that matches your torch, CUDA and Python versions from the Nunchaku GitHub releases page." },
      });
    } else {
      await route.fulfill({
        json: { ok: false, output: "ERROR: No matching distribution found for triton", capabilities: {} },
      });
    }
  });
  await openApp(page);
  await page.getByRole("navigation", { name: "Primary" }).getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("tab", { name: "Accelerators" }).click();
  const row = page.locator('tr[data-accelerator-id="nunchaku"]');
  await expect(row).toBeVisible();
  await row.getByRole("button", { name: /Install/ }).click();
  const why = page.locator('tr[data-accelerator-error="nunchaku"]');
  await expect(why.getByRole("alert")).toContainText("not installable from PyPI");
  await expect(why).toContainText("GitHub releases page");
  mkdirSync(SHOTS_DIR, { recursive: true });
  await page.locator(".accelerator-boost-pack, table").first().screenshot({ path: path.join(SHOTS_DIR, "accelerators-install-error.png") });
});
