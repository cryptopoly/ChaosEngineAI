import type { Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { API_BASE, TOKEN, openApp, openMenuIfCompact, openTab } from "./helpers";

// Review stream D: the backend stops while the app is in use. Before this the only sign was the small badge at
// the foot of the sidebar, which sits inside the closed drawer below 1180 px, and most pages kept showing their
// last data. The workspace now shows a notice until the backend answers again. The second half is the token
// case (FU-195): the backend comes back with another API token, the page reads it from /api/auth/session once
// and carries on. The stop is simulated by refusing every request to the backend, since the specs share one.

test.setTimeout(120_000);

const notice = (page: Page) => page.locator(".backend-offline-banner");
const badge = (page: Page) => page.locator(".sidebar-footer .badge");

async function stopBackend(page: Page): Promise<void> {
  await page.route(`${API_BASE}/**`, (route) => route.abort("connectionrefused"));
}

async function startBackend(page: Page): Promise<void> {
  await page.unroute(`${API_BASE}/**`);
}

test("a stopped backend is named on the page, every tab still opens, and the notice clears when it is back", async ({ page }) => {
  await openApp(page);
  await openTab(page, "Chat", "Chat");
  await expect(notice(page)).toHaveCount(0);
  await expect(badge(page)).toHaveText("Backend online");

  await stopBackend(page);
  await expect(notice(page)).toBeVisible({ timeout: 30_000 });
  await expect(notice(page)).toContainText("The backend is not answering.");
  await expect(notice(page)).toContainText("The app reconnects by itself.");
  await expect(notice(page)).toHaveAttribute("role", "status");
  await expect(badge(page)).toHaveText("Offline");

  // Moving around while it is down: no tab crashes, the notice stays.
  for (const [group, child] of [["Chat", "Models"], ["Images", "Studio"], ["Video", "Studio"], ["Voice", "Studio"], ["Music and Audio", "Studio"]] as const) {
    await openTab(page, group, child);
    await expect(page.locator(".error-boundary"), `${group} > ${child} crashed while offline`).toHaveCount(0);
    await expect(notice(page)).toBeVisible();
  }
  const nav = page.getByRole("navigation", { name: "Primary" });
  for (const single of ["Projects", "Settings", "Dashboard"]) {
    await nav.locator(".nav-button", { hasText: new RegExp(`^${single}`) }).click();
    await expect(page.locator(".error-boundary"), `${single} crashed while offline`).toHaveCount(0);
    await expect(notice(page)).toBeVisible();
  }

  await startBackend(page);
  await expect(notice(page)).toHaveCount(0, { timeout: 30_000 });
  await expect(badge(page)).toHaveText("Backend online");
  // And the app works again: Settings shows what the backend holds.
  await nav.locator(".nav-button", { hasText: /^Settings/ }).click();
  await expect(page.getByText(/^Accepted on .+\.$/)).toBeVisible();
});

test.describe("1100 x 760, the smallest window", () => {
  test.use({ viewport: { width: 1100, height: 760 } });

  test("the notice shows where the sidebar badge is out of sight", async ({ page }) => {
    await openApp(page);
    await openTab(page, "Chat", "Chat");
    // The drawer is closed, so the badge that used to be the only sign cannot be seen.
    await expect(badge(page)).toBeHidden();
    await stopBackend(page);
    await expect(notice(page)).toBeVisible({ timeout: 30_000 });
    await expect(notice(page)).toBeInViewport();
    const box = (await notice(page).boundingBox())!;
    expect(box.x + box.width).toBeLessThanOrEqual(1100);
    await startBackend(page);
    await expect(notice(page)).toHaveCount(0, { timeout: 30_000 });
  });
});

test("a backend that comes back with another token: the page reads the new one and carries on", async ({ page }) => {
  await openApp(page);
  const fresh = "token-after-restart";
  const cors = { "access-control-allow-origin": new URL(page.url()).origin, "access-control-allow-headers": "*" };
  let refused = 0;
  let accepted = 0;
  // From here the backend "has restarted": the token the page was built with is refused (401), the session route
  // hands out the new one, and a request that carries the new one is passed on with the token the real backend knows.
  await page.route(`${API_BASE}/**`, async (route) => {
    const request = route.request();
    const { pathname } = new URL(request.url());
    if (request.method() === "OPTIONS" || pathname === "/api/health") return route.fallback();
    if (pathname === "/api/auth/session") {
      return route.fulfill({ status: 200, headers: cors, json: { apiToken: fresh, apiBase: API_BASE, tokenType: "Bearer" } });
    }
    const sent = request.headers()["authorization"] ?? "";
    if (sent === `Bearer ${fresh}`) {
      accepted += 1;
      return route.fallback({ headers: { ...request.headers(), authorization: `Bearer ${TOKEN}` } });
    }
    refused += 1;
    return route.fulfill({ status: 401, headers: cors, json: { detail: "Unauthorized. Supply the ChaosEngineAI API token as a Bearer token." } });
  });

  // The next polls are refused once, then carry the new token.
  await expect.poll(() => accepted, { timeout: 30_000 }).toBeGreaterThan(0);
  expect(refused).toBeGreaterThan(0);
  await expect(notice(page)).toHaveCount(0);
  await openMenuIfCompact(page);
  await expect(badge(page)).toHaveText("Backend online");

  // A page that loads its own data works with the new token, with no further refusals.
  const before = refused;
  await page.getByRole("navigation", { name: "Primary" }).locator(".nav-button", { hasText: /^Settings/ }).click();
  await page.locator(".settings-subtab-bar [role=tab]", { hasText: "Diagnostics" }).click();
  await expect(page.getByRole("button", { name: "Copy diagnostics" })).toBeVisible();
  await expect(page.getByText("CPU logical cores")).toBeVisible({ timeout: 20_000 });
  await openTab(page, "Images", "Gallery");
  await expect(page.locator(".error-boundary")).toHaveCount(0);
  await page.waitForTimeout(6000);
  expect(refused, "requests were still sent with the old token").toBe(before);
});
