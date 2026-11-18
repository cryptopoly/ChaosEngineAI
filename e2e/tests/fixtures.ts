import { test as base, expect } from "@playwright/test";

const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

/** Every test: abort any request whose host is not loopback, so a test can never reach the web. */
export const test = base.extend({
  context: async ({ context }, use) => {
    await context.route("**/*", (route) => {
      const url = new URL(route.request().url());
      if (url.protocol === "data:" || url.protocol === "blob:" || url.protocol === "about:" || LOCAL_HOSTS.has(url.hostname)) {
        return route.fallback();
      }
      return route.abort("blockedbyclient");
    });
    await use(context);
  },
  page: async ({ page }, use) => {
    await use(page);
    // A poll answered through a spec's own route (`route.fetch()` on /api/workspace, say) can still be in
    // flight when the test body ends, and Playwright then fails the finished test with "route.fetch: Test
    // ended". It showed once the shared openApp stopped waiting 5 s; dropping the routes here ignores it.
    await page.unrouteAll({ behavior: "ignoreErrors" });
  },
});

export { expect };
