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
});

export { expect };
