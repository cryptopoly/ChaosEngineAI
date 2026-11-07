# Playwright security review (test-only dev dependency for browser E2E)

Date: 2026-11-07. Research only, nothing installed. Scope: `@playwright/test` driving the Vite dev server or built frontend against a localhost FastAPI backend, on macOS now and Windows later.

## Verdict

**Approve with conditions.** The "significant security issues" claim is overstated. There is one real, fixed, High-rated advisory in the npm package (macOS browser-reinstall scripts, fixed Oct 2025). There is a separate High advisory in a different package, `@playwright/mcp`, which we would not use. The remaining risk is the normal risk of any automated browser plus a ~2 MB-of-code, 2-package npm tree with no install scripts. Conditions are in section 6.

## 1. Known CVEs and advisories

| Id | Package | Affected | Fixed | Severity | What |
|---|---|---|---|---|---|
| CVE-2025-59288 / GHSA-7mvr-c777-76hp | `playwright` (npm) | < 1.55.1 | 1.55.1 | High, CVSS v4 8.7 (GitLab scores it 5.3 Medium) | Five macOS scripts in `playwright-core/bin/` (`reinstall_chrome_{beta,stable}_mac.sh`, `reinstall_msedge_{dev,beta,stable}_mac.sh`) used `curl -k`, so a network attacker could swap the installer, which then ran with `sudo installer`. Only hit if you ran the "reinstall Chrome/Edge channel" command. Fix removes `-k`. Published 2025-10-14. |
| GHSA-qxm8-4v54-964r | same issue, Socket's report | lists 1.55.1 affected, 1.56.0+ patched | 1.56.0 | High | Same bug. **Version conflict**: GitHub's advisory says fixed in 1.55.1, Socket's says fixed in 1.56.0. Any version >= 1.56.0 is safe under both readings. |
| CVE-2025-9611 / GHSA-6fg3-hvw7-2fwq | `@playwright/mcp` | < 0.0.40 | 0.0.40 | High, CVSS v4 7.2 | MCP server did not validate the Origin header, so DNS rebinding from a web page could call its tools on a local server. Published 2026-01-07. Not relevant unless we add the MCP server. |
| microsoft/playwright issue #43899 | bundled in `playwright-core/lib/utilsBundle.js` | 1.65.x and earlier (reported against `1.66.0-next`) | Fixed on main, closed 2026-11-04, shipping in the 1.66 line | npm audit shows 2 high, 2 moderate, all DoS via crafted input (zod 4.4.3, brace-expansion 5.0.7, minimatch 10.2.5, ip-address 10.7.2) | Maintainer calls exposure low. Expect `npm audit` noise on 1.65.0 until 1.66 ships. Not exploitable from our usage (no untrusted input reaches these parsers). |

Not Playwright: GHSA-687h-xw6f-q2qw / CVE-2026-44439 is Lookyloo's "PlaywrightCapture", a different project with a similar name.

No other GHSA for `playwright`, `playwright-core` or `@playwright/test` turned up. I did not query osv.dev directly (its API rejected the fetch tool with HTTP 405), so the "no other advisory" claim rests on the GitHub advisory pages, npmscan summaries and search results, not an osv.dev query. Run `npm audit` and `scripts/audit-dependencies.py` (osv.dev based) after install to confirm.

Security policy: reports go to MSRC, coordinated disclosure (SECURITY.md). Microsoft-maintained, Apache-2.0.

## 2. Supply chain and browser download

- **No install-time script.** npm registry metadata for `playwright`, `playwright-core` and `@playwright/test` 1.65.0 shows no `scripts` (no postinstall). `npm install` does not download browsers. Browsers come only from an explicit `npx playwright install chromium`. So `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD` is mostly moot on current versions; I could only confirm it is honoured in `installBrowsersForNpmInstall` in the registry source, and the docs page I fetched does not mention it. Keeping it set in CI is harmless.
- **Download hosts** (source, `registry/index.ts`): `cdn.playwright.dev` and `playwright.download.prss.microsoft.com` (Microsoft ESRP CDN). `playwright.azureedge.net` no longer appears in the code (the old Azure CDN host). Override with `PLAYWRIGHT_DOWNLOAD_HOST`.
- **Integrity.** I could not confirm any checksum or signature verification of the downloaded browser archive: the docs say nothing, and the fetcher file that would do it (`browserFetcher`) was not readable in this pass. Trust is HTTPS plus Microsoft's CDN. That is the real residual download risk, and it is the same trust level as any browser download. Mitigate by pinning the Playwright version (it pins exact browser revisions) and caching the browser directory in CI.
- **Cache location**: macOS `~/Library/Caches/ms-playwright`, Windows `%LOCALAPPDATA%\ms-playwright`, override with `PLAYWRIGHT_BROWSERS_PATH`.
- **Dependency footprint (latest 1.65.0, published 2026-11-05):** `@playwright/test` -> `playwright` -> `playwright-core`. Registry metadata shows no other runtime dependencies and no optionalDependencies (older releases had optional `fsevents`; not shown now). `playwright-core` unpacked is about 14 MB because it bundles its own deps into `utilsBundle.js`, which is why bundled-dependency advisories show up (see #43899) and why `npm audit` cannot fix them by overrides. Licence Apache-2.0 on all three, which meets the project licence rule. Engines: Node >= 20. Needs a `THIRD_PARTY_NOTICES.md` entry for dev tooling per project rules.
- **MCP variant** (`@playwright/mcp`): lets an AI agent drive a browser. Risks: CVE-2025-9611 above, prompt injection from page content steering the agent, and the agent having a logged-in browser. Not needed for scripted E2E. **Recommend not adding it** to this project's dependencies. A third-party blog claims the current npm version is 0.0.82; I did not verify that.
- **Registry hygiene**: general npm worm/typosquat risk applies to any package. Playwright has a first-party scope (`@playwright/test`) and an unambiguous maintainer (Microsoft). I did not audit the maintainer list on npm.

## 3. Risks of running an automated browser

- **Sandbox is off by default.** The docs for `browserType.launch` list `chromiumSandbox` with default `false` ("Enable Chromium sandboxing"). That means a renderer exploit has a weaker barrier. The docs do not mention `--no-sandbox` itself; the usual mapping (false -> `--no-sandbox` flag) is widely known but I did not confirm it in source this pass. Mitigation: set `chromiumSandbox: true` and only load localhost. A social-media post (2026-11-02) says Playwright 1.64.1 bundles a Chromium with several unfixed CVEs; **unverified**, no primary source. What is verifiable: Playwright ships a pinned Chromium (main branch pins 158.0.8127.18 for revision 1254) that lags Chrome stable by some days to weeks, so pointing it at untrusted web content is the risk, not our localhost use.
- **Remote debugging.** Playwright talks to Chromium over a pipe by default (not a TCP port). A TCP debugging port exists only if you pass `--remote-debugging-port` or use `connectOverCDP`/`launchServer`. Do not do either. (The pipe default is from my background knowledge; I did not re-verify it in this pass.)
- **Traces, videos, screenshots, HAR.** The trace viewer docs say traces hold full network requests with request/response headers and bodies, complete DOM snapshots, console logs and screencast frames. The page has no warning about secrets, but anything in an `Authorization` header, a token in a request body or an API key shown on screen ends up in the trace zip. For us the likely secrets are the HF token in Settings and any backend bearer token. Keep `trace: 'off'` or `'retain-on-failure'`, store output under a git-ignored dir, never upload artifacts from runs that used real tokens, and run against a throwaway data dir.
- **Untrusted pages.** Playwright is fine for our own app. Risk appears only if a test navigates to the web (the app has model-hub and docs links). Block it: route-abort every non-localhost request.
- **Test-created state.** The E2E exercises a backend that can download multi-GB models, run inference and write to `~/.chaosengine`. Use a temp `CHAOSENGINE` data dir and mocked or tiny models so a test cannot trigger huge downloads or touch real chats.
- **Backend exposure.** The FastAPI backend should stay bound to 127.0.0.1. A DNS rebinding style issue like CVE-2025-9611 is a class that applies to our backend too (Origin/Host validation); worth a separate check, not a Playwright issue.

## 4. Alternatives

| Option | Fit for this app | Risk comparison |
|---|---|---|
| **Vitest browser mode** (`@vitest/browser-playwright` provider) | Component-level tests in a real browser, already on vitest 4.1. | Still uses Playwright underneath, so same browser download and sandbox concerns. Not a way to avoid Playwright. The `preview` provider avoids Playwright but Vitest says it is not recommended for CI (it simulates events). The webdriverio provider is the other option. |
| **Puppeteer** | Chromium only, comparable features. | Same class of risk (downloads Chromium at install by default via postinstall, which is worse for supply chain than Playwright's explicit install). Single-vendor (Google). No clear security advantage; I did not review its advisory history. |
| **WebdriverIO** | Could drive Chrome/Edge and, via its Tauri service, the real Tauri window on Windows/Linux. | Larger dependency tree (many npm packages), which means more supply chain surface. I did not measure the tree. |
| **tauri-driver** | Tests the real desktop shell. | Driven directly it supports only Windows and Linux; macOS has no WKWebView driver (Tauri docs). CrabNebula's fork supports macOS but needs a paid API key. Unusable on the Mac dev box without a paid service, so it cannot be the main E2E now. |
| **Existing vitest + jsdom/testing-library** | No real browser. | Lowest risk, but does not test real rendering, CSS or backend integration. Already in use; keep it. |

Net: Playwright is a reasonable choice. No alternative is clearly safer for real-browser E2E, and the safer ones do less.

## 5. Claims I could not verify

- Any osv.dev query result (API refused the fetch tool).
- Browser archive checksum verification in the Playwright downloader.
- That `chromiumSandbox: false` maps to `--no-sandbox` in source, and that the default transport is a pipe.
- The X/Twitter claim about Chromium CVEs in 1.64.1.
- The `@playwright/mcp` 0.0.82 current version.
- Puppeteer and WebdriverIO advisory history and dependency counts.
- Whether 1.55.1 or 1.56.0 is the exact fix (sources disagree; >= 1.56.0 satisfies both).
- Version note: npm `latest` was 1.65.0 on 2026-11-07 (published 2026-11-05); a third-party page on 2026-11-02 listed 1.64.1, which is consistent.

## 6. Mitigations (conditions for approval)

1. Add `@playwright/test` to `devDependencies` with an **exact** version (no caret), >= 1.56.0. Today that means `1.65.0`, or `1.66.x` once released, since it carries the bundled-dependency fixes. Commit `package-lock.json`. Install with `npm ci`.
2. Keep it out of the default dev path: a separate script (`npm run e2e:browser`), not part of `npm test`, not run by `pre-build-check.sh` unless asked. Consider an `e2e/` sub-package with its own `package.json` so the app's lockfile and release build never see Playwright.
3. Browser install is explicit and pinned: `npx playwright install chromium` (add `--only-shell` for headless). Set `PLAYWRIGHT_BROWSERS_PATH` to a project-local or CI-cached dir. Never run `npx playwright install` with `--with-deps` as root outside CI, and never use the Chrome/Edge channel reinstall commands (the old vulnerable path).
4. Launch options: `headless: true`, `chromiumSandbox: true`, no custom `args`, no `--remote-debugging-port`, no `connectOverCDP`, no `launchServer`.
5. Network lockdown: in a shared fixture, abort all requests whose host is not `127.0.0.1` / `localhost` (`context.route('**/*', ...)`), and set `baseURL` to localhost only.
6. Secrets: tests use a fresh temp data dir and a dummy token; no real HF or API tokens in the environment of the test run. `trace: 'retain-on-failure'` at most, output in a git-ignored folder (`test-results/`, `playwright-report/`), never uploaded from a run with real credentials.
7. Audit: run `npm audit --omit=dev` is not enough since this is a dev dependency, so run plain `npm audit` and `.venv/bin/python scripts/audit-dependencies.py` after adding. Accept the bundled-dependency DoS advisories in `scripts/dependency-audit-ignore.json` only with a reason (tests only, no untrusted input) and an expiry date, or upgrade to 1.66.
8. Do not add `@playwright/mcp` or any agent-driven browser server to the project dependencies. If wanted later, treat as a separate decision.
9. Bookkeeping: add Playwright to `THIRD_PARTY_NOTICES.md` (Apache-2.0, dev tooling) and to the upstream dependency table in `CLAUDE.md` with a check command (`npm view @playwright/test version`).
10. Windows later: run the same pinned Chromium; skip `tauri-driver` on macOS (unsupported); revisit it for Windows-only shell tests.

## Sources

- https://github.com/advisories/GHSA-7mvr-c777-76hp
- https://github.com/SocketDev/security-research/security/advisories/GHSA-qxm8-4v54-964r
- https://advisories.gitlab.com/pkg/npm/playwright/CVE-2025-59288/
- https://github.com/advisories/GHSA-6fg3-hvw7-2fwq
- https://github.com/microsoft/playwright/issues/43899
- https://github.com/microsoft/playwright/blob/main/SECURITY.md
- https://registry.npmjs.org/playwright, /playwright-core, /@playwright/test (queried 2026-11-07)
- https://playwright.dev/docs/browsers , /docs/api/class-browsertype , /docs/trace-viewer
- https://raw.githubusercontent.com/microsoft/playwright/main/packages/playwright-core/src/server/registry/index.ts
- https://raw.githubusercontent.com/microsoft/playwright/main/packages/playwright-core/browsers.json
- https://v2.tauri.app/develop/tests/webdriver/
- https://vitest.dev/guide/browser/
- https://npmscan.io/package/npm/playwright@1.3 , https://x.com/nisten/status/2117322879859568756 (unverified claim)
