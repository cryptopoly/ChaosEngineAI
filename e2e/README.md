# Browser E2E (Playwright)

Dev-only. Not part of `npm test`, `npm install` or the release build; the root `package.json` has no Playwright dependency. Conditions and risks: [docs/research/playwright-security.md](../docs/research/playwright-security.md).

## One-time setup

```
cd e2e
npm ci --ignore-scripts
npx playwright install chromium --only-shell   # explicit browser download, headless shell only
```

Never use `--with-deps` outside CI, the Chrome/Edge channel reinstall commands, or `@playwright/mcp`.

## Run

From the repo root: `npm run e2e` (the root needs its own `npm ci` so Vite is available). Or `cd e2e && npm test`.

Playwright starts both servers itself:

| Server | Port | Notes |
|---|---|---|
| Vite dev server | 8909 | `VITE_CHAOSENGINE_API_BASE` points it at the backend |
| Backend (repo `.venv`, or `E2E_PYTHON`) | 8910 | fresh `HOME` at `e2e/.run/home` on every run, dummy token, empty HF tokens |

Two runs at once (another worktree): `E2E_WEB_PORT=8911 E2E_API_PORT=8912 npm run e2e`.
The Vite server for the run sets `VITE_CHAOSENGINE_HOME_STUDIO=1` so the Music and Storyboard tabs exist.

Every run starts as a first launch, so the terms screen appears. Specs run in file order with one worker and share
one backend, so `01-smoke` (which accepts the terms last) must stay before specs that assume they are accepted.
Anything that is not loopback is aborted in `tests/fixtures.ts`; import `test` from there, not from `@playwright/test`.

Output (git-ignored): `e2e/test-results/` (screenshots, traces on failure only), `e2e/playwright-report/`, `e2e/.run/`.
Traces contain request bodies and headers; do not share them from a run that used real tokens (there are none by default).

## Specs

- `01-smoke`: shell renders, sidebar order, terms screen on a clean first run, accepting it sticks.
- `02-gallery-project-filter`: the Project dropdown in the Image gallery (open, search, keyboard, Escape returns focus). Gallery outputs are mocked at `/api/images/outputs`.
- `03-libraries-projects`: Video, Music and Voice libraries with a Storyboard project: dropdown, filtering, the "From project X" badge, header counts, Open in Storyboard (library data mocked, the project seeded through the backend).
- `04-music-sounds-licence`: Music | Sounds switch, the Stable Audio Open licence screen and its "Powered by Stability AI" line (the backend's own catalog).
- `05-storyboard`: create a project (Aspect ratio, Resolution, Frame rate, and the dialog at 375 px), Project defaults format, add shots, takes and lock, captions panel, export panel (forced AI tag, Upscale, run estimate, Film list). Projects are seeded with a tiny still and a tone; nothing is rendered.
- `06-busy-advice`: `/api/busy-advice` mocked for ok, slower and risky: the Chat load dialog ("Load anyway" gating) and the Music Studio estimate strip.
- `07-voice-clone`: (Voice Studio > Voices, and Speak for the cloned voice) the clone statement, the new-voice form (whose voice, speaker's name, permission tick, a sample from an uploaded file or the microphone through Chromium's fake device, wrong file type and a too-short recording refused, the fields sent), AI label on cloned speech, the form at 375 px.
- `08-settings-terms`: Settings terms panel (read only), and an older accepted version bringing the terms back.
- `09-mobile`: 375 px wide, no sideways scroll, a full-page screenshot per screen in `e2e/screenshots/` (git-ignored). One known failure is marked `test.fail` (F-1 in [docs/research/e2e-findings.md](../docs/research/e2e-findings.md)).
- `14-voice-speak`: Voice Studio Speak (style box and chips greyed out for voices without style, model line and Download, the `/api/voice/speak` body), Voices (kind filters, Use in Speak, Use in Chat and the chat voice badge, samples for cloned voices not kept, Chat and calls, delete with confirm), the ffmpeg note before Generate, autosave to the Library and the Voice Models speech-model rows, plus 375 px. Shared mocks in `tests/voiceMocks.ts`.
- `27-benchmarks-conversion`: Benchmarks Run and History (empty state, a speed run compared only with speed runs, an accuracy run with its score and no tok/s line, the mode sent with a run and the result dialog in that mode, the MLX-only note for perplexity and accuracy with a GGUF model, a backend error in the dialog, Cancel benchmark (the run answers cancelled, nothing saved, the dialog says whether the model was unloaded), the progress dialog driven by the run's own progress (a percentage only for counted questions, none for a load or a generation), no Quality figure anywhere, accuracy runs scored by different methods shown without a delta, History list, model filter, baseline by right-click and by double click, a baseline of another kind, views, Copy CSV and Delete run with its confirmation, 375 px) and Conversion to MLX (nothing to convert until a source is picked, only non-MLX models offered, a Transformers source at 8-bit and the request sent, a GGUF picked as one weights file still shown as the source, Cancel conversion and the cancelled dialog, the progress dialog driven by the converter's own progress (files downloaded, bytes written, the size estimate called an estimate), the measured size ratio, bits per weight and perplexity comparison in the completion dialog, a comparison that could not be measured with its reason, nothing projected with no source picked, a backend error in the dialog, 375 px). The library, the saved runs, both long requests and the two progress endpoints are mocked.
- `28-finetuning`: Tools > Fine-Tuning. Empty state (no adapters, Start off, the "not built" sentence, nothing sent to `/api/finetuning/start`), an adapter folder under the run's `~/Models` shown as a card (this state used to crash the tab), a row with fields missing, a failing adapter list, and 375 px. In a worktree without its own `.venv`, set `E2E_PYTHON`.
- `29-prompts-in-chat`: (D-87) the Prompts panel beside the Chat composer: the five built-ins, search focused on open, Up and Down between the search box and the rows, Escape steps back then closes with focus on the button, a save with no name or bad variables refused with a sentence, a template with a variable filled in and previewed, "Use as system prompt" sent as `systemPrompt` (chat faked as in `16-skills`) and "Stop using it", "Insert into message", "Save message as prompt", edit, delete with a confirm, a failed load, 1280 and 375 px (screenshots `29-prompts-chat-*.png`). The sidebar has no Prompts or Plugins entry and `/api/plugins` is 404; a remembered Prompts id opens Chat and a request for Plugins opens the Dashboard.
- `31-server-activity`: Server page (D-88): no Logs sidebar entry, the connect block is one closed line that remembers being opened, Overview's recent activity, Activity's timeline, filters, detail row (command line, engine output, environment), latest failure, select, copy and export, pause, remembered view, 375 px. The event stream is mocked; nothing is loaded.
- `32-backend-offline`: the backend stops answering mid-use (requests refused): the workspace notice, every tab still opens, the notice clears when it is back; the same at 1100 x 760 where the sidebar badge is out of sight; and the backend coming back with another API token (401 once, then the page reads `/api/auth/session`).
- `33-window-sizes`: every tab outside Benchmarks, Tools, Server and Logs at 1100 x 760 (the smallest window), 1400 x 920 and 375 px: no panel cuts its content, no two panels overlap, no crash. Plus Settings in the drawer with the Chat or Benchmarks group open, the Settings > General controls, and the Storage folder rows.
- `34-locales`: the first-launch language banner on a German system (it states the language in use and offers Keep and Use English; both buttons), every shipped locale (no raw keys, no leftover placeholders, no sideways scroll), and a pseudo-locale (accented, 40 % longer, served in place of the German catalogue) on Settings General, Storage, Diagnostics and Accelerators.
- `35-keyboard-dialogs`: first-run terms by keyboard, Select Model in Chat and Compare as real dialogs (Tab stays inside, Escape closes, focus returns), the read-only terms, and the Chat composer with no model loaded.
- `36-video-gallery-lazy`: forty mocked clips; only the cards in view ask for their file, the last one only after scrolling to it.
- `37-chat-local-model`: a local model that is not a catalogue row (added to the workspace answer): Chat in a new thread sends the model's own repo, not a look-alike catalogue row's, and opens a new thread.
- `38-diagnostics-i18n`: Settings > Diagnostics in German and Japanese, every label checked against that language's catalogue (no English left); the Language panel's title and labelled rows; Compare and HTML Challenge slot names and the picker's buttons in German.
- `39-key-masking`: API keys on screen are masked (dots and the last 4 characters) until Reveal: the Server key field, the Connect samples, the Test commands, Settings > Integrations and Agent access; each Copy puts the real key on a stubbed clipboard.
- `40-experimental-tabs`: Settings > Appearance > Show experimental tabs (off by default): the tabs flagged `experimental` in `src/constants/tabs.ts` are not in the Tools group until it is on, the choice survives a reload, and a tabs-style sidebar that remembered a hidden tab does not reopen it. Spec 28 turns the switch on before it loads the page (`showExperimentalTabs` in `tests/helpers.ts`).

A sibling worktree without its own `.venv` needs `E2E_PYTHON=/path/to/.venv/bin/python`. Stop a run with `kill -9` on the `playwright test` process: after a plain `kill` the runner outlives its two servers and runs the remaining tests against whatever listens on the same ports.

Spec order matters: `01-smoke` accepts the terms; `08` accepts them again against the same backend and leaves them accepted. `tests/helpers.ts` has the shared helpers (open a tab, seed a project, fixture media under the run's throwaway HOME).
