# MiniMax H3 licence check (2026-11-20)

**This is not legal advice.** It is a plain reading of the licence text by an engineer, written so a lawyer can confirm or correct it quickly. Decision row: D-99. Earlier notes: [minimax-h3-and-new-video-image-2026-11-10.md](minimax-h3-and-new-video-image-2026-11-10.md), FU-166, D-76.

## Headline

1. **The owner's premise is half wrong.** None of the three Hugging Face repos involved is gated today. `MiniMaxAI/MiniMax-H3`, `leejet/MiniMax-H3-GGUF` and `Comfy-Org/MiniMax-H3` all return `"gated": false` from the Hub API (checked 2026-11-20). A user does not click "accept" on Hugging Face before downloading. The app's own licence screen and tick are the only acceptance step anywhere in the chain. The app also does the download itself (on the user's click, through its own download worker); it never bundles the weights.
2. **The licence binds whoever uses the weights, and accepting is by use.** The preamble says running, using or displaying the model is acceptance. So each user who runs H3 is a Licensee, whether or not they clicked anything. That supports the owner's view: compliance is the user's job.
3. **The open question is whether ChaosEngineAI itself is caught by §V.2 and §V.5.** Both are written for a Licensee who offers "a product ... that permits the generation of Outputs". The app ships no weights and its maintainers never run H3, so the better reading is that the owner is not a Licensee at all. But §V.2 and §V.5 do not require that the product contains the weights, so a lawyer should confirm this.
4. **Verdict on the gate: about right, slightly loose in two places, not too strict.** Flip `H3_RELEASED` once a lawyer answers the questions below, after two small changes: the tick also accepts the licence and its Acceptable Use Policy, and the screen says that turning on remote access makes the user a hosted service under §V.5.

## What was checked

| Source | Revision / state on 2026-11-20 |
|---|---|
| Licence, `MiniMaxAI/MiniMax-H3/LICENSE` | `main` is byte-identical to the revision the app pins (`0919fb6`, 17,604 bytes). Licence date 2 August 2026; AUP last revised 2 August 2026. No change since the app was built. |
| Official model card `MiniMaxAI/MiniMax-H3` | sha `1cc7549`, last edit 2026-11-20 (README only). Links a licence Q&A and an application form for the excluded territories. |
| Licence Q&A `docs/QA-about-License.md` | Explains the exclusion as regulatory caution; says it means "not yet", and offers a separate licence on application. |
| `leejet/MiniMax-H3-GGUF` (the DiT and text encoder the app downloads) | sha `eee1dd2`, not gated, no licence metadata; README says the files follow the original licence. |
| `Comfy-Org/MiniMax-H3` (the two VAEs the app downloads) | sha `4b53a35`, not gated, `license: other` linking the MiniMax licence. |
| App code | `backend_service/licence_gate.py`, `routes/licence_gate.py`, `catalog/video_minimax_h3.py`, `helpers/images.py`, `helpers/video.py`, `routes/images.py`, `routes/video.py`, `state/downloads.py`, `mcp_bridge/tools.py`, `src/components/TerritoryGateNotice.tsx`, `src/components/LicenceScreen.tsx`, `THIRD_PARTY_NOTICES.md` (branch `test/rtx4090` at `a66a3616`). |

## What the licence says, point by point

Clause numbers are from the licence. Paraphrase unless quoted.

| Topic | What it says | What it means for the app |
|---|---|---|
| Territory | §I.3 and §I.5: the Applicable Territory is the world minus the EU, the UK, the Republic of Korea and the USA. §II grants rights "Solely within the Applicable Territory". §V.4 forbids use, reproduction, distribution or display of the Works or their Outputs outside it. AUP item 1 repeats it. | Most of the app's likely users are in an excluded territory. The tick is the only control, and the app cannot check location (it does not look it up, by design). |
| Who the licensee is | §I.9: any natural or legal person exercising rights or using the Works "for any purpose in any field of use". The preamble makes running or using the Works an acceptance. | The user who downloads and runs H3 is the Licensee. The owner becomes one only by using, reproducing, distributing, running or displaying the Works, which ChaosEngineAI does not do (no weights in the repo or the bundles; maintainers never ran it). |
| What counts as the Works | §I.6, §I.7, §I.11: the model, its weights and inference code as released by MiniMax, plus Model Derivatives (any modification or "work based on" H3, and distillations). Outputs are not Model Derivatives. | The leejet GGUF and Comfy-Org files are Model Derivatives, under the same licence. Whether stable-diffusion.cpp's independent H3 implementation (MIT code, no weights) is a "work based on MiniMax H3" is unclear: see lawyer question 2. The app's catalog row and command line are not Works on any reading. |
| Distribution | §III: a distributor must hand on a copy of the licence, mark modified files, and ship a NOTICE file; all only inside the territory. The last sentence of §III exempts someone who receives the Works inside "an integrated end-user product" from §III, but not from §V and the AUP. | The app distributes no Works today, so §III does not apply to it. If a release bundle ever ships the weights, §III applies and the territory rule makes global distribution impossible. Do not bundle. |
| Hosted Services | §I.8: services via API, web access "or any other electronic or remote means". §III.4 exempts hosted services from the NOTICE file only. §V.2 and §V.5 cover hosted services. | Off by default. If a user turns on **Allow remote connections** or the `/mcp` bridge and lets other people generate H3 through it, that user is running a Hosted Service and carries §V.2 and §V.5 personally. |
| Binding downstream users | §V.2: "you must bind each recipient or user to enforceable terms" at least as protective as §V and the AUP, and tell them those restrictions apply. | If a lawyer says the app is a product "incorporating" the Works, the app must bind and notify each user. The current tick only states territory; it does not accept the licence or the AUP. Cheap fix below. |
| Safeguards duty | §V.5: anyone who makes available to a Third Party a product that "permits the generation of Outputs using MiniMax H3" must build, test and review safeguards against AUP breaches, keep a way to report violations, and act on reports. | The hardest clause for an app. A desktop app cannot moderate what a user makes offline, and the maintainers never see Outputs. If it applies to ChaosEngineAI, the app needs at least a reporting route and documented safeguards, or H3 should stay hidden. This is lawyer question 1. |
| Acceptable Use | Exhibit A, 20 items, including: use outside the territory (1), breaking any law or IP right (2), harming minors (6), election or harmful disinformation (7), fake engagement (8), harassment (9), undisclosed machine-made content in public (12), impersonation without consent (13), high-risk automated decisions (14), military use (19). MiniMax may update it at any time. | The app already summarises 1, 12, 13 and 19 on the licence screen. That is enough as a summary if the screen links the full text and the tick accepts it. Because MiniMax can change the AUP, the screen should link the live licence page as well as the pinned one. |
| No training other models | §V.3: Works and Outputs may not be used to improve any other AI model. | Already on the screen. Relevant if the app ever feeds H3 videos into fine-tuning; it does not today. |
| Commercial use | §IV.1: written authorisation needed above US$20 million yearly revenue of the "commercial products and services". §IV.2: a commercial product or service that uses H3 must prominently show "MiniMax H3" in its interface. | Users: on the screen already. The app: the threshold is irrelevant at its size; if ChaosEngineAI is sold or monetised, §IV.2 may apply to it, and the row name "MiniMax H3" on the Models and Video pages already meets it. |
| Attribution | §III.3 *encourages* (does not require) "Powered by MiniMax H3", an AI-generation identifier in output files, and a public write-up. §VI.2 licenses the "MiniMax H3" mark only to comply with §III.3, and only inside the territory. | Optional. The app names the model as a plain description (nominative use), which §VI.2 also allows "as reasonably and customarily necessary to describe" the Works. The app writes a machine-generated tag for music and cloned voices but not for video; adding one to H3 videos would follow the encouragement and AUP item 12. |
| Indemnity | §VI.3: a Licensee must indemnify MiniMax against third-party claims arising from use or distribution of the Works "by you or by any Third Party". | Another reason the owner should stay a non-Licensee: never download, run, test or redistribute H3. |
| Law and courts | §IX: Hong Kong law, Hong Kong courts. | The lawyer needs Hong Kong contract-law input, or at least awareness of it. |

## What the gate does today

All of this is code on `test/rtx4090`; nothing has been released.

1. **Release switch.** `licence_gate.H3_RELEASED = False`. While false: the H3 row is dropped from every image and video catalog list (`filter_families`), `/api/licence-gate` returns no models, so the "Models with territory limits" notice does not render, confirm is refused, and every route that downloads, loads, previews or generates the row returns 403 ("not available in this version"). A test asserts the line still reads `False`.
2. **When released.** The Video > Models page shows one collapsed notice, "Models with territory limits". **Review** opens the licence screen: the excluded territories, the five summary lines, a line saying the app does not check where the user is, links to the pinned licence, and a tick on the sentence *"I confirm that I am not in the European Union, the United Kingdom, the Republic of Korea or the United States, and that I will not use this model or what it makes there."* Confirm stays disabled until ticked.
3. **Record.** Confirm stores key, time, app version, the exact sentence and the licence URL in `licence-gate.json` beside `settings.json`, and appends to `licence-gate-audit.jsonl`. If a later version changes the sentence, the old record no longer counts and the row hides until the user ticks again. **Withdraw** removes the record and hides the row.
4. **Who can confirm.** Only the app's own window. An agent token, the `/mcp` bridge or another machine is refused, and no agent tool can confirm. Agents can use the row only after the user has confirmed it.
5. **Enforcement points.** Catalog listings (images and video), `POST /api/images/download` and `/api/video/download` (by row and by repo, including companion repos), the shared download worker (`state/downloads.py`, by repo), preload and generate, and the MCP `generate_video` path.
6. **Not bundled.** No weights in the repo or the release bundles. Release CI does not build `sd`, so H3 also needs a locally built `sd`. `THIRD_PARTY_NOTICES.md` says the weights are user-downloaded and links the licence.

**What happens if the user has accepted a Hugging Face gate:** nothing different. The app never reads Hugging Face gate status for H3 (the row says `gated: false`, which is true of all three repos), and a Hugging Face token or prior download does not unhide the row or lift the 403. Only `H3_RELEASED = True` plus the in-app tick does. A user who downloads the files outside the app still sees no H3 row until they tick.

## Verdict (plain language, not legal advice)

- **The owner's position is sound for the user's own use.** A user who downloads and runs H3 is the Licensee and must follow the territory rule and the AUP; the app cannot and need not police a private offline machine.
- **The weak point is §V.2 and §V.5, not the territory.** They are written broadly enough ("product ... that permits the generation of Outputs") that MiniMax could argue an app with a one-click H3 download and generate button is such a product. If a lawyer agrees, the app needs binding terms, a reporting route and documented safeguards before release. If a lawyer disagrees (the owner never takes a licence, so the licence binds only users), the gate as built is enough with the two changes below.
- **Too strict?** No. Hiding until a tick is the least a responsible app can do with a licence that excludes most of its users. Do not loosen it to "visible but warned".
- **Too loose?** Slightly, in two places: the tick states territory only and does not accept the licence and AUP (a cheap fix for §V.2), and nothing warns that sharing generation over the network (remote access, `/mcp`) turns the user into a hosted service. A third, optional gap: H3 videos carry no machine-generated tag.
- **The owner must never download, run or test H3 from the UK.** That would make the owner a Licensee using the Works in an Excluded Territory. Real-model testing has to be done by someone outside the excluded territories, or not at all (the row already says "not tested by the maintainers").

## Questions for a lawyer

1. Does §V.5 (and §V.2) bind the author of a free desktop app that contains no MiniMax H3 weights but offers a one-click download from third-party Hugging Face repos and a generate button, when the author never uses the model? If yes, what is the minimum that counts as "reasonable and proportionate" safeguards for software that runs entirely on the user's machine?
2. Is stable-diffusion.cpp's MIT-licensed implementation of the H3 architecture a "Model Derivative" ("work based on MiniMax H3", §I.11)? If yes, would shipping an `sd` binary with H3 support in a worldwide release be distribution outside the territory under §V.4, even with no weights?
3. Does a UK-resident developer who writes, publishes and distributes worldwide the integration code (catalog row, command line, download button) risk liability to MiniMax for inducing or facilitating breaches by users in excluded territories, given the tick is a self-statement and the app does no location check? Does the tick and audit record materially reduce that risk?
4. Is a click-to-confirm tick in a desktop app an "enforceable term at least as protective" as §V and Exhibit A for §V.2, if its sentence accepts the licence and AUP by link? Does it need to be in the app's own end-user terms (the app is Apache 2.0 with no EULA today)?
5. Under §VI.2, may the app use the name "MiniMax H3" in the catalog, licence screen and notices while its author is outside the Applicable Territory and not a Licensee (nominative description), and does monetising ChaosEngineAI in future change the §IV.2 position?

## Recommended changes

Do these in order; only step 4 ships H3.

1. **Fix the record of the premise.** In D-99 and FU-166, replace "the user must pass the HF gate" with: none of the H3 repos are gated; the in-app licence screen is the only acceptance step.
2. **Change the tick sentence** in `TERRITORY_MODELS["minimax-h3"]["confirmSentence"]` (existing records stop counting, which is the intended behaviour; no user has one yet):

   > I am not in the European Union, the United Kingdom, the Republic of Korea or the United States, and I will not use this model or what it makes there. I accept the MiniMax H3 Community License, including its Acceptable Use Policy, and I am responsible for following it.

3. **Add one line to the licence screen and the row's `note`** (new i18n key with `null` placeholders in other locales; check `tests/test_licence_gate.py` and the `TerritoryGateNotice` tests for text they pin):

   > ChaosEngineAI does not include or distribute MiniMax H3. You download it from Hugging Face yourself, and you are the licensee. If you let other people generate with it (for example through remote connections or the agent bridge), you are offering a hosted service and the licence's duties for services (sections V.2 and V.5) apply to you. MiniMax may update its Acceptable Use Policy; read the current licence on Hugging Face.

   Link both the pinned licence (as now) and the live page `https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE`, so AUP updates are visible.
4. **Flip `H3_RELEASED = True` only after a lawyer answers questions 1 and 3** (the test that asserts `False` changes with it). If the answer to question 1 is "yes, §V.5 applies to you", keep it `False` until a reporting route (for example a "Report misuse" link to an email address or issue form) and a short safeguards page exist, or leave H3 out.
5. **Never bundle `sd` with H3 support in a worldwide release until question 2 is answered.** Today release CI builds no `sd`, so nothing changes now; note it in FU-166 so a later "ship sd in the bundle" task does not cross it silently.
6. **Optional (encouraged by §III.3.b and AUP item 12):** write the existing machine-generated tag, with `model: "MiniMax H3"`, into H3 video files, as music and cloned voices already do.

Unchanged: no territory lookup (the app never guesses location, D-76), no bundling, no default, UI-only confirm, owner never runs H3.
