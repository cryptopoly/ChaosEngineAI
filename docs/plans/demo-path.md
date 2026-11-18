# Podcast demo path (v0.10.0, Mac)

Status: **rehearsed four times with real models on 2026-11-18; R3 and R4 ran the script below end to end, twice in a row, with no stumble left unfixed.** Stream F of the overnight review (`docs/plans/overnight-e2e-review.md`). Evidence and every stumble: `docs/progress/review-f.md`, screenshots in `docs/progress/review-f-shots/`.

A six to eight minute live path on the M4 Max (64 GB) that shows ChaosEngineAI at its best: a fresh start, a fast local answer, a tool call, a film clip made from one sentence in chat, the sample film, a spoken line, and the OpenAI-compatible server. Nothing in the path downloads anything.

## The short version

| # | Step | Clicks | R3 time |
|---|---|---|---|
| 1 | First screen | Launch, I agree | 1 to 3 s |
| 2 | Load a model | Chat > Chat > Load Model > Qwen3.8 27B 4-bit > Load for Chat | 7 to 10 s |
| 3 | Fast answer, then a tool | type, Enter; Tools, type, Enter | 2 s, then 6 s |
| 4 | A film clip from chat | Studio, type, Enter, Approve | plan card in 38 to 60 s, render about 1 min 40 s |
| 5 | While it renders: Server | Server (collapse Connect your app) | instant |
| 6 | While it renders: the sample film | Projects > Open the sample > Play film | about 1 s |
| 7 | Back to the clip | Chat (the thread opens at the clip) | instant |
| 8 | Voice | Voice > Studio, type, Excited, Generate audio | 5 to 15 s |

R3 took 4 min 50 s and R4 3 min 44 s of clicks from opening Chat to the spoken line, with no talking. With talking it fills six to eight minutes.

## Before going on air (15 minutes, off camera)

1. **Model choice: Qwen3.8 27B MLX 4-bit (`mlx-community/Qwen3.8-27B-4bit`), context 32K.** In rehearsal it answered at about 30 tok/s, called the calculator once and cleanly, and made a correct Studio plan first time. LFM2.5 24B-A2B is four times faster (114 tok/s) but in R1 it sent malformed tool calls (four calculator cards, two marked "raw" and "unreadable"), asked for an 8-frame clip, and in Keep going it called `list_models` in a loop without making a project. Do not use it on air.
2. **Screen share at 1920x1080 with Settings > General > Appearance > UI scale 125 %** (new in this branch). At 1280x720 keep 90 % and collapse the Chats list (the `<` button next to New thread); at 720p and 110 % or more the thread has no room for messages.
3. **Settings > General > Appearance > Keep all sidebar groups open: on.** With it off, opening one group closes the one above and every row below moves while the pointer is on its way; in R1 three clicks landed on the wrong item.
4. **Warm up once:** load Qwen3.8 27B, ask one question, unload; run one Video Studio clip at SD; speak one line. A cold first load after a reboot is slower than the times below.
5. **Quit everything else that uses memory.** The path peaks at about 37 GB (chat model 15 GB, LTX-2.5 about 17 GB, speech 3.5 GB).
6. **Never open on air:** Server > Connect your app (expanded, its code samples and the Test dialog print the API key in clear text; on `review/integrate` the key is masked until Reveal, so this applies only to builds without it), Settings > Storage (data, delivery and model folders), Settings > Integrations (its snippets print the API key too), Logs, Diagnostics, Chat > Models row details (paths start with your home folder). Collapse Connect your app before the show; it re-opens when you leave the page and come back, so collapse it again on the way in (step 5).
7. **Chat > Models shows "Downloading 1"**: a paused partial download (Nemotron 3.5 Bolt E4B OptiQ, 0.1 of 17.8 GB) in your Hugging Face cache. Cancel it or keep Chat > Models off the path. The path below never opens it.
8. The model list contains a test model (`optimum-intel-internal-testing/tiny-random-lfm3`, 0.0 GB). Type in the search box rather than scrolling the list.
9. Focus mode on, notifications off, a second browser tab with the R3 clip open as the fallback video.
10. **Chat starts with Tools and Studio off.** Both switches are remembered between sessions. With Studio left on, the plain question in step 3 carries the studio tool list and waits 15 to 20 s on "Processing prompt". Turning Studio off moves the composer up a line; click into the text box again before typing.

## The script

### 1. First screen (target 0:20)

- **Click:** launch the app. Terms of use appears (about 1 s after the window). Click **I agree** (closes in about 1.3 s).
- **Say:** "This is ChaosEngineAI. Everything here runs on this laptop: chat, images, video, voice, music. No account, nothing leaves the machine."
- **Shows:** Dashboard: memory in use, model disk, the hardware fit card.
- **Fallback:** if the window stays on "Starting backend sidecar" past 20 s, quit and relaunch once; then go on from the Chat page, which needs only the backend.
- Do not read the Hardware Fit card aloud: "Recommended target: 70B class @ Native f16" reads like a 70B model at 16-bit, which does not fit in 64 GB (it means the f16 KV cache).

### 2. Load a model already on disk (target 0:30)

- **Click:** sidebar **Chat > Chat**, then **Load Model** in the empty thread (it opens the model picker; before this branch it went to the 131-row catalogue). Type `Qwen3.8-27B-4bit` in the search, click the row, click **Load for Chat**. Leave the context at 8K: R3 ran the whole path at 8K. (R2 used 32K; prompt processing was 5 s shorter, not worth the slider on air.)
- **Say:** "I already have a few dozen models on disk. I pick one, it loads into memory, about ten seconds."
- **Shows:** the picker with size, format, context and capability chips; the header reads the model name and READY.
- **Fallback:** if the load passes 30 s, press **Cancel load** and pick `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` (loads in under 20 s; plain chat only, skip step 4's Studio part and show the sample film instead).

### 3. A fast answer, then a tool (target 0:45)

- **Type:** `In two sentences: why does running AI on your own computer matter?` Enter.
- **Shows:** the reply streams in about 2 s; the chips under it show the speed (about 30 tok/s) and memory free.
- **Click:** **Tools** (turns blue, "Tools ON"). **Type:** `Use the calculator: what is 17.5% of 48,250, plus the square root of 7,921?` Enter.
- **Say:** "It can use tools: here a calculator, it also has code, web search and file reading."
- **Shows:** one calculator card with the expression, then the answer 8,532.75. About 7 s.
- **Fallback:** if the model answers without calling the tool, say "it did that one in its head" and move on. Avoid "Run Python to ...": the model routes it to the calculator with a list comprehension, which fails twice before it gets there (18 s, three cards).

### 4. A film clip from one sentence (target 1:00 of talk, render runs on)

- **Click:** **Studio** (turns blue, "Studio ON"; a **Keep going** button appears next to it).
- **Type:** `Make a 3-second video clip at 768x512: a lighthouse at dusk, waves crashing on the rocks, seagulls overhead.` Enter.
- **Shows:** "Processing prompt" for 15 to 20 s (the studio tools make a long prompt), a `list_models` card, then a **Plan to approve** card: model LTX-2.5, size, length, time estimate, memory. Nothing renders until you approve.
- **Say:** "It plans first and asks. Nothing runs until I say yes."
- **Click:** **Approve**. A job card shows the step, elapsed time and time left.
- **Time:** 1 min 43 s at 768x512 (R3; the card estimated 1 min 29 s), 1 min 48 s in R1; 3 min 36 s at 1280x704 in R2, when the prompt did not name a size. Name the size.
- **Fallback:** if a red "Invalid input" card appears, the model fixes the call itself on the next turn; wait for the second plan card. If no plan card after 60 s, press **Stop**, and open the fallback tab with the R3 clip.

### 5. While it renders: the server (target 0:45)

- **Click:** sidebar **Server**. If **Connect your app** is open, click its title first to collapse it (it holds the API key in clear text).
- **Say:** "The same models are an OpenAI-compatible server on localhost. Any app that speaks the OpenAI API can point here instead of the cloud."
- **Shows:** RUNNING, the address `http://127.0.0.1:8876/v1`, the loaded model card with its model id, port, warm pool, request count.
- **Do not click:** Test (its dialog prints the key), Reveal, Copy next to the key.

### 6. While it renders: the sample film (target 1:00)

- **Click:** sidebar **Projects**, **Open the sample** (about 1 s), **Play film**, then the round play button in the player (it opens paused).
- **Say:** "Chat can also plan whole films: shots, stills, clips, sound. This four-shot sample was made by the app on this Mac."
- **Shows:** The Lighthouse rough cut, 6.8 s, shot markers, captions button.
- **Fallback:** none needed; it plays files already on disk.

### 7. Back to the clip (target 0:30)

- **Click:** sidebar **Chat > Chat**. The thread opens at the bottom with the clip's player and Open in Library; scroll up a little to show the whole frame, click play.
- **Say:** "Text to video, on a laptop, in under two minutes."
- **Fallback:** if it is still rendering, read the time left from the job card and do step 8 first; come back after.

### 8. Voice (target 0:45)

- **Click:** sidebar **Voice > Studio**. **Type:** `Everything you just saw ran on this laptop. No cloud, no account, no meter running.` Click the **Excited** style chip, then **Generate audio**.
- **Shows:** 5 s in R3, 12 to 16 s in R1 and R2 (first use loads the speech model), then the player scrolls into view (new in this branch). Click play.
- **Do not** start it while the video renders: speech waits for the render to finish (R2: 3 min). This branch now says so on screen ("Waiting for the video render to finish").
- **Fallback:** pick a quick voice (Change voice > a Kokoro voice) if the natural voice is slow.

### Optional wow: Keep going (only with time to spare, and only rehearsed again)

Keep going lets the app send the next step after each turn until a film is done. With Qwen3.8 27B a two-shot film needs several nudges and minutes of rendering. It was not rehearsed end to end with Qwen in this run (R1 with LFM2.5 stalled), so do not run it live. Talk over a recording instead.

## Rehearsal timings

R1: 1280x720, UI scale 90 %, LFM2.5 24B-A2B, 8K context. R2: 1920x1080, 125 %, Qwen3.8 27B, 32K. R3: the script above.

R3: 1920x1080, 125 %, Qwen3.8 27B, 8K, the script above, all fixes in, a fresh data folder (terms accepted once before the timed run). The full pytest suite was running in the background during R3.

R4: the same as R3 from a true first launch (new data folder and the window's storage cleared), with the Playwright suite running in the background.

| Step | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| Backend ready (health) | about 3 s | 2.7 s | not timed | 2.5 s |
| Terms dialog shown after window | about 2 s | about 1 s | (accepted before) | 1.0 s |
| I agree closes | about 3 s | 1.3 s | (accepted before) | 1.3 s |
| Dashboard after reload | 2 s | 2 s | 2.6 s | (first launch) |
| Model load | 10 to 18 s (LFM2.5) | 8.6 s (Qwen3.8 27B) | 10.4 s | 7.0 s |
| Fast answer | 2.7 s, 114 tok/s | 2.3 s, 29.6 tok/s | 2.0 s, 29.5 tok/s | 2.2 s, 29.7 tok/s |
| Tool answer | 23 s, 4 calls, 2 failed | 18.3 s (primes, 3 calls); 6.9 s (calculator prompt, 1 call) | 5.8 s, 1 call | 7.2 s, 1 call |
| Studio plan card shown | 71 s and a failed call, then 7 s after "make it 3 seconds" | 34 s | 40 to 60 s | 37.9 s |
| Clip render | 1 min 48 s, 768x512, after a second Approve (model name refused) | 3 min 36 s, 1280x704 | 1 min 43 s, 768x512 | 1 min 38 s, 768x512 |
| Server page | instant | instant | instant | instant |
| Sample film open | under 1 s | about 2 s | about 1 s (after the dead-end fix) | about 1 s |
| Speech | 12 to 16 s | 15 s after a 3 min wait behind the render | 4.8 s | 6.2 s |
| Speech started during a render | not tried | 3 min, still bar, no word | says "Waiting for the video render to finish", done 28 s later | not tried |
| Keep going | stopped at 90 s, no project | not run | not run | not run |
| Stumbles left | many (fixed after) | voice wait (fixed after) | Projects dead end (fixed during) | none |

## Fixed in this branch (`review/f`)

- Chat model picker gave `Qwen3-0.6B-GGUF` the size and badges of a 39.8B catalogue row, then of the 27B row (format and quantization alone counted as a match).
- Empty-chat Load Model opened the catalogue page; it now opens the picker. Escape closes the picker.
- UI scale 110 % and 125 %.
- Calculator takes a bare-string expression and `sum(1, 2, 3)` (R1's failed cards).
- Studio tools accept an installed model's name ("LTX-2.5") for its id (R1's refused Approve).
- The chat thread follows tool cards, plan cards and job cards as they arrive (R1 and R2: the newest card sat below the fold).
- Speak scrolls the finished clip into view, and says when it waits behind another job.
- Projects opened on "This project is not there any more" when the remembered project was gone (R3, after a data folder change); it now opens the list.

## On-air risks, ranked by how much they would embarrass you

Fixed ones are in the list above. These are left, worst first.

1. **The API key on screen.** *(Masking fixed on `review/integrate`, `d6aa020b`: dots and the last 4 characters until Reveal, Copy copies the real key. LAN access is still the owner's to turn off.)* Server > Connect your app is open by default and its Python, JavaScript and config samples print the key in clear text next to the masked field; Test prints it in a curl line; Settings > Integrations prints it too; the masked field shows its first and last four characters. The Server page belongs to stream E (redesign, collapsed by default). Recommendation: mask the key in every sample and in Test until Reveal is pressed, and remember the collapsed state. Until then, follow prep item 6. Separately, your real settings have **LAN access on and Require API token off**: anyone on the studio's Wi-Fi could use the server during the show. Turn LAN access off for the day.
2. **The studio plan takes 40 to 60 s to appear** with Qwen3.8 27B (15 to 20 s of that is processing the studio tool list). Talk over it; the script does. Recommendation: cache the studio system prompt between turns (prefix cache) or shorten the tool descriptions; measure before and after.
3. **The model picks the video size.** Without "at 768x512" in the prompt, Qwen chose 1280x720 and the render took 3 min 36 s instead of 1 min 43 s. Recommendation: `generate_video` uses the model's recommended size unless the user's message asks for another (needs a change to the tool description and the golden tool list, so not done here).
4. **Small fast models misfire tool calls.** LFM2.5 sent unparseable calls, asked for frames under 8, and looped in Keep going. The calculator and model-name sides are fixed; the model is not. Stay on Qwen3.8 27B for anything with tools.
5. **Keep going is not demo-ready.** Not rehearsed with Qwen3.8 27B; with LFM2.5 it made no project in 90 s. Keep it out of the live path.
6. **First screen copy.** *(Fixed on `review/integrate`, `59da5ffa`: "70B class at 4-bit", and the activity line is written after the MLX probe.)* Dashboard reads "Recommended target: 70B class @ Native f16" and the activity feed "Recommended target: 70B GGUF" on an MLX Mac. Recommendation: say what fits ("70B class at 4-bit") and drop "Native f16" from the title.
7. **Server page names other apps** *(Owner, 2026-11-18, D-93: keep; compatibility targets are allowed, the rule bars inspiration and competitor mentions.)* ("Ollama-compatible apps", Open WebUI, Continue.dev) inside Connect your app, against the editorial rule. Stream E's file.
8. **Video Studio's heads-up** says "A loaded chat model holds about 30 GB" with the 15 GB Qwen loaded. Off the path; check the figure.
9. **A clean data folder is not a clean first launch.** The window's own storage keeps the last tab, the Studio and Tools switches, UI scale and the open project. Test the real first launch with a new macOS user if that matters on air.
10. **At 1280x720 the chat is cramped**: Thread name takes a full row and the composer toolbar wraps, leaving about 180 px for messages. Use 1920x1080.
11. **Accessibility gaps seen on the way** (not visible on air): sidebar buttons have no accessible names, the Tools and Studio chips have no pressed state, the launch sliders have no labels.

## Not tested

- The packaged app (Tauri window, its splash, its first 60 seconds); only the dev server in a browser pane at emulated sizes.
- A light theme: the app has none; it is dark whatever the system setting.
- Keep going with Qwen3.8 27B; voice in Chat (Call, read-aloud); web search on a real network (the backend ran with Hugging Face offline).
- Windows and Linux.
