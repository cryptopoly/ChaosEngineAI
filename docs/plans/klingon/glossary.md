# Klingon glossary and style guide

Source kinds: **canon** = a word from Marc Okrand's published vocabulary used in its published sense; **canon-sense** = a canon word used in a new, software sense (a speaker will recognise the word and the metaphor); **coined** = a compound or derivation built with Klingon grammar from canon parts, not attested; **English** = kept as is. Confidence: A = sure of the canon word and its fit, B = sure of the word, fit debatable, C = coined and needs a speaker's review. Every coined or canon-sense term is open to correction; `qa.md` lists them again for the reviewer.

## Style rules used throughout

1. **Orthography.** Okrand's Latin romanisation, case-sensitive: `a b ch D e gh H I j l m n ng o p q Q r S t tlh u v w y '`. `I`, `D`, `H`, `S`, `Q` are always capital; `q` is always lower. No sentence-initial capital. The apostrophe is a letter (`'`): never drop it, never curl it. No pIqaD.
2. **Buttons and menu items are imperatives.** Prefix `yI-` (singular or no object), `tI-` (plural object): `yIpol` Save, `yIqIl` Cancel, `Hoch tIwIv` Select all. A "don't" uses `-Qo'`: `yIpolQo'` Don't save.
3. **Status lines use the indefinite subject `-lu'`.** `yab chu'lu'ta'` the model has been loaded; `yab lI'lu'taH` the model is being downloaded. Aspect: `-ta'` done on purpose, `-pu'` done, `-taH` ongoing, `-lI'` ongoing towards a known end (progress bars), `-choH` change of state.
4. **Labels are bare nouns or noun-noun compounds** (the first noun qualifies the second): `yab pong` model name, `qawHaq lo'` memory usage. Adjectival verbs follow the noun: `yab chu'` new model.
5. **Numbers precede the noun, nouns do not take `-mey` after a number.** ICU plurals therefore have one branch: `{count, plural, other {# yab}}`. `#` stays. `-DIch` makes ordinals, `-logh` times.
6. **Word order is object-verb-subject.** `yab lI' De'wI'` the computer downloads the model. Time and manner adverbs come first: `DaH`, `wej`, `tugh`, `nom`.
7. **Yes/no questions take `-'a'`** on the verb: `yab DaQaw' DaneH'a'?` Do you want to delete the model? Confirmations are phrased as questions plus `HIja'` / `ghobe'` buttons.
8. **Possession.** `-lIj` your (thing), `-wIj` my (thing), `-Daj` its; `-lI'` / `-wI'` for beings. Models, files and projects are things.
9. **Keep English for** model names and repos, file formats and extensions, GPU, CPU, RAM, VRAM, MLX, GGUF, CUDA, Metal, MPS, API, URL, JSON, HTTP, LoRA, KV, FP16, INT4 and other quant names, token (both senses), seed, sampler, CFG, fps, Hz, GB, MB, ms, %, keyboard keys, product and company names, `{placeholders}`, code, paths, commands, and anything inside backticks. `ChaosEngineAI`, `Hugging Face`, `Kokoro`, `Whisper`, `Parakeet`, `FLUX`, `Wan`, `LTX`, `Stable Diffusion`, `ACE-Step`, `Stable Audio Open` are names.
10. **Length.** Klingon is terse; where a label would still be long, prefer the shorter canon noun over a precise compound and put the precision in the help text.
11. **Punctuation.** English punctuation is kept (`.`, `?`, `…`, `·`, `:`), since the UI relies on it for layout. `—` becomes `,` or a new sentence.

## Core verbs (canon, confidence A unless marked)

| English | Klingon | Kind | Note |
|---|---|---|---|
| save, keep | pol | canon | `yIpol` |
| cancel | qIl | canon | `yIqIl` |
| delete, destroy | Qaw' | canon | `yIQaw'`; heavier than English "delete", which fits a warrior UI |
| remove | teq | canon | for removing an item from a list, not from disk |
| install | jom | canon | TKD "install (a device)" |
| uninstall | jomHa' | coined (C) | `-Ha'` undo |
| load, activate, switch on | chu' | canon | TKD "engage, activate"; homophone of "be new" |
| unload, switch off | chu'Ha' | canon | TKD "disengage" |
| download | lI' | canon-sense (B) | TKD "transmit data to a place"; community usage for download |
| upload | lab | canon-sense (B) | TKD "transmit data away from a place" |
| import, bring in | qem | canon | |
| export, send out | ngeH | canon | "send" |
| generate, make, create | chenmoH | canon | |
| start, begin | tagh | canon | |
| stop | mev | canon | |
| pause | yev | canon | |
| continue, resume | taH | canon | |
| retry | nIDqa' | canon + `-qa'` | "try again" |
| open | poSmoH | canon | |
| close | SoQmoH | canon | |
| search, look for | nej | canon | |
| find, locate | Sam | canon | |
| discover, find | tu' | canon | |
| choose, select | wIv | canon | noun "wIv" = choice |
| change, edit | choH | canon | |
| update, renew, refresh | chu'moH | canon | "make new" |
| reset to default | motlhmoH | coined (C) | "make usual" |
| clear, empty | chImmoH | canon | |
| copy | qon | canon-sense (B) | TKD "record"; used for copy to clipboard |
| paste, place | lan | canon | |
| show, display | cha' | canon | TKD "display (a picture)" |
| hide | So' | canon | |
| view, see | legh | canon | |
| watch | bej | canon | |
| listen | 'Ij | canon | |
| hear | Qoy | canon | |
| speak, say | jatlh | canon | |
| record (audio) | qon | canon | |
| write, transcribe | ghItlh | canon | |
| read | laD | canon | |
| send | ngeH | canon | |
| receive | Hev | canon | |
| connect | rar | canon | |
| use | lo' | canon | |
| test, try out, sample, benchmark | waH | canon-sense (B) | |
| examine, inspect, scan, diagnose | nuD | canon | |
| estimate, judge | noH | canon | |
| recommend | chup | canon | `chuplu'` recommended |
| require | poQ | canon | |
| need | nIS | canon | |
| be able | -laH | canon suffix | |
| accept | laj | canon | |
| decline, refuse | lajQo' | canon | |
| verify, confirm | 'ol | canon | |
| warn | ghuHmoH | canon | |
| wait | loS | canon | `yIloS` please wait |
| finish | rIn | canon | `rIn` is finished; `pItlh` done! |
| succeed, work, function | Qap | canon | |
| fail | luj | canon | |
| happen | qaS | canon | |
| run (operate) | Qap / chu' | canon-sense (B) | "run a model" = `yab chu'`; "running" = `QaptaH` |
| quit, leave | mej | canon | |
| return, go back | chegh | canon | |
| be next | veb | canon | |
| precede | nung | canon | previous = `nungbogh` |
| train (a model) | qeq | canon-sense (B) | TKD "practice, train, drill" |
| convert | choH | canon | `choHmoH` convert something |
| compare | 'ov | canon-sense (C) | TKD "compete"; "pit against each other" |
| mix | DuD | canon | |
| press, click, tap | 'uy | canon | |
| drag | luH | canon | "yank" |
| drop | lan | canon | |
| preview | legh nung | coined (C) | "see first" |
| share | nob | canon | "give" |
| report (tell) | ja' | canon | |
| learn | ghoj | canon | |
| understand | yaj | canon | |
| know | Sov | canon | unknown = `Sovbe'lu'` |
| remember | qaw | canon | |
| forget | lIj | canon | |
| ask, request | tlhob | canon | |
| answer, reply | jang | canon | |
| permit, allow | chaw' | canon | noun "chaw'" permit, licence |
| lock | ngaQmoH | canon | `ngaQ` be locked |
| unlock | ngaQHa'moH | canon | |

## Core nouns

| English | Klingon | Kind | Conf. | Note |
|---|---|---|---|---|
| model (AI) | yab | canon-sense | B | TKD "mind, brain". One term for chat, image, video, music and voice models. Alternative a reviewer might prefer: `velqa'` (replica, model). |
| language model, chat model | ja'chuq yab | coined | C | |
| image model | mIllogh yab | coined | C | |
| video model | mIllogh vIH yab | coined | C | |
| voice model | ghogh yab | coined | C | |
| music model | QoQ yab | coined | C | |
| data | De' | canon | A | |
| computer, machine | De'wI' | canon | A | "this machine" = `De'wI'vam` |
| file | nav | canon-sense | B | TKD "paper, sheet" |
| folder, directory | nav ghom | coined | C | "paper group" |
| path | He | canon | A | "route, course" |
| memory (RAM) | qawHaq | canon | A | TKD "memory banks" |
| storage, disk space | polmeH Daq | coined | C | "place for keeping" |
| on disk | pollu' | coined | C | "is kept" |
| cache | cache | English | | technical token (KV cache, First Block Cache) |
| engine (inference) | jonta' | canon | A | TKD "engine" |
| backend | qoD pat | coined | C | "inner system" |
| worker (subprocess) | vumwI' | canon derivation | A | |
| process, procedure, step | mIw | canon | A | |
| system | pat | canon | A | |
| device, tool | jan | canon | A | |
| machinery, hardware | mIqta' | canon | A | |
| screen, display | HaSta | canon | A | TKD "visual display" |
| window | Qorwagh | canon | A | |
| panel, board | 'echlet | canon | A | |
| switch, button | leQ | canon | A | |
| control panel, settings | SeHlaw | canon | A | TKD "control panel"; used for the Settings tab and Preferences |
| option, choice, setting | wIv | canon | A | |
| default | motlh | canon | A | "usual" |
| custom, special, advanced | le' | canon | B | "special"; advanced options = `wIv le'` |
| list, menu | tetlh | canon | A | TKD "list, roll, scroll" |
| collection, library, gallery | ghom | canon-sense | B | "group"; Voice Library = `ghogh ghom`, Image Gallery = `mIllogh ghom`, My Models = `yabmeywIj` |
| project | Qu' | canon-sense | B | TKD "task, mission, duty"; a storyboard project is a mission |
| film | lut | canon-sense | B | TKD "story" |
| workspace | vumpa' | coined | C | "work room" |
| dashboard | wIy | canon | A | TKD "tactical display" |
| chat, conversation | ja'chuq | canon | A | TKD "discuss, confer" |
| session, thread | ja'chuq | canon-sense | B | same word |
| message, prompt | QIn | canon | B | TKD "message"; prompt also `QIn` |
| negative prompt | QIn Qo' | coined | C | "no-message"; help text explains |
| reply, response | jang | canon | A | |
| word | mu' | canon | A | |
| sentence, line of text | mu'tlhegh | canon | A | |
| text, writing, document, transcript, caption | ghItlh | canon | A | |
| name | pong | canon | A | |
| number, count | mI' | canon | A | |
| size | tInghach | coined | C | `-ghach` on a bare verb; marked but understandable |
| length (duration), time | poH | canon | A | |
| speed | Do | canon | A | TKD "velocity" |
| quality, level, rank, rating | patlh | canon | B | |
| strength | HoS | canon | A | also power, energy |
| temperature (sampler) | Hat | canon | A | |
| context (window) | ghu' | canon-sense | B | "situation" |
| image, picture, still | mIllogh | canon | A | |
| video, clip | mIllogh vIH | coined | C | "moving picture"; clip also `mIllogh vIH` |
| frame | mIllogh | canon | B | counted frames of a clip |
| voice | ghogh | canon | A | |
| speech (the sound) | QIch | canon | A | |
| sound, audio, noise | wab | canon | A | |
| music | QoQ | canon | A | |
| song, track | bom | canon | A | |
| microphone | QoywI' | coined | C | "hearer" |
| speaker (person) | jatlhwI' | canon derivation | A | |
| voice clone | ghogh velqa' | coined | C | "voice replica" |
| copy, replica, template, example | velqa' | canon | B | TKD "replica, model" |
| sample | waH | canon-sense | B | |
| recording | qonta'ghach | coined | C | prefer `qon` as verb |
| language | Hol | canon | A | |
| English | DIvI' Hol | canon | A | "Federation language" |
| Klingon | tlhIngan Hol | canon | A | |
| user | lo'wI' | canon derivation | A | |
| person, character (cast) | ghot | canon | A | cast = `ghot ghom` |
| place (scene location) | Daq | canon | A | also "location" generally |
| shot (storyboard) | 'ay' | canon-sense | B | "section, part" |
| take | nID | canon-sense | B | "attempt" |
| storyboard | lut 'echlet | coined | C | "story board" |
| mixer | DuDwI' | canon derivation | A | |
| agent | Duy | canon | A | TKD "agent, emissary" |
| skill | laH | canon | A | "ability" |
| plugin, extension | chelwI' | coined | C | "adder" |
| server | jabwI' | canon-sense | B | TKD "waiter, server (of food)" |
| job, task | Qu' | canon | A | |
| error, mistake | Qagh | canon | A | |
| fault, blame | pIch | canon | A | |
| defect, malfunction | Duy' | canon | A | |
| warning, alert | ghuH | canon | A | |
| danger, risk | Qob | canon | A | |
| damage | QIH | canon | A | |
| status, condition | Dotlh | canon | A | |
| progress | ruch | canon-sense | C | "proceed" |
| history | qun | canon | A | |
| log, journal | QonoS | canon | A | |
| fact, finding | ngoD | canon | A | |
| advice, tip, hint | qeS | canon | A | |
| plan | nab | canon | A | |
| goal | ngoQ | canon | A | |
| resources | jo | canon | A | |
| power, energy | HoS | canon | A | |
| success | Qapla' | canon | A | |
| battle, run (benchmark run) | may' | canon-sense | C | a benchmark run is a battle; `waH` is used where "test" is meant |
| treaty, agreement, terms | mab | canon | A | |
| permit, licence | chaw' | canon | A | |
| source | Hal | canon | A | |
| shape, aspect ratio | chen | canon | B | |
| route, path | He | canon | A | |
| range | chuq | canon | A | |
| maximum | 'aqroS | canon | A | |
| minimum | rav | canon | A | |
| all, every | Hoch | canon | A | |
| none, nothing, zero | pagh | canon | A | |
| other, another | latlh | canon | A | |
| some | 'op | canon | A | |
| second / minute / hour / day / month / year | lup / tup / rep / jaj / jar / DIS | canon | A | |
| percent | vatlhvI' | canon | A | |
| now | DaH | canon | A | |
| not yet | wej | canon | A | |
| soon | tugh | canon | A | |
| recently | qen | canon | A | |
| here, local | naDev | canon | A | "on this machine" = `De'wI'vamDaq` |
| ready | ghuH | canon | B | "be alert, prepared" |
| idle | leStaH | canon | A | "is resting" |
| busy, working | vumtaH | canon | A | |
| loading (model) | chu'lI' | canon | A | |
| loading (data) | qemlI' | canon | A | "bringing" |
| downloading | lI'lI' | canon-sense | B | |
| generating | chenmoHlI' | canon | A | |
| installed | jomlu'ta' | canon | A | |
| selected | wIvlu'ta' | canon | A | |
| available | tu'lu' | canon | A | "there is" |
| missing, lacking | Hutlh | canon | A | |
| unknown | Sovbe'lu' | canon | A | |
| new | chu' | canon | A | |
| old | ngo' | canon | A | |
| empty | chIm | canon | A | |
| full | buy' | canon | A | |
| fast / slow | nom / QIt | canon | A | adverbs |
| big / small | tIn / mach | canon | A | |
| long / short (time) | nI' / run | canon | A | |
| good / bad | QaQ / qab | canon | A | |
| excellent | pov | canon | A | |
| stable / unstable | ngaD / ngaDHa' | canon | A | |
| correct / wrong | lugh / muj | canon | A | |
| same / different | rap / pIm | canon | A | |
| open / closed | poS / SoQ | canon | A | |
| on / off | chu'lu' / chu'Ha'lu' | canon | B | "engaged / disengaged" |
| yes / no | HIja' / ghobe' | canon | A | |
| OK, fine | lu' | canon | A | |
| done! | pItlh | canon | A | interjection |
| enough | yap | canon | A | |
| many / few | law' / puS | canon | A | |
| left / right | poS / nIH | canon | A | |
| up / down (area above/below) | Dung / bIng | canon | A | |
| before / after | nungbogh / vebbogh | canon derivation | B | |
| first / last | wa'DIch / Qav | canon | A | |
| what / where / how / why / when / who / how many | nuq / nuqDaq / chay' / qatlh / ghorgh / 'Iv / 'ar | canon | A | |
| AI | AI | English | | kept; "De'wI' yab" (computer mind) is used only in prose |
| GPU, CPU, RAM, VRAM, MLX, GGUF, CUDA, Metal, MPS, API, URL, JSON, token, seed, sampler, CFG, fps, LoRA, KV, FP16, INT4, GB, MB | same | English | | |
| version | version | English | | written as `v0.10.0` where possible |
| Hugging Face, Kokoro, Whisper, Parakeet, Nemotron, FLUX, Wan, LTX, ACE-Step, ChaosEngineAI | same | English | | names |

## Phrase patterns

| English | Klingon | Note |
|---|---|---|
| Are you sure? | DaneHbej'a'? | "do you certainly want it?" |
| This cannot be undone. | choHHa'laHbe'lu'. | |
| Please wait… | yIloS… | |
| Something went wrong. | Qagh tu'lu'. | "there is an error" |
| Try again. | yInIDqa'. | |
| No results. | pagh tu'lu'. | |
| Not installed. | jomlu'be'. | |
| Not loaded yet. | wej chu'lu'. | |
| X is loaded. | X chu'lu'ta'. | |
| X is ready. | ghuH X. | |
| {n} of {total} | {total}vo' {n} | "n out of total" |
| Learn more | yIghoj | |
| Open on GitHub | GitHubDaq yIpoSmoH | |
| Coming soon | tugh | |
| Beta, experimental | waHlu'taH | "is being tested" |
| Settings → Language | SeHlaw → Hol | |

## Coined terms for a speaker to review

`jomHa'` uninstall · `motlhmoH` reset · `legh nung` preview · `yab` model (sense) · `ja'chuq yab` chat model · `mIllogh yab` image model · `mIllogh vIH` video · `mIllogh vIH yab` video model · `ghogh yab` voice model · `QoQ yab` music model · `nav ghom` folder · `polmeH Daq` storage · `pollu'` on disk · `qoD pat` backend · `vumpa'` workspace · `QIn Qo'` negative prompt · `tInghach` size · `QoywI'` microphone · `ghogh velqa'` voice clone · `qonta'ghach` recording · `lut 'echlet` storyboard · `chelwI'` plugin · `ruch` progress · `may'` benchmark run · `'ov` compare · `lI'` download · `lab` upload · `qon` copy · `waH` benchmark · `qeq` train · `nav` file · `ghom` library · `Qu'` project · `lut` film · `ghu'` context · `'ay'` shot · `nID` take · `jabwI'` server · `velqa'` template · `patlh` quality · `chen` aspect ratio · `Hat` temperature · `He` path · `chu'` load · `yabmeywIj` My Models · `DIvI' Hol` English.

## Added during translation (per file, for the speaker review)

Terms the per-file translators needed beyond the table above. Same marking: a canon word in a new sense, or a compound built from canon parts. None is presented as canon in its new sense.

| Klingon | Meaning and source | File |
|---|---|---|
| `'ab` | height (canon verb 'have a height of', used as the label) | studio |
| `'ab tIn` | vertical (tall) | studio |
| `'ay' choH` | inpaint (canon 'section' + 'change') | library |
| `'el` | access (canon 'enter') | library |
| `'IQ` | sad (canon) | studio |
| `'olghach` | verification | studio |
| `ABI package` | ABI-dependent package (kept English) | studio |
| `bombe'` | instrumental (no song / no singing) | chat |
| `bot` | blocked (canon 'prevent, block') | studio |
| `buS nIS` | needs attention (needs focus) | library |
| `cha'ghach` | playback (the displaying) | studio |
| `chegh nav ghom` | rollback directory (return folder) | studio |
| `chenmoHghach` | generation (a generating, the act); -ghach on chenmoH, used as the noun 'generation run' | studio |
| `chenmoHwI' pong` | attribution (maker's name) | studio |
| `chenta'` | render (the made thing) | chat |
| `chevwI'` | filter (separator) | studio |
| `chu'moHghach` | update, upgrade (noun) | studio |
| `chu'wI'` | playhead / transport (activator); also 'hook' in templates | studio |
| `Daw'` | random (canon: revolt; used as 'random/unruly') | studio |
| `De' 'el` | input (data that enters; 'el = enter, canon) | chat |
| `De' naQ` | technical details (complete data) | studio |
| `De' nav` | web page (data sheet) | chat |
| `De' pat` | network (data system), in the LongLive install note | library |
| `De'wI' wIv` | automatic (computer's choice) | studio |
| `Delghach` | description | studio |
| `Do DubwI'` | performance booster (speed improver) | studio |
| `Dochmeyvam` | these items (Doch = thing, canon) | chat |
| `DuD choH` | crossfade (mix change) | studio |
| `DuH` | probability (canon: be possible) | chat |
| `Dung / bIng` | ascending / descending (canon 'area above' / 'area below') | library |
| `ghItlh jatlhwI'` | text-to-speech (text speaker) | studio |
| `ghItlh nap` | draft (simple writing) | studio |
| `ghItlhghach` | dictation, the act of writing | studio |
| `ghogh 'el` | voice input / dictation | chat |
| `ghogh ghom jan` | voice pack (voice collection device) | studio |
| `ghogh jo` | voice pack (voice resources) | chat |
| `ghoj Do` | learning rate (learning speed) | chat |
| `ghom Qu'` | batch (group task) | studio |
| `ghor` | broken (canon 'break') | library |
| `ghoS SIbI'` | go straight to | studio |
| `Hal bejlu'taH` | watching upstream (source is being watched) | library |
| `Hal Daq` | repository (source place) | studio |
| `Hap` | file format (canon 'matter, material'; used as the stuff a model file is made of: GGUF, MLX, safetensors) | library |
| `He (lane)` | lane in the mixer (route) | studio |
| `Hop puS` | shallow depth of field (short distance) | studio |
| `Huv` | clear/sharp, 'Huvbe'' blurry (canon 'be clear, not obstructed') | studio |
| `ja'chuq (session)` | this session (as in activity log) | studio |
| `jatlh mev tu'wI'` | turn / voice-activity detection (detector of speech stopping) | chat |
| `jen` | high, tall; portrait orientation, high quality (canon 'be high') | studio |
| `jo` | pack (canon: resources) | studio |
| `jo ghom` | bundle (resources group) | studio |
| `jomghach` | install, installation (noun) | studio |
| `jonta' wIv A/B waH` | runtime-profile A/B test | chat |
| `jot` | calm (canon 'be calm') | studio |
| `juch` | width / landscape (canon verb 'have a width of', used as the label) | studio |
| `laH 'agh` | tech demo (ability demonstration) | chat |
| `lajghach` | acknowledgement (the accepting) | studio |
| `lan (ngoqvo')` | paste (place from the clipboard) | studio |
| `legh nung` | preview (see first) | studio |
| `lI'ghach` | download (noun, the act of downloading) | studio |
| `lI'lu'ta'bogh` | the download (the files that were downloaded), used for 'delete download' | library |
| `loS reD rap` | square (four equal sides) | studio |
| `loStaH` | pending (is waiting) | library |
| `machmoH` | compressed (KV cache) (canon derivation: make small) | library |
| `mech chaw'` | commercial licence (mech trade + chaw' permit) | studio |
| `mech QIn` | advertisement (trade message) | studio |
| `meq` | reasoning (canon 'reason, logic') | library |
| `mI' SImwI'` | calculator (number computer; SIm = calculate, canon) | chat |
| `mIr` | sort order (canon: chain) | studio |
| `mIS ngoQ` | target entropy (confusion goal) | chat |
| `mIw rIt` | function calling (canon 'procedure' + 'summon') | library |
| `mugh` | translate (canon) | studio |
| `nab` | cache strategy (canon 'plan') | library |
| `nab 'echlet` | plan card (plan panel) | chat |
| `naDev ja'chuq chu' yItagh` | branch / fork from here (start a new chat here) | chat |
| `naQbe'` | incomplete (canon 'be whole' + not) | library |
| `nargh` | released (canon 'appear'; column header for release date) | library |
| `nav laDwI'` | file reader | chat |
| `nav wIvwI'` | file picker | studio |
| `nen` | age, maturity (canon 'mature' - used as noun, review) | studio |
| `ngI'` | weights (model weights) (canon 'have a weight of') | library |
| `ngoD` | result / output (canon: fact) | chat |
| `ngoqmoH` | encode (make into code) | studio |
| `ngoqwI'` | encoder (code maker) | studio |
| `ngu'` | identify, label (canon) | studio |
| `ngutlh` | character, letter (canon) | studio |
| `nobwI'` | provider (canon derivation: giver) | library |
| `nommoHwI'` | accelerator, acceleration (canon derivation: thing that makes fast) | library |
| `nuqneH` | welcome (canon greeting, literally what do you want) | chat |
| `parHa'` | favourite, like (canon 'like', the undo of par dislike) | library |
| `pe'` | trim, cut (canon: cut) | studio |
| `per` | tag, label (canon: label) | studio |
| `pIH` | expect (canon) | studio |
| `pImwI'` | variation (thing that differs) | studio |
| `pixel chenmoH` | decode (make pixels) | studio |
| `po'wI'` | expert (canon derivation from po' be expert) | studio |
| `poH He` | timeline (time route) | studio |
| `pong yIchoH` | rename (change the name); used for Rename session | chat |
| `poS / ngaQ` | open / gated access (canon 'be open' / 'be locked') | library |
| `qaD` | challenge (canon noun, used for HTML Challenge) | chat |
| `qawHaq polwI'` | memory saver | chat |
| `Qeb` | loop (canon: ring) | studio |
| `QeH` | angry (canon) | studio |
| `QIch ghItlhwI'` | speech-to-text (speech writer) | studio |
| `QIch yajwI'` | speech recognition (speech understander) | chat |
| `Qob nuD` | security scan (danger inspection) | chat |
| `Qoj / yIQojmoH` | random / randomise (Qoj = be random, canon) | chat |
| `qon Daq` | clipboard (recording place) | studio |
| `qonghach` | recording (noun); used instead of the glossary's qonta'ghach | studio |
| `qorDu'` | model family (canon 'family') | library |
| `Qub` | thinking (canon 'think') | library |
| `Qub mIw` | thinking mode | chat |
| `ra' tetlh (/)` | slash command menu (command list) | chat |
| `ratlh` | remain (canon) | studio |
| `rItlh` | chroma/pigment (canon 'paint, pigment') | studio |
| `SaHbe'` | random (does not care) | studio |
| `Say'moH` | tidy up, clean (canon 'make clean') | studio |
| `Seg` | variant, kind (canon 'type, kind'; used for model variants) | library |
| `Sor chuq` | tree budget (tree range) | chat |
| `Sum` | be near (canon) | studio |
| `tam` | swap (canon 'exchange, substitute') | studio |
| `tammoH / tammoHHa'` | mute / unmute (make quiet / undo) | chat |
| `teS QoywI'` | headphones (ear hearer) | chat |
| `tI'` | repair (canon) | studio |
| `tlhoS` | tight fit (canon 'almost'; 'almost does not fit') | library |
| `tlhup` | whisper (canon) | studio |
| `tujchoHtaHvIS` | while warming up (literal: while becoming hot) | studio |
| `tujmo' QIt` | thermal throttle (slow because of heat) | chat |
| `vaQ` | excited (canon 'be vigorous, aggressive, effective'; no canon word for excited) | studio |
| `veH` | limit, threshold (canon: boundary) | studio |
| `velqa' mIw` | simulation (replica process) | chat |
| `veQ` | trash (canon: garbage), .trash folder | chat |
| `vumpa'` | Studio (glossary workspace term reused for Image Studio / Video Studio: mIllogh vumpa') | library |
| `wab HoS ngoQ` | loudness target | studio |
| `web leghwI'` | web browser (web viewer) | chat |
| `wIv le'` | override (special choice) | chat |
| `wIy` | overview tab (tactical display) | studio |
| `wovmoHwI' neon` | neon lights | studio |
| `wuqghach` | decision (wuq = decide) | studio |
| `yab HaSta` | Models page (model + display) | library |
| `yab nav` | model card (model + paper) | library |
| `yab nung` | draft model (preceding model) | chat |
| `yap` | fit, be enough (canon) | studio |
| `yevlu'ta'` | paused (canon 'pause' + passive) | library |
| `yIbagh / yIbaghHa'` | pin / unpin (bagh = tie, canon) | chat |
| `yIlaD 'ej yIjatlh` | read aloud (read and speak) | chat |

Also in common.json: `jang pIH mIw` speculative decoding (reply-anticipation method), `mIw cha'DIch` fallback (second method), `yab tuj` warm model, `Daq chIm` headroom, `qawHaq buy'` memory pressure, `HoS polwI'` battery, `wanI' tetlh` activity feed, `De' nav` web page, `qaD` challenge, `laD neH token` read-only token, `Duy 'el` agent access, `He Hal` primary navigation, `ta'` action (column), `raS` table (view), `yIbagh` pin, `nIteb tagh` auto-start, `jabwI'` server, `lu'` OK, `pItlh` done.
