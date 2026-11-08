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
