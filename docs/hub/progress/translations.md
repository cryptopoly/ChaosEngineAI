---
title: Progress translations
status: done
---
# Progress: translations

Updated: 2026-10-29 (second pass added). Owner: translations engineer agent. Branch: `feat/models-fit` (no commit made by me; first pass is committed in the merge, second pass is in the working tree).

## Goal

Give all 10 shipping locales the new UI keys from `ui-core.md`, `settings-rec.md` (first pass), `ui-video.md` and `ui-chat.md` (second pass), so the catalogs match the inline English `defaultValue` text used in the code.

## Done

- 107 keys added to each of the 10 locales (`en` plus 9 translations): 81 from ui-core (5 in `common`, 76 in `library`, including the two `imageDiscover.sortLabel.*` keys and the three temporary `models.legacy.*` keys) and 26 from settings-rec (`studio`). Files touched: `src/locales/<locale>/{common,library,studio}.json` only (30 files).
- No source key left unplaced. The parser checked the English text of every key against the progress files (zero mismatches).
- 43 of the 107 keys per locale are copies of existing translations in the same locale (so the old and new pages say the same thing): `library:imageDiscover.{action,status,filter,summary}.*`, `library:imageModels.{action.openModelCard,action.browseCatalog,status.downloadFailed}`, `library:onlineModels.memory.{comfortableLabel,tightLabel}` (Fits / Tight), `library:card.license`, `library:filter.all`, `runtime:controls.engine`, `common:tabGroups.models`, `common:imageDiscover.sortLabel.{likes,downloads}`, `studio:voice.library.filter` (Show), and the studio labels (`image.qualityPreset|steps|guidance|sampler|diffusionCache`, `videoStudio.fields.frames|fps`, `imageStudio.toggles.cfgDecay|previewVae|fp8Layerwise` with the " (CUDA Ada+)" suffix removed).
- Checks run: `npm run i18n:validate` (100 % parity, ICU compile OK, no orphans), `npx vitest run src/locales` (10 passed). Every message with `{`/`'` was also rendered with `intl-messageformat` for counts 1, 2, 5, 21 (all plural forms correct, no swallowed apostrophes). Only additions: every existing key is unchanged and in the same position; `git diff` shows the new lines plus the unavoidable trailing-comma line changes (50 deleted lines across the 30 files).

## Second pass (ui-video, ui-chat)

- 56 keys added to each of the 10 locales (`en` plus 9 translations), all in `src/locales/<locale>/library.json`: 13 new from `ui-video.md` (`models.video.*` x5, `models.detail.ggufFile|textEncoder|catalogSize`, `videoModels.sharedRepo|sharedRepoTitle`, `videoModels.action.deleteShared{Gguf,TextEncoder,Component}`), the 3 older keys that were never in `en` (`videoDiscover.status.installing`, `videoDiscover.sortLabel.likes|downloads`; copied from the existing `common:videoDiscover.*` translations), and 40 from `ui-chat.md` (`models.chat.*`). No source key left unplaced; the parser compared the English text of all 56 against the two progress files (zero mismatches).
- Reworded `myModels.strategy.dflashEmptyTooltip` in all 10 locales. English: "... Download a base model from the Models page to enable DFlash." (was "from Discover"). Only the last sentence changed; the other translations point at "Models" (zh 模型 / ja モデル / de Modelle / ru Модели / ko 모델 / fr Modèles / es Modelos / pt Modelos). The sidebar entry for this page is actually called "Chat Models" (`tabs.chatModels`); say if the tooltip should name that instead.
- The three temporary `models.legacy.*` keys were removed by the lead before this pass and are not re-added. Net keys added by me per locale: 104 (first pass) + 56 = 160. `i18n:validate` coverage is 3003/3003 for every locale.
- Reused existing translations where the text matched: sorts and column labels (`imageModels.sort.*`, `myModels.column.format|backend`, `imageModels.column.released`), and the RAM sorts derived from the RAM/VRAM ones with "/VRAM" removed (`models.chat.sort.ram*`), so the old and new pages read the same.
- Subtitle `models.chat.subtitle` is built from the first-pass `models.subtitle` translation plus "{size} on disk" and a `{dirs}` plural (ru has one/few/many/other).
- Checks: `npm run i18n:validate` OK, `npx vitest run src/locales` 10 passed, every new message rendered with `intl-messageformat` (no errors, no swallowed apostrophes). Against HEAD the only changed existing value in any file is `dflashEmptyTooltip`; all other existing entries are unchanged and in the same order.
- Not mine: `tsc`-visible code still calls keys that have no catalog entry: `video.ltx2.install|installing|notInstalled|retry` (Ltx2EngineBanner), `videoStudio.ltx2.engineMessage` (VideoStudioTab), `installLog.statusNoun.ltx2` (InstallLogPanel). They show their inline English default in every locale.
- Judgement calls: "Capability" (filter label) is 能力 / 能力 / 機能 / Fähigkeit / Возможность / 기능 / Capacité / Capacidad / Capacidade; "Acceleration" follows the existing "All acceleration" wording; "thread" follows the existing chat sidebar (zh 线程, zh-TW 執行緒, ja スレッド, de Thread, ru поток, ko 스레드, fr fil, es hilo, pt-BR thread); ru keeps "скачивание" for download nouns in the library namespace (the existing `common` sort labels say "загрузок"); "Download" as the source badge (`models.chat.sourceDownload`) is the noun form (Скачивание, Téléchargement, Descarga).

## Not done

- Nothing from the two sources. Not touched on purpose: locale strings that still say Discover / My Models (listed in ui-core "Requests").

## Glossary chosen (verdicts and key concepts)

Order: zh-CN / zh-TW / ja / de / ru / ko / fr / es / pt-BR.

- Fits (existing): 可放入 / 可容納 / 収まる / Passt / Помещается / 적합 / Tient / Cabe / Cabe
- Tight (existing): 紧张 / 勉強 / ギリギリ / Knapp / Впритык / 빠듯함 / Juste / Justo / Apertado
- Won't fit (new): 放不下 / 放不下 / 収まらない / Passt nicht / Не поместится / 부적합 / Ne tient pas / No cabe / Não cabe
- Cannot run on this machine: 无法在本机运行 / 無法在本機執行 / このマシンでは実行できません / Läuft nicht auf diesem Rechner / Не запускается на этом компьютере / 이 컴퓨터에서 실행할 수 없음 / Ne peut pas s'exécuter sur cette machine / No se puede ejecutar en esta máquina / Não pode ser executado nesta máquina
- Not for this machine (`models.fit.unknown`): 不适合本机 / 不適合本機 / このマシン向けではありません / Nicht für diesen Rechner / Не для этого компьютера / 이 컴퓨터용이 아님 / Pas pour cette machine / No para esta máquina / Não para esta máquina
- Best fit: 最适合 / 最適合 / 最適 / Beste Wahl / Лучший выбор / 가장 적합 / Meilleur choix / Mejor opción / Melhor opção
- measured (tag; Studio badge is the capitalised form): 实测 / 實測 / 実測 / gemessen / измерено / 측정값 / mesuré / medido / medido
- estimated: 估算 / 估算 / 推定 / geschätzt / расчётно (short: расч.) / 추정값 / estimé / estimado / estimado
- "this machine": 本机 / 本機 / このマシン / diesem Rechner / этом компьютере / 이 컴퓨터 / cette machine / esta máquina / esta máquina
- Bucket chips (Installed / Available / Needs attention): 已安装, 可下载, 需要关注 / 已安裝, 可下載, 需要留意 / インストール済み, ダウンロード可能, 要確認 / Installiert, Verfügbar, Zu prüfen / Установленные, Доступные, Требуют внимания / 설치됨, 다운로드 가능, 확인 필요 / Installés, Disponibles, À vérifier / Instalados, Disponibles, Requieren atención / Instalados, Disponíveis, Precisam de atenção

## Ambiguities and judgement calls

1. `library:models.empty.availability` takes `{bucket}` as a lower-cased bucket label. To avoid grammatical agreement problems, every locale puts the label inside quotation marks as a category name (for example ru: «{bucket}» in "в разделе «{bucket}» нет моделей"), not as an adjective. A per-bucket sentence would read better in ru, de, fr, es.
2. `library:models.fit.unknown` has the English text "Not for this machine"; translated by the text, not by the key name.
3. "Available" is translated as "available to download" (zh 可下载, ja ダウンロード可能, ko 다운로드 가능) because it means models that are not installed yet. de / fr / es / pt-BR / ru use the plain word.
4. zh-TW "勉強" for Tight is the existing translation (Discover). It reads as "barely/reluctantly" in Taiwan Chinese; 吃緊 or 緊湊 may be better. Kept for consistency, flagged for review.
5. zh "Estimated": existing catalog uses 预计 / 預估 / 예상 (expected); here I used 估算 / 估算 / 추정값 because it is shown next to 实测 / 측정값 as a measured-versus-estimated pair.
6. Units: ru and fr keep the existing locale convention ГБ / Go in the new strings (the old strings in those files do the same); the other locales keep GB. The brief said not to translate units; say if you want GB everywhere.
7. `recommended.labels.size`: zh-CN / zh-TW use 尺寸 (dimensions) rather than the catalog's 大小 (file size), since this is image width x height. ko 크기, others as in the catalog.
8. `recommended.use` / `inUse` / `applyAll` mean "apply this value"; ja uses 適用, de Übernehmen / Aktiv, ru Применить / Применено, fr Appliquer / Appliqué, es and pt Usar / En uso, Em uso. `recommended.on` / `off` follow the existing toggle wording (de An / Aus, fr Activé / Désactivé, es Activado / Desactivado).
9. `recommended.measuredTitle` "Timed on {machine}." means a benchmark was run on that machine; translated as "measured on" (gemessen, mesuré, chronométré in fr).
10. `library:models.detail.fit` ("Fit on this machine") is a row label, so it is a noun in most locales (de Eignung für diesen Rechner, fr Adéquation à cette machine, ru Пригодность для этого компьютера).
11. Russian "Downloading" copies the library wording (Скачивается; "скачивание" in actions), while the old common namespace says "загрузка"; the library namespace is internally consistent, the two namespaces are not (existing).
12. Plurals: zh / ja / ko use only `other`; ru has one / few / many / other; the verb part of `confirmShared` in de / fr / es / pt-BR / ru is also a plural so the sentence agrees with the count (ru uses `=1` so 21 stays plural).
13. Seen but not mine: `videoDiscover.sortLabel.*` exists in `common`, `models.sort.*` pairs with `common:imageDiscover.sortLabel.*` wording on purpose. `library:imageDiscover.sortLabel.likes|downloads` are copies of the `common` ones.

## How to resume

1. `cd /Users/dan/ChaosEngineAI && git status` (locale files and this file are uncommitted).
2. `npm run i18n:validate && npx vitest run src/locales`.
3. Next step: have native speakers review de, ja, ru, zh-TW, then the lead commits. Generator scripts are in the session scratchpad only (not committed); re-running them is not needed.

## Decisions to propose

- Native-speaker review of the verdict wording above (Tight in zh-TW, Won't fit in ko 부적합, Best fit in all locales).
- Whether to write a per-bucket empty-state sentence instead of the `{bucket}` quote pattern (point 1).
- Whether ru / fr should show GB instead of ГБ / Go (point 6).
