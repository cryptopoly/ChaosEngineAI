# Translation doubts, 2026-11-23 pass (feeds FU-254 and FU-421)

Strings filled by the FU-420 final pass (104 keys per locale, machine translation) that the translator was unsure about. One section per locale; native-speaker review welcome. Terms were taken from each locale's existing wording (take, vocal, shot, seed, VRAM, offload, menu paths such as Chat > Models).

Not translations: the `meta.supportedLocales` array in each of the nine locale files now also carries the experimental `tlh` entry (code, endonym, flags are language-neutral and copied from `en`; only `en`'s array is read by the app). The `tlh` locale files were not touched.

## Common to all nine locales

- "vocal stem" (videoStudio.lipSync.hint, board.song.vocals.*, board.editor.vocalTakesHint): no earlier string used the word. Kept the loanword "stem" where the music world does (ja ステム, de Stem, ru стем, ko 스템, fr/es/pt stem) and 人声分轨 / 人聲分軌 in Chinese. Native check wanted.
- "sync score / sync scoring": a new term. Rendered as 同步评分 / 同步評分, 同期スコア, Sync-Bewertung, оценка синхронизации, 싱크 점수, score de synchro, puntuación de sincronía, pontuação de sincronia.
- "Sung vocal" label (lipSync.singing, board.editor.vocalSinging): short checkbox label contrasting with speech; wording is a guess in every locale.
- "Experts in system RAM" (runtime.moeOffload.*): "experts" rendered with each locale's mixture-of-experts term (专家, エキスパート, Experten, эксперты, 전문가, experts, expertos, especialistas); no earlier string used it.

## zh-CN

- videoStudio.lipSync.takes / board.editor.vocalTakes: "Takes" as 版本 (matches the existing 版本 for takes and variations; could be mistaken for software versions).
- models.fit.fitsOffload: 卸载后可放入; existing strings keep "CPU offload" untranslated, 卸载 is used only in zh-TW.
- runtime.moeOffload.body: 350 亿 / 30 亿 parameter figures written out; check the number style.

## zh-TW

- videoStudio.lipSync.* prose uses 嘴型同步 (as in board.editor.vocalHint) while the title keeps the existing 對嘴人聲; mixed terms.
- videoStudio.lipSync.takes: 版本 for takes (see zh-CN).
- models.fit.fitsOffload: 卸載後可容納.
- common.settingsTab.providers.subtitle: 資料目錄 for "data folder" to match the existing Data Directory title.

## ja

- videoStudio.lipSync.singing / board.editor.vocalSinging: 歌唱ボーカル; may read stiff as a toggle label.
- models.fit.fitsOffload: オフロードで収まる; "offload" is a loanword.
- runtime.moeOffload.hint: "Mixture-of-Experts" left in Latin script; エキスパート used for the experts themselves.
- models.chat.recommended.popular: 最も人気 as a short chip label.
- board.song.vocals.stemRemove: 代わりに曲を使う; check it reads as a button.

## de

- videoStudio.lipSync.singing / board.editor.vocalSinging: "Singstimme" for "Sung vocal".
- Shot terms follow the section: "Einstellung" in the editor strings that already existed, "Shot" in board.song.* (as board.song.cut.*); mixed.
- models.fit.fitsOffload: "Passt mit Auslagerung"; "Auslagerung" may be unfamiliar next to "Offload".
- models.chat.recommended.fastest / newest / popular: "Am schnellsten", "Neueste", "Am beliebtesten" as chip labels; gender and form against "Beste Wahl".
- board.song.vocals.done: dash and "aus dem Stem / aus dem Song".

## ru

- videoStudio.lipSync.takes: "Дубли" (matches the existing дубли); estimate and score.* use дубль / дубля / дублей.
- videoStudio.lipSync.singing / board.editor.vocalSinging: "Пение" as a label for "Sung vocal".
- Shot terms follow the section: "шот" in existing editor strings, "кадр" in board.song.* (as board.song.cut.*).
- board.song.vocals.done: "из стема" (genitive of a loanword); "получили свой вокал" is a bit stiff.
- models.chat.recommended.fastest / newest / popular: feminine "Самая быстрая ... " assumes the label stands for "модель".
- runtime.moeOffload.label: "Эксперты в системной RAM" (RAM left in Latin script as in existing strings).

## ko

- videoStudio.lipSync.* and board.editor.vocal*: 샷 in editor strings (existing), 컷 in board.song.*; the same shot is named two ways.
- models.fit.fitsOffload: 오프로드하면 적합 (existing strings use 오프로드).
- runtime.moeOffload.body: quotes around '메모리에 맞추기' must match that control's real label (runtime.controls.fitInMemory).
- board.song.vocals.done: 컷 #개에 보컬을 지정했습니다(...기준) is a paraphrase of the English "from the stem/song".

## fr

- Takes as "prises" (matches the existing "prises"); "graine" for seed (matches the existing "Graine").
- videoStudio.lipSync.hint, board.song.vocals.hint: "stem vocal" is a loanword; "piste vocale" might be preferred.
- models.fit.fitsOffload: "Tient avec déchargement"; "offload" has no settled French term.
- runtime.moeOffload.body: guillemets around « Tenir en mémoire » must match the control label.
- dashboard.startHere.step2 / recommended.intro / bestFit.*: "Discussion" used for the Chat tab as in the existing path "Discussion > Modèles".
- board.song.vocals.title: "Synchro labiale de la chanson" as a heading.

## es

- Takes as "tomas" (matches the existing "tomas"); "semilla" for seed (voice.speak.seed uses Semilla, image strings keep Seed).
- models.fit.fitsOffload: "Cabe con offload" (loanword).
- "equipo" used for "this computer"; existing strings mix "esta máquina" and "este equipo".
- board.editor.vocalStart: "Inicio en la voz (s)" short label.
- runtime.moeOffload.body: «Ajustar a memoria» must match the real label of runtime.controls.fitInMemory.

## pt-BR

- Takes as "versões" (matches the existing board.song.cut.confirmBody and chat variations); the tab caption uses "takes". Pick one.
- models.fit.fitsOffload: "Cabe com offload" (loanword).
- runtime.moeOffload.*: "especialistas" for experts (pt mixture-of-experts term); aspas around "Caber na memória" must match the real label.
- models.chat.recommended.popular: "Mais popular" as a chip label.
- videoStudio.lipSync.takesHint: "seed" kept as loanword (as existing).
