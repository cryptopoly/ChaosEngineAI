# Translation doubts, 2026-11-20 pass (feeds FU-254)

Strings filled by the D-97 translation pass (254 keys per locale, machine translation) that the translator was unsure about. One section per locale; native-speaker review welcome.

## zh-CN

- serverActivity.field.bytes: "Weights" rendered as 权重 (could be read as model weights vs file size)
- serverActivity.reason.user: 由你操作 (reads "by you"; maybe 手动 is more natural)
- serverActivity.summary.parked / kind.parked: "Kept warm" -> 保持预热; term 热池 exists for warm pool, 预热 chosen for the verb
- benchmarkTab.scoring.nextToken: 下一个 Token 选择 (awkward, native check)
- voice.speak.queue.*: "Speech" rendered as 朗读 (matches Speak tab term assumed, not verified)
- voice.models.macOnlyNote: 朗读/语音 used as tab names; verify against actual zh-CN tab labels
- voice.models.sttGuide.*: word "pick" -> 首选; marketing tone check
- conversionTab.qualityChange: {change}% sentence assumes change carries its own sign (e.g. +3 / -2); "较源模型{change}%" may read oddly
- musicStudio.copy.wayDeveloper: Windows menu path names 开发人员模式/开发者选项 depend on Windows build
- voice.runtime.aboutTickedGb: "turn detection" -> 话轮检测

## zh-TW

- serverActivity.kind.generation / summary.generation: "生成" vs existing "產出" for Makes; check consistency.
- serverActivity.summary.parked, kind.parked: "保持預熱" for "Kept warm"; native review.
- serverActivity.field.prefillSeconds: "預填" (Prefill) is a technical term, unsure about reader familiarity.
- benchmarkTab.evalNeedsMlx: "輸送量" for Throughput; verify against existing zh-TW mode label.
- studio voice.models.wordTimingsLine: long and technical; sentence structure may read awkwardly.
- voice.runtime.aboutTickedGb: "輪替偵測" for "turn detection" (probably better "對話輪次偵測"); native review.
- voice.speak.needsSpeechGpu: menu path uses "語音 > 模型 > 語音執行階段 > 附加元件"; must match actual zh-TW labels.
- musicStudio.copy.wayDeveloper: Windows menu names (設定、系統、開發人員專用) should match Windows zh-TW UI.
- board.editor.vocal*: "對嘴人聲" for "Vocal for lip sync"; "鏡頭" used for shot.
- fineTuning.notAvailableBody: "adapter" left untranslated; check existing zh-TW term for adapters.
- promptLibrary.inUseNote: "對話串" for thread; check existing term.
- serverActivity.reason.user: "由你操作" (by you) used with "已卸載(…)".

## ja

- serverActivity.reason.user: "by you" rendered as ユーザー操作 (neutral); check it reads well inside "アンロード済み (ユーザー操作)".
- serverActivity.summary.parked / kind.parked: "Kept warm" as ウォーム保持; "warm" transliteration follows existing ウォームプール.
- serverActivity.field.prefillSeconds: プリフィル (technical term, may want プロンプト処理).
- benchmarkTab.ofMaxTokens: "of {max} allowed" shortened to 上限 {max}; check label context.
- benchmarkTab.scoring.note: long technical sentence, verify wording.
- voice.models.sttGuide.parakeet-tdt-v3: "times each word on its own" rendered as 単語ごとのタイミングも自力で取得; 自力 may sound odd.
- voice.models.sttGuide.whisper-small: "one of the two that can translate" phrasing awkward.
- voice.models.wordTimingsLine: long, structure rearranged.
- voice.speak.needsSpeechGpu: menu path uses > with Japanese menu names; confirm names match existing ja UI (音声 / モデル / 音声ランタイム / アドオン).
- voice.studio.needsPackGb: "Voice" tab translated as 音声 per existing glossary.
- musicStudio.sounds.needLicenceStableAudio3: legal wording (利用規約 for Gemma terms).
- musicStudio.copy.wayDeveloper: Windows Settings path names (設定 > システム > 開発者向け) should be checked against real Japanese Windows UI.
- musicUnsupported.reason / soundReason: double negative "何もインストールもダウンロードもされません" reads clunky.
- board.editor.vocalHint: "mouth follows" / "keeps this sound" paraphrased.
- modelStoragePanel.lmstudioHint: refers to button "LM Studio と共有"; must match library models.chat.share.action.
- conversionTab.qualityChange: "{change}% perplexity vs the source" order changed.

## de

- serverActivity.field.prefill / prefillSeconds: "Prefill" left as English technical term; native review.
- serverActivity.field.promptTokens / completionTokens: "Prompt-Tokens"/"Completion-Tokens" kept as loanwords.
- serverActivity.paused: plural with adjective agreement "# neues/neue" (Paused, 1 new [event]) - gender guess.
- serverActivity.reason.user: "durch dich" in lowercase after "Unloaded (...)"; check reads naturally.
- serverActivity.summary.parked / kind.parked: "Warm gehalten" vs existing Warm-Pool term; native review.
- voice.models.sttGuide.parakeet-tdt-v3, whisper-large-v3-turbo: "zeitet jedes Wort" is awkward for "times each word"; better phrasing wanted (e.g. "liefert Zeitstempel pro Wort").
- voice.models.wordTimings / wordTimingsLine: "Wortzeiten" term and "Zeitet" verb; same concern.
- voice.runtime.aboutTickedGb: "Turn-Erkennung", "Sprecherkennzeichnung" for turn detection / speaker labels.
- board.editor.vocal*: "Gesang" used for "vocal"; "Standbild dieser Einstellung" for "this shot's still" (shot = Einstellung; existing de term for shot not verified).
- musicStudio.copy.why / wayDeveloper: Windows UI names (Entwicklermodus, Für Entwickler) assumed from German Windows.
- musicStudio.sounds.*: "Sounds" kept English; "Musik- und Audiomodelle" assumed page name, not verified against existing de label.
- voice.speak.needsSpeechGpu / studio.needsPackGb: menu path names (Voice > Modelle > Voice-Runtime > Add-ons) assumed to match existing de labels.
- modelStoragePanel.lmstudioHint: quoted "Mit LM Studio teilen" must match library.models.chat.share.action.
- benchmarkTab.evalNeedsMlx: "Durchsatz" must match existing de label for Throughput mode (not checked).
- voice.models.licenceName.CC-BY-4_0: parenthetical "(Urheber nennen)".

## ru

- serverActivity.summary.parked / kind.parked: "Держится тёплой" for "Kept warm" is awkward, check wording.
- serverActivity.reason.*: grammatical fit inside "Выгружено ({reason})" (вами / другой моделью / из-за простоя) needs review.
- serverActivity.field.prefillSeconds: "Префилл" kept as loanword.
- voice.speak.needsSpeechGpu, voice.runtime.addon.*: Russian names of add-ons ("Речь на GPU") and menu path labels must match actual Russian UI labels.
- voice.models.macOnlyNoteVoices: awkward "они и их голоса ({voices})".
- voice.models.sttGuide.parakeet-tdt-v3: "сама отмечает время каждого слова" - feminine agreement assumes "модель".
- board.editor.vocalHint: "шот" used for shot; check against existing ru storyboard term.
- benchmarkTab.evalNeedsMlx: "Пропускная способность" assumed to be the ru label of the Throughput mode.
- conversionTab.qualityChange: "{change}% перплексии относительно источника" - sign/direction wording.
- promptLibrary.empty: "Создать" assumed to match the New button label.
- musicStudio.copy.wayDeveloper: Windows settings path names are approximate Russian UI names.

## ko

- serverActivity.* (warm/parked): "웜" transliteration follows existing "웜 풀"; "Kept warm" phrasing may read oddly.
- serverActivity.field.prefillSeconds: "프리필" transliteration, native check.
- serverActivity.field.completionTokens: "완성 토큰" vs "출력 토큰".
- serverActivity.reason.user: "사용자 요청" (English "by you" is a short adverbial; fits "언로드됨 (사용자 요청)").
- serverActivity.paused: plural collapsed to `other` only; "새 항목 #개" wording.
- benchmarkTab.progress.phases.evaluatingAccuracy: "문제 푸는 중" casual register?
- benchmarkTab.scoring.note: long technical sentence, check clarity.
- promptLibrary.empty: refers to button "새로 만들기" - must match the existing ko label for "New".
- modelStoragePanel.lmstudioHint / models.chat.share.*: "LM Studio와 공유" must match settings label; "{where}(으)로" particle after placeholder.
- voice.models.sttGuide.*: marketing-style terse phrases; "스스로 감지" for "finds the language itself".
- voice.models.wordTimingsLine: long, ordering of clauses.
- voice.models.licenceName.CC-BY-4_0 / OpenMDW-1_1: parenthetical explanations only.
- voice.runtime.addon.*: "추가 기능" for add-on, check against existing ko term.
- board.editor.vocalHint: "스틸" for still image; "샷" term not used here.
- musicStudio.copy.why: "링크" for symlink/link, may need "심볼릭 링크".

## fr

- serverActivity.summary.loadedWarm / kind.parked / phase.warm: "copie chaude", "Gardé au chaud", "Chaud" follow existing "Pool chaud"; check naturalness.
- serverActivity.field.bytes: "Poids" (weights) may read ambiguous; "Taille des poids" is an alternative.
- serverActivity.field.prefillSeconds: "Préremplissage" technical term.
- studioChat.plan.expertFps: "i/s" (images par seconde) vs keeping "fps"; native review.
- voice.models.sttGuide.*: "détecte la langue lui-même" masculine agreement with "modèle" assumed.
- voice.models.sttFitMeasured: "Mesuré sur" masculine/neutral phrasing; check.
- voice.models.wordTimings / wordTimingsLine: "Horodatage des mots" vs existing glossary for "word timings".
- voice.runtime.aboutTickedGb: "détection de tour de parole" (turn detection) is a guess.
- board.editor.vocal*: "synchro labiale", "plan" for shot (existing fr glossary for "shot" not verified).
- benchmarkTab.evalNeedsMlx: "Débit" for Throughput mode assumed; check against existing mode label.
- benchmarkTab.scoring.nextToken: terse label.
- benchmarkHistory.metric.scoring / benchmarkTab.scoring.*: "Évaluation"/scoring.
- promptLibrary.*: "modèle" for template collides with model; "modèle de prompt" used once; check existing glossary.
- promptLibrary.samplersInvalid: "échantillonneurs prédéfinis" for preset samplers.
- musicStudio.copy.*: "lien" for symlink; Windows setting names ("Pour les développeurs") must match the French Windows UI.
- modelStoragePanel.lmstudioHint: Share with LM Studio rendered "Partager avec LM Studio" (must match library.models.chat.share.action).
- voice.models.licenceName.*: parenthetical translations only.

## es

- serverActivity.kind.parked / summary.parked: "en caliente" for warm pool; matches existing "Grupo en caliente" but may read oddly.
- serverActivity.field.prefill / Streaming / prompt: left as English technical terms (Prefill, Streaming, prompt).
- serverActivity.reason.user: "por ti" informal; check register in the activity log.
- voice.models.licenceName.CC-BY-4_0: "da crédito al autor" imperative inside a label.
- voice.models.sttGuide.*: "detecta el idioma por sí mismo" vs a more neutral "detecta el idioma automáticamente".
- benchmarkTab.modal.cancelUnconfirmed: "La app" (informal) wording.
- promptLibrary.samplersInvalid: "samplers" kept in English, "preconfiguración" for preset.
- conversionTab.qualityChange: "{change}% de perplejidad" - Spanish usually writes "{change} %"; kept existing style without space.
- musicStudio.copy.wayDeveloper: Windows UI names "Para programadores" assumed from Spanish Windows.
- board.editor.vocal: "Voz para sincronía labial" and "fotograma fijo" for "still".
- voice.speak.queue.*: "vídeo" with accent; check whether existing es files use "video".

## pt-BR

- serverActivity.summary.loadedWarm / parked / phase.warm / field.warm: "quente" / "pool quente" follows the existing "Pool quente" glossary; check it reads naturally.
- serverActivity.field.prefill / streaming / backend: kept as English technical terms (Prefill, Streaming, Backend).
- promptLibrary.*: "Template" rendered as "modelo de prompt" to avoid clashing with "modelo" (AI model); check against how existing pt-BR strings name templates.
- promptLibrary.variablesInvalid: "matriz JSON" for JSON array; "lista" may be clearer to users.
- promptLibrary.samplersInvalid: "samplers da predefinição" (kept sampler as is).
- benchmarkTab.scoring.note: "acurácia" vs "precisão" - consistency with existing pt-BR benchmark strings not verified.
- board.editor.vocalHint: "quadro fixo deste plano" for "this shot's still"; existing pt-BR term for "shot" may be "plano" or "cena" - verify.
- voice.models.firstUseLanguageLine: "o detectam quando está em detectar" is clumsy; depends on the pt-BR label of the Language dropdown option.
- voice.models.sttFit: "Cabe" for "Fits".
- voice.runtime.aboutTickedGb: "detecção de turnos e rótulos de falantes" (turn detection and speaker labels).
- musicStudio.copy.wayDeveloper: Windows menu names translated from English from memory (Configurações, Sistema, Para desenvolvedores).
- modelStoragePanel.lmstudioHint: "vincula" for "links"; "unidade" for drive.
