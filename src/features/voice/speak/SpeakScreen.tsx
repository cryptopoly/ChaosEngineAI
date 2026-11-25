import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Drawer } from "../../../components/Drawer";
import { Panel } from "../../../components/Panel";
import { cancelSpeak, saveSpeech, speakVoice } from "../../../api";
import type { TabId, VoiceEntry } from "../../../types";
import { useVoice } from "../VoiceContext";
import { joinWavs, splitSpeech, type SpeechPart } from "../studio/speech";
import { requestLibraryFilter } from "../libraryVoice";
import { unsupportedModelText } from "../models/voiceModelsLogic";
import { describeVoiceError, type VoiceErrorView } from "../voiceErrorText";
import { SpeakResultRow, type SpeakOutput } from "./SpeakResultRow";
import { VoicePicker } from "./VoicePicker";
import { VoiceCanLine, VoiceRow } from "./VoiceRow";
import {
  MAX_REQUEST_CHARS,
  MAX_SPEAK_CHARS,
  STYLE_CHIPS,
  addRecent,
  canGenerate,
  cloneRequestText,
  costLabel,
  cloneTextTooLong,
  defaultLanguage,
  downloadSizeGb,
  errorTarget,
  ffmpegLimits,
  insertTag,
  installedCloneModels,
  installedDesignModels,
  modelNote,
  recommendedNaturalModel,
  sampleTooShort,
  showInstallCard,
  styleGap,
  styleState,
  voiceAbilities,
  voiceControls,
} from "./speakLogic";
import { GuidanceSlider, LanguageAndExpression, SpeakError, StyleGapNote, TagChips } from "./SpeakControls";
import { InstallVoiceCard } from "./InstallVoiceCard";
import { useCostText } from "./costText";
import type { VoiceCatalog } from "./useVoiceCatalog";
import { useVoiceSample } from "./useVoiceSample";
import { speakQueueLineKey, useSpeakQueue } from "./speakQueue";

function base64ToBytes(data: string): Uint8Array<ArrayBuffer> {
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

const CHIP_LABELS: Record<string, string> = {
  neutral: "Neutral",
  angry: "Angry",
  sad: "Sad",
  excited: "Excited",
  whisper: "Whisper",
  calm: "Calm",
};

/** Speak: pick a voice, say how it should sound, and make the audio. One screen for every voice. */
export function SpeakScreen({
  catalog,
  voiceId,
  onVoiceChange,
  onTabChange,
  onNewClone,
  textRequest,
}: {
  catalog: VoiceCatalog;
  voiceId: string;
  onVoiceChange: (id: string) => void;
  onTabChange?: (tab: TabId) => void;
  /** Open the new cloned voice form (the Voices tab). */
  onNewClone?: () => void;
  /** Text to put in the box (the Library's Speak again); a new nonce replaces the text again. */
  textRequest?: { text: string; nonce: number } | null;
}) {
  const { t, i18n } = useTranslation("studio");
  const { settings, runtime } = useVoice();
  const textRef = useRef<HTMLTextAreaElement | null>(null);
  const [designModelKey, setDesignModelKey] = useState("");
  const [language, setLanguage] = useState("");
  const [expression, setExpression] = useState<number | null>(null);
  const [guidance, setGuidance] = useState<number | null>(null);
  const [text, setText] = useState(textRequest?.text ?? "");
  // Style text is kept per voice, so a preset's "angry" never becomes the description of a designed voice.
  const [styles, setStyles] = useState<Record<string, string>>({});
  const [cloneModelKey, setCloneModelKey] = useState("");
  const [speed, setSpeed] = useState(settings.speed);
  const [pitch, setPitch] = useState(0);
  const [seed, setSeed] = useState("");
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const queueLine = speakQueueLineKey(useSpeakQueue(progress !== null));
  const [error, setError] = useState<VoiceErrorView | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  // The session's last five clips stay under Generate, newest first.
  const [clips, setClips] = useState<SpeakOutput[]>([]);
  const clipsRef = useRef<SpeakOutput[]>([]);
  // The picker is a drawer behind "Change voice", so the form and Generate stay above the fold (D-73 S1).
  const [picking, setPicking] = useState(false);
  const cancelRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const sample = useVoiceSample(speed);
  const costs = useCostText();

  useEffect(() => () => clipsRef.current.forEach((clip) => URL.revokeObjectURL(clip.url)), []);

  useEffect(() => {
    if (textRequest) setText(textRequest.text);
    // Only a new request replaces the text, not every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [textRequest?.nonce]);

  const voice: VoiceEntry | null = catalog.voices.find((v) => v.id === voiceId) ?? null;
  const style = styles[voiceId] ?? "";
  const setStyle = (value: string) => setStyles((prev) => ({ ...prev, [voiceId]: value }));
  const cloned = voice?.kind === "clone";
  const designing = voice?.kind === "design";
  const note = modelNote(voice, catalog.models, designing ? designModelKey : cloneModelKey);
  const styleInfo = styleState(voice, catalog.models, note.model);
  const activeStyle = styleInfo.enabled ? style.trim() : "";
  const gap = styleGap(styleInfo, catalog.models, catalog.modelsAvailable);
  // The "Style needs a natural voice" line carries its own download, so the card would only repeat it.
  const installModel =
    showInstallCard(catalog.models, catalog.modelsAvailable) && gap?.kind !== "needsNatural" ? recommendedNaturalModel(catalog.models) : null;
  const cloneChoices = installedCloneModels(voice, catalog.models);
  const designChoices = installedDesignModels(voice, catalog.models);
  const controls = voiceControls(note.model);
  const chosenLanguage = controls.languages.includes(language) ? language : defaultLanguage(controls.languages, i18n.language ?? "en");
  const chosenExpression = expression ?? controls.expression?.default ?? 0.5;
  const chosenGuidance = guidance ?? controls.guidance?.default ?? 0.5;
  const shortSample = sampleTooShort(voice, note.model);
  const steered = cloned && note.model?.family === "voxcpm2";
  // Only the models that take them get the new fields, so every other request body stays as it was.
  const modelFields = {
    ...(controls.languages.length > 0 ? { language: chosenLanguage } : {}),
    ...(controls.expression ? { expression: chosenExpression } : {}),
    ...(controls.guidance ? { guidance: chosenGuidance } : {}),
  };
  const addTag = (tag: string) => {
    const box = textRef.current;
    const start = box?.selectionStart ?? text.length;
    const end = box?.selectionEnd ?? text.length;
    const next = insertTag(text, start, end, tag);
    setText(next.text);
    requestAnimationFrame(() => {
      box?.focus();
      box?.setSelectionRange(next.cursor, next.cursor);
    });
  };
  const parts: SpeechPart[] = cloned ? (text.trim() ? [{ text: cloneRequestText(text), gapAfter: 0 }] : []) : splitSpeech(text);
  const tooLong = cloned && cloneTextTooLong(text);
  // Without ffmpeg the pitch slider (and speed, on voices that need it) is disabled up front, and the request ignores it.
  const limits = ffmpegLimits(voice);
  const requestSpeed = limits.speed ? speed : 1;
  const requestPitch = limits.pitch ? pitch : 0;
  const abilities = voiceAbilities(voice, controls, styleInfo.enabled);
  const ready = canGenerate(voice, text, activeStyle, note) && !tooLong && !shortSample;
  const seedValue = seed.trim() === "" ? null : Math.max(0, Math.min(2 ** 31 - 1, Math.floor(Number(seed))));

  const generate = async () => {
    if (!voice) return;
    setError(null);
    setNotes([]);
    cancelRef.current = false;
    const controller = new AbortController();
    abortRef.current = controller;
    setProgress({ done: 0, total: parts.length });
    try {
      const wavs: ArrayBuffer[] = [];
      const seen: string[] = [];
      let last = null as Awaited<ReturnType<typeof speakVoice>> | null;
      for (const [index, part] of parts.entries()) {
        if (cancelRef.current) return;
        last = await speakVoice(
          {
            text: part.text,
            voice: voice.id,
            style: activeStyle,
            speed: requestSpeed,
            pitch: requestPitch,
            seed: Number.isFinite(seedValue) ? seedValue : null,
            // A cloned or designed voice runs with the model shown on the model line (the backend would otherwise take
            // its default).
            ...((cloned || designing) && note.model ? { model: note.model.key } : {}),
            ...modelFields,
          },
          controller.signal,
        );
        wavs.push(base64ToBytes(last.audio).buffer);
        for (const line of last.notes) if (!seen.includes(line)) seen.push(line);
        setProgress({ done: index + 1, total: parts.length });
      }
      if (cancelRef.current || !last) return;
      const joined =
        wavs.length === 1
          ? { blob: new Blob([wavs[0]], { type: "audio/wav" }), durationS: last.durationS }
          : joinWavs(wavs, parts.map((p) => p.gapAfter));
      // Every result goes to the Library (cloned speech is filed by the backend already). A failed save keeps the
      // clip here with Save to Library offered again.
      const voiceName = voice.kind === "design" ? activeStyle.slice(0, 120) : voice.name;
      let libraryId = last.libraryId;
      if (!libraryId) {
        try {
          libraryId = (await saveSpeech(joined.blob, { text, voice: voice.id, speed: requestSpeed, voiceName, voiceKind: voice.kind })).id;
        } catch (err) {
          seen.push(
            t("voice.speak.autosaveFailed", {
              defaultValue: "The clip could not be saved to the Library ({error}). Use Save to Library to try again.",
              error: err instanceof Error ? err.message : String(err),
            }),
          );
        }
      }
      setNotes(seen);
      const kept = addRecent(clipsRef.current, {
        url: URL.createObjectURL(joined.blob),
        blob: joined.blob,
        durationS: joined.durationS,
        text,
        voice: voice.id,
        voiceName,
        voiceKind: voice.kind,
        speed: requestSpeed,
        cloned: last.clonedVoice,
        libraryId,
      });
      kept.dropped.forEach((clip) => URL.revokeObjectURL(clip.url));
      clipsRef.current = kept.list;
      setClips(kept.list);
    } catch (err) {
      if (!cancelRef.current) setError(describeVoiceError(err));
    } finally {
      abortRef.current = null;
      setProgress(null);
    }
  };

  const cancel = () => {
    cancelRef.current = true;
    abortRef.current?.abort();
    void cancelSpeak(); // the abort alone leaves the backend job running
  };

  /** The button on a coded error does what its action says; null when nothing here can do it. */
  const errorHandler = (view: VoiceErrorView): (() => void) | null => {
    switch (errorTarget(view.action)) {
      case "retry":
        return () => void generate();
      case "models":
        return onTabChange ? () => onTabChange("voice-models") : null;
      case "settings":
        return onTabChange ? () => onTabChange("settings") : null;
      case "download":
        if (note.model && note.model.supported && !note.model.installed && !note.downloading) {
          const key = note.model.key;
          return () => void catalog.download(key);
        }
        return onTabChange ? () => onTabChange("voice-models") : null;
      case "picker":
        return () => setPicking(true);
      case "clone":
        return onNewClone ?? (() => setPicking(true));
      default:
        return null;
    }
  };

  return (
    <Panel
      title={t("voice.speak.title", { defaultValue: "Speak" })}
      subtitle={t("voice.speak.subtitleLong", { defaultValue: "Turn text into speech. Blank lines start a new paragraph; [pause] adds a pause." })}
      className="span-2"
    >
      <div className="voice-studio voice-speak">
        <textarea
          ref={textRef}
          className="text-input voice-textarea voice-textarea--long"
          value={text}
          maxLength={MAX_SPEAK_CHARS}
          aria-label={t("voice.speak.textLabel", { defaultValue: "Text to speak" })}
          onChange={(e) => setText(e.target.value)}
          placeholder={t("voice.speak.placeholder", { defaultValue: "Type or paste text. Markdown and code are tidied up before speaking." })}
        />
        {cloned ? (
          <p className="muted-text voice-speak__hint">
            {t("voice.speak.cloneLimit", {
              defaultValue: "A cloned voice speaks up to {max} characters at a time, as one clip saved to your Library.",
              max: MAX_REQUEST_CHARS.toLocaleString(),
            })}
          </p>
        ) : null}
        <VoiceRow
          voice={voice}
          model={note.model}
          playing={voice ? sample.isPlaying(voice.id) : false}
          loading={voice ? sample.isLoading(voice.id) : false}
          onSample={() => {
            if (voice) void sample.play(voice);
          }}
          onChange={() => setPicking(true)}
        />
        <VoiceCanLine abilities={abilities} />
        {installModel ? <InstallVoiceCard model={installModel} onDownload={() => void catalog.download(installModel.key)} /> : null}

        {designing && designChoices.length >= 2 ? (
          <label className="voice-settings__field voice-speak__clone-model">
            <span>{t("voice.speak.designModel", { defaultValue: "Design model" })}</span>
            <select className="text-input" value={note.model?.key ?? ""} onChange={(e) => setDesignModelKey(e.target.value)}>
              {designChoices.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        {cloned && cloneChoices.length >= 2 ? (
          <label className="voice-settings__field voice-speak__clone-model">
            <span>{t("voice.speak.cloneModel", { defaultValue: "Cloning model" })}</span>
            <select className="text-input" value={note.model?.key ?? ""} onChange={(e) => setCloneModelKey(e.target.value)}>
              {cloneChoices.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        <TagChips controls={controls} onInsert={addTag} />

        <div className="voice-speak__style">
          {styleInfo.enabled ? (
            <>
              <label className="voice-settings__field">
                <span>{designing ? t("voice.speak.describe", { defaultValue: "Describe the voice" }) : t("voice.speak.style", { defaultValue: "Style" })}</span>
                <textarea
                  className="text-input voice-textarea voice-speak__style-box"
                  value={style}
                  maxLength={500}
                  onChange={(e) => setStyle(e.target.value)}
                  placeholder={
                    designing
                      ? t("voice.speak.describePlaceholder", { defaultValue: "A warm woman in her thirties, calm and clear, speaking at an easy pace." })
                      : t("voice.speak.stylePlaceholderFree", { defaultValue: "Tone, pace, accent, in your own words. For example: a tired, gentle bedtime voice, slow, slight Irish accent." })
                  }
                />
              </label>
              {!designing ? (
                <div className="voice-speak__chips" role="group" aria-label={t("voice.speak.styleChips", { defaultValue: "Style shortcuts" })}>
                  {STYLE_CHIPS.map((chip) => (
                    <button
                      key={chip.id}
                      type="button"
                      className={`voice-speak__chip${style === chip.instruction && chip.instruction !== "" ? " voice-speak__chip--on" : ""}`}
                      onClick={() => setStyle(chip.instruction)}
                    >
                      {t(`voice.speak.chips.${chip.id}`, { defaultValue: CHIP_LABELS[chip.id] })}
                    </button>
                  ))}
                </div>
              ) : null}
            </>
          ) : gap ? (
            <StyleGapNote
              gap={gap}
              onDownload={(key) => void catalog.download(key)}
              onChoose={() => setPicking(true)}
              sizeGb={gap.model ? downloadSizeGb(gap.model) : 0}
            />
          ) : null}
          {designing ? (
            <p className="muted-text voice-speak__hint">
              {t("voice.speak.describeHint", {
                defaultValue: "Say the age, gender, pitch, tone, accent and pace. The voice changes on every run; set a seed under Advanced to keep it.",
              })}
            </p>
          ) : null}
          {steered ? (
            <p className="muted-text voice-speak__hint">
              {t("voice.speak.steeredNote", {
                defaultValue: "VoxCPM2 can follow a style for a cloned voice; with a style it keeps a little less of the likeness. Leave it empty for the closest match.",
              })}
            </p>
          ) : null}
        </div>

        <LanguageAndExpression
          controls={controls}
          language={chosenLanguage}
          onLanguage={setLanguage}
          expression={chosenExpression}
          onExpression={setExpression}
        />

        <div className="voice-studio__row">
          <label className="voice-settings__field">
            <span>
              {t("voice.speak.speed", { defaultValue: "Speed" })} ({requestSpeed.toFixed(1)}×)
            </span>
            <input
              type="range"
              min={0.5}
              max={2}
              step={0.1}
              value={requestSpeed}
              disabled={!limits.speed}
              aria-describedby={limits.speed ? undefined : "voice-speak-speed-reason"}
              onChange={(e) => setSpeed(parseFloat(e.target.value))}
            />
          </label>
        </div>
        {limits.speed ? null : (
          <p id="voice-speak-speed-reason" className="muted-text voice-speak__hint">
            {t("voice.speak.speedNeedsFfmpeg", { defaultValue: "Speed for this voice needs ffmpeg, which is not installed. Install it from Settings." })}
          </p>
        )}

        <details className="voice-speak__advanced">
          <summary>{t("voice.speak.advanced", { defaultValue: "Advanced" })}</summary>
          <label className="voice-settings__field">
            <span>
              {t("voice.speak.pitch", { defaultValue: "Pitch" })} ({requestPitch > 0 ? "+" : ""}
              {requestPitch})
            </span>
            <input
              type="range"
              min={-12}
              max={12}
              step={1}
              value={requestPitch}
              disabled={!limits.pitch}
              aria-describedby={limits.pitch ? undefined : "voice-speak-pitch-reason"}
              onChange={(e) => setPitch(parseInt(e.target.value, 10))}
            />
          </label>
          {limits.pitch ? null : (
            <p id="voice-speak-pitch-reason" className="muted-text voice-speak__hint">
              {t("voice.speak.pitchNeedsFfmpeg", { defaultValue: "Pitch needs ffmpeg, which is not installed. Install it from Settings." })}
            </p>
          )}
          <label className="voice-settings__field">
            <span>{t("voice.speak.seed", { defaultValue: "Seed (leave empty for a different take each time)" })}</span>
            <input
              type="number"
              className="text-input"
              min={0}
              max={2147483647}
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
            />
          </label>
          <GuidanceSlider range={controls.guidance} value={chosenGuidance} onChange={setGuidance} />
        </details>

        <p className="muted-text voice-speak__model" aria-live="polite">
          {voice == null
            ? ""
            : note.model == null && voice.kind === "builtin"
              ? t("voice.speak.modelKokoroCost", { defaultValue: "Runs with Kokoro, {detail}.", detail: costs.detail(costLabel(null)) })
              : note.model
                ? note.model.installed
                  ? t("voice.speak.modelReadyCost", { defaultValue: "Runs with {name}: {detail}.", name: note.model.name, detail: costs.detail(costLabel(note.model)) })
                  : note.downloading
                    ? note.model.download.phase === "watermark"
                      ? t("voice.clone.installingWatermark", { defaultValue: "Installing the watermark component…" })
                      : t("voice.speak.modelDownloading", { defaultValue: "Downloading {name}…", name: note.model.name })
                    : !note.model.supported
                      ? `${note.model.name}: ${unsupportedModelText(note.model, runtime)}`
                      : `${t("voice.speak.modelNeeds", { defaultValue: "Needs {name} ({gb} GB), not downloaded yet.", name: note.model.name, gb: note.model.sizeGb })} ${t("voice.speak.modelNeedsCost", { defaultValue: "Once downloaded: {detail}.", detail: costs.detail(costLabel(note.model)) })}`
                : (voice.needs ?? "")}{" "}
          {note.model && note.model.supported && !note.model.installed && !note.downloading ? (
            <button type="button" className="secondary-button" onClick={() => void catalog.download(note.model!.key)}>
              {t("voice.models.download", { defaultValue: "Download" })}
            </button>
          ) : null}
          {voice && onTabChange ? (
            <>
              {" "}
              <button type="button" className="link-button" onClick={() => onTabChange("voice-models")}>
                {t("voice.speak.manageInModels", { defaultValue: "Manage in Models" })}
              </button>
            </>
          ) : null}
        </p>

        {shortSample && note.model ? (
          <p className="voice-error" role="alert">
            {t("voice.speak.sampleTooShort", {
              defaultValue: "{name} needs a voice sample of at least {min} seconds; this one is {secs}. Pick another cloning model.",
              name: note.model.name,
              min: note.model.minReferenceS ?? 0,
              secs: (voice?.durationS ?? 0).toFixed(1),
            })}
          </p>
        ) : null}
        {tooLong ? (
          <p className="voice-error" role="alert">
            {t("voice.speak.cloneTooLong", { defaultValue: "This is too long for one cloned-voice clip. Shorten it to {max} characters.", max: MAX_REQUEST_CHARS.toLocaleString() })}
          </p>
        ) : null}
        {designing && !style.trim() ? (
          <p className="muted-text voice-speak__hint">{t("voice.speak.describeFirst", { defaultValue: "Describe the voice you want before generating." })}</p>
        ) : null}
        {error ? <SpeakError error={error} onAction={errorHandler(error)} /> : null}
        {notes.map((line) => (
          <p key={line} className="muted-text voice-speak__hint">
            {line}
          </p>
        ))}
        <div className="button-row">
          {progress ? (
            <>
              <span className="voice-studio__progress" aria-live="polite">
                {t("voice.speak.progress", { defaultValue: "Part {done} of {total}", done: Math.min(progress.done + 1, progress.total), total: progress.total })}
              </span>
              <progress max={progress.total} value={progress.done} />
              <button type="button" className="secondary-button" onClick={cancel}>
                {t("voice.cancel", { defaultValue: "Cancel" })}
              </button>
              {queueLine ? (
                <span className="muted-text" aria-live="polite">
                  {t(`voice.speak.queue.${queueLine}`, { defaultValue: "Waiting for another job to finish. Speech starts right after it." })}
                </span>
              ) : null}
            </>
          ) : (
            <button type="button" className="primary-button" disabled={!ready || parts.length === 0} onClick={() => void generate()}>
              {parts.length > 1
                ? t("voice.speak.generateParts", { defaultValue: "Generate audio ({n} parts)", n: parts.length })
                : t("voice.speak.generate", { defaultValue: "Generate audio" })}
            </button>
          )}
        </div>
        {clips.length > 0 ? (
          <section className="voice-speak__recent" aria-label={t("voice.speak.recent", { defaultValue: "Recent clips" })}>
            <p className="muted-text voice-speak__hint">
              <strong>{t("voice.speak.recent", { defaultValue: "Recent clips" })}</strong>
            </p>
            {clips.map((clip) => (
              <SpeakResultRow
                key={clip.url}
                output={clip}
                onTabChange={onTabChange}
                onError={(err) => setError(describeVoiceError(err))}
                onOpenLibrary={
                  onTabChange
                    ? () => {
                        requestLibraryFilter("speech");
                        onTabChange("voice-gallery");
                      }
                    : undefined
                }
              />
            ))}
          </section>
        ) : null}
      </div>
      {picking ? (
        <Drawer
          title={t("voice.picker.drawerTitle", { defaultValue: "Choose a voice" })}
          subtitle={t("voice.picker.drawerHint", { defaultValue: "Click a voice to use it in Speak. The sample button plays a short line." })}
          onClose={() => setPicking(false)}
          className="voice-drawer"
        >
          <VoicePicker
            voices={catalog.voices}
            models={catalog.models}
            selectedId={voiceId}
            onSelect={onVoiceChange}
            onChoose={() => setPicking(false)}
            modelsAvailable={catalog.modelsAvailable}
            isPlaying={sample.isPlaying}
            isLoading={sample.isLoading}
            onSample={(entry) => void sample.play(entry)}
            onNewClone={
              catalog.cloneAllowed && onNewClone
                ? () => {
                    setPicking(false);
                    onNewClone();
                  }
                : undefined
            }
            onGetMore={onTabChange ? () => onTabChange("voice-models") : undefined}
            onAddonInstalled={() => void catalog.refresh()}
          />
        </Drawer>
      ) : null}
    </Panel>
  );
}
