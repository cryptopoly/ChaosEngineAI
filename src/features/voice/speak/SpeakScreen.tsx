import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../../components/Panel";
import { cancelSpeak, speakVoice } from "../../../api";
import type { TabId, VoiceEntry } from "../../../types";
import { useVoice } from "../VoiceContext";
import { joinWavs, splitSpeech, type SpeechPart } from "../studio/speech";
import { CloneStatement } from "../clone/CloneStatement";
import { SpeakResultRow, type SpeakOutput } from "./SpeakResultRow";
import { VoicePicker } from "./VoicePicker";
import {
  MAX_REQUEST_CHARS,
  MAX_SPEAK_CHARS,
  STYLE_CHIPS,
  canGenerate,
  cloneRequestText,
  costLabel,
  cloneTextTooLong,
  installedCloneModels,
  modelNote,
  recommendedNaturalModel,
  showInstallCard,
  styleState,
} from "./speakLogic";
import { InstallVoiceCard } from "./InstallVoiceCard";
import { useCostText } from "./costText";
import type { VoiceCatalog } from "./useVoiceCatalog";
import { useVoiceSample } from "./useVoiceSample";

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
}: {
  catalog: VoiceCatalog;
  voiceId: string;
  onVoiceChange: (id: string) => void;
  onTabChange?: (tab: TabId) => void;
}) {
  const { t } = useTranslation("studio");
  const { settings } = useVoice();
  const [text, setText] = useState("");
  // Style text is kept per voice, so a preset's "angry" never becomes the description of a designed voice.
  const [styles, setStyles] = useState<Record<string, string>>({});
  const [cloneModelKey, setCloneModelKey] = useState("");
  const [speed, setSpeed] = useState(settings.speed);
  const [pitch, setPitch] = useState(0);
  const [seed, setSeed] = useState("");
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [output, setOutput] = useState<SpeakOutput | null>(null);
  const cancelRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const sample = useVoiceSample(speed);
  const costs = useCostText();

  useEffect(
    () => () => {
      if (output) URL.revokeObjectURL(output.url);
    },
    [output],
  );

  const voice: VoiceEntry | null = catalog.voices.find((v) => v.id === voiceId) ?? null;
  const style = styles[voiceId] ?? "";
  const setStyle = (value: string) => setStyles((prev) => ({ ...prev, [voiceId]: value }));
  const cloned = voice?.kind === "clone";
  const designing = voice?.kind === "design";
  const styleInfo = styleState(voice, catalog.models);
  const activeStyle = styleInfo.enabled ? style.trim() : "";
  const note = modelNote(voice, catalog.models, cloneModelKey);
  const installModel = showInstallCard(catalog.models, catalog.modelsAvailable) ? recommendedNaturalModel(catalog.models) : null;
  const cloneChoices = installedCloneModels(voice, catalog.models);
  const parts: SpeechPart[] = cloned ? (text.trim() ? [{ text: cloneRequestText(text), gapAfter: 0 }] : []) : splitSpeech(text);
  const tooLong = cloned && cloneTextTooLong(text);
  const ready = canGenerate(voice, text, activeStyle, note) && !tooLong;
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
            speed,
            pitch,
            seed: Number.isFinite(seedValue) ? seedValue : null,
            // A cloned voice runs with the cloning model shown on the model line (the backend would otherwise take its default).
            ...(cloned && note.model ? { model: note.model.key } : {}),
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
      setNotes(seen);
      setOutput({
        url: URL.createObjectURL(joined.blob),
        blob: joined.blob,
        durationS: joined.durationS,
        text,
        voice: voice.id,
        speed,
        cloned: last.clonedVoice,
        libraryId: last.libraryId,
      });
    } catch (err) {
      if (!cancelRef.current) setError(err instanceof Error ? err.message : String(err));
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

  const styleReason =
    styleInfo.reason === "builtin"
      ? t("voice.speak.styleBuiltin", { defaultValue: "Quick voices have no style control. Pick a natural voice to use it." })
      : styleInfo.reason === "clone"
        ? t("voice.speak.styleClone", { defaultValue: "Cloned voices have no style control." })
        : styleInfo.reason === "smallModel"
          ? t("voice.speak.styleSmall", { defaultValue: "The small model ignores style. The 1.7B model reads it." })
          : "";

  return (
    <Panel
      title={t("voice.speak.title", { defaultValue: "Speak" })}
      subtitle={t("voice.speak.subtitleLong", { defaultValue: "Turn text into speech. Blank lines start a new paragraph; [pause] adds a pause." })}
      className="span-2"
    >
      <div className="voice-studio voice-speak">
        <textarea
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

        <VoicePicker
          voices={catalog.voices}
          models={catalog.models}
          lead={installModel ? <InstallVoiceCard model={installModel} onDownload={() => void catalog.download(installModel.key)} /> : null}
          selectedId={voiceId}
          onSelect={onVoiceChange}
          modelsAvailable={catalog.modelsAvailable}
          isPlaying={sample.isPlaying}
          isLoading={sample.isLoading}
          onSample={(entry) => void sample.play(entry)}
        />
        {cloned ? <CloneStatement /> : null}

        <div className="voice-speak__style">
          <label className="voice-settings__field">
            <span>{designing ? t("voice.speak.describe", { defaultValue: "Describe the voice" }) : t("voice.speak.style", { defaultValue: "Style" })}</span>
            <textarea
              className="text-input voice-textarea voice-speak__style-box"
              value={style}
              maxLength={500}
              disabled={!styleInfo.enabled}
              aria-describedby={styleReason ? "voice-speak-style-reason" : undefined}
              onChange={(e) => setStyle(e.target.value)}
              placeholder={
                designing
                  ? t("voice.speak.describePlaceholder", { defaultValue: "A warm woman in her thirties, calm and clear, speaking at an easy pace." })
                  : t("voice.speak.stylePlaceholder", { defaultValue: "How should it sound? For example: a tired, gentle bedtime voice." })
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
                  disabled={!styleInfo.enabled}
                  onClick={() => setStyle(chip.instruction)}
                >
                  {t(`voice.speak.chips.${chip.id}`, { defaultValue: CHIP_LABELS[chip.id] })}
                </button>
              ))}
            </div>
          ) : null}
          {styleReason ? (
            <p id="voice-speak-style-reason" className="muted-text voice-speak__hint">
              {styleReason}
            </p>
          ) : null}
        </div>

        <div className="voice-studio__row">
          <label className="voice-settings__field">
            <span>
              {t("voice.speak.speed", { defaultValue: "Speed" })} ({speed.toFixed(1)}×)
            </span>
            <input type="range" min={0.5} max={2} step={0.1} value={speed} onChange={(e) => setSpeed(parseFloat(e.target.value))} />
          </label>
          <label className="voice-settings__field">
            <span>
              {t("voice.speak.pitch", { defaultValue: "Pitch" })} ({pitch > 0 ? "+" : ""}
              {pitch})
            </span>
            <input type="range" min={-12} max={12} step={1} value={pitch} onChange={(e) => setPitch(parseInt(e.target.value, 10))} />
          </label>
        </div>

        <details className="voice-speak__advanced">
          <summary>{t("voice.speak.advanced", { defaultValue: "Advanced" })}</summary>
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
        </details>

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
                    : `${t("voice.speak.modelNeeds", { defaultValue: "Needs {name} ({gb} GB), not downloaded yet.", name: note.model.name, gb: note.model.sizeGb })} ${t("voice.speak.modelNeedsCost", { defaultValue: "Once downloaded: {detail}.", detail: costs.detail(costLabel(note.model)) })}`
                : (voice.needs ?? "")}{" "}
          {note.model && !note.model.installed && !note.downloading ? (
            <button type="button" className="secondary-button" onClick={() => void catalog.download(note.model!.key)}>
              {t("voice.models.download", { defaultValue: "Download" })}
            </button>
          ) : null}
          {styleInfo.unlockWith && !styleInfo.unlockWith.installed && styleInfo.unlockWith.download.state !== "downloading" ? (
            <button type="button" className="secondary-button" onClick={() => void catalog.download(styleInfo.unlockWith!.key)}>
              {t("voice.speak.downloadForStyle", { defaultValue: "Download {name} for style", name: styleInfo.unlockWith.name })}
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

        {tooLong ? (
          <p className="voice-error" role="alert">
            {t("voice.speak.cloneTooLong", { defaultValue: "This is too long for one cloned-voice clip. Shorten it to {max} characters.", max: MAX_REQUEST_CHARS.toLocaleString() })}
          </p>
        ) : null}
        {designing && !style.trim() ? (
          <p className="muted-text voice-speak__hint">{t("voice.speak.describeFirst", { defaultValue: "Describe the voice you want before generating." })}</p>
        ) : null}
        {error ? (
          <p className="voice-error" role="alert">
            {error}
          </p>
        ) : null}
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
            </>
          ) : (
            <button type="button" className="primary-button" disabled={!ready || parts.length === 0} onClick={() => void generate()}>
              {parts.length > 1
                ? t("voice.speak.generateParts", { defaultValue: "Generate audio ({n} parts)", n: parts.length })
                : t("voice.speak.generate", { defaultValue: "Generate audio" })}
            </button>
          )}
        </div>
        {output ? <SpeakResultRow key={output.url} output={output} onTabChange={onTabChange} onError={setError} /> : null}
      </div>
    </Panel>
  );
}
