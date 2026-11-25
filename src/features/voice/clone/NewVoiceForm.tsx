import { useEffect, useRef, useState } from "react";
import { CloneStatement } from "./CloneStatement";
import { useTranslation } from "react-i18next";
import { MicSettingsButton } from "../MicSettingsButton";
import { formatClock, micErrorMessage, startMicCapture, type MicCapture } from "../audio";
import { useVoice } from "../VoiceContext";
import { describeVoiceError, type VoiceErrorView } from "../voiceErrorText";
import { CloneModelPicker } from "./CloneModelPicker";
import { createClonedVoice, downloadCloneModel, getCloneModels, type CloneModel, type ClonedVoice } from "./cloneApi";
import {
  SAMPLE_EXTENSIONS,
  canSaveVoice,
  cloneErrorFix,
  isSampleFile,
  pickCloneModel,
  sampleLengthProblem,
  wavSeconds,
  type CloneBasis,
  type NewVoiceDetails,
} from "./cloneLogic";
import { VoiceErrorNotice } from "./VoiceErrorNotice";

type StepName = "permission" | "sample" | "name" | "model";

/**
 * Making a cloned voice, in four numbered steps: 1 Permission (whose voice it is and the one tick), 2 Sample (a file
 * from this computer or a microphone recording), 3 Name, 4 Model. The backend checks the sample's type and length
 * and stores the speaker's name, the basis and the time with the voice. A failure shows one sentence and at most one
 * button (``describeVoiceError``), with the raw text under Details.
 */
export function NewVoiceForm({
  onCreated,
  onCancel,
  onOpenSetup,
}: {
  onCreated: (voice: ClonedVoice) => void;
  /** Shows a Cancel button; leave out when the form is a screen of its own. */
  onCancel?: () => void;
  /** The fix for a missing runtime, add-on or ffmpeg; Voice Models when not given. */
  onOpenSetup?: () => void;
}) {
  const { t } = useTranslation("studio");
  const { platform, openVoiceModels } = useVoice();
  const [details, setDetails] = useState<NewVoiceDetails>({ basis: null, speakerName: "", voiceName: "", confirmed: false });
  const [file, setFile] = useState<File | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [recording, setRecording] = useState<{ blob: Blob; url: string; seconds: number } | null>(null);
  const [recordingNow, setRecordingNow] = useState(false);
  const [sampleText, setSampleText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<VoiceErrorView | null>(null);
  // Only a blocked microphone has its own button (open the system settings); it reads the raw text.
  const [micError, setMicError] = useState<string | null>(null);
  const captureRef = useRef<MicCapture | null>(null);
  const stepRefs = useRef<Partial<Record<StepName, HTMLElement | null>>>({});
  const [models, setModels] = useState<CloneModel[]>([]);
  const [chosenModel, setChosenModel] = useState("");

  const refreshModels = async () => {
    try {
      setModels((await getCloneModels()).models);
    } catch {
      // The picker then shows the download prompt; saving stays off until a model is known.
    }
  };
  useEffect(() => {
    void refreshModels();
  }, []);
  // A download in progress: look again until it finishes.
  const fetching = models.some((m) => m.download.state === "downloading");
  useEffect(() => {
    if (!fetching) return;
    const timer = window.setInterval(() => void refreshModels(), 3000);
    return () => window.clearInterval(timer);
  }, [fetching]);
  // The choice, or the default while the person has not changed it (or the chosen model is gone).
  const preselected = pickCloneModel(models);
  const modelKey = models.some((m) => m.key === chosenModel && m.installed && m.supported) ? chosenModel : (preselected?.key ?? "");

  const set = (patch: Partial<NewVoiceDetails>) => setDetails((current) => ({ ...current, ...patch }));
  const fail = (err: unknown) => {
    setMicError(null);
    setError(describeVoiceError(err));
  };
  const clearError = () => {
    setError(null);
    setMicError(null);
  };
  const stepTitle = (n: number, label: string) => t("voice.clone.stepTitle", { defaultValue: "{n}. {label}", n, label });

  const discardRecording = () => {
    if (recording) URL.revokeObjectURL(recording.url);
    setRecording(null);
  };

  const chooseFile = (next: File | null) => {
    clearError();
    setFile(next);
    // One sample at a time: a chosen file replaces a recording.
    if (next) discardRecording();
  };

  const toggleRecording = async () => {
    clearError();
    if (captureRef.current) {
      const blob = await captureRef.current.stop();
      captureRef.current = null;
      setRecordingNow(false);
      discardRecording();
      setRecording({ blob, url: URL.createObjectURL(blob), seconds: wavSeconds(blob.size) });
      // A recording replaces a chosen file.
      setFile(null);
      setFileInputKey((key) => key + 1);
      return;
    }
    try {
      captureRef.current = await startMicCapture();
      setRecordingNow(true);
    } catch (err) {
      const message = micErrorMessage(err, platform);
      setMicError(message);
      setError(describeVoiceError(message));
    }
  };

  const fileTypeProblem = file !== null && !isSampleFile(file.name);
  const lengthProblem = file ? null : sampleLengthProblem(recording?.seconds ?? null);
  const hasSample = (file !== null && !fileTypeProblem) || (recording !== null && lengthProblem === null);

  const save = async () => {
    const sample = file ?? recording?.blob;
    if (!sample || !details.basis) return;
    setBusy(true);
    clearError();
    try {
      onCreated(
        await createClonedVoice({
          speakerName: details.speakerName.trim(),
          basis: details.basis as CloneBasis,
          name: details.voiceName.trim() || details.speakerName.trim(),
          sample,
          sampleName: file ? file.name : "recording.wav",
          referenceText: sampleText.trim(),
          model: modelKey,
        }),
      );
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const goTo = (step: StepName) => stepRefs.current[step]?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  // The one button of an error: what this form can do for its action; none for the others (the sentence stands alone).
  const errorAction = (() => {
    const fix = error ? cloneErrorFix(error.action) : null;
    switch (fix) {
      case "retry":
        return () => void save();
      case "sample":
        return () => {
          chooseFile(null);
          setFileInputKey((key) => key + 1);
          goTo("sample");
        };
      case "permission":
        return () => goTo("permission");
      case "model":
        return () => goTo("model");
      case "setup":
        return onOpenSetup ?? openVoiceModels;
      default:
        return null;
    }
  })();

  const section = (step: StepName) => (node: HTMLElement | null) => {
    stepRefs.current[step] = node;
  };

  return (
    <div className="voice-studio voice-clone__new">
      <CloneStatement />
      <fieldset className="voice-clone__basis voice-clone__step" ref={section("permission")}>
        <legend>{stepTitle(1, t("voice.clone.stepPermission", { defaultValue: "Permission" }))}</legend>
        <p className="muted-text">{t("voice.clone.whose", { defaultValue: "Whose voice is it?" })}</p>
        <label>
          <input type="radio" name="clone-basis" checked={details.basis === "own"} onChange={() => set({ basis: "own" })} />{" "}
          {t("voice.clone.basisOwn", { defaultValue: "My own voice" })}
        </label>
        <label>
          <input type="radio" name="clone-basis" checked={details.basis === "permission"} onChange={() => set({ basis: "permission" })} />{" "}
          {t("voice.clone.basisPermission", { defaultValue: "Someone who gave me permission" })}
        </label>
        <label className="voice-clone__confirm">
          <input type="checkbox" checked={details.confirmed} onChange={(e) => set({ confirmed: e.target.checked })} />{" "}
          {t("voice.clone.confirm", { defaultValue: "This is my own voice, or the speaker has given me permission to clone it." })}
        </label>
      </fieldset>
      <div className="voice-clone__sample voice-clone__step" ref={section("sample")}>
        <h4>{stepTitle(2, t("voice.clone.stepSample", { defaultValue: "Sample" }))}</h4>
        <p className="muted-text">
          {t("voice.clone.sampleHelp", {
            defaultValue:
              "3 to 30 seconds of the same person speaking clearly. Upload a WAV, MP3, M4A, FLAC or OGG file, or record with the microphone. 10 to 20 seconds works best.",
          })}
        </p>
        <p className="muted-text">
          {t("voice.clone.sampleMinimumTurbo", {
            defaultValue: "Chatterbox-Turbo needs at least 5 seconds. A sample of 10 seconds or more works with every cloning model.",
          })}
        </p>
        <input
          key={fileInputKey}
          type="file"
          aria-label={t("voice.clone.sampleFile", { defaultValue: "Voice sample file" })}
          accept={SAMPLE_EXTENSIONS.join(",")}
          onChange={(e) => chooseFile(e.target.files?.[0] ?? null)}
        />
        <div className="button-row">
          <button type="button" className={recordingNow ? "primary-button" : "secondary-button"} onClick={() => void toggleRecording()}>
            {recordingNow
              ? t("voice.stopRecording", { defaultValue: "Stop recording" })
              : recording
                ? t("voice.clone.recordAgain", { defaultValue: "Record again" })
                : t("voice.record", { defaultValue: "Record" })}
          </button>
          {recording ? (
            <>
              <audio controls src={recording.url} aria-label={t("voice.clone.recordingPlayback", { defaultValue: "Your recording" })} />
              <span className="muted-text">{formatClock(recording.seconds)}</span>
            </>
          ) : null}
        </div>
        <label className="voice-settings__field">
          <span>{t("voice.clone.sampleText", { defaultValue: "Words spoken in the sample (optional; filled in automatically)" })}</span>
          <input className="text-input" maxLength={2000} value={sampleText} onChange={(e) => setSampleText(e.target.value)} />
        </label>
        {fileTypeProblem ? (
          <p className="voice-error" role="alert">
            {t("voice.clone.sampleType", { defaultValue: "Choose a WAV, MP3, M4A, FLAC or OGG file." })}
          </p>
        ) : null}
        {lengthProblem ? (
          <p className="voice-error" role="alert">
            {lengthProblem === "short"
              ? t("voice.clone.tooShort", { defaultValue: "The recording is shorter than 3 seconds. Record a little more." })
              : t("voice.clone.tooLong", { defaultValue: "The recording is longer than 30 seconds. Record a shorter sample." })}
          </p>
        ) : null}
      </div>
      <div className="voice-clone__name voice-clone__step" ref={section("name")}>
        <h4>{stepTitle(3, t("voice.clone.stepName", { defaultValue: "Name" }))}</h4>
        <div className="voice-studio__row">
          <label className="voice-settings__field">
            <span>{t("voice.clone.speakerName", { defaultValue: "Speaker's name" })}</span>
            <input className="text-input" maxLength={60} value={details.speakerName} onChange={(e) => set({ speakerName: e.target.value })} />
          </label>
          <label className="voice-settings__field">
            <span>{t("voice.clone.voiceName", { defaultValue: "Name for this voice (optional)" })}</span>
            <input className="text-input" maxLength={80} value={details.voiceName} onChange={(e) => set({ voiceName: e.target.value })} />
          </label>
        </div>
      </div>
      <div className="voice-clone__step" ref={section("model")}>
        <CloneModelPicker
          title={stepTitle(4, t("voice.clone.stepModel", { defaultValue: "Model" }))}
          models={models}
          selected={modelKey}
          onSelect={setChosenModel}
          onDownload={(key) =>
            void downloadCloneModel(key).then(refreshModels, (err: unknown) => {
              fail(err);
              return refreshModels();
            })
          }
        />
      </div>
      <div className="button-row">
        <button
          type="button"
          className="primary-button"
          disabled={!canSaveVoice(details, hasSample, modelKey) || recordingNow || busy}
          onClick={() => void save()}
        >
          {busy ? t("voice.clone.saving", { defaultValue: "Saving the voice…" }) : t("voice.clone.save", { defaultValue: "Save voice" })}
        </button>
        {onCancel ? (
          <button type="button" className="secondary-button" disabled={busy} onClick={onCancel}>
            {t("voice.cancel", { defaultValue: "Cancel" })}
          </button>
        ) : null}
      </div>
      {error ? <VoiceErrorNotice view={error} onAction={errorAction} /> : null}
      <MicSettingsButton error={micError} platform={platform} />
    </div>
  );
}
