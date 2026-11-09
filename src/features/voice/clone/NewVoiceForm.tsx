import { useRef, useState } from "react";
import { CloneStatement } from "./CloneStatement";
import { useTranslation } from "react-i18next";
import { MicSettingsButton } from "../MicSettingsButton";
import { formatClock, micErrorMessage, startMicCapture, type MicCapture } from "../audio";
import { useVoice } from "../VoiceContext";
import { createClonedVoice, type ClonedVoice } from "./cloneApi";
import {
  SAMPLE_EXTENSIONS,
  canSaveVoice,
  isSampleFile,
  sampleLengthProblem,
  wavSeconds,
  type CloneBasis,
  type NewVoiceDetails,
} from "./cloneLogic";

/**
 * Making a cloned voice: whose voice it is, the speaker's name, a voice sample (a file from this computer or a
 * microphone recording) and one permission tick. The backend checks the sample's type and length and stores the
 * speaker's name, the basis and the time with the voice.
 */
export function NewVoiceForm({ onCreated, onCancel }: { onCreated: (voice: ClonedVoice) => void; onCancel: () => void }) {
  const { t } = useTranslation("studio");
  const { platform } = useVoice();
  const [details, setDetails] = useState<NewVoiceDetails>({ basis: null, speakerName: "", voiceName: "", confirmed: false });
  const [file, setFile] = useState<File | null>(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [recording, setRecording] = useState<{ blob: Blob; url: string; seconds: number } | null>(null);
  const [recordingNow, setRecordingNow] = useState(false);
  const [sampleText, setSampleText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const captureRef = useRef<MicCapture | null>(null);

  const set = (patch: Partial<NewVoiceDetails>) => setDetails((current) => ({ ...current, ...patch }));

  const discardRecording = () => {
    if (recording) URL.revokeObjectURL(recording.url);
    setRecording(null);
  };

  const chooseFile = (next: File | null) => {
    setError(null);
    setFile(next);
    // One sample at a time: a chosen file replaces a recording.
    if (next) discardRecording();
  };

  const toggleRecording = async () => {
    setError(null);
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
      setError(micErrorMessage(err, platform));
    }
  };

  const fileTypeProblem = file !== null && !isSampleFile(file.name);
  const lengthProblem = file ? null : sampleLengthProblem(recording?.seconds ?? null);
  const hasSample = (file !== null && !fileTypeProblem) || (recording !== null && lengthProblem === null);

  const save = async () => {
    const sample = file ?? recording?.blob;
    if (!sample || !details.basis) return;
    setBusy(true);
    setError(null);
    try {
      onCreated(
        await createClonedVoice({
          speakerName: details.speakerName.trim(),
          basis: details.basis as CloneBasis,
          name: details.voiceName.trim() || details.speakerName.trim(),
          sample,
          sampleName: file ? file.name : "recording.wav",
          referenceText: sampleText.trim(),
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="voice-studio voice-clone__new">
      <CloneStatement />
      <fieldset className="voice-clone__basis">
        <legend>{t("voice.clone.whose", { defaultValue: "Whose voice is it?" })}</legend>
        <label>
          <input type="radio" name="clone-basis" checked={details.basis === "own"} onChange={() => set({ basis: "own" })} />{" "}
          {t("voice.clone.basisOwn", { defaultValue: "My own voice" })}
        </label>
        <label>
          <input type="radio" name="clone-basis" checked={details.basis === "permission"} onChange={() => set({ basis: "permission" })} />{" "}
          {t("voice.clone.basisPermission", { defaultValue: "Someone who gave me permission" })}
        </label>
      </fieldset>
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
      <div className="voice-clone__sample">
        <h4>{t("voice.clone.sampleTitle", { defaultValue: "Voice sample" })}</h4>
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
      </div>
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
      <label className="voice-clone__confirm">
        <input type="checkbox" checked={details.confirmed} onChange={(e) => set({ confirmed: e.target.checked })} />{" "}
        {t("voice.clone.confirm", { defaultValue: "This is my own voice, or the speaker has given me permission to clone it." })}
      </label>
      <div className="button-row">
        <button
          type="button"
          className="primary-button"
          disabled={!canSaveVoice(details, hasSample) || recordingNow || busy}
          onClick={() => void save()}
        >
          {busy ? t("voice.clone.saving", { defaultValue: "Saving the voice…" }) : t("voice.clone.save", { defaultValue: "Save voice" })}
        </button>
        <button type="button" className="secondary-button" disabled={busy} onClick={onCancel}>
          {t("voice.cancel", { defaultValue: "Cancel" })}
        </button>
      </div>
      {error ? (
        <p className="voice-error" role="alert">
          {error}
        </p>
      ) : null}
      <MicSettingsButton error={error} platform={platform} />
    </div>
  );
}
