import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { formatClock, micErrorMessage, startMicCapture, type MicCapture } from "../audio";
import { useVoice } from "../VoiceContext";
import { createClonedVoice, startConsent, type ClonedVoice, type ConsentChallenge } from "./cloneApi";
import { canStartConsent, sampleLengthProblem, spacedCode, wavSeconds, type CloneBasis, type ConsentForm } from "./cloneLogic";

/**
 * Making a cloned voice: whose voice and the statement, then the speaker reads a sentence with a
 * random code on the microphone. The backend checks the words and code with our own speech-to-text
 * and stores the consent record with the voice; there is no way to skip this step.
 */
export function NewVoiceForm({ onCreated, onCancel }: { onCreated: (voice: ClonedVoice) => void; onCancel: () => void }) {
  const { t } = useTranslation("studio");
  const { platform } = useVoice();
  const [form, setForm] = useState<ConsentForm>({ basis: null, speakerName: "", voiceName: "", confirmed: false });
  const [challenge, setChallenge] = useState<ConsentChallenge | null>(null);
  const [recording, setRecording] = useState<{ blob: Blob; url: string; seconds: number } | null>(null);
  const [recordingNow, setRecordingNow] = useState(false);
  const [sample, setSample] = useState<File | null>(null);
  const [sampleText, setSampleText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const captureRef = useRef<MicCapture | null>(null);

  const set = (patch: Partial<ConsentForm>) => setForm((current) => ({ ...current, ...patch }));

  const getSentence = async () => {
    setError(null);
    try {
      setChallenge(await startConsent(form.speakerName.trim(), form.basis as CloneBasis));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const toggleRecording = async () => {
    setError(null);
    if (captureRef.current) {
      const blob = await captureRef.current.stop();
      captureRef.current = null;
      setRecordingNow(false);
      if (recording) URL.revokeObjectURL(recording.url);
      setRecording({ blob, url: URL.createObjectURL(blob), seconds: wavSeconds(blob.size) });
      return;
    }
    try {
      captureRef.current = await startMicCapture();
      setRecordingNow(true);
    } catch (err) {
      setError(micErrorMessage(err, platform));
    }
  };

  const save = async () => {
    if (!challenge || !recording) return;
    setBusy(true);
    setError(null);
    try {
      onCreated(
        await createClonedVoice({
          challengeId: challenge.id,
          name: form.voiceName.trim() || form.speakerName.trim(),
          consent: recording.blob,
          reference: sample,
          referenceText: sampleText.trim(),
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const lengthProblem = sample ? null : sampleLengthProblem(recording?.seconds ?? null);

  return (
    <div className="voice-studio voice-clone__new">
      {!challenge ? (
        <>
          <fieldset className="voice-clone__basis">
            <legend>{t("voice.clone.whose", { defaultValue: "Whose voice is it?" })}</legend>
            <label>
              <input type="radio" name="clone-basis" checked={form.basis === "own"} onChange={() => set({ basis: "own" })} />{" "}
              {t("voice.clone.basisOwn", { defaultValue: "My own voice" })}
            </label>
            <label>
              <input type="radio" name="clone-basis" checked={form.basis === "permission"} onChange={() => set({ basis: "permission" })} />{" "}
              {t("voice.clone.basisPermission", { defaultValue: "Someone who gave me permission (they read the sentence themselves)" })}
            </label>
          </fieldset>
          <div className="voice-studio__row">
            <label className="voice-settings__field">
              <span>{t("voice.clone.speakerName", { defaultValue: "Speaker's name" })}</span>
              <input className="text-input" maxLength={60} value={form.speakerName} onChange={(e) => set({ speakerName: e.target.value })} />
            </label>
            <label className="voice-settings__field">
              <span>{t("voice.clone.voiceName", { defaultValue: "Name for this voice (optional)" })}</span>
              <input className="text-input" maxLength={80} value={form.voiceName} onChange={(e) => set({ voiceName: e.target.value })} />
            </label>
          </div>
          <label className="voice-clone__confirm">
            <input type="checkbox" checked={form.confirmed} onChange={(e) => set({ confirmed: e.target.checked })} />{" "}
            {t("voice.clone.confirm", {
              defaultValue:
                "This is my own voice, or the speaker has given me permission to copy it. It is not a public figure, a child or someone who has died.",
            })}
          </label>
          <div className="button-row">
            <button type="button" className="primary-button" disabled={!canStartConsent(form)} onClick={() => void getSentence()}>
              {t("voice.clone.getSentence", { defaultValue: "Get the consent sentence" })}
            </button>
            <button type="button" className="secondary-button" onClick={onCancel}>
              {t("voice.cancel", { defaultValue: "Cancel" })}
            </button>
          </div>
        </>
      ) : (
        <>
          <p className="muted-text">
            {t("voice.clone.readThis", {
              defaultValue: "{name} reads this sentence aloud, including the code one digit at a time:",
              name: challenge.speakerName,
            })}
          </p>
          <blockquote className="voice-clone__sentence" aria-live="polite">
            {challenge.sentence}
          </blockquote>
          <p className="muted-text">
            {t("voice.clone.codeHint", { defaultValue: "Code: {code}. The recording is also used as the voice sample unless you add one below.", code: spacedCode(challenge.code) })}
          </p>
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
                <audio controls src={recording.url} />
                <span className="muted-text">{formatClock(recording.seconds)}</span>
              </>
            ) : null}
          </div>
          <div className="voice-clone__sample">
            <h4>{t("voice.clone.sampleTitle", { defaultValue: "Upload a voice sample (optional)" })}</h4>
            <p className="muted-text">
              {t("voice.clone.sampleHelp", {
                defaultValue: "3 to 30 seconds of the same person speaking clearly, from a file on this computer. 10 to 20 seconds works best.",
              })}
            </p>
            <input type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg" onChange={(e) => setSample(e.target.files?.[0] ?? null)} />
            <label className="voice-settings__field">
              <span>{t("voice.clone.sampleText", { defaultValue: "Words spoken in the sample (optional; filled in automatically)" })}</span>
              <input className="text-input" maxLength={2000} value={sampleText} onChange={(e) => setSampleText(e.target.value)} />
            </label>
          </div>
          {lengthProblem ? (
            <p className="voice-error" role="alert">
              {lengthProblem === "short"
                ? t("voice.clone.tooShort", { defaultValue: "The recording is shorter than 3 seconds. Read the whole sentence." })
                : t("voice.clone.tooLong", { defaultValue: "The recording is longer than 30 seconds. Record just the sentence." })}
            </p>
          ) : null}
          <div className="button-row">
            <button type="button" className="primary-button" disabled={!recording || recordingNow || busy || lengthProblem !== null} onClick={() => void save()}>
              {busy ? t("voice.clone.checking", { defaultValue: "Checking the recording…" }) : t("voice.clone.save", { defaultValue: "Check and save voice" })}
            </button>
            <button type="button" className="secondary-button" disabled={busy} onClick={onCancel}>
              {t("voice.cancel", { defaultValue: "Cancel" })}
            </button>
          </div>
        </>
      )}
      {error ? (
        <p className="voice-error" role="alert">
          {error}
        </p>
      ) : null}
    </div>
  );
}
