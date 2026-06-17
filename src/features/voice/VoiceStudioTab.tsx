import { useCallback, useEffect, useRef, useState } from "react";
import { Panel } from "../../components/Panel";
import { transcribeAudio, synthesizeSpeech } from "../../api";
import type { TabId, VoiceRuntime, SttModel, TtsVoice } from "../../types";

export interface VoiceStudioTabProps {
  voiceRuntime: VoiceRuntime | null;
  backendOnline: boolean;
  onSendToChat: (text: string) => void;
  onTabChange: (tab: TabId) => void;
}

type RecordingState = "idle" | "recording" | "transcribing";

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function VoiceStudioTab({ voiceRuntime, backendOnline, onSendToChat }: VoiceStudioTabProps) {
  // ── STT state ──────────────────────────────────────────────────────────────
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [transcript, setTranscript] = useState("");
  const [transcribeError, setTranscribeError] = useState<string | null>(null);
  const [selectedSttModel, setSelectedSttModel] = useState<string>("");
  const [copyLabel, setCopyLabel] = useState("Copy");

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── TTS state ──────────────────────────────────────────────────────────────
  const [ttsText, setTtsText] = useState("");
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  const [speed, setSpeed] = useState(1.0);
  const [ttsLoading, setTtsLoading] = useState(false);
  const [ttsError, setTtsError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const activeAudioUrl = useRef<string | null>(null);

  // Seed selectors from runtime
  useEffect(() => {
    if (voiceRuntime) {
      const defaultModel = voiceRuntime.sttModels.find((m: SttModel) => m.default);
      if (defaultModel && !selectedSttModel) {
        setSelectedSttModel(defaultModel.id);
      }
      if (voiceRuntime.ttsVoices.length > 0 && !selectedVoice) {
        setSelectedVoice(voiceRuntime.ttsVoices[0].id);
      }
    }
  }, [voiceRuntime, selectedSttModel, selectedVoice]);

  // Clean up blob URL on unmount
  useEffect(() => {
    return () => {
      if (activeAudioUrl.current) {
        URL.revokeObjectURL(activeAudioUrl.current);
      }
    };
  }, []);

  // ── STT handlers ───────────────────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    setTranscribeError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
        setRecordingState("transcribing");
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", blob, "recording.webm");
        formData.append("model", selectedSttModel || "mlx-community/whisper-large-v3-turbo-q4");
        try {
          const result = await transcribeAudio(formData);
          setTranscript(result.text);
        } catch (err) {
          setTranscribeError(err instanceof Error ? err.message : "Transcription failed.");
        } finally {
          setRecordingState("idle");
          setRecordingSeconds(0);
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecordingState("recording");
      setRecordingSeconds(0);
      timerRef.current = setInterval(() => {
        setRecordingSeconds((s) => s + 1);
      }, 1000);
    } catch (err) {
      setTranscribeError(
        err instanceof Error ? err.message : "Could not access microphone.",
      );
    }
  }, [selectedSttModel]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  const handleRecordClick = useCallback(() => {
    if (recordingState === "idle") {
      void startRecording();
    } else if (recordingState === "recording") {
      stopRecording();
    }
  }, [recordingState, startRecording, stopRecording]);

  const handleCopy = useCallback(() => {
    if (!transcript) return;
    void navigator.clipboard.writeText(transcript).then(() => {
      setCopyLabel("Copied");
      setTimeout(() => setCopyLabel("Copy"), 2000);
    });
  }, [transcript]);

  // ── TTS handlers ───────────────────────────────────────────────────────────

  const handleSynthesize = useCallback(async () => {
    if (!ttsText.trim()) return;
    setTtsLoading(true);
    setTtsError(null);
    if (activeAudioUrl.current) {
      URL.revokeObjectURL(activeAudioUrl.current);
      activeAudioUrl.current = null;
      setAudioUrl(null);
    }
    try {
      const blob = await synthesizeSpeech(ttsText, selectedVoice || "af_heart", speed);
      const url = URL.createObjectURL(blob);
      activeAudioUrl.current = url;
      setAudioUrl(url);
    } catch (err) {
      setTtsError(err instanceof Error ? err.message : "Synthesis failed.");
    } finally {
      setTtsLoading(false);
    }
  }, [ttsText, selectedVoice, speed]);

  // ── Derived ────────────────────────────────────────────────────────────────

  const sttModels = voiceRuntime?.sttModels ?? [];
  const ttsVoices = voiceRuntime?.ttsVoices ?? [];
  const sttDisabled = !backendOnline || recordingState === "transcribing";

  let recordLabel = "Record";
  if (recordingState === "recording") recordLabel = `Stop  ${formatDuration(recordingSeconds)}`;
  if (recordingState === "transcribing") recordLabel = "Transcribing…";

  return (
    <div className="content-grid image-page-grid">
      {/* ── STT Panel ─────────────────────────────────────────────────────── */}
      <Panel title="Transcribe" subtitle="Record audio and convert to text">
        <div className="voice-studio-stt">
          {/* Model selector */}
          {sttModels.length > 0 && (
            <div className="form-row">
              <label htmlFor="stt-model-select" className="form-label">
                STT Model
              </label>
              <select
                id="stt-model-select"
                className="select-input"
                value={selectedSttModel}
                onChange={(e) => setSelectedSttModel(e.target.value)}
                disabled={recordingState !== "idle"}
              >
                {sttModels.map((m: SttModel) => (
                  <option key={m.id} value={m.id}>
                    {m.name}{m.installed ? "" : " (not installed)"}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Record button */}
          <div className="voice-record-row">
            <button
              type="button"
              className={`voice-record-button${recordingState === "recording" ? " voice-record-button--active" : ""}`}
              onClick={handleRecordClick}
              disabled={sttDisabled}
              aria-label={recordLabel}
            >
              <span className="voice-record-dot" aria-hidden="true" />
            </button>
            <span className="voice-record-label">
              {recordingState === "idle" && "Click to record"}
              {recordingState === "recording" && `Recording… ${formatDuration(recordingSeconds)}`}
              {recordingState === "transcribing" && "Transcribing…"}
            </span>
          </div>

          {transcribeError && (
            <p className="error-text" style={{ marginTop: 8 }}>{transcribeError}</p>
          )}

          {/* Transcript output */}
          <div className="form-row" style={{ marginTop: 12 }}>
            <label className="form-label">
              Transcript
              {transcript && (
                <span className="muted-text" style={{ marginLeft: 8, fontWeight: "normal" }}>
                  {transcript.length} chars
                </span>
              )}
            </label>
            <textarea
              className="prompt-textarea"
              style={{ minHeight: 120 }}
              readOnly
              value={transcript}
              placeholder="Transcript will appear here after recording…"
            />
          </div>

          {/* Actions */}
          <div className="button-row" style={{ marginTop: 8 }}>
            <button
              type="button"
              className="secondary-button"
              onClick={handleCopy}
              disabled={!transcript}
            >
              {copyLabel}
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={() => onSendToChat(transcript)}
              disabled={!transcript}
            >
              Send to Chat
            </button>
          </div>
        </div>
      </Panel>

      {/* ── TTS Panel ─────────────────────────────────────────────────────── */}
      <Panel title="Speak" subtitle="Convert text to speech">
        <div className="voice-studio-tts">
          <div className="form-row">
            <label htmlFor="tts-text" className="form-label">Text</label>
            <textarea
              id="tts-text"
              className="prompt-textarea"
              style={{ minHeight: 100 }}
              placeholder="Enter text to synthesize…"
              value={ttsText}
              onChange={(e) => setTtsText(e.target.value)}
            />
          </div>

          {/* Voice selector */}
          {ttsVoices.length > 0 && (
            <div className="form-row">
              <label htmlFor="tts-voice-select" className="form-label">Voice</label>
              <select
                id="tts-voice-select"
                className="select-input"
                value={selectedVoice}
                onChange={(e) => setSelectedVoice(e.target.value)}
              >
                {ttsVoices.map((v: TtsVoice) => (
                  <option key={v.id} value={v.id}>
                    {v.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Speed slider */}
          <div className="form-row">
            <label htmlFor="tts-speed" className="form-label">
              Speed <span className="muted-text">({speed.toFixed(1)}×)</span>
            </label>
            <input
              id="tts-speed"
              type="range"
              min={0.5}
              max={2.0}
              step={0.1}
              value={speed}
              onChange={(e) => setSpeed(parseFloat(e.target.value))}
              style={{ width: "100%" }}
            />
          </div>

          {ttsError && (
            <p className="error-text" style={{ marginTop: 4 }}>{ttsError}</p>
          )}

          <div className="button-row" style={{ marginTop: 8 }}>
            <button
              type="button"
              className="primary-button"
              onClick={() => void handleSynthesize()}
              disabled={!ttsText.trim() || ttsLoading || !backendOnline}
            >
              {ttsLoading ? "Generating…" : "Generate"}
            </button>
          </div>

          {audioUrl && (
            <div style={{ marginTop: 12 }}>
              {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
              <audio controls src={audioUrl} style={{ width: "100%" }} />
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
}
