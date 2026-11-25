import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../../components/Panel";
import { getLibraryAudio, getLibraryItem, transcribeBlob } from "../../../api";
import type { VoiceLibraryItem, SttModel, VoiceRuntime } from "../../../types";
import { MicSettingsButton } from "../MicSettingsButton";
import { formatClock, micErrorMessage, startMicCapture, type MicCapture } from "../audio";
import { useVoice } from "../VoiceContext";
import { describeVoiceError, type VoiceErrorView } from "../voiceErrorText";
import { VOICE_LANGUAGES, sttFindsLanguageItself, sttLanguageMode, sttLanguageModeText, voiceLanguageName } from "../voiceLanguages";
import { TranscriptEditor } from "./TranscriptEditor";

const ACCEPTED_FILES = "audio/*,video/*,.wav,.mp3,.m4a,.aac,.flac,.ogg,.opus,.webm,.mp4,.mov,.mkv,.m4v,.aiff,.caf";
const MAX_FILE_BYTES = 500 * 1024 * 1024;
const SPEAKER_COUNTS = [0, 2, 3, 4, 5, 6, 8];

/** Prefer a downloaded model, then the recommended one. */
export function pickDefaultSttModel(models: SttModel[], preferred?: string): SttModel | undefined {
  const runnable = models.filter((m) => m.backendInstalled !== false);
  const pool = runnable.length > 0 ? runnable : models;
  const chosen = preferred ? pool.find((m) => m.id === preferred || m.key === preferred) : undefined;
  if (chosen) return chosen;
  const recommended = pool.find((m) => m.default);
  if (recommended?.installed) return recommended;
  return pool.find((m) => m.installed) ?? recommended ?? pool[0];
}

interface JobOptions {
  model: string;
  language: string;
  translate: boolean;
  speakers: boolean;
  speakerCount: number;
}

interface Job {
  id: number;
  name: string;
  file: Blob;
  options: JobOptions;
  status: "queued" | "running" | "done" | "error" | "cancelled";
  error?: VoiceErrorView;
  libraryId?: string;
  startedAt?: number;
  seconds?: number;
  note?: string;
}

export interface OpenRequest {
  id: string;
  /** Changes on every request, so opening the same item twice works. */
  nonce: number;
}

export interface TranscribePanelProps {
  runtime: VoiceRuntime;
  openRequest: OpenRequest | null;
  onSendToChat: (text: string) => void;
  onOpenModels: () => void;
}

export function TranscribePanel({ runtime, openRequest, onSendToChat, onOpenModels }: TranscribePanelProps) {
  const { t, i18n } = useTranslation("studio");
  const tf = t as unknown as (key: string, options?: Record<string, unknown>) => string;
  const { settings, platform, saveSettings } = useVoice();
  const [options, setOptions] = useState<JobOptions>({
    model: pickDefaultSttModel(runtime.sttModels, settings.sttModel)?.id ?? "",
    language: settings.language,
    translate: false,
    speakers: false,
    speakerCount: 0,
  });
  const [jobs, setJobs] = useState<Job[]>([]);
  const [recording, setRecording] = useState(false);
  const [level, setLevel] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [remember, setRemember] = useState(false);
  const [now, setNow] = useState(Date.now());
  const [open, setOpen] = useState<{ item: VoiceLibraryItem; audioUrl: string | null } | null>(null);
  const captureRef = useRef<MicCapture | null>(null);
  const abortRef = useRef<{ id: number; controller: AbortController } | null>(null);
  const nextId = useRef(1);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const modelSelect = useRef<HTMLSelectElement | null>(null);
  const languageSelect = useRef<HTMLSelectElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);

  const model = runtime.sttModels.find((m) => m.id === options.model);
  const canTranslate = model?.translate === true;
  const findsLanguage = sttFindsLanguageItself(model?.key);

  // Change the model or language; once "Remember as default" is ticked, the change is saved as the default too
  // (Chat's Dictate and voice calls read the same two settings).
  const persistDefault = async (next: Pick<JobOptions, "model" | "language">) => {
    const picked = runtime.sttModels.find((m) => m.id === next.model);
    try {
      await saveSettings({ sttModel: picked?.key ?? picked?.id ?? "", language: next.language });
    } catch (err) {
      setError(describeVoiceError(err).sentence);
    }
  };
  const change = (patch: Partial<Pick<JobOptions, "model" | "language">>) => {
    const next = { ...options, ...patch };
    setOptions(next);
    if (remember) void persistDefault(next);
  };

  useEffect(() => {
    if (!options.model) {
      const picked = pickDefaultSttModel(runtime.sttModels, settings.sttModel)?.id ?? "";
      if (picked) setOptions((o) => ({ ...o, model: picked }));
    }
  }, [options.model, runtime.sttModels, settings.sttModel]);

  const running = jobs.some((j) => j.status === "running");
  useEffect(() => {
    if (!running && !recording) return;
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [running, recording]);

  const showItem = useCallback((item: VoiceLibraryItem, audio: Blob | null) => {
    if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    audioUrlRef.current = audio ? URL.createObjectURL(audio) : null;
    setOpen({ item, audioUrl: audioUrlRef.current });
  }, []);

  useEffect(
    () => () => {
      captureRef.current?.cancel();
      abortRef.current?.controller.abort();
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    },
    [],
  );

  // Open a Library item sent here from the Library page.
  useEffect(() => {
    if (!openRequest) return;
    let cancelled = false;
    void (async () => {
      try {
        const item = await getLibraryItem(openRequest.id);
        const audio = item.audio ? await getLibraryAudio(item.id).catch(() => null) : null;
        if (!cancelled) showItem(item, audio);
      } catch (err) {
        if (!cancelled) setError(describeVoiceError(err).sentence);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [openRequest, showItem]);

  const patchJob = (id: number, patch: Partial<Job>) => setJobs((all) => all.map((j) => (j.id === id ? { ...j, ...patch } : j)));

  // Work through the queue one file at a time.
  useEffect(() => {
    if (running) return;
    const job = jobs.find((j) => j.status === "queued");
    if (!job) return;
    const controller = new AbortController();
    abortRef.current = { id: job.id, controller };
    patchJob(job.id, { status: "running", startedAt: Date.now() });
    void (async () => {
      try {
        const result = await transcribeBlob(job.file, job.name, {
          model: job.options.model || undefined,
          language: job.options.language || undefined,
          task: job.options.translate ? "translate" : "transcribe",
          words: true,
          speakers: job.options.speakers,
          speakerCount: job.options.speakerCount,
          save: true,
          signal: controller.signal,
        });
        patchJob(job.id, {
          status: "done",
          libraryId: result.libraryId,
          seconds: result.seconds ?? undefined,
          note: result.note ?? undefined,
        });
        if (result.libraryId) {
          const item = await getLibraryItem(result.libraryId);
          // Show the first finished file, or this one if nothing is open.
          setOpen((current) => {
            if (current) return current;
            if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
            audioUrlRef.current = URL.createObjectURL(job.file);
            return { item, audioUrl: audioUrlRef.current };
          });
        }
      } catch (err) {
        if (controller.signal.aborted) patchJob(job.id, { status: "cancelled" });
        else patchJob(job.id, { status: "error", error: describeVoiceError(err) });
      } finally {
        if (abortRef.current?.id === job.id) abortRef.current = null;
      }
    })();
  }, [jobs, running]);

  const enqueue = (files: { blob: Blob; name: string }[]) => {
    const tooBig = files.filter((f) => f.blob.size > MAX_FILE_BYTES);
    if (tooBig.length) setError(t("voice.transcribe.tooBig", { defaultValue: "{name} is larger than 500 MB.", name: tooBig[0].name }));
    const accepted = files.filter((f) => f.blob.size <= MAX_FILE_BYTES && f.blob.size > 0);
    const snapshot = { ...options, translate: options.translate && canTranslate, speakers: options.speakers && runtime.speakersAvailable };
    setJobs((all) => [
      ...all,
      ...accepted.map((f) => ({ id: nextId.current++, name: f.name, file: f.blob, options: snapshot, status: "queued" as const })),
    ]);
  };

  const onFiles = (list: FileList | null) => {
    if (list?.length) enqueue(Array.from(list, (file) => ({ blob: file, name: file.name })));
  };

  const stopRecording = useCallback(async () => {
    const capture = captureRef.current;
    captureRef.current = null;
    setRecording(false);
    setLevel(0);
    if (!capture) return;
    const wav = await capture.stop();
    const stamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    enqueue([{ blob: wav, name: `${t("voice.transcribe.recordingName", { defaultValue: "Recording" })} ${stamp}.wav` }]);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- enqueue reads the latest options when called
  }, [options, canTranslate, runtime.speakersAvailable, t]);

  const record = async () => {
    if (recording) {
      await stopRecording();
      return;
    }
    setError(null);
    try {
      captureRef.current = await startMicCapture({ onLevel: setLevel, onLimit: () => void stopRecording() });
      setSeconds(0);
      setRecording(true);
      const started = Date.now();
      const tick = window.setInterval(() => {
        if (!captureRef.current) window.clearInterval(tick);
        else setSeconds((Date.now() - started) / 1000);
      }, 250);
    } catch (err) {
      setError(micErrorMessage(err, platform));
    }
  };

  const cancelJob = (job: Job) => {
    if (job.status === "running") abortRef.current?.controller.abort();
    else patchJob(job.id, { status: "cancelled" });
  };

  const openJob = async (job: Job) => {
    if (!job.libraryId) return;
    try {
      showItem(await getLibraryItem(job.libraryId), job.file);
    } catch (err) {
      setError(describeVoiceError(err).sentence);
    }
  };

  // The one button an error offers: try the file again, point at the setting to change, or go where the part is added.
  const runErrorAction = (job: Job) => {
    switch (job.error?.action) {
      case "retry":
        patchJob(job.id, { status: "queued", error: undefined });
        break;
      case "pick-another-language":
        languageSelect.current?.focus();
        break;
      case "pick-another-model":
      case "smaller-model":
        modelSelect.current?.focus();
        break;
      default:
        onOpenModels();
    }
  };

  const statusText = (job: Job) => {
    switch (job.status) {
      case "queued":
        return t("voice.queue.queued", { defaultValue: "Waiting" });
      case "running":
        return `${t("voice.queue.running", { defaultValue: "Transcribing" })} ${formatClock((now - (job.startedAt ?? now)) / 1000)}`;
      case "done":
        return job.seconds != null
          ? t("voice.queue.doneIn", { defaultValue: "Done in {s} s", s: job.seconds.toFixed(1) })
          : t("voice.queue.done", { defaultValue: "Done" });
      case "cancelled":
        return t("voice.queue.cancelled", { defaultValue: "Cancelled" });
      default:
        return job.error?.sentence ?? t("voice.queue.failed", { defaultValue: "Failed" });
    }
  };

  return (
    <>
      <Panel
        title={t("voice.transcribe.title", { defaultValue: "Transcribe" })}
        subtitle={t("voice.transcribe.subtitleLibrary", { defaultValue: "Record, or drop audio and video files. Transcripts are kept in the Library." })}
      >
        <div className="voice-studio">
          <div className="voice-studio__row">
            <label className="voice-settings__field">
              <span>{t("voice.transcribe.model", { defaultValue: "Model" })}</span>
              <select ref={modelSelect} className="text-input" value={options.model} onChange={(e) => change({ model: e.target.value })}>
                {runtime.sttModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {t("voice.transcribe.modelOption", {
                      defaultValue: "{name}, {language}",
                      name: m.name,
                      language: sttLanguageModeText(sttLanguageMode(m.key, options.language), tf),
                    })}
                  </option>
                ))}
              </select>
            </label>
            <label className="voice-settings__field">
              <span>{t("voice.transcribe.language", { defaultValue: "Language" })}</span>
              <select
                ref={languageSelect}
                className="text-input"
                value={findsLanguage ? "" : options.language}
                disabled={findsLanguage}
                title={findsLanguage ? t("voice.transcribe.languageFixed", { defaultValue: "This model finds the language itself." }) : undefined}
                onChange={(e) => change({ language: e.target.value })}
              >
                {VOICE_LANGUAGES.map(([code, label]) => (
                  <option key={code} value={code}>
                    {voiceLanguageName(code, label, i18n.language)}
                  </option>
                ))}
              </select>
            </label>
            <label
              className="checkbox-label"
              title={t("voice.transcribe.rememberTitle", { defaultValue: "Dictate in Chat and voice calls use this model and language too." })}
            >
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => {
                  setRemember(e.target.checked);
                  if (e.target.checked) void persistDefault(options);
                }}
              />
              <span>{t("voice.transcribe.remember", { defaultValue: "Remember as default" })}</span>
            </label>
          </div>
          {model && !model.installed ? (
            <p className="muted-text voice-studio__download-note">
              {t("voice.transcribe.downloadsFirst", { defaultValue: "Downloads {gb} GB before the first file.", gb: model.sizeGb })}
            </p>
          ) : null}

          <div className="voice-studio__options">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={options.translate && canTranslate}
                disabled={!canTranslate}
                onChange={(e) => setOptions({ ...options, translate: e.target.checked })}
              />
              <span>{t("voice.transcribe.translateWhich", { defaultValue: "Translate to English (Whisper Small or Base)" })}</span>
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={options.speakers && runtime.speakersAvailable}
                disabled={!runtime.speakersAvailable}
                onChange={(e) => setOptions({ ...options, speakers: e.target.checked })}
              />
              <span>{t("voice.transcribe.speakers", { defaultValue: "Label speakers" })}</span>
            </label>
            {options.speakers && runtime.speakersAvailable ? (
              <select
                className="text-input voice-studio__count"
                value={options.speakerCount}
                aria-label={t("voice.transcribe.speakerCount", { defaultValue: "Number of speakers" })}
                onChange={(e) => setOptions({ ...options, speakerCount: Number(e.target.value) })}
              >
                {SPEAKER_COUNTS.map((n) => (
                  <option key={n} value={n}>
                    {n === 0 ? t("voice.transcribe.speakersAuto", { defaultValue: "Find how many" }) : t("voice.transcribe.speakersN", { defaultValue: "{n} speakers", n })}
                  </option>
                ))}
              </select>
            ) : null}
            {model?.needsAligner && runtime.aligner && !runtime.aligner.installed ? (
              <span className="muted-text voice-studio__word-timings">
                {t("voice.transcribe.needsWordTimings", {
                  defaultValue: "{name} gives segment times only until the Word timings download ({gb} GB) is in.",
                  name: model.name,
                  gb: runtime.aligner.sizeGb,
                })}{" "}
                <button type="button" className="link-button" onClick={onOpenModels}>
                  {t("voice.studio.openModels", { defaultValue: "Open Voice Models" })}
                </button>
              </span>
            ) : null}
            {!runtime.speakersAvailable ? (
              <span className="muted-text">
                {t("voice.transcribe.speakersNeedUpdate", { defaultValue: "Speaker labels need a voice runtime update." })}{" "}
                <button type="button" className="link-button" onClick={onOpenModels}>
                  {t("voice.studio.openModels", { defaultValue: "Open Voice Models" })}
                </button>
              </span>
            ) : null}
          </div>

          <div
            className={`voice-drop${dragging ? " voice-drop--active" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              onFiles(e.dataTransfer.files);
            }}
          >
            <button
              type="button"
              className={`voice-record${recording ? " voice-record--active" : ""}`}
              onClick={() => void record()}
              aria-label={recording ? t("voice.stopRecording", { defaultValue: "Stop recording" }) : t("voice.record", { defaultValue: "Record" })}
            >
              <span className="voice-record__dot" />
            </button>
            <div className="voice-drop__text">
              {recording ? (
                <>
                  <strong>{formatClock(seconds)}</strong>
                  <span className="voice-level">
                    <span className="voice-level__fill" style={{ transform: `scaleX(${0.05 + level * 0.95})` }} />
                  </span>
                </>
              ) : (
                <span>
                  {t("voice.transcribe.hintQueue", { defaultValue: "Record, or drop files here (several at once is fine)." })}{" "}
                  <button type="button" className="link-button" onClick={() => fileInput.current?.click()}>
                    {t("voice.transcribe.chooseFiles", { defaultValue: "Choose files" })}
                  </button>
                  <small className="muted-text voice-drop__types">
                    {t("voice.transcribe.fileTypes", { defaultValue: "WAV, MP3, M4A, FLAC, OGG, MP4, MOV and more" })}
                  </small>
                </span>
              )}
            </div>
            <input
              ref={fileInput}
              type="file"
              accept={ACCEPTED_FILES}
              multiple
              hidden
              onChange={(e) => {
                onFiles(e.target.files);
                e.target.value = "";
              }}
            />
          </div>

          {error ? (
            <p className="voice-error" role="alert">
              {error}
            </p>
          ) : null}
          <MicSettingsButton error={error} platform={platform} />

          {jobs.length > 0 ? (
            <ul className="voice-queue" aria-label={t("voice.queue.title", { defaultValue: "Queue" })}>
              {jobs.map((job) => (
                <li key={job.id} className={`voice-queue__item voice-queue__item--${job.status}${open?.item.id === job.libraryId ? " voice-queue__item--open" : ""}`}>
                  <span className="voice-queue__name" title={job.name}>
                    {job.name}
                  </span>
                  <span className="voice-queue__status" title={job.note}>
                    {statusText(job)}
                    {job.note ? ` · ${job.note}` : ""}
                  </span>
                  {job.status === "done" ? (
                    <button type="button" className="link-button" onClick={() => void openJob(job)}>
                      {t("voice.queue.open", { defaultValue: "Open" })}
                    </button>
                  ) : job.status === "queued" || job.status === "running" ? (
                    <button type="button" className="link-button" onClick={() => cancelJob(job)}>
                      {t("voice.cancel", { defaultValue: "Cancel" })}
                    </button>
                  ) : (
                    <button type="button" className="link-button" onClick={() => setJobs((all) => all.filter((j) => j.id !== job.id))}>
                      {t("voice.queue.remove", { defaultValue: "Remove" })}
                    </button>
                  )}
                  {job.status === "error" && job.error && (job.error.actionLabel || job.error.details) ? (
                    <span className="voice-queue__error" style={{ gridColumn: "1 / -1" }}>
                      {job.error.actionLabel ? (
                        <button type="button" className="link-button" onClick={() => runErrorAction(job)}>
                          {job.error.actionLabel}
                        </button>
                      ) : null}
                      {job.error.details ? (
                        <details className="voice-error-details">
                          <summary>{t("voice.errors.detailsToggle", { defaultValue: "Details" })}</summary>
                          <span className="muted-text">{job.error.details}</span>
                        </details>
                      ) : null}
                    </span>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </Panel>

      <Panel
        title={open?.item.title ?? t("voice.editor.heading", { defaultValue: "Transcript" })}
        subtitle={
          open
            ? [open.item.durationS != null ? formatClock(open.item.durationS) : null, open.item.language, open.item.task === "translate" ? t("voice.editor.translated", { defaultValue: "translated to English" }) : null]
                .filter(Boolean)
                .join(" · ")
            : t("voice.editor.subtitle", { defaultValue: "Click a time or a word to play from there; double-click a line to edit it." })
        }
      >
        {open ? (
          <TranscriptEditor
            key={open.item.id}
            item={open.item}
            audioUrl={open.audioUrl}
            onSaved={(item) => setOpen((current) => (current && current.item.id === item.id ? { ...current, item } : current))}
            onSendToChat={onSendToChat}
          />
        ) : (
          <div className="empty-state">
            <p className="muted-text">{t("voice.editor.none", { defaultValue: "Finished transcripts open here and are kept in the Library." })}</p>
          </div>
        )}
      </Panel>
    </>
  );
}
