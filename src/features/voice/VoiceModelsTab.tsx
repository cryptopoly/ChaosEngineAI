import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import {
  cancelVoiceInstall,
  downloadModel,
  getDownloadStatus,
  getVoiceInstallStatus,
  startVoiceInstall,
  stopVoiceWorker,
} from "../../api";
import type { DownloadStatus } from "../../api";
import type { SttModel, TtsVoice, VoiceInstallJob, VoiceRuntime } from "../../types";
import { speechPlayer, useSpeechPlayer } from "./speechPlayer";
import { useVoice } from "./VoiceContext";

export interface VoiceModelsTabProps {
  voiceRuntime: VoiceRuntime | null;
  backendOnline: boolean;
  onRefreshVoiceRuntime: () => void;
}

const PREVIEW_TEXT: Record<string, string> = {
  "en-US": "Hello! This is how I sound when I read your chats aloud.",
  "en-GB": "Hello! This is how I sound when I read your chats aloud.",
  es: "¡Hola! Así sueno cuando leo tus conversaciones en voz alta.",
  fr: "Bonjour ! Voici ma voix quand je lis vos conversations.",
  hi: "नमस्ते! जब मैं आपकी बातचीत पढ़ता हूँ तो मेरी आवाज़ ऐसी होती है।",
  it: "Ciao! Ecco come suono quando leggo le tue conversazioni.",
  ja: "こんにちは。チャットを読み上げるときの声です。",
  "pt-BR": "Olá! É assim que eu soo quando leio suas conversas.",
  zh: "你好！这是我朗读对话时的声音。",
};

const LANGUAGES: Array<[string, string]> = [
  ["", "Detect automatically"],
  ["en", "English"],
  ["de", "German"],
  ["fr", "French"],
  ["es", "Spanish"],
  ["it", "Italian"],
  ["pt", "Portuguese"],
  ["nl", "Dutch"],
  ["pl", "Polish"],
  ["ru", "Russian"],
  ["uk", "Ukrainian"],
  ["ja", "Japanese"],
  ["zh", "Chinese"],
  ["ko", "Korean"],
  ["hi", "Hindi"],
  ["ar", "Arabic"],
  ["tr", "Turkish"],
];

export const VOICE_LANGUAGES = LANGUAGES;

function laneDescription(runtime: VoiceRuntime): string {
  if (runtime.lane === "mlx") return "Apple Silicon: Parakeet and Whisper on MLX, Kokoro voices";
  if (runtime.cuda) return "NVIDIA GPU: Whisper on CUDA, Kokoro voices";
  if (runtime.nvidiaGpu) return "Whisper (CUDA libraries are added by the install), Kokoro voices";
  return "Whisper on the CPU, Kokoro voices";
}

function VoicePackPanel({ runtime, onChanged }: { runtime: VoiceRuntime; onChanged: () => void }) {
  const { t } = useTranslation("studio");
  const [job, setJob] = useState<VoiceInstallJob>(runtime.install);
  const [extras, setExtras] = useState<string[]>(runtime.extras.filter((e) => e.installed).map((e) => e.id));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!job.running) return;
    const timer = window.setInterval(() => {
      void getVoiceInstallStatus()
        .then((next) => {
          setJob(next);
          if (!next.running) onChanged();
        })
        .catch(() => {});
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job.running, onChanged]);

  const install = async () => {
    setError(null);
    try {
      setJob(await startVoiceInstall(extras));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const extrasChanged =
    runtime.installed &&
    runtime.extras.some((extra) => extra.installed !== extras.includes(extra.id));

  return (
    <Panel
      title={t("voice.pack.title", { defaultValue: "Voice pack" })}
      subtitle={laneDescription(runtime)}
      className="span-2"
    >
      <div className="voice-pack">
        <div className="voice-pack__status">
          {runtime.installed ? (
            <span className="badge success">{t("voice.pack.installed", { defaultValue: "Installed" })}</span>
          ) : (
            <span className="badge muted">{t("voice.pack.notInstalled", { defaultValue: "Not installed" })}</span>
          )}
          <span className="muted-text">
            {t("voice.pack.about", {
              defaultValue:
                "Speech-to-text and text-to-speech run in their own environment on this computer (about 2 GB, plus the models you download). Nothing is sent online.",
            })}
          </span>
        </div>

        {runtime.extras.length > 0 ? (
          <div className="voice-pack__extras">
            {runtime.extras.map((extra) => (
              <label key={extra.id} className="checkbox-label">
                <input
                  type="checkbox"
                  checked={extras.includes(extra.id)}
                  disabled={job.running}
                  onChange={(event) =>
                    setExtras((current) =>
                      event.target.checked ? [...current, extra.id] : current.filter((id) => id !== extra.id),
                    )
                  }
                />
                {extra.label}
              </label>
            ))}
          </div>
        ) : null}

        {job.running ? (
          <div className="progress-row">
            <div className="progress-meta">
              <span>
                {job.phaseLabel} ({job.phaseIndex}/{job.phaseTotal})
              </span>
              <button type="button" className="secondary-button" onClick={() => void cancelVoiceInstall().then(setJob)}>
                {t("voice.pack.cancel", { defaultValue: "Cancel" })}
              </button>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${Math.max(4, job.percent)}%` }} />
            </div>
          </div>
        ) : (
          <div className="button-row">
            {!runtime.installed || extrasChanged ? (
              <button type="button" className="primary-button" onClick={() => void install()}>
                {runtime.installed
                  ? t("voice.pack.update", { defaultValue: "Update voice pack" })
                  : t("voice.pack.install", { defaultValue: "Install voice pack" })}
              </button>
            ) : (
              <button type="button" className="secondary-button" onClick={() => void install()}>
                {t("voice.pack.reinstall", { defaultValue: "Repair / reinstall" })}
              </button>
            )}
          </div>
        )}

        {job.phase === "error" || job.phase === "cancelled" || error ? (
          <p className="voice-error" role="alert">
            {error ?? job.error}
          </p>
        ) : null}
        {job.phase === "done" && !job.running ? (
          <p className="muted-text">{t("voice.pack.done", { defaultValue: "Voice pack installed." })}</p>
        ) : null}
        {job.log.length > 0 ? (
          <details className="install-log-panel">
            <summary className="install-log-summary">{t("voice.pack.log", { defaultValue: "Install log" })}</summary>
            <pre className="install-log-terminal">{job.log.join("\n")}</pre>
          </details>
        ) : null}
      </div>
    </Panel>
  );
}

function SttModelCard({
  model,
  selected,
  onSelect,
  onInstalled,
}: {
  model: SttModel;
  selected: boolean;
  onSelect: () => void;
  onInstalled: () => void;
}) {
  const { t } = useTranslation("studio");
  const [status, setStatus] = useState<DownloadStatus | null>(null);

  useEffect(() => {
    if (status?.state !== "downloading") return;
    const timer = window.setInterval(() => {
      void getDownloadStatus().then((all) => {
        const mine = all.find((d) => d.repo === model.id);
        if (!mine) return;
        setStatus(mine);
        if (mine.state === "completed") onInstalled();
      });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [status, model.id, onInstalled]);

  const download = async () => {
    try {
      setStatus(await downloadModel(model.id));
    } catch (err) {
      setStatus({ repo: model.id, state: "failed", progress: 0, downloadedGb: 0, totalGb: null, error: String(err) });
    }
  };

  return (
    <article className={`image-library-card voice-model-card${selected ? " voice-model-card--selected" : ""}`}>
      <div className="image-library-card-head">
        <div>
          <h3 className="voice-model-card__name">{model.name}</h3>
          <p className="muted-text voice-model-card__meta">
            {model.languages} · {model.sizeGb} GB · {model.backend}
          </p>
        </div>
        <div className="voice-model-card__badges">
          {model.default ? <span className="badge subtle">{t("voice.models.recommended", { defaultValue: "Recommended" })}</span> : null}
          {model.installed ? (
            <span className="badge success">{t("voice.models.downloaded", { defaultValue: "Downloaded" })}</span>
          ) : null}
        </div>
      </div>
      <div className="voice-model-card__actions">
        <label className="checkbox-label">
          <input type="radio" name="voice-stt-model" checked={selected} onChange={onSelect} />
          {t("voice.models.use", { defaultValue: "Use for dictation" })}
        </label>
        {!model.installed ? (
          status?.state === "downloading" ? (
            <span className="muted-text">{Math.round((status.progress ?? 0) * 100)}%</span>
          ) : (
            <button type="button" className="secondary-button" onClick={() => void download()}>
              {t("voice.models.download", { defaultValue: "Download" })}
            </button>
          )
        ) : null}
      </div>
      {!model.installed && status?.state !== "downloading" ? (
        <p className="muted-text voice-model-card__meta">
          {t("voice.models.firstUse", { defaultValue: "Downloads on first use if you skip this." })}
        </p>
      ) : null}
      {status?.state === "failed" ? <p className="voice-error">{status.error}</p> : null}
    </article>
  );
}

function VoicesPanel({ voices }: { voices: TtsVoice[] }) {
  const { t } = useTranslation("studio");
  const { settings, saveSettings, runtime } = useVoice();
  const player = useSpeechPlayer();
  const groups = useMemo(() => {
    const byLanguage = new Map<string, TtsVoice[]>();
    for (const voice of voices) {
      const key = voice.languageLabel ?? voice.language;
      byLanguage.set(key, [...(byLanguage.get(key) ?? []), voice]);
    }
    return [...byLanguage.entries()];
  }, [voices]);

  return (
    <Panel
      title={t("voice.voices.title", { defaultValue: "Voices" })}
      subtitle={t("voice.voices.subtitle", { defaultValue: "Kokoro: 54 voices in 9 languages. Pick the one chat reads aloud with." })}
      className="span-2"
    >
      <div className="voice-groups">
        {groups.map(([language, list]) => (
          <section key={language} className="voice-group">
            <h4 className="voice-group__title">
              {language}
              {list[0]?.available === false ? (
                <span className="muted-text voice-group__note">
                  {list[0].language === "zh"
                    ? t("voice.voices.needsChinese", { defaultValue: "Tick Mandarin Chinese voices in the voice pack." })
                    : t("voice.voices.notYet", { defaultValue: "Not available on this computer yet." })}
                </span>
              ) : null}
            </h4>
            <div className="voice-chips">
              {list.map((voice) => {
                const previewId = `preview-${voice.id}`;
                const previewing = player.ownerId === previewId && player.status !== "idle";
                const selected = settings.ttsVoice === voice.id;
                return (
                  <div key={voice.id} className={`voice-chip${selected ? " voice-chip--selected" : ""}`}>
                    <button
                      type="button"
                      className="voice-chip__name"
                      disabled={voice.available === false}
                      onClick={() => void saveSettings({ ttsVoice: voice.id })}
                      title={t("voice.voices.useVoice", { defaultValue: "Use this voice" })}
                    >
                      {voice.name}
                      <span className="voice-chip__gender">{voice.gender === "male" ? "M" : "F"}</span>
                    </button>
                    <button
                      type="button"
                      className="voice-chip__play"
                      disabled={voice.available === false || !runtime?.ttsAvailable}
                      aria-label={t("voice.voices.preview", { defaultValue: "Preview" })}
                      title={t("voice.voices.preview", { defaultValue: "Preview" })}
                      onClick={() =>
                        previewing
                          ? speechPlayer.stop()
                          : void speechPlayer.play(previewId, PREVIEW_TEXT[voice.language] ?? PREVIEW_TEXT["en-US"], voice.id, settings.speed)
                      }
                    >
                      {previewing ? "■" : "▶"}
                    </button>
                  </div>
                );
              })}
            </div>
          </section>
        ))}
      </div>
    </Panel>
  );
}

function SettingsPanel({ runtime }: { runtime: VoiceRuntime }) {
  const { t } = useTranslation("studio");
  const { settings, saveSettings, refresh } = useVoice();
  return (
    <Panel title={t("voice.settings.title", { defaultValue: "Voice settings" })} className="span-2">
      <div className="voice-settings">
        <label className="checkbox-label">
          <input type="checkbox" checked={settings.autoSend} onChange={(e) => void saveSettings({ autoSend: e.target.checked })} />
          {t("voice.settings.autoSend", { defaultValue: "Send the message as soon as dictation finishes" })}
        </label>
        <label className="checkbox-label">
          <input type="checkbox" checked={settings.autoRead} onChange={(e) => void saveSettings({ autoRead: e.target.checked })} />
          {t("voice.settings.autoRead", { defaultValue: "Read every chat reply aloud when it finishes" })}
        </label>
        <label className="voice-settings__field">
          <span>{t("voice.settings.language", { defaultValue: "Dictation language" })}</span>
          <select className="text-input" value={settings.language} onChange={(e) => void saveSettings({ language: e.target.value })}>
            {LANGUAGES.map(([code, label]) => (
              <option key={code} value={code}>
                {label}
              </option>
            ))}
          </select>
          <small className="muted-text">
            {t("voice.settings.languageHint", { defaultValue: "Whisper uses it; Parakeet detects the language itself." })}
          </small>
        </label>
        <label className="voice-settings__field">
          <span>
            {t("voice.settings.speed", { defaultValue: "Reading speed" })} ({settings.speed.toFixed(1)}×)
          </span>
          <input
            type="range"
            min={0.5}
            max={2}
            step={0.1}
            value={settings.speed}
            onChange={(e) => void saveSettings({ speed: parseFloat(e.target.value) })}
          />
        </label>
        <label className="voice-settings__field">
          <span>{t("voice.settings.idle", { defaultValue: "Free voice memory after (minutes idle, 0 = never)" })}</span>
          <input
            type="number"
            className="text-input"
            min={0}
            max={240}
            value={settings.idleUnloadMinutes}
            onChange={(e) => void saveSettings({ idleUnloadMinutes: Math.max(0, Math.min(240, Number(e.target.value) || 0)) })}
          />
        </label>
        <div className="voice-settings__worker">
          <span className="muted-text">
            {runtime.workerPid
              ? t("voice.settings.workerRunning", { defaultValue: "Voice models are loaded (process {pid}).", pid: runtime.workerPid })
              : t("voice.settings.workerStopped", { defaultValue: "Voice models are not loaded." })}
          </span>
          {runtime.workerPid ? (
            <button type="button" className="secondary-button" onClick={() => void stopVoiceWorker().then(refresh)}>
              {t("voice.settings.unload", { defaultValue: "Unload now" })}
            </button>
          ) : null}
        </div>
      </div>
    </Panel>
  );
}

export function VoiceModelsTab({ voiceRuntime, backendOnline, onRefreshVoiceRuntime }: VoiceModelsTabProps) {
  const { t } = useTranslation("studio");
  const { settings, saveSettings } = useVoice();

  if (!backendOnline || !voiceRuntime) {
    return (
      <div className="content-grid discover-page voice-page">
        <Panel title={t("voice.models.title", { defaultValue: "Voice" })} className="span-2">
          <div className="empty-state">
            <p className="muted-text">{t("voice.offline", { defaultValue: "Waiting for the backend…" })}</p>
          </div>
        </Panel>
      </div>
    );
  }

  const selectedModel =
    voiceRuntime.sttModels.find((m) => m.id === settings.sttModel || m.key === settings.sttModel)?.id ??
    voiceRuntime.sttModels.find((m) => m.default)?.id;

  return (
    <div className="content-grid discover-page voice-page">
      <VoicePackPanel runtime={voiceRuntime} onChanged={onRefreshVoiceRuntime} />
      {voiceRuntime.installed ? (
        <>
          <Panel
            title={t("voice.models.sttTitle", { defaultValue: "Speech-to-text models" })}
            subtitle={t("voice.models.sttSubtitle", { defaultValue: "Used by the microphone in chat and by Voice Studio." })}
            className="span-2"
          >
            <div className="image-library-grid">
              {voiceRuntime.sttModels.map((model) => (
                <SttModelCard
                  key={model.id}
                  model={model}
                  selected={model.id === selectedModel}
                  onSelect={() => void saveSettings({ sttModel: model.key ?? model.id })}
                  onInstalled={onRefreshVoiceRuntime}
                />
              ))}
            </div>
          </Panel>
          <VoicesPanel voices={voiceRuntime.ttsVoices} />
          <SettingsPanel runtime={voiceRuntime} />
        </>
      ) : null}
    </div>
  );
}
