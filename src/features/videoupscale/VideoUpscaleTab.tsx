/**
 * The Upscale tab in the Video group (FU-617, fast tier): install the add-on on the first visit, then choose a
 * video (its size, frame rate, length, sound and field order are read with the probe), a quality row and a size,
 * start, and follow the job (cancel, resume, play or open the result). One job at a time; the newest jobs are
 * listed to look at again, and the last job's id is remembered so a restart shows it with Resume.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { FitBadge } from "../../components/models/FitBadge";
import {
  cancelVideoUpscaleJob,
  fetchVideoOutputBlobUrl,
  getVideoUpscaleAddonStatus,
  getVideoUpscaleJob,
  getVideoUpscaleModels,
  listVideoUpscaleJobs,
  probeVideoUpscaleInput,
  resumeVideoUpscaleJob,
  startVideoUpscale,
  startVideoUpscaleInstall,
  type VideoUpscaleAddonStatus,
  type VideoUpscaleJob,
  type VideoUpscaleModel,
  type VideoUpscaleProbe,
} from "../../api";
import { UpscaleInstallCard } from "./UpscaleInstallCard";
import { UpscaleJobCard } from "./UpscaleJobCard";
import { UpscaleRecentJobs } from "./UpscaleRecentJobs";
import { useUpscaleMedia } from "./useUpscaleMedia";
import {
  MAX_SIDE,
  SIZE_CHOICES,
  defaultModelId,
  defaultSizeChoice,
  deinterlaceDefault,
  durationText,
  fileNameOf,
  fpsText,
  isLive,
  missingModelIds,
  probeSize,
  qualityOption,
  showInstallCard,
  sizeOption,
  sizeRequest,
  type SizeChoice,
  type VideoSize,
} from "./upscaleLogic";
import { clearUpscaleRequest, peekUpscaleRequest, readLastJob, writeLastJob, type UpscaleSource } from "./upscaleSource";

const POLL_MS = 1500;
const VIDEO_EXTENSIONS = ["mp4", "mov", "m4v", "webm", "mkv", "avi", "mpg", "mpeg", "ts", "m2ts"];

export interface VideoUpscaleTabProps {
  onRevealPath: (path: string) => void;
  onOpenPath: (path: string) => void;
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** A playable URL of a Video Gallery clip (its file route needs the API token, so it comes as a blob URL). */
function useClipUrl(artifactId: string | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    setUrl(null);
    if (!artifactId) return undefined;
    let cancelled = false;
    let made: string | null = null;
    fetchVideoOutputBlobUrl(artifactId)
      .then((next) => {
        if (cancelled) {
          URL.revokeObjectURL(next);
          return;
        }
        made = next;
        setUrl(next);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      if (made) URL.revokeObjectURL(made);
    };
  }, [artifactId]);
  return url;
}

export function VideoUpscaleTab({ onRevealPath, onOpenPath }: VideoUpscaleTabProps) {
  const { t } = useTranslation("studio");
  const { t: tCommon } = useTranslation("common");
  const [addon, setAddon] = useState<VideoUpscaleAddonStatus | null>(null);
  const [addonError, setAddonError] = useState<string | null>(null);
  const [installError, setInstallError] = useState<string | null>(null);
  const [models, setModels] = useState<VideoUpscaleModel[]>([]);
  const [source, setSource] = useState<UpscaleSource | null>(() => peekUpscaleRequest());
  const [pathInput, setPathInput] = useState(() => peekUpscaleRequest()?.path ?? "");
  const [previewSize, setPreviewSize] = useState<VideoSize | null>(null);
  const [modelId, setModelId] = useState<string | null>(null);
  const [sizeChoice, setSizeChoice] = useState<SizeChoice>("2x");
  const [deinterlace, setDeinterlace] = useState(false);
  const [probe, setProbe] = useState<VideoUpscaleProbe | null>(null);
  const [probing, setProbing] = useState(false);
  const [probeError, setProbeError] = useState<string | null>(null);
  const [deinterlaceAvailable, setDeinterlaceAvailable] = useState(true);
  const [job, setJob] = useState<VideoUpscaleJob | null>(null);
  const [recent, setRecent] = useState<VideoUpscaleJob[]>([]);
  const [startError, setStartError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [pickError, setPickError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const installPoll = useRef<number | null>(null);

  // The probe's size is the file's own; a Gallery clip's size and the preview's size hold until it answers.
  const sourceSize: VideoSize | null = probeSize(probe)
    ?? (source?.width && source?.height ? { width: source.width, height: source.height } : previewSize);

  // The Gallery's request is taken once the tab shows it.
  useEffect(() => {
    clearUpscaleRequest();
  }, []);

  const loadModels = useCallback(() => {
    getVideoUpscaleModels()
      .then((next) => {
        setModels(next.models);
        setDeinterlaceAvailable(next.deinterlaceAvailable !== false);
        setModelId((current) => defaultModelId(next.models, current));
      })
      .catch(() => undefined);
  }, []);

  const loadRecent = useCallback(() => {
    listVideoUpscaleJobs().then(setRecent).catch(() => undefined);
  }, []);

  const stopInstallPoll = () => {
    if (installPoll.current != null) window.clearInterval(installPoll.current);
    installPoll.current = null;
  };

  const followInstall = useCallback(() => {
    stopInstallPoll();
    installPoll.current = window.setInterval(() => {
      getVideoUpscaleAddonStatus()
        .then((next) => {
          setAddon(next);
          if (!next.job.running) {
            stopInstallPoll();
            loadModels();
          }
        })
        .catch(stopInstallPoll);
    }, POLL_MS);
  }, [loadModels]);

  useEffect(() => {
    let cancelled = false;
    getVideoUpscaleAddonStatus()
      .then((next) => {
        if (cancelled) return;
        setAddon(next);
        if (next.job.running) followInstall();
      })
      .catch((err: unknown) => {
        if (!cancelled) setAddonError(message(err));
      });
    loadModels();
    const remembered = readLastJob();
    if (remembered) {
      getVideoUpscaleJob(remembered.id)
        .then((found) => {
          if (!cancelled) setJob(found);
        })
        .catch(() => writeLastJob(null)); // gone (finished and cleaned up, or never started)
    }
    return () => {
      cancelled = true;
      stopInstallPoll();
    };
  }, [followInstall, loadModels]);

  const install = (ids?: string[]) => {
    setInstallError(null);
    startVideoUpscaleInstall(ids)
      .then((next) => {
        setAddon(next);
        followInstall();
      })
      .catch((err: unknown) => setInstallError(message(err)));
  };

  // Follow a live job.
  const jobId = job?.id ?? null;
  const jobLive = job ? isLive(job.state) : false;
  useEffect(() => {
    if (!jobId || !jobLive) return undefined;
    const timer = window.setInterval(() => {
      getVideoUpscaleJob(jobId).then(setJob).catch(() => undefined);
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [jobId, jobLive]);

  // The list follows the job: a new one, and every change of state (done, failed, cancelled).
  const jobState = job?.state ?? null;
  useEffect(() => {
    loadRecent();
  }, [loadRecent, jobId, jobState]);

  const previewUrl = useClipUrl(source?.artifactId ?? null);
  const media = useUpscaleMedia(job);

  // Read the file's own size, frame rate, length, sound and field order whenever the path changes.
  const sourcePath = source?.path ?? null;
  useEffect(() => {
    setProbe(null);
    setProbeError(null);
    setDeinterlace(false);
    if (!sourcePath) {
      setProbing(false);
      return undefined;
    }
    let cancelled = false;
    setProbing(true);
    probeVideoUpscaleInput(sourcePath)
      .then((found) => {
        if (cancelled) return;
        setProbe(found);
        setDeinterlace(deinterlaceDefault(found));
      })
      .catch((err: unknown) => {
        if (!cancelled) setProbeError(message(err));
      })
      .finally(() => {
        if (!cancelled) setProbing(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sourcePath]);

  // Keep the size choice on something the new source allows.
  useEffect(() => {
    setSizeChoice((current) => (sizeOption(current, sourceSize).block ? defaultSizeChoice(sourceSize) : current));
  }, [sourceSize?.width, sourceSize?.height]); // eslint-disable-line react-hooks/exhaustive-deps

  const choosePath = (path: string, extra: Partial<UpscaleSource> = {}) => {
    const trimmed = path.trim();
    setPathInput(trimmed);
    setPreviewSize(null);
    setStartError(null);
    setSource(trimmed ? { path: trimmed, ...extra } : null);
  };

  const browse = async () => {
    setPickError(null);
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        multiple: false,
        title: t("videoUpscale.source.browseTitle", { defaultValue: "Choose a video to upscale" }),
        filters: [{ name: t("videoUpscale.source.filterName", { defaultValue: "Videos" }), extensions: VIDEO_EXTENSIONS }],
      });
      if (typeof selected === "string" && selected) choosePath(selected);
    } catch {
      setPickError(t("videoUpscale.source.browseFailed", {
        defaultValue: "The file picker is not available here. Paste the full path of the video instead.",
      }));
    }
  };

  const sizePart = sizeRequest(sizeChoice, sourceSize);
  const selectedModel = models.find((model) => model.id === modelId) ?? null;
  const modelOk = selectedModel ? qualityOption(selectedModel).selectable : false;
  const canStart = Boolean(source?.path) && modelOk && sizePart !== null && !jobLive && !busy && Boolean(addon?.installed);

  const start = () => {
    if (!source || !modelId || !sizePart) return;
    setBusy(true);
    setStartError(null);
    setActionError(null);
    startVideoUpscale({ inputPath: source.path, modelId, ...sizePart, deinterlace: deinterlace && deinterlaceAvailable })
      .then((started) => {
        setJob(started);
        writeLastJob({ id: started.id, source });
      })
      .catch((err: unknown) => setStartError(message(err)))
      .finally(() => setBusy(false));
  };

  const cancel = () => {
    if (!job) return;
    setBusy(true);
    setActionError(null);
    cancelVideoUpscaleJob(job.id)
      .then(() => getVideoUpscaleJob(job.id).then(setJob))
      .catch((err: unknown) => setActionError(message(err)))
      .finally(() => setBusy(false));
  };

  const resumeJob = (target: VideoUpscaleJob) => {
    setJob(target); // an error then shows on its card
    setBusy(true);
    setActionError(null);
    resumeVideoUpscaleJob(target.id)
      .then(setJob)
      .catch((err: unknown) => setActionError(message(err)))
      .finally(() => setBusy(false));
  };

  const inspect = (target: VideoUpscaleJob) => {
    setJob(target);
    setActionError(null);
  };

  const dismiss = () => {
    setJob(null);
    setActionError(null);
    writeLastJob(null);
  };

  const licence = useMemo(() => {
    const row = models.find((model) => model.default) ?? models[0];
    return row ? { licence: row.licence, note: row.licenceNote } : null;
  }, [models]);

  const blockText = (choice: SizeChoice): string | null => {
    const option = sizeOption(choice, sourceSize);
    if (option.block === "limit" && option.size) {
      return t("videoUpscale.size.limit", {
        defaultValue: "The result would be {width}x{height}; the limit is {max} pixels on a side. Choose a smaller scale or a smaller target size.",
        width: option.size.width, height: option.size.height, max: MAX_SIDE,
      });
    }
    if (option.block === "needsSize") {
      return t("videoUpscale.size.needsSize", {
        defaultValue: "The target sizes need the size of the video, which is read when you choose the file. Choose 2x or 4x until it has been read.",
      });
    }
    if (option.block === "notBigger") {
      return t("videoUpscale.size.notBigger", { defaultValue: "The video is already this tall or taller." });
    }
    return null;
  };
  const blockNotes = Array.from(new Set(SIZE_CHOICES.map(blockText).filter((note): note is string => Boolean(note))));
  const chosenSize = sizeOption(sizeChoice, sourceSize).size;
  const missing = addon ? missingModelIds(addon) : [];

  const sizeLabel = (choice: SizeChoice) =>
    choice === "2x"
      ? t("videoUpscale.size.x2", { defaultValue: "2x" })
      : choice === "4x"
        ? t("videoUpscale.size.x4", { defaultValue: "4x" })
        : t("videoUpscale.size.target", { defaultValue: "{height}p", height: choice.replace("p", "") });

  return (
    <div className="content-grid image-page-grid">
      <Panel
        title={tCommon("tabs.videoUpscale", { defaultValue: "Upscale" })}
        subtitle={tCommon("tabCaptions.videoUpscale", { defaultValue: "Make a video sharper and bigger" })}
        className="span-2 video-upscale"
      >
        {addonError ? (
          <p className="error-text">
            {t("videoUpscale.statusFailed", { defaultValue: "Could not read the upscaler status: {error}", error: addonError })}
          </p>
        ) : null}
        {addon && showInstallCard(addon) ? (
          <UpscaleInstallCard
            status={addon}
            licence={licence?.licence ?? null}
            licenceNote={licence?.note ?? null}
            requestError={installError}
            onInstall={() => install(missingModelIds(addon))}
          />
        ) : null}

        {job ? (
          <UpscaleJobCard
            job={job}
            modelLabel={models.find((model) => model.id === job.modelId)?.label ?? null}
            media={media}
            busy={busy}
            actionError={actionError}
            onCancel={cancel}
            onResume={() => resumeJob(job)}
            onReveal={() => onRevealPath(job.output)}
            onPlay={media.play}
            onOpen={() => onOpenPath(job.output)}
            onDismiss={dismiss}
          />
        ) : null}

        {addon?.installed ? (
          <div className="image-form-stack video-upscale-form">
            <div className="control-stack">
              <span className="eyebrow">{t("videoUpscale.source.label", { defaultValue: "Video" })}</span>
              <div className="video-upscale-path-row">
                <input
                  className="text-input"
                  type="text"
                  value={pathInput}
                  placeholder={t("videoUpscale.source.placeholder", { defaultValue: "Full path of a video file on this computer" })}
                  aria-label={t("videoUpscale.source.label", { defaultValue: "Video" })}
                  onChange={(event) => setPathInput(event.target.value)}
                  onBlur={() => {
                    if (pathInput.trim() !== (source?.path ?? "")) choosePath(pathInput);
                  }}
                />
                <button type="button" className="secondary-button" onClick={() => void browse()}>
                  {t("videoUpscale.source.browse", { defaultValue: "Browse…" })}
                </button>
              </div>
              <p className="muted-text">
                {t("videoUpscale.source.hint", {
                  defaultValue: "Or use Upscale this on a clip in the Video Gallery. The new film is saved in your video folder, under upscaled.",
                })}
              </p>
              {pickError ? <p className="error-text">{pickError}</p> : null}
              {previewUrl ? (
                <video
                  className="video-upscale-preview"
                  src={previewUrl}
                  preload="metadata"
                  muted
                  playsInline
                  controls
                  onLoadedMetadata={(event) => {
                    const video = event.currentTarget;
                    if (video.videoWidth > 0 && video.videoHeight > 0) setPreviewSize({ width: video.videoWidth, height: video.videoHeight });
                  }}
                />
              ) : null}
              {source ? (
                <p className="muted-text" title={source.path}>
                  {probe
                    ? t("videoUpscale.probe.line", {
                        defaultValue: "{name}: {width} × {height}, {fps} frames a second, {duration}, {audio}",
                        name: fileNameOf(source.path), width: probe.width, height: probe.height, fps: fpsText(probe.fps),
                        duration: durationText(probe.durationSeconds),
                        audio: probe.hasAudio
                          ? t("videoUpscale.probe.withSound", { defaultValue: "with sound" })
                          : t("videoUpscale.probe.noSound", { defaultValue: "no sound" }),
                      })
                    : sourceSize
                      ? t("videoUpscale.source.chosenSized", {
                          defaultValue: "{name}, {width} × {height}",
                          name: fileNameOf(source.path), width: sourceSize.width, height: sourceSize.height,
                        })
                      : fileNameOf(source.path)}
                  {probing ? ` ${t("videoUpscale.probe.reading", { defaultValue: "Reading the video…" })}` : ""}
                </p>
              ) : null}
              {probeError ? <p className="error-text">{probeError}</p> : null}
            </div>

            <div className="control-stack">
              <span className="eyebrow">{t("videoUpscale.quality.label", { defaultValue: "Quality" })}</span>
              <div className="image-pill-row" role="radiogroup" aria-label={t("videoUpscale.quality.label", { defaultValue: "Quality" })}>
                {models.map((model) => {
                  const option = qualityOption(model);
                  return (
                    <button
                      key={model.id}
                      type="button"
                      role="radio"
                      aria-checked={modelId === model.id}
                      className={`pill-button${modelId === model.id ? " active" : ""}`}
                      disabled={!option.selectable}
                      title={model.fit?.reason || undefined}
                      onClick={() => setModelId(model.id)}
                    >
                      <strong>{model.label}</strong>
                      <span>{model.name}</span>
                      {model.speedNote ? <span>{model.speedNote}</span> : null}
                      <FitBadge fit={model.fit} />
                      {!option.installed ? (
                        <span>{t("videoUpscale.quality.notInstalled", { defaultValue: "Not downloaded yet" })}</span>
                      ) : option.verdict === "wont-fit" ? (
                        <span>{t("videoUpscale.quality.wontFit", { defaultValue: "Needs more graphics memory than this computer has" })}</span>
                      ) : null}
                    </button>
                  );
                })}
              </div>
              {selectedModel?.notes ? <p className="muted-text">{selectedModel.notes}</p> : null}
              {missing.length > 0 && !addon.job.running ? (
                <div className="button-row">
                  <button type="button" className="secondary-button" onClick={() => install(missing)}>
                    {t("videoUpscale.quality.downloadMissing", { defaultValue: "Download the missing models" })}
                  </button>
                </div>
              ) : null}
            </div>

            <div className="control-stack">
              <span className="eyebrow">{t("videoUpscale.size.label", { defaultValue: "Size" })}</span>
              <div className="image-pill-row" role="radiogroup" aria-label={t("videoUpscale.size.label", { defaultValue: "Size" })}>
                {SIZE_CHOICES.map((choice) => {
                  const option = sizeOption(choice, sourceSize);
                  return (
                    <button
                      key={choice}
                      type="button"
                      role="radio"
                      aria-checked={sizeChoice === choice}
                      className={`pill-button${sizeChoice === choice ? " active" : ""}`}
                      disabled={option.block !== null}
                      title={blockText(choice) ?? undefined}
                      onClick={() => setSizeChoice(choice)}
                    >
                      <strong>{sizeLabel(choice)}</strong>
                      {option.size ? <span>{option.size.width} × {option.size.height}</span> : null}
                    </button>
                  );
                })}
              </div>
              {chosenSize ? (
                <p className="muted-text">
                  {t("videoUpscale.size.result", { defaultValue: "The result will be {width} × {height}.", width: chosenSize.width, height: chosenSize.height })}
                </p>
              ) : null}
              {blockNotes.map((note) => <p key={note} className="muted-text">{note}</p>)}
            </div>

            <div className="control-stack">
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={deinterlace && deinterlaceAvailable}
                  disabled={!deinterlaceAvailable}
                  onChange={(event) => setDeinterlace(event.target.checked)}
                />
                <span>
                  {t("videoUpscale.deinterlace", {
                    defaultValue: "Deinterlace first (for TV or tape recordings that show comb-like lines on movement)",
                  })}
                </span>
              </label>
              {!deinterlaceAvailable ? (
                <p className="muted-text">
                  {t("videoUpscale.deinterlaceUnavailable", {
                    defaultValue: "Deinterlacing is not available: the video tools on this computer cannot do it.",
                  })}
                </p>
              ) : probe?.interlaced === true ? (
                <p className="muted-text">
                  {t("videoUpscale.probe.interlaced", { defaultValue: "This video looks interlaced, so the box is ticked." })}
                </p>
              ) : null}
            </div>

            <div className="button-row">
              <button type="button" className="primary-button" disabled={!canStart} onClick={start}>
                {t("videoUpscale.start", { defaultValue: "Start upscaling" })}
              </button>
              {jobLive ? (
                <span className="muted-text">{t("videoUpscale.oneAtATime", { defaultValue: "One upscale at a time: wait for this one or cancel it." })}</span>
              ) : null}
            </div>
            {startError ? <p className="error-text">{startError}</p> : null}
          </div>
        ) : null}

        <UpscaleRecentJobs jobs={recent} selectedId={jobId} resumeDisabled={busy || jobLive} onSelect={inspect} onResume={resumeJob} />
      </Panel>
    </div>
  );
}
