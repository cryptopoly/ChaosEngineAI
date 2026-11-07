/**
 * Music > Studio: describe a track, pick a length, press Make track. The default view is the mode switch,
 * the description, the lyrics (songs only), the length and one action; the run estimate sits above the
 * action; tempo, key, variations and seed are in Options, and the model, the AI tag and the idle unload
 * are in Expert. States: offline, loading, no pack or model, ready, queued, running, done, error.
 */
import { useEffect, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import type { TabRequest } from "../../constants";
import { Disclosure } from "../../components/Disclosure";
import { LiveProgress, type LiveProgressPhase } from "../../components/LiveProgress";
import { FitBadge } from "../../components/models";
import { readFit } from "../../components/models/fit";
import { RunEstimateStrip } from "../../components/RunEstimateStrip";
import { StudioShell } from "../../components/StudioShell";
import { useBusyAdvice } from "../../hooks/useBusyAdvice";
import type { UseMusicState } from "../../hooks/useMusicState";
import type { MusicResult } from "../../types";
import { useSendToStoryboard } from "../storyboard/useSendToStoryboard";
import { findNamedArtist } from "./artistNames";
import {
  CAPTION_MAX,
  LENGTH_PRESETS,
  LYRICS_MAX,
  MUSIC_PHASE_IDS,
  MUSIC_PHASE_SECONDS,
  SECONDS_MAX,
  SECONDS_MIN,
  againForm,
  clampSeconds,
  formFromItem,
  isQueued,
  lengthPillLabel,
  shorterSeconds,
  stripFromEstimate,
  validateForm,
  varyForm,
  type FormIssue,
  type MusicForm,
  type MusicMode,
  type TFn,
} from "./musicLogic";
import { MusicExpertBody, MusicOptionsBody, expertSummary, optionsSummary } from "./MusicStudioOptions";
import { MusicResultCard } from "./MusicResultCard";
import { SoundsStudio } from "./SoundsStudio";

export interface MusicStudioTabProps {
  music: UseMusicState;
  backendOnline: boolean;
  onTabChange: (request: TabRequest) => void;
}

const MODES: MusicMode[] = ["song", "instrumental"];
const STUDIO_MODES = ["music", "sounds"] as const;

export function MusicStudioTab({ music, backendOnline, onTabChange }: MusicStudioTabProps) {
  const { t } = useTranslation("studio");
  const tf = t as unknown as TFn;
  const { studio, runtime, selectedVariant, installedVariants } = music;
  const { form, generation, running, progress, estimate } = studio;
  const storyboard = useSendToStoryboard((tab) => onTabChange(tab));
  const issues = validateForm(form);
  const artist = findNamedArtist(form.caption);
  const queued = isQueued(progress);
  const installed = Boolean(runtime?.installed);
  const ready = installed && music.hasInstalledModel && !runtime?.unavailableReason && Boolean(selectedVariant?.availableLocally);
  const canMake = ready && issues.length === 0 && !running;
  const strip = stripFromEstimate(estimate, progress?.queue ?? null, running);
  const busyAdvice = useBusyAdvice(strip.peakGb);
  const fit = readFit(estimate?.fit ?? selectedVariant?.fit);

  // Leaving a finished run's error up while the user edits is noise: it goes with the next edit.
  const { dismissError } = studio;
  const errorKind = generation.error?.kind;
  useEffect(() => {
    if (errorKind === "cancelled") {
      const timer = window.setTimeout(dismissError, 6000);
      return () => window.clearTimeout(timer);
    }
  }, [errorKind, dismissError]);

  function make() {
    if (canMake) void studio.generate();
  }

  function onKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      make();
    }
  }

  const studioMode = music.soundVariant ? music.studioMode : "music";
  const soundsMode = studioMode === "sounds";
  const modeLabels2: Record<(typeof STUDIO_MODES)[number], string> = {
    music: t("musicStudio.studioMode.music", { defaultValue: "Music" }),
    sounds: t("musicStudio.studioMode.sounds", { defaultValue: "Sounds" }),
  };
  const header = (
    <>
      <div className="music-studio__title">
        <h2>{t("musicStudio.title", { defaultValue: "Music Studio" })}</h2>
        {music.soundVariant ? (
          <div className="segmented music-studio__studio-modes" role="group" aria-label={t("musicStudio.studioMode.label", { defaultValue: "Music or sound effects" })}>
            {STUDIO_MODES.map((mode) => (
              <button
                key={mode}
                type="button"
                className={`segment${studioMode === mode ? " active" : ""}`}
                aria-pressed={studioMode === mode}
                disabled={running || music.sounds.running}
                onClick={() => music.setStudioMode(mode)}
              >
                {modeLabels2[mode]}
              </button>
            ))}
          </div>
        ) : null}
        {soundsMode ? (
          <div className="music-studio__status" role="status">
            <span className="music-studio__model">{music.soundVariant?.name}</span>
            {music.soundVariant?.licence?.attribution ? <span className="muted-text">{music.soundVariant.licence.attribution}</span> : null}
          </div>
        ) : (
        <div className="music-studio__status" role="status">
          {selectedVariant ? <span className="music-studio__model">{selectedVariant.name}</span> : null}
          {selectedVariant && ready ? (
            <span className="badge success">
              {runtime?.loadedModel && runtime.loadedModel === selectedVariant.engineModel
                ? t("musicStudio.status.loaded", { defaultValue: "Loaded" })
                : t("musicStudio.status.installed", { defaultValue: "Installed" })}
            </span>
          ) : null}
          {selectedVariant && !ready && backendOnline ? (
            <span className="badge muted">{t("musicStudio.status.notInstalled", { defaultValue: "Not installed" })}</span>
          ) : null}
          <FitBadge fit={fit} />
        </div>
        )}
      </div>
      <div className="button-row">
        <button type="button" className="secondary-button" onClick={() => onTabChange("music-models")}>
          {t("studioKit.models", { defaultValue: "Models" })}
        </button>
        <button type="button" className="secondary-button" onClick={() => onTabChange("music-library")}>
          {t("musicStudio.library", { defaultValue: "Library" })}
        </button>
      </div>
    </>
  );

  if (!backendOnline) {
    return (
      <div className="content-grid music-page">
        <section className="panel span-2 music-studio">
          <StudioShell
            studioId="music-studio"
            header={header}
            inputs={<p className="muted-text">{t("musicStudio.offline", { defaultValue: "Waiting for the backend…" })}</p>}
          />
        </section>
      </div>
    );
  }

  if (soundsMode) {
    return <SoundsStudio music={music} backendOnline={backendOnline} onTabChange={onTabChange} header={header} />;
  }

  // Not ready: one sentence and one button (nothing to fill in yet).
  if (!ready) {
    let message: string;
    if (!music.catalogLoaded || !runtime) message = t("musicStudio.loading", { defaultValue: "Loading the music models…" });
    else if (runtime.unavailableReason) message = runtime.unavailableReason;
    else if (!installed) {
      message = t("musicStudio.packMissing", { defaultValue: "The music pack is not installed yet. Install it from Music Models (about 1.6 GB)." });
    } else {
      message = t("musicStudio.noModel", { defaultValue: "No music model installed yet." });
    }
    const size = selectedVariant?.sizeGb;
    return (
      <div className="content-grid music-page">
        <section className="panel span-2 music-studio">
          <StudioShell
            studioId="music-studio"
            header={header}
            inputs={
              <div className="empty-state music-studio__empty">
                <p className="muted-text">{message}</p>
                {size && installed && !runtime?.unavailableReason ? (
                  <p className="muted-text">
                    {t("musicStudio.noModelSize", { defaultValue: "{name} is about {size} GB.", name: selectedVariant.name, size: size.toFixed(1) })}
                  </p>
                ) : null}
                {music.catalogLoaded && runtime ? (
                  <button type="button" className="primary-button" onClick={() => onTabChange("music-models")}>
                    {t("musicStudio.openModels", { defaultValue: "Open Music Models" })}
                  </button>
                ) : null}
              </div>
            }
          />
        </section>
      </div>
    );
  }

  // i18n-extract: musicStudio.progress.load
  // i18n-extract: musicStudio.progress.encode
  // i18n-extract: musicStudio.progress.compose
  // i18n-extract: musicStudio.progress.decode
  // i18n-extract: musicStudio.progress.save
  const phases: LiveProgressPhase[] = MUSIC_PHASE_IDS.map((id) => ({
    id,
    label: t(`musicStudio.progress.${id}`, { defaultValue: id }),
    estimatedSeconds: MUSIC_PHASE_SECONDS[id],
  }));
  const startedAt = generation.startedAt ?? Date.now();

  function runFrom(next: MusicForm) {
    studio.replaceForm(next);
    void studio.generate(next);
  }
  const again = (result: MusicResult) => runFrom(againForm({ ...formFromItem(result.item, selectedVariant), variations: form.variations }));
  const vary = (result: MusicResult) => runFrom(varyForm({ ...formFromItem(result.item, selectedVariant), variations: form.variations }, result.seed));

  const modeLabels: Record<MusicMode, string> = {
    song: t("musicStudio.mode.song", { defaultValue: "Song" }),
    instrumental: t("musicStudio.mode.instrumental", { defaultValue: "Instrumental" }),
  };

  const inputs = (
    <>
      <div className="segmented music-studio__modes" role="group" aria-label={t("musicStudio.mode.label", { defaultValue: "Kind of track" })}>
        {MODES.map((mode) => (
          <button
            key={mode}
            type="button"
            className={`segment${form.mode === mode ? " active" : ""}`}
            aria-pressed={form.mode === mode}
            disabled={running}
            onClick={() => studio.setForm({ mode })}
          >
            {modeLabels[mode]}
          </button>
        ))}
      </div>

      <label className="music-studio__field">
        <span className="music-studio__label">
          {t("musicStudio.prompt.label", { defaultValue: "Describe the music" })}
          <small className="muted-text">{form.caption.length} / {CAPTION_MAX}</small>
        </span>
        <textarea
          className="text-input"
          rows={2}
          maxLength={CAPTION_MAX}
          value={form.caption}
          disabled={running}
          onChange={(event) => studio.setForm({ caption: event.target.value })}
          placeholder={t("musicStudio.prompt.placeholder", { defaultValue: "warm lo-fi hip hop, vinyl crackle, mellow piano, 85 bpm" })}
        />
      </label>
      {artist ? (
        <div className="callout warning music-studio__artist" role="status">
          <p>{t("musicStudio.artistWarning", { defaultValue: "Naming an artist can produce a copy of their style. Describe the sound instead." })}</p>
        </div>
      ) : null}

      {form.mode === "song" ? (
        <label className="music-studio__field">
          <span className="music-studio__label">
            {t("musicStudio.lyrics.label", { defaultValue: "Lyrics" })}
            <small className="muted-text">{form.lyrics.length} / {LYRICS_MAX}</small>
          </span>
          <textarea
            className="text-input music-studio__lyrics"
            rows={6}
            maxLength={LYRICS_MAX}
            value={form.lyrics}
            disabled={running}
            onChange={(event) => studio.setForm({ lyrics: event.target.value })}
            placeholder={t("musicStudio.lyrics.placeholder", { defaultValue: "[verse] ... [chorus] ..." })}
          />
          <small className="muted-text">{t("musicStudio.lyrics.rights", { defaultValue: "You are responsible for rights in lyrics you type." })}</small>
        </label>
      ) : null}
    </>
  );

  const presets = (
    <div className="preset-row music-studio__length" role="group" aria-label={t("musicStudio.length.label", { defaultValue: "Length" })}>
      <span className="preset-row-label">{t("musicStudio.length.label", { defaultValue: "Length" })}</span>
      {LENGTH_PRESETS.map((seconds) => (
        <button
          key={seconds}
          type="button"
          className={`preset-pill${!form.customLength && form.seconds === seconds ? " active" : ""}`}
          aria-pressed={!form.customLength && form.seconds === seconds}
          disabled={running}
          onClick={() => studio.setForm({ seconds, customLength: false })}
        >
          <span className="preset-pill-label">{lengthPillLabel(tf, seconds)}</span>
        </button>
      ))}
      <button
        type="button"
        className={`preset-pill${form.customLength ? " active" : ""}`}
        aria-pressed={form.customLength}
        disabled={running}
        onClick={() => studio.setForm({ customLength: true })}
      >
        <span className="preset-pill-label">{t("musicStudio.length.custom", { defaultValue: "Custom" })}</span>
        <span className="preset-pill-sub">{t("musicStudio.length.range", { defaultValue: "{min} to {max} s", min: SECONDS_MIN, max: SECONDS_MAX })}</span>
      </button>
      {form.customLength ? (
        <input
          className="text-input music-studio__seconds"
          type="number"
          inputMode="numeric"
          min={SECONDS_MIN}
          max={SECONDS_MAX}
          aria-label={t("musicStudio.length.customLabel", { defaultValue: "Length in seconds" })}
          disabled={running}
          value={Number.isFinite(form.seconds) ? form.seconds : ""}
          onChange={(event) => studio.setForm({ seconds: event.target.value === "" ? Number.NaN : Math.round(Number(event.target.value)) })}
          onBlur={() => studio.setForm({ seconds: clampSeconds(form.seconds, selectedVariant) })}
        />
      ) : null}
    </div>
  );

  const hint: FormIssue | undefined = issues[0];
  const hintText: Record<FormIssue, string> = {
    captionEmpty: t("musicStudio.issue.captionEmpty", { defaultValue: "Describe the music to make a track." }),
    captionTooLong: t("musicStudio.issue.captionTooLong", { defaultValue: "The description is longer than {max} characters.", max: CAPTION_MAX }),
    lyricsEmpty: t("musicStudio.issue.lyricsEmpty", { defaultValue: "Add lyrics, or switch to Instrumental." }),
    lyricsTooLong: t("musicStudio.issue.lyricsTooLong", { defaultValue: "The lyrics are longer than {max} characters.", max: LYRICS_MAX }),
    secondsRange: t("musicStudio.issue.secondsRange", { defaultValue: "Length must be {min} to {max} seconds.", min: SECONDS_MIN, max: SECONDS_MAX }),
    bpmRange: t("musicStudio.issue.bpmRange", { defaultValue: "BPM must be 30 to 300, or Auto." }),
    seedRange: t("musicStudio.issue.seedRange", { defaultValue: "The seed must be a whole number from 0 up." }),
    variations: t("musicStudio.issue.variations", { defaultValue: "Variations must be 1, 2 or 4." }),
  };

  const estimateNode = (
    <>
      <RunEstimateStrip
        seconds={strip.seconds}
        peakGb={strip.peakGb}
        totalGb={strip.totalGb}
        diskMb={strip.diskMb}
        basis={strip.basis}
        risk={strip.risk}
        queue={strip.queue}
        busy={busyAdvice}
        action={
          <button type="button" className="primary-button" disabled={!canMake} onClick={make}>
            {running
              ? t("musicStudio.making", { defaultValue: "Making…" })
              : t("musicStudio.make", { defaultValue: "Make track" })}
          </button>
        }
      />
      {hint && !running ? <p className="muted-text music-studio__hint">{hintText[hint]}</p> : null}
      {running ? (
        <div className="music-studio__running">
          <LiveProgress
            title={t("musicStudio.running", { defaultValue: "Making your track" })}
            subtitle={progress?.runLabel ?? undefined}
            phases={phases}
            startedAt={startedAt}
            accent="audio"
            // Waiting behind another job: the snapshot may be that job's, so show only the queue line and estimates.
            realProgress={queued && progress ? { ...progress, active: false } : progress}
          />
          <div className="button-row">
            <button type="button" className="secondary-button" disabled={generation.cancelling} onClick={() => void studio.cancel()}>
              {generation.cancelling
                ? t("musicStudio.cancelling", { defaultValue: "Cancelling…" })
                : t("musicStudio.cancel", { defaultValue: "Cancel" })}
            </button>
          </div>
          {queued ? (
            <p className="muted-text" role="status">
              {t("musicStudio.queued", { defaultValue: "Queued: yours starts after the job ahead of it." })}
            </p>
          ) : null}
        </div>
      ) : null}

    </>
  );

  const options = (
    <Disclosure id="options" title={t("disclosure.options", { defaultValue: "Options" })} summary={optionsSummary(tf, form)}>
      <MusicOptionsBody form={form} onChange={studio.setForm} estimateSeconds={estimate?.estimate.seconds ?? null} disabled={running} />
    </Disclosure>
  );

  const expert = (
    <Disclosure
      id="expert"
      title={t("disclosure.expert", { defaultValue: "Expert" })}
      summary={expertSummary(tf, runtime, selectedVariant, installedVariants.length)}
    >
      <MusicExpertBody
        runtime={runtime}
        variants={installedVariants}
        selected={selectedVariant}
        onSelectModel={music.setSelectedModelId}
        onEmbedTag={(on) => void music.setEmbedTag(on)}
        onIdleMinutes={(minutes) => void music.setIdleMinutes(minutes)}
        disabled={running}
      />
    </Disclosure>
  );

  const error = generation.error;
  const headline: Record<NonNullable<typeof error>["kind"], string> = {
    oom: t("musicStudio.error.memory", { defaultValue: "There is not enough free memory for this track. Try a shorter length." }),
    notReady: t("musicStudio.error.notReady", { defaultValue: "The music model is not ready to run." }),
    cancelled: t("musicStudio.error.cancelled", { defaultValue: "Cancelled." }),
    offline: t("musicStudio.error.offline", { defaultValue: "The backend did not answer. Check that it is running, then try again." }),
    invalid: t("musicStudio.error.invalid", { defaultValue: "The track could not be started with these settings." }),
    generic: t("musicStudio.error.generic", { defaultValue: "The track could not be made." }),
  };

  const results = (
    <>
      {error && !running ? (
        <div className={`callout ${error.kind === "cancelled" ? "quiet" : "error"} music-studio__error`} role={error.kind === "cancelled" ? "status" : "alert"}>
          <p>{headline[error.kind]}</p>
          {error.detail && error.kind !== "generic" && error.kind !== "offline" ? <p className="muted-text">{error.detail}</p> : null}
          <div className="button-row">
            {error.kind === "oom" && form.seconds > SECONDS_MIN ? (
              <button
                type="button"
                className="secondary-button"
                onClick={() => {
                  studio.setForm({ seconds: shorterSeconds(form.seconds), customLength: false });
                  studio.dismissError();
                }}
              >
                {t("musicStudio.error.useShorter", { defaultValue: "Use a shorter length" })}
              </button>
            ) : null}
            {error.kind === "notReady" ? (
              <button type="button" className="secondary-button" onClick={() => onTabChange("music-models")}>
                {t("musicStudio.openModels", { defaultValue: "Open Music Models" })}
              </button>
            ) : null}
            {error.kind !== "cancelled" && generation.lastForm ? (
              <button type="button" className="secondary-button" disabled={!canMake} onClick={make}>
                {t("error.retry", { defaultValue: "Retry" })}
              </button>
            ) : null}
          </div>
          {error.kind !== "cancelled" ? (
            <details className="debug-details">
              <summary>{t("error.technicalDetails", { defaultValue: "Technical details" })}</summary>
              <pre className="install-log-terminal">{error.technical}</pre>
            </details>
          ) : null}
        </div>
      ) : null}

      <div className="music-studio__results-head">
        <h3>{t("musicStudio.latest", { defaultValue: "Latest" })}</h3>
        <button type="button" className="secondary-button" onClick={() => onTabChange("music-library")}>
          {t("musicStudio.openLibrary", { defaultValue: "Open Library" })}
        </button>
      </div>
      {generation.results.length === 0 && !running ? (
        <div className="empty-state">
          <p className="muted-text">{t("musicStudio.empty", { defaultValue: "No music yet. Describe a track and press Make track." })}</p>
        </div>
      ) : (
        <div className="music-studio__results" role="list">
          {generation.results.map((result) => (
            <div role="listitem" key={result.item.id}>
              <MusicResultCard
                result={result}
                libraryDir={runtime?.libraryDir ?? null}
                storyboard={storyboard}
                busy={running}
                onOpenLibrary={() => onTabChange("music-library")}
                onAgain={again}
                onVary={vary}
              />
            </div>
          ))}
        </div>
      )}
    </>
  );

  return (
    <div className="content-grid music-page">
      <section className="panel span-2 music-studio" onKeyDown={onKeyDown}>
        <StudioShell
          studioId="music-studio"
          header={header}
          inputs={inputs}
          presets={presets}
          estimate={estimateNode}
          options={options}
          expert={expert}
          results={results}
        />
      </section>
    </div>
  );
}
