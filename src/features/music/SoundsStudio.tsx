/**
 * Music Studio > Sounds: describe a sound, pick a length (1 s up to the selected model's maximum: 11 s for Small,
 * 47 s for the full 1.0 model), press Make sound. The default view is the description, the length pills and
 * slider, a Loop switch and one action; the run estimate sits above the action; variations and seed are in
 * Options. Sounds come from Stable Audio Open, a gated model: until the licence is accepted, the model
 * downloaded and the sounds pack installed, the Studio says which step is next
 * in one sentence and opens Music and Audio Models. "Powered by Stability AI" is shown under the results, as the licence
 * asks. States: offline, loading, not ready, ready, queued, running, done, error.
 */
import { useEffect, type KeyboardEvent, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { TabRequest } from "../../constants";
import { AudioPlayer } from "../../components/AudioPlayer";
import { Disclosure } from "../../components/Disclosure";
import { LiveProgress, type LiveProgressPhase } from "../../components/LiveProgress";
import { RunEstimateStrip, formatDuration } from "../../components/RunEstimateStrip";
import { SendToStoryboardMenu } from "../../components/SendToStoryboardMenu";
import { StudioShell } from "../../components/StudioShell";
import { summaryLine } from "../../components/studioSummary";
import type { UseMusicState } from "../../hooks/useMusicState";
import type { MusicResult } from "../../types";
import { useSendToStoryboard, type UseSendToStoryboard } from "../storyboard/useSendToStoryboard";
import { clock, isQueued, itemFilePath, megabytes, stripFromEstimate } from "./musicLogic";
import { isLicenceAccepted } from "./musicLicence";
import {
  SOUND_CAPTION_MAX,
  SOUND_VARIATION_CHOICES,
  againSoundForm,
  soundBlocker,
  soundLimits,
  validateSoundForm,
  varySoundForm,
  soundFormFromItem,
  type SoundForm,
  type SoundIssue,
} from "./soundsLogic";
import { useTrackAudio } from "./useTrackAudio";

export interface SoundsStudioProps {
  music: UseMusicState;
  backendOnline: boolean;
  onTabChange: (request: TabRequest) => void;
  /** The Studio header (title, Music | Sounds switch, links), owned by `MusicStudioTab`. */
  header: ReactNode;
}

const SOUND_PHASE_IDS = ["load", "encode", "compose", "decode", "save"] as const;
const SOUND_PHASE_SECONDS: Record<(typeof SOUND_PHASE_IDS)[number], number> = { load: 20, encode: 1, compose: 3, decode: 2, save: 1 };
/** Rough seconds per phase; the compose step takes as long as the model's run estimate says (minutes for the full model). */
const COMPOSE_MIN_SECONDS = 3;

function SoundResultCard({
  result,
  libraryDir,
  storyboard,
  busy,
  onOpenLibrary,
  onAgain,
  onVary,
}: {
  result: MusicResult;
  libraryDir: string | null;
  storyboard: UseSendToStoryboard;
  busy: boolean;
  onOpenLibrary: () => void;
  onAgain: (result: MusicResult) => void;
  onVary: (result: MusicResult) => void;
}) {
  const { t } = useTranslation("studio");
  const { item } = result;
  const duration = item.durationS ?? 0;
  const audio = useTrackAudio(item.id, duration);
  const time = result.jobSeconds != null ? formatDuration(t as unknown as Parameters<typeof formatDuration>[0], result.jobSeconds) : null;
  const line = [
    time ? t("musicStudio.result.doneIn", { defaultValue: "Done in {time}", time }) : t("musicStudio.result.done", { defaultValue: "Done" }),
    megabytes(item.sizeBytes),
    t("musicStudio.result.seed", { defaultValue: "seed {seed}", seed: result.seed }),
  ]
    .filter(Boolean)
    .join(" · ");
  const send = { kind: "sfx" as const, path: itemFilePath(libraryDir, item), label: item.title, seconds: duration || null };

  return (
    <article className="music-result" aria-label={item.title}>
      <div className="music-result__head">
        <strong className="music-result__title">{item.title}</strong>
        <span className="muted-text">
          {item.loop ? `${t("musicStudio.sounds.loopBadge", { defaultValue: "Loop" })} · ` : ""}
          {clock(duration)}
        </span>
      </div>
      {audio.url ? (
        <AudioPlayer src={audio.url} durationS={duration} peaks={audio.peaks} loop={Boolean(item.loop)} label={item.title} />
      ) : (
        <p className="muted-text music-result__loading" role="status">
          {audio.failed
            ? t("musicStudio.result.audioFailed", { defaultValue: "The audio file could not be loaded." })
            : t("musicStudio.result.audioLoading", { defaultValue: "Loading the audio…" })}
        </p>
      )}
      <p className="muted-text music-result__line">{line}</p>
      <div className="button-row music-result__actions">
        <button type="button" className="secondary-button" onClick={onOpenLibrary}>
          {t("musicStudio.result.openLibrary", { defaultValue: "Open in Library" })}
        </button>
        <button type="button" className="secondary-button" disabled={busy} title={t("musicStudio.result.againTitle", { defaultValue: "Same settings, a new random seed." })} onClick={() => onAgain(result)}>
          {t("musicStudio.result.again", { defaultValue: "Again" })}
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={busy}
          title={t("musicStudio.result.varyTitle", { defaultValue: "Same settings, the next seed ({seed}).", seed: result.seed + 1 })}
          onClick={() => onVary(result)}
        >
          {t("musicStudio.result.vary", { defaultValue: "Vary" })}
        </button>
        <SendToStoryboardMenu
          item={send}
          projects={storyboard.projects}
          loaded={storyboard.loaded}
          busy={storyboard.busy}
          enabled={storyboard.enabled}
          notice={storyboard.notice}
          onOpen={() => void storyboard.loadProjects()}
          onSend={(projectId, target) => storyboard.send(send, projectId, target)}
          onDismissNotice={storyboard.dismissNotice}
        />
      </div>
    </article>
  );
}

export function SoundsStudio({ music, backendOnline, onTabChange, header }: SoundsStudioProps) {
  const { t } = useTranslation("studio");
  const { sounds: studio, runtime, soundVariant } = music;
  const { form, run, running, progress, estimate } = studio;
  const storyboard = useSendToStoryboard((tab) => onTabChange(tab));
  const issues = validateSoundForm(form, soundVariant);
  const limits = soundLimits(soundVariant);
  const queued = isQueued(progress);
  const licence = soundVariant?.licence ?? null;
  const blocker = soundBlocker({
    catalogLoaded: music.catalogLoaded && Boolean(runtime),
    variant: soundVariant,
    packInstalled: Boolean(runtime?.sound?.installed),
    unavailableReason: runtime?.sound?.unavailableReason ?? null,
    licenceAccepted: soundVariant && licence ? isLicenceAccepted(soundVariant.id, licence.id) : false,
  });
  const canMake = blocker === null && issues.length === 0 && !running;
  const strip = stripFromEstimate(estimate, progress?.queue ?? null, running);

  const { dismissError } = studio;
  const errorKind = run.error?.kind;
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

  const shell = (inner: Omit<Parameters<typeof StudioShell>[0], "studioId" | "header">) => (
    <div className="content-grid music-page">
      <section className="panel span-2 music-studio music-studio--sounds" onKeyDown={onKeyDown}>
        <StudioShell studioId="sounds-studio" header={header} {...inner} />
      </section>
    </div>
  );

  if (!backendOnline) {
    return shell({ inputs: <p className="muted-text">{t("musicStudio.offline", { defaultValue: "Waiting for the backend…" })}</p> });
  }

  if (blocker !== null) {
    const messages: Record<NonNullable<typeof blocker>, string> = {
      offline: t("musicStudio.offline", { defaultValue: "Waiting for the backend…" }),
      loading: t("musicStudio.loading", { defaultValue: "Loading the music models…" }),
      unsupported: runtime?.sound?.unavailableReason ?? "",
      licence: t("musicStudio.sounds.needLicence", {
        defaultValue: "Sound effects use Stable Audio Open, which asks you to accept its licence first. Open Music and Audio Models to read it.",
      }),
      download: t("musicStudio.sounds.needDownload", {
        defaultValue: "Download {name} from Music and Audio Models (about {size} GB). Accept the licence on Hugging Face first.",
        name: soundVariant?.name ?? "Stable Audio Open Small",
        size: (soundVariant?.sizeGb ?? 2.6).toFixed(1),
      }),
      textEncoder: t("musicStudio.sounds.needTextEncoder", { defaultValue: "The sound model is missing its text encoder. Download it again from Music and Audio Models." }),
      pack: t("musicStudio.sounds.needPack", { defaultValue: "Sounds also need a small pack (about 1 GB). Install it from Music and Audio Models." }),
    };
    return shell({
      inputs: (
        <div className="empty-state music-studio__empty">
          <p className="muted-text">{messages[blocker]}</p>
          {blocker !== "loading" && blocker !== "unsupported" ? (
            <button type="button" className="primary-button" onClick={() => onTabChange("music-models")}>
              {t("musicStudio.openModels", { defaultValue: "Open Music and Audio Models" })}
            </button>
          ) : null}
        </div>
      ),
    });
  }

  // i18n-extract: musicStudio.progress.load
  // i18n-extract: musicStudio.progress.encode
  // i18n-extract: musicStudio.progress.compose
  // i18n-extract: musicStudio.progress.decode
  // i18n-extract: musicStudio.progress.save
  const phases: LiveProgressPhase[] = SOUND_PHASE_IDS.map((id) => ({
    id,
    label: t(`musicStudio.progress.${id}`, { defaultValue: id }),
    estimatedSeconds: id === "compose" ? Math.max(COMPOSE_MIN_SECONDS, Math.round(estimate?.estimate.seconds ?? SOUND_PHASE_SECONDS.compose)) : SOUND_PHASE_SECONDS[id],
  }));

  function runFrom(next: SoundForm) {
    studio.replaceForm(next);
    void studio.generate(next);
  }
  const again = (result: MusicResult) => runFrom(againSoundForm({ ...soundFormFromItem(result.item), variations: form.variations }));
  const vary = (result: MusicResult) => runFrom(varySoundForm({ ...soundFormFromItem(result.item), variations: form.variations }, result.seed));

  const inputs = (
    <>
      <label className="music-studio__field">
        <span className="music-studio__label">
          {t("musicStudio.sounds.prompt", { defaultValue: "Describe the sound" })}
          <small className="muted-text">
            {form.caption.length} / {SOUND_CAPTION_MAX}
          </small>
        </span>
        <textarea
          className="text-input"
          rows={2}
          maxLength={SOUND_CAPTION_MAX}
          value={form.caption}
          disabled={running}
          onChange={(event) => studio.setForm({ caption: event.target.value })}
          placeholder={t("musicStudio.sounds.placeholder", { defaultValue: "heavy wooden door creaking open, close microphone" })}
        />
        <small className="muted-text">{t("musicStudio.sounds.english", { defaultValue: "English works best. Effects and loops, not songs or speech." })}</small>
      </label>
    </>
  );

  const presets = (
    <div className="preset-row music-studio__length" role="group" aria-label={t("musicStudio.length.label", { defaultValue: "Length" })}>
      <span className="preset-row-label">{t("musicStudio.length.label", { defaultValue: "Length" })}</span>
      <input
        type="range"
        className="music-studio__length-slider"
        aria-label={t("musicStudio.sounds.lengthSlider", { defaultValue: "Length in seconds" })}
        aria-valuetext={t("musicStudio.length.seconds", { defaultValue: "{n} s", n: form.seconds })}
        min={limits.min}
        max={limits.max}
        step={1}
        value={Math.min(limits.max, Math.max(limits.min, Math.round(form.seconds)))}
        disabled={running}
        onChange={(event) => studio.setForm({ seconds: Number(event.target.value) })}
      />
      <output className="music-studio__length-value">{t("musicStudio.length.seconds", { defaultValue: "{n} s", n: form.seconds })}</output>
      <label className="music-studio__loop">
        <input type="checkbox" checked={form.loop} disabled={running} onChange={(event) => studio.setForm({ loop: event.target.checked })} />
        <span>{t("musicStudio.sounds.loop", { defaultValue: "Make it a seamless loop" })}</span>
      </label>
    </div>
  );

  const hint: SoundIssue | undefined = issues[0];
  const hintText: Record<SoundIssue, string> = {
    captionEmpty: t("musicStudio.sounds.issue.captionEmpty", { defaultValue: "Describe the sound to make it." }),
    captionTooLong: t("musicStudio.sounds.issue.captionTooLong", { defaultValue: "The description is longer than {max} characters.", max: SOUND_CAPTION_MAX }),
    secondsRange: t("musicStudio.sounds.issue.secondsRange", { defaultValue: "Length must be {min} to {max} seconds.", min: limits.min, max: limits.max }),
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
        action={
          <button type="button" className="primary-button" disabled={!canMake} onClick={make}>
            {running ? t("musicStudio.sounds.making", { defaultValue: "Making…" }) : t("musicStudio.sounds.make", { defaultValue: "Make sound" })}
          </button>
        }
      />
      {hint && !running ? <p className="muted-text music-studio__hint">{hintText[hint]}</p> : null}
      {running ? (
        <div className="music-studio__running">
          <LiveProgress
            title={t("musicStudio.sounds.running", { defaultValue: "Making your sound" })}
            subtitle={progress?.runLabel ?? undefined}
            phases={phases}
            startedAt={run.startedAt ?? Date.now()}
            accent="audio"
            realProgress={queued && progress ? { ...progress, active: false } : progress}
          />
          {progress?.warning ? (
            <p className="callout warning" role="status">
              {progress.warning === "cpu-fallback"
                ? t("musicStudio.sounds.cpuFallback", {
                    defaultValue: "The Apple GPU could not run this model, so it restarted on the CPU. This is much slower: it can take many minutes. You can cancel.",
                  })
                : t("musicStudio.sounds.cpuOnly", {
                    defaultValue: "This sound is running on the CPU, not a GPU. This is much slower: it can take many minutes. You can cancel.",
                  })}
            </p>
          ) : null}
          <div className="button-row">
            <button type="button" className="secondary-button" disabled={run.cancelling} onClick={() => void studio.cancel()}>
              {run.cancelling ? t("musicStudio.cancelling", { defaultValue: "Cancelling…" }) : t("musicStudio.cancel", { defaultValue: "Cancel" })}
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
    <Disclosure
      id="options"
      title={t("disclosure.options", { defaultValue: "Options" })}
      summary={
        summaryLine([
          t("musicStudio.options.summaryVariations", { defaultValue: "{n, plural, one {# variation} other {# variations}}", n: form.variations }),
          form.seed === null
            ? t("musicStudio.options.randomSeed", { defaultValue: "Random seed" })
            : t("musicStudio.options.summarySeed", { defaultValue: "Seed {n}", n: form.seed }),
        ]) ?? ""
      }
    >
      <div className="music-options">
        <div className="music-options__group">
          <span className="preset-row-label">{t("musicStudio.options.variations", { defaultValue: "Variations" })}</span>
          <div className="preset-row">
            {SOUND_VARIATION_CHOICES.map((choice) => (
              <button
                key={choice}
                type="button"
                className={`preset-pill${form.variations === choice ? " active" : ""}`}
                aria-pressed={form.variations === choice}
                disabled={running}
                onClick={() => studio.setForm({ variations: choice })}
              >
                <span className="preset-pill-label">{choice}</span>
              </button>
            ))}
          </div>
        </div>
        <label className="music-options__field">
          <span>{t("musicStudio.options.seed", { defaultValue: "Seed" })}</span>
          <input
            className="text-input"
            type="number"
            inputMode="numeric"
            min={0}
            disabled={running}
            placeholder={t("musicStudio.options.random", { defaultValue: "Random" })}
            value={form.seed ?? ""}
            onChange={(event) => studio.setForm({ seed: event.target.value.trim() === "" ? null : Math.round(Number(event.target.value)) })}
          />
        </label>
      </div>
    </Disclosure>
  );

  const error = run.error;
  const headline: Record<NonNullable<typeof error>["kind"], string> = {
    oom: t("musicStudio.sounds.error.memory", { defaultValue: "There is not enough free memory for this sound. Free some memory and try again." }),
    notReady: t("musicStudio.error.notReady", { defaultValue: "The music model is not ready to run." }),
    cancelled: t("musicStudio.error.cancelled", { defaultValue: "Cancelled." }),
    offline: t("musicStudio.error.offline", { defaultValue: "The backend did not answer. Check that it is running, then try again." }),
    invalid: t("musicStudio.sounds.error.invalid", { defaultValue: "The sound could not be started with these settings." }),
    generic: t("musicStudio.sounds.error.generic", { defaultValue: "The sound could not be made." }),
  };

  const results = (
    <>
      {error && !running ? (
        <div className={`callout ${error.kind === "cancelled" ? "quiet" : "error"} music-studio__error`} role={error.kind === "cancelled" ? "status" : "alert"}>
          <p>{headline[error.kind]}</p>
          {error.detail && error.kind !== "generic" && error.kind !== "offline" ? <p className="muted-text">{error.detail}</p> : null}
          <div className="button-row">
            {error.kind === "notReady" ? (
              <button type="button" className="secondary-button" onClick={() => onTabChange("music-models")}>
                {t("musicStudio.openModels", { defaultValue: "Open Music and Audio Models" })}
              </button>
            ) : null}
            {error.kind !== "cancelled" && run.lastForm ? (
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
      {run.results.length === 0 && !running ? (
        <div className="empty-state">
          <p className="muted-text">{t("musicStudio.sounds.empty", { defaultValue: "No sounds yet. Describe one and press Make sound." })}</p>
        </div>
      ) : (
        <div className="music-studio__results" role="list">
          {run.results.map((result) => (
            <div role="listitem" key={result.item.id}>
              <SoundResultCard
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

  return shell({ inputs, presets, estimate: estimateNode, options, results });
}
