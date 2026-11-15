/**
 * "Try the sample project" (S12): the bundled four-shot film, offered on an empty Projects list. Opening it copies
 * the sample into the storyboards folder as an ordinary project (edit, redo, export or delete it like any other)
 * and opens it. Hidden when this install has no sample.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { fetchStoryboardSamplePosterUrl, getStoryboardSample, installStoryboardSample } from "../../api";
import type { StoryboardSampleInfo } from "../../api/storyboard";
import { VideoIcon } from "../../components/icons/SidebarIcons";
import { RESOLUTION_ALIAS } from "./formatLogic";

export interface SampleProjectCardProps {
  /** Opens the copied project. */
  onOpened: (projectId: string) => void;
}

export function SampleProjectCard({ onOpened }: SampleProjectCardProps) {
  const { t } = useTranslation("studio");
  const [sample, setSample] = useState<StoryboardSampleInfo | null>(null);
  const [poster, setPoster] = useState<string | null>(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    getStoryboardSample()
      .then((info) => live && setSample(info))
      .catch(() => live && setSample({ available: false }));
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    if (!sample?.poster) return;
    let live = true;
    let made: string | null = null;
    fetchStoryboardSamplePosterUrl()
      .then((url) => {
        if (live) {
          made = url;
          setPoster(url);
        } else URL.revokeObjectURL(url);
      })
      .catch(() => undefined);
    return () => {
      live = false;
      if (made) URL.revokeObjectURL(made);
    };
  }, [sample?.poster]);

  if (!sample?.available) return null;

  async function open() {
    setWorking(true);
    setError(null);
    try {
      const payload = await installStoryboardSample(t("board.sample.projectName", { defaultValue: "The Lighthouse" }));
      onOpened(payload.project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setWorking(false);
    }
  }

  const meta = t("board.sample.meta", {
    defaultValue: "{shots, plural, one {# shot} other {# shots}} · {seconds} s · {aspect} · {resolution}",
    shots: sample.shots ?? 0,
    seconds: Math.round(sample.seconds ?? 0),
    aspect: sample.aspect ?? "16:9",
    resolution: (RESOLUTION_ALIAS as Record<string, string | undefined>)[sample.resolution ?? ""] ?? sample.resolution ?? "",
  });

  return (
    <article className="sample-card" aria-labelledby="sample-card-title">
      <div className="sample-card__poster">
        {poster ? (
          <img src={poster} alt={t("board.sample.posterAlt", { defaultValue: "The first shot of the sample film" })} />
        ) : (
          <VideoIcon width={36} height={36} aria-hidden="true" />
        )}
      </div>
      <div className="sample-card__body">
        <span className="eyebrow">{t("board.sample.eyebrow", { defaultValue: "New here?" })}</span>
        <h3 id="sample-card-title">{t("board.sample.title", { defaultValue: "Try the sample project" })}</h3>
        <p>
          {t("board.sample.body", {
            defaultValue: "A finished four-shot film with its stills and clips, ready to play. Change a prompt, redo a shot or compare takes: it is an ordinary project, so you can delete it when you are done.",
          })}
        </p>
        <p className="sample-card__meta">{meta}</p>
        <div className="sample-card__actions">
          <button type="button" className="primary-button" disabled={working} aria-busy={working} onClick={() => void open()}>
            {working
              ? t("board.sample.opening", { defaultValue: "Copying the sample…" })
              : t("board.sample.open", { defaultValue: "Open the sample" })}
          </button>
        </div>
        {error ? (
          <div className="callout error" role="alert">
            <p>{t("board.sample.failed", { defaultValue: "The sample project could not be copied." })}</p>
            <details className="debug-details">
              <summary>{t("error.technicalDetails", { defaultValue: "Technical details" })}</summary>
              <pre>{error}</pre>
            </details>
          </div>
        ) : null}
      </div>
    </article>
  );
}
