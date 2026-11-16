/**
 * The song a music video is cut to (S-M2, on the S-M1 routes), on the Plan tab: attach one from the Music library or a
 * file; then its length, tempo and sections as a ruler with labels to rename, the rough-analysis note, "Make a cut
 * plan" (shots proposed from the sections and beats, reviewed before they replace the board's shots in one Undo),
 * the lyrics with "Time the lyrics", and Remove. Attaching and every edit go through the project's op queue, so Undo
 * and the revision check hold.
 */
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { attachStoryboardSong, proposeStoryboardCutPlan } from "../../../api/storyboard";
import type { StoryboardCutPlan, StoryboardProject } from "../../../types";
import type { UseStoryboardResult } from "../useStoryboard";
import { CutPlanReview } from "./CutPlanReview";
import { SongAttach } from "./SongAttach";
import { SongLyrics } from "./SongLyrics";
import { SongRuler } from "./SongRuler";
import { clock, relabel } from "./songLogic";
import "./song.css";

export function SongPanel({ story, project }: { story: UseStoryboardResult; project: StoryboardProject }) {
  const { t } = useTranslation("studio");
  const song = project.song ?? null;
  const [working, setWorking] = useState<"attach" | "cut" | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [plan, setPlan] = useState<StoryboardCutPlan | null>(null);
  const [changing, setChanging] = useState(false);
  const review = useRef<HTMLDivElement>(null);

  async function attach(source: { libraryId: string } | { path: string }) {
    if (!story.call) return;
    setWorking("attach");
    setProblem(null);
    setPlan(null);
    try {
      if (await story.call((id, revision) => attachStoryboardSong(id, { revision, ...source }))) setChanging(false);
    } catch (err) {
      setProblem(err instanceof Error ? err.message : String(err));
    } finally {
      setWorking(null);
    }
  }

  async function cut() {
    setWorking("cut");
    setProblem(null);
    try {
      setPlan(await proposeStoryboardCutPlan(project.id, {}));
      requestAnimationFrame(() => review.current?.querySelector<HTMLElement>("#cut-plan-title")?.focus());
    } catch (err) {
      setProblem(err instanceof Error ? err.message : String(err));
    } finally {
      setWorking(null);
    }
  }

  async function use() {
    if (!plan) return;
    if (await story.op("apply_plan", { mode: "replace", shots: plan.shots })) setPlan(null);
  }

  const busy = story.busy || working !== null;
  return (
    <section className="song-panel" aria-labelledby="song-panel-title">
      <div className="plan-section-head">
        <div>
          <h3 id="song-panel-title">{t("board.song.title", { defaultValue: "Song" })}</h3>
          <p className="muted-text">
            {song
              ? t("board.song.summary", {
                  defaultValue: "{title}: {time}{bpm, select, none {} other {, {bpm} BPM}}, {n, plural, one {# section} other {# sections}}",
                  title: song.title || t("board.song.untitled", { defaultValue: "Untitled" }),
                  time: clock(song.seconds),
                  bpm: song.bpm != null ? String(Math.round(song.bpm)) : "none",
                  n: song.sections.length,
                })
              : t("board.song.intro", { defaultValue: "For a music video: attach the song, and the shots can be cut to its sections and beats." })}
          </p>
        </div>
        {song && !changing ? (
          <div className="song-panel__head-actions">
            <button type="button" className="secondary-button plan-quiet-button" disabled={busy} onClick={() => setChanging(true)}>
              {t("board.song.change", { defaultValue: "Change song" })}
            </button>
            <button type="button" className="secondary-button plan-quiet-button" disabled={busy} onClick={() => void story.op("clear_song", {})}>
              {t("board.song.remove", { defaultValue: "Remove" })}
            </button>
          </div>
        ) : null}
      </div>

      {!song || changing ? (
        <>
          <SongAttach
            disabled={busy}
            onLibrary={(item) => void attach({ libraryId: item.id })}
            onFile={(path) => void attach({ path })}
          />
          {working === "attach" ? (
            <p className="muted-text" role="status">{t("board.song.analysing", { defaultValue: "Listening for the beat and the sections…" })}</p>
          ) : null}
          {changing ? (
            <button type="button" className="secondary-button plan-quiet-button" onClick={() => setChanging(false)}>
              {t("storyboard.cancel", { defaultValue: "Cancel" })}
            </button>
          ) : null}
        </>
      ) : (
        <>
          <SongRuler
            seconds={song.seconds}
            sections={song.sections}
            disabled={busy}
            onRename={(index, label) => {
              const sections = relabel(song.sections, index, label);
              if (sections) void story.op("edit_song", { sections });
            }}
          />
          {song.rough ? (
            <p className="muted-text song-panel__rough">
              {song.note || t("board.song.rough", { defaultValue: "Sections and tempo are estimates; check them by ear and rename or fix them here." })}
            </p>
          ) : null}
          <div className="song-panel__actions">
            <button type="button" className="primary-button" disabled={busy} onClick={() => void cut()}>
              {working === "cut" ? t("board.song.cutting", { defaultValue: "Cutting to the music…" }) : t("board.song.cutPlan", { defaultValue: "Make a cut plan" })}
            </button>
          </div>
          <div ref={review}>
            {plan ? <CutPlanReview plan={plan} project={project} busy={story.busy} onUse={() => void use()} onDiscard={() => setPlan(null)} /> : null}
          </div>
          <SongLyrics story={story} song={song} projectId={project.id} />
        </>
      )}
      {problem ? <p className="plan-error" role="alert">{problem}</p> : null}
    </section>
  );
}
