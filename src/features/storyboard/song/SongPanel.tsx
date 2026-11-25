/**
 * The song a music video is cut to (S-M2, on the S-M1 routes), on the Plan tab: attach one from the Music library or a
 * file; then its length, tempo and sections as a ruler with labels to rename, the rough-analysis note, the one
 * question "Does someone sing on screen?" (audit change 8) that leads to one cut: singing shots of 2 to 5 s followed by
 * each shot's vocal, or the ordinary cut to the beat. The cut is reviewed before it replaces the board's shots in one
 * Undo. Under it the lip-sync section with the coverage table (audit change 3), then the lyrics and Remove. Attaching
 * and every edit go through the project's op queue, so Undo and the revision check hold.
 */
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { attachStoryboardSong, attachStoryboardSongVocals, proposeStoryboardCutPlan } from "../../../api/storyboard";
import type { StoryboardCutPlan, StoryboardProject, StoryboardSongVocals } from "../../../types";
import type { UseStoryboardResult } from "../useStoryboard";
import { CutPlanReview } from "./CutPlanReview";
import { SingQuestion } from "./SingQuestion";
import { SongAttach } from "./SongAttach";
import { SongLyrics } from "./SongLyrics";
import { SongRuler } from "./SongRuler";
import { SongVocals } from "./SongVocals";
import { totalSeconds } from "../planLogic";
import { isBatchRunning } from "../storyboardLogic";
import { coverageInUse, cutOptions, initialSingAnswer, type FixAction, type SingAnswer } from "./coverageLogic";
import { clock, relabel } from "./songLogic";
import { useSongCoverage } from "./useSongCoverage";
import "./song.css";

export interface SongPanelProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  /** Open a shot in the shot editor, on its length (a coverage row's Lengthen or Shorten). The Plan tab always
   * passes it; without it those two buttons are not offered. */
  onOpenShot?: (shotId: string, field: "length") => void;
}

export function SongPanel({ story, project, onOpenShot }: SongPanelProps) {
  const { t } = useTranslation("studio");
  const song = project.song ?? null;
  const [working, setWorking] = useState<"attach" | "cut" | "vocals" | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [plan, setPlan] = useState<{ cut: StoryboardCutPlan; singing: boolean } | null>(null);
  const [changing, setChanging] = useState(false);
  const [answer, setAnswer] = useState<SingAnswer | null>(() => initialSingAnswer(project));
  const [stem, setStem] = useState<string | null>(null);
  const [vocals, setVocals] = useState<StoryboardSongVocals | null>(null);
  const [vocalsProblem, setVocalsProblem] = useState<string | null>(null);
  const { coverage, show } = useSongCoverage(project.id, project.revision, Boolean(song), story.guide);
  const review = useRef<HTMLDivElement>(null);
  const panel = useRef<HTMLElement>(null);

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
      setPlan({ cut: await proposeStoryboardCutPlan(project.id, cutOptions(answer)), singing: answer === "yes" });
      requestAnimationFrame(() => review.current?.querySelector<HTMLElement>("#cut-plan-title")?.focus());
    } catch (err) {
      setProblem(err instanceof Error ? err.message : String(err));
    } finally {
      setWorking(null);
    }
  }

  async function assignVocals() {
    if (!story.call) return;
    setWorking("vocals");
    setVocalsProblem(null);
    setVocals(null);
    try {
      await story.call(async (id, revision) => {
        const out = await attachStoryboardSongVocals(id, { revision, ...(stem ? { stemPath: stem } : {}) });
        setVocals(out.vocals);
        show(out.guide?.coverage ?? out.vocals.coverage, out.project.revision);
        return out;
      });
    } catch (err) {
      setVocalsProblem(err instanceof Error ? err.message : String(err));
    } finally {
      setWorking(null);
    }
  }

  async function use() {
    if (!plan) return;
    const singing = plan.singing;
    if (!(await story.op("apply_plan", { mode: "replace", shots: plan.cut.shots }))) return;
    setPlan(null);
    if (singing) await assignVocals();
  }

  function fix(action: FixAction) {
    switch (action.kind) {
      case "open-length":
        onOpenShot?.(action.shotId, "length");
        break;
      case "unlock":
        void story.op("lock", { shotId: action.shotId, locked: false });
        break;
      case "real-speed":
        void story.op("edit_shot", { shotId: action.shotId, speed: 1 });
        break;
      case "assign-vocals":
        void assignVocals();
        break;
      case "make-lipsync":
        void story.startBatch("both", action.shotIds);
        break;
    }
  }

  const busy = story.busy || working !== null;
  const lipSyncShown = answer === "yes" || coverageInUse(coverage);
  return (
    <section className="song-panel" aria-labelledby="song-panel-title" ref={panel}>
      <div className="plan-section-head">
        <div>
          <h3 id="song-panel-title" tabIndex={-1}>{t("board.song.title", { defaultValue: "Song" })}</h3>
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
              return sections ? story.op("edit_song", { sections }) : undefined;
            }}
          />
          {song.rough ? (
            <p className="muted-text song-panel__rough">
              {song.note || t("board.song.rough", { defaultValue: "Sections and tempo are estimates; check them by ear and rename or fix them here." })}
            </p>
          ) : null}
          <SingQuestion
            answer={answer}
            onAnswer={setAnswer}
            stem={stem}
            onStem={setStem}
            disabled={story.busy || working === "attach"}
            working={working === "cut" || working === "vocals" ? working : null}
            hasShots={project.shots.length > 0}
            hasTakes={project.shots.some((shot) => shot.takes.length > 0)}
            onCut={() => void cut()}
            onVocalsOnly={() => void assignVocals()}
          />
          <div ref={review}>
            {plan ? (
              <CutPlanReview
                plan={plan.cut}
                project={project}
                busy={story.busy}
                singing={plan.singing}
                onUse={() => void use()}
                onDiscard={() => setPlan(null)}
              />
            ) : null}
          </div>
          {lipSyncShown ? (
            <SongVocals
              coverage={coverage}
              shots={project.shots}
              result={vocals}
              problem={vocalsProblem}
              disabled={story.busy || (working !== null && working !== "vocals")}
              working={working === "vocals"}
              batchRunning={isBatchRunning(story.batch)}
              canOpenShot={Boolean(onOpenShot)}
              onAssign={() => void assignVocals()}
              onFix={fix}
            />
          ) : null}
          <SongLyrics
            story={story}
            song={song}
            projectId={project.id}
            filmSeconds={totalSeconds(project.shots)}
            onCutPlan={() =>
              answer ? void cut() : panel.current?.querySelector<HTMLElement>('input[name="sing-on-screen"]')?.focus()
            }
            cutDisabled={working !== null}
          />
        </>
      )}
      {problem ? <p className="plan-error" role="alert">{problem}</p> : null}
    </section>
  );
}
