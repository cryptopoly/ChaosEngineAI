/**
 * The song's lip-sync section on the Plan tab (FU-371, audit change 3): the coverage table, kept with the project,
 * the last run's summary, and "Give each shot its vocal" to run the vocal cut again (a stem if one is given, else
 * the song; a 0.4 s lead-in and the singing switch on, FU-380). The slices go in as one op, so one Undo takes them
 * all off. Running it is the parent's job, since "Yes, lip-sync" chains it after the cut.
 */
import { useTranslation } from "react-i18next";
import type { StoryboardLipSyncCoverage, StoryboardShot, StoryboardSongVocals } from "../../../types";
import { LIP_SYNC_RANGE, type FixAction } from "./coverageLogic";
import { LipSyncCoverage } from "./LipSyncCoverage";

export interface SongVocalsProps {
  coverage: StoryboardLipSyncCoverage | null;
  shots: StoryboardShot[];
  /** The last run's answer in this session (the table itself comes from the project). */
  result: StoryboardSongVocals | null;
  problem: string | null;
  disabled: boolean;
  working: boolean;
  batchRunning: boolean;
  canOpenShot?: boolean;
  onAssign: () => void;
  onFix: (action: FixAction) => void;
}

export function SongVocals({ coverage, shots, result, problem, disabled, working, batchRunning, canOpenShot, onAssign, onFix }: SongVocalsProps) {
  const { t } = useTranslation("studio");
  const busy = disabled || working;
  const unassigned = coverage?.counts.unassigned ?? 0;
  return (
    <div className="song-vocals">
      <h4>{t("board.song.vocals.title", { defaultValue: "Lip-sync the song" })}</h4>
      <p className="muted-text">
        {t("board.song.vocals.hintEach", {
          defaultValue:
            "Each shot gets its part of the vocal: it starts 0.4 s before the shot, is marked sung and makes 3 takes.",
        })}{" "}
        {t(LIP_SYNC_RANGE.key, { defaultValue: LIP_SYNC_RANGE.defaultValue, ...LIP_SYNC_RANGE.values })}
      </p>
      {result ? (
        <p className="muted-text" role="status">
          {t("board.song.vocals.done", {
            defaultValue:
              "{attached, plural, =0 {No shot got a vocal} one {# shot got its vocal} other {# shots got their vocal}} from the {source, select, stem {stem} other {song}}.",
            attached: result.attached,
            source: result.source,
          })}
        </p>
      ) : null}
      {coverage ? (
        <LipSyncCoverage coverage={coverage} shots={shots} busy={busy} batchRunning={batchRunning} canOpenShot={canOpenShot} onFix={onFix} />
      ) : null}
      <div className="song-panel__actions">
        <button type="button" className="secondary-button" disabled={busy} onClick={onAssign}>
          {working
            ? t("board.song.vocals.working", { defaultValue: "Cutting the vocal…" })
            : t("board.song.vocals.attach", { defaultValue: "Give each shot its vocal" })}
        </button>
        {coverage && coverage.counts.sings > 0 ? (
          <span className="muted-text">
            {t("board.song.vocals.again", {
              defaultValue: "Runs again for every shot, including shots you took the vocal off.",
            })}
          </span>
        ) : unassigned > 0 ? (
          <span className="muted-text">
            {t("board.song.vocals.unassigned", {
              defaultValue: "{n, plural, one {# shot has no vocal yet.} other {# shots have no vocal yet.}}",
              n: unassigned,
            })}
          </span>
        ) : null}
      </div>
      {problem ? <p className="plan-error" role="alert">{problem}</p> : null}
    </div>
  );
}
