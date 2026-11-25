/**
 * The lip-sync coverage table (audit change 3): for every shot its song time, length and state (synced, waiting for
 * its clip, not lip-synced, no vocal), the reason in plain words, and one button that fixes it with an action that
 * already exists. It is drawn from the project's `guide.coverage`, so it is the same after a tab switch or a reload.
 */
import { useTranslation } from "react-i18next";
import type { StoryboardLipSyncCoverage, StoryboardShot } from "../../../types";
import { fixAction, fixLabel, reasonText, rowFix, stateLabel, waitingShotIds, type FixAction, type Text } from "./coverageLogic";
import { clock } from "./songLogic";
import "./coverage.css";

export interface LipSyncCoverageProps {
  coverage: StoryboardLipSyncCoverage;
  shots: StoryboardShot[];
  /** An op or run is in flight: the buttons wait. */
  busy: boolean;
  /** A batch is running: the clip buttons wait for it. */
  batchRunning: boolean;
  /** Lengthen and Shorten open the shot editor; false hides them. */
  canOpenShot?: boolean;
  onFix: (action: FixAction) => void;
}

export function LipSyncCoverage({ coverage, shots, busy, batchRunning, canOpenShot = true, onFix }: LipSyncCoverageProps) {
  const { t } = useTranslation("studio");
  const say = (text: Text) => t(text.key, { defaultValue: text.defaultValue, ...text.values });
  const { counts } = coverage;
  const titles = new Map(shots.map((shot) => [shot.id, shot.title]));
  const waiting = waitingShotIds(coverage);

  return (
    <section className="lip-sync-coverage" id="lip-sync-coverage" aria-labelledby="lip-sync-coverage-title">
      <div className="lip-sync-coverage__head">
        <h4 id="lip-sync-coverage-title">
          {t("board.song.coverage.title", {
            defaultValue: "Lip sync: {sings} of {total, plural, one {# shot sings} other {# shots sing}}",
            sings: counts.sings,
            total: counts.total,
          })}
        </h4>
        {coverage.source ? (
          <span className="muted-text">
            {t("board.song.coverage.source", {
              defaultValue: "{source, select, stem {Vocal from the stem} other {Vocal from the song}}",
              source: coverage.source,
            })}
          </span>
        ) : null}
      </div>
      <p className="muted-text lip-sync-coverage__counts">
        {t("board.song.coverage.counts", {
          defaultValue: "{synced} synced · {waiting} waiting for a clip · {skipped} not lip-synced · {unassigned} without a vocal",
          synced: counts.synced,
          waiting: counts.waiting,
          skipped: counts.skipped,
          unassigned: counts.unassigned,
        })}
      </p>
      {coverage.shots.length === 0 ? (
        <p className="muted-text">{t("board.song.coverage.empty", { defaultValue: "No shots yet." })}</p>
      ) : (
        <div className="lip-sync-coverage__scroll">
          <table className="lip-sync-coverage__table">
            <thead>
              <tr>
                <th scope="col">{t("board.song.coverage.col.shot", { defaultValue: "Shot" })}</th>
                <th scope="col">{t("board.song.coverage.col.time", { defaultValue: "Song time" })}</th>
                <th scope="col">{t("board.song.coverage.col.length", { defaultValue: "Length" })}</th>
                <th scope="col">{t("board.song.coverage.col.state", { defaultValue: "Lip sync" })}</th>
              </tr>
            </thead>
            <tbody>
              {coverage.shots.map((row) => {
                const fix = rowFix(row, canOpenShot);
                const title = titles.get(row.shotId);
                const clipBlocked = fix === "make-lipsync" && batchRunning;
                return (
                  <tr key={row.shotId} className={`lip-sync-coverage__row lip-sync-coverage__row--${row.state}`}>
                    <th scope="row">
                      <span className="plan-number">{row.number}</span>
                      {title ? <span className="lip-sync-coverage__title">{title}</span> : null}
                    </th>
                    <td>{`${clock(row.songStart)} – ${clock(row.songEnd)}`}</td>
                    <td>{t("board.plan.seconds", { defaultValue: "{n} s", n: Math.round(row.seconds * 10) / 10 })}</td>
                    <td>
                      <span className={`lip-sync-coverage__state lip-sync-coverage__state--${row.state}`}>{say(stateLabel(row.state))}</span>
                      {row.code ? <span className="muted-text lip-sync-coverage__reason">{say(reasonText(row.code, row))}</span> : null}
                      {fix ? (
                        <button
                          type="button"
                          className="secondary-button lip-sync-coverage__fix"
                          disabled={busy || clipBlocked}
                          onClick={() => onFix(fixAction(fix, row.shotId))}
                        >
                          {say(fixLabel(fix, row))}
                        </button>
                      ) : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {waiting.length > 0 ? (
        <div className="song-panel__actions">
          <button
            type="button"
            className="primary-button"
            disabled={busy || batchRunning}
            onClick={() => onFix({ kind: "make-lipsync", shotIds: waiting })}
          >
            {t("board.song.coverage.makeAll", {
              defaultValue: "{n, plural, one {Make # lip-sync clip} other {Make # lip-sync clips}}",
              n: waiting.length,
            })}
          </button>
          {batchRunning ? (
            <span className="muted-text">{t("board.song.coverage.batchRunning", { defaultValue: "Waits for the shots being made now." })}</span>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
