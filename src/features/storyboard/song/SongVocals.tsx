/**
 * Song to lip-sync shots (FU-371), under the song on the Plan tab: cut the song into sung shots of 3 to 5 s, then
 * give each shot its slice of the vocal (a stem if one is given, else the song), with a 0.4 s lead-in and the
 * singing switch on (FU-380). The slices go in as one op, so one Undo takes them all off.
 */
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { attachStoryboardSongVocals } from "../../../api/storyboard";
import type { StoryboardSongVocals } from "../../../types";
import { PathField } from "../PathField";
import type { UseStoryboardResult } from "../useStoryboard";

export const SUNG_SHOT_MIN_SECONDS = 3;
export const SUNG_SHOT_MAX_SECONDS = 5;

export interface SongVocalsProps {
  story: UseStoryboardResult;
  disabled: boolean;
  onCutForSinging: () => void;
}

export function SongVocals({ story, disabled, onCutForSinging }: SongVocalsProps) {
  const { t } = useTranslation("studio");
  const [stem, setStem] = useState<string | null>(null);
  const [working, setWorking] = useState(false);
  const [result, setResult] = useState<StoryboardSongVocals | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  async function attach() {
    if (!story.call) return;
    setWorking(true);
    setProblem(null);
    setResult(null);
    try {
      await story.call(async (id, revision) => {
        const out = await attachStoryboardSongVocals(id, { revision, ...(stem ? { stemPath: stem } : {}) });
        setResult(out.vocals);
        return out;
      });
    } catch (err) {
      setProblem(err instanceof Error ? err.message : String(err));
    } finally {
      setWorking(false);
    }
  }

  const busy = disabled || working;
  return (
    <div className="song-vocals">
      <h4>{t("board.song.vocals.title", { defaultValue: "Lip-sync the song" })}</h4>
      <p className="muted-text">
        {t("board.song.vocals.hint", {
          defaultValue:
            "1. Cut the song into sung shots of {min} to {max} s. 2. Give each shot its part of the vocal: it starts 0.4 s before the shot, is marked sung and makes 3 takes. A vocal stem (the voice alone) follows best; without one the song itself is used.",
          min: SUNG_SHOT_MIN_SECONDS,
          max: SUNG_SHOT_MAX_SECONDS,
        })}
      </p>
      <div className="song-panel__actions">
        <button type="button" className="secondary-button" disabled={busy} onClick={onCutForSinging}>
          {t("board.song.vocals.cut", { defaultValue: "Cut for singing ({min} to {max} s shots)", min: SUNG_SHOT_MIN_SECONDS, max: SUNG_SHOT_MAX_SECONDS })}
        </button>
      </div>
      {stem ? (
        <div className="lip-sync-picker__file">
          <code title={stem}>{stem}</code>
          <button type="button" className="secondary-button" disabled={busy} onClick={() => setStem(null)}>
            {t("board.song.vocals.stemRemove", { defaultValue: "Use the song instead" })}
          </button>
        </div>
      ) : (
        <PathField
          label={t("board.song.vocals.stem", { defaultValue: "Vocal stem (optional, WAV or FLAC)" })}
          buttonLabel={t("board.song.vocals.stemUse", { defaultValue: "Use this stem" })}
          extensions={["wav", "flac"]}
          disabled={busy}
          onSubmit={setStem}
        />
      )}
      <div className="song-panel__actions">
        <button type="button" className="primary-button" disabled={busy} onClick={() => void attach()}>
          {working
            ? t("board.song.vocals.working", { defaultValue: "Cutting the vocal…" })
            : t("board.song.vocals.attach", { defaultValue: "Give each shot its vocal" })}
        </button>
      </div>
      {result ? (
        <div role="status">
          <p className="muted-text">
            {t("board.song.vocals.done", {
              defaultValue:
                "{attached, plural, =0 {No shot got a vocal} one {# shot got its vocal} other {# shots got their vocal}} from the {source, select, stem {stem} other {song}}.",
              attached: result.attached,
              source: result.source,
            })}
          </p>
          {result.skipped.length ? (
            <ul className="muted-text">
              {result.skipped.map((item) => (
                <li key={item.shotId}>{item.reason}</li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
      {problem ? <p className="plan-error" role="alert">{problem}</p> : null}
    </div>
  );
}
