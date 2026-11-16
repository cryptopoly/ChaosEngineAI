/**
 * The song's lyrics and "Time the lyrics" (S-M2): one caption line per lyric line, saved with `edit_song` when the box
 * loses focus; timing from the song's sections, or by speech-to-text when it is installed here (the backend says so,
 * and why not). The captions land on the song's mixer clip, so burn-in and the .srt are the Export switches.
 */
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { getLyricsSpeechStatus, makeStoryboardLyricCaptions } from "../../../api/storyboard";
import type { StoryboardSong } from "../../../types";
import type { UseStoryboardResult } from "../useStoryboard";
import { lyricsToLines, sameLines } from "./songLogic";

type Timing = "auto" | "sections" | "speech";

export function SongLyrics({ story, song, projectId }: { story: UseStoryboardResult; song: StoryboardSong; projectId: string }) {
  const { t } = useTranslation("studio");
  const [text, setText] = useState(song.lyrics.join("\n"));
  const saved = useRef(song.lyrics);
  const [speech, setSpeech] = useState<{ ready: boolean; reason: string | null } | null>(null);
  const [timing, setTiming] = useState<Timing>("auto");
  const [working, setWorking] = useState(false);
  const [result, setResult] = useState<{ used: "sections" | "speech"; notes: string[] } | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    // Take lyrics that changed elsewhere (Undo, the agent), unless the box holds typing not saved yet.
    setText((current) => (sameLines(lyricsToLines(current), saved.current) ? song.lyrics.join("\n") : current));
    saved.current = song.lyrics;
  }, [song.lyrics]);

  useEffect(() => {
    let live = true;
    getLyricsSpeechStatus()
      .then((status) => live && setSpeech(status))
      .catch(() => live && setSpeech({ ready: false, reason: null }));
    return () => {
      live = false;
    };
  }, []);

  function save() {
    const lines = lyricsToLines(text);
    if (sameLines(lines, saved.current)) return;
    saved.current = lines;
    void story.op("edit_song", { lyrics: lines });
  }

  async function time() {
    if (!story.call) return;
    setWorking(true);
    setProblem(null);
    setResult(null);
    const answer: { used: "sections" | "speech"; notes: string[] }[] = [];
    try {
      await story.call(async (id, revision) => {
        const reply = await makeStoryboardLyricCaptions(id, { revision, timing });
        answer.push({ used: reply.timing, notes: reply.notes });
        return reply;
      });
      if (answer[0]) setResult(answer[0]);
    } catch (err) {
      // A 409 "no-stt" says why speech timing cannot run here; every other refusal is shown in its own words too.
      setProblem(err instanceof Error ? err.message : String(err));
    } finally {
      setWorking(false);
    }
  }

  const lines = lyricsToLines(text);
  const noClip = !song.clipId;
  return (
    <div className="song-lyrics">
      <label className="eyebrow" htmlFor={`song-lyrics-${projectId}`}>{t("board.song.lyrics", { defaultValue: "Lyrics" })}</label>
      <textarea
        id={`song-lyrics-${projectId}`}
        className="text-area song-lyrics__box"
        rows={5}
        value={text}
        placeholder={t("board.song.lyricsPlaceholder", { defaultValue: "One caption line per line. Leave out [verse] and other tags." })}
        onChange={(event) => setText(event.target.value)}
        onBlur={save}
      />
      <div className="song-lyrics__row">
        <label className="song-lyrics__timing">
          <span>{t("board.song.timing", { defaultValue: "Timing" })}</span>
          <select className="text-input" value={timing} onChange={(event) => setTiming(event.target.value as Timing)}>
            <option value="auto">{t("board.song.timingAuto", { defaultValue: "Automatic" })}</option>
            <option value="sections">{t("board.song.timingSections", { defaultValue: "From the sections" })}</option>
            <option value="speech" disabled={speech !== null && !speech.ready}>{t("board.song.timingSpeech", { defaultValue: "By ear (speech-to-text)" })}</option>
          </select>
        </label>
        <button type="button" className="secondary-button" disabled={working || story.busy || lines.length === 0 || noClip} onClick={() => void time()}>
          {working ? t("board.song.timingWorking", { defaultValue: "Timing the lyrics…" }) : t("board.song.timeLyrics", { defaultValue: "Time the lyrics" })}
        </button>
      </div>
      {speech && !speech.ready && speech.reason ? <p className="muted-text">{speech.reason}</p> : null}
      {noClip ? <p className="muted-text">{t("board.song.noClip", { defaultValue: "The song is not on the mixer, so there is no clip to put captions on. Attach it again with the mixer on." })}</p> : null}
      {problem ? <p className="plan-error" role="alert">{problem}</p> : null}
      {result ? (
        <div className="song-lyrics__done" role="status">
          <p>
            {result.used === "speech"
              ? t("board.song.timedSpeech", { defaultValue: "Timed by ear. The lines are captions on the song's clip in the Mixer; Undo takes them off." })
              : t("board.song.timedSections", { defaultValue: "Timed from the sections. The lines are captions on the song's clip in the Mixer; Undo takes them off." })}
          </p>
          {result.notes.map((note) => (
            <p key={note} className="muted-text">{note}</p>
          ))}
        </div>
      ) : null}
    </div>
  );
}
