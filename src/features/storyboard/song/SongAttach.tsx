/**
 * Choose the song for a project (S-M2): a track from the Music library (listed with its length), or an audio file on
 * this machine (typed or picked with the desktop file dialog). The caller attaches it; the backend checks the bytes
 * and analyses it.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { getMusicLibrary } from "../../../api/music";
import type { MusicLibraryItem } from "../../../types";
import { PathField } from "../PathField";
import { clock } from "./songLogic";

export const SONG_EXTENSIONS = ["wav", "flac", "mp3", "ogg", "m4a", "aac"];

export interface SongAttachProps {
  disabled: boolean;
  onLibrary: (item: MusicLibraryItem) => void;
  onFile: (path: string) => void;
}

export function SongAttach({ disabled, onLibrary, onFile }: SongAttachProps) {
  const { t } = useTranslation("studio");
  const [tracks, setTracks] = useState<MusicLibraryItem[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [choice, setChoice] = useState("");

  useEffect(() => {
    let live = true;
    getMusicLibrary()
      .then((list) => live && setTracks(list.items.filter((item) => item.kind === "track")))
      .catch(() => {
        if (live) {
          setTracks([]);
          setFailed(true);
        }
      });
    return () => {
      live = false;
    };
  }, []);

  const picked = tracks?.find((item) => item.id === choice) ?? null;
  return (
    <div className="song-attach">
      <div className="song-attach__library">
        <label className="eyebrow" htmlFor="song-library-choice">{t("board.song.fromLibrary", { defaultValue: "From the Music library" })}</label>
        {tracks === null ? (
          <p className="muted-text" role="status">{t("board.song.libraryLoading", { defaultValue: "Reading the Music library…" })}</p>
        ) : tracks.length === 0 ? (
          <p className="muted-text">
            {failed
              ? t("board.song.libraryFailed", { defaultValue: "The Music library could not be read. Choose a file instead." })
              : t("board.song.libraryEmpty", { defaultValue: "No tracks in the Music library yet. Make one in Music, or choose a file." })}
          </p>
        ) : (
          <div className="path-field__row">
            <select id="song-library-choice" className="text-input" value={choice} disabled={disabled} onChange={(event) => setChoice(event.target.value)}>
              <option value="">{t("board.song.pickTrack", { defaultValue: "Choose a track" })}</option>
              {tracks.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.durationS ? `${item.title || item.id} (${clock(item.durationS)})` : item.title || item.id}
                </option>
              ))}
            </select>
            <button type="button" className="primary-button" disabled={disabled || !picked} onClick={() => picked && onLibrary(picked)}>
              {t("board.song.attach", { defaultValue: "Attach" })}
            </button>
          </div>
        )}
      </div>
      <PathField
        label={t("board.song.fromFile", { defaultValue: "Or an audio file (WAV, FLAC, MP3, OGG, M4A)" })}
        buttonLabel={t("board.song.attach", { defaultValue: "Attach" })}
        extensions={SONG_EXTENSIONS}
        disabled={disabled}
        onSubmit={onFile}
      />
    </div>
  );
}
