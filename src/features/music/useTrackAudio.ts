/**
 * A track's audio for the player: the file is fetched with the bearer token and played through a blob URL
 * (revoked when the card goes away), with waveform peaks decoded on the side. `enabled` lets a Library card
 * wait until it is on screen before it loads anything.
 */
import { useEffect, useState } from "react";
import { fetchMusicAudioBlob } from "../../api";
import { computePeaks } from "./musicAudio";

export interface TrackAudio {
  url: string | null;
  peaks: number[] | undefined;
  failed: boolean;
}

export function useTrackAudio(id: string, durationS: number, enabled = true): TrackAudio {
  const [state, setState] = useState<TrackAudio>({ url: null, peaks: undefined, failed: false });

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let objectUrl: string | null = null;
    setState({ url: null, peaks: undefined, failed: false });
    fetchMusicAudioBlob(id)
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setState({ url: objectUrl, peaks: undefined, failed: false });
        void computePeaks(id, blob, durationS).then((peaks) => {
          if (!cancelled && peaks) setState((previous) => ({ ...previous, peaks }));
        });
      })
      .catch(() => {
        if (!cancelled) setState({ url: null, peaks: undefined, failed: true });
      });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [id, durationS, enabled]);

  return state;
}
