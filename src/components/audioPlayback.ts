/**
 * One audio player plays at a time, app-wide. A player claims playback when
 * it starts; the previous owner is stopped. Same owner idea as
 * `speechPlayer` (read aloud), which is also stopped when a player starts.
 */
import { speechPlayer } from "../features/voice/speechPlayer";

interface Owner {
  id: string;
  stop: () => void;
}

let owner: Owner | null = null;

export function claimPlayback(id: string, stop: () => void): void {
  if (owner && owner.id !== id) owner.stop();
  owner = { id, stop };
  if (speechPlayer.getState().status !== "idle") speechPlayer.stop();
}

export function releasePlayback(id: string): void {
  if (owner?.id === id) owner = null;
}

export function playbackOwnerId(): string | null {
  return owner?.id ?? null;
}
