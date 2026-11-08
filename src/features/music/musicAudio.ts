/**
 * Waveform peaks for a track: decode the file with the Web Audio API and reduce it to a fixed number of
 * bars. Everything is in a try/catch: a webview that cannot decode FLAC, a file that is too long to decode
 * comfortably, or a failed decode all give `null`, and the player draws one flat bar instead.
 */

export const PEAK_BUCKETS = 200;
/** Longer tracks are not decoded for a waveform (a 10 minute stereo track is about 230 MB of samples). */
export const MAX_PEAK_SECONDS = 300;

/** The largest absolute sample per bucket, scaled so the loudest bar is 1. Empty input gives no bars. */
export function peaksFromChannel(samples: Float32Array, buckets = PEAK_BUCKETS): number[] {
  if (samples.length === 0 || buckets <= 0) return [];
  const count = Math.min(buckets, samples.length);
  const out: number[] = new Array(count).fill(0);
  let loudest = 0;
  for (let bucket = 0; bucket < count; bucket += 1) {
    const from = Math.floor((bucket * samples.length) / count);
    const to = Math.max(from + 1, Math.floor(((bucket + 1) * samples.length) / count));
    let peak = 0;
    for (let i = from; i < to; i += 1) peak = Math.max(peak, Math.abs(samples[i]));
    out[bucket] = peak;
    loudest = Math.max(loudest, peak);
  }
  return loudest > 0 ? out.map((peak) => Math.round((peak / loudest) * 1000) / 1000) : out;
}

const cache = new Map<string, number[] | null>();
let chain: Promise<unknown> = Promise.resolve();

/** Peaks of `blob`, decoded one at a time and remembered by `id`; `null` when it cannot be done. */
export function computePeaks(id: string, blob: Blob, durationS: number): Promise<number[] | null> {
  if (cache.has(id)) return Promise.resolve(cache.get(id) ?? null);
  if (durationS > MAX_PEAK_SECONDS) return Promise.resolve(null);
  const run = async (): Promise<number[] | null> => {
    let context: AudioContext | null = null;
    try {
      const Ctor: typeof AudioContext | undefined =
        window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) return null;
      context = new Ctor();
      const decoded = await context.decodeAudioData(await blob.arrayBuffer());
      const peaks = peaksFromChannel(decoded.getChannelData(0));
      return peaks.length > 0 ? peaks : null;
    } catch {
      return null;
    } finally {
      try {
        await context?.close();
      } catch {
        // closing an already closed context is harmless
      }
    }
  };
  const next = chain.then(run, run).then((peaks) => {
    cache.set(id, peaks);
    return peaks;
  });
  chain = next.catch(() => undefined);
  return next;
}

export function forgetPeaks(id: string): void {
  cache.delete(id);
}
