/**
 * Plays a rough cut (S9) in the app. Every file is fetched once (the project files route needs the API
 * token, so as blobs). Sound is decoded and scheduled with Web Audio, sample-exact against the audio
 * clock: the audio clips and each clip's own sound (at the shot's gain and speed), fades, music ducked
 * under voice. The audio clock is the film clock; the picture follows it: each shot has its own muted
 * <video> (or picture) layered on the stage, shown, faded and kept in step from an animation frame, with
 * small drift taken up by the playback rate and large drift by a seek. The picture is drawn late by the
 * audio output latency so what is seen matches what is heard.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiFetch } from "../../../api";
import { storyboardFilePath } from "../../../api/storyboard";
import {
  DUCK_GAIN,
  duckWindows,
  envelopePoints,
  mediaFiles,
  soundStart,
  sourceTimeAt,
  videoCorrection,
  visibleAt,
  type RoughCut,
} from "./roughCutLogic";

/** Seconds of lead before a scheduled start, so the first sample is never late. */
const START_LEAD_S = 0.06;
/** How far ahead an upcoming clip is cued to its first frame. */
const CUE_AHEAD_S = 1.5;
/** How often the readout (and anything else that renders) catches up with the clock while playing. */
const READOUT_MS = 100;

export interface RoughCutLoad {
  done: number;
  total: number;
  /** Project files that could not be read: their part plays blank or silent. */
  failed: string[];
}

export interface DriftStats {
  /** Largest |video time - wanted time| seen while playing, seconds. */
  maxS: number;
  /** Seeks made to catch up. */
  seeks: number;
  samples: number;
}

export interface UseRoughCutResult {
  ready: boolean;
  load: RoughCutLoad;
  playing: boolean;
  /** Film seconds, updated about ten times a second while playing. */
  timeS: number;
  /** Object URL of a project file once loaded. */
  urlOf: (rel: string) => string | null;
  /** The stage registers each shot's layer (a video, a picture or a card) by shot id. */
  registerLayer: (shotId: string, el: HTMLElement | null) => void;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (timeS: number) => void;
  /** No Web Audio here: the picture plays, the sound does not. */
  silent: boolean;
  drift: DriftStats;
}

type AudioCtor = typeof AudioContext;

function audioContextClass(): AudioCtor | null {
  const w = globalThis as unknown as { AudioContext?: AudioCtor; webkitAudioContext?: AudioCtor };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

async function fetchProjectFile(projectId: string, rel: string): Promise<Blob> {
  const response = await apiFetch(storyboardFilePath(projectId, rel), { method: "GET" });
  if (!response.ok) throw new Error(`${rel}: ${response.status}`);
  return await response.blob();
}

/** Run `work` over `items`, at most `limit` at a time. */
async function pool<T>(items: T[], limit: number, work: (item: T) => Promise<void>): Promise<void> {
  let next = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const item = items[next++];
      await work(item);
    }
  });
  await Promise.all(runners);
}

export function useRoughCut(projectId: string, cut: RoughCut): UseRoughCutResult {
  const files = useMemo(() => mediaFiles(cut), [cut]);
  const totalFiles = files.videos.length + files.stills.length + files.audio.length;
  const [load, setLoad] = useState<RoughCutLoad>({ done: 0, total: totalFiles, failed: [] });
  const [ready, setReady] = useState(totalFiles === 0);
  const [playing, setPlaying] = useState(false);
  const [timeS, setTimeS] = useState(0);
  const [drift, setDrift] = useState<DriftStats>({ maxS: 0, seeks: 0, samples: 0 });
  const urls = useRef(new Map<string, string>());
  const buffers = useRef(new Map<string, AudioBuffer>());
  const layers = useRef(new Map<string, HTMLElement>());
  const ctxRef = useRef<AudioContext | null>(null);
  const master = useRef<GainNode | null>(null);
  const running = useRef<AudioScheduledSourceNode[]>([]);
  const anchor = useRef<{ clock: number; film: number } | null>(null);
  const timeRef = useRef(0);
  const readoutAt = useRef(0);
  const driftRef = useRef<DriftStats>({ maxS: 0, seeks: 0, samples: 0 });
  const [, bump] = useState(0);

  // The audio context lives as long as the player.
  useEffect(() => {
    const Ctor = audioContextClass();
    if (!Ctor) return;
    const ctx = new Ctor();
    const gain = ctx.createGain();
    gain.connect(ctx.destination);
    ctxRef.current = ctx;
    master.current = gain;
    return () => {
      ctxRef.current = null;
      master.current = null;
      void ctx.close().catch(() => undefined);
    };
  }, []);

  // Fetch every file once; decode the sound of clips and audio files.
  useEffect(() => {
    let cancelled = false;
    const made: string[] = [];
    const failed: string[] = [];
    let done = 0;
    const jobs = [
      ...files.videos.map((rel) => ({ rel, url: true, decode: true })),
      ...files.stills.map((rel) => ({ rel, url: true, decode: false })),
      ...files.audio.map((rel) => ({ rel, url: false, decode: true })),
    ];
    void pool(jobs, 3, async (job) => {
      try {
        const blob = await fetchProjectFile(projectId, job.rel);
        if (cancelled) return;
        if (job.url) {
          const url = URL.createObjectURL(blob);
          made.push(url);
          urls.current.set(job.rel, url);
        }
        const ctx = ctxRef.current;
        if (job.decode && ctx) {
          try {
            buffers.current.set(job.rel, await ctx.decodeAudioData(await blob.arrayBuffer()));
          } catch {
            // A clip with no sound track: its shot is silent, which is what the export does too.
            if (!job.url) failed.push(job.rel);
          }
        }
      } catch {
        failed.push(job.rel);
      } finally {
        done += 1;
        if (!cancelled) setLoad({ done, total: jobs.length, failed: [...failed] });
      }
    }).then(() => {
      if (!cancelled) {
        setReady(true);
        bump((n) => n + 1);
      }
    });
    return () => {
      cancelled = true;
      made.forEach((url) => URL.revokeObjectURL(url));
      urls.current.clear();
      buffers.current.clear();
    };
  }, [projectId, files]);

  const latency = useCallback((): number => {
    const ctx = ctxRef.current as (AudioContext & { outputLatency?: number }) | null;
    if (!ctx) return 0;
    return (ctx.outputLatency || 0) + (ctx.baseLatency || 0);
  }, []);

  /** Film seconds now, as heard. */
  const clockNow = useCallback((): number => {
    const a = anchor.current;
    if (!a) return timeRef.current;
    const ctx = ctxRef.current;
    const now = ctx ? ctx.currentTime : performance.now() / 1000;
    return a.film + Math.max(0, now - a.clock);
  }, []);

  const stopSounds = useCallback(() => {
    for (const node of running.current) {
      try {
        node.stop();
        node.disconnect();
      } catch {
        // already stopped
      }
    }
    running.current = [];
  }, []);

  const scheduleSounds = useCallback(
    (from: number, startAt: number) => {
      const ctx = ctxRef.current;
      const out = master.current;
      if (!ctx || !out) return;
      const windows = duckWindows(cut);
      const at = (filmS: number) => startAt + Math.max(0, filmS - from);
      for (const sound of cut.sounds) {
        const buffer = buffers.current.get(sound.path);
        if (!buffer) continue;
        const start = soundStart(sound, from, buffer.duration);
        if (!start || start.offset >= buffer.duration - 1e-3) continue;
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.playbackRate.value = sound.rate;
        if (sound.loop) {
          source.loop = true;
          source.loopStart = Math.min(sound.offset, buffer.duration);
          source.loopEnd = buffer.duration;
        }
        const env = ctx.createGain();
        const points = envelopePoints(sound, from);
        env.gain.setValueAtTime((points[0]?.value ?? 1) * sound.gain, at(points[0]?.t ?? from));
        for (const point of points.slice(1)) env.gain.linearRampToValueAtTime(point.value * sound.gain, at(point.t));
        source.connect(env);
        if (sound.duck && windows.length) {
          const duck = ctx.createGain();
          const inside = windows.some(([a, b]) => from >= a && from < b);
          duck.gain.setValueAtTime(inside ? DUCK_GAIN : 1, startAt);
          for (const [a, b] of windows) {
            if (a > from) duck.gain.setTargetAtTime(DUCK_GAIN, at(a), 0.01);
            if (b > from) duck.gain.setTargetAtTime(1, at(b), 0.1);
          }
          env.connect(duck);
          duck.connect(out);
        } else {
          env.connect(out);
        }
        // `duration` counts source seconds; a sped-up shot uses more source per film second.
        source.start(startAt + start.delay, start.offset, sound.loop ? undefined : start.length * sound.rate);
        if (sound.loop) source.stop(startAt + start.delay + start.length);
        running.current.push(source);
      }
    },
    [cut],
  );

  /** Show, fade and step every layer for film time `t`. */
  const paint = useCallback(
    (t: number, isPlaying: boolean) => {
      const visible = new Map(visibleAt(cut, t).map((layer) => [layer.piece.shotId, layer.opacity]));
      for (const piece of cut.pieces) {
        const el = layers.current.get(piece.shotId);
        if (!el) continue;
        const opacity = visible.get(piece.shotId);
        el.style.opacity = opacity === undefined ? "0" : String(opacity);
        el.style.visibility = opacity === undefined ? "hidden" : "visible";
        if (!(el instanceof HTMLVideoElement)) continue;
        if (opacity === undefined) {
          if (!el.paused) el.pause();
          const upcoming = piece.start > t && piece.start - t < CUE_AHEAD_S;
          if (upcoming && Math.abs(el.currentTime - piece.sourceIn) > 0.05 && !el.seeking) el.currentTime = piece.sourceIn;
          continue;
        }
        const wanted = sourceTimeAt(piece, t);
        if (!isPlaying) {
          if (!el.paused) el.pause();
          if (Math.abs(el.currentTime - wanted) > 0.02 && !el.seeking) el.currentTime = wanted;
          continue;
        }
        if (el.paused) {
          el.currentTime = wanted;
          el.playbackRate = piece.speed;
          void el.play().catch(() => undefined);
          continue;
        }
        if (el.seeking) continue;
        const fix = videoCorrection(el.currentTime, wanted, piece.speed);
        const stats = driftRef.current;
        stats.samples += 1;
        stats.maxS = Math.max(stats.maxS, Math.abs(el.currentTime - wanted));
        if (fix.seek) {
          stats.seeks += 1;
          el.currentTime = wanted;
        }
        if (Math.abs(el.playbackRate - fix.rate) > 1e-3) el.playbackRate = fix.rate;
      }
    },
    [cut],
  );

  const pause = useCallback(() => {
    const t = Math.min(cut.total, clockNow());
    anchor.current = null;
    stopSounds();
    timeRef.current = t;
    setPlaying(false);
    setTimeS(t);
    paint(t, false);
  }, [cut.total, clockNow, paint, stopSounds]);

  const playFrom = useCallback(
    (from: number) => {
      const start = from >= cut.total - 0.05 ? 0 : Math.max(0, from);
      stopSounds();
      const ctx = ctxRef.current;
      if (ctx && ctx.state !== "running") void ctx.resume().catch(() => undefined);
      const clock = ctx ? ctx.currentTime + START_LEAD_S : performance.now() / 1000 + START_LEAD_S;
      anchor.current = { clock, film: start };
      timeRef.current = start;
      driftRef.current = { maxS: 0, seeks: 0, samples: 0 };
      scheduleSounds(start, clock);
      setPlaying(true);
      setTimeS(start);
    },
    [cut.total, scheduleSounds, stopSounds],
  );

  const play = useCallback(() => {
    if (!ready || cut.total <= 0) return;
    playFrom(timeRef.current);
  }, [ready, cut.total, playFrom]);

  const toggle = useCallback(() => {
    if (anchor.current) pause();
    else play();
  }, [pause, play]);

  const seek = useCallback(
    (to: number) => {
      const t = Math.max(0, Math.min(cut.total, to));
      if (anchor.current) {
        playFrom(t);
        return;
      }
      timeRef.current = t;
      setTimeS(t);
      paint(t, false);
    },
    [cut.total, paint, playFrom],
  );

  // The frame loop: picture follows the clock; the readout catches up a few times a second.
  useEffect(() => {
    if (!ready) return;
    let frame = 0;
    const tick = () => {
      if (anchor.current) {
        const heard = clockNow();
        const seen = Math.max(0, heard - latency());
        if (heard >= cut.total) {
          pause();
          timeRef.current = cut.total;
          setTimeS(cut.total);
        } else {
          timeRef.current = heard;
          paint(seen, true);
          const now = performance.now();
          if (now - readoutAt.current >= READOUT_MS) {
            readoutAt.current = now;
            setTimeS(heard);
            setDrift({ ...driftRef.current });
          }
        }
      }
      frame = requestAnimationFrame(tick);
    };
    paint(timeRef.current, false);
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [ready, cut.total, clockNow, latency, paint, pause]);

  // Stop the sound when the player goes away.
  useEffect(() => () => stopSounds(), [stopSounds]);

  const registerLayer = useCallback(
    (shotId: string, el: HTMLElement | null) => {
      if (el) {
        layers.current.set(shotId, el);
        if (el instanceof HTMLVideoElement) el.muted = true; // its sound comes through Web Audio
        if (!anchor.current) paint(timeRef.current, false); // while playing, the frame loop paints
      } else layers.current.delete(shotId);
    },
    [paint],
  );

  const urlOf = useCallback((rel: string) => urls.current.get(rel) ?? null, []);

  return {
    ready,
    load,
    playing,
    timeS,
    urlOf,
    registerLayer,
    play,
    pause,
    toggle,
    seek,
    silent: audioContextClass() === null,
    drift,
  };
}
