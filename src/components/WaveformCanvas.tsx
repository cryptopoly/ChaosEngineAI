/**
 * Waveform drawn on a canvas from precomputed peaks (0 to 1). Decorative:
 * the canvas is `aria-hidden`; the player carries the text equivalent.
 * With no peaks it draws one flat bar. Colours are read from the design
 * tokens (`--muted`, `--accent`) at draw time.
 */
import { useEffect, useLayoutEffect, useRef } from "react";

export interface WaveformCanvasProps {
  peaks: number[];
  /** 0 to 1: the played part is drawn in the accent colour. */
  progress?: number;
  height?: number;
}

const BAR_WIDTH = 3;
const BAR_GAP = 2;
const useIsoLayoutEffect = typeof window !== "undefined" ? useLayoutEffect : useEffect;

/** Reduce (or stretch) `peaks` to exactly `count` bars: each bar is the max of its bucket. */
export function barsFor(peaks: number[], count: number): number[] {
  if (count <= 0) return [];
  if (peaks.length === 0) return Array.from({ length: count }, () => 0);
  return Array.from({ length: count }, (_, i) => {
    const from = Math.floor((i * peaks.length) / count);
    const to = Math.max(from + 1, Math.floor(((i + 1) * peaks.length) / count));
    let max = 0;
    for (let j = from; j < to && j < peaks.length; j += 1) max = Math.max(max, Math.abs(peaks[j]));
    return Math.min(1, max);
  });
}

export function WaveformCanvas({ peaks, progress = 0, height = 40 }: WaveformCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useIsoLayoutEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const cssWidth = Math.max(1, canvas.clientWidth || canvas.parentElement?.clientWidth || 1);
      canvas.width = Math.round(cssWidth * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.scale(dpr, dpr);
      const styles = getComputedStyle(canvas);
      const muted = styles.getPropertyValue("--muted").trim() || "#9ea8b4";
      const accent = styles.getPropertyValue("--accent").trim() || "#8fb4ff";
      const count = Math.max(1, Math.floor(cssWidth / (BAR_WIDTH + BAR_GAP)));
      const bars = barsFor(peaks, count);
      const played = Math.max(0, Math.min(1, progress)) * count;
      const mid = height / 2;
      bars.forEach((value, i) => {
        // A flat bar when there is nothing to show, never an empty canvas.
        const barHeight = Math.max(2, value * (height - 4));
        ctx.fillStyle = i < played ? accent : muted;
        ctx.fillRect(i * (BAR_WIDTH + BAR_GAP), mid - barHeight / 2, BAR_WIDTH, barHeight);
      });
    };
    draw();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [peaks, progress, height]);

  return <canvas ref={ref} className="waveform" aria-hidden="true" width={300} height={height} />;
}
