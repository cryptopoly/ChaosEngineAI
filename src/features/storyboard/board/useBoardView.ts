/**
 * Pan and zoom of the freestyle board (S-M2). The wheel pans; with Ctrl or Cmd (and a trackpad pinch, which the
 * browser sends as a Ctrl wheel) it zooms about the pointer. A drag on the empty board pans with a mouse, a pen or one
 * finger; two fingers pinch to zoom and pan together. The zoom buttons zoom about the middle, and Fit shows every card.
 */
import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type RefObject } from "react";
import { CARD_H, CARD_W, fitView, pinchStep, zoomAbout, type PinchState, type Pos, type View } from "./boardLogic";

export interface BoardViewControls {
  view: View;
  zoomBy: (factor: number) => void;
  fit: (positions: Pos[]) => void;
  reveal: (p: Pos) => void;
  /** Pointer handlers for the empty board (the caller sends card pointers elsewhere). */
  onPanStart: (event: ReactPointerEvent<HTMLElement>) => void;
  onPanMove: (event: ReactPointerEvent<HTMLElement>) => void;
  onPanEnd: (event: ReactPointerEvent<HTMLElement>) => void;
}

export function useBoardView(viewport: RefObject<HTMLElement | null>): BoardViewControls {
  const [view, setView] = useState<View>({ x: 24, y: 24, zoom: 1 });
  const viewRef = useRef(view);
  viewRef.current = view;
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef<PinchState | null>(null);

  const local = (clientX: number, clientY: number) => {
    const rect = viewport.current?.getBoundingClientRect();
    return { x: clientX - (rect?.left ?? 0), y: clientY - (rect?.top ?? 0) };
  };

  /** The first two fingers' midpoint (viewport pixels) and distance. */
  const pinchOf = (): PinchState => {
    const [a, b] = [...pointers.current.values()];
    return { mid: local((a.x + b.x) / 2, (a.y + b.y) / 2), distance: Math.hypot(a.x - b.x, a.y - b.y) };
  };

  useEffect(() => {
    const el = viewport.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = el.getBoundingClientRect();
      if (event.ctrlKey || event.metaKey) {
        const factor = Math.exp(-event.deltaY * 0.01);
        setView((v) => zoomAbout(v, factor, event.clientX - rect.left, event.clientY - rect.top));
      } else {
        setView((v) => ({ ...v, x: v.x - event.deltaX, y: v.y - event.deltaY }));
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [viewport]);

  const zoomBy = useCallback(
    (factor: number) => {
      const rect = viewport.current?.getBoundingClientRect();
      setView((v) => zoomAbout(v, factor, (rect?.width ?? 0) / 2, (rect?.height ?? 0) / 2));
    },
    [viewport],
  );

  const fit = useCallback(
    (positions: Pos[]) => {
      const rect = viewport.current?.getBoundingClientRect();
      setView(fitView(positions, rect?.width ?? 0, rect?.height ?? 0));
    },
    [viewport],
  );

  /** Pan just enough that a card at `p` is on screen (keyboard focus and moves). */
  const reveal = useCallback(
    (p: Pos) => {
      const rect = viewport.current?.getBoundingClientRect();
      if (!rect) return;
      setView((v) => {
        const margin = 16;
        const left = p.x * v.zoom + v.x;
        const top = p.y * v.zoom + v.y;
        const right = left + CARD_W * v.zoom;
        const bottom = top + CARD_H * v.zoom;
        let { x, y } = v;
        if (left < margin) x += margin - left;
        else if (right > rect.width - margin) x -= right - (rect.width - margin);
        if (top < margin) y += margin - top;
        else if (bottom > rect.height - margin) y -= bottom - (rect.height - margin);
        return x === v.x && y === v.y ? v : { ...v, x, y };
      });
    },
    [viewport],
  );

  const onPanStart = (event: ReactPointerEvent<HTMLElement>) => {
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    event.currentTarget.setPointerCapture?.(event.pointerId);
    if (pointers.current.size === 2) pinch.current = pinchOf();
  };

  const onPanMove = (event: ReactPointerEvent<HTMLElement>) => {
    const last = pointers.current.get(event.pointerId);
    if (!last) return;
    const now = { x: event.clientX, y: event.clientY };
    pointers.current.set(event.pointerId, now);
    if (pointers.current.size >= 2 && pinch.current) {
      const prev = pinch.current;
      const next = pinchOf();
      setView((v) => pinchStep(v, prev, next));
      pinch.current = next;
      return;
    }
    setView((v) => ({ ...v, x: v.x + now.x - last.x, y: v.y + now.y - last.y }));
  };

  const onPanEnd = (event: ReactPointerEvent<HTMLElement>) => {
    pointers.current.delete(event.pointerId);
    if (pointers.current.size < 2) pinch.current = null;
  };

  return { view, zoomBy, fit, reveal, onPanStart, onPanMove, onPanEnd };
}
