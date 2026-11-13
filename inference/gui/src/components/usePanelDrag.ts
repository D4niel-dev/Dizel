import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type {
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
} from "react";

const EDGE = 12;
const KEEP_VISIBLE = 148;
const MIN_TOP = 44;

function clampPosition(x: number, y: number, width: number) {
  const minX = EDGE - width + KEEP_VISIBLE;
  const maxX = Math.max(minX, window.innerWidth - KEEP_VISIBLE);
  const minY = MIN_TOP;
  const maxY = Math.max(minY, window.innerHeight - 56);
  return {
    x: Math.min(Math.max(x, minX), maxX),
    y: Math.min(Math.max(y, minY), maxY),
  };
}

interface DragState {
  pointerId: number;
  startX: number;
  startY: number;
  originX: number;
  originY: number;
  width: number;
}

/** Pointer and keyboard movement for a free-floating panel. */
export function usePanelDrag() {
  const ref = useRef<HTMLDivElement | null>(null);
  const drag = useRef<DragState | null>(null);
  const livePos = useRef<{ x: number; y: number } | null>(null);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (!pos) return;
    const onResize = () => {
      const el = ref.current;
      if (!el) return;
      const next = clampPosition(pos.x, pos.y, el.offsetWidth);
      livePos.current = next;
      setPos(next);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [pos]);

  // Pointer moves write directly to the element; React only commits on release.
  // Re-applying the live position after a render keeps unrelated store updates
  // from snapping the panel back to where the drag began.
  useLayoutEffect(() => {
    if (!dragging) return;
    const el = ref.current;
    const next = livePos.current;
    if (!el || !next) return;
    el.style.left = `${next.x}px`;
    el.style.top = `${next.y}px`;
  });

  const beginDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    if ((event.target as HTMLElement).closest("button")) return;
    const el = ref.current;
    if (!el) return;

    const box = el.getBoundingClientRect();
    const start = clampPosition(box.left, box.top, box.width);
    livePos.current = start;
    drag.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: start.x,
      originY: start.y,
      width: box.width,
    };

    // Switch to fixed before the first move so client coordinates and the
    // panel's own coordinate space agree immediately.
    el.classList.add("is-moved");
    el.style.left = `${start.x}px`;
    el.style.top = `${start.y}px`;
    setPos(start);
    setDragging(true);

    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      /* synthetic or already-released pointer */
    }
    event.preventDefault();
  };

  const moveDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    const state = drag.current;
    const el = ref.current;
    if (!state || !el || state.pointerId !== event.pointerId) return;

    const next = clampPosition(
      state.originX + event.clientX - state.startX,
      state.originY + event.clientY - state.startY,
      state.width,
    );
    livePos.current = next;
    el.style.left = `${next.x}px`;
    el.style.top = `${next.y}px`;
  };

  const stopDrag = () => {
    if (!drag.current) return;
    drag.current = null;
    setDragging(false);
    if (livePos.current) setPos(livePos.current);
  };

  const cancelDrag = () => {
    drag.current = null;
    livePos.current = null;
    setDragging(false);
  };

  const resetPosition = () => {
    const el = ref.current;
    if (el) {
      el.style.left = "";
      el.style.top = "";
      el.classList.remove("is-moved");
    }
    cancelDrag();
    setPos(null);
  };

  const nudge = (dx: number, dy: number) => {
    const el = ref.current;
    if (!el) return;
    const box = el.getBoundingClientRect();
    const base = livePos.current ?? { x: box.left, y: box.top };
    const next = clampPosition(base.x + dx, base.y + dy, box.width);
    livePos.current = next;
    setPos(next);
  };

  const onGripKeyDown = (event: ReactKeyboardEvent<HTMLSpanElement>) => {
    const step = event.shiftKey ? 48 : 16;
    if (event.key === "Home" || event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      resetPosition();
      return;
    }
    if (event.key === "ArrowLeft") nudge(-step, 0);
    else if (event.key === "ArrowRight") nudge(step, 0);
    else if (event.key === "ArrowUp") nudge(0, -step);
    else if (event.key === "ArrowDown") nudge(0, step);
    else return;
    event.preventDefault();
  };

  return {
    ref,
    pos,
    dragging,
    beginDrag,
    moveDrag,
    stopDrag,
    resetPosition,
    onGripKeyDown,
    cancelDrag,
  };
}
