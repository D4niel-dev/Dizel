import { useEffect, useRef, useState } from "react";
import type {
  CSSProperties,
  PointerEvent as ReactPointerEvent,
} from "react";
import { createPortal } from "react-dom";

import { Icon } from "./Icon";
import type { IconName } from "./Icon";
import { Picker } from "./Picker";
import {
  FAST_SPEEDS,
  THINKING_EFFORTS,
  effortById,
  modeColor,
  modeLabel,
  normalizeSpeed,
  speedById,
  type FastSpeed,
  type ThinkingEffort,
} from "../services/modes";

const MODE_ICONS: Record<string, IconName> = {
  Fast: "bolt",
  Planning: "listChecks",
  Coding: "code",
  Thinking: "brain",
};

type Panel = "effort" | "speed" | null;

interface ModePickerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  placement: "up" | "down";
  modes: string[];
  mode: string;
  effort: ThinkingEffort;
  speed: FastSpeed;
  onMode: (mode: string) => void;
  onEffort: (effort: ThinkingEffort) => void;
  onSpeed: (speed: FastSpeed) => void;
}

export function ModePicker({
  open,
  onOpenChange,
  placement,
  modes,
  mode,
  effort,
  speed,
  onMode,
  onEffort,
  onSpeed,
}: ModePickerProps) {
  const [panel, setPanel] = useState<Panel>(null);
  const [preview, setPreview] = useState<ThinkingEffort>(effort);
  const [speedPreview, setSpeedPreview] = useState<FastSpeed>(speed);
  const [dragging, setDragging] = useState(false);
  const [speedDragging, setSpeedDragging] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const arrowRef = useRef<HTMLButtonElement>(null);
  const rowRef = useRef<HTMLButtonElement>(null);
  const fastArrowRef = useRef<HTMLButtonElement>(null);
  const fastRowRef = useRef<HTMLButtonElement>(null);
  const scaleRef = useRef<HTMLDivElement>(null);
  const speedScaleRef = useRef<HTMLDivElement>(null);
  const closeTimer = useRef<number | null>(null);
  const pinnedRef = useRef(false);

  const clearClose = () => {
    if (closeTimer.current !== null) {
      window.clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  };
  const scheduleClose = () => {
    if (pinnedRef.current) return;
    clearClose();
    closeTimer.current = window.setTimeout(() => setPanel(null), 150);
  };
  const openPanel = (which: Exclude<Panel, null>, pin = false) => {
    clearClose();
    if (pin) pinnedRef.current = true;
    if (which === "effort") setPreview(effort);
    else setSpeedPreview(normalizeSpeed(speed));
    const anchor =
      which === "effort"
        ? arrowRef.current ?? rowRef.current
        : fastArrowRef.current ?? fastRowRef.current;
    if (!anchor) return;
    const rect = anchor.getBoundingClientRect();
    // Keep in step with `.effort-panel { width }` — the flyout is placed by hand,
    // so a width change here that misses the stylesheet pushes it off-screen.
    const width = 320;
    const height = 184;
    const left =
      window.innerWidth - rect.right >= width + 12
        ? rect.right + 8
        : Math.max(12, rect.left - width - 8);
    const top = Math.max(12, Math.min(rect.top - 8, window.innerHeight - height));
    setPosition({ top, left });
    setPanel(which);
  };
  const closePanel = () => {
    pinnedRef.current = false;
    setPanel(null);
  };

  useEffect(() => {
    if (!open) {
      pinnedRef.current = false;
      setPanel(null);
    }
  }, [open]);
  useEffect(() => () => clearClose(), []);

  const applyEffort = (next: ThinkingEffort) => {
    setPreview(next);
    onEffort(next);
    if (mode !== "Thinking") onMode("Thinking");
  };
  const applySpeed = (next: FastSpeed) => {
    setSpeedPreview(next);
    onSpeed(next);
    if (mode !== "Fast") onMode("Fast");
  };

  const previewInfo = effortById(preview);
  const speedInfo = speedById(speedPreview);
  const color = modeColor(mode, effort, normalizeSpeed(speed));

  const stopCount = THINKING_EFFORTS.length;
  const activeIndex = Math.max(
    0,
    THINKING_EFFORTS.findIndex((item) => item.id === preview),
  );
  const progress = activeIndex / (stopCount - 1);

  const speedCount = FAST_SPEEDS.length;
  const speedIndex = Math.max(
    0,
    FAST_SPEEDS.findIndex((item) => item.id === speedPreview),
  );
  const speedProgress = speedIndex / (speedCount - 1);

  // The outermost stop centres are measured rather than derived from the track:
  // each centre is inset by half a thumb, and this stays correct if that inset
  // changes in CSS. Shared by both sliders so drag behaves identically.
  const indexFromPointer = (
    clientX: number,
    scale: HTMLElement | null,
    count: number,
  ) => {
    if (!scale) return -1;
    const stops = scale.querySelectorAll<HTMLElement>(".effort-stop");
    if (stops.length < 2) return -1;
    const first = stops[0].getBoundingClientRect();
    const last = stops[stops.length - 1].getBoundingClientRect();
    const from = first.left + first.width / 2;
    const to = last.left + last.width / 2;
    if (to <= from) return -1;
    const ratio = Math.min(1, Math.max(0, (clientX - from) / (to - from)));
    return Math.min(count - 1, Math.max(0, Math.round(ratio * (count - 1))));
  };
  const chooseFromPointer = (clientX: number) => {
    const index = indexFromPointer(clientX, scaleRef.current, stopCount);
    if (index >= 0) setPreview(THINKING_EFFORTS[index].id);
  };
  const chooseSpeedFromPointer = (clientX: number) => {
    const index = indexFromPointer(clientX, speedScaleRef.current, speedCount);
    if (index >= 0) setSpeedPreview(FAST_SPEEDS[index].id);
  };
  const endDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragging) return;
    setDragging(false);
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      /* pointer capture is best-effort */
    }
    applyEffort(preview);
  };
  const endSpeedDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!speedDragging) return;
    setSpeedDragging(false);
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      /* pointer capture is best-effort */
    }
    applySpeed(speedPreview);
  };

  return (
    <>
      <Picker
        open={open}
        placement={placement}
        onOpenChange={onOpenChange}
        triggerClassName={mode !== "Fast" ? "bar-chip is-armed" : "bar-chip"}
        triggerTitle="Response mode"
        triggerStyle={{ color }}
        label={
          <>
            <Icon name={MODE_ICONS[mode] ?? "sparkles"} size={13} />
            <span>{modeLabel(mode, effort, normalizeSpeed(speed))}</span>
            <Icon name="chevronDown" size={11} />
          </>
        }
      >
        <div className="popover-label">Mode</div>
        {modes.map((name) => {
          const rowColor = modeColor(name, effort, normalizeSpeed(speed));
          const active = name === mode;
          const hasPanel = name === "Thinking" || name === "Fast";
          const tag =
            name === "Thinking"
              ? effortById(effort).label
              : name === "Fast" && normalizeSpeed(speed) !== "fast"
                ? speedById(normalizeSpeed(speed)).label
                : null;
          return (
            <div className="mode-row" key={name}>
              <button
                ref={name === "Thinking" ? rowRef : name === "Fast" ? fastRowRef : undefined}
                className={
                  active ? "popover-item mode-item active" : "popover-item mode-item"
                }
                onClick={() => {
                  setPanel(null);
                  onOpenChange(false);
                  onMode(name);
                }}
                onMouseEnter={
                  name === "Thinking"
                    ? () => openPanel("effort", false)
                    : name === "Fast"
                      ? () => openPanel("speed", false)
                      : undefined
                }
                onMouseLeave={hasPanel ? scheduleClose : undefined}
              >
                <span className="popover-item-main">
                  <Icon
                    name={MODE_ICONS[name] ?? "sparkles"}
                    size={13}
                    style={{ color: rowColor }}
                  />
                  <span style={{ color: active ? rowColor : undefined }}>{name}</span>
                  {tag && (
                    <span className="mode-effort-tag" style={{ color: rowColor }}>
                      {tag}
                    </span>
                  )}
                </span>
                {active && <Icon name="check" size={13} style={{ color: rowColor }} />}
              </button>
              {hasPanel && (
                <button
                  ref={name === "Thinking" ? arrowRef : fastArrowRef}
                  type="button"
                  className="mode-arrow"
                  aria-label={name === "Thinking" ? "Thinking effort" : "Fast speed"}
                  aria-expanded={panel === (name === "Thinking" ? "effort" : "speed")}
                  onClick={() => {
                    const which = name === "Thinking" ? "effort" : "speed";
                    if (panel === which && pinnedRef.current) {
                      closePanel();
                      return;
                    }
                    openPanel(which, true);
                  }}
                  onMouseEnter={() => openPanel(name === "Thinking" ? "effort" : "speed", false)}
                  onMouseLeave={scheduleClose}
                >
                  <Icon name="chevronRight" size={12} />
                </button>
              )}
            </div>
          );
        })}
      </Picker>

      {open &&
        panel === "effort" &&
        createPortal(
          <div
            className="effort-panel"
            data-picker-keep-open=""
            style={{ top: position.top, left: position.left }}
            role="group"
            aria-label="Thinking effort"
            onMouseEnter={clearClose}
            onMouseLeave={scheduleClose}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                closePanel();
              }
            }}
          >
            <div className="effort-head">
              <span>Thinking effort</span>
              <b style={{ color: previewInfo.color }}>{previewInfo.label}</b>
            </div>
            <div
              ref={scaleRef}
              className={dragging ? "effort-scale is-dragging" : "effort-scale"}
              style={{ "--progress": progress } as CSSProperties}
              role="slider"
              tabIndex={0}
              aria-valuemin={0}
              aria-valuemax={THINKING_EFFORTS.length - 1}
              aria-valuenow={THINKING_EFFORTS.findIndex((item) => item.id === preview)}
              aria-valuetext={previewInfo.label}
              onPointerDown={(event) => {
                setDragging(true);
                try {
                  event.currentTarget.setPointerCapture(event.pointerId);
                } catch {
                  // Synthetic pointers and some browsers have no active capture;
                  // pointermove still works from the element itself.
                }
                chooseFromPointer(event.clientX);
              }}
              onPointerMove={(event) => {
                if (dragging) chooseFromPointer(event.clientX);
              }}
              onPointerUp={endDrag}
              onKeyDown={(event) => {
                const index = THINKING_EFFORTS.findIndex(
                  (item) => item.id === preview,
                );
                if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
                  event.preventDefault();
                  applyEffort(THINKING_EFFORTS[Math.max(0, index - 1)].id);
                } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
                  event.preventDefault();
                  applyEffort(
                    THINKING_EFFORTS[
                      Math.min(THINKING_EFFORTS.length - 1, index + 1)
                    ].id,
                  );
                } else if (event.key === "Home") {
                  event.preventDefault();
                  applyEffort(THINKING_EFFORTS[0].id);
                } else if (event.key === "End") {
                  event.preventDefault();
                  applyEffort(THINKING_EFFORTS[THINKING_EFFORTS.length - 1].id);
                }
              }}
            >
              <div className="effort-line" aria-hidden="true" />
              <div
                className="effort-fill"
                aria-hidden="true"
                style={
                  {
                    "--fill-color": previewInfo.color,
                    opacity: activeIndex === 0 ? 0 : 1,
                  } as CSSProperties
                }
              />
              {THINKING_EFFORTS.map((item, index) => (
                <button
                  key={item.id}
                  type="button"
                  className={[
                    "effort-stop",
                    item.id === preview ? "active" : "",
                    index < activeIndex ? "is-passed" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  style={
                    {
                      "--stop-color": item.color,
                      "--p": index / (THINKING_EFFORTS.length - 1),
                    } as CSSProperties
                  }
                  aria-label={item.label}
                  aria-pressed={item.id === effort}
                  onClick={() => applyEffort(item.id)}
                />
              ))}
            </div>
            <div className="effort-names" aria-hidden="true">
              {THINKING_EFFORTS.map((item, index) => (
                <span
                  key={item.id}
                  className={item.id === preview ? "is-active" : ""}
                  style={
                    {
                      "--p": index / (THINKING_EFFORTS.length - 1),
                      color: item.id === preview ? item.color : undefined,
                    } as CSSProperties
                  }
                >
                  {item.label}
                </span>
              ))}
            </div>
            <p className="effort-note">{previewInfo.note}</p>
          </div>,
          document.body,
        )}

      {open &&
        panel === "speed" &&
        createPortal(
          <div
            className="effort-panel speed-panel"
            data-picker-keep-open=""
            style={{ top: position.top, left: position.left }}
            role="group"
            aria-label="Fast speed"
            onMouseEnter={clearClose}
            onMouseLeave={scheduleClose}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                closePanel();
              }
            }}
          >
            <div className="effort-head">
              <span>Reply speed</span>
              <b style={{ color: speedInfo.color }}>{speedInfo.label}</b>
            </div>
            <div
              ref={speedScaleRef}
              className={speedDragging ? "effort-scale is-dragging" : "effort-scale"}
              style={{ "--progress": speedProgress } as CSSProperties}
              role="slider"
              tabIndex={0}
              aria-valuemin={0}
              aria-valuemax={FAST_SPEEDS.length - 1}
              aria-valuenow={speedIndex}
              aria-valuetext={speedInfo.label}
              onPointerDown={(event) => {
                setSpeedDragging(true);
                try {
                  event.currentTarget.setPointerCapture(event.pointerId);
                } catch {
                  // Synthetic pointers and some browsers have no active capture;
                  // pointermove still works from the element itself.
                }
                chooseSpeedFromPointer(event.clientX);
              }}
              onPointerMove={(event) => {
                if (speedDragging) chooseSpeedFromPointer(event.clientX);
              }}
              onPointerUp={endSpeedDrag}
              onKeyDown={(event) => {
                if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
                  event.preventDefault();
                  applySpeed(FAST_SPEEDS[Math.max(0, speedIndex - 1)].id);
                } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
                  event.preventDefault();
                  applySpeed(
                    FAST_SPEEDS[Math.min(FAST_SPEEDS.length - 1, speedIndex + 1)].id,
                  );
                } else if (event.key === "Home") {
                  event.preventDefault();
                  applySpeed(FAST_SPEEDS[0].id);
                } else if (event.key === "End") {
                  event.preventDefault();
                  applySpeed(FAST_SPEEDS[FAST_SPEEDS.length - 1].id);
                }
              }}
            >
              <div className="effort-line" aria-hidden="true" />
              <div
                className="effort-fill"
                aria-hidden="true"
                style={
                  {
                    "--fill-color": speedInfo.color,
                    opacity: speedIndex === 0 ? 0 : 1,
                  } as CSSProperties
                }
              />
              {FAST_SPEEDS.map((item, index) => (
                <button
                  key={item.id}
                  type="button"
                  className={[
                    "effort-stop",
                    item.id === speedPreview ? "active" : "",
                    index < speedIndex ? "is-passed" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  style={
                    {
                      "--stop-color": item.color,
                      "--p": index / (FAST_SPEEDS.length - 1),
                    } as CSSProperties
                  }
                  aria-label={item.label}
                  aria-pressed={item.id === normalizeSpeed(speed)}
                  onClick={() => applySpeed(item.id)}
                />
              ))}
            </div>
            <div className="effort-names" aria-hidden="true">
              {FAST_SPEEDS.map((item, index) => (
                <span
                  key={item.id}
                  className={item.id === speedPreview ? "is-active" : ""}
                  style={
                    {
                      "--p": index / (FAST_SPEEDS.length - 1),
                      color: item.id === speedPreview ? item.color : undefined,
                    } as CSSProperties
                  }
                >
                  {item.label}
                </span>
              ))}
            </div>
            <p className="effort-note">
              {speedInfo.note} Higher speeds allow a larger reply budget and cost more
              tokens. Tokens per second is unchanged.
            </p>
          </div>,
          document.body,
        )}
    </>
  );
}
