import { useEffect, useRef } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react";

interface PickerProps {
  /** Content of the trigger button. */
  label: ReactNode;
  triggerClassName?: string;
  triggerTitle?: string;
  placement?: "up" | "down";
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: ReactNode;
}

/**
 * A trigger button plus a popover panel. Owns the outside-click and Escape
 * handling so callers only track a boolean.
 */
export function Picker({
  label,
  triggerClassName = "bar-btn",
  triggerTitle,
  placement = "up",
  open,
  onOpenChange,
  children,
}: PickerProps) {
  const ref = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;

    const onPointerDown = (event: MouseEvent) => {
      if (!ref.current?.contains(event.target as Node)) onOpenChange(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onOpenChange(false);
        triggerRef.current?.focus();
      }
    };

    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open, onOpenChange]);

  // Put keyboard focus inside the menu so ArrowUp/Down, Home and End work.
  useEffect(() => {
    if (!open) return;
    const frame = window.requestAnimationFrame(() => {
      ref.current
        ?.querySelector<HTMLButtonElement>(".popover-item:not(:disabled)")
        ?.focus();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [open]);

  const onMenuKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      onOpenChange(false);
      triggerRef.current?.focus();
      return;
    }
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;

    const items = Array.from(
      event.currentTarget.querySelectorAll<HTMLButtonElement>(
        ".popover-item:not(:disabled)",
      ),
    );
    if (items.length === 0) return;

    const current = items.indexOf(document.activeElement as HTMLButtonElement);
    let next = 0;
    if (event.key === "ArrowDown") {
      next = current < 0 ? 0 : (current + 1) % items.length;
    } else if (event.key === "ArrowUp") {
      next = current <= 0 ? items.length - 1 : current - 1;
    } else if (event.key === "End") {
      next = items.length - 1;
    }

    event.preventDefault();
    items[next]?.focus();
  };

  return (
    <div className="picker" ref={ref}>
      <button
        className={triggerClassName}
        title={triggerTitle}
        aria-label={triggerTitle}
        aria-haspopup="menu"
        aria-expanded={open}
        ref={triggerRef}
        onClick={() => onOpenChange(!open)}
      >
        {label}
      </button>

      {open && (
        <div
          className={`popover ${placement}`}
          role="menu"
          onKeyDown={onMenuKeyDown}
        >
          {children}
        </div>
      )}
    </div>
  );
}
