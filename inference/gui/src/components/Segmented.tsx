import { useRef } from "react";
import type { KeyboardEvent } from "react";

interface SegmentedOption {
  value: string;
  label: string;
}

/**
 * A small radiogroup styled as a segmented pill. Arrow keys move selection,
 * Home/End jump to the ends, and only the checked segment is in the tab order.
 */
export function Segmented({
  value,
  options,
  onChange,
  ariaLabel,
}: {
  value: string;
  options: SegmentedOption[];
  onChange: (value: string) => void;
  ariaLabel: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const selected = options.findIndex((option) => option.value === value);
  const current = selected >= 0 ? selected : 0;

  const choose = (index: number) => {
    const option = options[index];
    if (!option) return;
    onChange(option.value);
    window.requestAnimationFrame(() => refs.current[index]?.focus());
  };

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let next: number | null = null;
    switch (event.key) {
      case "ArrowRight":
      case "ArrowDown":
        next = (index + 1) % options.length;
        break;
      case "ArrowLeft":
      case "ArrowUp":
        next = (index - 1 + options.length) % options.length;
        break;
      case "Home":
        next = 0;
        break;
      case "End":
        next = options.length - 1;
        break;
      default:
        return;
    }
    event.preventDefault();
    choose(next);
  };

  return (
    <div className="segmented" role="radiogroup" aria-label={ariaLabel}>
      {options.map((option, index) => (
        <button
          key={option.value}
          ref={(node) => {
            refs.current[index] = node;
          }}
          type="button"
          role="radio"
          aria-checked={option.value === value}
          tabIndex={index === current ? 0 : -1}
          className="segmented-item"
          onClick={() => choose(index)}
          onKeyDown={(event) => onKeyDown(event, index)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
