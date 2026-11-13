import type { ReactNode } from "react";

export function Section({  title,
  hint,
  action,
  children,
  className,
}: {
  title: string;
  hint?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={className ? `set-section ${className}` : "set-section"}>
      <div className="set-section-head">
        <div>
          <h3>{title}</h3>
          {hint && <p className="set-section-hint">{hint}</p>}
        </div>
        {action}
      </div>
      <div className="set-list">{children}</div>
    </section>
  );
}

/** A real toggle. The old row rendered the words "On" and "Off" as a button,
 *  which reads as a label until you notice it is clickable. */
export function Switch({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className={checked ? "set-switch is-on" : "set-switch"}
      onClick={onChange}
    >
      <span className="set-switch-knob" />
    </button>
  );
}

export function Row({
  label,
  hint,
  children,
  onClick,
  spec,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
  onClick?: () => void;
  /** Read-only fact: label and value line up in columns rather than the value
   *  being thrown to the right edge. */
  spec?: boolean;
}) {
  return (
    <div className={spec ? "set-row is-spec" : "set-row"} onClick={onClick}>
      <div className="set-row-text">
        <span>{label}</span>
        {hint && <small>{hint}</small>}
      </div>
      <div className="set-row-control">{children}</div>
    </div>
  );
}
