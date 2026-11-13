import { useEffect, useId, useRef, useState } from "react";

import { Icon } from "./Icon";
import type { FeedbackRating } from "../types";

const OPTIONS: Record<FeedbackRating, string[]> = {
  up: ["Accurate", "Clear", "Helpful", "Well formatted"],
  down: ["Incorrect", "Incomplete", "Off-topic", "Unclear", "Too long"],
};

export function FeedbackDialog({
  rating,
  reasons,
  note,
  hasFeedback,
  onSave,
  onSkip,
  onRemove,
}: {
  rating: FeedbackRating;
  reasons: string[];
  note: string;
  hasFeedback: boolean;
  onSave: (reasons: string[], note: string) => void;
  onSkip: () => void;
  onRemove: () => void;
}) {
  const [selected, setSelected] = useState(reasons);
  const [draft, setDraft] = useState(note);
  const noteRef = useRef<HTMLTextAreaElement>(null);
  const skipRef = useRef(onSkip);
  const titleId = useId();
  const noteId = useId();

  useEffect(() => {
    skipRef.current = onSkip;
  }, [onSkip]);

  useEffect(() => {
    noteRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") skipRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const toggle = (option: string) => {
    setSelected((current) =>
      current.includes(option)
        ? current.filter((item) => item !== option)
        : [...current, option],
    );
  };

  const positive = rating === "up";

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => event.target === event.currentTarget && onSkip()}
    >
      <div
        className={`modal feedback-modal ${positive ? "is-up" : "is-down"}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
      >
        <div className="feedback-head">
          <span className="feedback-mark" aria-hidden="true">
            <Icon name={positive ? "thumbUp" : "thumbDown"} size={18} />
          </span>
          <div className="feedback-heading">
            <h2 id={titleId}>{positive ? "Good response" : "What went wrong?"}</h2>
            <p>
              The rating is saved locally. Add a reason or note if you want — it&apos;s
              optional.
            </p>
          </div>
          <button className="modal-x" onClick={onSkip} aria-label="Close feedback">
            <Icon name="close" size={13} />
          </button>
        </div>

        <div className="feedback-body">
          <div className="feedback-chips" role="group" aria-label="Reason">
            {OPTIONS[rating].map((option) => (
              <button
                key={option}
                type="button"
                className="feedback-chip"
                aria-pressed={selected.includes(option)}
                onClick={() => toggle(option)}
              >
                {option}
              </button>
            ))}
          </div>

          <label className="feedback-note" htmlFor={noteId}>
            <span>
              {positive ? "Anything else that was good?" : "What was wrong?"}
              <em>optional</em>
            </span>
            <textarea
              ref={noteRef}
              id={noteId}
              value={draft}
              maxLength={600}
              rows={3}
              onChange={(event) => setDraft(event.target.value)}
              placeholder={
                positive
                  ? "e.g. The code example was correct and easy to run."
                  : "e.g. It missed the part about…"
              }
            />
          </label>
        </div>

        <footer className="feedback-foot">
          {hasFeedback && (
            <button className="btn ghost feedback-remove" onClick={onRemove}>
              Remove feedback
            </button>
          )}
          <span className="spacer" />
          <button className="btn ghost" onClick={onSkip}>
            Skip note
          </button>
          <button className="btn primary" onClick={() => onSave(selected, draft.trim())}>
            Save feedback
          </button>
        </footer>
      </div>
    </div>
  );
}
