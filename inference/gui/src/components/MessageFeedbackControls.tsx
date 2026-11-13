import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { FeedbackDialog } from "./FeedbackDialog";
import { Icon } from "./Icon";
import { useChat } from "../store/chatStore";
import { assistantTurnIndex } from "../lib/variants";
import { feedbackKey, getFeedback, setFeedbackEntry } from "../lib/feedback";
import { modelName } from "../services/model";
import * as api from "../services/api";
import type { FeedbackRating, MessageFeedback } from "../types";

/** Thumb controls and the optional reason dialog for one assistant reply. */
export function MessageFeedbackControls({
  messageId,
  content,
}: {
  messageId: string;
  content: string;
}) {
  const sessionId = useChat((state) => state.sessionId);
  const [feedback, setFeedback] = useState<MessageFeedback | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const current = useChat.getState();
    const turn = assistantTurnIndex(current.messages, messageId);
    setFeedback(getFeedback(current.sessionId, turn, content));
  }, [sessionId, messageId, content]);

  const persist = (entry: MessageFeedback | null) => {
    const current = useChat.getState();
    const turn = assistantTurnIndex(current.messages, messageId);
    if (!current.sessionId || turn < 0) return;

    setFeedbackEntry(current.sessionId, turn, content, entry);

    const index = current.messages.findIndex((message) => message.id === messageId);
    const prompt =
      [...current.messages.slice(0, Math.max(0, index))]
        .reverse()
        .find((message) => message.role === "user")?.content ?? "";
    const status = current.status;

    void api
      .saveFeedback({
        rating: entry?.rating ?? "removed",
        session_id: current.sessionId,
        message_key: feedbackKey(turn, content),
        reasons: entry?.reasons ?? [],
        note: entry?.note ?? "",
        response: content,
        prompt,
        model: status?.api_model || modelName(status),
        provider: status?.provider || "local",
      })
      .catch(() => undefined);
  };

  const choose = (rating: FeedbackRating) => {
    const same = feedback?.rating === rating;
    const next: MessageFeedback = {
      rating,
      reasons: same ? feedback.reasons : [],
      note: same ? feedback.note : "",
      createdAt: Date.now(),
    };
    setFeedback(next);
    persist(next);
    setOpen(true);
  };

  const save = (reasons: string[], note: string) => {
    if (!feedback) return;
    const next: MessageFeedback = { ...feedback, reasons, note, createdAt: Date.now() };
    setFeedback(next);
    persist(next);
    setOpen(false);
  };

  const remove = () => {
    setFeedback(null);
    persist(null);
    setOpen(false);
  };

  const positive = feedback?.rating === "up";
  const negative = feedback?.rating === "down";

  return (
    <>
      <span className="feedback-actions" role="group" aria-label="Rate this response">
        <button
          type="button"
          className={`act act-icon${positive ? " is-up" : ""}`}
          aria-pressed={positive}
          aria-label={positive ? "Edit positive feedback" : "Good response"}
          title={positive ? "Edit positive feedback" : "Good response"}
          onClick={() => choose("up")}
        >
          <Icon name="thumbUp" size={13} />
        </button>
        <button
          type="button"
          className={`act act-icon${negative ? " is-down" : ""}`}
          aria-pressed={negative}
          aria-label={negative ? "Edit negative feedback" : "Bad response"}
          title={negative ? "Edit negative feedback" : "Bad response"}
          onClick={() => choose("down")}
        >
          <Icon name="thumbDown" size={13} />
        </button>
      </span>

      {open &&
        feedback &&
        createPortal(
          <FeedbackDialog
            rating={feedback.rating}
            reasons={feedback.reasons}
            note={feedback.note}
            hasFeedback
            onSave={save}
            onSkip={() => setOpen(false)}
            onRemove={remove}
          />,
          document.body,
        )}
    </>
  );
}
