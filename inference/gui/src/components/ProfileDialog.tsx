import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";

import { Icon } from "./Icon";
import { Segmented } from "./Segmented";
import { greeting } from "../services/greeting";
import { modelName } from "../services/model";
import { useChat } from "../store/chatStore";
import type { GreetingStyle, ProfileAccent } from "../types";

const ACCENTS: ProfileAccent[] = ["blue", "teal", "violet", "rose", "amber"];

function sinceLabel(value: string): string {
  if (!value) return "Today";
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString(undefined, { month: "short", year: "numeric" });
}

export function ProfileDialog() {
  const open = useChat((s) => s.profileOpen);
  const close = useChat((s) => s.closeProfile);
  const username = useChat((s) => s.username);
  const avatarUrl = useChat((s) => s.avatarUrl);
  const profile = useChat((s) => s.profile);
  const save = useChat((s) => s.saveProfile);
  const sessionCount = useChat((s) => s.sessions.length);
  const status = useChat((s) => s.status);

  const [name, setName] = useState(username);
  const [bio, setBio] = useState(profile.bio);
  const [pronouns, setPronouns] = useState(profile.pronouns);
  const [accent, setAccent] = useState<ProfileAccent>(profile.accent);
  const [greetingStyle, setGreetingStyle] = useState<GreetingStyle>(profile.greetingStyle);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(avatarUrl);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setName(username);
    setBio(profile.bio);
    setPronouns(profile.pronouns);
    setAccent(profile.accent);
    setGreetingStyle(profile.greetingStyle);
    setPreview(avatarUrl);
    setFile(null);
    setBusy(false);
  }, [open, username, avatarUrl, profile]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close]);

  if (!open) return null;

  const choose = (event: ChangeEvent<HTMLInputElement>) => {
    const picked = event.target.files?.[0];
    if (!picked) return;
    setFile(picked);
    setPreview(URL.createObjectURL(picked));
  };

  const details = {
    bio: bio.trim(),
    pronouns: pronouns.trim(),
    accent,
    greetingStyle,
    createdAt: profile.createdAt,
  };

  const submit = async () => {
    setBusy(true);
    await save(name.trim() || "You", details, file ?? undefined);
    setBusy(false);
  };

  const removePhoto = async () => {
    setBusy(true);
    await save(name.trim() || "You", details, undefined, true);
    setBusy(false);
  };

  const displayName = name.trim() || "You";
  const headline =
    greetingStyle === "simple" ? `Hey ${displayName}.` : `${greeting()}, ${displayName}!`;

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => event.target === event.currentTarget && close()}
    >
      <div
        className="modal profile-modal"
        data-accent={accent}
        role="dialog"
        aria-modal="true"
        aria-label="Profile"
      >
        <button className="modal-x profile-close" onClick={close} aria-label="Close profile">
          <Icon name="close" size={13} />
        </button>

        <div className="profile-layout">
          <aside className="profile-side">
            <div className="profile-header-glow" />

            <div className="profile-identity">
              <div className="profile-avatar-ring">
                <button
                  className="profile-avatar"
                  data-accent={accent}
                  onClick={() => inputRef.current?.click()}
                  title="Change photo"
                >
                  {preview ? (
                    <img src={preview} alt="" />
                  ) : (
                    <span>{displayName.slice(0, 1).toUpperCase()}</span>
                  )}
                  <span className="profile-avatar-badge">
                    <Icon name="compose" size={11} />
                  </span>
                </button>
              </div>

              <div className="profile-id-row">
                <h2>{displayName}</h2>
                <span className="profile-badge profile-badge-local">Local</span>
              </div>
              <div className="profile-meta">
                {pronouns.trim() && <span className="profile-chip">{pronouns.trim()}</span>}
                <span className="profile-meta-bio">
                  {bio.trim() || "Add a short status below"}
                </span>
              </div>
            </div>

            <div className="profile-stats">
              <div className="profile-stat">
                <span>Saved chats</span>
                <b>{sessionCount}</b>
              </div>
              <div className="profile-stat">
                <span>Active model</span>
                <b>{modelName(status)}</b>
              </div>
              <div className="profile-stat">
                <span>Local since</span>
                <b>{sinceLabel(profile.createdAt)}</b>
              </div>
            </div>

            <span className="profile-privacy">
              <Icon name="info" size={12} />
              Stored only in settings.json
            </span>
          </aside>

          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/gif"
            hidden
            onChange={choose}
          />

          <div className="profile-body">
            <section className="profile-section">
              <div className="profile-section-head">
                <h3>Identity</h3>
              </div>
              <p className="profile-section-hint">
                How Dizel addresses you. Varied picks a fresh line each chat; Time uses the
                clock.
              </p>
              <div className="profile-fields">
                <div className="profile-pair">
                  <label className="profile-field">
                    <span>
                      Display name
                      <b>{name.length}/40</b>
                    </span>
                    <input
                      className="set-input"
                      value={name}
                      maxLength={40}
                      spellCheck={false}
                      onChange={(event) => setName(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") void submit();
                      }}
                    />
                  </label>

                  <label className="profile-field">
                    <span>
                      Status
                      <b>{bio.length}/80</b>
                    </span>
                    <input
                      className="set-input"
                      value={bio}
                      maxLength={80}
                      spellCheck={false}
                      placeholder="Building local AI"
                      onChange={(event) => setBio(event.target.value)}
                    />
                  </label>
                </div>

                <div className="profile-pair">
                  <label className="profile-field">
                    <span>Pronouns</span>
                    <input
                      className="set-input"
                      value={pronouns}
                      maxLength={20}
                      spellCheck={false}
                      placeholder="they/them"
                      onChange={(event) => setPronouns(event.target.value)}
                    />
                  </label>
                  <div className="profile-field profile-greeting">
                    <span>Greeting</span>
                    <Segmented
                      value={greetingStyle}
                      options={[
                        { value: "warm", label: "Varied" },
                        { value: "time", label: "Time" },
                        { value: "simple", label: "Simple" },
                      ]}
                      onChange={(value) => setGreetingStyle(value as GreetingStyle)}
                      ariaLabel="Greeting style"
                    />
                  </div>
                </div>
              </div>
            </section>

            <section className="profile-section">
              <div className="profile-section-head">
                <h3>Avatar</h3>
                <button
                  className="profile-photo-remove"
                  onClick={() => void removePhoto()}
                  disabled={!preview || busy}
                >
                  Remove photo
                </button>
              </div>
              <div className="profile-accent-row">
                {ACCENTS.map((value) => (
                  <button
                    key={value}
                    type="button"
                    className="profile-accent"
                    data-accent={value}
                    aria-label={`${value} avatar color`}
                    aria-pressed={accent === value}
                    onClick={() => setAccent(value)}
                  >
                    {accent === value && <Icon name="check" size={13} />}
                  </button>
                ))}
              </div>
            </section>

            <section className="profile-section">
              <div className="profile-section-head">
                <h3>Preview</h3>
                <span className="profile-preview-note">Updates live</span>
              </div>
              <div className="profile-preview-card">
                <span className="profile-preview-avatar" data-accent={accent}>
                  {preview ? (
                    <img src={preview} alt="" />
                  ) : (
                    displayName.slice(0, 1).toUpperCase()
                  )}
                </span>
                <div className="profile-preview-copy">
                  <strong>{headline}</strong>
                  {bio.trim() && <em>{bio.trim()}</em>}
                  <span>What&apos;s on your mind?</span>
                </div>
              </div>
            </section>
          </div>

          <footer className="set-foot profile-foot">
            <span className="spacer" />
            <button className="btn" onClick={close}>
              Cancel
            </button>
            <button
              className="btn primary"
              onClick={() => void submit()}
              disabled={busy || !name.trim()}
            >
              {busy ? "Saving…" : "Save"}
            </button>
          </footer>
        </div>
      </div>
    </div>
  );
}
