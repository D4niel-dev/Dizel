import { useCallback, useEffect, useRef, useState } from "react";

import { Icon } from "./Icon";
import { Section, Switch } from "./settingsUi";
import * as api from "../services/api";
import type { McpServerInfo, McpServerState } from "../services/api";

const STATE_LABEL: Record<McpServerState, string> = {
  ready: "Connected",
  starting: "Starting…",
  error: "Error",
  disabled: "Off",
};

interface Draft {
  id?: string;
  name: string;
  transport: "stdio" | "http";
  command: string;
  args: string;
  env: string;
  url: string;
  headers: string;
  enabled: boolean;
}

const EMPTY: Draft = {
  id: undefined,
  name: "",
  transport: "stdio",
  command: "",
  args: "",
  env: "",
  url: "",
  headers: "",
  enabled: true,
};

function draftFrom(server: McpServerInfo): Draft {
  return {
    id: server.id,
    name: server.name,
    transport: server.transport,
    command: server.command ?? "",
    args: (server.args ?? []).join("\n"),
    env: server.env_keys.join("\n"),
    url: server.url ?? "",
    headers: server.header_keys.join("\n"),
    enabled: server.enabled,
  };
}

function parseKv(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const eq = trimmed.indexOf("=");
    if (eq === 0) continue;
    if (eq < 0) {
      out[trimmed] = "";
      continue;
    }
    out[trimmed.slice(0, eq).trim()] = trimmed.slice(eq + 1).trim();
  }
  return out;
}

export function McpSettings() {
  const [servers, setServers] = useState<McpServerInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [draft, setDraft] = useState<Draft | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [prefs, setPrefs] = useState<api.McpPreferences | null>(null);
  const [prefsBusy, setPrefsBusy] = useState(false);
  const [prefsNotice, setPrefsNotice] = useState("");
  const alive = useRef(true);
  // Synchronous re-entry guard: `busy` only exists on the next render, so a
  // second click in the same tick would still see it as false.
  const saveInFlight = useRef(false);
  // Bumped before every save. A status poll that started earlier carries the
  // old value and is dropped instead of overwriting the saved list with a
  // pre-save snapshot.
  const listSeq = useRef(0);

  const load = useCallback(async () => {
    // While a save is in flight a poll could start after the counter bump but
    // before the PUT commits and would then apply a snapshot older than the
    // save's response; skip it. Polls that started before the save are
    // already covered by the sequence check below.
    if (saveInFlight.current) return;
    const seq = listSeq.current;
    try {
      const data = await api.mcpServers();
      if (alive.current && seq === listSeq.current) {
        setServers(data.servers);
        setError("");
      }
    } catch (err) {
      if (alive.current && seq === listSeq.current) setError((err as Error).message);
    } finally {
      if (alive.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    alive.current = true;
    void load();
    const timer = window.setInterval(() => void load(), 2000);
    return () => {
      alive.current = false;
      window.clearInterval(timer);
    };
  }, [load]);

  useEffect(() => {
    let cancelled = false;
    api
      .mcpPreferences()
      .then((value) => {
        if (!cancelled) setPrefs(value);
      })
      .catch((err) => {
        if (!cancelled) setError((err as Error).message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const savePrefs = async () => {
    if (!prefs || prefsBusy) return;
    setPrefsBusy(true);
    setError("");
    setPrefsNotice("");
    try {
      const saved = await api.setMcpPreferences(prefs);
      if (alive.current) {
        setPrefs(saved);
        setPrefsNotice("Saved — applies on the next connect or call.");
      }
    } catch (err) {
      if (alive.current) setError((err as Error).message);
    } finally {
      if (alive.current) setPrefsBusy(false);
    }
  };

  // Entries are either full server records or partial payloads; the sidecar
  // merges them into the stored config.
  const save = async (next: unknown[]) => {
    if (saveInFlight.current) return;
    saveInFlight.current = true;
    listSeq.current += 1;
    setBusy(true);
    setError("");
    try {
      const data = await api.setMcpServers(next);
      if (alive.current) {
        setServers(data.servers);
        setDraft(null);
        setNotice("Saved.");
      }
    } catch (err) {
      if (alive.current) setError((err as Error).message);
    } finally {
      saveInFlight.current = false;
      if (alive.current) setBusy(false);
    }
  };

  const payloadFor = (entry: McpServerInfo | null, draftValue: Draft) => {
    const base: Record<string, unknown> = {
      id: entry?.id,
      name: draftValue.name.trim(),
      transport: draftValue.transport,
      enabled: draftValue.enabled,
    };
    if (draftValue.transport === "stdio") {
      base.command = draftValue.command.trim();
      base.args = draftValue.args
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
      base.env = parseKv(draftValue.env);
    } else {
      base.url = draftValue.url.trim();
      base.headers = parseKv(draftValue.headers);
    }
    return base;
  };

  const commitDraft = () => {
    if (busy || saveInFlight.current) return;
    if (!draft) return;
    if (!draft.name.trim()) {
      setError("A name is required.");
      return;
    }
    const current = draft.id ? servers.find((s) => s.id === draft.id) ?? null : null;
    const payload = payloadFor(current, draft);
    const next = current
      ? servers.map((s) => (s.id === current.id ? { ...s, ...payload } : s))
      : [...servers, payload];
    void save(next);
  };

  const toggle = (server: McpServerInfo) => {
    if (busy || saveInFlight.current) return;
    void save(
      servers.map((s) => (s.id === server.id ? { ...s, enabled: !s.enabled } : s)),
    );
  };

  const remove = (server: McpServerInfo) => {
    if (busy || saveInFlight.current) return;
    setConfirmId(null);
    void save(servers.filter((s) => s.id !== server.id));
  };

  const test = async (server: McpServerInfo) => {
    if (busy || saveInFlight.current) return;
    setBusy(true);
    try {
      const result = await api.testMcpServer(server.id);
      if (alive.current) {
        setNotice(
          result.state === "ready"
            ? `${server.name}: connected.`
            : `${server.name}: ${result.error || result.state}.`,
        );
      }
      await load();
    } catch (err) {
      if (alive.current) setError((err as Error).message);
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  const addButton = (
    <button className="btn small" onClick={() => setDraft({ ...EMPTY })} disabled={busy}>
      <Icon name="plus" size={12} />
      Add server
    </button>
  );

  const errorBlock = error ? (
    <div className="mcp-list-error" role="alert">
      <Icon name="alert" size={12} />
      <span>{error}</span>
    </div>
  ) : null;

  return (
    <>
      {prefs && (
        <Section
          className="mcp-section"
          title="Connection & limits"
          hint="One set of rules for every server. Saved alongside the server list."
        >
          <label className="set-row">
            <span className="set-row-text">
              <span>Connect timeout</span>
              <small>Seconds to wait for a server to start and list its tools</small>
            </span>
            <span className="set-row-control slider-control">
              <b>{prefs.connect_timeout}s</b>
              <input
                className="set-range"
                type="range"
                min={5}
                max={60}
                step={1}
                value={prefs.connect_timeout}
                onChange={(event) =>
                  setPrefs({ ...prefs, connect_timeout: Number(event.target.value) })
                }
              />
            </span>
          </label>
          <label className="set-row">
            <span className="set-row-text">
              <span>Call timeout</span>
              <small>Longest a single tool call may run</small>
            </span>
            <span className="set-row-control slider-control">
              <b>{prefs.call_timeout}s</b>
              <input
                className="set-range"
                type="range"
                min={10}
                max={300}
                step={5}
                value={prefs.call_timeout}
                onChange={(event) =>
                  setPrefs({ ...prefs, call_timeout: Number(event.target.value) })
                }
              />
            </span>
          </label>
          <label className="set-row">
            <span className="set-row-text">
              <span>Tool output cap</span>
              <small>Characters of a tool result handed to the model</small>
            </span>
            <span className="set-row-control slider-control">
              <b>{(prefs.max_output_chars / 1000).toFixed(1)}k</b>
              <input
                className="set-range"
                type="range"
                min={1000}
                max={32000}
                step={500}
                value={prefs.max_output_chars}
                onChange={(event) =>
                  setPrefs({ ...prefs, max_output_chars: Number(event.target.value) })
                }
              />
            </span>
          </label>
          <div className="set-row">
            <div className="set-row-text">
              <span>Retry failed calls</span>
              <small>Reconnect once and retry when a tool call fails</small>
            </div>
            <div className="set-row-control">
              <Switch
                checked={prefs.retry_failed_calls}
                onChange={() =>
                  setPrefs({ ...prefs, retry_failed_calls: !prefs.retry_failed_calls })
                }
                label="Retry failed tool calls"
              />
            </div>
          </div>
          <div className="set-row">
            <div className="set-row-text">
              <span>Auto-connect on launch</span>
              <small>Start enabled servers when the app opens</small>
            </div>
            <div className="set-row-control">
              <Switch
                checked={prefs.auto_connect}
                onChange={() => setPrefs({ ...prefs, auto_connect: !prefs.auto_connect })}
                label="Auto-connect MCP servers on launch"
              />
            </div>
          </div>
          {prefsNotice && (
            <div className="mcp-notice" role="status">
              <Icon name="check" size={12} />
              <span>{prefsNotice}</span>
            </div>
          )}
          <div className="mcp-prefs-actions">
            <button
              className="btn small primary"
              onClick={() => void savePrefs()}
              disabled={prefsBusy}
            >
              Save preferences
            </button>
          </div>
        </Section>
      )}

      <Section
        className="mcp-section"
        title="MCP servers"
        hint="Enabled servers connect on launch when auto-connect is on. Their tools appear when the model's tools switch is on."
        action={servers.length > 0 ? addButton : undefined}
      >
        {loading && <div className="mcp-hint-row">Reading servers…</div>}
        {!loading && servers.length === 0 && (
          <div className="mcp-empty">
            <span className="mcp-empty-mark" aria-hidden="true">
              <Icon name="plug" size={20} filled />
            </span>
            <div className="mcp-empty-copy">
              <h4>Connect your first MCP server</h4>
              <p>
                Add filesystem access, GitHub, a database — anything that speaks MCP.
                Its tools join the model&apos;s tool belt, right alongside the built-ins.
              </p>
              <div className="mcp-empty-actions">
                <button
                  className="btn small primary"
                  onClick={() => setDraft({ ...EMPTY })}
                  disabled={busy}
                >
                  <Icon name="plus" size={12} />
                  Add your first server
                </button>
                <span className="mcp-empty-note">You can turn each server off at any time.</span>
              </div>
            </div>
          </div>
        )}
        <div className="mcp-list">
          {servers.map((server) => (
            <div className="mcp-card" key={server.id}>
              <div className="mcp-card-head">
                <span className="mcp-state" data-state={server.state}>
                  <span className="mcp-dot" />
                  {STATE_LABEL[server.state]}
                </span>
                <span className="mcp-card-main">
                  <span className="mcp-card-name" title={server.name}>
                    {server.name}
                  </span>
                  <span className="mcp-badge">{server.transport}</span>
                  {server.state === "ready" && (
                    <span className="mcp-card-tools">
                      {server.tools.length} tool{server.tools.length === 1 ? "" : "s"}
                    </span>
                  )}
                </span>
                <span className="mcp-card-actions">
                  <button
                    className="mcp-icon-btn"
                    title="Test connection"
                    aria-label={`Test ${server.name}`}
                    onClick={() => void test(server)}
                    disabled={busy}
                  >
                    <Icon name="refresh" size={13} />
                  </button>
                  <button
                    className="mcp-icon-btn"
                    title="Edit"
                    aria-label={`Edit ${server.name}`}
                    onClick={() => setDraft(draftFrom(server))}
                    disabled={busy}
                  >
                    <Icon name="settings" size={13} />
                  </button>
                  <button
                    className="mcp-icon-btn is-danger"
                    title="Remove"
                    aria-label={`Remove ${server.name}`}
                    onClick={() => setConfirmId(server.id)}
                    disabled={busy}
                  >
                    <Icon name="trash" size={13} />
                  </button>
                </span>
                {/* Switch has no disabled prop; toggle() refuses while a
                    request is running, and this dims/inerts it for feedback. */}
                <span className="mcp-switch" aria-disabled={busy || undefined}>
                  <Switch
                    checked={server.enabled}
                    onChange={() => toggle(server)}
                    label={`Enable ${server.name}`}
                  />
                </span>
              </div>
              {server.error && (
                <div className="mcp-error" role="alert">
                  <Icon name="alert" size={12} />
                  <span>{server.error}</span>
                </div>
              )}
              {confirmId === server.id && (
                <div className="mcp-confirm">
                  <span>Remove “{server.name}” and its tools?</span>
                  <button className="btn small" onClick={() => setConfirmId(null)} disabled={busy}>
                    Cancel
                  </button>
                  <button className="btn small danger" onClick={() => remove(server)} disabled={busy}>
                    Remove
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
        {notice && (
          <div className="mcp-notice" role="status">
            <Icon name="check" size={12} />
            <span>{notice}</span>
          </div>
        )}
        {!draft && errorBlock}
      </Section>

      {draft && (
        <Section
          className="mcp-section"
          title={draft.id ? `Edit ${draft.name || "server"}` : "New server"}
          hint="Values are stored in settings.json on this machine. Leave a value blank to keep the stored one."
        >
          <div className="mcp-form">
            <label>
              Name
              <input
                className="set-input"
                value={draft.name}
                onChange={(event) => setDraft({ ...draft, name: event.target.value })}
                spellCheck={false}
              />
            </label>
            <label>
              Transport
              <select
                className="set-select"
                value={draft.transport}
                onChange={(event) =>
                  setDraft({ ...draft, transport: event.target.value as Draft["transport"] })
                }
              >
                <option value="stdio">stdio — spawn a local command</option>
                <option value="http">http — Streamable HTTP endpoint</option>
              </select>
            </label>
            {draft.transport === "stdio" ? (
              <>
                <label>
                  Command
                  <input
                    className="set-input mcp-mono"
                    value={draft.command}
                    placeholder="npx"
                    onChange={(event) => setDraft({ ...draft, command: event.target.value })}
                    spellCheck={false}
                  />
                </label>
                <label>
                  Arguments, one per line
                  <textarea
                    className="set-textarea mcp-mono"
                    rows={3}
                    value={draft.args}
                    placeholder={"-y\n@modelcontextprotocol/server-filesystem\nC:\\work"}
                    onChange={(event) => setDraft({ ...draft, args: event.target.value })}
                    spellCheck={false}
                  />
                </label>
                <label>
                  Environment, KEY=VALUE per line
                  <textarea
                    className="set-textarea mcp-mono"
                    rows={2}
                    value={draft.env}
                    onChange={(event) => setDraft({ ...draft, env: event.target.value })}
                    spellCheck={false}
                  />
                </label>
              </>
            ) : (
              <>
                <label>
                  URL
                  <input
                    className="set-input mcp-mono"
                    value={draft.url}
                    placeholder="https://mcp.example.com/mcp"
                    onChange={(event) => setDraft({ ...draft, url: event.target.value })}
                    spellCheck={false}
                  />
                </label>
                <label>
                  Headers, KEY=VALUE per line
                  <textarea
                    className="set-textarea mcp-mono"
                    rows={2}
                    value={draft.headers}
                    onChange={(event) => setDraft({ ...draft, headers: event.target.value })}
                    spellCheck={false}
                  />
                </label>
              </>
            )}
            {errorBlock}
            <div className="mcp-form-actions">
              <button className="btn" onClick={() => setDraft(null)}>
                Cancel
              </button>
              <button className="btn primary" onClick={commitDraft} disabled={busy}>
                Save server
              </button>
            </div>
          </div>
        </Section>
      )}
    </>
  );
}
