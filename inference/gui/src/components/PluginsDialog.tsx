import { useEffect, useMemo, useState } from "react";

import { Icon } from "./Icon";
import { callMcpTool, mcpTools, setTools } from "../services/api";
import type { McpPluginCatalog, McpPluginServer, McpPluginTool } from "../services/api";
import { useChat } from "../store/chatStore";

/**
 * Direct runner for connected MCP servers.
 *
 * Two ways to use a tool: run it here and read the output, or insert a
 * prefilled <tool> call so the model runs it through the normal tools loop.
 * Prompt-path calls only work while the model's tools switch is on, so the
 * dialog shows that switch and can flip it in place.
 */

type FieldValue = string | boolean;

interface JsonSchema {
  type?: string | string[];
  properties?: Record<string, JsonSchema>;
  required?: string[];
  enum?: unknown[];
  default?: unknown;
  description?: string;
}

function schemaOf(tool: McpPluginTool): JsonSchema {
  const raw = tool.input_schema;
  return raw && typeof raw === "object" ? (raw as JsonSchema) : {};
}

function kindOf(spec: JsonSchema | undefined): string {
  if (!spec) return "string";
  const declared = Array.isArray(spec.type)
    ? spec.type.find((entry) => entry !== "null")
    : spec.type;
  if (declared) return declared;
  if (spec.enum) return "string";
  if (spec.properties) return "object";
  return "string";
}

function initialValues(tool: McpPluginTool): Record<string, FieldValue> {
  const values: Record<string, FieldValue> = {};
  for (const [key, spec] of Object.entries(schemaOf(tool).properties ?? {})) {
    const kind = kindOf(spec);
    if (kind === "boolean") {
      values[key] = spec?.default === true;
    } else if (spec?.default !== undefined && spec.default !== null) {
      values[key] =
        kind === "array" || kind === "object"
          ? JSON.stringify(spec.default, null, 2)
          : String(spec.default);
    } else {
      values[key] = "";
    }
  }
  return values;
}

function buildArguments(
  tool: McpPluginTool,
  values: Record<string, FieldValue>,
): { args: Record<string, unknown>; error: string } {
  const schema = schemaOf(tool);
  const required = new Set(schema.required ?? []);
  const args: Record<string, unknown> = {};
  const missing: string[] = [];

  for (const [key, spec] of Object.entries(schema.properties ?? {})) {
    const kind = kindOf(spec);
    const value = values[key];

    if (kind === "boolean") {
      // A checkbox shows both states, so send both states. Omitting an
      // unchecked box would let the server's default override what is on screen.
      args[key] = value === true;
      continue;
    }

    const raw = String(value ?? "");
    if (!raw.trim()) {
      if (required.has(key)) missing.push(key);
      continue;
    }

    if (kind === "number" || kind === "integer") {
      const number = Number(raw);
      if (!Number.isFinite(number)) {
        return { args, error: `${key} must be a number.` };
      }
      args[key] = kind === "integer" ? Math.trunc(number) : number;
    } else if (kind === "array" || kind === "object") {
      try {
        args[key] = JSON.parse(raw);
      } catch {
        return { args, error: `${key} must be valid JSON.` };
      }
    } else {
      args[key] = raw;
    }
  }

  if (missing.length > 0) {
    return { args, error: `Required: ${missing.join(", ")}.` };
  }
  return { args, error: "" };
}

function splitKey(key: string): [string, string] {
  const at = key.indexOf("\u0000");
  return at < 0 ? ["", ""] : [key.slice(0, at), key.slice(at + 1)];
}

function toolKey(server: McpPluginServer, tool: McpPluginTool): string {
  return `${server.id}\u0000${tool.name}`;
}

const STATE_LABEL: Record<string, string> = {
  ready: "connected",
  starting: "connecting",
  error: "error",
  disabled: "off",
};

function Field({
  name,
  spec,
  required,
  value,
  onChange,
}: {
  name: string;
  spec: JsonSchema;
  required: boolean;
  value: FieldValue;
  onChange: (next: FieldValue) => void;
}) {
  const kind = kindOf(spec);
  const hint = spec.description ? <small>{spec.description}</small> : null;
  const label = (
    <span className="plugins-field-label">
      {name}
      {required && <em className="plugins-req">required</em>}
    </span>
  );

  if (kind === "boolean") {
    return (
      <label className="plugins-check">
        <input
          type="checkbox"
          checked={value === true}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>
          {label}
          {hint}
        </span>
      </label>
    );
  }

  if (spec.enum && spec.enum.length > 0) {
    return (
      <label className="plugins-field">
        {label}
        <select
          className="set-input"
          value={String(value)}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">{required ? "Select…" : "—"}</option>
          {spec.enum.map((option) => (
            <option key={String(option)} value={String(option)}>
              {String(option)}
            </option>
          ))}
        </select>
        {hint}
      </label>
    );
  }

  if (kind === "array" || kind === "object") {
    return (
      <label className="plugins-field">
        {label}
        <textarea
          className="set-input plugins-json"
          rows={3}
          spellCheck={false}
          value={String(value)}
          placeholder={kind === "array" ? '["value"]' : '{"key": "value"}'}
          onChange={(event) => onChange(event.target.value)}
        />
        {hint}
      </label>
    );
  }

  return (
    <label className="plugins-field">
      {label}
      <input
        className="set-input"
        type={kind === "number" || kind === "integer" ? "number" : "text"}
        step={kind === "integer" ? 1 : "any"}
        spellCheck={false}
        value={String(value)}
        onChange={(event) => onChange(event.target.value)}
      />
      {hint}
    </label>
  );
}

function ToolPanel({
  server,
  tool,
  toolsEnabled,
  enabling,
  onEnableTools,
}: {
  server: McpPluginServer;
  tool: McpPluginTool;
  toolsEnabled: boolean;
  enabling: boolean;
  onEnableTools: () => void;
}) {
  const [values, setValues] = useState<Record<string, FieldValue>>(() =>
    initialValues(tool),
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState("");
  const [copied, setCopied] = useState(false);

  const schema = schemaOf(tool);
  const properties = Object.entries(schema.properties ?? {});
  const required = new Set(schema.required ?? []);

  const run = async () => {
    const built = buildArguments(tool, values);
    if (built.error) {
      setError(built.error);
      return;
    }
    setError("");
    setBusy(true);
    try {
      const response = await callMcpTool(server.id, tool.name, built.args);
      setResult(response.result || "(no output)");
    } catch (caught) {
      setError((caught as Error).message);
      setResult("");
    } finally {
      setBusy(false);
    }
  };

  const insert = () => {
    const built = buildArguments(tool, values);
    if (built.error) {
      setError(built.error);
      return;
    }
    const state = useChat.getState();
    const call = `<tool name="${tool.namespaced}">${JSON.stringify(built.args)}</tool>`;
    const current = state.draft.trim();
    state.setDraft(current ? `${state.draft.replace(/\s+$/, "")}\n\n${call}` : call);
    state.closePlugins();
    state.requestComposerFocus();
    state.notify(
      toolsEnabled
        ? `Inserted ${tool.name} — edit the call, then press Enter.`
        : `Inserted ${tool.name}. Turn on model tools so the model can run it.`,
    );
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(result);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setError("Clipboard unavailable.");
    }
  };

  return (
    <div className="plugins-tool">
      <div className="plugins-tool-head">
        <div>
          <h3>{tool.name}</h3>
          <code className="plugins-namespaced">{tool.namespaced}</code>
        </div>
        <span className={`plugins-state is-${server.state}`}>
          {STATE_LABEL[server.state] ?? server.state}
        </span>
      </div>

      <p className="plugins-tool-desc">
        {tool.description || "No description from the server."}
      </p>

      {!toolsEnabled && (
        <div className="plugins-callout">
          <Icon name="info" size={13} />
          <span>
            Model tools are off. Direct runs work now; the model needs the switch on
            to execute an inserted call.
          </span>
          <button className="btn" onClick={onEnableTools} disabled={enabling}>
            {enabling ? "Enabling…" : "Enable"}
          </button>
        </div>
      )}

      <div className="plugins-fields">
        {properties.length === 0 ? (
          <p className="plugins-empty-fields">This tool takes no arguments.</p>
        ) : (
          properties.map(([name, spec]) => (
            <Field
              key={name}
              name={name}
              spec={spec}
              required={required.has(name)}
              value={values[name] ?? ""}
              onChange={(next) =>
                setValues((current) => ({ ...current, [name]: next }))
              }
            />
          ))
        )}
      </div>

      <div className="plugins-actions">
        <button className="btn" onClick={insert} disabled={busy}>
          <Icon name="sparkles" size={13} /> Insert for model
        </button>
        <button className="btn primary" onClick={() => void run()} disabled={busy}>
          {busy ? "Running…" : "Run now"}
        </button>
      </div>

      {error && <div className="plugins-error">{error}</div>}

      {result !== "" && (
        <div className="plugins-result">
          <div className="plugins-result-head">
            <span>Result</span>
            <button className="btn" onClick={() => void copy()}>
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <pre>{result}</pre>
        </div>
      )}
    </div>
  );
}

export function PluginsDialog() {
  const open = useChat((s) => s.pluginsOpen);
  const close = useChat((s) => s.closePlugins);
  const openSettings = useChat((s) => s.openSettings);
  const [catalog, setCatalog] = useState<McpPluginCatalog | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const [enabling, setEnabling] = useState(false);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    setLoading(true);
    setError("");
    void mcpTools()
      .then((next) => {
        if (!alive) return;
        setCatalog(next);
        setSelected((previous) => {
          const stillThere = next.servers.some((server) =>
            server.tools.some((tool) => toolKey(server, tool) === previous),
          );
          if (stillThere) return previous;
          const firstServer = next.servers.find((server) => server.tools.length > 0);
          return firstServer
            ? toolKey(firstServer, firstServer.tools[0])
            : "";
        });
      })
      .catch((caught) => {
        if (alive) setError((caught as Error).message);
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [open, reloadKey]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close]);

  const filtered = useMemo(() => {
    const servers = catalog?.servers ?? [];
    const needle = query.trim().toLowerCase();
    if (!needle) return servers;
    return servers
      .map((server) => ({
        ...server,
        tools: server.tools.filter((tool) =>
          `${server.name} ${server.id} ${tool.name} ${tool.namespaced} ${tool.description}`
            .toLowerCase()
            .includes(needle),
        ),
      }))
      .filter((server) => server.tools.length > 0);
  }, [catalog, query]);

  if (!open) return null;

  const [selectedServerId, selectedToolName] = splitKey(selected);
  const selectedServer = catalog?.servers.find(
    (server) => server.id === selectedServerId,
  );
  const selectedTool = selectedServer?.tools.find(
    (tool) => tool.name === selectedToolName,
  );
  const toolsEnabled = catalog?.tools_enabled ?? false;
  const serverCount = catalog?.servers.length ?? 0;
  const toolCount =
    catalog?.servers.reduce((sum, server) => sum + server.tools.length, 0) ?? 0;

  const toggleTools = async () => {
    if (!catalog || enabling) return;
    setEnabling(true);
    try {
      const next = await setTools(!catalog.tools_enabled);
      setCatalog({ ...catalog, tools_enabled: next.enabled });
      useChat
        .getState()
        .notify(next.enabled ? "Model tools on." : "Model tools off.");
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setEnabling(false);
    }
  };

  const openMcpSettings = () => {
    close();
    openSettings("MCP");
  };

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => event.target === event.currentTarget && close()}
    >
      <div
        className="modal plugins-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Plugins"
      >
        <div className="modal-head plugins-head">
          <h2>
            <Icon name="plug" size={15} /> Plugins
          </h2>
          {catalog && (
            <button
              className={toolsEnabled ? "plugins-tools-pill is-on" : "plugins-tools-pill"}
              onClick={() => void toggleTools()}
              disabled={enabling}
              aria-pressed={toolsEnabled}
              title="Toggle the model's tools switch"
            >
              <span className="plugins-dot" />
              Model tools {toolsEnabled ? "on" : "off"}
            </button>
          )}
          <button
            className="modal-x"
            onClick={() => setReloadKey((key) => key + 1)}
            title="Refresh plugins"
            aria-label="Refresh plugins"
          >
            <Icon name="refresh" size={13} />
          </button>
          <button className="modal-x" onClick={close} aria-label="Close plugins">
            <Icon name="close" size={13} />
          </button>
        </div>

        <p className="plugins-intro">
          {serverCount > 0
            ? `${toolCount} tool${toolCount === 1 ? "" : "s"} across ${serverCount} server${serverCount === 1 ? "" : "s"}. Run one here, or insert its call for the model.`
            : "MCP servers and their tools live here. Add one in Settings → MCP."}
        </p>

        {loading && !catalog ? (
          <div className="modal-loading">
            <span className="pill-spinner" />
            Reading connected plugins…
          </div>
        ) : error && !catalog ? (
          <div className="plugins-error plugins-error-page">
            {error}
            <button className="btn" onClick={() => setReloadKey((key) => key + 1)}>
              Retry
            </button>
          </div>
        ) : serverCount === 0 ? (
          <div className="plugins-empty">
            <Icon name="plug" size={24} />
            <h3>No plugins connected</h3>
            <p>
              Add an MCP server once, and every tool it exposes appears here for
              direct runs and model calls.
            </p>
            <button className="btn primary" onClick={openMcpSettings}>
              Open MCP settings
            </button>
          </div>
        ) : (
          <div className="plugins-layout">
            <aside className="plugins-side">
              <label className="plugins-search">
                <Icon name="search" size={13} />
                <input
                  value={query}
                  spellCheck={false}
                  placeholder="Filter tools…"
                  aria-label="Filter plugin tools"
                  onChange={(event) => setQuery(event.target.value)}
                />
              </label>

              <div className="plugins-scroll">
                {filtered.length === 0 ? (
                  <p className="plugins-list-empty">No tools match “{query.trim()}”.</p>
                ) : (
                  filtered.map((server) => (
                    <div key={server.id} className="plugins-server">
                      <div className="plugins-server-head">
                        <span className="plugins-server-name" title={server.id}>
                          {server.name}
                        </span>
                        <span className={`plugins-state is-${server.state}`}>
                          {STATE_LABEL[server.state] ?? server.state}
                        </span>
                      </div>
                      {server.error && (
                        <p className="plugins-server-error">{server.error}</p>
                      )}
                      {server.tools.length === 0 ? (
                        <p className="plugins-server-empty">
                          {server.enabled
                            ? "No tools ready yet."
                            : "Server is off — enable it in Settings → MCP."}
                        </p>
                      ) : (
                        server.tools.map((tool) => {
                          const key = toolKey(server, tool);
                          return (
                            <button
                              key={key}
                              className={
                                selected === key
                                  ? "plugins-tool-row is-active"
                                  : "plugins-tool-row"
                              }
                              onClick={() => setSelected(key)}
                            >
                              <span className="plugins-tool-name">{tool.name}</span>
                              <small>
                                {tool.description || "No description"}
                              </small>
                            </button>
                          );
                        })
                      )}
                    </div>
                  ))
                )}
              </div>
            </aside>

            <section className="plugins-detail">
              {selectedServer && selectedTool ? (
                <ToolPanel
                  key={selected}
                  server={selectedServer}
                  tool={selectedTool}
                  toolsEnabled={toolsEnabled}
                  enabling={enabling}
                  onEnableTools={() => void toggleTools()}
                />
              ) : (
                <div className="plugins-empty plugins-empty-detail">
                  <Icon name="plug" size={20} />
                  <h3>Pick a tool</h3>
                  <p>
                    Tools appear once a server connects. Check Settings → MCP if a
                    server is off or failing.
                  </p>
                  <button className="btn" onClick={openMcpSettings}>
                    Open MCP settings
                  </button>
                </div>
              )}
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
