import type { SkillInfo } from "../types";

export type CommandKind = "system" | "skill" | "debug";

export interface CommandEntry {
  name: string;
  hint: string;
  kind: CommandKind;
  aliases: string[];
  category?: string;
  source?: string;
}

export interface CommandGroup {
  label: string;
  entries: CommandEntry[];
  showCount?: boolean;
}

/** Mirrors dizel_core.skills.CATEGORY_ORDER; keeps GUI and TUI in step. */
export const CATEGORY_ORDER = [
  "Writing",
  "Thinking",
  "Coding",
  "Learning",
  "Work",
  "Data",
  "Local",
  "General",
];

/** The always-available commands that are not themselves prompt skills. */
export const SKILLS_COMMAND: CommandEntry = {
  name: "skills",
  hint: "List prompt skills, or reload them from disk",
  kind: "system",
  aliases: [],
};

export const PLUGINS_COMMAND: CommandEntry = {
  name: "plugins",
  hint: "Run a connected MCP tool, or insert its call for the model",
  kind: "system",
  aliases: [],
};

function debug(name: string, hint: string, aliases: string[] = []): CommandEntry {
  return { name, hint, kind: "debug", aliases };
}

/** Mirrors the /help panel in chatStore; keep the two in step. */
export const DEBUG_COMMANDS: CommandEntry[] = [
  debug("ctx", "Context breakdown from the server"),
  debug("model", "Model and provider facts"),
  debug("capabilities", "What each surface can do"),
  debug("status", "One-screen state dump"),
  debug("history", "Messages sent to the model"),
  debug("bundle", "Copy a diagnostic bundle"),
  debug("echo", "Append a fake reply"),
  debug("markdown", "Append every markdown style", ["md"]),
  debug("thought", "Append a Thought fixture", ["think"]),
  debug("filler", "Append n words of filler"),
  debug("error", "Raise the error banner"),
  debug("simulate", "loading | error | streaming | off"),
  debug("clear", "Start a new chat"),
  debug("export", "Save this chat as JSON"),
  debug("replay", "Resend the last prompt"),
  debug("copy", "Copy the last reply"),
  debug("pin", "Pin or unpin this chat"),
  debug("archive", "Archive this chat"),
  debug("stop", "Stop the current generation"),
  debug("budget", "Force displayed context usage"),
  debug("ctxfull", "Fill the real context window"),
  debug("usage", "Window state; fill | exhaust | reset | expire"),
  debug("tools", "Let the model call tools"),
  debug("bench", "Time n runs (ttft + tok/s)"),
  debug("accent", "blue | teal | violet | rose"),
  debug("reset-ui", "Clear local state and reload"),
  debug("help", "List debug commands"),
  debug("debug", "Turn debug mode off"),
];

export const DEBUG_COMMAND_NAMES = new Set(
  DEBUG_COMMANDS.flatMap((entry) => [entry.name, ...entry.aliases]),
);

/**
 * /skills and /plugins first, then prompt skills grouped by category, then the
 * debug switch. A skill may not shadow a debug command while debug is on —
 * runSlash applies the same rule.
 */
export function buildCommandGroups(skills: SkillInfo[], debugOn: boolean): CommandGroup[] {
  const buckets = new Map<string, CommandEntry[]>();

  for (const skill of skills) {
    if (skill.name === SKILLS_COMMAND.name || skill.name === PLUGINS_COMMAND.name) continue;
    if (debugOn && DEBUG_COMMAND_NAMES.has(skill.name)) continue;
    const category = skill.category || "General";
    const entry: CommandEntry = {
      name: skill.name,
      hint: skill.description,
      kind: "skill",
      aliases: skill.aliases,
      category,
      source: skill.source,
    };
    const bucket = buckets.get(category);
    if (bucket) bucket.push(entry);
    else buckets.set(category, [entry]);
  }

  const rank = new Map(CATEGORY_ORDER.map((name, index) => [name, index]));
  const categories = [...buckets.keys()].sort((left, right) => {
    const leftRank = rank.get(left) ?? CATEGORY_ORDER.length;
    const rightRank = rank.get(right) ?? CATEGORY_ORDER.length;
    return leftRank - rightRank || left.localeCompare(right);
  });

  const groups: CommandGroup[] = [
    { label: "Skills", entries: [SKILLS_COMMAND], showCount: false },
    { label: "Plugins", entries: [PLUGINS_COMMAND], showCount: false },
  ];
  for (const category of categories) {
    groups.push({ label: category, entries: buckets.get(category) ?? [] });
  }
  if (debugOn) groups.push({ label: "Debug", entries: DEBUG_COMMANDS });
  return groups;
}

function subsequenceGaps(text: string, query: string): number | null {
  let cursor = 0;
  let gaps = 0;
  let last = -1;
  for (const char of query) {
    const found = text.indexOf(char, cursor);
    if (found < 0) return null;
    if (last >= 0) gaps += found - last - 1;
    last = found;
    cursor = found + 1;
  }
  return gaps;
}

function matchScore(text: string, query: string): number {
  if (!query) return 0;
  if (text === query) return 0;
  if (text.startsWith(query)) return 1 + (text.length - query.length) * 0.01;
  const index = text.indexOf(query);
  if (index >= 0) return 3 + index * 0.1;
  const gaps = subsequenceGaps(text, query);
  return gaps === null ? Infinity : 10 + gaps;
}

function entryScore(entry: CommandEntry, query: string): number {
  return Math.min(
    matchScore(entry.name, query),
    ...entry.aliases.map((alias) => matchScore(alias, query)),
  );
}

/** Character positions matched by a fuzzy query, for drawing the highlight. */
export function matchPositions(text: string, query: string): number[] {
  const positions: number[] = [];
  if (!query) return positions;
  const lower = text.toLowerCase();
  let cursor = 0;
  for (const char of query.toLowerCase()) {
    const found = lower.indexOf(char, cursor);
    if (found < 0) return [];
    positions.push(found);
    cursor = found + 1;
  }
  return positions;
}

export function filterCommands(groups: CommandGroup[], query: string): CommandGroup[] {
  const needle = query.toLowerCase();
  return groups
    .map((group) => ({
      ...group,
      entries: group.entries
        .map((entry) => ({ entry, score: entryScore(entry, needle) }))
        .filter(({ score }) => Number.isFinite(score))
        .sort(
          (left, right) =>
            left.score - right.score || left.entry.name.localeCompare(right.entry.name),
        )
        .map(({ entry }) => entry),
    }))
    .filter((group) => group.entries.length > 0);
}

export function flattenGroups(groups: CommandGroup[]): CommandEntry[] {
  return groups.flatMap((group) => group.entries);
}
