export type ThinkingEffort = "low" | "medium" | "high" | "xhigh" | "deep";
export type FastSpeed = "fast" | "turbo" | "ultrafast";

export interface SpeedInfo {
  id: FastSpeed;
  label: string;
  color: string;
  note: string;
}

/** Higher speed spends a larger reply budget and carries a stronger clause. */
export const FAST_SPEEDS: SpeedInfo[] = [
  {
    id: "fast",
    label: "Fast",
    color: "#38bdf8",
    note: "A direct answer without preamble or filler. The default.",
  },
  {
    id: "turbo",
    label: "Turbo",
    color: "#4fd6be",
    note: "A larger reply budget and a stronger prompt to answer promptly.",
  },
  {
    id: "ultrafast",
    label: "Ultrafast",
    color: "#fb923c",
    note: "The largest budget and the strongest prompt to answer immediately.",
  },
];

const SPEED_IDS = new Set<string>(FAST_SPEEDS.map((item) => item.id));

export function normalizeSpeed(value: unknown): FastSpeed {
  return typeof value === "string" && SPEED_IDS.has(value)
    ? (value as FastSpeed)
    : "fast";
}

export function speedById(id: FastSpeed): SpeedInfo {
  return FAST_SPEEDS.find((item) => item.id === id) ?? FAST_SPEEDS[0];
}

export interface EffortInfo {
  id: ThinkingEffort;
  label: string;
  color: string;
  note: string;
}

export const THINKING_EFFORTS: EffortInfo[] = [
  {
    id: "low",
    label: "Low",
    color: "#60a5fa",
    note: "Answers straight away with minimal reasoning. Fastest and cheapest — good for simple questions, lookups and small edits.",
  },
  {
    id: "medium",
    label: "Medium",
    color: "#a78bfa",
    note: "Thinks briefly before answering: weighs the obvious options and checks the usual pitfalls. The default balance of speed and care.",
  },
  {
    id: "high",
    label: "High",
    color: "#f472b6",
    note: "Works the problem properly — compares approaches, checks edge cases and verifies the answer before writing it.",
  },
  {
    id: "xhigh",
    label: "X-High",
    color: "#fb923c",
    note: "Breaks the task into parts, weighs the alternatives against each other and reviews its own reasoning before committing.",
  },
  {
    id: "deep",
    label: "Max",
    color: "#f43f5e",
    note: "Maximum effort. Explores several routes, stress-tests the answer against failure modes and spends the whole budget getting it right.",
  },
];

export const MODE_COLORS: Record<string, string> = {
  Fast: "#38bdf8",
  Planning: "#f59e0b",
  Coding: "#34d399",
};

const EFFORT_IDS = new Set<string>(THINKING_EFFORTS.map((item) => item.id));

export function normalizeEffort(value: unknown): ThinkingEffort {
  return typeof value === "string" && EFFORT_IDS.has(value)
    ? (value as ThinkingEffort)
    : "medium";
}

export function effortById(id: ThinkingEffort): EffortInfo {
  return THINKING_EFFORTS.find((item) => item.id === id) ?? THINKING_EFFORTS[1];
}

export function modeColor(
  mode: string,
  effort: ThinkingEffort,
  speed: FastSpeed = "fast",
): string {
  if (mode === "Fast") return speedById(speed).color;
  return mode === "Thinking"
    ? effortById(effort).color
    : MODE_COLORS[mode] ?? "#9aa6b2";
}

export function modeLabel(
  mode: string,
  effort: ThinkingEffort,
  speed: FastSpeed = "fast",
): string {
  // The chip is the tier itself in the Fast family, not "Fast · Turbo".
  if (mode === "Fast") return speedById(speed).label;
  return mode === "Thinking" ? effortById(effort).label : mode;
}
