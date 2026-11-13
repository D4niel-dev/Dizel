import type { DayPart } from "./greeting";

export interface WelcomeCopy {
  headline: string;
  subline: string;
}

/** Time-bucketed headline templates; {name} and {greeting} stay live. */
const HEADLINES: Record<DayPart, string[]> = {
  morning: [
    "{greeting}, {name}!",
    "Morning, {name}.",
    "Morning, {name}!",
    "Early start, {name}?",
    "Early bird, {name}?",
    "Early riser, {name}?",
    "Good morning to you, {name}.",
    "A fresh morning, {name}.",
    "New day, {name}.",
    "Morning already, {name}?",
    "Good morning, {name}.",
    "Top of the morning, {name}.",
    "Mornin', {name}.",
    "Morning to you, {name}.",
    "Bright and early, {name}.",
    "A new day, {name}.",
  ],
  afternoon: [
    "{greeting}, {name}!",
    "Afternoon, {name}.",
    "Afternoon, {name}!",
    "Good afternoon to you, {name}.",
    "Good afternoon, {name}.",
    "Afternoon already, {name}?",
    "Back at it, {name}?",
    "Halfway through, {name}?",
    "How's the afternoon, {name}?",
    "Having a good afternoon, {name}?",
    "Good day, {name}.",
    "Good day to you, {name}.",
    "Afternoon to you, {name}.",
    "Mid-afternoon, {name}.",
    "The afternoon's here, {name}.",
    "Still at it, {name}?",
  ],
  evening: [
    "{greeting}, {name}!",
    "Evening, {name}.",
    "Evening, {name}!",
    "Good evening to you, {name}.",
    "Good evening, {name}.",
    "Evening already, {name}?",
    "Still going, {name}?",
    "Still up, {name}?",
    "How's the evening, {name}?",
    "Having a good evening, {name}?",
    "Winding down, {name}?",
    "A good evening to you, {name}.",
    "Evening to you, {name}.",
    "Late evening, {name}.",
    "The evening's here, {name}.",
    "Hope your evening's going well, {name}.",
  ],
};

const SUBLINES = [
  "What's on your mind?",
  "Ask anything. Start anywhere.",
  "A question, a task, or a half-formed idea.",
  "Write, debug, plan, or think out loud.",
  "Paste the messy part.",
  "Built to run on your machine.",
  "Local when you want it.",
  "What should we figure out?",
  "One prompt is enough to begin.",
  "Turn a rough idea into a plan.",
  "Ask for the short version or the deep one.",
  "Drop in code, a file, or a question.",
  "Let's get the first draft down.",
  "What would make today easier?",
  "Start with the part that's stuck.",
  "Plain answers, no detours.",
  "What are we improving?",
  "Think out loud; I'll keep up.",
  "A good place to start is the problem.",
  "Bring the context; we'll find the thread.",
  "Ask for code, a plan, or an explanation.",
  "The model is ready when you are.",
  "What's the next small step?",
  "Pick up a saved prompt, or start clean.",
];

function pick(list: string[]): string {
  return list[Math.floor(Math.random() * list.length)];
}

export function sampleWelcome(part: DayPart): WelcomeCopy {
  return { headline: pick(HEADLINES[part]), subline: pick(SUBLINES) };
}

export function fillWelcome(
  template: string,
  values: { greeting: string; name: string },
): string {
  return template
    .replace(/\{greeting\}/g, values.greeting)
    .replace(/\{name\}/g, values.name);
}