export type DayPart = "morning" | "afternoon" | "evening";

export function dayPart(date = new Date()): DayPart {
  const hour = date.getHours();
  if (hour < 12) return "morning";
  if (hour < 18) return "afternoon";
  return "evening";
}

/** Time-of-day greeting, shared so the profile preview can't drift from the app. */
export function greeting(date = new Date()): string {
  return `Good ${dayPart(date)}`;
}