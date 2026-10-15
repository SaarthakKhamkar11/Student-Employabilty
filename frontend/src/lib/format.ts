// Sentinels from the backend ("Not yet evaluated", "Insufficient evidence", "Not detected", ...) are values with a
// meaning. They are never coerced to 0 or hidden: isSentinel() lets components render them literally.

export const SENTINELS = new Set([
  "Not yet evaluated",
  "Insufficient evidence",
  "Not detected",
  "Skill not detected",
  "Not available in source data",
]);

export function isSentinel(v: unknown): v is string {
  return typeof v === "string" && SENTINELS.has(v);
}

export function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

/** 0-1 suitability -> "87.3%". */
export function pct(v: number, digits = 1): string {
  return `${(v * 100).toFixed(digits)}%`;
}

export function num(v: number, digits = 3): string {
  return v.toFixed(digits);
}

export function dims(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function fmtDate(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function ms(v: number): string {
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  return `${v.toFixed(v < 10 ? 1 : 0)} ms`;
}

export function initials(name?: string | null): string {
  if (!name) return "?";
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((s) => s[0]!.toUpperCase())
    .join("");
}

export const SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];

export function hueFor(key: string): string {
  let h = 0;
  for (const c of key) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return SERIES[h % SERIES.length]!;
}
