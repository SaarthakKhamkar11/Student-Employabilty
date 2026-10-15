import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const AXIS = { fontSize: 11.5, fill: "#64748b" };
const GRID = "#e6eaee";

interface Point {
  label: string;
  value: number;
  hint?: string;
}

function Tip({ active, payload, unit }: { active?: boolean; payload?: { payload: Point }[]; unit: string }) {
  if (!active || !payload?.length) return null;
  const p = payload[0]!.payload;
  return (
    <div className="card" style={{ padding: "8px 10px", fontSize: 12.5, boxShadow: "var(--shadow)" }}>
      <b>{p.label}</b>
      {p.hint && <div className="muted">{p.hint}</div>}
      <div>
        {p.value.toFixed(unit === "%" ? 1 : 3)}
        {unit}
      </div>
    </div>
  );
}

/** One smooth area series (the reference style's main visual). Single series: no legend, the card title names it. */
export function AreaSeries({ points, unit = "%", color = "#16a34a", domain }: { points: Point[]; unit?: string; color?: string; domain?: [number, number] }) {
  const id = `g${color.replace("#", "")}`;
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={points} margin={{ top: 10, right: 12, bottom: 4, left: 0 }}>
        <defs>
          <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.22} />
            <stop offset="100%" stopColor={color} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={GRID} strokeDasharray="4 4" vertical={false} />
        <XAxis dataKey="label" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} interval={0} height={44} tickFormatter={(v: string) => (v.length > 14 ? v.slice(0, 13) + "…" : v)} />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} width={44} domain={domain ?? [0, "auto"]} tickFormatter={(v: number) => `${unit === "%" ? v.toFixed(0) : v.toFixed(2)}${unit === "%" ? "%" : ""}`} />
        <Tooltip content={<Tip unit={unit} />} cursor={{ stroke: "#94a3b8", strokeDasharray: "3 3" }} />
        <Area isAnimationActive={false} type="monotone" dataKey="value" stroke={color} strokeWidth={2.2} fill={`url(#${id})`} dot={{ r: 4, fill: color, stroke: "#fff", strokeWidth: 2 }} activeDot={{ r: 6 }} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/** Horizontal bars, one series. `highlight` marks the selected entity; everything else is the same hue. */
export function HBars({
  points,
  unit = "",
  color = "#2a78d6",
  highlight,
  domain,
  digits = 3,
}: {
  points: Point[];
  unit?: string;
  color?: string;
  highlight?: string;
  domain?: [number, number | "auto"];
  digits?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={points} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 4 }} barCategoryGap={6}>
        <CartesianGrid stroke={GRID} strokeDasharray="4 4" horizontal={false} />
        <XAxis type="number" tick={AXIS} tickLine={false} axisLine={false} domain={domain ?? [0, "auto"]} />
        <YAxis type="category" dataKey="label" tick={AXIS} tickLine={false} axisLine={false} width={150} tickFormatter={(v: string) => (v.length > 22 ? v.slice(0, 21) + "…" : v)} />
        <Tooltip content={<Tip unit={unit} />} cursor={{ fill: "rgba(15,23,42,0.04)" }} />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive={false}>
          {points.map((p) => (
            <Cell key={p.label} fill={color} opacity={highlight && p.label !== highlight ? 0.45 : 1} />
          ))}
          <LabelList dataKey="value" position="right" style={{ fontSize: 11.5, fill: "#475569" }} formatter={(v: number) => `${v.toFixed(digits)}${unit}`} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
