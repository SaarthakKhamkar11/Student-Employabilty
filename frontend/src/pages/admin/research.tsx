// Research-dashboard building blocks: the spec §31 leaderboard table, metric bars, the SHAP beeswarm and the agent
// trace tree. Every value is read from the admin API (experiment_runs / experiment_metrics / agent_runs).
import { Fragment, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, ChevronDown, ChevronRight } from "lucide-react";
import { fmtDate, ms } from "../../lib/format";
import { NotEvaluated, StatusTag } from "../../components/ui";

export interface Cell { value: number | null; status: string; split: string | null }
export interface Column { key: string; label: string; metric: string; fmt: string; unit?: string; note?: string; per_k?: boolean }
export interface LbRow { model: string; display: string; role: string | null; slot: string | null; version: string | null; selected: boolean; eligible: boolean | null; cells: Record<string, Record<string, Cell>> }
export interface ExperimentSummary {
  experiment_id: string; kind: string; finished_at: string; started_at: string; runtime_seconds: number; dataset_version: string;
  dataset_size: Record<string, unknown>; feature_version: string; embedding_model: string; embedding_version: string; prediction_model: string;
  random_seed: number; hardware: Record<string, unknown>; labels: Record<string, unknown>; validity: string | null;
  recommendation: { decision?: string; model?: string; line?: string } | null; best_models: Record<string, string> | null;
  config: Record<string, unknown> | null; notes: string; limitations: string; metric_rows?: number | unknown[];
}
export interface Leaderboard { experiment: ExperimentSummary; ks: number[]; primary_k: number; columns: Column[]; rows: LbRow[]; not_run: Record<string, { display: string; status: string; reason: string; spec_slot?: string }> }

export function fmtCell(c: Cell | undefined, fmt: string) {
  if (!c || c.value === null || c.value === undefined) {
    const label = c?.status === "not_feasible" ? "Not feasible" : "Not yet evaluated";
    return <NotEvaluated label={label} title={c ? `status: ${c.status}` : "no metric row recorded for this model"} />;
  }
  const v = c.value;
  const text = fmt === "score" ? v.toFixed(3) : fmt === "ms" ? (v >= 100 ? v.toFixed(0) : v.toFixed(2)) : fmt === "mb" ? v.toFixed(1) : fmt === "s" ? v.toFixed(2) : String(Math.round(v));
  return <span title={`${c.status}${c.split && c.split !== "n/a" ? ` · ${c.split} split` : ""}`}>{text}{c.status === "extrapolated" ? "*" : ""}</span>;
}

export function LeaderboardTable({ lb: raw, sortKey }: { lb: Leaderboard; sortKey?: string }) {
  // Overall Score (the ranking key) goes right after the model name, so it is never off-screen
  const lb = { ...raw, columns: [...raw.columns.filter((c) => c.key === "overall"), ...raw.columns.filter((c) => c.key !== "overall")] };
  const overall = (r: LbRow) => r.cells.overall ? Object.values(r.cells.overall)[0]?.value ?? -1 : -1;
  const rows = [...lb.rows].sort((a, b) => (sortKey ? 0 : overall(b) - overall(a)));
  return (
    <div className="table-wrap">
      <table className="data" style={{ fontSize: 12.5 }}>
        <thead>
          <tr>
            <th>#</th><th>Model</th>
            {lb.columns.map((c) => <th className="num" key={c.key} title={c.note ?? c.unit}>{c.label}{c.per_k ? <div className="tiny muted" style={{ fontWeight: 400 }}>{lb.ks.map((k) => `@${k}`).join(" / ")}</div> : c.unit ? <div className="tiny muted" style={{ fontWeight: 400 }}>{c.unit.split(" (")[0]}</div> : null}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.model} className={r.selected ? "highlight" : ""}>
              <td>{overall(r) >= 0 ? i + 1 : "—"}</td>
              <td style={{ minWidth: 170, maxWidth: 230 }}>
                <b>{r.display}</b>
                {r.selected && <> <StatusTag tone="good">selected</StatusTag></>}
                {r.role === "floor" && <> <StatusTag tone="neutral">floor</StatusTag></>}
                {r.eligible === false && r.role !== "floor" && <> <StatusTag tone="warn">not eligible</StatusTag></>}
                <div className="tiny muted mono" style={{ overflowWrap: "anywhere" }}>{r.version ?? r.model}</div>
              </td>
              {lb.columns.map((c) => {
                const cells = r.cells[c.key] ?? {};
                if (!c.per_k) return <td className="num" key={c.key}>{fmtCell(Object.values(cells)[0], c.fmt)}</td>;
                const vals = lb.ks.map((k) => cells[c.metric.replace("{k}", String(k))]);
                return <td className="num" key={c.key} style={{ whiteSpace: "nowrap" }}>{vals.every((v) => !v || v.value === null) ? fmtCell(vals[0], c.fmt) : vals.map((v, i) => <span key={i}>{i ? <span className="muted"> / </span> : null}{fmtCell(v, c.fmt)}</span>)}</td>;
              })}
            </tr>
          ))}
          {Object.entries(lb.not_run).map(([k, n]) => (
            <tr key={k}>
              <td>—</td>
              <td><b>{n.display}</b><div className="tiny muted">{n.status}</div></td>
              <td colSpan={lb.columns.length} title={n.reason} style={{ whiteSpace: "normal" }}><NotEvaluated label="Not yet evaluated" /> <span className="tiny muted">{n.reason}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="tiny muted" style={{ marginTop: 8 }}>
        {lb.columns.filter((c) => c.unit || c.note).map((c) => `${c.label}: ${c.note ?? c.unit}`).join(" · ")}. * extrapolated, not measured. Hover a cell for its status and split.
      </div>
    </div>
  );
}

/** One metric across models (one series, so one hue; the selected model is emphasised). Missing values listed. */
export function MetricBars({ lb, column, k, height = 260, color = "#2a78d6" }: { lb: Leaderboard; column: Column; k?: number; height?: number; color?: string }) {
  const metric = column.metric.replace("{k}", String(k ?? lb.primary_k));
  const pts = lb.rows.map((r) => ({ label: r.display, value: (r.cells[column.key] ?? {})[metric]?.value ?? null, selected: r.selected }));
  const have = pts.filter((p) => p.value !== null) as { label: string; value: number; selected: boolean }[];
  const missing = pts.filter((p) => p.value === null).map((p) => p.label);
  const lowerBetter = ["ms", "mb", "s"].includes(column.fmt);
  have.sort((a, b) => (lowerBetter ? a.value - b.value : b.value - a.value));
  return (
    <div>
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <BarChart data={have} layout="vertical" margin={{ top: 4, right: 60, bottom: 4, left: 4 }} barCategoryGap={5}>
            <CartesianGrid stroke="#e6eaee" strokeDasharray="4 4" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={false} scale={column.fmt === "ms" ? "log" : "auto"} domain={column.fmt === "ms" && have.length ? [Math.min(...have.map((p) => p.value)) / 2, "auto"] : [0, "auto"]} allowDataOverflow />
            <YAxis type="category" dataKey="label" width={170} tick={{ fontSize: 11.5, fill: "#475569" }} tickLine={false} axisLine={false} tickFormatter={(v: string) => (v.length > 26 ? v.slice(0, 25) + "…" : v)} />
            <Tooltip formatter={(v: number) => [column.fmt === "score" ? v.toFixed(3) : v.toFixed(2), `${column.label.replace("@K", `@${k ?? lb.primary_k}`)}${column.unit ? ` (${column.unit})` : ""}`]} cursor={{ fill: "rgba(15,23,42,0.04)" }} />
            <Bar isAnimationActive={false} dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={16} label={{ position: "right", fontSize: 11, fill: "#475569", formatter: (v: number) => (column.fmt === "score" ? v.toFixed(3) : v >= 100 ? v.toFixed(0) : v.toFixed(2)) }}>
              {have.map((p) => <Cell key={p.label} fill={color} opacity={p.selected ? 1 : 0.55} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="tiny muted">
        {lowerBetter ? "Lower is better" : "Higher is better"}{column.fmt === "ms" ? " · log scale" : ""}. Solid bar = selected model.
        {missing.length ? <> Not yet evaluated: {missing.join(", ")}.</> : null}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ SHAP beeswarm
export interface GlobalExplanation {
  matching_model: string; matching_model_version: string; artifacts_experiment: string; embedding_model: string | null; embedding_version: string;
  feature_version: string; feature_matrix: string; explainer: string; output_space: string; base_value: number; split: string; rows: number;
  additivity_max_abs_error: number;
  importance_ranking: { rank: number; feature: string; mean_abs_shap: number; used_by_model: boolean; skill_feature: boolean; tied: boolean }[];
  distributions: Record<string, { mean: number; std: number; mean_abs: number; min: number; max: number; p05: number; p50: number; p95: number; share_positive: number; share_negative: number; rows_missing_value: number; histogram: { edges: number[]; counts: number[] } }>;
  summary_plot: { rows: number; sampling: string; features: Record<string, { feature_value: (number | null)[]; shap: number[] }>; pair_ids?: string[] };
  notes: string[];
}

const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]; // sequential blue, low -> high

export function Beeswarm({ g, features }: { g: GlobalExplanation; features: string[] }) {
  const W = 760, rowH = 34, left = 190, right = 20;
  const all = features.flatMap((f) => g.summary_plot.features[f]?.shap ?? []);
  if (!all.length) return <NotEvaluated label="No summary-plot sample" />;
  const lo = Math.min(...all), hi = Math.max(...all);
  const x = (v: number) => left + ((v - lo) / (hi - lo || 1)) * (W - left - right);
  const H = features.length * rowH + 30;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="SHAP summary plot">
      <line x1={x(0)} x2={x(0)} y1={0} y2={H - 24} stroke="#94a3b8" strokeDasharray="3 3" />
      {features.map((f, i) => {
        const d = g.summary_plot.features[f];
        if (!d) return null;
        const vals = d.feature_value.filter((v): v is number => v !== null);
        const vlo = Math.min(...vals), vhi = Math.max(...vals);
        const cy = i * rowH + rowH / 2;
        return (
          <g key={f}>
            <text x={left - 8} y={cy + 4} textAnchor="end" fontSize={11.5} fill="#334155">{f}</text>
            <line x1={left} x2={W - right} y1={cy} y2={cy} stroke="#eef2f6" />
            {d.shap.map((s, j) => {
              const v = d.feature_value[j];
              const color = v === null || v === undefined ? "#c0c6cf" : RAMP[Math.min(RAMP.length - 1, Math.floor(((v - vlo) / (vhi - vlo || 1)) * (RAMP.length - 0.001)))]!;
              const jitter = (((j * 2654435761) % 1000) / 1000 - 0.5) * (rowH * 0.62);
              return <circle key={j} cx={x(s)} cy={cy + jitter} r={2.4} fill={color} opacity={0.85}><title>{`${f} = ${v ?? "missing"} → SHAP ${s.toFixed(3)}`}</title></circle>;
            })}
          </g>
        );
      })}
      {[lo, 0, hi].map((t) => <text key={t} x={x(t)} y={H - 8} textAnchor="middle" fontSize={11} fill="#64748b">{t.toFixed(1)}</text>)}
    </svg>
  );
}

export function RampLegend() {
  return (
    <div className="row tiny muted" style={{ gap: 6 }}>
      feature value: low {RAMP.map((c) => <i key={c} style={{ display: "inline-block", width: 14, height: 8, background: c }} />)} high · <i style={{ display: "inline-block", width: 10, height: 8, background: "#c0c6cf" }} /> missing
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ trace tree
export interface TraceNode {
  kind: "supervisor" | "agent" | "tool" | "llm" | "skip"; name: string; id: string; status: string; started_at: string; finished_at: string; duration_ms: number;
  attempt?: number | null; retry_count?: number | null; tool_call_count?: number | null; tool_latency_ms?: number | null; model_used?: string | null;
  model_version?: string | null; termination_reason?: string | null; error?: string | null; details: Record<string, unknown>; children: TraceNode[];
}
export interface TraceTree { request_id: string; task_type: string; status: string; started_at: string; finished_at: string; duration_ms: number; routing_source: string | null; root: TraceNode; counts: Record<string, number> }

const KIND_LABEL: Record<string, string> = { supervisor: "Supervisor", agent: "Agent", tool: "Tool call", llm: "LLM call", skip: "Skipped" };

function Node({ n, t0, total, depth }: { n: TraceNode; t0: number; total: number; depth: number }) {
  const [open, setOpen] = useState(depth < 2);
  const start = new Date(n.started_at).getTime() - t0;
  const kind = n.kind === "supervisor" ? "agent" : n.kind;
  const ok = ["success", "ok", "skipped"].includes(n.status);
  return (
    <div className={`trace-node ${depth === 0 ? "trace-root" : ""}`}>
      <div className={`trace-box kind-${kind} status-${n.status}`}>
        {n.children.length > 0 ? (
          <button onClick={() => setOpen(!open)} style={{ border: 0, background: "none", cursor: "pointer", padding: 0 }} aria-label="toggle">{open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</button>
        ) : <span style={{ width: 14 }} />}
        <span className="tiny muted" style={{ width: 62 }}>{KIND_LABEL[n.kind]}</span>
        <b style={{ fontSize: 13 }}>{n.name}</b>
        {n.attempt && n.attempt > 1 ? <StatusTag tone="warn">attempt {n.attempt}</StatusTag> : null}
        <StatusTag tone={ok ? (n.status === "skipped" ? "neutral" : "good") : n.status === "partial" ? "warn" : "bad"}>{n.status}</StatusTag>
        {n.model_used && <span className="tiny muted">LLM {n.model_used}</span>}
        {n.model_version && <span className="tiny muted mono">{n.model_version}</span>}
        {n.error && <span className="tiny" style={{ color: "var(--bad)" }}>{n.error}</span>}
        {n.kind === "skip" && n.details?.reason ? <span className="tiny muted">{String(n.details.reason)}</span> : null}
        {n.kind === "llm" && n.details?.http_status ? <StatusTag tone="bad">HTTP {String(n.details.http_status)}</StatusTag> : null}
        <span className="trace-dur">
          <span className="trace-bar" title={`starts at +${ms(start)}`}><i style={{ left: `${(start / total) * 100}%`, width: `${Math.max(1.5, (n.duration_ms / total) * 100)}%`, background: kind === "tool" ? "var(--series-3)" : kind === "llm" ? "var(--series-7)" : kind === "skip" ? "#94a3b8" : "var(--series-1)" }} /></span>
          {ms(n.duration_ms)}
        </span>
      </div>
      {open && n.children.map((c) => <Node key={c.id} n={c} t0={t0} total={total} depth={depth + 1} />)}
    </div>
  );
}

export function TraceView({ tree }: { tree: TraceTree }) {
  const t0 = new Date(tree.started_at).getTime();
  return (
    <div className="trace">
      <div className="row small" style={{ marginBottom: 10, gap: 14 }}>
        <span>Task <b>{tree.task_type}</b></span>
        <span>Routing <b>{tree.routing_source ?? "—"}</b></span>
        <span>Started {fmtDate(tree.started_at)}</span>
        <span>Total <b>{ms(tree.duration_ms)}</b></span>
        {Object.entries(tree.counts).map(([k, v]) => <span key={k}>{k.replace(/_/g, " ")} <b>{v}</b></span>)}
      </div>
      <div className="legend" style={{ marginBottom: 8 }}>
        <span><i style={{ background: "var(--series-1)" }} />Supervisor / agent</span>
        <span><i style={{ background: "var(--series-3)" }} />Tool call</span>
        <span><i style={{ background: "var(--series-7)" }} />LLM call</span>
        <span><i style={{ background: "#94a3b8" }} />Skipped agent</span>
        <span>Bars: when each step ran within the request (waterfall)</span>
      </div>
      <Node n={tree.root} t0={t0} total={Math.max(tree.duration_ms, 1)} depth={0} />
    </div>
  );
}

export function Kv({ obj }: { obj: Record<string, unknown> | null | undefined }) {
  if (!obj) return <NotEvaluated />;
  return (
    <dl className="kv">
      {Object.entries(obj).map(([k, v]) => (
        <Fragment key={k}>
          <dt>{k.replace(/_/g, " ")}</dt>
          <dd>{v === null || v === undefined ? <NotEvaluated label="Not recorded" /> : typeof v === "object" ? <code style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(v)}</code> : typeof v === "boolean" ? (v ? <CheckCircle2 size={13} color="var(--good)" /> : "no") : String(v)}</dd>
        </Fragment>
      ))}
    </dl>
  );
}
