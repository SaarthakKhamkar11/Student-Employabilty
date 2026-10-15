import type { ReactNode } from "react";
import { AlertCircle, Cpu, Inbox, Sparkles } from "lucide-react";
import { isNum, isSentinel } from "../lib/format";

/** A missing value, rendered literally. Never a blank, never a 0. */
export function NotEvaluated({ label = "Not yet evaluated", title }: { label?: string; title?: string }) {
  return (
    <span className="nye" title={title ?? "No measured value exists for this yet"}>
      {label}
    </span>
  );
}

/** Render a number with a formatter, or the backend's sentinel (or "Not yet evaluated") literally. */
export function Val({ v, fmt, missing }: { v: unknown; fmt?: (n: number) => string; missing?: string }) {
  if (isNum(v)) return <>{fmt ? fmt(v) : String(v)}</>;
  if (isSentinel(v)) return <NotEvaluated label={v} />;
  if (typeof v === "string" && v.trim()) return <>{v}</>;
  return <NotEvaluated label={missing ?? "Not yet evaluated"} />;
}

export type Tone = "good" | "warn" | "bad" | "info" | "neutral" | "ai";

export function StatusTag({ tone, children, dot = true }: { tone: Tone; children: ReactNode; dot?: boolean }) {
  return (
    <span className={`tag tag-${tone}`}>
      {dot && <span className="dot" />}
      {children}
    </span>
  );
}

export function EmptyState({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="empty">
      <div style={{ marginBottom: 6 }}>{icon ?? <Inbox size={20} />}</div>
      <b>{title}</b>
      {children && <div className="small">{children}</div>}
    </div>
  );
}

export function Loading({ label = "Loading from the backend…" }: { label?: string }) {
  return (
    <div className="loading">
      <span className="spinner" /> {label}
    </div>
  );
}

export function ErrorBox({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div className="alert alert-error" role="alert">
      <AlertCircle size={16} style={{ flexShrink: 0, marginTop: 1 }} />
      <div style={{ flex: 1 }}>{error}</div>
      {onRetry && (
        <button className="btn btn-sm" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Card({
  title,
  sub,
  right,
  children,
  className = "",
  style,
}: {
  title?: ReactNode;
  sub?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <section className={`card ${className}`} style={style}>
      {(title || right) && (
        <div className="card-head">
          <div>
            {title && <h3>{title}</h3>}
            {sub && <div className="sub">{sub}</div>}
          </div>
          {right && <div className="right">{right}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

/** Small inline sparkline (single series, no axes). Values are real data points; nothing is invented. */
export function Sparkline({ values, color, kind = "line" }: { values: number[]; color: string; kind?: "line" | "bars" }) {
  const w = 56,
    h = 26;
  if (values.length < 2) return null;
  const max = Math.max(...values, 1e-9);
  const min = Math.min(0, ...values);
  if (kind === "bars") {
    const bw = w / values.length;
    return (
      <svg width={w} height={h} aria-hidden>
        {values.map((v, i) => {
          const bh = Math.max(2, ((v - min) / (max - min || 1)) * (h - 2));
          return <rect key={i} x={i * bw + 1} y={h - bh} width={bw - 2} height={bh} rx={2} fill={color} opacity={0.35 + (0.65 * (i + 1)) / values.length} />;
        })}
      </svg>
    );
  }
  const pts = values.map((v, i) => [(i / (values.length - 1)) * w, h - 3 - ((v - min) / (max - min || 1)) * (h - 6)] as const);
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  return (
    <svg width={w} height={h} aria-hidden>
      <path d={`${d} L${w},${h} L0,${h} Z`} fill={color} opacity={0.1} />
      <path d={d} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function KpiCard({
  icon,
  color,
  label,
  value,
  foot,
  spark,
  sparkKind,
}: {
  icon: ReactNode;
  color: string;
  label: string;
  value: ReactNode;
  foot?: ReactNode;
  spark?: number[];
  sparkKind?: "line" | "bars";
}) {
  // color, spark and sparkKind are accepted for callers but no longer drawn: one quiet accent, and the label is
  // never squeezed by a sparkline
  void color, void spark, void sparkKind;
  return (
    <div className="card kpi">
      <div className="kpi-top">
        <div className="kpi-icon">{icon}</div>
        <div className="kpi-label">{label}</div>
      </div>
      <div className="kpi-value">{value}</div>
      {foot && <div className="kpi-foot">{foot}</div>}
    </div>
  );
}

export function ChartCard({ title, sub, right, children, height = 300 }: { title: string; sub?: ReactNode; right?: ReactNode; children: ReactNode; height?: number }) {
  return (
    <Card title={title} sub={sub} right={right}>
      <div style={{ width: "100%", height }}>{children}</div>
    </Card>
  );
}

export interface InsightSection {
  heading: string;
  items: string[];
  empty?: string;
}

/** Right-hand "assistant" panel. Shows stored, evidence-backed lines; the AI button is opt-in (never on load). */
export function InsightPanel({
  title,
  sub,
  sections,
  badge,
  notice,
  onAskAI,
  askLabel = "Ask AI for a summary",
  asking,
  footer,
}: {
  title: string;
  sub?: ReactNode;
  sections: InsightSection[];
  badge?: ReactNode;
  notice?: ReactNode;
  onAskAI?: () => void;
  askLabel?: string;
  asking?: boolean;
  footer?: ReactNode;
}) {
  return (
    <Card title={title} sub={sub} right={badge}>
      {notice}
      <div className="insight-scroll">
      {sections.map((s) => (
        <div key={s.heading}>
          <div className="insight-section">{s.heading}</div>
          {s.items.length ? (
            s.items.map((t, i) => (
              <div className="insight-item" key={i}>
                <Sparkles size={14} className="spark" />
                <div>{t}</div>
              </div>
            ))
          ) : (
            <div className="insight-item muted">
              <div>{s.empty ?? "Insufficient evidence"}</div>
            </div>
          )}
        </div>
      ))}
      </div>
      {onAskAI && (
        <button className="btn btn-ai" onClick={onAskAI} disabled={asking} style={{ marginTop: 6 }}>
          {asking ? <span className="spinner" /> : <Sparkles size={14} />} {askLabel}
        </button>
      )}
      {footer}
    </Card>
  );
}

export function ListRow({
  thumb,
  thumbColor,
  title,
  meta,
  tag,
  score,
  action,
  selected,
  onClick,
}: {
  thumb: ReactNode;
  thumbColor: string;
  title: ReactNode;
  meta?: ReactNode;
  tag?: ReactNode;
  score?: ReactNode;
  action?: ReactNode;
  selected?: boolean;
  onClick?: () => void;
}) {
  return (
    <div className={`list-row ${selected ? "selected" : ""}`} onClick={onClick} style={onClick ? { cursor: "pointer" } : undefined}>
      <div className="list-thumb" data-color={thumbColor}>
        {thumb}
      </div>
      <div className="list-main">
        <div className="list-title">{title}</div>
        {meta && <div className="list-meta">{meta}</div>}
      </div>
      {tag}
      {score !== undefined && <div className="score-pill">{score}</div>}
      {action}
    </div>
  );
}

/** Which models produced a result (Rule 16 metadata). Shown, not hidden. */
export function ModelBadge({ versions, compact }: { versions?: Record<string, unknown> | null; compact?: boolean }) {
  if (!versions || typeof versions !== "object") return <NotEvaluated label="Model version: not yet evaluated" />;
  const v = versions as Record<string, string>;
  const matching = v.matching_model_version ?? v.matching ?? v.matching_model;
  const emb = v.embedding_version ? `${v.embedding_model ?? ""} ${v.embedding_version}`.trim() : v.embedding;
  const title = Object.entries(v)
    .map(([k, x]) => `${k}: ${typeof x === "string" ? x : JSON.stringify(x)}`)
    .join("\n");
  return (
    <span className="model-badge" title={title}>
      <Cpu size={12} />
      {compact ? (
        <span>{matching ?? emb ?? "model"}</span>
      ) : (
        <span>
          {matching ? <>matching <b>{matching}</b></> : null}
          {matching && emb ? " · " : null}
          {emb ? <>embedding <b>{emb}</b></> : null}
          {!matching && !emb ? "model metadata" : null}
        </span>
      )}
    </span>
  );
}

export function Steps({ steps, current }: { steps: string[]; current: number }) {
  return (
    <div className="steps">
      {steps.map((s, i) => (
        <div key={s} style={{ display: "contents" }}>
          {i > 0 && <span className="step-sep" />}
          <div className={`step ${i < current ? "done" : i === current ? "current" : ""}`}>
            <span className="n">{i < current ? "✓" : i + 1}</span>
            {s}
          </div>
        </div>
      ))}
    </div>
  );
}

export function ScoreBar({ value, color }: { value: number; color?: string }) {
  return (
    <div className="bar-track" style={{ flex: 1 }}>
      <div className="bar-fill" style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: color }} />
    </div>
  );
}
