import { useState } from "react";
import { Route, Routes, useNavigate, useParams } from "react-router-dom";
import { Activity, Boxes, Brain, Database, FlaskConical, GitBranch, LayoutDashboard, LineChart, Medal, Network, Timer, Trophy } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useApi } from "../../lib/useApi";
import { fmtDate, ms } from "../../lib/format";
import { PageHead, Shell, type NavEntry } from "../../components/Shell";
import { Card, ChartCard, EmptyState, ErrorBox, KpiCard, Loading, ModelBadge, NotEvaluated, StatusTag } from "../../components/ui";
import { HBars } from "../../components/charts";
import { Beeswarm, Kv, LeaderboardTable, MetricBars, RampLegend, TraceView, type ExperimentSummary, type GlobalExplanation, type Leaderboard, type TraceTree } from "./research";

const NAV: NavEntry[] = [
  { to: "/admin", label: "Overview", icon: <LayoutDashboard size={16} />, end: true },
  { to: "/admin/embedding", label: "Embedding Models", icon: <Boxes size={16} /> },
  { to: "/admin/matching", label: "Matching Models", icon: <GitBranch size={16} /> },
  { to: "/admin/leaderboard", label: "Leaderboard", icon: <Trophy size={16} /> },
  { to: "/admin/metrics", label: "Metrics", icon: <LineChart size={16} /> },
  { to: "/admin/shap", label: "SHAP", icon: <Brain size={16} /> },
  { to: "/admin/experiments", label: "Experiments", icon: <FlaskConical size={16} /> },
  { to: "/admin/traces", label: "Agent Traces", icon: <Network size={16} /> },
];

type Kind = "embedding" | "matching";

interface Overview {
  counts: Record<string, number>;
  embeddings: { entity_type: string; model_name: string; model_version: string; entities: number }[];
  deployed: { embedding_model: string; embedding_version: string; embedding_dimension: number; embedding_fallback: string | null; matching: Record<string, unknown> };
  latest_experiments: Record<Kind, ExperimentSummary | null>;
  benchmark_configuration: Record<string, Record<string, unknown>>;
}

function useExperiments() {
  return useApi<(ExperimentSummary & { metric_rows: number })[]>("/admin/experiments");
}

function ExperimentPicker({ kind, value, onChange }: { kind: Kind; value: string; onChange: (v: string) => void }) {
  const ex = useExperiments();
  const list = (ex.data ?? []).filter((e) => e.kind === `${kind}_selection`);
  return (
    <select className="btn" value={value} onChange={(e) => onChange(e.target.value)} title="Experiment">
      <option value="">Newest human-label run (default)</option>
      {list.map((e) => <option key={e.experiment_id} value={e.experiment_id}>{e.experiment_id} · {String(e.labels.used_source ?? "no labels")}</option>)}
    </select>
  );
}

function ExperimentHeader({ e }: { e: ExperimentSummary }) {
  const ok = (e.validity ?? "").includes("OK");
  return (
    <Card>
      <div className="row" style={{ gap: 18 }}>
        <div><div className="tiny muted">Experiment</div><b className="mono">{e.experiment_id}</b></div>
        <div><div className="tiny muted">Finished</div>{fmtDate(e.finished_at)}</div>
        <div><div className="tiny muted">Labels</div>{String(e.labels.used_source ?? "none")} · {String(e.labels.used_split ?? "—")} split · {String(e.labels.used_pairs ?? 0)} pairs</div>
        <div><div className="tiny muted">Dataset version</div><span className="mono">{e.dataset_version}</span></div>
        <div style={{ flex: 1, minWidth: 260 }}>{e.validity ? <StatusTag tone={ok ? "good" : "warn"}>{e.validity}</StatusTag> : <NotEvaluated label="validity not recorded" />}</div>
      </div>
      {e.recommendation?.line && <div className="alert alert-info" style={{ margin: "12px 0 0" }}><Medal size={15} /> <div><b>Engine decision ({e.recommendation.decision}):</b> {e.recommendation.line}</div></div>}
    </Card>
  );
}

// ------------------------------------------------------------------------------------------------- Overview
function OverviewPage() {
  const o = useApi<Overview>("/admin/overview");
  const lat = useApi<{ agents: { agent_name: string; runs: number; mean_ms: number; p50_ms: number; p95_ms: number; not_success: number }[] }>("/admin/agent-latency");
  if (o.loading) return <Loading />;
  if (o.error || !o.data) return <ErrorBox error={o.error ?? "No data"} onRetry={o.reload} />;
  const d = o.data;
  const m = d.deployed.matching as { active?: string; artifacts_experiment?: string; selection_experiment?: string; embedding_model?: string; fallback?: Record<string, { active: string; selection_experiment: string }> };
  const evalCfg = (d.latest_experiments.embedding?.config as { evaluation?: Record<string, unknown> } | null)?.evaluation;
  const mEval = (d.latest_experiments.matching?.config as { evaluation?: Record<string, unknown> } | null)?.evaluation;
  return (
    <>
      <PageHead title="Research overview" sub="What is deployed, what it was selected on, and the data it runs over: all read live from the database and config." />
      <div className="stack">
        <div className="grid kpis">
          <KpiCard icon={<FlaskConical size={16} />} color="#6250d6" label="Experiments recorded" value={d.counts.experiments} foot="experiment_runs (Phases 6 & 8)" />
          <KpiCard icon={<Activity size={16} />} color="#2a78d6" label="Requests traced" value={d.counts.requests} foot={`${d.counts.trace_messages} trace messages`} />
          <KpiCard icon={<Database size={16} />} color="#1baf7a" label="Student profiles" value={d.counts.profiles} foot={`${d.counts.users} accounts`} />
          <KpiCard icon={<Boxes size={16} />} color="#eb6834" label="Jobs" value={d.counts.jobs} foot={`${d.counts.portal_jobs} posted via the portal`} />
          <KpiCard icon={<Medal size={16} />} color="#e87ba4" label="Feedback events" value={d.counts.feedback} foot="shortlists etc. (Phase 15 input)" />
        </div>
        <div className="grid three">
          <Card title="Deployed embedding model">
            <dl className="kv">
              <dt>Model</dt><dd><b>{d.deployed.embedding_model}</b></dd>
              <dt>Version</dt><dd className="mono">{d.deployed.embedding_version}</dd>
              <dt>Dimension</dt><dd>{d.deployed.embedding_dimension}</dd>
              <dt>Fallback active</dt><dd>{d.deployed.embedding_fallback ? <StatusTag tone="warn">{d.deployed.embedding_fallback}</StatusTag> : <StatusTag tone="good">no: primary model in use</StatusTag>}</dd>
            </dl>
          </Card>
          <Card title="Deployed matching model" sub="config/matching_model.yaml">
            <dl className="kv">
              <dt>Active</dt><dd><b>{m.active}</b> on {m.embedding_model}</dd>
              <dt>Trained in</dt><dd className="mono">{m.artifacts_experiment}</dd>
              <dt>Selected in</dt><dd className="mono">{m.selection_experiment}</dd>
              {m.fallback && Object.entries(m.fallback).map(([k, v]) => <div key={k} style={{ display: "contents" }}><dt>Fallback ({k})</dt><dd>{v.active} · <span className="mono">{v.selection_experiment}</span></dd></div>)}
            </dl>
          </Card>
          <Card title="Dataset version" sub="As recorded by the latest selection runs">
            <dl className="kv">
              <dt>Embedding run</dt><dd className="mono">{d.latest_experiments.embedding?.dataset_version ?? "—"}</dd>
              <dt>Matching run</dt><dd className="mono">{d.latest_experiments.matching?.dataset_version ?? "—"}</dd>
              <dt>Labels</dt><dd>{String(d.latest_experiments.matching?.labels.path ?? "—")} · {String(d.latest_experiments.matching?.labels.pairs ?? "—")} pairs</dd>
              <dt>Labels sha256</dt><dd className="mono tiny">{String(d.latest_experiments.matching?.labels.sha256 ?? "—").slice(0, 24)}…</dd>
              <dt>Features</dt><dd className="mono">{d.latest_experiments.matching?.feature_version ?? "—"}</dd>
            </dl>
          </Card>
        </div>
        <div className="grid two">
          <Card title="Benchmark configuration" sub="Evaluation protocol the engines recorded with each run">
            <div className="insight-section" style={{ marginTop: 0 }}>Embedding selection</div>
            <Kv obj={evalCfg ?? null} />
            <div className="insight-section">Matching selection</div>
            <Kv obj={mEval ?? null} />
            <details className="disclosure" style={{ marginTop: 10 }}>
              <summary>Selection weights and candidates (config files)</summary>
              {Object.entries(d.benchmark_configuration).map(([k, v]) => <div key={k} style={{ marginTop: 8 }}><b className="small">{k}</b><pre className="mono" style={{ whiteSpace: "pre-wrap", background: "var(--surface-2)", padding: 10, borderRadius: 8, maxHeight: 220, overflow: "auto" }}>{JSON.stringify(v, null, 1)}</pre></div>)}
            </details>
          </Card>
          <Card title="Stored embeddings" sub="Vectors in the database, by model">
            <table className="data">
              <thead><tr><th>Entity</th><th>Model</th><th>Version</th><th className="num">Entities</th></tr></thead>
              <tbody>{d.embeddings.map((e) => <tr key={`${e.entity_type}${e.model_name}`}><td>{e.entity_type}</td><td>{e.model_name}</td><td className="mono tiny">{e.model_version}</td><td className="num">{e.entities}</td></tr>)}</tbody>
            </table>
            <div className="insight-section">Agent latency (persisted traces)</div>
            {lat.data ? (
              <div style={{ height: Math.max(160, lat.data.agents.length * 30) }}>
                <HBars unit=" ms" digits={0} points={lat.data.agents.map((a) => ({ label: a.agent_name, value: a.p50_ms, hint: `${a.runs} runs · mean ${ms(a.mean_ms)} · p95 ${ms(a.p95_ms)}${a.not_success ? ` · ${a.not_success} not successful` : ""}` }))} />
              </div>
            ) : <Loading />}
            <div className="tiny muted">Median per agent execution; hover for mean, p95 and failures.</div>
          </Card>
        </div>
      </div>
    </>
  );
}

// ------------------------------------------------------------------------------------------------- Model comparison
function ComparisonPage({ kind }: { kind: Kind }) {
  const [exp, setExp] = useState("");
  const lb = useApi<Leaderboard>(`/admin/leaderboard/${kind}${exp ? `?experiment_id=${exp}` : ""}`);
  const title = kind === "embedding" ? "Embedding model comparison" : "Matching model comparison";
  const charts = kind === "embedding" ? ["quality", "mrr", "latency", "memory"] : ["f1", "ndcg", "latency", "training_time"];
  return (
    <>
      <PageHead title={title} sub={`Phase ${kind === "embedding" ? 6 : 8}'s Model Evaluation Engine, read from experiment_runs / experiment_metrics.`} actions={<ExperimentPicker kind={kind} value={exp} onChange={setExp} />} />
      {lb.loading ? <Loading /> : lb.error || !lb.data ? <ErrorBox error={lb.error ?? "No data"} /> : (
        <div className="stack">
          <ExperimentHeader e={lb.data.experiment} />
          <div className="grid two">
            {charts.map((key) => {
              const col = lb.data!.columns.find((c) => c.key === key)!;
              return (
                <Card key={key} title={col.label.replace("@K", `@${lb.data!.primary_k}`)} sub={col.note ?? col.unit}>
                  <MetricBars lb={lb.data!} column={col} height={Math.max(200, lb.data!.rows.length * 30)} color={key === "latency" || key === "memory" || key === "training_time" ? "#eb6834" : "#2a78d6"} />
                </Card>
              );
            })}
          </div>
          {lb.data.experiment.best_models && (
            <Card title="Engine's findings" sub="Recorded verbatim in the experiment report">
              {Object.entries(lb.data.experiment.best_models).map(([k, v]) => <div key={k} className="insight-item"><b style={{ minWidth: 120, textTransform: "capitalize" }}>{k.replace(/_/g, " ")}</b><div>{v}</div></div>)}
            </Card>
          )}
          <Card title="Full table"><LeaderboardTable lb={lb.data} /></Card>
        </div>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------------- Leaderboard
function LeaderboardPage() {
  const emb = useApi<Leaderboard>("/admin/leaderboard/embedding");
  const mat = useApi<Leaderboard>("/admin/leaderboard/matching");
  return (
    <>
      <PageHead title="Model leaderboard" sub="Spec §31 columns, ranked by each engine's Overall Score (balanced scenario). Missing metrics read Not yet evaluated, never 0." />
      <div className="stack">
        {[["Embedding leaderboard", emb], ["Matching leaderboard", mat]].map(([t, q]) => {
          const s = q as typeof emb;
          return (
            <Card key={t as string} title={t as string} sub={s.data ? <>Experiment <span className="mono">{s.data.experiment.experiment_id}</span> · {s.data.experiment.validity}</> : undefined}>
              {s.loading ? <Loading /> : s.error || !s.data ? <ErrorBox error={s.error ?? "No data"} /> : <LeaderboardTable lb={s.data} />}
            </Card>
          );
        })}
      </div>
    </>
  );
}

// ------------------------------------------------------------------------------------------------- Metrics
function MetricsPage() {
  const [kind, setKind] = useState<Kind>("matching");
  const [k, setK] = useState(10);
  const lb = useApi<Leaderboard>(`/admin/leaderboard/${kind}`);
  const groups: Record<Kind, string[]> = {
    embedding: ["mrr", "ndcg", "precision", "recall", "latency", "memory", "size"],
    matching: ["accuracy", "precision", "recall", "f1", "mrr", "ndcg", "latency", "memory"],
  };
  return (
    <>
      <PageHead title="Metrics" sub="Latency, memory, accuracy / F1, MRR, NDCG@K, Precision@K and Recall@K from the stored runs."
        actions={<>
          <div className="tabs" style={{ margin: 0, border: 0 }}>{(["embedding", "matching"] as Kind[]).map((x) => <button key={x} className={`tab ${kind === x ? "active" : ""}`} onClick={() => setKind(x)}>{x}</button>)}</div>
          <select className="btn" value={k} onChange={(e) => setK(Number(e.target.value))}>{(lb.data?.ks ?? [5, 10]).map((x) => <option key={x} value={x}>K = {x}</option>)}</select>
        </>} />
      {lb.loading ? <Loading /> : lb.error || !lb.data ? <ErrorBox error={lb.error ?? "No data"} /> : (
        <div className="stack">
          <ExperimentHeader e={lb.data.experiment} />
          <div className="grid two">
            {groups[kind].map((key) => {
              const col = lb.data!.columns.find((c) => c.key === key);
              if (!col) return null;
              const cost = ["ms", "mb", "s"].includes(col.fmt);
              return (
                <Card key={key} title={col.per_k ? col.label.replace("@K", `@${k}`) : col.label} sub={col.unit ?? col.note}>
                  <MetricBars lb={lb.data!} column={col} k={k} height={Math.max(200, lb.data!.rows.length * 30)} color={cost ? "#eb6834" : "#2a78d6"} />
                </Card>
              );
            })}
          </div>
          <History kind={kind} />
        </div>
      )}
    </>
  );
}

/** One metric across every recorded run of the engine: how the numbers moved as labels changed. */
function History({ kind }: { kind: Kind }) {
  const rows = useApi<{ experiment_id: string; finished_at: string; label_source: string | null; model_name: string; metric: string; split: string; value: number | null; status: string }[]>(`/admin/metrics/${kind}`);
  const [metric, setMetric] = useState(kind === "embedding" ? "ndcg@5" : "ndcg@10");
  if (!rows.data) return rows.loading ? <Loading /> : null;
  const metrics = Array.from(new Set(rows.data.map((r) => r.metric))).sort();
  const runs = Array.from(new Set(rows.data.map((r) => r.experiment_id)));
  const models = Array.from(new Set(rows.data.map((r) => r.model_name)));
  return (
    <Card title="Across recorded runs" sub="The same metric in every stored experiment of this engine" right={<select className="btn btn-sm" value={metric} onChange={(e) => setMetric(e.target.value)}>{metrics.map((m) => <option key={m}>{m}</option>)}</select>}>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>Model</th>{runs.map((r) => { const x = rows.data!.find((y) => y.experiment_id === r)!; return <th key={r} className="num">{r.replace(/^(emb|mms)-/, "")}<div className="tiny muted" style={{ fontWeight: 400 }}>{x.label_source ?? "no labels"}</div></th>; })}</tr></thead>
          <tbody>
            {models.map((m) => (
              <tr key={m}><td>{m}</td>{runs.map((r) => { const c = rows.data!.find((y) => y.experiment_id === r && y.model_name === m && y.metric === metric); return <td key={r} className="num">{c && c.value !== null ? c.value.toFixed(3) : <NotEvaluated label={c?.status === "not_feasible" ? "Not feasible" : "Not yet evaluated"} />}</td>; })}</tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="tiny muted" style={{ marginTop: 6 }}>Weakly-labelled runs are diagnostics, not selections (their labels come from rules, not people).</div>
    </Card>
  );
}

// ------------------------------------------------------------------------------------------------- SHAP
function ShapPage() {
  const s = useApi<{ global_explanation: GlobalExplanation; duration_ms: number }>("/admin/shap");
  const [feat, setFeat] = useState<string | null>(null);
  if (s.loading) return <Loading label="The Explainability Agent is computing global SHAP values (a few seconds)…" />;
  if (s.error || !s.data) return <ErrorBox error={s.error ?? "No data"} onRetry={s.reload} />;
  const g = s.data.global_explanation;
  const top = g.importance_ranking.slice(0, 12).map((r) => r.feature);
  const f = feat ?? top[0]!;
  const dist = g.distributions[f];
  return (
    <>
      <PageHead title="SHAP summaries" sub="get_global_explanation (Phase 11) for the active matching model, over its validation split." actions={<ModelBadge versions={{ matching_model_version: g.matching_model_version, embedding_model: g.embedding_model ?? "", embedding_version: g.embedding_version }} />} />
      <div className="stack">
        <div className="grid kpis">
          <KpiCard icon={<Brain size={16} />} color="#6250d6" label="Rows explained" value={g.rows.toLocaleString()} foot={`${g.split} split`} />
          <KpiCard icon={<Activity size={16} />} color="#2a78d6" label="Base value" value={g.base_value.toFixed(3)} foot={g.output_space} />
          <KpiCard icon={<Timer size={16} />} color="#1baf7a" label="Additivity error" value={g.additivity_max_abs_error.toExponential(1)} foot="max |sum(SHAP) + base − output|" />
          <KpiCard icon={<Boxes size={16} />} color="#eb6834" label="Features" value={g.importance_ranking.length} foot={g.feature_version} />
        </div>
        <div className="grid split-main">
          <ChartCard title="Global importance" sub="Mean |SHAP| per feature (log-odds)" height={Math.max(300, g.importance_ranking.length * 24)}>
            <HBars digits={3} highlight={f} points={g.importance_ranking.map((r) => ({ label: r.feature, value: r.mean_abs_shap, hint: `rank ${r.rank}${r.tied ? " (tied)" : ""}${r.used_by_model ? "" : " · not used by the model"}` }))} />
          </ChartCard>
          <Card title="Explainer">
            <Kv obj={{ explainer: g.explainer, matching_model: g.matching_model_version, trained_in: g.artifacts_experiment, feature_matrix: g.feature_matrix, computed_in: ms(s.data.duration_ms) }} />
            <div className="insight-section">Notes from the agent</div>
            {g.notes.map((n, i) => <div className="insight-item" key={i}>{n}</div>)}
          </Card>
        </div>
        <Card title="Summary plot" sub={`Each dot is one student-job pair (${g.summary_plot.sampling}); position = SHAP value, colour = feature value`} right={<RampLegend />}>
          <Beeswarm g={g} features={top} />
        </Card>
        <Card title={`Distribution: ${f}`} right={<select className="btn btn-sm" value={f} onChange={(e) => setFeat(e.target.value)}>{g.importance_ranking.map((r) => <option key={r.feature}>{r.feature}</option>)}</select>}>
          {dist ? (
            <div className="grid split-main">
              <div style={{ height: 230 }}>
                <ResponsiveContainer>
                  <BarChart data={dist.histogram.counts.map((c, i) => ({ bin: `${dist.histogram.edges[i]!.toFixed(2)}`, count: c }))} margin={{ top: 6, right: 8, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke="#e6eaee" strokeDasharray="4 4" vertical={false} />
                    <XAxis dataKey="bin" tick={{ fontSize: 10.5, fill: "#64748b" }} interval={3} tickLine={false} />
                    <YAxis tick={{ fontSize: 11, fill: "#64748b" }} tickLine={false} axisLine={false} width={46} />
                    <Tooltip formatter={(v: number) => [v, "pairs"]} labelFormatter={(l) => `SHAP from ${l}`} />
                    <Bar isAnimationActive={false} dataKey="count" fill="#6250d6" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <Kv obj={{ mean: dist.mean.toFixed(3), "mean |SHAP|": dist.mean_abs.toFixed(3), p05: dist.p05.toFixed(3), median: dist.p50.toFixed(3), p95: dist.p95.toFixed(3), "share raising the score": `${(dist.share_positive * 100).toFixed(1)}%`, "rows with the value missing": dist.rows_missing_value }} />
            </div>
          ) : <NotEvaluated />}
        </Card>
      </div>
    </>
  );
}

// ------------------------------------------------------------------------------------------------- Experiments
function ExperimentsPage() {
  const ex = useExperiments();
  const navigate = useNavigate();
  return (
    <>
      <PageHead title="Experiment history" sub="Every run the Phase 6 and Phase 8 engines recorded (experiment_runs)." />
      <Card>
        {ex.loading ? <Loading /> : ex.error ? <ErrorBox error={ex.error} /> : ex.data?.length ? (
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>Experiment</th><th>Engine</th><th>Finished</th><th>Labels</th><th>Validity</th><th>Decision</th><th>Dataset</th><th className="num">Runtime</th><th className="num">Metric rows</th></tr></thead>
              <tbody>
                {ex.data.map((e) => (
                  <tr key={e.experiment_id} style={{ cursor: "pointer" }} onClick={() => navigate(`/admin/experiments/${e.experiment_id}`)}>
                    <td className="mono"><a>{e.experiment_id}</a></td>
                    <td>{e.kind === "embedding_selection" ? "Phase 6 embeddings" : "Phase 8 matching"}</td>
                    <td className="small">{fmtDate(e.finished_at)}</td>
                    <td>{e.labels.used_source ? <StatusTag tone={e.labels.used_source === "human_reviewed" ? "good" : "warn"}>{String(e.labels.used_source)}</StatusTag> : <StatusTag tone="neutral">none (cost only)</StatusTag>}</td>
                    <td className="small" style={{ maxWidth: 280 }}>{e.validity ?? <NotEvaluated label="not recorded" />}</td>
                    <td className="small">{e.recommendation?.decision ? <><b>{e.recommendation.decision}</b>{e.recommendation.model ? ` ${e.recommendation.model}` : ""}</> : "—"}</td>
                    <td className="mono tiny">{e.dataset_version}</td>
                    <td className="num">{e.runtime_seconds ? `${e.runtime_seconds.toFixed(0)} s` : "—"}</td>
                    <td className="num">{e.metric_rows}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <EmptyState title="No experiments recorded">Run an engine with --record-db.</EmptyState>}
      </Card>
    </>
  );
}

function ExperimentDetail() {
  const { id } = useParams();
  const e = useApi<Omit<ExperimentSummary, "metric_rows"> & { metric_rows: { model_name: string; model_version: string; metric: string; split: string; value: number | null; status: string }[] }>(`/admin/experiments/${id}`);
  if (e.loading) return <Loading />;
  if (e.error || !e.data) return <ErrorBox error={e.error ?? "Not found"} />;
  const d = e.data;
  const metrics = Array.from(new Set(d.metric_rows.map((r) => r.metric))).sort();
  const models = Array.from(new Set(d.metric_rows.map((r) => r.model_name)));
  return (
    <>
      <PageHead title={d.experiment_id} sub={d.kind} />
      <div className="stack">
        <ExperimentHeader e={d as ExperimentSummary} />
        <div className="grid two">
          <Card title="Run"><Kv obj={{ started: fmtDate(d.started_at), finished: fmtDate(d.finished_at), runtime_s: d.runtime_seconds, seed: d.random_seed, feature_version: d.feature_version, embedding: `${d.embedding_model} ${d.embedding_version}`, prediction_model: d.prediction_model, dataset_size: d.dataset_size, hardware: d.hardware }} /></Card>
          <Card title="Labels and configuration"><Kv obj={{ ...d.labels, evaluation: (d.config as { evaluation?: unknown } | null)?.evaluation }} /><div className="insight-section">Limitations</div><div className="small">{d.limitations || "—"}</div></Card>
        </div>
        <Card title="Every recorded metric">
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>Model</th>{metrics.map((m) => <th key={m} className="num">{m}</th>)}</tr></thead>
              <tbody>{models.map((mo) => <tr key={mo}><td>{mo}</td>{metrics.map((m) => { const c = d.metric_rows.find((r) => r.model_name === mo && r.metric === m); return <td key={m} className="num">{c && c.value !== null ? (Math.abs(c.value) >= 100 ? c.value.toFixed(0) : c.value.toFixed(3)) : <NotEvaluated label={c?.status === "not_feasible" ? "Not feasible" : "Not yet evaluated"} />}</td>; })}</tr>)}</tbody>
            </table>
          </div>
        </Card>
      </div>
    </>
  );
}

// ------------------------------------------------------------------------------------------------- Traces
interface TraceRow { request_id: string; task_type: string; status: string; started_at: string; duration_ms: number; routing_source: string | null; model_used: string | null; student_id: string | null; job_id: string | null; agents_called: string[]; executions: number; retry_count: number }

function TracesPage() {
  const [task, setTask] = useState("");
  const [status, setStatus] = useState("");
  const list = useApi<TraceRow[]>(`/admin/traces?limit=100${task ? `&task=${task}` : ""}${status ? `&status=${status}` : ""}`);
  const [sel, setSel] = useState<string | null>(null);
  const id = sel ?? list.data?.[0]?.request_id ?? null;
  const tree = useApi<TraceTree>(id ? `/admin/traces/${id}` : null);
  const tasks = ["recommend_jobs_for_student", "rank_candidates_for_job", "explain_match", "learning_recommendations", "cohort_benchmark", "ingest_student_profile", "ingest_job", "skill_gap_analysis"];
  return (
    <>
      <PageHead title="Agent execution traces" sub="Supervisor → agents → tools / LLM calls, rebuilt from agent_runs + agent_messages (get_agent_trace)."
        actions={<>
          <select className="btn" value={task} onChange={(e) => (setTask(e.target.value), setSel(null))}><option value="">All tasks</option>{tasks.map((t) => <option key={t}>{t}</option>)}</select>
          <select className="btn" value={status} onChange={(e) => (setStatus(e.target.value), setSel(null))}><option value="">Any status</option>{["success", "partial", "failed"].map((t) => <option key={t}>{t}</option>)}</select>
        </>} />
      <div className="grid" style={{ gridTemplateColumns: "minmax(280px, 360px) minmax(0, 1fr)", alignItems: "start" }}>
        <Card title="Requests" sub={list.data ? `${list.data.length} most recent` : undefined}>
          {list.loading ? <Loading /> : list.error ? <ErrorBox error={list.error} /> : list.data?.length ? (
            <div className="list" style={{ maxHeight: 720, overflowY: "auto" }}>
              {list.data.map((r) => (
                <div key={r.request_id} className={`list-row ${r.request_id === id ? "selected" : ""}`} style={{ cursor: "pointer", padding: "9px 8px" }} onClick={() => setSel(r.request_id)}>
                  <div className="list-main">
                    <div className="list-title" style={{ fontSize: 13 }}>{r.task_type}</div>
                    <div className="list-meta"><span>{fmtDate(r.started_at)}</span><span>{ms(r.duration_ms)}</span><span>{r.executions} executions</span></div>
                    <div className="list-meta"><span className="mono">{r.student_id ?? ""}{r.student_id && r.job_id ? " · " : ""}{r.job_id ?? ""}</span></div>
                  </div>
                  <StatusTag tone={r.status === "success" ? "good" : r.status === "partial" ? "warn" : "bad"}>{r.status}</StatusTag>
                </div>
              ))}
            </div>
          ) : <EmptyState title="No traced requests" />}
        </Card>
        <Card title="Execution tree" sub={id ? <span className="mono">{id}</span> : undefined}>
          {tree.loading ? <Loading /> : tree.error ? <ErrorBox error={tree.error} /> : tree.data ? <TraceView tree={tree.data} /> : <EmptyState title="Select a request" />}
        </Card>
      </div>
    </>
  );
}

export default function AdminApp() {
  return (
    <Shell nav={NAV}>
      <Routes>
        <Route index element={<OverviewPage />} />
        <Route path="embedding" element={<ComparisonPage kind="embedding" />} />
        <Route path="matching" element={<ComparisonPage kind="matching" />} />
        <Route path="leaderboard" element={<LeaderboardPage />} />
        <Route path="metrics" element={<MetricsPage />} />
        <Route path="shap" element={<ShapPage />} />
        <Route path="experiments" element={<ExperimentsPage />} />
        <Route path="experiments/:id" element={<ExperimentDetail />} />
        <Route path="traces" element={<TracesPage />} />
      </Routes>
    </Shell>
  );
}
