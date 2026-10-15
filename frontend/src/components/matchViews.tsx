import { useState } from "react";
import { BookOpen, CheckCircle2, CircleDashed, ExternalLink, PlayCircle, RefreshCw, Shuffle, XCircle } from "lucide-react";
import { api } from "../lib/api";
import { fmtDate, isNum, isSentinel, ms, num, pct } from "../lib/format";
import type { AiSummary, LearningOutput, MatchExplanation, Run, SkillGapReport } from "../lib/types";
import { Card, EmptyState, InsightPanel, ModelBadge, NotEvaluated, StatusTag } from "./ui";

/** When and how a result was produced: stored or fresh, the request id (its trace), and a refresh action. */
export function RunMeta({ run, onRefresh, refreshing }: { run: Run | null; onRefresh?: () => void; refreshing?: boolean }) {
  if (!run) return null;
  return (
    <div className="row tiny muted" style={{ gap: 8 }}>
      <StatusTag tone={run.status === "success" ? "good" : run.status === "partial" ? "warn" : "bad"}>{run.status}</StatusTag>
      <span title={`request ${run.request_id}`}>
        {run.stored ? "Stored result" : "Computed now"} · {fmtDate(run.finished_at)} · {ms(run.duration_ms)}
      </span>
      {onRefresh && (
        <button className="btn btn-sm" onClick={onRefresh} disabled={refreshing} title="Run the agents again">
          {refreshing ? <span className="spinner" /> : <RefreshCw size={13} />} Refresh
        </button>
      )}
    </div>
  );
}

/** Agent outcomes that were not successes, said out loud instead of hidden. */
export function RunProblems({ run, agents }: { run: Run | null; agents: string[] }) {
  if (!run) return null;
  const problems = agents
    .map((a) => [a, run.agents[a]] as const)
    .filter(([, r]) => r && r.status !== "success");
  if (!problems.length) return null;
  return (
    <div className="alert alert-warn">
      <div>
        {problems.map(([a, r]) => (
          <div key={a}>
            <b>{a.replace(/_/g, " ")}</b>: {r!.status}
            {r!.skip_reason ? ` (${r!.skip_reason})` : ""}
            {r!.error_codes?.length ? ` [${r!.error_codes.join(", ")}]` : ""}
          </div>
        ))}
      </div>
    </div>
  );
}

function gapLine(s: { skill: string; importance_score: number | null }) {
  return (
    <span className="chip" key={s.skill} title={isNum(s.importance_score) ? `model importance ${num(s.importance_score)}` : "importance not scored"}>
      <XCircle size={12} color="var(--bad)" /> {s.skill}
    </span>
  );
}

/** Phase 9's SkillGapReport as matched / missing / transferable. */
export function SkillGapView({ report, perspective = "student" }: { report: SkillGapReport | string | null | undefined; perspective?: "student" | "company" }) {
  if (!report || isSentinel(report) || typeof report === "string") {
    return <EmptyState title="Skill gap report not available">{typeof report === "string" ? report : "Insufficient evidence"}</EmptyState>;
  }
  const missing = [...report.missing_required.map((g) => ({ ...g, req: "required" })), ...report.missing_preferred.map((g) => ({ ...g, req: "preferred" }))];
  const transferable = missing.filter((g) => g.transferable_candidate);
  const who = perspective === "student" ? "your profile" : "the candidate's profile";
  return (
    <div className="stack" style={{ gap: 14 }}>
      <div>
        <div className="insight-section" style={{ marginTop: 0 }}>
          <CheckCircle2 size={12} color="var(--good)" style={{ verticalAlign: -1 }} /> Matched ({report.matched_skills.length})
        </div>
        {report.matched_skills.length ? (
          <div className="chips">
            {report.matched_skills.map((s) => (
              <span className="chip" key={s}>
                <CheckCircle2 size={12} color="var(--good)" /> {s}
              </span>
            ))}
          </div>
        ) : (
          <div className="small muted">No job skill was detected in {who}.</div>
        )}
      </div>
      <div>
        <div className="insight-section">Missing: required ({report.missing_required.length})</div>
        {report.missing_required.length ? <div className="chips">{report.missing_required.map(gapLine)}</div> : <div className="small muted">None.</div>}
      </div>
      <div>
        <div className="insight-section">Missing: preferred ({report.missing_preferred.length})</div>
        {report.missing_preferred.length ? <div className="chips">{report.missing_preferred.map(gapLine)}</div> : <div className="small muted">None.</div>}
      </div>
      <div>
        <div className="insight-section">
          <Shuffle size={12} style={{ verticalAlign: -1 }} /> Transferable ({transferable.length + report.partially_related.length})
        </div>
        {transferable.length || report.partially_related.length ? (
          <table className="data">
            <thead>
              <tr>
                <th>Job skill</th>
                <th>Closest skill in {who}</th>
                <th className="num">Name similarity</th>
                <th>Strength</th>
              </tr>
            </thead>
            <tbody>
              {transferable.map((g) => (
                <tr key={`t-${g.skill}`}>
                  <td>{g.skill}</td>
                  <td>{g.transferable_candidate}</td>
                  <td className="num">{isNum(g.transferable_similarity) ? num(g.transferable_similarity, 2) : <NotEvaluated />}</td>
                  <td><StatusTag tone="info">transferable</StatusTag></td>
                </tr>
              ))}
              {report.partially_related.map((g) => (
                <tr key={`p-${g.skill}`}>
                  <td>{g.skill}</td>
                  <td>{g.related_student_skill}</td>
                  <td className="num">{num(g.similarity, 2)}</td>
                  <td><StatusTag tone="neutral">partially related</StatusTag></td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="small muted">No missing skill is close to a skill in {who}.</div>
        )}
      </div>
      <div className="tiny muted">
        {report.note} Required vs preferred is inferred from the job text. Importance: <code>{report.importance_model || "—"}</code> · similarity:{" "}
        <code>{report.similarity_model || "—"}</code>
        {Object.keys(report.thresholds || {}).length ? ` · thresholds ${Object.entries(report.thresholds).map(([k, v]) => `${k} ${v}`).join(", ")}` : ""}
      </div>
    </div>
  );
}

/** Phase 10 learning recommendations, including the honest empty case per skill. */
export function LearningList({ output, compact }: { output: LearningOutput | null; compact?: boolean }) {
  if (!output) return <EmptyState title="No learning recommendations">The Learning Agent did not run for this job.</EmptyState>;
  if (typeof output.recommendations === "string") return <EmptyState title="Learning recommendations">{output.recommendations}</EmptyState>;
  if (!output.recommendations.length) return <EmptyState title="No skill gaps to learn for">Every job skill was detected in your profile.</EmptyState>;
  return (
    <div className="list">
      {output.recommendations.map((r) => (
        <div className="feed-item" key={r.skill}>
          <div className="feed-icon" style={{ background: r.requirement === "required" ? "var(--bad-bg)" : "var(--warn-bg)" }}>
            <BookOpen size={15} color={r.requirement === "required" ? "var(--bad)" : "var(--warn)"} />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="row" style={{ gap: 6 }}>
              <b>{r.skill}</b>
              <StatusTag tone={r.requirement === "required" ? "bad" : "warn"}>{r.requirement}</StatusTag>
              <span className="tiny muted">priority {r.priority}{r.tied_with.length ? ` (tied with ${r.tied_with.length})` : ""}</span>
            </div>
            {r.resources.length ? (
              <div style={{ marginTop: 4 }}>
                {r.resources.slice(0, compact ? 2 : 5).map((res) => (
                  <div key={res.resource_url} className="small" style={{ marginTop: 5 }}>
                    <a href={res.resource_url} target="_blank" rel="noreferrer" style={{ display: "inline", overflowWrap: "break-word" }}>
                      {res.kind === "video" ? <PlayCircle size={12} style={{ verticalAlign: -2 }} /> : <BookOpen size={12} style={{ verticalAlign: -2 }} />}{" "}
                      {res.resource_name} <ExternalLink size={10} />
                    </a>
                    <div className="tiny muted">
                      {res.kind} · {res.provider} · relevance {num(res.relevance_score, 2)}
                      {res.estimated_effort && !isSentinel(res.estimated_effort) ? ` · ${res.estimated_effort}` : ""}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="small" style={{ marginTop: 4 }}>
                <CircleDashed size={12} style={{ verticalAlign: -1 }} /> <i>{r.message ?? "No resources found for this skill in the current catalog."}</i>
              </div>
            )}
            {!compact && <div className="tiny muted" style={{ marginTop: 4 }}>{r.explanation}</div>}
          </div>
        </div>
      ))}
      {!compact && <div className="tiny muted" style={{ marginTop: 8 }}>{output.note}</div>}
    </div>
  );
}

/**
 * Phase 11 explanation in the insight panel. Shows the STORED (validated, evidence-backed) explanation by default.
 * Gemini is called only when the user presses the button; failures are shown, never replaced by invented text.
 */
export function ExplanationPanel({
  explanation,
  initialAi,
  askPath,
  title = "Match explanation",
}: {
  explanation: MatchExplanation | null;
  initialAi?: AiSummary | null;
  askPath: string | null;
  title?: string;
}) {
  const [ai, setAi] = useState<AiSummary | null>(initialAi ? { status: "ok", ...initialAi } : null);
  const [asking, setAsking] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  if (!explanation) {
    return (
      <Card title={title}>
        <EmptyState title="No explanation">Insufficient evidence: the Explainability Agent did not produce one for this match.</EmptyState>
      </Card>
    );
  }
  const shown = ai?.status === "ok" && ai.explanation ? ai.explanation : explanation;
  const isAi = shown.source === "llm_validated";

  async function ask() {
    if (!askPath) return;
    setAsking(true);
    setNotice(null);
    try {
      const r = await api<AiSummary>(askPath, { method: "POST" });
      if (r.status === "ok") setAi(r);
      else setNotice(r.message ?? "AI summary unavailable. Showing stored explanation.");
    } catch (e) {
      setNotice(`AI summary unavailable (${(e as Error).message}). Showing stored explanation.`);
    } finally {
      setAsking(false);
    }
  }

  return (
    <InsightPanel
      title={title}
      sub={
        <>
          {explanation.job_title ?? explanation.job_id} · suitability {pct(explanation.suitability_score)}
        </>
      }
      badge={isAi ? <StatusTag tone="ai">AI summary · {ai?.llm_model ?? "Gemini"}{ai?.cached ? " · cached" : ""}</StatusTag> : <StatusTag tone="neutral">Stored explanation</StatusTag>}
      notice={notice ? <div className="alert alert-warn">{notice}</div> : null}
      sections={[
        { heading: "Positive evidence", items: shown.positive_evidence, empty: "Insufficient evidence" },
        { heading: "Potential gaps", items: shown.potential_gaps, empty: "No gaps detected" },
        { heading: "Model evidence", items: shown.model_evidence, empty: "Insufficient evidence" },
        { heading: "How to improve", items: shown.how_to_improve, empty: "Nothing to suggest from this evidence" },
      ]}
      onAskAI={askPath && !isAi ? ask : undefined}
      askLabel="Ask AI (Gemini) to rephrase"
      asking={asking}
      footer={
        <div className="tiny muted" style={{ marginTop: 10 }}>
          {isAi
            ? "Gemini rephrased the evidence; every line was validated against the evidence items before display."
            : "Built deterministically from the model's SHAP values, the skill-gap report and similarity percentiles; every line cites its evidence."}
          <div style={{ marginTop: 6 }}>
            <ModelBadge versions={shown.model_versions} />
          </div>
        </div>
      }
    />
  );
}
