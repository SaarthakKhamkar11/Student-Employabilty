import { useState } from "react";
import { BookOpen, CheckCircle2, CircleDashed, ExternalLink, PlayCircle, RefreshCw, Shuffle, XCircle } from "lucide-react";
import { api } from "../lib/api";
import { isNum, isSentinel, pct } from "../lib/format";
import type { AiSummary, LearningOutput, MatchExplanation, MatchResult, Run, SkillGapReport } from "../lib/types";
import { Card, EmptyState, InsightPanel, StatusTag } from "./ui";

/** A refresh action for a result. Run status and timing are internal details and are not shown. */
export function RunMeta({ run, onRefresh, refreshing }: { run: Run | null; onRefresh?: () => void; refreshing?: boolean }) {
  if (!run || !onRefresh) return null;
  return (
    <button className="btn btn-sm" onClick={onRefresh} disabled={refreshing} title="Recalculate with your latest profile">
      {refreshing ? <span className="spinner" /> : <RefreshCw size={13} />} Refresh
    </button>
  );
}

/** A plain notice when part of a result could not be computed. */
export function RunProblems({ run, agents }: { run: Run | null; agents: string[] }) {
  if (!run) return null;
  const failed = agents.some((a) => run.agents[a]?.status === "failed");
  if (!failed) return null;
  return <div className="alert alert-warn">Some details could not be computed for this result. Try Refresh in a minute.</div>;
}

export function joinNames(names: string[], max = 4): string {
  const xs = names.slice(0, max);
  const more = names.length - xs.length;
  if (more > 0) return `${xs.join(", ")} and ${more} more`;
  if (xs.length <= 1) return xs.join("");
  return `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`;
}

function gapChip(s: { skill: string }) {
  return (
    <span className="chip" key={s.skill}>
      <XCircle size={12} color="var(--bad)" /> {s.skill}
    </span>
  );
}

/** Matched / to develop / carry-over skills for one job. */
export function SkillGapView({ report, perspective = "student" }: { report: SkillGapReport | string | null | undefined; perspective?: "student" | "company" }) {
  if (!report || isSentinel(report) || typeof report === "string") {
    return <EmptyState title="Skill comparison not available">Try Refresh in a minute.</EmptyState>;
  }
  const missing = [...report.missing_required, ...report.missing_preferred];
  const transferable = missing.filter((g) => g.transferable_candidate);
  const who = perspective === "student" ? "your profile" : "the candidate's profile";
  return (
    <div className="stack" style={{ gap: 14 }}>
      <div>
        <div className="insight-section" style={{ marginTop: 0 }}>
          <CheckCircle2 size={12} color="var(--good)" style={{ verticalAlign: -1 }} /> Skills already in {who} ({report.matched_skills.length})
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
          <div className="small muted">None of the job's listed skills are in {who} yet.</div>
        )}
      </div>
      <div>
        <div className="insight-section">To develop: required ({report.missing_required.length})</div>
        {report.missing_required.length ? <div className="chips">{report.missing_required.map(gapChip)}</div> : <div className="small muted">None.</div>}
      </div>
      <div>
        <div className="insight-section">To develop: preferred ({report.missing_preferred.length})</div>
        {report.missing_preferred.length ? <div className="chips">{report.missing_preferred.map(gapChip)}</div> : <div className="small muted">None.</div>}
      </div>
      {(transferable.length > 0 || report.partially_related.length > 0) && (
        <div>
          <div className="insight-section">
            <Shuffle size={12} style={{ verticalAlign: -1 }} /> Skills that carry over
          </div>
          <table className="data">
            <thead>
              <tr>
                <th>Job skill</th>
                <th>Closest skill in {who}</th>
                <th>Overlap</th>
              </tr>
            </thead>
            <tbody>
              {transferable.map((g) => (
                <tr key={`t-${g.skill}`}>
                  <td>{g.skill}</td>
                  <td>{g.transferable_candidate}</td>
                  <td><StatusTag tone="info">strong</StatusTag></td>
                </tr>
              ))}
              {report.partially_related.map((g) => (
                <tr key={`p-${g.skill}`}>
                  <td>{g.skill}</td>
                  <td>{g.related_student_skill}</td>
                  <td><StatusTag tone="neutral">partial</StatusTag></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/** Courses and videos per skill to develop: name, where it is, relevance, and the effort for courses. */
export function LearningList({ output, compact }: { output: LearningOutput | null; compact?: boolean }) {
  if (!output || typeof output.recommendations === "string")
    return <EmptyState title="No learning suggestions yet">Try Refresh in a minute.</EmptyState>;
  if (!output.recommendations.length) return <EmptyState title="Nothing to learn for this job">Your profile already covers its listed skills.</EmptyState>;
  return (
    <div className="list">
      {output.recommendations.map((r) => (
        <div className="feed-item" key={r.skill}>
          <div className="feed-icon" style={{ background: "var(--accent-soft)" }}>
            <BookOpen size={15} color="var(--accent)" />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="row" style={{ gap: 6 }}>
              <b>{r.skill}</b>
              <StatusTag tone={r.requirement === "required" ? "bad" : "warn"} dot={false}>{r.requirement}</StatusTag>
            </div>
            {r.resources.length ? (
              <div style={{ marginTop: 4 }}>
                {r.resources.slice(0, compact ? 2 : 4).map((res) => (
                  <div key={res.resource_url} className="resource">
                    <a href={res.resource_url} target="_blank" rel="noreferrer">
                      {res.kind === "video" ? <PlayCircle size={13} style={{ verticalAlign: -2 }} /> : <BookOpen size={13} style={{ verticalAlign: -2 }} />}{" "}
                      {res.resource_name} <ExternalLink size={10} />
                    </a>
                    <div className="tiny muted">
                      {res.kind === "video" ? `YouTube · ${res.provider.replace(/^YouTube:\s*/, "")}` : res.provider} · relevance {Math.round(res.relevance_score * 100)}%
                      {res.kind === "course" && res.estimated_effort && !isSentinel(res.estimated_effort) ? ` · ${res.estimated_effort}` : ""}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="small" style={{ marginTop: 4 }}>
                <CircleDashed size={12} style={{ verticalAlign: -1 }} /> <i>No course found for this skill yet.</i>
              </div>
            )}
          </div>
        </div>
      ))}
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
        <EmptyState title="No explanation yet">Try Refresh in a minute.</EmptyState>
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
          {explanation.job_title ?? explanation.job_id} · {pct(explanation.suitability_score, 0)} match
        </>
      }
      badge={isAi ? <StatusTag tone="ai">AI summary</StatusTag> : undefined}
      notice={notice ? <div className="alert alert-warn">{notice}</div> : null}
      sections={[
        { heading: "Positive evidence", items: shown.positive_evidence, empty: "Insufficient evidence" },
        { heading: "Potential gaps", items: shown.potential_gaps, empty: "No gaps detected" },
        { heading: "How to improve", items: shown.how_to_improve, empty: "Nothing to suggest from this evidence" },
      ]}
      onAskAI={askPath && !isAi ? ask : undefined}
      askLabel="Rephrase with AI"
      asking={asking}
    />
  );
}

export interface CareerPath {
  role: string;
  jobs: number;
  best_rank: number;
  mean_suitability: number;
  example_titles: string[];
}

const DIM_WORDS: Record<string, string> = {
  experience_match: "experience",
  education_match: "education",
  project_relevance: "projects",
  certification_relevance: "certifications",
};

/**
 * The student's explanation in plain sentences: why the job was recommended (up to 4 lines), the gaps (up to 3),
 * why the role suits them (up to 4) and how to improve. Built from the match's own structured results (skills
 * found and missing, employability dimensions, the role's place among the student's matches); no model terms or
 * scores are shown.
 */
export function MatchSummary({
  match,
  explanation,
  careerPaths,
  title = "Why this job?",
}: {
  match: MatchResult;
  explanation: MatchExplanation | null;
  careerPaths?: CareerPath[];
  title?: string;
}) {
  const report = typeof match.skill_gap === "object" && match.skill_gap ? match.skill_gap : null;
  const emp = typeof match.employability === "object" ? match.employability : null;
  const dim = (k: string) => (emp && isNum(emp.dimensions[k]) ? (emp.dimensions[k] as number) : null);
  const jobTitle = match.summary.job_title ?? "this job";
  const role = match.summary.role ?? jobTitle;

  const why: string[] = [];
  if (report?.matched_skills.length) why.push(`Your profile already shows ${joinNames(report.matched_skills)}, which this role asks for.`);
  const carry = report ? [...report.missing_required, ...report.missing_preferred].filter((g) => g.transferable_candidate) : [];
  // a carry-over skill is still listed as "to develop": say it's a head start, not that the gap is closed
  if (carry.length) why.push(`Your ${carry[0]!.transferable_candidate} gives you a head start on ${carry[0]!.skill.toLowerCase()}.`);
  const strong = Object.keys(DIM_WORDS).filter((k) => (dim(k) ?? 0) >= 75).map((k) => DIM_WORDS[k]!);
  if (strong.length) why.push(`Your ${joinNames(strong)} line up well with what the role involves.`);
  if ((dim("education_match") ?? 0) >= 80 && why.length < 4) why.push("Your degree meets the job's education requirement.");
  if (!why.length) why.push("Your overall profile is a close fit for the work this role describes.");

  const gaps: string[] = [];
  const req = report?.missing_required.map((g) => g.skill) ?? [];
  const pref = report?.missing_preferred.map((g) => g.skill) ?? [];
  if (req.length) gaps.push(`The role asks for ${joinNames(req, 3)}, which your profile doesn't show yet.`);
  if (pref.length) gaps.push(`${joinNames(pref, 3)} would strengthen your application.`);
  const weak = Object.keys(DIM_WORDS).filter((k) => { const v = dim(k); return v !== null && v < 50; }).map((k) => DIM_WORDS[k]!)
    .filter((w) => !strong.includes(w));
  if (weak.length) gaps.push(`Your ${joinNames(weak)} ${weak.length > 1 ? "are" : "is"} less closely related to this role than the rest of your profile.`);
  if (!gaps.length) gaps.push("No gaps found: your profile covers every skill this job lists.");

  const career: string[] = [];
  const path = careerPaths?.find((c) => c.role === role);
  if (path && careerPaths) {
    const pos = careerPaths.indexOf(path);
    career.push(pos === 0 ? `${role} is your strongest direction: it leads your recommendations.` : `${role} is one of the ${careerPaths.length} directions your recommendations point to.`);
    if (path.jobs > 1) career.push(`${path.jobs} of your recommended jobs are in this role, so the fit isn't a one-off.`);
  }
  if ((dim("career_interest_alignment") ?? 0) >= 70) career.push("It matches the career interests and domain in your profile.");
  if (isNum(match.semantic_similarity) && match.semantic_similarity >= 0.75) career.push("Your projects and experience read close to the day-to-day work of this role.");
  if (strong.includes("projects")) career.push("Your project work is directly relevant, which is often what recruiters for this role look for first.");
  if (!career.length) career.push(`${role} builds on the skills and education already in your profile.`);

  const improve = explanation?.how_to_improve.length ? explanation.how_to_improve : req.length ? [`Develop ${joinNames(req, 3)} first: they are required for this role.`] : [];

  return (
    <InsightPanel
      title={title}
      sub={<>{jobTitle}{isNum(match.suitability_score) ? <> · {pct(match.suitability_score, 0)} match</> : null}</>}
      sections={[
        { heading: "Why this job is recommended", items: why.slice(0, 4) },
        { heading: "Gaps", items: gaps.slice(0, 3) },
        { heading: "Why this role suits you", items: career.slice(0, 4) },
        { heading: "How to improve", items: improve, empty: "Nothing to add: keep your profile up to date." },
      ]}
    />
  );
}
