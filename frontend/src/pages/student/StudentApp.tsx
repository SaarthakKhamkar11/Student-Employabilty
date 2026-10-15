import { createContext, useContext, useState } from "react";
import { Route, Routes, useNavigate, useParams } from "react-router-dom";
import {
  Award, BarChart3, BookOpen, Briefcase, Compass, FileText, FolderKanban, GraduationCap, LayoutDashboard, Puzzle,
  Target, User, Users,
} from "lucide-react";
import { api } from "../../lib/api";
import { useAuth } from "../../lib/auth";
import { useApi } from "../../lib/useApi";
import { dims, isNum, isSentinel, pct } from "../../lib/format";
import { data, type AiSummary, type CohortBenchmark, type LearningOutput, type MatchExplanation, type MatchingOutput, type MatchResult, type Run } from "../../lib/types";
import { PageHead, Shell, type NavEntry } from "../../components/Shell";
import { Card, ChartCard, EmptyState, ErrorBox, KpiCard, ListRow, Loading, ModelBadge, NotEvaluated, ScoreBar, StatusTag, Val } from "../../components/ui";
import { ExplanationPanel, LearningList, RunMeta, RunProblems, SkillGapView } from "../../components/matchViews";
import { AreaSeries, HBars } from "../../components/charts";
import { ConfirmProfile, ResumeUpload } from "./Onboarding";
import { ProfileFacts, type StudentProfileData } from "./ProfileEditor";

const NAV: NavEntry[] = [
  { to: "/student", label: "Overview", icon: <LayoutDashboard size={16} />, end: true },
  { to: "/student/profile", label: "Profile", icon: <User size={16} /> },
  { to: "/student/resume", label: "Resume", icon: <FileText size={16} /> },
  { to: "/student/jobs", label: "Jobs", icon: <Briefcase size={16} /> },
  { to: "/student/skill-gaps", label: "Skill Gaps", icon: <Puzzle size={16} /> },
  { to: "/student/learning", label: "Learning", icon: <BookOpen size={16} /> },
  { to: "/student/career", label: "Career", icon: <Compass size={16} /> },
  { to: "/student/benchmark", label: "Benchmark", icon: <Users size={16} /> },
];

// The five spec dimensions shown as KPI cards (Phase 7 EmployabilityScore keys)
const DIMENSIONS: { key: string; label: string; color: string; icon: JSX.Element }[] = [
  { key: "skill_match", label: "Skills", color: "#2a78d6", icon: <Target size={16} /> },
  { key: "experience_match", label: "Experience", color: "#eb6834", icon: <Briefcase size={16} /> },
  { key: "education_match", label: "Education", color: "#1baf7a", icon: <GraduationCap size={16} /> },
  { key: "project_relevance", label: "Projects", color: "#6250d6", icon: <FolderKanban size={16} /> },
  { key: "certification_relevance", label: "Certifications", color: "#e87ba4", icon: <Award size={16} /> },
];

interface Overview {
  run: Run;
  career_paths: { role: string; jobs: number; best_rank: number; mean_suitability: number; example_titles: string[] }[];
}

const Ctx = createContext<{ overview: ReturnType<typeof useApi<Overview>>; refresh: () => void; refreshing: boolean } | null>(null);
const useOverview = () => useContext(Ctx)!;

function matchesOf(run: Run | null | undefined): MatchResult[] {
  const m = data<MatchingOutput>(run, "matching_agent")?.matches;
  return Array.isArray(m) ? m : [];
}

function title(m: MatchResult) {
  return m.summary.job_title ?? m.job_id;
}

function missingRequired(m: MatchResult): number | null {
  return typeof m.skill_gap === "object" && m.skill_gap ? m.skill_gap.missing_required.length : null;
}

/** Loading / error / "agents are running" states for the overview, shared by every page. */
function WithOverview({ children }: { children: (o: Overview) => JSX.Element }) {
  const { overview } = useOverview();
  if (overview.loading && !overview.data)
    return <Loading label="Loading your results. If nothing is stored yet, the agents run now (up to a minute or two)…" />;
  if (overview.error) return <ErrorBox error={overview.error} onRetry={overview.reload} />;
  if (!overview.data) return null;
  return children(overview.data);
}

// ------------------------------------------------------------------------------------------------- Overview
function OverviewPage() {
  const navigate = useNavigate();
  const { refresh, refreshing } = useOverview();
  const [chart, setChart] = useState<"jobs" | "dims">("jobs");
  return (
    <WithOverview>
      {(o) => {
        const matches = matchesOf(o.run);
        const top = matches[0];
        const emp = top && typeof top.employability === "object" ? top.employability : null;
        const expl = data<{ match_explanation: MatchExplanation | string }>(o.run, "explainability_agent")?.match_explanation;
        const learning = data<LearningOutput>(o.run, "learning_agent");
        const series = (key: string) => matches.map((m) => (typeof m.employability === "object" ? m.employability.dimensions[key] : null)).filter(isNum);
        return (
          <>
            <PageHead
              title="Overview"
              sub={top ? <>Your employability against your best-matching job, <b>{title(top)}</b>, and your ranked recommendations.</> : "Your employability and ranked job recommendations."}
              actions={<RunMeta run={o.run} onRefresh={refresh} refreshing={refreshing} />}
            />
            <RunProblems run={o.run} agents={["profile_agent", "matching_agent", "learning_agent", "explainability_agent"]} />
            {!top ? (
              <EmptyState title="No job recommendations">{typeof data<MatchingOutput>(o.run, "matching_agent")?.matches === "string" ? String(data<MatchingOutput>(o.run, "matching_agent")?.matches) : "The Matching Agent returned no eligible job."}</EmptyState>
            ) : (
              <div className="stack">
                <div className="grid kpis">
                  <KpiCard icon={<BarChart3 size={16} />} color="#0f4c51" label="Overall" value={<Val v={emp?.overall} fmt={(v) => `${v.toFixed(1)}`} />}
                    foot={<>out of 100 · for {title(top)}</>} spark={matches.map((m) => (typeof m.employability === "object" && isNum(m.employability.overall) ? m.employability.overall : 0))} sparkKind="bars" />
                  {DIMENSIONS.map((d) => (
                    <KpiCard key={d.key} icon={d.icon} color={d.color} label={d.label} value={<Val v={emp?.dimensions[d.key]} fmt={(v) => v.toFixed(1)} />}
                      foot="out of 100 · top match" spark={series(d.key)} />
                  ))}
                </div>
                <div className="grid split-main">
                  <ChartCard
                    title={chart === "jobs" ? "Suitability of your top recommended jobs" : `Score breakdown for ${title(top)}`}
                    sub={chart === "jobs" ? "Matching model's suitability score per job, in rank order" : "Employability dimensions, 0-100"}
                    right={
                      <select className="btn btn-sm" value={chart} onChange={(e) => setChart(e.target.value as "jobs" | "dims")}>
                        <option value="jobs">Top jobs</option>
                        <option value="dims">Score breakdown</option>
                      </select>
                    }
                    height={300}
                  >
                    {chart === "jobs" ? (
                      <AreaSeries points={matches.filter((m) => isNum(m.suitability_score)).map((m) => ({ label: `#${m.rank}`, value: (m.suitability_score as number) * 100, hint: `${title(m)} · ${m.summary.role ?? ""}` }))} />
                    ) : (
                      <HBars unit="" digits={1} domain={[0, 100]}
                        points={Object.entries(emp?.dimensions ?? {}).filter(([, v]) => isNum(v)).map(([k, v]) => ({ label: dims(k), value: v as number }))} />
                    )}
                  </ChartCard>
                  <ExplanationPanel explanation={typeof expl === "object" ? expl : null} askPath={`/student/match/${top.job_id}/ask-ai`} title={`Why ${title(top)}?`} />
                </div>
                {emp && Object.entries(emp.dimensions).some(([, v]) => isSentinel(v)) && (
                  <div className="alert alert-info">
                    Not scored for this match:{" "}
                    {Object.entries(emp.dimensions).filter(([, v]) => isSentinel(v)).map(([k, v]) => `${dims(k)} (${v})`).join(", ")}.
                  </div>
                )}
                <div className="grid split-main">
                  <Card title="Recommended jobs" sub="Ranked by the matching model" right={<ModelBadge versions={top.model_versions} compact />}>
                    <div className="list">
                      {matches.map((m) => {
                        const miss = missingRequired(m);
                        return (
                          <ListRow key={m.job_id} thumb={`#${m.rank}`} thumbColor="#0f4c51" title={title(m)}
                            meta={<><span>{m.summary.role}</span>{m.summary.country && <span>{m.summary.country}</span>}<span className="mono">{m.job_id}</span></>}
                            tag={miss === null ? <NotEvaluated label="gaps: not evaluated" /> : miss ? <StatusTag tone="bad">{miss} required skill{miss > 1 ? "s" : ""} missing</StatusTag> : <StatusTag tone="good">all required skills</StatusTag>}
                            score={<Val v={m.suitability_score} fmt={(v) => pct(v)} />}
                            action={<button className="btn btn-primary btn-sm" onClick={() => navigate(`/student/jobs/${m.job_id}`)}>View match</button>} />
                        );
                      })}
                    </div>
                  </Card>
                  <Card title="Learning for your top match" sub={`Phase 10 resources for ${title(top)}'s missing skills`}>
                    <LearningList output={learning} compact />
                  </Card>
                </div>
                <FunnelNote run={o.run} />
              </div>
            )}
          </>
        );
      }}
    </WithOverview>
  );
}

function FunnelNote({ run }: { run: Run }) {
  const m = data<MatchingOutput>(run, "matching_agent");
  if (!m || typeof m.funnel !== "object" || !m.funnel) return null;
  const f = m.funnel as Record<string, unknown>;
  return (
    <Card title="How these were chosen" sub="The Matching Agent's funnel for this run">
      <div className="row small" style={{ gap: 16 }}>
        <span>Checked <b>{String(f.checked)}</b> postings</span>
        <span>Excluded by hard rules: {Object.entries((f.excluded as Record<string, number>) ?? {}).filter(([, n]) => n).map(([k, n]) => `${dims(k)} ${n}`).join(", ") || "none"}</span>
        <span>Eligible <b>{String(f.eligible)}</b></span>
        <span>Retrieved <b>{String(f.retrieved)}</b></span>
        <span>Same role collapsed {String(f.collapsed_same_role)}</span>
        <span>Returned <b>{String(f.returned)}</b></span>
      </div>
      <div className="tiny muted" style={{ marginTop: 8 }}>{String(f.location_rule)}. {m.notes.join(" ")}</div>
    </Card>
  );
}

// ------------------------------------------------------------------------------------------------- Jobs & match
function JobsPage() {
  const navigate = useNavigate();
  const [q, setQ] = useState("");
  return (
    <WithOverview>
      {(o) => {
        const matches = matchesOf(o.run).filter((m) => `${title(m)} ${m.summary.role}`.toLowerCase().includes(q.toLowerCase()));
        return (
          <>
            <PageHead title="Jobs" sub="Your ranked recommendations. Open one to see its skill gaps, learning and explanation." actions={<RunMeta run={o.run} />} />
            <Card title="Recommendations" right={<input className="btn btn-sm" placeholder="Filter by title or role" value={q} onChange={(e) => setQ(e.target.value)} style={{ cursor: "text" }} />}>
              {matches.length ? (
                <div className="table-wrap">
                  <table className="data">
                    <thead>
                      <tr>
                        <th>Rank</th><th>Job</th><th>Role</th><th className="num">Suitability</th><th className="num">Overall</th><th className="num">Semantic similarity (pct.)</th><th>Required skills</th><th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {matches.map((m) => (
                        <tr key={m.job_id}>
                          <td>#{m.rank}</td>
                          <td><b>{title(m)}</b><div className="tiny muted mono">{m.job_id}</div></td>
                          <td>{m.summary.role ?? <NotEvaluated label="Not detected" />}</td>
                          <td className="num"><Val v={m.suitability_score} fmt={(v) => pct(v)} /></td>
                          <td className="num"><Val v={typeof m.employability === "object" ? m.employability.overall : m.employability} fmt={(v) => v.toFixed(1)} /></td>
                          <td className="num"><Val v={m.semantic_similarity} fmt={(v) => `${(v * 100).toFixed(0)}th`} /></td>
                          <td>{missingRequired(m) === null ? <NotEvaluated /> : `${missingRequired(m)} missing`}</td>
                          <td><button className="btn btn-primary btn-sm" onClick={() => navigate(`/student/jobs/${m.job_id}`)}>View match</button></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <EmptyState title="No jobs to show" />
              )}
            </Card>
          </>
        );
      }}
    </WithOverview>
  );
}

function MatchPage() {
  const { jobId } = useParams();
  const [refreshTick, setRefreshTick] = useState(0);
  const match = useApi<{ run: Run; ai_summary: AiSummary | null }>(jobId ? `/student/match/${jobId}${refreshTick ? `?refresh=true&t=${refreshTick}` : ""}` : null);
  const learning = useApi<{ run: Run }>(jobId ? `/student/learning/${jobId}` : null);
  const run = match.data?.run;
  const m = matchesOf(run)[0];
  const matching = data<MatchingOutput>(run, "matching_agent");
  const expl = data<{ match_explanation: MatchExplanation | string }>(run, "explainability_agent")?.match_explanation;
  const emp = m && typeof m.employability === "object" ? m.employability : null;
  return (
    <>
      <PageHead title={m ? title(m) : "Match"} sub={m ? <>{m.summary.role} · <span className="mono">{jobId}</span></> : <span className="mono">{jobId}</span>}
        actions={<RunMeta run={run ?? null} onRefresh={() => setRefreshTick((t) => t + 1)} refreshing={match.loading} />} />
      {match.error && <ErrorBox error={match.error} onRetry={match.reload} />}
      {match.loading && !run && <Loading label="The agents are scoring and explaining this match…" />}
      {run && (
        <div className="stack">
          <RunProblems run={run} agents={["matching_agent", "explainability_agent"]} />
          {matching?.pair_check && !matching.pair_check.eligible && (
            <div className="alert alert-warn">This job would normally be filtered out: {matching.pair_check.failed_rules.join(", ")}. It is scored here because you asked for it.</div>
          )}
          {!m ? (
            <EmptyState title="This job could not be scored">{typeof matching?.matches === "string" ? matching.matches : "Insufficient evidence"}</EmptyState>
          ) : (
            <>
              <div className="grid kpis">
                <KpiCard icon={<Target size={16} />} color="#16a34a" label="Suitability" value={<Val v={m.suitability_score} fmt={(v) => pct(v)} />} foot="matching model score (a ranking score, not a probability)" />
                <KpiCard icon={<BarChart3 size={16} />} color="#0f4c51" label="Overall employability" value={<Val v={emp?.overall} fmt={(v) => v.toFixed(1)} />} foot="out of 100 for this job" />
                <KpiCard icon={<Compass size={16} />} color="#2a78d6" label="Semantic similarity" value={<Val v={m.semantic_similarity} fmt={(v) => `${(v * 100).toFixed(0)}th pct.`} />} foot="percentile among all student-job pairs" />
              </div>
              <div className="grid split-main">
                <div className="stack">
                  <Card title="Score breakdown" sub={emp?.method}>
                    {emp ? (
                      <div className="stack" style={{ gap: 8 }}>
                        {Object.entries(emp.dimensions).map(([k, v]) => (
                          <div className="row" key={k} style={{ flexWrap: "nowrap" }}>
                            <span style={{ width: 210 }} className="small">{dims(k)}</span>
                            {isNum(v) ? <><ScoreBar value={v} /><b className="small" style={{ width: 44, textAlign: "right" }}>{v.toFixed(1)}</b></> : <NotEvaluated label={String(v)} />}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <NotEvaluated label={String(m.employability)} />
                    )}
                  </Card>
                  <Card title="Skill gaps" sub="Phase 9 skill-gap report for this job">
                    <SkillGapView report={m.skill_gap} />
                  </Card>
                </div>
                <ExplanationPanel key={run.request_id} explanation={typeof expl === "object" ? expl : null} initialAi={match.data?.ai_summary} askPath={`/student/match/${jobId}/ask-ai`} />
              </div>
              <Card title="What to learn for this job" sub="Learning Agent (Phase 10)" right={<RunMeta run={learning.data?.run ?? null} />}>
                {learning.loading && <Loading label="The Learning Agent is ranking resources (this can take a minute on CPU)…" />}
                {learning.error && <ErrorBox error={learning.error} onRetry={learning.reload} />}
                {learning.data && (
                  <>
                    <RunProblems run={learning.data.run} agents={["learning_agent"]} />
                    {learning.data.run.agents.learning_agent?.status === "skipped" ? (
                      <EmptyState title="Nothing to learn for this job">{learning.data.run.agents.learning_agent.skip_reason}</EmptyState>
                    ) : (
                      <LearningList output={data<LearningOutput>(learning.data.run, "learning_agent")} />
                    )}
                  </>
                )}
              </Card>
              <Card title="Model and version information"><ModelBadge versions={m.model_versions} /> <span className="tiny muted">feature {m.model_versions.feature_version} · dataset {m.model_versions.dataset_version} · experiment {m.model_versions.experiment_id}</span></Card>
            </>
          )}
        </div>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------------- Skill gaps
function JobPicker({ matches, value, onChange }: { matches: MatchResult[]; value: string; onChange: (v: string) => void }) {
  return (
    <select className="btn btn-sm" value={value} onChange={(e) => onChange(e.target.value)}>
      {matches.map((m) => (
        <option key={m.job_id} value={m.job_id}>#{m.rank} {title(m)}</option>
      ))}
    </select>
  );
}

function SkillGapsPage() {
  const [job, setJob] = useState<string>("");
  return (
    <WithOverview>
      {(o) => {
        const matches = matchesOf(o.run);
        const sel = matches.find((m) => m.job_id === job) ?? matches[0];
        return (
          <>
            <PageHead title="Skill gaps" sub="What each recommended job asks for that was not detected in your profile." actions={sel && <JobPicker matches={matches} value={sel.job_id} onChange={setJob} />} />
            {sel ? (
              <div className="grid split-main">
                <Card title={title(sel)} sub={`Rank #${sel.rank} · suitability ${isNum(sel.suitability_score) ? pct(sel.suitability_score) : sel.suitability_score}`}>
                  <SkillGapView report={sel.skill_gap} />
                </Card>
                <Card title="Across your top jobs" sub="Required skills not detected, per job">
                  <div style={{ height: Math.max(180, matches.length * 30) }}>
                    <HBars digits={0} highlight={`#${sel.rank} ${title(sel)}`} points={matches.filter((m) => missingRequired(m) !== null).map((m) => ({ label: `#${m.rank} ${title(m)}`, value: missingRequired(m)! }))} />
                  </div>
                </Card>
              </div>
            ) : (
              <EmptyState title="No recommendations yet" />
            )}
          </>
        );
      }}
    </WithOverview>
  );
}

// ------------------------------------------------------------------------------------------------- Learning
function LearningPage() {
  const [job, setJob] = useState<string>("");
  const { overview } = useOverview();
  const matches = matchesOf(overview.data?.run);
  const sel = matches.find((m) => m.job_id === job) ?? matches[0];
  const fromOverview = sel && sel === matches[0] ? data<LearningOutput>(overview.data?.run, "learning_agent") : null;
  const other = useApi<{ run: Run }>(sel && !fromOverview ? `/student/learning/${sel.job_id}` : null);
  return (
    <WithOverview>
      {(o) => (
        <>
          <PageHead title="Learning" sub="Courses and videos for your missing skills, from the curated catalogue and the YouTube Data API." actions={sel && <JobPicker matches={matches} value={sel.job_id} onChange={setJob} />} />
          <Card title={sel ? `For ${title(sel)}` : "Learning"} right={<RunMeta run={fromOverview ? o.run : other.data?.run ?? null} />}>
            {fromOverview ? (
              <LearningList output={fromOverview} />
            ) : other.loading ? (
              <Loading label="The Learning Agent is ranking resources (this can take a minute on CPU)…" />
            ) : other.error ? (
              <ErrorBox error={other.error} onRetry={other.reload} />
            ) : other.data?.run.agents.learning_agent?.status === "skipped" ? (
              <EmptyState title="Nothing to learn for this job">{other.data.run.agents.learning_agent.skip_reason}</EmptyState>
            ) : (
              <LearningList output={data<LearningOutput>(other.data?.run, "learning_agent")} />
            )}
          </Card>
        </>
      )}
    </WithOverview>
  );
}

// ------------------------------------------------------------------------------------------------- Career
function CareerPage() {
  return (
    <WithOverview>
      {(o) => {
        const top = matchesOf(o.run)[0];
        const align = top && typeof top.employability === "object" ? top.employability.dimensions.career_interest_alignment : undefined;
        return (
          <>
            <PageHead title="Career" sub="Career directions your ranked matches point to." />
            <div className="alert alert-info">
              There is no separate career model in this system yet. These directions are your ranked job matches grouped by role (an aggregation of
              the Matching Agent's output), so they change when your profile or the job pool changes.
            </div>
            {o.career_paths.length ? (
              <div className="grid split-main">
                <ChartCard title="Mean suitability by role" sub="Across your recommended jobs in each role" height={Math.max(220, o.career_paths.length * 34)}>
                  <HBars unit="%" digits={1} points={o.career_paths.map((c) => ({ label: c.role, value: c.mean_suitability * 100, hint: `${c.jobs} job(s), best rank #${c.best_rank}` }))} />
                </ChartCard>
                <Card title="Directions">
                  <div className="list">
                    {o.career_paths.map((c) => (
                      <ListRow key={c.role} thumb={`#${c.best_rank}`} thumbColor="#2a78d6" title={c.role} meta={<><span>{c.jobs} matching job(s)</span><span>e.g. {c.example_titles.join(", ")}</span></>} score={pct(c.mean_suitability)} />
                    ))}
                  </div>
                  <div className="insight-section">Career-interest alignment (top match)</div>
                  <Val v={align} fmt={(v) => `${v.toFixed(1)} / 100`} />
                  {isSentinel(align) && <div className="tiny muted" style={{ marginTop: 4 }}>Add your career interests in Profile to let the model score this.</div>}
                </Card>
              </div>
            ) : (
              <EmptyState title="No career directions yet">No recommended job has a stated role.</EmptyState>
            )}
          </>
        );
      }}
    </WithOverview>
  );
}

// ------------------------------------------------------------------------------------------------- Benchmark
function BenchmarkView({ b }: { b: CohortBenchmark }) {
  const s = typeof b.suitability === "object" ? b.suitability : null;
  const cov = typeof b.skill_coverage === "object" ? b.skill_coverage : null;
  const gap = typeof b.relative_skill_gap === "object" ? b.relative_skill_gap : null;
  return (
    <div className="stack">
      <div className="grid kpis">
        <KpiCard icon={<Users size={16} />} color="#2a78d6" label="Comparable peers" value={b.comparable_size} foot={`most similar of ${b.cohort_size} in the cohort`} />
        <KpiCard icon={<Target size={16} />} color="#16a34a" label="Your percentile" value={s ? `${s.percentile.toFixed(0)}th` : <Val v={b.suitability} />} foot={s ? `suitability for ${s.job_title ?? s.job_id}` : undefined} />
        <KpiCard icon={<Puzzle size={16} />} color="#6250d6" label="Common-skill coverage" value={cov ? <Val v={cov.coverage} fmt={(v) => pct(v, 0)} /> : <Val v={b.skill_coverage} />} foot="of the skills most peers hold" />
        <KpiCard icon={<BarChart3 size={16} />} color="#eb6834" label="Fewer gaps than" value={gap ? `${gap.fewer_missing_than_percentile.toFixed(0)}%` : <Val v={b.relative_skill_gap} />} foot="of peers, for this job" />
      </div>
      <div className="grid two">
        <ChartCard title="Suitability: you vs. comparable peers" sub={s ? `${s.peers_scored} peers scored, ${s.peers_unscored} not comparable` : undefined} height={190}>
          {s ? (
            <HBars unit="" digits={4} highlight="You" color="#16a34a" points={[{ label: "You", value: s.student_score }, { label: "Cohort average", value: s.cohort_average }, { label: "Cohort median", value: s.cohort_median }]} />
          ) : (
            <EmptyState title="Not available"><Val v={b.suitability} /></EmptyState>
          )}
        </ChartCard>
        <Card title="Skills compared with peers" sub="Shares of comparable peers, never individual records">
          {gap ? (
            <div className="stack" style={{ gap: 10 }}>
              {([["Distinctive gaps (most peers have them)", gap.distinctive_gaps, "bad"], ["Shared gaps (most peers lack them too)", gap.shared_gaps, "neutral"], ["Your advantages", gap.advantages, "good"]] as const).map(([h, list, tone]) => (
                <div key={h}>
                  <div className="insight-section" style={{ marginTop: 0 }}>{h}</div>
                  {list.length ? <div className="chips">{list.map((x) => <StatusTag key={x.skill} tone={tone} dot={false}>{x.skill} · {pct(x.peers_with_share, 0)}</StatusTag>)}</div> : <span className="small muted">None</span>}
                </div>
              ))}
            </div>
          ) : (
            <Val v={b.relative_skill_gap} />
          )}
        </Card>
      </div>
      <Card title="How the cohort was formed">
        <table className="data">
          <thead><tr><th>Filter</th><th>Your value</th><th>Applied</th><th>Note</th></tr></thead>
          <tbody>
            {b.filters.map((f, i) => (
              <tr key={i}><td>{f.name === "[REDACTED]" ? <span title="The stored copy of this run had this key scrubbed by the storage PII filter (it redacts any key called 'name')"><NotEvaluated label="name redacted in stored run" /></span> : dims(f.name)}</td><td>{f.value ?? <NotEvaluated label="Not detected" />}</td><td>{f.applied ? <StatusTag tone="good">yes</StatusTag> : <StatusTag tone="neutral">no</StatusTag>}</td><td className="small">{f.note}</td></tr>
            ))}
          </tbody>
        </table>
        <div className="tiny muted" style={{ marginTop: 8 }}>
          {b.relaxed.length ? `Relaxed because the cohort was too small: ${b.relaxed.join(", ")}. ` : ""}
          Privacy: {Object.entries(b.privacy).map(([k, v]) => `${dims(k)}: ${typeof v === "object" ? JSON.stringify(v) : String(v)}`).join(" · ")}. {b.notes.join(" ")}
        </div>
        <div style={{ marginTop: 8 }}><ModelBadge versions={typeof b.model_versions === "object" ? b.model_versions : null} /> <span className="tiny muted">method {b.method_version}</span></div>
      </Card>
    </div>
  );
}

function BenchmarkPage() {
  const [job, setJob] = useState("");
  const { overview } = useOverview();
  const matches = matchesOf(overview.data?.run);
  const fromOverview = data<MatchingOutput>(overview.data?.run, "matching_agent")?.cohort_benchmark;
  const pick = job && job !== matches[0]?.job_id ? job : null;
  const other = useApi<{ run: Run }>(pick ? `/student/benchmark?job_id=${pick}` : null);
  const b = pick ? data<{ cohort_benchmark: CohortBenchmark | string }>(other.data?.run, "profile_agent")?.cohort_benchmark : fromOverview;
  return (
    <WithOverview>
      {() => (
        <>
          <PageHead title="Benchmark" sub="How you compare with similar students: aggregates only, no other student is ever shown." actions={matches.length > 0 && <JobPicker matches={matches} value={job || matches[0]!.job_id} onChange={setJob} />} />
          {pick && other.loading ? <Loading label="Cohort analytics running…" /> : pick && other.error ? <ErrorBox error={other.error} /> : b && typeof b === "object" ? <BenchmarkView b={b} /> : <EmptyState title="Cohort benchmark not available">{typeof b === "string" ? b : "Insufficient evidence"}</EmptyState>}
        </>
      )}
    </WithOverview>
  );
}

// ------------------------------------------------------------------------------------------------- Profile & resume
function ProfilePage() {
  const prof = useApi<{ profile: StudentProfileData; stated_at_signup: Record<string, unknown>; confirmed_at: string | null; embeddings: { model_name: string; model_version: string; section: string; dimension: number }[] }>("/student/profile");
  const [editing, setEditing] = useState(false);
  const { refresh } = useOverview();
  if (prof.loading) return <Loading />;
  if (prof.error || !prof.data) return <ErrorBox error={prof.error ?? "No profile"} />;
  const p = prof.data.profile;
  const models = Array.from(new Set(prof.data.embeddings.map((e) => `${e.model_name}|${e.model_version}|${e.dimension}`)));
  return (
    <>
      <PageHead title="Profile" sub={prof.data.confirmed_at ? `Confirmed by you · ${new Date(prof.data.confirmed_at).toLocaleString()}` : "Not confirmed yet"} actions={<button className="btn btn-primary" onClick={() => setEditing(!editing)}>{editing ? "Cancel" : "Correct my profile"}</button>} />
      {editing ? (
        <ConfirmProfile profile={p} onConfirmed={() => (setEditing(false), prof.reload(), refresh())} confirmLabel="Save corrections" />
      ) : (
        <div className="grid split-main">
          <Card title="Structured profile">
            <div className="stack" style={{ gap: 12 }}>
              <div className="kv">
                {(["degree", "branch", "graduation_year", "domain", "career_interests", "achievements"] as const).map((k) => (
                  <div key={k} style={{ display: "contents" }}><dt>{dims(k)}</dt><dd><Val v={p[k]} /></dd></div>
                ))}
              </div>
              <div>
                <div className="insight-section">Skills</div>
                {Array.isArray(p.skills) ? <div className="chips">{p.skills.map((s) => <span className="chip" key={s.canonical} title={`${s.raw} · ${s.method}${s.evidence ? ` · ${s.evidence}` : ""}`}>{s.canonical}</span>)}</div> : <NotEvaluated label={p.skills} />}
              </div>
              {(["projects", "experience", "certifications"] as const).map((k) => (
                <div key={k}>
                  <div className="insight-section">{dims(k)}</div>
                  {Array.isArray(p[k]) ? <ul style={{ margin: 0, paddingLeft: 18 }}>{(p[k] as string[]).map((x, i) => <li key={i}>{x}</li>)}</ul> : <NotEvaluated label={String(p[k])} />}
                </div>
              ))}
            </div>
          </Card>
          <div className="stack">
            <Card title="Parser details"><ProfileFacts profile={p} /></Card>
            <Card title="Stated at signup"><dl className="kv">{Object.entries(prof.data.stated_at_signup).map(([k, v]) => <div key={k} style={{ display: "contents" }}><dt>{dims(k)}</dt><dd><Val v={v} missing="Not stated" /></dd></div>)}</dl></Card>
            <Card title="Embeddings stored">
              {models.length ? models.map((m) => { const [n, v, d] = m.split("|"); return <div key={m} className="small" style={{ marginBottom: 6 }}><ModelBadge versions={{ embedding_model: n!, embedding_version: v! }} /> <span className="muted">{d} dims · {prof.data!.embeddings.filter((e) => e.model_name === n).map((e) => e.section).join(", ")}</span></div>; }) : <NotEvaluated />}
            </Card>
          </div>
        </div>
      )}
    </>
  );
}

function ResumePage() {
  const [profile, setProfile] = useState<StudentProfileData | null>(null);
  const { refresh } = useOverview();
  const navigate = useNavigate();
  return (
    <>
      <PageHead title="Resume" sub="Upload a new resume. You will review the parsed profile before it replaces your results." />
      {profile ? (
        <ConfirmProfile profile={profile} onConfirmed={() => (refresh(), navigate("/student"))} confirmLabel="Confirm new profile" />
      ) : (
        <Card title="Upload a resume"><ResumeUpload onDone={setProfile} /></Card>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------------- app
export default function StudentApp() {
  const { me } = useAuth();
  const [refreshing, setRefreshing] = useState(false);
  const overview = useApi<Overview>("/student/overview");
  const refresh = async () => {
    setRefreshing(true);
    try {
      overview.setData(await api<Overview>("/student/overview?refresh=true"));
    } catch (e) {
      alert((e as Error).message);
    } finally {
      setRefreshing(false);
    }
  };
  const top = matchesOf(overview.data?.run)[0];
  if (me && !me.onboarding.profile_confirmed_at) {
    window.location.replace("/student/onboarding");
    return null;
  }
  return (
    <Ctx.Provider value={{ overview, refresh, refreshing }}>
      <Shell nav={NAV} searchPlaceholder="Search jobs on the Jobs page…" topRight={top ? <ModelBadge versions={top.model_versions} compact /> : undefined}>
        <Routes>
          <Route index element={<OverviewPage />} />
          <Route path="profile" element={<ProfilePage />} />
          <Route path="resume" element={<ResumePage />} />
          <Route path="jobs" element={<JobsPage />} />
          <Route path="jobs/:jobId" element={<MatchPage />} />
          <Route path="skill-gaps" element={<SkillGapsPage />} />
          <Route path="learning" element={<LearningPage />} />
          <Route path="career" element={<CareerPage />} />
          <Route path="benchmark" element={<BenchmarkPage />} />
        </Routes>
      </Shell>
    </Ctx.Provider>
  );
}


