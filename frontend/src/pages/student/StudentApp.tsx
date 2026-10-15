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
import { Card, ChartCard, EmptyState, ErrorBox, KpiCard, ListRow, Loading, NotEvaluated, ScoreBar, StatusTag, Val } from "../../components/ui";
import { LearningList, MatchSummary, RunMeta, SkillGapView } from "../../components/matchViews";
import { JobCardList, skillCounts, useJobDetails } from "../../components/jobCards";
import { AreaSeries, HBars } from "../../components/charts";
import { ConfirmProfile, ResumeUpload } from "./Onboarding";
import { type StudentProfileData } from "./ProfileEditor";

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
  return skillCounts(m)?.req ?? null;
}

/** Loading / error states for the overview, shared by every page. */
function WithOverview({ children }: { children: (o: Overview) => JSX.Element }) {
  const { overview } = useOverview();
  if (overview.loading && !overview.data)
    return <Loading label="Loading your results (the first time can take a minute or two)…" />;
  if (overview.error) return <ErrorBox error={overview.error} onRetry={overview.reload} />;
  if (!overview.data) return null;
  return children(overview.data);
}

// ------------------------------------------------------------------------------------------------- Overview
function OverviewPage() {
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
        return (
          <>
            <PageHead
              title="Overview"
              sub={top ? <>Your employability against your best-matching job, <b>{title(top)}</b>, and your ranked recommendations.</> : "Your employability and ranked job recommendations."}
              actions={<RunMeta run={o.run} onRefresh={refresh} refreshing={refreshing} />}
            />
            {!top ? (
              <EmptyState title="No job recommendations yet">Try Refresh in a minute, or add more detail to your profile.</EmptyState>
            ) : (
              <div className="stack">
                <div className="grid kpis">
                  <KpiCard icon={<BarChart3 size={16} />} color="" label="Overall" value={<Val v={emp?.overall} fmt={(v) => `${v.toFixed(0)}`} />}
                    foot={<>out of 100 · for {title(top)}</>} />
                  {DIMENSIONS.map((d) => (
                    <KpiCard key={d.key} icon={d.icon} color="" label={d.label} value={<Val v={emp?.dimensions[d.key]} fmt={(v) => v.toFixed(0)} />}
                      foot="out of 100" />
                  ))}
                </div>
                <div className="grid split-main">
                  <Card title="Recommended jobs" sub="Your best matches first. Click a job for its details.">
                    <JobCardList matches={matches} />
                  </Card>
                  <MatchSummary match={top} explanation={typeof expl === "object" ? expl : null} careerPaths={o.career_paths} title={`Why ${title(top)}?`} />
                </div>
                {emp && Object.entries(emp.dimensions).some(([, v]) => isSentinel(v)) && (
                  <div className="alert alert-info">
                    Not scored for this match yet:{" "}
                    {Object.entries(emp.dimensions).filter(([, v]) => isSentinel(v)).map(([k]) => dims(k)).join(", ")}. Adding these to your profile gives a fuller picture.
                  </div>
                )}
                <div className="grid split-main">
                  <ChartCard
                    title={chart === "jobs" ? "How well your top jobs fit" : `Score breakdown for ${title(top)}`}
                    sub={chart === "jobs" ? "Match per job, in rank order" : "Out of 100"}
                    right={
                      <select className="btn btn-sm" value={chart} onChange={(e) => setChart(e.target.value as "jobs" | "dims")}>
                        <option value="jobs">Top jobs</option>
                        <option value="dims">Score breakdown</option>
                      </select>
                    }
                    height={280}
                  >
                    {chart === "jobs" ? (
                      <AreaSeries points={matches.filter((m) => isNum(m.suitability_score)).map((m) => ({ label: `#${m.rank}`, value: (m.suitability_score as number) * 100, hint: `${title(m)} · ${m.summary.role ?? ""}` }))} />
                    ) : (
                      <HBars unit="" digits={0} domain={[0, 100]}
                        points={Object.entries(emp?.dimensions ?? {}).filter(([, v]) => isNum(v)).map(([k, v]) => ({ label: dims(k), value: v as number }))} />
                    )}
                  </ChartCard>
                  <Card title="What to learn next" sub={`For ${title(top)}`}>
                    <LearningList output={learning} compact />
                  </Card>
                </div>
              </div>
            )}
          </>
        );
      }}
    </WithOverview>
  );
}

// ------------------------------------------------------------------------------------------------- Jobs & match
function JobsPage() {
  const [q, setQ] = useState("");
  return (
    <WithOverview>
      {(o) => {
        const matches = matchesOf(o.run).filter((m) => `${title(m)} ${m.summary.role}`.toLowerCase().includes(q.toLowerCase()));
        return (
          <>
            <PageHead title="Jobs" sub="Your ranked recommendations. Click a job for its details, then View match for the full comparison." />
            <Card title="Recommendations" right={<input className="btn btn-sm" placeholder="Filter by title or role" value={q} onChange={(e) => setQ(e.target.value)} style={{ cursor: "text" }} />}>
              {matches.length ? (
                <JobCardList matches={matches} />
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
  const { overview } = useOverview();
  const [refreshTick, setRefreshTick] = useState(0);
  const match = useApi<{ run: Run; ai_summary: AiSummary | null }>(jobId ? `/student/match/${jobId}${refreshTick ? `?refresh=true&t=${refreshTick}` : ""}` : null);
  const learning = useApi<{ run: Run }>(jobId ? `/student/learning/${jobId}` : null);
  const run = match.data?.run;
  const m = matchesOf(run)[0];
  const matching = data<MatchingOutput>(run, "matching_agent");
  const expl = data<{ match_explanation: MatchExplanation | string }>(run, "explainability_agent")?.match_explanation;
  const emp = m && typeof m.employability === "object" ? m.employability : null;
  const details = useJobDetails(m ? [m] : []);
  const d = jobId ? details.data?.[jobId] : undefined;
  const counts = m ? skillCounts(m) : null;
  return (
    <>
      <PageHead title={d?.job_title ?? (m ? title(m) : "Match")}
        sub={d ? [d.company_name ?? "Company not stated", d.location, d.salary, d.work_type].filter(Boolean).join(" · ") : m?.summary.role ?? undefined}
        actions={<RunMeta run={run ?? null} onRefresh={() => setRefreshTick((t) => t + 1)} refreshing={match.loading} />} />
      {match.error && <ErrorBox error={match.error} onRetry={match.reload} />}
      {match.loading && !run && <Loading label="Loading this match…" />}
      {run && (
        <div className="stack">
          {matching?.pair_check && !matching.pair_check.eligible && (
            <div className="alert alert-warn">This job is outside your usual matches ({matching.pair_check.failed_rules.map(dims).join(", ").toLowerCase()}), but here is how you compare.</div>
          )}
          {!m ? (
            <EmptyState title="This job could not be scored">Try Refresh in a minute.</EmptyState>
          ) : (
            <>
              <div className="grid kpis">
                <KpiCard icon={<Target size={16} />} color="" label="Match" value={<Val v={m.suitability_score} fmt={(v) => pct(v, 0)} />} foot="how well this job fits your profile" />
                <KpiCard icon={<BarChart3 size={16} />} color="" label="Overall employability" value={<Val v={emp?.overall} fmt={(v) => v.toFixed(0)} />} foot="out of 100 for this job" />
                <KpiCard icon={<Puzzle size={16} />} color="" label="Required skills to develop" value={counts ? counts.req : "—"} foot={counts ? `${counts.have} of the job's ${counts.total} listed skills already in your profile` : undefined} />
              </div>
              <div className="grid split-main">
                <div className="stack">
                  <Card title="Score breakdown" sub="Out of 100">
                    {emp ? (
                      <div className="stack" style={{ gap: 8 }}>
                        {Object.entries(emp.dimensions).map(([k, v]) => (
                          <div className="row" key={k} style={{ flexWrap: "nowrap" }}>
                            <span style={{ width: 210 }} className="small">{dims(k)}</span>
                            {isNum(v) ? <><ScoreBar value={v} /><b className="small" style={{ width: 44, textAlign: "right" }}>{v.toFixed(0)}</b></> : <NotEvaluated label={String(v)} />}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <NotEvaluated label={String(m.employability)} />
                    )}
                  </Card>
                  <Card title="Skills for this job">
                    <SkillGapView report={m.skill_gap} />
                  </Card>
                </div>
                <MatchSummary key={run.request_id} match={m} explanation={typeof expl === "object" ? expl : null} careerPaths={overview.data?.career_paths} />
              </div>
              <Card title="What to learn for this job">
                {learning.loading && <Loading label="Finding courses (this can take a minute the first time)…" />}
                {learning.error && <ErrorBox error={learning.error} onRetry={learning.reload} />}
                {learning.data && (
                  learning.data.run.agents.learning_agent?.status === "skipped" ? (
                    <EmptyState title="Nothing to learn for this job">Your profile already covers its listed skills.</EmptyState>
                  ) : (
                    <LearningList output={data<LearningOutput>(learning.data.run, "learning_agent")} />
                  )
                )}
              </Card>
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
                <Card title={title(sel)} sub={`Rank #${sel.rank}${isNum(sel.suitability_score) ? ` · ${pct(sel.suitability_score, 0)} match` : ""}`}>
                  <SkillGapView report={sel.skill_gap} />
                </Card>
                <Card title="Across your top jobs" sub="Required skills to develop, per job">
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
      {() => (
        <>
          <PageHead title="Learning" sub="Courses for the skills each job asks for." actions={sel && <JobPicker matches={matches} value={sel.job_id} onChange={setJob} />} />
          <Card title={sel ? `For ${title(sel)}` : "Learning"}>
            {fromOverview ? (
              <LearningList output={fromOverview} />
            ) : other.loading ? (
              <Loading label="Finding courses (this can take a minute the first time)…" />
            ) : other.error ? (
              <ErrorBox error={other.error} onRetry={other.reload} />
            ) : other.data?.run.agents.learning_agent?.status === "skipped" ? (
              <EmptyState title="Nothing to learn for this job">Your profile already covers its listed skills.</EmptyState>
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
        return (
          <>
            <PageHead title="Career" sub="Career directions your ranked matches point to." />
            {o.career_paths.length ? (
              <div className="grid split-main">
                <ChartCard title="Fit by role" sub="Average match across your recommended jobs in each role" height={Math.max(220, o.career_paths.length * 34)}>
                  <HBars unit="%" digits={1} points={o.career_paths.map((c) => ({ label: c.role, value: c.mean_suitability * 100, hint: `${c.jobs} job(s), best rank #${c.best_rank}` }))} />
                </ChartCard>
                <Card title="Directions">
                  <div className="list">
                    {o.career_paths.map((c) => (
                      <ListRow key={c.role} thumb={`#${c.best_rank}`} thumbColor="" title={c.role} meta={<><span>{c.jobs} matching job{c.jobs > 1 ? "s" : ""}</span><span>e.g. {c.example_titles.join(", ")}</span></>} score={pct(c.mean_suitability, 0)} />
                    ))}
                  </div>
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
        <KpiCard icon={<Users size={16} />} color="" label="Comparable peers" value={b.comparable_size} foot={`most similar of ${b.cohort_size} students like you`} />
        <KpiCard icon={<Target size={16} />} color="" label="Your percentile" value={s ? `${s.percentile.toFixed(0)}th` : <Val v={b.suitability} />} foot={s ? `for ${s.job_title ?? "this job"}` : undefined} />
        <KpiCard icon={<Puzzle size={16} />} color="" label="Common-skill coverage" value={cov ? <Val v={cov.coverage} fmt={(v) => pct(v, 0)} /> : <Val v={b.skill_coverage} />} foot="of the skills most peers have" />
        <KpiCard icon={<BarChart3 size={16} />} color="" label="Fewer gaps than" value={gap ? `${gap.fewer_missing_than_percentile.toFixed(0)}%` : <Val v={b.relative_skill_gap} />} foot="of peers, for this job" />
      </div>
      <div className="grid two">
        <ChartCard title="Match: you vs. comparable peers" sub="For the selected job, out of 100" height={190}>
          {s ? (
            <HBars unit="" digits={0} highlight="You" points={[{ label: "You", value: s.student_score * 100 }, { label: "Peer average", value: s.cohort_average * 100 }, { label: "Peer median", value: s.cohort_median * 100 }]} />
          ) : (
            <EmptyState title="Not available"><Val v={b.suitability} /></EmptyState>
          )}
        </ChartCard>
        <Card title="Skills compared with peers" sub="Share of comparable peers; no individual is ever shown">
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
      <Card title="Who you are compared with">
        <table className="data">
          <thead><tr><th>Filter</th><th>Your value</th><th>Used</th></tr></thead>
          <tbody>
            {b.filters.map((f, i) => (
              // filter names can arrive redacted from storage; their order is fixed: branch, graduation year, target role
              <tr key={i}><td>{["Branch", "Graduation year", "Target role"][i] ?? dims(f.name)}</td><td>{f.value ?? <NotEvaluated label="Not stated" />}</td><td>{f.applied ? <StatusTag tone="good">yes</StatusTag> : <StatusTag tone="neutral">{f.note.startsWith("dropped") ? "widened to find enough peers" : "no"}</StatusTag>}</td></tr>
            ))}
          </tbody>
        </table>
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
          {pick && other.loading ? <Loading label="Comparing you with similar students…" /> : pick && other.error ? <ErrorBox error={other.error} /> : b && typeof b === "object" ? <BenchmarkView b={b} /> : <EmptyState title="Benchmark not available yet">Try Refresh on the Overview page.</EmptyState>}
        </>
      )}
    </WithOverview>
  );
}

// ------------------------------------------------------------------------------------------------- Profile & resume
/** "Title: description" entries (projects, experience) as a bold title over the description. */
function Entry({ text, split }: { text: string; split: boolean }) {
  const at = split ? text.indexOf(": ") : -1;
  if (at > 0 && at < 140) {
    return <div className="entry"><b>{text.slice(0, at)}</b><p>{text.slice(at + 2)}</p></div>;
  }
  return <div className="entry">{text}</div>;
}

function ProfilePage() {
  const prof = useApi<{ profile: StudentProfileData; stated_at_signup: Record<string, unknown>; confirmed_at: string | null; needs_reupload: boolean }>("/student/profile");
  const [editing, setEditing] = useState(false);
  const { refresh } = useOverview();
  const navigate = useNavigate();
  if (prof.loading) return <Loading />;
  if (prof.error || !prof.data) return <ErrorBox error={prof.error ?? "No profile"} />;
  const p = prof.data.profile;
  const signup = prof.data.stated_at_signup;
  // one value per fact: what you confirmed in your profile, else what you gave at sign-up
  const fact = (k: string) => {
    const v = (p as unknown as Record<string, unknown>)[k];
    return isNum(v) || (typeof v === "string" && v.trim() && !isSentinel(v)) ? v : signup[k];
  };
  const facts: [string, unknown][] = [
    ["College", signup.college], ["Degree", fact("degree")], ["Branch", fact("branch")],
    ["Graduation year", fact("graduation_year")], ["Domain", fact("domain")], ["Career interests", fact("career_interests")],
  ];
  return (
    <>
      <PageHead title="Profile" sub={prof.data.confirmed_at ? `Last confirmed ${new Date(prof.data.confirmed_at).toLocaleDateString()}` : "Not confirmed yet"}
        actions={<button className="btn btn-primary" onClick={() => setEditing(!editing)}>{editing ? "Cancel" : "Edit profile"}</button>} />
      {prof.data.needs_reupload && !editing && (
        <div className="alert alert-warn" style={{ alignItems: "center" }}>
          <span style={{ flex: 1 }}>Your resume was read with an older version of the reader, so some text may run together. Upload it again to fix this.</span>
          <button className="btn btn-sm" onClick={() => navigate("/student/resume")}>Upload resume</button>
        </div>
      )}
      {editing ? (
        <ConfirmProfile profile={p} onConfirmed={() => (setEditing(false), prof.reload(), refresh())} confirmLabel="Save changes" />
      ) : (
        <Card>
          <div className="stack" style={{ gap: 16 }}>
            <dl className="kv">
              {facts.map(([k, v]) => <div key={k} style={{ display: "contents" }}><dt>{k}</dt><dd><Val v={v} missing="Not stated" /></dd></div>)}
            </dl>
            <div>
              <div className="insight-section">Skills</div>
              {Array.isArray(p.skills) && p.skills.length ? <div className="chips">{p.skills.map((s) => <span className="chip" key={s.canonical}>{s.canonical}</span>)}</div> : <span className="small muted">None added</span>}
            </div>
            {(["projects", "experience", "certifications"] as const).map((k) => (
              <div key={k}>
                <div className="insight-section">{dims(k)}</div>
                {Array.isArray(p[k]) && (p[k] as string[]).length
                  ? (p[k] as string[]).map((x, i) => <Entry key={i} text={x} split={k !== "certifications"} />)
                  : <span className="small muted">None added</span>}
              </div>
            ))}
          </div>
        </Card>
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
  if (me && !me.onboarding.profile_confirmed_at) {
    window.location.replace("/student/onboarding");
    return null;
  }
  return (
    <Ctx.Provider value={{ overview, refresh, refreshing }}>
      <Shell nav={NAV} searchPlaceholder="Search jobs on the Jobs page…">
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


