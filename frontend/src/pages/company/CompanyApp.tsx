import { useMemo, useState } from "react";
import { Route, Routes, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { Briefcase, ChevronDown, ChevronRight, LayoutDashboard, PlusCircle, Star, Target, UserCheck, Users } from "lucide-react";
import { api } from "../../lib/api";
import { useAuth } from "../../lib/auth";
import { useApi } from "../../lib/useApi";
import { fmtDate, isNum, pct } from "../../lib/format";
import { data, type AiSummary, type MatchExplanation, type MatchingOutput, type MatchResult, type Run } from "../../lib/types";
import { PageHead, Shell, type NavEntry } from "../../components/Shell";
import { Card, ChartCard, EmptyState, ErrorBox, KpiCard, ListRow, Loading, NotEvaluated, StatusTag, Val } from "../../components/ui";
import { ExplanationPanel, RunMeta, RunProblems, SkillGapView } from "../../components/matchViews";
import { AreaSeries } from "../../components/charts";
import { ConfirmJob, PostJob, type Posting } from "./JobEditor";

const NAV: NavEntry[] = [
  { to: "/company", label: "Overview", icon: <LayoutDashboard size={16} />, end: true },
  { to: "/company/jobs", label: "Jobs", icon: <Briefcase size={16} /> },
  { to: "/company/post", label: "Post a Job", icon: <PlusCircle size={16} /> },
  { to: "/company/candidates", label: "Candidates", icon: <Users size={16} /> },
  { to: "/company/shortlist", label: "Shortlist", icon: <Star size={16} /> },
];

interface JobRow { job_id: string; job_title: string; created_at: string; confirmed_at: string | null; required_skills: string[] | string | null; location: string | null; work_type: string | null }
interface Overview {
  jobs_posted: number; candidates_matched: number | null; shortlisted: number; avg_top_suitability: number | null;
  jobs: { job_id: string; job_title: string; confirmed: boolean; ranked_at: string | null; candidates: number | null; top5_mean_suitability: number | null; scores: number[] }[];
}

const missingReq = (m: MatchResult) => (typeof m.skill_gap === "object" && m.skill_gap ? m.skill_gap.missing_required.length : null);

// ------------------------------------------------------------------------------------------------- Overview
function OverviewPage() {
  const ov = useApi<Overview>("/company/overview");
  const navigate = useNavigate();
  const [sel, setSel] = useState<string>("");
  if (ov.loading) return <Loading />;
  if (ov.error || !ov.data) return <ErrorBox error={ov.error ?? "No data"} onRetry={ov.reload} />;
  const o = ov.data;
  const job = o.jobs.find((j) => j.job_id === sel) ?? o.jobs.find((j) => j.scores.length) ?? o.jobs[0];
  return (
    <>
      <PageHead title="Overview" sub="Your jobs, matched candidates and shortlist, from the ranking runs stored for each job." actions={<button className="btn btn-primary" onClick={() => navigate("/company/post")}><PlusCircle size={14} /> Post a job</button>} />
      <div className="stack">
        <div className="grid kpis">
          <KpiCard icon={<Briefcase size={16} />} color="#2a78d6" label="Jobs posted" value={o.jobs_posted} foot={`${o.jobs.filter((j) => j.confirmed).length} confirmed`} />
          <KpiCard icon={<Users size={16} />} color="#eb6834" label="Candidates matched" value={<Val v={o.candidates_matched} missing="Not yet evaluated" />} foot="in the latest stored ranking of each job" spark={o.jobs.map((j) => j.candidates ?? 0)} sparkKind="bars" />
          <KpiCard icon={<UserCheck size={16} />} color="#1baf7a" label="Shortlisted" value={o.shortlisted} foot="current decisions (feedback events)" />
          <KpiCard icon={<Target size={16} />} color="#6250d6" label="Avg. suitability, top candidates" value={<Val v={o.avg_top_suitability} fmt={(v) => pct(v)} />} foot="mean of each job's top 5" spark={o.jobs.map((j) => j.top5_mean_suitability ?? 0)} />
        </div>
        {job ? (
          <div className="grid split-main">
            <ChartCard title={`Candidate suitability: ${job.job_title}`} sub="Ranked candidates' suitability scores, in rank order" height={300}
              right={<select className="btn btn-sm" value={job.job_id} onChange={(e) => setSel(e.target.value)}>{o.jobs.map((j) => <option key={j.job_id} value={j.job_id}>{j.job_title}</option>)}</select>}>
              {job.scores.length ? <AreaSeries points={job.scores.map((s, i) => ({ label: `#${i + 1}`, value: s * 100 }))} /> : <EmptyState title="Not ranked yet">{job.confirmed ? "Open Candidates to rank this job." : "Confirm the parsed job first."}</EmptyState>}
            </ChartCard>
            <Card title="Your jobs">
              <div className="list">
                {o.jobs.map((j) => (
                  <ListRow key={j.job_id} thumb={j.job_title.slice(0, 2).toUpperCase()} thumbColor="#0f4c51" title={j.job_title}
                    meta={<><span className="mono">{j.job_id}</span><span>{j.ranked_at ? `ranked ${fmtDate(j.ranked_at)}` : "not ranked yet"}</span></>}
                    tag={j.confirmed ? <StatusTag tone="good">confirmed</StatusTag> : <StatusTag tone="warn">needs confirmation</StatusTag>}
                    action={<button className="btn btn-primary btn-sm" onClick={() => navigate(j.confirmed ? `/company/candidates?job=${j.job_id}` : `/company/jobs/${j.job_id}`)}>{j.confirmed ? "Candidates" : "Confirm"}</button>} />
                ))}
              </div>
            </Card>
          </div>
        ) : (
          <EmptyState title="No jobs yet">Post a job to see matched candidates.</EmptyState>
        )}
      </div>
    </>
  );
}

// ------------------------------------------------------------------------------------------------- Jobs
function JobsPage() {
  const jobs = useApi<JobRow[]>("/company/jobs");
  const navigate = useNavigate();
  return (
    <>
      <PageHead title="Jobs" sub="Every job you posted, and whether its parsed structure is confirmed." />
      <Card>
        {jobs.loading ? <Loading /> : jobs.error ? <ErrorBox error={jobs.error} /> : jobs.data?.length ? (
          <table className="data">
            <thead><tr><th>Job</th><th>Location</th><th>Work type</th><th>Required skills</th><th>Posted</th><th>Status</th><th></th></tr></thead>
            <tbody>
              {jobs.data.map((j) => (
                <tr key={j.job_id}>
                  <td><b>{j.job_title}</b><div className="tiny muted mono">{j.job_id}</div></td>
                  <td><Val v={j.location} missing="Not detected" /></td>
                  <td><Val v={j.work_type} missing="Not detected" /></td>
                  <td>{Array.isArray(j.required_skills) ? j.required_skills.join(", ") : <Val v={j.required_skills} />}</td>
                  <td className="small">{fmtDate(j.created_at)}</td>
                  <td>{j.confirmed_at ? <StatusTag tone="good">confirmed</StatusTag> : <StatusTag tone="warn">needs confirmation</StatusTag>}</td>
                  <td className="row" style={{ flexWrap: "nowrap" }}>
                    <button className="btn btn-sm" onClick={() => navigate(`/company/jobs/${j.job_id}`)}>{j.confirmed_at ? "Edit" : "Confirm"}</button>
                    {j.confirmed_at && <button className="btn btn-primary btn-sm" onClick={() => navigate(`/company/candidates?job=${j.job_id}`)}>Candidates</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <EmptyState title="No jobs yet" />}
      </Card>
    </>
  );
}

function JobDetail() {
  const { jobId } = useParams();
  const job = useApi<{ posting: Posting; confirmed_at: string | null }>(`/company/jobs/${jobId}`);
  const navigate = useNavigate();
  if (job.loading) return <Loading />;
  if (job.error || !job.data) return <ErrorBox error={job.error ?? "Not found"} />;
  return (
    <>
      <PageHead title={job.data.posting.job_title} sub={job.data.confirmed_at ? `Confirmed ${fmtDate(job.data.confirmed_at)}` : "Not confirmed yet"} />
      <ConfirmJob posting={job.data.posting} label={job.data.confirmed_at ? "Save corrections" : "Confirm job"} onConfirmed={() => navigate(`/company/candidates?job=${jobId}`)} />
    </>
  );
}

function PostPage() {
  const { me } = useAuth();
  const [posting, setPosting] = useState<Posting | null>(null);
  const [hints, setHints] = useState<Record<string, string>>({});
  const navigate = useNavigate();
  return (
    <>
      <PageHead title="Post a job" sub="Add your company name and the job description. You review the details we pick out before candidates are ranked." />
      {posting ? <ConfirmJob posting={posting} hints={hints} onConfirmed={() => navigate(`/company/candidates?job=${posting.job_id}`)} /> : <PostJob defaultCompany={me?.company?.name ?? ""} onParsed={(p, h) => (setPosting(p), setHints(h))} />}
    </>
  );
}

// ------------------------------------------------------------------------------------------------- Candidates
interface Filters { minScore: number; maxMissing: string; branch: string; degree: string; shortlistedOnly: boolean }

function CandidateDetail({ jobId, studentId }: { jobId: string; studentId: string }) {
  const d = useApi<{ run: Run; ai_summary: AiSummary | null }>(`/company/jobs/${jobId}/candidates/${studentId}`);
  if (d.loading) return <Card title="Match explanation"><Loading label="Loading this candidate…" /></Card>;
  if (d.error || !d.data) return <ErrorBox error={d.error ?? "No data"} />;
  const expl = data<{ match_explanation: MatchExplanation | string }>(d.data.run, "explainability_agent")?.match_explanation;
  return <ExplanationPanel key={studentId} title={`Why ${studentId}?`} explanation={typeof expl === "object" ? expl : null} initialAi={d.data.ai_summary} askPath={`/company/jobs/${jobId}/candidates/${studentId}/ask-ai`} />;
}

function CandidatesPage() {
  const [params, setParams] = useSearchParams();
  const jobs = useApi<JobRow[]>("/company/jobs");
  const confirmed = (jobs.data ?? []).filter((j) => j.confirmed_at);
  const jobId = params.get("job") ?? confirmed[0]?.job_id ?? null;
  const [tick, setTick] = useState(0);
  const cands = useApi<{ run: Run; shortlisted: string[] }>(jobId ? `/company/jobs/${jobId}/candidates${tick ? `?refresh=true&t=${tick}` : ""}` : null);
  const [selected, setSelected] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const [f, setF] = useState<Filters>({ minScore: 0, maxMissing: "", branch: "", degree: "", shortlistedOnly: false });

  const run = cands.data?.run;
  const out = data<MatchingOutput>(run, "matching_agent");
  const all = Array.isArray(out?.matches) ? out!.matches : [];
  const shortlisted = new Set(cands.data?.shortlisted ?? []);
  const branches = useMemo(() => Array.from(new Set(all.map((m) => m.summary.branch).filter(Boolean))) as string[], [all]);
  const list = all.filter((m) =>
    (!isNum(m.suitability_score) || m.suitability_score * 100 >= f.minScore) &&
    (f.maxMissing === "" || (missingReq(m) ?? 99) <= Number(f.maxMissing)) &&
    (!f.branch || m.summary.branch === f.branch) &&
    (!f.degree || (m.summary.degree ?? "").toUpperCase().includes(f.degree.toUpperCase())) &&
    (!f.shortlistedOnly || shortlisted.has(m.student_id)));
  const sel = selected ?? list[0]?.student_id ?? null;

  async function toggle(m: MatchResult) {
    const action = shortlisted.has(m.student_id) ? "unshortlist" : "shortlist";
    setBusy(m.student_id);
    setNotice(null);
    try {
      const row = await api<{ id: number; event_type: string; created_at: string }>(`/company/jobs/${jobId}/shortlist`, {
        method: "POST",
        json: { student_id: m.student_id, action, request_id: run?.request_id, rank: m.rank, suitability_score: isNum(m.suitability_score) ? m.suitability_score : null, model_versions: m.model_versions },
      });
      const next = new Set(shortlisted);
      if (action === "shortlist") next.add(m.student_id);
      else next.delete(m.student_id);
      cands.setData({ ...cands.data!, shortlisted: Array.from(next) });
      setNotice({ tone: "ok", text: `${action === "shortlist" ? "Shortlisted" : "Removed"} ${m.student_id}: saved as feedback event #${row.id} at ${fmtDate(row.created_at)}.` });
    } catch (e) {
      setNotice({ tone: "error", text: `Could not save the decision: ${(e as Error).message}` });
    } finally {
      setBusy(null);
    }
  }

  return (
    <>
      <PageHead title="Candidates" sub="Ranked by the matching model for one confirmed job."
        actions={<>
          {confirmed.length > 0 && <select className="btn" value={jobId ?? ""} onChange={(e) => (setParams({ job: e.target.value }), setSelected(null), setOpen(null))}>{confirmed.map((j) => <option key={j.job_id} value={j.job_id}>{j.job_title} ({j.job_id})</option>)}</select>}
          <RunMeta run={run ?? null} onRefresh={() => setTick((t) => t + 1)} refreshing={cands.loading} />
        </>} />
      {jobs.data && !confirmed.length && <EmptyState title="No confirmed job">Post a job and confirm its parsed structure to rank candidates.</EmptyState>}
      {cands.error && <ErrorBox error={cands.error} onRetry={cands.reload} />}
      {cands.loading && !run && <Loading label="Ranking candidates…" />}
      {notice && <div className={`alert ${notice.tone === "ok" ? "alert-ok" : "alert-error"}`}>{notice.text}</div>}
      {run && (
        <div className="stack">
          <RunProblems run={run} agents={["matching_agent", "explainability_agent"]} />
          <Card>
            <div className="row" style={{ gap: 14 }}>
              <b className="small">Filters</b>
              <label className="small row" style={{ gap: 6 }}>Min. suitability <input type="range" min={0} max={100} value={f.minScore} onChange={(e) => setF({ ...f, minScore: Number(e.target.value) })} /> {f.minScore}%</label>
              <label className="small row" style={{ gap: 6 }}>Max. missing required
                <select className="btn btn-sm" value={f.maxMissing} onChange={(e) => setF({ ...f, maxMissing: e.target.value })}><option value="">any</option>{[0, 1, 2, 3, 5].map((n) => <option key={n} value={n}>{n}</option>)}</select></label>
              <label className="small row" style={{ gap: 6 }}>Branch
                <select className="btn btn-sm" value={f.branch} onChange={(e) => setF({ ...f, branch: e.target.value })} style={{ maxWidth: 240 }}><option value="">any</option>{branches.map((b) => <option key={b}>{b}</option>)}</select></label>
              <label className="small row" style={{ gap: 6 }}>Degree <input className="btn btn-sm" style={{ width: 90, cursor: "text" }} placeholder="e.g. B.E" value={f.degree} onChange={(e) => setF({ ...f, degree: e.target.value })} /></label>
              <label className="small row" style={{ gap: 6 }}><input type="checkbox" checked={f.shortlistedOnly} onChange={(e) => setF({ ...f, shortlistedOnly: e.target.checked })} /> Shortlisted only</label>
              <span className="spacer" />
              <span className="small muted">{list.length} of {all.length} shown</span>
            </div>
          </Card>
          <div className="grid split-main">
            <Card title="Ranked candidates" sub={out && typeof out.funnel === "object" ? `${String((out.funnel as Record<string, unknown>).eligible)} eligible of ${String((out.funnel as Record<string, unknown>).checked)} students · top ${all.length} returned` : undefined}
>
              {list.length ? (
                <div className="list">
                  {list.map((m) => {
                    const miss = missingReq(m);
                    const on = shortlisted.has(m.student_id);
                    return (
                      <div key={m.student_id}>
                        <ListRow selected={sel === m.student_id} onClick={() => setSelected(m.student_id)} thumb={`#${m.rank}`} thumbColor={on ? "#16a34a" : "#0f4c51"}
                          title={<>{m.student_id} <span className="muted" style={{ fontWeight: 400 }}>· {m.summary.branch ?? "branch not detected"}</span></>}
                          meta={<><span>{m.summary.degree ?? "degree not detected"}</span><span>{m.summary.domain}</span>
                            <button className="btn-ghost tiny" style={{ border: 0, background: "none", cursor: "pointer", color: "var(--info)", padding: 0 }} onClick={(e) => (e.stopPropagation(), setOpen(open === m.student_id ? null : m.student_id))}>
                              {open === m.student_id ? <ChevronDown size={12} /> : <ChevronRight size={12} />} skill comparison
                            </button></>}
                          tag={miss === null ? <NotEvaluated label="gaps: not evaluated" /> : miss ? <StatusTag tone="bad">{miss} required missing</StatusTag> : <StatusTag tone="good">all required</StatusTag>}
                          score={<Val v={m.suitability_score} fmt={(v) => pct(v)} />}
                          action={<button className={`btn btn-sm ${on ? "" : "btn-primary"}`} disabled={busy === m.student_id} onClick={(e) => (e.stopPropagation(), toggle(m))}>
                            {busy === m.student_id ? <span className="spinner" /> : <Star size={13} fill={on ? "#eda100" : "none"} color={on ? "#b45309" : "currentColor"} />} {on ? "Shortlisted" : "Shortlist"}
                          </button>} />
                        {open === m.student_id && <div style={{ padding: "4px 4px 16px 58px" }}><SkillGapView report={m.skill_gap} perspective="company" /></div>}
                      </div>
                    );
                  })}
                </div>
              ) : <EmptyState title="No candidate matches these filters" />}
            </Card>
            {sel && jobId ? <CandidateDetail jobId={jobId} studentId={sel} /> : <Card title="Match explanation"><EmptyState title="Select a candidate" /></Card>}
          </div>
          {out?.notes?.length ? <div className="tiny muted">{out.notes.join(" ")}</div> : null}
        </div>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------------- Shortlist
function ShortlistPage() {
  const sl = useApi<{ job_id: string; job_title: string; student_id: string; shortlisted_at: string; event_id: number; payload: Record<string, unknown> }[]>("/company/shortlist");
  const navigate = useNavigate();
  async function remove(job: string, sid: string) {
    try {
      await api(`/company/jobs/${job}/shortlist`, { method: "POST", json: { student_id: sid, action: "unshortlist" } });
      sl.reload();
    } catch (e) {
      alert((e as Error).message);
    }
  }
  return (
    <>
      <PageHead title="Shortlist" sub="Every shortlist decision is stored as a feedback event, the input to the feedback loop." />
      <Card>
        {sl.loading ? <Loading /> : sl.error ? <ErrorBox error={sl.error} /> : sl.data?.length ? (
          <table className="data">
            <thead><tr><th>Candidate</th><th>Job</th><th className="num">Rank when shortlisted</th><th className="num">Match then</th><th>Shortlisted</th><th>Event</th><th></th></tr></thead>
            <tbody>
              {sl.data.map((r) => (
                <tr key={`${r.job_id}-${r.student_id}`}>
                  <td><b>{r.student_id}</b></td>
                  <td>{r.job_title}<div className="tiny muted mono">{r.job_id}</div></td>
                  <td className="num"><Val v={r.payload.rank} /></td>
                  <td className="num"><Val v={r.payload.suitability_score} fmt={(v) => pct(v)} /></td>
                  <td className="small">{fmtDate(r.shortlisted_at)}</td>
                  <td className="mono">#{r.event_id}</td>
                  <td className="row" style={{ flexWrap: "nowrap" }}>
                    <button className="btn btn-sm" onClick={() => navigate(`/company/candidates?job=${r.job_id}`)}>Open</button>
                    <button className="btn btn-sm btn-danger" onClick={() => remove(r.job_id, r.student_id)}>Remove</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <EmptyState title="Nobody shortlisted yet">Use the Shortlist button on the Candidates page.</EmptyState>}
      </Card>
    </>
  );
}

export default function CompanyApp() {
  return (
    <Shell nav={NAV}>
      <Routes>
        <Route index element={<OverviewPage />} />
        <Route path="jobs" element={<JobsPage />} />
        <Route path="jobs/:jobId" element={<JobDetail />} />
        <Route path="post" element={<PostPage />} />
        <Route path="candidates" element={<CandidatesPage />} />
        <Route path="shortlist" element={<ShortlistPage />} />
      </Routes>
    </Shell>
  );
}
