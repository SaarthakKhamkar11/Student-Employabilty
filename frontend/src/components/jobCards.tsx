import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Briefcase, ChevronDown, ChevronUp, GraduationCap, Wallet, MapPin, Clock } from "lucide-react";
import { useApi } from "../lib/useApi";
import { isNum, pct } from "../lib/format";
import type { MatchResult } from "../lib/types";
import { StatusTag } from "./ui";

export interface JobDetails {
  job_id: string;
  job_title: string;
  role: string | null;
  company_name: string | null;
  location: string | null;
  work_type: string | null;
  salary: string | null;
  experience: string | null;
  qualification: string | null;
  required_skills: string[];
  preferred_skills: string[];
  responsibilities: string[];
  description: string | null;
}

/** Job facts for a list of matches, in one request. */
export function useJobDetails(matches: MatchResult[]) {
  const ids = matches.map((m) => m.job_id).join(",");
  return useApi<Record<string, JobDetails>>(ids ? `/student/jobs/details?ids=${ids}` : null);
}

/** The counts every page uses for one match, so the same job never shows two different numbers. */
export function skillCounts(m: MatchResult) {
  const g = typeof m.skill_gap === "object" && m.skill_gap ? m.skill_gap : null;
  if (!g) return null;
  const have = g.matched_skills.length;
  const req = g.missing_required.length;
  const pref = g.missing_preferred.length;
  return { have, req, pref, total: have + req + pref };
}

export function RequiredTag({ m }: { m: MatchResult }) {
  const c = skillCounts(m);
  if (!c) return null;
  return c.req ? <StatusTag tone="bad">{c.req} required skill{c.req > 1 ? "s" : ""} to develop</StatusTag> : <StatusTag tone="good">no required skills missing</StatusTag>;
}

const show = (v: string | null | undefined) => v || "Not stated";

/** Title, company and match; click to see the job's details, then "View match" for the full comparison. */
export function JobCard({ m, d, open, onToggle }: { m: MatchResult; d?: JobDetails; open: boolean; onToggle: () => void }) {
  const navigate = useNavigate();
  const title = d?.job_title ?? m.summary.job_title ?? m.job_id;
  return (
    <div className={`job-card ${open ? "open" : ""}`}>
      <button type="button" className="job-card-head" onClick={onToggle} aria-expanded={open}>
        <div className="list-thumb">#{m.rank}</div>
        <div className="list-main">
          <div className="list-title">{title}</div>
          <div className="list-meta">
            <span>{d?.company_name ?? "Company not stated"}</span>
            {d?.location && <span>{d.location}</span>}
          </div>
        </div>
        <RequiredTag m={m} />
        <div className="score-pill">{isNum(m.suitability_score) ? pct(m.suitability_score, 0) : "—"}<div className="tiny muted" style={{ fontWeight: 500 }}>match</div></div>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>
      {open && (
        <div className="job-card-body">
          {!d ? (
            <div className="small muted">Loading job details…</div>
          ) : (
            <>
              <div className="job-facts">
                <span><MapPin size={13} /> {show(d.location)}</span>
                <span><Wallet size={13} /> {show(d.salary)}</span>
                <span><Briefcase size={13} /> {show(d.work_type)}</span>
                <span><Clock size={13} /> {show(d.experience)}</span>
                <span><GraduationCap size={13} /> {show(d.qualification)}</span>
              </div>
              <div className="insight-section">Required skills</div>
              {d.required_skills.length ? <div className="chips">{d.required_skills.map((s) => <span key={s} className="chip">{s}</span>)}</div> : <div className="small muted">Not stated</div>}
              {d.preferred_skills.length > 0 && (
                <>
                  <div className="insight-section">Preferred skills</div>
                  <div className="chips">{d.preferred_skills.map((s) => <span key={s} className="chip">{s}</span>)}</div>
                </>
              )}
              {d.responsibilities.length > 0 && (
                <>
                  <div className="insight-section">Responsibilities</div>
                  <ul className="entries">{d.responsibilities.map((r, i) => <li key={i}>{r}</li>)}</ul>
                </>
              )}
              {!d.responsibilities.length && d.description && (
                <>
                  <div className="insight-section">About the job</div>
                  <p className="small" style={{ margin: 0 }}>{d.description}</p>
                </>
              )}
              <div className="row" style={{ marginTop: 14 }}>
                <button className="btn btn-primary btn-sm" onClick={() => navigate(`/student/jobs/${m.job_id}`)}>View match</button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/** The ranked list of job cards; one card open at a time. */
export function JobCardList({ matches }: { matches: MatchResult[] }) {
  const details = useJobDetails(matches);
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="job-cards">
      {matches.map((m) => (
        <JobCard key={m.job_id} m={m} d={details.data?.[m.job_id]} open={open === m.job_id} onToggle={() => setOpen(open === m.job_id ? null : m.job_id)} />
      ))}
    </div>
  );
}
