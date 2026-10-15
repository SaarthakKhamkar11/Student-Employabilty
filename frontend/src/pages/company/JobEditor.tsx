import { useState } from "react";
import { Plus, X } from "lucide-react";
import { api } from "../../lib/api";
import { isNum, isSentinel } from "../../lib/format";
import { Card, ErrorBox, NotEvaluated, StatusTag } from "../../components/ui";

export interface JobSkill { raw: string; canonical: string; category: string; method: string; requirement: "required" | "preferred"; requirement_basis: string }
export interface Posting {
  job_id: string;
  source: string;
  job_title: string;
  role: string;
  company_name: string;
  location: string;
  country: string;
  work_type: string;
  job_description: string;
  responsibilities: string[] | string;
  min_degree: string;
  min_degree_level: number | string;
  min_experience_years: number | string;
  max_experience_years: number | string;
  skills: JobSkill[] | string;
  required_skills: string[] | string;
  preferred_skills: string[] | string;
  skill_split: { method: string; note: string } | string;
  duplicate_check: { near_duplicate_of: string | null } | string;
  pii: { detected: boolean; counts: Record<string, number> } | string;
  missing_fields: string[];
  extractor_version: string;
}

const txt = (v: unknown) => (typeof v === "string" && !isSentinel(v) ? v : "");
const DEGREES = ["", "Diploma", "Bachelor's", "Master's", "Doctorate"];

function Parsed({ v }: { v: unknown }) {
  if (isSentinel(v)) return <NotEvaluated label={v} />;
  if (Array.isArray(v)) return <>{v.length} item(s)</>;
  if (v === null || v === undefined || v === "") return <NotEvaluated label="Not detected" />;
  return <>{String(v)}</>;
}

function SkillList({ label, items, onChange, tone }: { label: string; items: string[]; onChange: (v: string[]) => void; tone: "bad" | "warn" }) {
  const [add, setAdd] = useState("");
  const push = () => {
    const s = add.trim();
    if (s && !items.some((x) => x.toLowerCase() === s.toLowerCase())) onChange([...items, s]);
    setAdd("");
  };
  return (
    <div className="field">
      <label>{label}</label>
      <div className="chips" style={{ marginBottom: 6 }}>
        {items.length ? items.map((s) => (
          <span key={s} className={`tag tag-${tone}`}>
            {s}
            <button type="button" onClick={() => onChange(items.filter((x) => x !== s))} style={{ border: 0, background: "none", cursor: "pointer", color: "inherit", padding: 0 }} aria-label={`remove ${s}`}><X size={11} /></button>
          </span>
        )) : <NotEvaluated label="None" />}
      </div>
      <div className="row">
        <input style={{ flex: 1 }} value={add} placeholder="Add a skill" onChange={(e) => setAdd(e.target.value)} onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), push())} />
        <button type="button" className="btn btn-sm" onClick={push}><Plus size={13} /> Add</button>
      </div>
    </div>
  );
}

/** The parsed JD shown back to the company. Nothing is ranked until the company confirms it. */
export function ConfirmJob({ posting, onConfirmed, label = "Confirm job", hints = {} }: { posting: Posting; onConfirmed: (p: Posting) => void; label?: string; hints?: Record<string, string> }) {
  const [f, setF] = useState({
    job_title: txt(posting.job_title), role: txt(posting.role), location: txt(posting.location) || hints.location || "", country: txt(posting.country),
    work_type: txt(posting.work_type) || hints.work_type || "", job_description: txt(posting.job_description),
    responsibilities: Array.isArray(posting.responsibilities) ? posting.responsibilities.join("\n") : "",
    min_degree: DEGREES.includes(txt(posting.min_degree)) ? txt(posting.min_degree) : "",
    min_experience_years: isNum(posting.min_experience_years) ? String(posting.min_experience_years) : "",
    max_experience_years: isNum(posting.max_experience_years) ? String(posting.max_experience_years) : "",
  });
  const [req, setReq] = useState<string[]>(Array.isArray(posting.required_skills) ? posting.required_skills : []);
  const [pref, setPref] = useState<string[]>(Array.isArray(posting.preferred_skills) ? posting.preferred_skills : []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (k: keyof typeof f) => (e: { target: { value: string } }) => setF({ ...f, [k]: e.target.value });

  async function confirm() {
    if (!f.job_title.trim()) return setError("Job title is required");
    const lo = f.min_experience_years ? Number(f.min_experience_years) : undefined;
    const hi = f.max_experience_years ? Number(f.max_experience_years) : undefined;
    if (lo !== undefined && hi !== undefined && hi < lo) return setError("Maximum experience is below the minimum");
    setBusy(true);
    setError(null);
    try {
      const r = await api<{ posting: Posting }>(`/company/jobs/${posting.job_id}`, {
        method: "PUT",
        json: {
          job_title: f.job_title, role: f.role, location: f.location, country: f.country, work_type: f.work_type,
          job_description: f.job_description, responsibilities: f.responsibilities.split("\n").map((x) => x.trim()).filter(Boolean),
          required_skills: req, preferred_skills: pref, min_degree: f.min_degree, min_experience_years: lo, max_experience_years: hi,
        },
      });
      onConfirmed(r.posting);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const fld = (k: keyof typeof f, label: string, parsed: unknown) => (
    <div className="field">
      <label>{label} <span className="tiny muted" style={{ fontWeight: 400 }}>· parsed: <Parsed v={parsed} /></span></label>
      <input value={f[k]} onChange={set(k)} />
    </div>
  );
  const pii = typeof posting.pii === "object" ? posting.pii : null;
  const dup = typeof posting.duplicate_check === "object" ? posting.duplicate_check : null;
  const basis = Array.isArray(posting.skills) ? Array.from(new Set(posting.skills.map((s) => s.requirement_basis))) : [];
  return (
    <div className="grid split-main">
      <Card title="Review the parsed job" sub="Correct anything wrong. Candidates are ranked only after you confirm.">
        {error && <ErrorBox error={error} />}
        <div className="form-grid">
          {fld("job_title", "Job title", posting.job_title)}
          {fld("role", "Role", posting.role)}
          {fld("location", "Location", posting.location)}
          {fld("work_type", "Work type", posting.work_type)}
        </div>
        <div className="form-grid">
          <div className="field">
            <label>Minimum degree <span className="tiny muted" style={{ fontWeight: 400 }}>· parsed: <Parsed v={posting.min_degree} /></span></label>
            <select value={f.min_degree} onChange={set("min_degree")}>{DEGREES.map((d) => <option key={d} value={d}>{d || "Not required / not stated"}</option>)}</select>
          </div>
          <div className="form-grid" style={{ gap: "0 8px" }}>
            <div className="field"><label>Min. years <span className="tiny muted" style={{ fontWeight: 400 }}>· <Parsed v={posting.min_experience_years} /></span></label><input type="number" min={0} step={0.5} value={f.min_experience_years} onChange={set("min_experience_years")} /></div>
            <div className="field"><label>Max. years <span className="tiny muted" style={{ fontWeight: 400 }}>· <Parsed v={posting.max_experience_years} /></span></label><input type="number" min={0} step={0.5} value={f.max_experience_years} onChange={set("max_experience_years")} /></div>
          </div>
        </div>
        <SkillList label="Required skills" items={req} onChange={setReq} tone="bad" />
        <SkillList label="Preferred skills" items={pref} onChange={setPref} tone="warn" />
        <div className="field"><label>Responsibilities (one per line) <span className="tiny muted" style={{ fontWeight: 400 }}>· parsed: <Parsed v={posting.responsibilities} /></span></label><textarea value={f.responsibilities} onChange={set("responsibilities")} /></div>
        <div className="field"><label>Description <span className="tiny muted" style={{ fontWeight: 400 }}>· parsed: <Parsed v={posting.job_description} /></span></label><textarea value={f.job_description} onChange={set("job_description")} /></div>
        <button className="btn btn-primary" onClick={confirm} disabled={busy}>{busy ? <span className="spinner" /> : null} {busy ? "Saving and re-embedding…" : label}</button>
      </Card>
      <Card title="Parser details">
        <dl className="kv">
          <dt>Job id</dt><dd><code>{posting.job_id}</code></dd>
          <dt>Company</dt><dd>{posting.company_name}</dd>
          <dt>Built from</dt><dd>{posting.source.replace(/_/g, " ")}</dd>
          <dt>Required / preferred</dt><dd>{typeof posting.skill_split === "object" ? <><StatusTag tone="warn">inferred</StatusTag> {posting.skill_split.note}</> : <NotEvaluated />}</dd>
          <dt>Skill basis</dt><dd>{basis.length ? basis.join(", ") : <NotEvaluated />}</dd>
          <dt>PII removed</dt><dd>{pii ? (pii.detected ? Object.entries(pii.counts).map(([k, n]) => `${n} ${k}`).join(", ") : "none detected") : <NotEvaluated />}</dd>
          <dt>Duplicate check</dt><dd>{dup ? (dup.near_duplicate_of ? `near-duplicate of ${dup.near_duplicate_of}` : "no duplicate found") : <NotEvaluated />}</dd>
          <dt>Still missing</dt><dd>{posting.missing_fields.length ? posting.missing_fields.join(", ") : "nothing"}</dd>
          <dt>Extractor</dt><dd><code>{posting.extractor_version}</code></dd>
        </dl>
      </Card>
    </div>
  );
}

const WORK_TYPES = ["Full-Time", "Part-Time", "Internship", "Contract", "Temporary"];

/** Job creation form OR JD upload/paste. Both go to the Job Intelligence Agent; returns the parsed posting. */
export function PostJob({ onParsed }: { onParsed: (p: Posting, hints: Record<string, string>) => void }) {
  const [mode, setMode] = useState<"form" | "upload">("form");
  const [f, setF] = useState({ job_title: "", location: "", work_type: "Full-Time", description: "", responsibilities: "", required: "", preferred: "", qualification: "", experience: "" });
  const [touched, setTouched] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (k: keyof typeof f) => (e: { target: { value: string } }) => setF({ ...f, [k]: e.target.value });
  const list = (s: string) => s.split(/[\n,]/).map((x) => x.trim()).filter(Boolean);
  const errs: Record<string, string> = {};
  if (f.job_title.trim().length < 2) errs.job_title = "Job title is required";
  if (f.description.trim().length < 20) errs.description = "At least 20 characters";
  if (!list(f.required).length) errs.required = "List at least one required skill";

  async function submit() {
    setError(null);
    setTouched(true);
    let body: RequestInit & { json?: unknown };
    let path = "/company/jobs";
    if (mode === "form") {
      if (Object.keys(errs).length) return;
      body = { method: "POST", json: { job_title: f.job_title, location: f.location || null, work_type: f.work_type || null, description: f.description,
        responsibilities: f.responsibilities.split("\n").map((x) => x.trim()).filter(Boolean), required_skills: list(f.required), preferred_skills: list(f.preferred),
        qualification: f.qualification || null, experience: f.experience || null } };
    } else {
      if (!file && !text.trim()) return setError("Choose a file or paste the job description");
      if (file && !/\.(pdf|docx|txt)$/i.test(file.name)) return setError("Only .pdf, .docx or .txt files are supported");
      const form = new FormData();
      if (file) form.append("file", file);
      else form.append("jd_text", text);
      body = { method: "POST", body: form };
      path = "/company/jobs/upload";
    }
    setBusy(true);
    try {
      const r = await api<{ posting: Posting }>(path, body);
      // what the company typed in the form, offered as defaults where the parser found nothing
      onParsed(r.posting, mode === "form" ? { location: f.location, work_type: f.work_type } : {});
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const e = (k: string) => (touched ? errs[k] : undefined);
  return (
    <Card title="Post a job" right={<div className="tabs" style={{ margin: 0, border: 0 }}><button className={`tab ${mode === "form" ? "active" : ""}`} onClick={() => setMode("form")}>Job form</button><button className={`tab ${mode === "upload" ? "active" : ""}`} onClick={() => setMode("upload")}>Upload / paste JD</button></div>}>
      {error && <ErrorBox error={error} />}
      {mode === "form" ? (
        <>
          <div className="form-grid">
            <div className={`field ${e("job_title") ? "invalid" : ""}`}><label>Job title</label><input value={f.job_title} onChange={set("job_title")} />{e("job_title") && <span className="field-error">{e("job_title")}</span>}</div>
            <div className="form-grid" style={{ gap: "0 8px" }}>
              <div className="field"><label>Location</label><input value={f.location} onChange={set("location")} /></div>
              <div className="field"><label>Work type</label><select value={f.work_type} onChange={set("work_type")}>{WORK_TYPES.map((w) => <option key={w}>{w}</option>)}</select></div>
            </div>
          </div>
          <div className={`field ${e("description") ? "invalid" : ""}`}><label>Description</label><textarea value={f.description} onChange={set("description")} />{e("description") && <span className="field-error">{e("description")}</span>}</div>
          <div className="field"><label>Responsibilities (one per line)</label><textarea value={f.responsibilities} onChange={set("responsibilities")} /></div>
          <div className="form-grid">
            <div className={`field ${e("required") ? "invalid" : ""}`}><label>Required skills (comma separated)</label><input value={f.required} onChange={set("required")} />{e("required") && <span className="field-error">{e("required")}</span>}</div>
            <div className="field"><label>Preferred skills (comma separated)</label><input value={f.preferred} onChange={set("preferred")} /></div>
            <div className="field"><label>Qualification</label><input placeholder="e.g. B.Tech or BCA" value={f.qualification} onChange={set("qualification")} /></div>
            <div className="field"><label>Experience</label><input placeholder="e.g. 0-2 years" value={f.experience} onChange={set("experience")} /></div>
          </div>
        </>
      ) : (
        <>
          <label className="empty" style={{ display: "block", cursor: "pointer", marginBottom: 12 }}>
            <b>{file ? file.name : "Choose a JD file (.pdf, .docx, .txt)"}</b>
            <input type="file" accept=".pdf,.docx,.txt" style={{ display: "none" }} onChange={(ev) => setFile(ev.target.files?.[0] ?? null)} />
          </label>
          <div className="field"><label>…or paste the job description</label><textarea value={text} onChange={(ev) => setText(ev.target.value)} disabled={!!file} style={{ minHeight: 180 }} /></div>
        </>
      )}
      <button className="btn btn-primary" onClick={submit} disabled={busy}>{busy ? <span className="spinner" /> : null} {busy ? "The Job Intelligence Agent is parsing and ranking…" : "Parse job"}</button>
    </Card>
  );
}
