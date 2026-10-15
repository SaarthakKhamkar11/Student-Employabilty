import { useState } from "react";
import { Plus, X } from "lucide-react";
import { api } from "../../lib/api";
import { isNum, isSentinel } from "../../lib/format";
import { Card, ErrorBox, NotEvaluated } from "../../components/ui";

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
  salary_range: string;
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

/** The job details picked out of the JD, shown back to the company. Nothing is ranked until the company confirms it. */
export function ConfirmJob({ posting, onConfirmed, label = "Confirm job", hints = {} }: { posting: Posting; onConfirmed: (p: Posting) => void; label?: string; hints?: Record<string, string> }) {
  const [f, setF] = useState({
    job_title: txt(posting.job_title), role: txt(posting.role), company_name: txt(posting.company_name) || hints.company_name || "",
    location: txt(posting.location) || hints.location || "", country: txt(posting.country),
    work_type: txt(posting.work_type) || hints.work_type || "", salary_range: txt(posting.salary_range), job_description: txt(posting.job_description),
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
          job_title: f.job_title, role: f.role, company_name: f.company_name, location: f.location, country: f.country, work_type: f.work_type,
          salary_range: f.salary_range, job_description: f.job_description,
          responsibilities: f.responsibilities.split("\n").map((x) => x.trim()).filter(Boolean),
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

  const fld = (k: keyof typeof f, label: string, placeholder?: string) => (
    <div className="field">
      <label>{label}</label>
      <input value={f[k]} onChange={set(k)} placeholder={placeholder ?? "Not stated in the JD"} />
    </div>
  );
  return (
    <Card title="Review the job details" sub="We picked these out of your job description. Correct anything wrong; candidates are ranked only after you confirm.">
      {error && <ErrorBox error={error} />}
      <div className="form-grid">
        {fld("company_name", "Company")}
        {fld("job_title", "Job title")}
        {fld("role", "Role")}
        {fld("location", "Location")}
        {fld("work_type", "Work type")}
        {fld("salary_range", "Salary")}
      </div>
      <div className="form-grid">
        <div className="field">
          <label>Minimum degree</label>
          <select value={f.min_degree} onChange={set("min_degree")}>{DEGREES.map((d) => <option key={d} value={d}>{d || "Not required / not stated"}</option>)}</select>
        </div>
        <div className="form-grid" style={{ gap: "0 8px" }}>
          <div className="field"><label>Min. years of experience</label><input type="number" min={0} step={0.5} value={f.min_experience_years} onChange={set("min_experience_years")} /></div>
          <div className="field"><label>Max. years</label><input type="number" min={0} step={0.5} value={f.max_experience_years} onChange={set("max_experience_years")} /></div>
        </div>
      </div>
      <SkillList label="Required skills" items={req} onChange={setReq} tone="bad" />
      <SkillList label="Preferred skills" items={pref} onChange={setPref} tone="warn" />
      <div className="field"><label>Responsibilities (one per line)</label><textarea value={f.responsibilities} onChange={set("responsibilities")} /></div>
      <div className="field"><label>Description</label><textarea value={f.job_description} onChange={set("job_description")} /></div>
      <button className="btn btn-primary" onClick={confirm} disabled={busy}>{busy ? <span className="spinner" /> : null} {busy ? "Saving…" : label}</button>
    </Card>
  );
}

/** The company gives its name and the job description (a file or pasted text); everything else is picked out of the JD. */
export function PostJob({ onParsed, defaultCompany = "" }: { onParsed: (p: Posting, hints: Record<string, string>) => void; defaultCompany?: string }) {
  const [company, setCompany] = useState(defaultCompany);
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setError(null);
    if (!company.trim()) return setError("Company name is required");
    if (!file && text.trim().length < 40) return setError("Upload the job description or paste it (at least a few lines)");
    if (file && !/\.(pdf|docx|txt)$/i.test(file.name)) return setError("Only .pdf, .docx or .txt files are supported");
    const form = new FormData();
    form.append("company_name", company.trim());
    if (file) form.append("file", file);
    else form.append("jd_text", text);
    setBusy(true);
    try {
      const r = await api<{ posting: Posting }>("/company/jobs/upload", { method: "POST", body: form });
      onParsed(r.posting, { company_name: company.trim() });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Post a job">
      {error && <ErrorBox error={error} />}
      <div className="field"><label>Company name</label><input value={company} onChange={(e) => setCompany(e.target.value)} /></div>
      <label className="empty" style={{ display: "block", cursor: "pointer", marginBottom: 12 }}>
        <b>{file ? file.name : "Upload the job description (.pdf, .docx, .txt)"}</b>
        {file && <button type="button" className="btn btn-sm" style={{ marginTop: 8 }} onClick={(e) => (e.preventDefault(), setFile(null))}>Remove file</button>}
        <input type="file" accept=".pdf,.docx,.txt" style={{ display: "none" }} onChange={(ev) => setFile(ev.target.files?.[0] ?? null)} />
      </label>
      <div className="field"><label>…or paste the job description</label><textarea value={text} onChange={(ev) => setText(ev.target.value)} disabled={!!file} style={{ minHeight: 200 }}
        placeholder="Paste the full JD: title, location, salary, responsibilities and requirements are picked out automatically." /></div>
      <button className="btn btn-primary" onClick={submit} disabled={busy}>{busy ? <span className="spinner" /> : null} {busy ? "Uploading…" : "Upload"}</button>
    </Card>
  );
}
