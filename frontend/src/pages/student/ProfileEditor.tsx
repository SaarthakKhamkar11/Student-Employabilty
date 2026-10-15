import { useState } from "react";
import { Plus, X } from "lucide-react";
import { isSentinel } from "../../lib/format";
import { NotEvaluated, StatusTag } from "../../components/ui";

export interface StudentProfileData {
  student_id: string;
  source: string;
  degree: string;
  branch: string;
  graduation_year: number | string;
  avg_cgpa_sem5: number | string;
  education: string;
  skills: { raw: string; canonical: string; category: string; method: string; evidence: string | null }[] | string;
  projects: string[] | string;
  experience: string[] | string;
  certifications: string[] | string;
  achievements: string;
  career_interests: string;
  domain: string;
  pii: { detector_version: string; detected: boolean; counts: Record<string, number> } | string;
  missing_fields: string[];
  extractor_version: string;
}

export interface Correction {
  degree: string;
  branch: string;
  graduation_year: string;
  career_interests: string;
  domain: string;
  skills: string[];
  projects: string;
  experience: string;
  certifications: string;
}

const txt = (v: unknown) => (typeof v === "string" && !isSentinel(v) ? v : "");
// one entry per paragraph: a blank line between entries keeps long project descriptions readable
const lines = (v: unknown) => (Array.isArray(v) ? v.join("\n\n") : "");

export function initialCorrection(p: StudentProfileData): Correction {
  return {
    degree: txt(p.degree),
    branch: txt(p.branch),
    graduation_year: typeof p.graduation_year === "number" ? String(p.graduation_year) : "",
    career_interests: txt(p.career_interests),
    domain: txt(p.domain),
    skills: Array.isArray(p.skills) ? p.skills.map((s) => s.canonical) : [],
    projects: lines(p.projects),
    experience: lines(p.experience),
    certifications: lines(p.certifications),
  };
}

export function toPayload(c: Correction) {
  // entries are separated by blank lines; a single line break inside an entry is just wrapping
  const split = (s: string) => s.split(/\n\s*\n/).map((x) => x.replace(/\s*\n\s*/g, " ").trim()).filter(Boolean);
  return {
    degree: c.degree,
    branch: c.branch,
    graduation_year: c.graduation_year ? Number(c.graduation_year) : undefined,
    career_interests: c.career_interests,
    domain: c.domain,
    skills: c.skills,
    projects: split(c.projects),
    experience: split(c.experience),
    certifications: split(c.certifications),
  };
}

/** The parsed profile shown back to the student, field by field, with what the parser found beside each input. */
export function ProfileEditor({ profile, value, onChange }: { profile: StudentProfileData; value: Correction; onChange: (c: Correction) => void }) {
  const [newSkill, setNewSkill] = useState("");
  const set = (k: keyof Correction) => (e: { target: { value: string } }) => onChange({ ...value, [k]: e.target.value });
  const skillMeta = new Map(Array.isArray(profile.skills) ? profile.skills.map((s) => [s.canonical, s]) : []);

  function addSkill() {
    const s = newSkill.trim();
    if (s && !value.skills.some((x) => x.toLowerCase() === s.toLowerCase())) onChange({ ...value, skills: [...value.skills, s] });
    setNewSkill("");
  }

  const field = (k: keyof Correction, label: string, hint?: string, input?: JSX.Element) => (
    <div className="field">
      <label>{label}</label>
      {input ?? <input value={value[k] as string} onChange={set(k)} />}
      {hint && <span className="field-hint">{hint}</span>}
    </div>
  );

  return (
    <div>
      <div className="form-grid">
        {field("degree", "Degree")}
        {field("branch", "Branch / major")}
        {field("graduation_year", "Graduation year", undefined,
          <input type="number" min={1990} max={2035} value={value.graduation_year} onChange={set("graduation_year")} />)}
        {field("domain", "Domain", "e.g. Data Science, Web Development")}
      </div>
      {field("career_interests", "Career interests", "The roles you want to work in.",
        <input value={value.career_interests} onChange={set("career_interests")} placeholder="e.g. Data Analyst, Business Intelligence" />)}

      <div className="field">
        <label>
          Skills <span className="tiny muted" style={{ fontWeight: 400 }}>· remove wrong ones, add missing ones</span>
        </label>
        <div className="chips" style={{ marginBottom: 8 }}>
          {value.skills.length ? (
            value.skills.map((s) => {
              const m = skillMeta.get(s);
              return (
                <span className="chip" key={s}>
                  {s}
                  {!m && <StatusTag tone="info" dot={false}>added</StatusTag>}
                  <button type="button" className="btn-ghost" style={{ border: 0, background: "none", cursor: "pointer", padding: 0 }}
                    onClick={() => onChange({ ...value, skills: value.skills.filter((x) => x !== s) })} aria-label={`remove ${s}`}>
                    <X size={12} />
                  </button>
                </span>
              );
            })
          ) : (
            <NotEvaluated label="Skill not detected" />
          )}
        </div>
        <div className="row">
          <input style={{ flex: 1 }} value={newSkill} placeholder="Add a skill" onChange={(e) => setNewSkill(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addSkill())} />
          <button type="button" className="btn btn-sm" onClick={addSkill}>
            <Plus size={13} /> Add
          </button>
        </div>
      </div>

      {field("projects", "Projects", "Leave a blank line between projects.",
        <textarea className="entries" value={value.projects} onChange={set("projects")} />)}
      {field("experience", "Experience", "Leave a blank line between roles.",
        <textarea className="entries" value={value.experience} onChange={set("experience")} />)}
      {field("certifications", "Certifications", "Leave a blank line between certifications.",
        <textarea className="entries" style={{ minHeight: 110 }} value={value.certifications} onChange={set("certifications")} />)}
    </div>
  );
}
