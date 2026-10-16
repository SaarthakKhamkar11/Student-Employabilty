import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { MailCheck } from "lucide-react";
import { api } from "../../lib/api";
import { homeFor, useAuth } from "../../lib/auth";
import { ErrorBox } from "../../components/ui";
import { AuthLayout, EMAIL_RE, Field, PasswordInput, passwordProblem, PasswordRules, RoleToggle, type SideRole } from "./AuthLayout";

const SIZES = ["1-10", "11-50", "51-200", "201-1000", "1001-5000", "5000+"];
const THIS_YEAR = new Date().getFullYear();

type Form = Record<string, string>;
const EMPTY: Form = {
  full_name: "", email: "", password: "", college: "", degree: "", branch: "", graduation_year: "",
  company_name: "", recruiter_name: "", industry: "", company_size: "", website: "",
};

function validate(role: SideRole, f: Form): Form {
  const e: Form = {};
  const req = (k: string, label: string) => !f[k]?.trim() && (e[k] = `${label} is required`);
  if (!f.email.trim()) e.email = "Email is required";
  else if (!EMAIL_RE.test(f.email.trim())) e.email = "Enter a valid email address";
  const pw = passwordProblem(f.password);
  if (pw) e.password = pw;
  if (role === "student") {
    req("full_name", "Full name");
    req("college", "College / university");
    req("degree", "Degree");
    req("branch", "Branch / major");
    const y = Number(f.graduation_year);
    if (!f.graduation_year) e.graduation_year = "Graduation year is required";
    else if (!Number.isInteger(y) || y < 1990 || y > 2035) e.graduation_year = "Between 1990 and 2035";
  } else {
    req("company_name", "Company name");
    req("recruiter_name", "Recruiter name");
    req("industry", "Industry");
    req("company_size", "Company size");
    if (f.website.trim() && !/^(https?:\/\/)?[^\s.]+\.[^\s]+$/i.test(f.website.trim())) e.website = "Enter a URL such as https://example.com";
  }
  return e;
}

export default function Signup() {
  const [role, setRole] = useState<SideRole>("student");
  const [f, setF] = useState<Form>(EMPTY);
  const [touched, setTouched] = useState(false);
  const [stage, setStage] = useState<"form" | "otp">("form");
  const [otp, setOtp] = useState("");
  const [info, setInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const errs = validate(role, f);
  const show = (k: string) => (touched ? errs[k] : undefined);
  const set = (k: string) => (e: { target: { value: string } }) => setF({ ...f, [k]: e.target.value });

  async function requestOtp(e?: FormEvent) {
    e?.preventDefault();
    setTouched(true);
    setError(null);
    if (Object.keys(errs).length) return;
    setBusy(true);
    try {
      const r = await api<{ message: string }>("/auth/signup/request-otp", { method: "POST", json: { email: f.email.trim(), role } });
      setInfo(r.message);
      setStage("otp");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function verifyAndCreate(e: FormEvent) {
    e.preventDefault();
    setError(null);
    if (!/^\d{6}$/.test(otp)) return setError("Enter the 6-digit code from the email");
    setBusy(true);
    try {
      const v = await api<{ signup_token: string }>("/auth/signup/verify-otp", { method: "POST", json: { email: f.email.trim(), otp } });
      const body =
        role === "student"
          ? {
              signup_token: v.signup_token, password: f.password, full_name: f.full_name,
              student_profile: { college: f.college, degree: f.degree, branch: f.branch, graduation_year: Number(f.graduation_year) },
            }
          : {
              signup_token: v.signup_token, password: f.password, full_name: f.recruiter_name,
              recruiter_profile: { company_name: f.company_name, industry: f.industry, company_size: f.company_size, website: f.website || null },
            };
      const t = await api<{ access_token: string }>("/auth/signup/complete", { method: "POST", json: body });
      const me = await login(t.access_token);
      navigate(homeFor(me), { replace: true });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (stage === "otp") {
    return (
      <AuthLayout>
        <div className="row" style={{ marginBottom: 6 }}>
          <MailCheck size={22} color="var(--accent)" />
          <h2>Verify your email</h2>
        </div>
        {info && <div className="alert alert-ok">{info}. It is valid for 10 minutes.</div>}
        {error && <ErrorBox error={error} />}
        <form onSubmit={verifyAndCreate} noValidate>
          <Field label="6-digit code">
            <input inputMode="numeric" maxLength={6} value={otp} onChange={(e) => setOtp(e.target.value.replace(/\D/g, ""))} autoFocus />
          </Field>
          <button className="btn btn-primary" style={{ width: "100%", padding: 11 }} disabled={busy}>
            {busy ? <span className="spinner" /> : null} Verify and create account
          </button>
        </form>
        <div className="row" style={{ marginTop: 14, justifyContent: "space-between" }}>
          <button className="btn btn-ghost btn-sm" onClick={() => (setStage("form"), setError(null))}>
            ← Edit details
          </button>
          <button className="btn btn-ghost btn-sm" onClick={() => requestOtp()} disabled={busy}>
            Resend code
          </button>
        </div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout>
      <h2>Create your account</h2>
      <RoleToggle role={role} onChange={(r) => (setRole(r), setTouched(false), setError(null))} />
      {error && <ErrorBox error={error} />}
      <form onSubmit={requestOtp} noValidate>
        {role === "student" ? (
          <>
            <Field label="Full name" error={show("full_name")}>
              <input autoComplete="name" value={f.full_name} onChange={set("full_name")} />
            </Field>
            <Field label="Email" error={show("email")}>
              <input type="email" autoComplete="email" value={f.email} onChange={set("email")} />
            </Field>
            <Field label="Password" error={show("password")}>
              <PasswordInput autoComplete="new-password" value={f.password} onChange={(v) => setF({ ...f, password: v })} />
              <PasswordRules pw={f.password} />
            </Field>
            <Field label="College / university" error={show("college")}>
              <input value={f.college} onChange={set("college")} />
            </Field>
            <div className="form-grid">
              <Field label="Degree" error={show("degree")}>
                <input placeholder="e.g. B.Tech" value={f.degree} onChange={set("degree")} />
              </Field>
              <Field label="Graduation year" error={show("graduation_year")}>
                <input type="number" min={1990} max={2035} placeholder={String(THIS_YEAR + 1)} value={f.graduation_year} onChange={set("graduation_year")} />
              </Field>
            </div>
            <Field label="Branch / major" error={show("branch")}>
              <input placeholder="e.g. Computer Engineering" value={f.branch} onChange={set("branch")} />
            </Field>
          </>
        ) : (
          <>
            <Field label="Company name" error={show("company_name")}>
              <input autoComplete="organization" value={f.company_name} onChange={set("company_name")} />
            </Field>
            <Field label="Recruiter name" error={show("recruiter_name")}>
              <input autoComplete="name" value={f.recruiter_name} onChange={set("recruiter_name")} />
            </Field>
            <Field label="Work email" error={show("email")}>
              <input type="email" autoComplete="email" value={f.email} onChange={set("email")} />
            </Field>
            <Field label="Password" error={show("password")}>
              <PasswordInput autoComplete="new-password" value={f.password} onChange={(v) => setF({ ...f, password: v })} />
              <PasswordRules pw={f.password} />
            </Field>
            <div className="form-grid">
              <Field label="Industry" error={show("industry")}>
                <input placeholder="e.g. IT services" value={f.industry} onChange={set("industry")} />
              </Field>
              <Field label="Company size" error={show("company_size")}>
                <select value={f.company_size} onChange={set("company_size")}>
                  <option value="">Select…</option>
                  {SIZES.map((s) => (
                    <option key={s} value={s}>
                      {s} employees
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <Field label="Website (optional)" error={show("website")}>
              <input placeholder="https://" value={f.website} onChange={set("website")} />
            </Field>
          </>
        )}
        <button className="btn btn-primary" style={{ width: "100%", padding: 11 }} disabled={busy}>
          {busy ? <span className="spinner" /> : null} Continue
        </button>
      </form>
      <p className="small muted" style={{ marginTop: 16, textAlign: "center" }}>
        Already have an account? <Link to="/login">Log in</Link>
      </p>
    </AuthLayout>
  );
}
