import { useState, type FormEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { api, ApiError } from "../../lib/api";
import { homeFor, useAuth } from "../../lib/auth";
import { ErrorBox } from "../../components/ui";
import { AuthLayout, EMAIL_RE, Field, PasswordInput, RoleToggle, type SideRole } from "./AuthLayout";

export default function Login() {
  const [role, setRole] = useState<SideRole>("student");
  const loc = useLocation();
  const after = loc.state as { email?: string; reset?: boolean } | null;
  const [email, setEmail] = useState(after?.email ?? "");
  const [password, setPassword] = useState("");
  const [touched, setTouched] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [busy, setBusy] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const errs = {
    email: !email ? "Email is required" : !EMAIL_RE.test(email) ? "Enter a valid email address" : undefined,
    password: !password ? "Password is required" : undefined,
  };

  async function submit(e: FormEvent) {
    e.preventDefault();
    setTouched(true);
    setError(null);
    setNotFound(false);
    if (errs.email || errs.password) return;
    setBusy(true);
    try {
      const t = await api<{ access_token: string }>("/auth/login", { method: "POST", json: { email, password, role } });
      const me = await login(t.access_token);
      navigate(homeFor(me), { replace: true });
    } catch (err) {
      setError((err as Error).message);
      setNotFound(err instanceof ApiError && err.status === 404);
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthLayout>
      <h2>Welcome back</h2>
      <div className="muted">Log in to your {role === "student" ? "student" : "company"} account.</div>
      <RoleToggle role={role} onChange={(r) => (setRole(r), setError(null))} />
      {after?.reset && !error && <div className="alert alert-ok">Password updated. Log in with your new password.</div>}
      {error && <ErrorBox error={error} />}
      {notFound && (
        <div className="alert alert-info">
          <span>New to CareerCompass? <Link to="/signup">Create an account</Link> with this email.</span>
        </div>
      )}
      <form onSubmit={submit} noValidate>
        <Field label={role === "student" ? "Email" : "Work email"} error={touched ? errs.email : undefined}>
          <input type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} />
        </Field>
        <Field label="Password" error={touched ? errs.password : undefined}
          aside={<Link className="small" to="/forgot-password" state={{ email }}>Forgot password?</Link>}>
          <PasswordInput autoComplete="current-password" value={password} onChange={setPassword} />
        </Field>
        <button className="btn btn-primary" style={{ width: "100%", padding: 11 }} disabled={busy}>
          {busy ? <span className="spinner" /> : null} Log in
        </button>
      </form>
      <p className="small muted" style={{ marginTop: 16, textAlign: "center" }}>
        New here? <Link to="/signup">Create an account</Link>
      </p>
    </AuthLayout>
  );
}
