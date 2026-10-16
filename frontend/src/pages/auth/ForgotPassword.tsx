import { useState, type FormEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { KeyRound } from "lucide-react";
import { api } from "../../lib/api";
import { ErrorBox } from "../../components/ui";
import { AuthLayout, EMAIL_RE, Field, PasswordInput, passwordProblem, PasswordRules } from "./AuthLayout";

/** Forgot password: a 6-digit code is emailed (POST /auth/password/forgot), then code + new password
 * (POST /auth/password/reset). Back on the login page the email is filled in. */
export default function ForgotPassword() {
  const loc = useLocation();
  const navigate = useNavigate();
  const [email, setEmail] = useState(((loc.state as { email?: string } | null)?.email ?? "").trim());
  const [stage, setStage] = useState<"email" | "reset">("email");
  const [otp, setOtp] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [touched, setTouched] = useState(false);
  const [info, setInfo] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const emailErr = !email.trim() ? "Email is required" : !EMAIL_RE.test(email.trim()) ? "Enter a valid email address" : undefined;
  const errs = {
    otp: !/^\d{6}$/.test(otp) ? "Enter the 6-digit code from the email" : undefined,
    password: passwordProblem(password),
    confirm: confirm !== password ? "Passwords do not match" : undefined,
  };

  async function sendCode(e?: FormEvent) {
    e?.preventDefault();
    setTouched(true);
    setError(null);
    if (emailErr) return;
    setBusy(true);
    try {
      const r = await api<{ message: string }>("/auth/password/forgot", { method: "POST", json: { email: email.trim() } });
      setInfo(r.message);
      setStage("reset");
      setTouched(false);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function reset(e: FormEvent) {
    e.preventDefault();
    setTouched(true);
    setError(null);
    if (errs.otp || errs.password || errs.confirm) return;
    setBusy(true);
    try {
      await api("/auth/password/reset", { method: "POST", json: { email: email.trim(), otp, password } });
      navigate("/login", { replace: true, state: { email: email.trim(), reset: true } });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthLayout>
      <div className="row" style={{ marginBottom: 6 }}>
        <KeyRound size={22} color="var(--accent)" />
        <h2>Reset your password</h2>
      </div>
      {stage === "email" ? (
        <>
          <div className="muted" style={{ marginBottom: 18 }}>Enter your account's email and we'll send you a 6-digit code.</div>
          {error && <ErrorBox error={error} />}
          <form onSubmit={sendCode} noValidate>
            <Field label="Email" error={touched ? emailErr : undefined}>
              <input type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} autoFocus />
            </Field>
            <button className="btn btn-primary" style={{ width: "100%", padding: 11 }} disabled={busy}>
              {busy ? <span className="spinner" /> : null} Send code
            </button>
          </form>
        </>
      ) : (
        <>
          {info && <div className="alert alert-ok">{info}. It is valid for 10 minutes.</div>}
          {error && <ErrorBox error={error} />}
          <form onSubmit={reset} noValidate>
            <Field label="6-digit code" error={touched ? errs.otp : undefined}>
              <input inputMode="numeric" maxLength={6} value={otp} onChange={(e) => setOtp(e.target.value.replace(/\D/g, ""))} autoFocus />
            </Field>
            <Field label="New password" error={touched ? errs.password : undefined}>
              <PasswordInput autoComplete="new-password" value={password} onChange={setPassword} />
              <PasswordRules pw={password} />
            </Field>
            <Field label="Confirm new password" error={touched ? errs.confirm : undefined}>
              <PasswordInput autoComplete="new-password" value={confirm} onChange={setConfirm} />
            </Field>
            <button className="btn btn-primary" style={{ width: "100%", padding: 11 }} disabled={busy}>
              {busy ? <span className="spinner" /> : null} Update password
            </button>
          </form>
          <div className="row" style={{ marginTop: 14, justifyContent: "space-between" }}>
            <button className="btn btn-ghost btn-sm" onClick={() => (setStage("email"), setError(null), setInfo(null))}>
              ← Change email
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => sendCode()} disabled={busy}>
              Resend code
            </button>
          </div>
        </>
      )}
      <p className="small muted" style={{ marginTop: 16, textAlign: "center" }}>
        Remembered it? <Link to="/login" state={{ email: email.trim() }}>Back to log in</Link>
      </p>
    </AuthLayout>
  );
}
