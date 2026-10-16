import { useState, type ReactNode } from "react";
import { Building2, Check, Eye, EyeOff, GraduationCap } from "lucide-react";
import { Logo } from "../../components/Shell";
import { ThemeToggle } from "../../lib/theme";

export type SideRole = "student" | "recruiter";

export function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="auth">
      <div className="auth-brand">
        <Logo />
        <div className="tagline">
          Know where you stand. <span>See what to learn next.</span>
        </div>
      </div>
      <div className="auth-form">
        <div className="auth-theme"><ThemeToggle /></div>
        <div className="card auth-card" style={{ padding: 30 }}>
          {children}
        </div>
      </div>
    </div>
  );
}

export function RoleToggle({ role, onChange }: { role: SideRole; onChange: (r: SideRole) => void }) {
  return (
    <div className="role-toggle" role="tablist" aria-label="Account type">
      <button type="button" role="tab" aria-selected={role === "student"} className={role === "student" ? "on" : ""} onClick={() => onChange("student")}>
        <GraduationCap size={16} /> I'm a Student
      </button>
      <button type="button" role="tab" aria-selected={role === "recruiter"} className={role === "recruiter" ? "on" : ""} onClick={() => onChange("recruiter")}>
        <Building2 size={16} /> I'm a Company
      </button>
    </div>
  );
}

export function Field({
  label,
  error,
  hint,
  aside,
  children,
}: {
  label: string;
  error?: string;
  hint?: ReactNode;
  /** Shown at the right of the label row (e.g. "Forgot password?"). */
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className={`field ${error ? "invalid" : ""}`}>
      {aside ? <div className="field-label-row"><label>{label}</label>{aside}</div> : <label>{label}</label>}
      {children}
      {error ? <span className="field-error">{error}</span> : hint ? <span className="field-hint">{hint}</span> : null}
    </div>
  );
}

/** A password input with a show / hide button. */
export function PasswordInput({ value, onChange, autoComplete, autoFocus }: { value: string; onChange: (v: string) => void; autoComplete: string; autoFocus?: boolean }) {
  const [shown, setShown] = useState(false);
  return (
    <div className="pw-wrap">
      <input type={shown ? "text" : "password"} autoComplete={autoComplete} value={value} onChange={(e) => onChange(e.target.value)} autoFocus={autoFocus} />
      <button type="button" className="pw-eye" onClick={() => setShown(!shown)} aria-label={shown ? "Hide password" : "Show password"} title={shown ? "Hide password" : "Show password"}>
        {shown ? <EyeOff size={16} /> : <Eye size={16} />}
      </button>
    </div>
  );
}

export const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

// Same rule as the backend (app/auth/schemas.py PASSWORD_RE): 8+ characters, a capital letter, a number and a
// special character.
export const PASSWORD_RE = /^(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9\s]).{8,}$/;

const RULES: [string, (pw: string) => boolean][] = [
  ["8+ characters", (pw) => pw.length >= 8],
  ["a capital letter", (pw) => /[A-Z]/.test(pw)],
  ["a number", (pw) => /\d/.test(pw)],
  ["a special character", (pw) => /[^A-Za-z0-9\s]/.test(pw)],
];

export function passwordProblem(pw: string): string | undefined {
  if (!pw) return "Password is required";
  if (!PASSWORD_RE.test(pw)) {
    const missing = RULES.filter(([, ok]) => !ok(pw)).map(([r]) => r);
    return `Needs ${missing.join(", ")}`;
  }
  if (new TextEncoder().encode(pw).length > 72) return "At most 72 bytes";
  return undefined;
}

/** The password rules, each ticked as it is met. */
export function PasswordRules({ pw }: { pw: string }) {
  return (
    <span className="pw-rules">
      {RULES.map(([label, ok]) => (
        <span key={label} className={ok(pw) ? "ok" : ""}>
          <Check size={11} strokeWidth={3} style={{ opacity: ok(pw) ? 1 : 0.35 }} /> {label}
        </span>
      ))}
    </span>
  );
}
