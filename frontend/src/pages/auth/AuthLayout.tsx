import type { ReactNode } from "react";
import { Building2, GraduationCap } from "lucide-react";
import { Logo } from "../../components/Shell";

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
        <div className="card auth-card" style={{ padding: 28 }}>
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
  children,
}: {
  label: string;
  error?: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className={`field ${error ? "invalid" : ""}`}>
      <label>{label}</label>
      {children}
      {error ? <span className="field-error">{error}</span> : hint ? <span className="field-hint">{hint}</span> : null}
    </div>
  );
}

export const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

export function passwordProblem(pw: string): string | undefined {
  if (pw.length < 8) return "At least 8 characters";
  if (!/[A-Za-z]/.test(pw) || !/\d/.test(pw)) return "Use at least one letter and one number";
  if (new TextEncoder().encode(pw).length > 72) return "At most 72 bytes";
  return undefined;
}
