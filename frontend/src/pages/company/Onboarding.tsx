import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../../lib/auth";
import { Logo } from "../../components/Shell";
import { Card, Steps } from "../../components/ui";
import { ConfirmJob, PostJob, type Posting } from "./JobEditor";

const STEPS = ["Create account", "Company profile", "Post a job", "Confirm job", "Dashboard"];

export default function CompanyOnboarding() {
  const { me, refresh } = useAuth();
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [posting, setPosting] = useState<Posting | null>(null);
  const [hints, setHints] = useState<Record<string, string>>({});
  const c = me?.company;
  return (
    <div style={{ minHeight: "100vh" }}>
      <div style={{ background: "var(--sidebar-bg)", padding: "14px 28px" }}><Logo /></div>
      <div className="content" style={{ margin: "0 auto" }}>
        <div className="page-head"><div><h1>Welcome, {me?.full_name.split(" ")[0]}</h1><p>Set up {c?.name ?? "your company"} and post your first job.</p></div></div>
        <Steps steps={STEPS} current={step} />
        {step === 1 && (
          <Card title="Company profile" sub="From your sign-up. Every job you post is attached to this company.">
            <dl className="kv" style={{ marginBottom: 16 }}>
              <dt>Company</dt><dd>{c?.name}</dd>
              <dt>Industry</dt><dd>{c?.industry ?? "—"}</dd>
              <dt>Size</dt><dd>{c?.company_size ? `${c.company_size} employees` : "—"}</dd>
              <dt>Website</dt><dd>{c?.website ? <a href={c.website} target="_blank" rel="noreferrer">{c.website}</a> : "—"}</dd>
              <dt>Recruiter</dt><dd>{me?.full_name}{me?.details.designation ? ` · ${String(me.details.designation)}` : ""}</dd>
            </dl>
            <button className="btn btn-primary" onClick={() => setStep(2)}>Looks right, continue</button>
          </Card>
        )}
        {step === 2 && <PostJob onParsed={(p, h) => (setPosting(p), setHints(h), setStep(3))} />}
        {step === 3 && posting && (
          <ConfirmJob posting={posting} hints={hints} label="Confirm and open my dashboard" onConfirmed={async () => { await refresh(); navigate(`/company/candidates?job=${posting.job_id}`, { replace: true }); }} />
        )}
      </div>
    </div>
  );
}
