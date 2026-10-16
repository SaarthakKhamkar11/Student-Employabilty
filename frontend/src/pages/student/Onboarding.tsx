import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileUp } from "lucide-react";
import { api } from "../../lib/api";
import { useAuth } from "../../lib/auth";
import { Logo } from "../../components/Shell";
import { ThemeToggle } from "../../lib/theme";
import { Card, ErrorBox, Loading, Steps } from "../../components/ui";
import { initialCorrection, ProfileEditor, toPayload, type Correction, type StudentProfileData } from "./ProfileEditor";

const STEPS = ["Create account", "Upload resume", "Confirm profile", "Dashboard"];

export function ResumeUpload({ onDone }: { onDone: (profile: StudentProfileData) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function upload() {
    if (!file) return setError("Choose a .pdf, .docx or .txt file first");
    if (!/\.(pdf|docx|txt)$/i.test(file.name)) return setError("Only .pdf, .docx or .txt files are supported");
    if (file.size > 5 * 1024 * 1024) return setError("The file is larger than 5 MB");
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const r = await api<{ profile: StudentProfileData }>("/student/resume", { method: "POST", body: form });
      onDone(r.profile);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      {error && <ErrorBox error={error} />}
      <label className="empty" style={{ display: "block", cursor: "pointer", padding: 34 }}>
        <FileUp size={26} color="var(--brand-strong)" />
        <b style={{ marginTop: 8 }}>{file ? file.name : "Choose your resume"}</b>
        <span className="small">PDF, DOCX or TXT, up to 5 MB.</span>
        <input type="file" accept=".pdf,.docx,.txt" style={{ display: "none" }} onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </label>
      <div className="row" style={{ marginTop: 14 }}>
        <button className="btn btn-primary" onClick={upload} disabled={busy || !file}>
          {busy ? <span className="spinner" /> : null} {busy ? "Uploading…" : "Upload"}
        </button>
      </div>
    </div>
  );
}

export function ConfirmProfile({ profile, onConfirmed, confirmLabel = "Confirm profile" }: { profile: StudentProfileData; onConfirmed: () => void; confirmLabel?: string }) {
  const [value, setValue] = useState<Correction>(() => initialCorrection(profile));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function confirm() {
    if (value.graduation_year && (Number(value.graduation_year) < 1990 || Number(value.graduation_year) > 2035))
      return setError("Graduation year must be between 1990 and 2035");
    setBusy(true);
    setError(null);
    try {
      await api("/student/profile", { method: "PUT", json: toPayload(value) });
      onConfirmed();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Review your profile" sub="Correct anything wrong. Nothing is final until you confirm.">
      {error && <ErrorBox error={error} />}
      <ProfileEditor profile={profile} value={value} onChange={setValue} />
      <button className="btn btn-primary" onClick={confirm} disabled={busy}>
        {busy ? <span className="spinner" /> : null} {busy ? "Saving…" : confirmLabel}
      </button>
    </Card>
  );
}

export default function StudentOnboarding() {
  const { me, refresh } = useAuth();
  const navigate = useNavigate();
  const [profile, setProfile] = useState<StudentProfileData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<{ profile: StudentProfileData | null }>("/student/profile")
      .then((r) => setProfile(r.profile))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const step = profile ? 2 : 1;
  return (
    <div style={{ minHeight: "100vh" }}>
      <div className="onboard-bar"><Logo /><ThemeToggle /></div>
      <div className="content" style={{ margin: "0 auto" }}>
        <div className="page-head">
          <div>
            <h1>Welcome, {me?.full_name.split(" ")[0]}</h1>
            <p>Two steps before your dashboard: upload your resume, then check what we understood.</p>
          </div>
        </div>
        <Steps steps={STEPS} current={step} />
        {error && <ErrorBox error={error} />}
        {loading ? (
          <Loading />
        ) : profile ? (
          <>
            <div className="row" style={{ marginBottom: 12 }}>
              <button className="btn btn-sm" onClick={() => setProfile(null)}>Upload a different resume</button>
            </div>
            <ConfirmProfile
              profile={profile}
              onConfirmed={async () => {
                await refresh();
                navigate("/student", { replace: true });
              }}
              confirmLabel="Confirm and open my dashboard"
            />
          </>
        ) : (
          <Card title="Upload your resume" sub="We'll pick out your skills, projects, experience and education for you to review.">
            <ResumeUpload onDone={(p) => setProfile(p)} />
          </Card>
        )}
      </div>
    </div>
  );
}
