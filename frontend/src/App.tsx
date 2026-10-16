import { lazy, Suspense, type ReactNode } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { homeFor, useAuth, type Role } from "./lib/auth";
import { Loading } from "./components/ui";
import Login from "./pages/auth/Login";
import Signup from "./pages/auth/Signup";
import ForgotPassword from "./pages/auth/ForgotPassword";

const StudentApp = lazy(() => import("./pages/student/StudentApp"));
const StudentOnboarding = lazy(() => import("./pages/student/Onboarding"));
const CompanyApp = lazy(() => import("./pages/company/CompanyApp"));
const CompanyOnboarding = lazy(() => import("./pages/company/Onboarding"));
const AdminApp = lazy(() => import("./pages/admin/AdminApp"));

/** Client-side guard for navigation only; the backend enforces every role (require_role) on its own. */
function Guard({ role, children }: { role: Role; children: ReactNode }) {
  const { me, loading } = useAuth();
  const loc = useLocation();
  if (loading) return <Loading label="Checking your session…" />;
  if (!me) return <Navigate to="/login" replace state={{ from: loc.pathname }} />;
  if (me.role !== role) return <Forbidden home={homeFor(me)} />;
  return <>{children}</>;
}

function Forbidden({ home }: { home: string }) {
  return (
    <div style={{ display: "grid", placeItems: "center", minHeight: "100vh", padding: 20 }}>
      <div className="card" style={{ maxWidth: 440, textAlign: "center" }}>
        <h2 style={{ marginBottom: 6 }}>Not available for your account</h2>
        <p className="muted">This dashboard belongs to a different account type.</p>
        <a className="btn btn-primary" href={home}>
          Go to my dashboard
        </a>
      </div>
    </div>
  );
}

function Home() {
  const { me, loading } = useAuth();
  if (loading) return <Loading label="Checking your session…" />;
  return <Navigate to={me ? homeFor(me) : "/login"} replace />;
}

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/student/onboarding" element={<Guard role="student"><StudentOnboarding /></Guard>} />
        <Route path="/student/*" element={<Guard role="student"><StudentApp /></Guard>} />
        <Route path="/company/onboarding" element={<Guard role="recruiter"><CompanyOnboarding /></Guard>} />
        <Route path="/company/*" element={<Guard role="recruiter"><CompanyApp /></Guard>} />
        <Route path="/admin/*" element={<Guard role="admin"><AdminApp /></Guard>} />
        <Route path="*" element={<Home />} />
      </Routes>
    </Suspense>
  );
}
