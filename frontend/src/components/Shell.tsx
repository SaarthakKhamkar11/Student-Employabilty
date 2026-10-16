import type { ReactNode } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { ChevronLeft, LogOut, Search } from "lucide-react";
import { useAuth } from "../lib/auth";
import { initials } from "../lib/format";
import { ThemeToggle } from "../lib/theme";

export interface NavEntry {
  to: string;
  label: string;
  icon: ReactNode;
  end?: boolean;
}

const ROLE_LABEL = { student: "Student", recruiter: "Company", admin: "Admin · Research" } as const;

export function Logo() {
  return (
    <div className="brand">
      <img src="/logo.svg" alt="" />
      <div className="brand-name">
        Career<span>Compass</span>
      </div>
    </div>
  );
}

/** Back to the previous page; with no in-app history (a direct link), up to the dashboard home. */
export function BackButton({ home }: { home: string }) {
  const navigate = useNavigate();
  const canGoBack = ((window.history.state as { idx?: number } | null)?.idx ?? 0) > 0;
  return (
    <button type="button" className="icon-btn" title="Back" aria-label="Back" onClick={() => (canGoBack ? navigate(-1) : navigate(home))}>
      <ChevronLeft size={19} />
    </button>
  );
}

export function Shell({
  nav,
  children,
  topRight,
  searchPlaceholder = "Search…",
  onSearch,
  profileTo,
}: {
  nav: NavEntry[];
  children: ReactNode;
  topRight?: ReactNode;
  searchPlaceholder?: string;
  onSearch?: (q: string) => void;
  /** Where the avatar in the top right leads (the account's edit-profile page); omitted, it is not a link. */
  profileTo?: string;
}) {
  const { me, logout } = useAuth();
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const home = nav[0]?.to ?? "/";
  const atHome = pathname.replace(/\/$/, "") === home;
  const who = me && (
    <>
      <div className="avatar" aria-hidden>
        {initials(me.full_name)}
      </div>
      <div className="user-meta">
        <b>{me.full_name}</b>
        <small>{me.role === "recruiter" && me.company ? `${me.company.name} · Company` : ROLE_LABEL[me.role]}</small>
      </div>
    </>
  );
  return (
    <div className="shell">
      <aside className="sidebar">
        <Logo />
        <nav className="nav">
          {nav.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
              {n.icon}
              {n.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <div className="main">
        <header className="topbar">
          {!atHome && <BackButton home={home} />}
          <label className="search">
            <Search size={15} />
            <input placeholder={searchPlaceholder} onChange={(e) => onSearch?.(e.target.value)} disabled={!onSearch} />
          </label>
          <div className="topbar-right">
            {topRight}
            <ThemeToggle />
            {me && (
              <>
                <span className="topbar-sep" />
                {profileTo ? (
                  <button type="button" className={`profile-chip ${pathname.startsWith(profileTo) ? "active" : ""}`}
                    title="Edit profile" onClick={() => navigate(profileTo)}>
                    {who}
                  </button>
                ) : (
                  <div className="profile-chip">{who}</div>
                )}
                <button
                  className="icon-btn"
                  title="Log out"
                  aria-label="Log out"
                  onClick={() => {
                    logout();
                    navigate("/login");
                  }}
                >
                  <LogOut size={16} />
                </button>
              </>
            )}
          </div>
        </header>
        <main className="content">
          <div className="page" key={pathname}>
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

export function PageHead({ title, sub, actions }: { title: string; sub?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {sub && <p>{sub}</p>}
      </div>
      {actions && <div className="actions">{actions}</div>}
    </div>
  );
}
