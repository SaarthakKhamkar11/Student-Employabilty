import type { ReactNode } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { LogOut, Search } from "lucide-react";
import { useAuth } from "../lib/auth";
import { initials } from "../lib/format";

export interface NavEntry {
  to: string;
  label: string;
  icon: ReactNode;
  end?: boolean;
}

const ROLE_LABEL = { student: "Student", recruiter: "Company", admin: "Admin · Research" } as const;

export function Logo({ light = true }: { light?: boolean }) {
  return (
    <div className="brand">
      <img src="/logo.svg" alt="" />
      <div className="brand-name" style={light ? undefined : { color: "var(--text)" }}>
        Career<span>Compass</span>
      </div>
    </div>
  );
}

export function Shell({
  nav,
  children,
  topRight,
  searchPlaceholder = "Search…",
  onSearch,
}: {
  nav: NavEntry[];
  children: ReactNode;
  topRight?: ReactNode;
  searchPlaceholder?: string;
  onSearch?: (q: string) => void;
}) {
  const { me, logout } = useAuth();
  const navigate = useNavigate();
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
        <div className="sidebar-foot">
          Every number on these pages comes from the live API. Missing values read <i>Not yet evaluated</i>.
        </div>
      </aside>
      <div className="main">
        <header className="topbar">
          <label className="search">
            <Search size={15} />
            <input placeholder={searchPlaceholder} onChange={(e) => onSearch?.(e.target.value)} disabled={!onSearch} />
          </label>
          <div className="topbar-right">
            {topRight}
            {me && (
              <>
                <div className="avatar" aria-hidden>
                  {initials(me.full_name)}
                </div>
                <div className="user-meta">
                  <b>{me.full_name}</b>
                  <small>{me.role === "recruiter" && me.company ? `${me.company.name} · Company` : ROLE_LABEL[me.role]}</small>
                </div>
                <button
                  className="btn btn-ghost btn-sm"
                  title="Log out"
                  onClick={() => {
                    logout();
                    navigate("/login");
                  }}
                >
                  <LogOut size={15} />
                </button>
              </>
            )}
          </div>
        </header>
        <main className="content">{children}</main>
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
