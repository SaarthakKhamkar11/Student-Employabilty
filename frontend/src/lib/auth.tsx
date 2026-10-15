import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, getToken, setToken } from "./api";

export type Role = "student" | "recruiter" | "admin";

export interface Me {
  user_id: string;
  email: string;
  role: Role;
  full_name: string;
  student_id: string | null;
  company: { company_id: number; name: string; industry: string | null; company_size: string | null; website: string | null } | null;
  details: Record<string, unknown>;
  onboarding: { profile_confirmed_at: string | null; resume_uploaded: boolean; jobs_posted?: number };
}

interface AuthState {
  me: Me | null;
  loading: boolean;
  login: (token: string) => Promise<Me>;
  logout: () => void;
  refresh: () => Promise<Me | null>;
}

const Ctx = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [me, setMe] = useState<Me | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    if (!getToken()) {
      setMe(null);
      return null;
    }
    try {
      const m = await api<Me>("/auth/me");
      setMe(m);
      return m;
    } catch {
      setToken(null);
      setMe(null);
      return null;
    }
  }, []);

  useEffect(() => {
    refresh().finally(() => setLoading(false));
    const onLogout = () => setMe(null);
    window.addEventListener("cc:logout", onLogout);
    return () => window.removeEventListener("cc:logout", onLogout);
  }, [refresh]);

  const login = useCallback(async (token: string) => {
    setToken(token);
    const m = await api<Me>("/auth/me");
    setMe(m);
    return m;
  }, []);

  const logout = useCallback(() => {
    setToken(null);
    setMe(null);
  }, []);

  return <Ctx.Provider value={{ me, loading, login, logout, refresh }}>{children}</Ctx.Provider>;
}

export function useAuth(): AuthState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAuth outside AuthProvider");
  return v;
}

export function homeFor(me: Me): string {
  if (me.role === "admin") return "/admin";
  if (me.role === "recruiter") return me.onboarding.jobs_posted ? "/company" : "/company/onboarding";
  return me.onboarding.profile_confirmed_at ? "/student" : "/student/onboarding";
}
