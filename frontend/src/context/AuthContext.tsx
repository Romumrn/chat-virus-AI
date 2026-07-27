/**
 * AuthContext — holds the current user (decoded from /api/auth/me) and the
 * login/register/logout actions. The JWT lives in localStorage (see lib/api);
 * this context resolves it to a user on mount and exposes role helpers used to
 * gate navigation and pages.
 */
import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import {
  api,
  setToken,
  clearToken,
  getToken,
  setUnauthorizedHandler,
  type UserInfo,
  type TokenResponse,
  type Role,
} from "@/lib/api";

const ROLE_LEVEL: Record<Role, number> = { user: 0, dev: 1, admin: 2 };

interface AuthState {
  user: UserInfo | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (body: RegisterBody) => Promise<void>;
  logout: () => void;
  hasRole: (min: Role) => boolean;
}

interface RegisterBody {
  first_name: string;
  last_name: string;
  email: string;
  password: string;
  registration_code?: string;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setUnauthorizedHandler(() => setUser(null));
    if (!getToken()) {
      setLoading(false);
      return;
    }
    api
      .get<UserInfo>("/api/auth/me")
      .then(setUser)
      .catch(() => clearToken())
      .finally(() => setLoading(false));
  }, []);

  async function login(email: string, password: string) {
    const res = await api.post<TokenResponse>("/api/auth/login", { email, password });
    setToken(res.access_token);
    setUser(await api.get<UserInfo>("/api/auth/me"));
  }

  async function register(body: RegisterBody) {
    const res = await api.post<TokenResponse>("/api/auth/register", body);
    setToken(res.access_token);
    setUser(await api.get<UserInfo>("/api/auth/me"));
  }

  function logout() {
    clearToken();
    setUser(null);
  }

  function hasRole(min: Role) {
    if (!user) return false;
    return ROLE_LEVEL[user.role] >= ROLE_LEVEL[min];
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, hasRole }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
