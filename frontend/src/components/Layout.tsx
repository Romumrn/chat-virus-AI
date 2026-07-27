/**
 * Layout — the authenticated shell: a left sidebar with role-gated navigation
 * and a header carrying the user badge, theme toggle and logout. The Admin link
 * shows only for admins, Dev for dev+; the API also enforces this server-side.
 */
import { useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import {
  MessageSquare,
  User as UserIcon,
  Shield,
  Wrench,
  LogOut,
  Moon,
  Sun,
  Info as InfoIcon,
} from "lucide-react";
import { useAuth } from "@/context/AuthContext";
import { Badge, Button } from "@/components/ui";
import { cn } from "@/lib/utils";
import type { Role } from "@/lib/api";

interface NavItem {
  to: string;
  label: string;
  icon: typeof MessageSquare;
  min?: Role;
}

const NAV: NavItem[] = [
  { to: "/", label: "Chat", icon: MessageSquare },
  { to: "/account", label: "My account", icon: UserIcon },
  { to: "/info", label: "Info", icon: InfoIcon },
  { to: "/dev", label: "Developer", icon: Wrench, min: "dev" },
  { to: "/admin", label: "Administration", icon: Shield, min: "admin" },
];

const ROLE_BADGE: Record<Role, string> = {
  user: "bg-secondary text-secondary-foreground",
  dev: "bg-blue-500/15 text-blue-500",
  admin: "bg-primary/15 text-primary",
};

export default function Layout() {
  const { user, logout, hasRole } = useAuth();
  const navigate = useNavigate();
  const [dark, setDark] = useState(
    () => document.documentElement.getAttribute("data-theme") === "dark",
  );

  function toggleTheme() {
    const next = !dark;
    setDark(next);
    document.documentElement.setAttribute("data-theme", next ? "dark" : "light");
  }

  function handleLogout() {
    logout();
    navigate("/login");
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="flex w-60 shrink-0 flex-col border-r border-border bg-card">
        <div className="flex items-center gap-2 px-5 py-4 text-lg font-semibold">
          <span>🦠</span> Viromech@t
        </div>
        <nav className="flex-1 space-y-1 px-3">
          {NAV.filter((item) => !item.min || hasRole(item.min)).map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                )
              }
            >
              <item.icon className="h-4 w-4" />
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-border p-3 text-xs text-muted-foreground">
          <div>Viromech@t · SHAPE-Med@Lyon</div>
          <a
            href="https://github.com/Romumrn/viromechat"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline"
          >
            Repo
          </a>
        </div>
      </aside>

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-border px-5">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">
              {user?.first_name} {user?.last_name}
            </span>
            {user && (
              <Badge className={ROLE_BADGE[user.role]}>{user.role.toUpperCase()}</Badge>
            )}
          </div>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" onClick={toggleTheme} title="Toggle theme">
              {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
            <Button variant="ghost" size="sm" onClick={handleLogout}>
              <LogOut className="h-4 w-4" /> Logout
            </Button>
          </div>
        </header>
        <main className="min-h-0 flex-1 overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
