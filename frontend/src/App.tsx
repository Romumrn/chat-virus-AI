import { Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { Spinner } from "@/components/ui";
import Layout from "@/components/Layout";
import Login from "@/pages/Login";
import Register from "@/pages/Register";
import Chat from "@/pages/Chat";
import Info from "@/pages/Info";
import Account from "@/pages/Account";
import Admin from "@/pages/Admin";
import Dev from "@/pages/Dev";
import type { Role } from "@/lib/api";

function RequireRole({ min, children }: { min?: Role; children: JSX.Element }) {
  const { user, loading, hasRole } = useAuth();
  if (loading)
    return (
      <div className="flex h-screen items-center justify-center">
        <Spinner className="h-8 w-8 text-primary" />
      </div>
    );
  if (!user) return <Navigate to="/login" replace />;
  if (min && !hasRole(min)) return <Navigate to="/" replace />;
  return children;
}

export default function App() {
  const { user, loading } = useAuth();

  return (
    <Routes>
      <Route path="/login" element={user && !loading ? <Navigate to="/" replace /> : <Login />} />
      <Route path="/register" element={user && !loading ? <Navigate to="/" replace /> : <Register />} />

      <Route
        element={
          <RequireRole>
            <Layout />
          </RequireRole>
        }
      >
        <Route path="/" element={<Chat />} />
        <Route path="/account" element={<Account />} />
        <Route path="/info" element={<Info />} />
        <Route
          path="/dev"
          element={
            <RequireRole min="dev">
              <Dev />
            </RequireRole>
          }
        />
        <Route
          path="/admin"
          element={
            <RequireRole min="admin">
              <Admin />
            </RequireRole>
          }
        />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
