/**
 * Admin — ADMIN-only console: platform stats, a searchable user table with
 * inline role changes and deletion, and a drill-down to read any user's
 * conversations. All actions hit /api/admin/* (also role-guarded server-side).
 */
import { useEffect, useState } from "react";
import { Search, Trash2, X } from "lucide-react";
import { api, ApiError, type UserInfo, type Conversation, type ChatMessage, type Role } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import { Button, Card, Input, Select } from "@/components/ui";
import MessageBubble from "@/components/chat/MessageBubble";

interface Stats {
  users_total: number;
  users_by_role: Record<string, number>;
  conversations_total: number;
  messages_total: number;
}

function StatCard({ label, value }: { label: string; value: number | string }) {
  return (
    <Card className="p-4">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{value}</p>
    </Card>
  );
}

export default function Admin() {
  const { user: me } = useAuth();
  const [stats, setStats] = useState<Stats | null>(null);
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");

  // Drill-down state.
  const [viewing, setViewing] = useState<UserInfo | null>(null);
  const [convs, setConvs] = useState<Conversation[]>([]);
  const [openConv, setOpenConv] = useState<{ conv: Conversation; messages: ChatMessage[] } | null>(null);

  useEffect(() => {
    loadStats();
    loadUsers("");
  }, []);

  async function loadStats() {
    setStats(await api.get<Stats>("/api/admin/stats"));
  }
  async function loadUsers(q: string) {
    setUsers(await api.get<UserInfo[]>(`/api/admin/users${q ? `?q=${encodeURIComponent(q)}` : ""}`));
  }

  async function changeRole(email: string, role: Role) {
    setError("");
    try {
      await api.put(`/api/admin/users/${encodeURIComponent(email)}/role`, { role });
      loadUsers(query);
      loadStats();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not change role");
    }
  }

  async function removeUser(email: string) {
    if (!confirm(`Delete ${email} and all their conversations? This cannot be undone.`)) return;
    setError("");
    try {
      await api.del(`/api/admin/users/${encodeURIComponent(email)}`);
      loadUsers(query);
      loadStats();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not delete user");
    }
  }

  async function viewUser(u: UserInfo) {
    setViewing(u);
    setOpenConv(null);
    setConvs(await api.get<Conversation[]>(`/api/admin/users/${encodeURIComponent(u.email)}/conversations`));
  }

  async function readConversation(conv: Conversation) {
    const messages = await api.get<ChatMessage[]>(`/api/admin/conversations/${conv.id}/messages`);
    setOpenConv({ conv, messages });
  }

  return (
    <div className="h-full space-y-6 overflow-y-auto p-8">
      <h1 className="text-2xl font-semibold">Administration</h1>

      {stats && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <StatCard label="Users" value={stats.users_total} />
          <StatCard label="Admins / Devs" value={`${stats.users_by_role.admin || 0} / ${stats.users_by_role.dev || 0}`} />
          <StatCard label="Conversations" value={stats.conversations_total} />
          <StatCard label="Messages" value={stats.messages_total} />
        </div>
      )}

      {error && <p className="text-sm text-destructive">{error}</p>}

      {/* User table */}
      <Card className="overflow-hidden">
        <div className="flex items-center gap-2 border-b border-border p-3">
          <Search className="h-4 w-4 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              loadUsers(e.target.value);
            }}
            placeholder="Search by name or email…"
            className="h-8 border-0 focus-visible:ring-0"
          />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted/50 text-left text-muted-foreground">
              <tr>
                <th className="px-4 py-2 font-medium">Email</th>
                <th className="px-4 py-2 font-medium">Name</th>
                <th className="px-4 py-2 font-medium">Role</th>
                <th className="px-4 py-2 font-medium">Chats</th>
                <th className="px-4 py-2 font-medium">Last login</th>
                <th className="px-4 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.email} className="border-t border-border hover:bg-accent/30">
                  <td className="px-4 py-2">
                    <button className="text-primary hover:underline" onClick={() => viewUser(u)}>
                      {u.email}
                    </button>
                  </td>
                  <td className="px-4 py-2">{u.first_name} {u.last_name}</td>
                  <td className="px-4 py-2">
                    <Select
                      value={u.role}
                      disabled={u.email === me?.email}
                      onChange={(e) => changeRole(u.email, e.target.value as Role)}
                      className="h-8 w-24"
                    >
                      <option value="user">user</option>
                      <option value="dev">dev</option>
                      <option value="admin">admin</option>
                    </Select>
                  </td>
                  <td className="px-4 py-2">{u.n_conversations ?? 0}</td>
                  <td className="px-4 py-2 text-muted-foreground">
                    {u.last_login ? new Date(u.last_login).toLocaleDateString() : "never"}
                  </td>
                  <td className="px-4 py-2 text-right">
                    <button
                      onClick={() => removeUser(u.email)}
                      disabled={u.email === me?.email}
                      className="text-muted-foreground hover:text-destructive disabled:opacity-30"
                      title="Delete user"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* User conversations drill-down */}
      {viewing && (
        <Card className="p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-medium">Conversations of {viewing.email}</h2>
            <Button variant="ghost" size="icon" onClick={() => setViewing(null)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          {convs.length === 0 ? (
            <p className="text-sm text-muted-foreground">No conversations.</p>
          ) : (
            <ul className="space-y-1">
              {convs.map((c) => (
                <li key={c.id}>
                  <button
                    className="text-sm text-primary hover:underline"
                    onClick={() => readConversation(c)}
                  >
                    {c.title || "Untitled"}
                  </button>
                  <span className="ml-2 text-xs text-muted-foreground">
                    {c.updated_at ? new Date(c.updated_at).toLocaleString() : ""}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      )}

      {/* Conversation reader modal */}
      {openConv && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          onClick={() => setOpenConv(null)}
        >
          <Card
            className="flex max-h-[80vh] w-full max-w-3xl flex-col p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-3 flex items-center justify-between">
              <h3 className="font-medium">{openConv.conv.title}</h3>
              <Button variant="ghost" size="icon" onClick={() => setOpenConv(null)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="min-h-0 flex-1 space-y-4 overflow-y-auto">
              {openConv.messages.map((m, i) => (
                <MessageBubble key={i} msg={m} />
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
