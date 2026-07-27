import { useState, type FormEvent } from "react";
import { api, ApiError } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";
import { Button, Card, Input, Label, Badge } from "@/components/ui";

export default function Account() {
  const { user } = useAuth();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setMsg(null);
    setBusy(true);
    try {
      await api.put("/api/auth/me/password", {
        current_password: current,
        new_password: next,
      });
      setMsg({ ok: true, text: "Password updated." });
      setCurrent("");
      setNext("");
    } catch (err) {
      setMsg({ ok: false, text: err instanceof ApiError ? err.message : "Update failed" });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 overflow-y-auto p-8">
      <h1 className="text-2xl font-semibold">My account</h1>

      <Card className="space-y-3 p-6">
        <h2 className="font-medium">Profile</h2>
        <dl className="grid grid-cols-[8rem_1fr] gap-y-2 text-sm">
          <dt className="text-muted-foreground">Name</dt>
          <dd>{user?.first_name} {user?.last_name}</dd>
          <dt className="text-muted-foreground">Email</dt>
          <dd>{user?.email}</dd>
          <dt className="text-muted-foreground">Role</dt>
          <dd>{user && <Badge>{user.role.toUpperCase()}</Badge>}</dd>
        </dl>
      </Card>

      <Card className="space-y-4 p-6">
        <h2 className="font-medium">Change password</h2>
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <Label>Current password</Label>
            <Input type="password" value={current} onChange={(e) => setCurrent(e.target.value)} required />
          </div>
          <div className="space-y-1.5">
            <Label>New password</Label>
            <Input type="password" value={next} onChange={(e) => setNext(e.target.value)} required />
            <p className="text-xs text-muted-foreground">
              At least 12 characters, with upper/lowercase, a digit and a special character.
            </p>
          </div>
          {msg && (
            <p className={msg.ok ? "text-sm text-primary" : "text-sm text-destructive"}>{msg.text}</p>
          )}
          <Button type="submit" disabled={busy}>
            {busy ? "Saving…" : "Update password"}
          </Button>
        </form>
      </Card>
    </div>
  );
}
