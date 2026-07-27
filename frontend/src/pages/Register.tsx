import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { Button, Card, Input, Label } from "@/components/ui";
import { ApiError } from "@/lib/api";

const PASSWORD_RULES: [RegExp | ((p: string) => boolean), string][] = [
  [(p) => p.length >= 12, "At least 12 characters"],
  [/[a-z]/, "1 lowercase letter"],
  [/[A-Z]/, "1 uppercase letter"],
  [/[0-9]/, "1 digit"],
  [/[^A-Za-z0-9]/, "1 special character"],
];

function checkRule(rule: RegExp | ((p: string) => boolean), p: string): boolean {
  return typeof rule === "function" ? rule(p) : rule.test(p);
}

export default function Register() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({
    first_name: "",
    last_name: "",
    email: "",
    password: "",
    registration_code: "",
  });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  function set(k: keyof typeof form, v: string) {
    setForm((f) => ({ ...f, [k]: v }));
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await register(form);
      navigate("/");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Registration failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md p-8">
        <div className="mb-6 text-center">
          <div className="text-3xl">🦠</div>
          <h1 className="mt-2 text-2xl font-semibold">Create an account</h1>
          <p className="text-sm text-muted-foreground">
            Use your institutional email address
          </p>
        </div>
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label>First name</Label>
              <Input value={form.first_name} onChange={(e) => set("first_name", e.target.value)} required />
            </div>
            <div className="space-y-1.5">
              <Label>Last name</Label>
              <Input value={form.last_name} onChange={(e) => set("last_name", e.target.value)} required />
            </div>
          </div>
          <div className="space-y-1.5">
            <Label>Institutional email</Label>
            <Input
              type="email"
              value={form.email}
              onChange={(e) => set("email", e.target.value)}
              placeholder="you@university.edu"
              required
            />
          </div>
          <div className="space-y-1.5">
            <Label>Password</Label>
            <Input
              type="password"
              value={form.password}
              onChange={(e) => set("password", e.target.value)}
              required
            />
            {form.password && (
              <ul className="mt-1 space-y-0.5 text-xs">
                {PASSWORD_RULES.map(([rule, label]) => {
                  const ok = checkRule(rule, form.password);
                  return (
                    <li key={label} className={ok ? "text-primary" : "text-muted-foreground"}>
                      {ok ? "✓" : "○"} {label}
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
          <div className="space-y-1.5">
            <Label>Registration code</Label>
            <Input
              value={form.registration_code}
              onChange={(e) => set("registration_code", e.target.value)}
              placeholder="Invite code (if required)"
            />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <Button type="submit" className="w-full" disabled={busy}>
            {busy ? "Creating…" : "Create account"}
          </Button>
        </form>
        <p className="mt-4 text-center text-sm text-muted-foreground">
          Already have an account?{" "}
          <Link to="/login" className="font-medium text-primary hover:underline">
            Sign in
          </Link>
        </p>
      </Card>
    </div>
  );
}
