/**
 * Dev — DEV/ADMIN tooling: inspect and invoke MCP tools directly, and tail the
 * agent log. Backed by /api/dev/* (role-guarded server-side).
 */
import { useEffect, useState } from "react";
import { Play, RefreshCw } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { Button, Card, Select, Textarea } from "@/components/ui";

interface ToolSpec {
  type: string;
  function: { name: string; description: string; parameters: any };
}

function McpTester() {
  const [tools, setTools] = useState<ToolSpec[]>([]);
  const [selected, setSelected] = useState("");
  const [args, setArgs] = useState("{}");
  const [result, setResult] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .get<ToolSpec[]>("/api/dev/mcp/tools")
      .then((t) => {
        setTools(t);
        if (t[0]) setSelected(t[0].function.name);
      })
      .catch((e) => setResult(`Failed to load tools: ${e.message}`));
  }, []);

  const current = tools.find((t) => t.function.name === selected);

  async function call() {
    setBusy(true);
    setResult("");
    let parsed: any;
    try {
      parsed = JSON.parse(args || "{}");
    } catch {
      setResult("Arguments are not valid JSON.");
      setBusy(false);
      return;
    }
    try {
      const res = await api.post("/api/dev/mcp/call", { name: selected, arguments: parsed });
      setResult(JSON.stringify(res, null, 2));
    } catch (err) {
      setResult(err instanceof ApiError ? `Error: ${err.message}` : "Call failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="space-y-4 p-6">
      <h2 className="font-medium">MCP tool tester</h2>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-3">
          <div>
            <label className="text-xs text-muted-foreground">Tool</label>
            <Select value={selected} onChange={(e) => setSelected(e.target.value)}>
              {tools.map((t) => (
                <option key={t.function.name} value={t.function.name}>
                  {t.function.name}
                </option>
              ))}
            </Select>
          </div>
          {current && (
            <p className="text-xs text-muted-foreground">{current.function.description}</p>
          )}
          <div>
            <label className="text-xs text-muted-foreground">Arguments (JSON)</label>
            <Textarea
              value={args}
              onChange={(e) => setArgs(e.target.value)}
              rows={6}
              className="font-mono text-xs"
            />
          </div>
          <Button onClick={call} disabled={busy || !selected}>
            <Play className="h-4 w-4" /> {busy ? "Calling…" : "Call tool"}
          </Button>
        </div>
        <div>
          <label className="text-xs text-muted-foreground">Schema & result</label>
          <pre className="mt-1 max-h-32 overflow-auto rounded bg-muted p-2 text-xs">
            {current ? JSON.stringify(current.function.parameters, null, 2) : ""}
          </pre>
          <pre className="mt-2 max-h-64 overflow-auto rounded bg-muted p-2 text-xs">
            {result || "— run a tool to see its result —"}
          </pre>
        </div>
      </div>
    </Card>
  );
}

function LogViewer() {
  const [lines, setLines] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  async function load() {
    setBusy(true);
    try {
      const res = await api.get<{ lines: string[] }>("/api/dev/logs?lines=300");
      setLines(res.lines);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <Card className="space-y-3 p-6">
      <div className="flex items-center justify-between">
        <h2 className="font-medium">Agent logs</h2>
        <Button variant="outline" size="sm" onClick={load} disabled={busy}>
          <RefreshCw className={busy ? "h-4 w-4 animate-spin" : "h-4 w-4"} /> Refresh
        </Button>
      </div>
      <pre className="max-h-96 overflow-auto rounded bg-muted p-3 text-xs leading-relaxed">
        {lines.length ? lines.join("\n") : "— no log lines —"}
      </pre>
    </Card>
  );
}

export default function Dev() {
  return (
    <div className="h-full space-y-6 overflow-y-auto p-8">
      <h1 className="text-2xl font-semibold">Developer tools</h1>
      <McpTester />
      <LogViewer />
    </div>
  );
}
