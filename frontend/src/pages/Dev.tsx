/**
 * Dev — DEV/ADMIN tooling: inspect and invoke MCP tools directly, and tail the
 * agent log. Backed by /api/dev/* (role-guarded server-side).
 */
import { useEffect, useState } from "react";
import { ChevronDown, ChevronRight, Play, RefreshCw } from "lucide-react";
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

interface ErrorReport {
  id: number;
  user_email: string | null;
  question: string;
  answer: string;
  executed_codes: string[];
  comment: string;
  recent_logs: string[];
  status: "open" | "in_progress" | "done";
  created_at: string | null;
  updated_at: string | null;
}

const STATUS_STYLES: Record<ErrorReport["status"], string> = {
  open: "bg-destructive/15 text-destructive",
  in_progress: "bg-yellow-500/15 text-yellow-600 dark:text-yellow-400",
  done: "bg-green-500/15 text-green-600 dark:text-green-400",
};
const STATUS_LABELS: Record<ErrorReport["status"], string> = {
  open: "Open",
  in_progress: "In progress",
  done: "Done",
};

function ReportRow({
  report,
  onStatus,
}: {
  report: ErrorReport;
  onStatus: (id: number, status: ErrorReport["status"]) => void;
}) {
  const [open, setOpen] = useState(false);
  const date = report.created_at ? new Date(report.created_at).toLocaleString() : "—";

  return (
    <div className="rounded-md border border-border">
      <div className="flex items-center gap-3 p-3">
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
        >
          {open ? (
            <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
          )}
          <span className="shrink-0 font-mono text-xs text-muted-foreground">#{report.id}</span>
          <span
            className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${STATUS_STYLES[report.status]}`}
          >
            {STATUS_LABELS[report.status]}
          </span>
          <span className="truncate text-sm">
            {report.comment || report.question || "(no comment)"}
          </span>
        </button>
        <span className="hidden shrink-0 text-xs text-muted-foreground sm:inline">
          {report.user_email || "unknown"}
        </span>
        <span className="hidden shrink-0 text-xs text-muted-foreground md:inline">{date}</span>
        <Select
          value={report.status}
          onChange={(e) => onStatus(report.id, e.target.value as ErrorReport["status"])}
          className="w-32 shrink-0"
        >
          <option value="open">Open</option>
          <option value="in_progress">In progress</option>
          <option value="done">Done</option>
        </Select>
      </div>
      {open && (
        <div className="space-y-3 border-t border-border p-3 text-sm">
          <div className="text-xs text-muted-foreground">
            {report.user_email || "unknown user"} · {date}
          </div>
          {report.comment && (
            <div>
              <div className="text-xs font-medium text-muted-foreground">Comment</div>
              <p className="whitespace-pre-wrap">{report.comment}</p>
            </div>
          )}
          <div>
            <div className="text-xs font-medium text-muted-foreground">Question</div>
            <p className="whitespace-pre-wrap">{report.question || "—"}</p>
          </div>
          <div>
            <div className="text-xs font-medium text-muted-foreground">Answer</div>
            <p className="whitespace-pre-wrap">{report.answer || "—"}</p>
          </div>
          {report.executed_codes.length > 0 && (
            <div>
              <div className="text-xs font-medium text-muted-foreground">Executed code</div>
              <pre className="mt-1 max-h-48 overflow-auto rounded bg-muted p-2 text-xs">
                {report.executed_codes.join("\n\n")}
              </pre>
            </div>
          )}
          {report.recent_logs.length > 0 && (
            <div>
              <div className="text-xs font-medium text-muted-foreground">Recent logs</div>
              <pre className="mt-1 max-h-48 overflow-auto rounded bg-muted p-2 text-xs">
                {report.recent_logs.join("\n")}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ErrorReports() {
  const [reports, setReports] = useState<ErrorReport[]>([]);
  const [filter, setFilter] = useState<"" | ErrorReport["status"]>("");
  const [busy, setBusy] = useState(false);

  async function load() {
    setBusy(true);
    try {
      const q = filter ? `?status_filter=${filter}` : "";
      const res = await api.get<{ reports: ErrorReport[] }>(`/api/dev/reports${q}`);
      setReports(res.reports);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter]);

  async function setStatus(id: number, status: ErrorReport["status"]) {
    // Optimistic update, then persist.
    setReports((rs) => rs.map((r) => (r.id === id ? { ...r, status } : r)));
    try {
      await api.patch(`/api/dev/reports/${id}`, { status });
    } catch {
      load(); // revert to server truth on failure
    }
  }

  const openCount = reports.filter((r) => r.status === "open").length;

  return (
    <Card className="space-y-3 p-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="font-medium">
          Error reports{" "}
          {openCount > 0 && (
            <span className="ml-1 rounded-full bg-destructive/15 px-2 py-0.5 text-xs font-medium text-destructive">
              {openCount} open
            </span>
          )}
        </h2>
        <div className="flex items-center gap-2">
          <Select
            value={filter}
            onChange={(e) => setFilter(e.target.value as "" | ErrorReport["status"])}
            className="w-40"
          >
            <option value="">All statuses</option>
            <option value="open">Open</option>
            <option value="in_progress">In progress</option>
            <option value="done">Done</option>
          </Select>
          <Button variant="outline" size="sm" onClick={load} disabled={busy}>
            <RefreshCw className={busy ? "h-4 w-4 animate-spin" : "h-4 w-4"} /> Refresh
          </Button>
        </div>
      </div>
      {reports.length ? (
        <div className="space-y-2">
          {reports.map((r) => (
            <ReportRow key={r.id} report={r} onStatus={setStatus} />
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">— no error reports —</p>
      )}
    </Card>
  );
}

export default function Dev() {
  return (
    <div className="h-full space-y-6 overflow-y-auto p-8">
      <h1 className="text-2xl font-semibold">Developer tools</h1>
      <McpTester />
      <LogViewer />
      <ErrorReports />
    </div>
  );
}
