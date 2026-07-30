/**
 * MessageBubble — one chat turn. User turns are a plain right-aligned bubble;
 * assistant turns render Markdown, any Plotly figures, and a collapsible
 * Sources panel (Wikipedia / PubMed / NCBI / executed code).
 */
import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen, ChevronDown, ChevronRight, Code2, Flag } from "lucide-react";
import PlotlyFigure from "./PlotlyFigure";
import { api, type ChatMessage } from "@/lib/api";
import { cn } from "@/lib/utils";

function Sources({ msg }: { msg: ChatMessage }) {
  const [open, setOpen] = useState(false);
  const wiki = msg.wikipedia_urls || [];
  const pubmed = msg.pubmed_urls || [];
  const ncbi = msg.ncbi_urls || [];
  const codes = msg.executed_codes || [];
  if (!wiki.length && !pubmed.length && !ncbi.length && !codes.length) return null;

  return (
    <div className="mt-3 rounded-md border border-border bg-muted/40">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-sm font-medium"
      >
        {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        <BookOpen className="h-4 w-4" /> Sources
        <span className="text-xs text-muted-foreground">
          ({wiki.length + pubmed.length + ncbi.length} refs
          {codes.length ? `, ${codes.length} code` : ""})
        </span>
      </button>
      {open && (
        <div className="space-y-3 px-4 pb-3 text-sm">
          {wiki.length > 0 && (
            <div>
              <p className="font-medium">📘 Wikipedia</p>
              {wiki.map((u) => (
                <a key={u} href={u} target="_blank" rel="noreferrer" className="block text-primary hover:underline">
                  {decodeURIComponent(u.split("/").pop() || u).replace(/_/g, " ")}
                </a>
              ))}
            </div>
          )}
          {pubmed.length > 0 && (
            <div>
              <p className="font-medium">🔬 PubMed</p>
              {pubmed.map((u) => (
                <a key={u} href={u} target="_blank" rel="noreferrer" className="block text-primary hover:underline">
                  PMID: {u.replace(/\/$/, "").split("/").pop()}
                </a>
              ))}
            </div>
          )}
          {ncbi.length > 0 && (
            <div>
              <p className="font-medium">🧬 NCBI Taxonomy</p>
              {ncbi.map((u) => (
                <a key={u} href={u} target="_blank" rel="noreferrer" className="block text-primary hover:underline">
                  TaxID: {u.split("id=").pop()}
                </a>
              ))}
            </div>
          )}
          {codes.length > 0 && (
            <div>
              <p className="flex items-center gap-1 font-medium">
                <Code2 className="h-4 w-4" /> Executed code
              </p>
              {codes.map((c, i) => (
                <pre key={i} className="mt-1 overflow-x-auto rounded bg-background p-2 text-xs">
                  <code>{c}</code>
                </pre>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * ReportError — the "Report an error" control: a button that opens an
 * optional-comment box and POSTs the question/answer/code to /api/report.
 * Only shown for assistant turns in the live chat (when `question` is
 * provided), never in the read-only admin viewer.
 */
function ReportError({ msg, question }: { msg: ChatMessage; question: string }) {
  const [open, setOpen] = useState(false);
  const [comment, setComment] = useState("");
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);

  if (sent) {
    return (
      <p className="mt-2 text-xs text-muted-foreground">
        ⚠️ Error reported — thank you for your feedback.
      </p>
    );
  }

  async function send() {
    setBusy(true);
    try {
      await api.post("/api/report", {
        question,
        answer: msg.content,
        executed_codes: msg.executed_codes || [],
        comment,
      });
      setSent(true);
      setOpen(false);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-destructive"
        title="Signal a wrong or misleading answer"
      >
        <Flag className="h-3.5 w-3.5" /> Report an error
      </button>
      {open && (
        <div className="mt-2 space-y-2 rounded-md border border-border p-3">
          <p className="text-xs font-medium">What went wrong? (optional)</p>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="e.g. Wrong species name, incorrect count, hallucinated data…"
            rows={2}
            className="w-full resize-none rounded-md border border-input bg-background p-2 text-sm"
          />
          <div className="flex gap-2">
            <button
              onClick={send}
              disabled={busy}
              className="rounded-md bg-primary px-3 py-1 text-xs font-medium text-primary-foreground disabled:opacity-50"
            >
              {busy ? "Sending…" : "Send report"}
            </button>
            <button
              onClick={() => setOpen(false)}
              className="rounded-md border border-border px-3 py-1 text-xs"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function MessageBubble({
  msg,
  question,
}: {
  msg: ChatMessage;
  question?: string;
}) {
  const isUser = msg.role === "user";
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-2.5",
          isUser
            ? "bg-primary text-primary-foreground"
            : "border border-border bg-card text-card-foreground",
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{msg.content}</p>
        ) : (
          <>
            <div className="prose-chat max-w-none">
              <Markdown remarkPlugins={[remarkGfm]}>{msg.content}</Markdown>
            </div>
            {(msg.figures || []).map((fig, i) => (
              <PlotlyFigure key={i} figure={fig} />
            ))}
            <Sources msg={msg} />
            {question !== undefined && <ReportError msg={msg} question={question} />}
          </>
        )}
      </div>
    </div>
  );
}
