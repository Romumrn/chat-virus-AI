/**
 * MessageBubble — one chat turn. User turns are a plain right-aligned bubble;
 * assistant turns render Markdown, any Plotly figures, and a collapsible
 * Sources panel (Wikipedia / PubMed / NCBI / executed code).
 */
import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen, ChevronDown, ChevronRight, Code2 } from "lucide-react";
import PlotlyFigure from "./PlotlyFigure";
import type { ChatMessage } from "@/lib/api";
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

export default function MessageBubble({ msg }: { msg: ChatMessage }) {
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
          </>
        )}
      </div>
    </div>
  );
}
