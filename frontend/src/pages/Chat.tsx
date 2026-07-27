/**
 * Chat — the main workspace: a conversation list on the left, the message
 * thread + composer on the right. Answers stream in over SSE (lib/sse); while
 * a turn is running we show a live activity panel (status + tool calls +
 * figures). DEV/ADMIN users also get an expandable Expert-mode settings panel.
 */
import { useEffect, useRef, useState } from "react";
import {
  Plus,
  Trash2,
  Send,
  Sliders,
  Loader2,
  MessageSquare,
} from "lucide-react";
import { api, type Conversation, type ChatMessage } from "@/lib/api";
import { streamChat, type AgentEvent } from "@/lib/sse";
import { useAuth } from "@/context/AuthContext";
import { Button, Card, Input, Select, Textarea } from "@/components/ui";
import MessageBubble from "@/components/chat/MessageBubble";
import PlotlyFigure from "@/components/chat/PlotlyFigure";
import { cn } from "@/lib/utils";

interface LiveTool {
  name: string;
  icon: string;
  label: string;
  keyword: string;
  ok?: boolean;
}

interface Expert {
  model: string;
  temperature: number;
  top_p: number;
  max_tool_calls: number;
}

export default function Chat() {
  const { hasRole } = useAuth();
  const isExpert = hasRole("dev");

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);

  // Live activity for the in-flight turn.
  const [status, setStatus] = useState("");
  const [tools, setTools] = useState<LiveTool[]>([]);
  const [liveFigures, setLiveFigures] = useState<any[]>([]);

  // Expert mode (dev/admin only).
  const [showExpert, setShowExpert] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [expert, setExpert] = useState<Expert>({
    model: "",
    temperature: 0.2,
    top_p: 0.9,
    max_tool_calls: 7,
  });

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    refreshConversations();
    if (isExpert) {
      api
        .get<{ models: string[] }>("/api/dev/models")
        .then((r) => setModels(r.models))
        .catch(() => {});
    }
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, status, tools, liveFigures]);

  async function refreshConversations() {
    setConversations(await api.get<Conversation[]>("/api/conversations"));
  }

  async function openConversation(id: number) {
    setActiveId(id);
    setMessages(await api.get<ChatMessage[]>(`/api/conversations/${id}/messages`));
  }

  function newConversation() {
    setActiveId(null);
    setMessages([]);
  }

  async function deleteConversation(id: number, e: React.MouseEvent) {
    e.stopPropagation();
    await api.del(`/api/conversations/${id}`);
    if (activeId === id) newConversation();
    refreshConversations();
  }

  function resetLive() {
    setStatus("");
    setTools([]);
    setLiveFigures([]);
  }

  async function send() {
    const text = input.trim();
    if (!text || streaming) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: text }]);
    setStreaming(true);
    resetLive();

    const collectedFigures: any[] = [];
    let sources: Partial<ChatMessage> = {};
    let currentId = activeId;

    const req = {
      message: text,
      conversation_id: activeId,
      ...(isExpert
        ? {
            model: expert.model || undefined,
            temperature: expert.temperature,
            top_p: expert.top_p,
            max_tool_calls: expert.max_tool_calls,
          }
        : {}),
    };

    await streamChat(req, (ev: AgentEvent) => {
      switch (ev.type) {
        case "conversation":
          currentId = ev.id;
          setActiveId(ev.id);
          if (ev.new) refreshConversations();
          break;
        case "status":
          setStatus(ev.text);
          break;
        case "tool_call":
          setTools((t) => [
            ...t,
            { name: ev.name, icon: ev.icon, label: ev.label, keyword: ev.keyword },
          ]);
          break;
        case "tool_result":
          setTools((t) => {
            const copy = [...t];
            for (let i = copy.length - 1; i >= 0; i--) {
              if (copy[i].name === ev.name && copy[i].ok === undefined) {
                copy[i] = { ...copy[i], ok: ev.ok };
                break;
              }
            }
            return copy;
          });
          break;
        case "figure":
          collectedFigures.push(ev.figure);
          setLiveFigures((f) => [...f, ev.figure]);
          break;
        case "sources":
          sources = {
            wikipedia_urls: ev.wikipedia,
            pubmed_urls: ev.pubmed,
            ncbi_urls: ev.ncbi,
          };
          break;
        case "final":
        case "error":
          setMessages((m) => [
            ...m,
            {
              role: "assistant",
              content: ev.type === "error" ? `⚠️ ${ev.message}` : ev.content,
              figures: collectedFigures,
              ...sources,
            },
          ]);
          break;
        case "saved":
          break;
      }
    });

    setStreaming(false);
    resetLive();
    if (currentId) refreshConversations();
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="flex h-full">
      {/* Conversation list */}
      <div className="flex w-64 shrink-0 flex-col border-r border-border bg-card">
        <div className="p-3">
          <Button className="w-full" onClick={newConversation}>
            <Plus className="h-4 w-4" /> New chat
          </Button>
        </div>
        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto px-2 pb-2">
          {conversations.map((c) => (
            <div
              key={c.id}
              onClick={() => openConversation(c.id)}
              className={cn(
                "group flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 text-sm",
                activeId === c.id ? "bg-accent text-accent-foreground" : "hover:bg-accent/50",
              )}
            >
              <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground" />
              <span className="min-w-0 flex-1 truncate">{c.title || "Untitled"}</span>
              <button
                onClick={(e) => deleteConversation(c.id, e)}
                className="opacity-0 group-hover:opacity-100"
                title="Delete"
              >
                <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
              </button>
            </div>
          ))}
          {conversations.length === 0 && (
            <p className="px-2 py-4 text-center text-sm text-muted-foreground">No conversations yet</p>
          )}
        </div>
      </div>

      {/* Chat area */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div ref={scrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto p-6">
          {messages.length === 0 && !streaming && (
            <div className="flex h-full flex-col items-center justify-center text-center text-muted-foreground">
              <div className="text-5xl">🦠</div>
              <h2 className="mt-4 text-xl font-medium text-foreground">Ask about viruses</h2>
              <p className="mt-1 max-w-md text-sm">
                Query viral taxonomy, hosts, and the literature. Answers are grounded in tools and
                cite their sources.
              </p>
            </div>
          )}
          {messages.map((m, i) => (
            <MessageBubble key={i} msg={m} />
          ))}

          {/* Live activity while a turn streams */}
          {streaming && (
            <div className="flex justify-start">
              <Card className="max-w-[85%] space-y-2 p-4">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {status || "Working"}…
                </div>
                {tools.map((t, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <span>{t.icon}</span>
                    <span>{t.label}</span>
                    {t.keyword && <span className="text-muted-foreground">— {t.keyword}</span>}
                    <span className="ml-auto">
                      {t.ok === undefined ? "⏳" : t.ok ? "✅" : "❌"}
                    </span>
                  </div>
                ))}
                {liveFigures.map((f, i) => (
                  <PlotlyFigure key={i} figure={f} />
                ))}
              </Card>
            </div>
          )}
        </div>

        {/* Expert panel */}
        {isExpert && showExpert && (
          <div className="border-t border-border bg-muted/30 px-6 py-3">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <div>
                <label className="text-xs text-muted-foreground">Model</label>
                <Select
                  value={expert.model}
                  onChange={(e) => setExpert({ ...expert, model: e.target.value })}
                >
                  <option value="">default</option>
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </Select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground">
                  Temperature {expert.temperature}
                </label>
                <Input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={expert.temperature}
                  onChange={(e) => setExpert({ ...expert, temperature: +e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">top_p {expert.top_p}</label>
                <Input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={expert.top_p}
                  onChange={(e) => setExpert({ ...expert, top_p: +e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">
                  Max tool calls {expert.max_tool_calls}
                </label>
                <Input
                  type="range"
                  min={1}
                  max={15}
                  step={1}
                  value={expert.max_tool_calls}
                  onChange={(e) => setExpert({ ...expert, max_tool_calls: +e.target.value })}
                />
              </div>
            </div>
          </div>
        )}

        {/* Composer */}
        <div className="border-t border-border p-4">
          <div className="flex items-end gap-2">
            {isExpert && (
              <Button
                variant={showExpert ? "primary" : "outline"}
                size="icon"
                onClick={() => setShowExpert((s) => !s)}
                title="Expert settings"
              >
                <Sliders className="h-4 w-4" />
              </Button>
            )}
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask about a virus, host, or dataset…"
              rows={1}
              className="max-h-40 min-h-[2.5rem] flex-1 resize-none"
              disabled={streaming}
            />
            <Button size="icon" onClick={send} disabled={streaming || !input.trim()}>
              {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
