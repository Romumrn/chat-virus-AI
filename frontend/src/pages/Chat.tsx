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
  PanelLeftClose,
  PanelLeftOpen,
  Mic,
  Square,
} from "lucide-react";
import { api, type Conversation, type ChatMessage } from "@/lib/api";
import { streamChat, type AgentEvent } from "@/lib/sse";
import { useAuth } from "@/context/AuthContext";
import { Button, Card, Input, Select, Textarea } from "@/components/ui";
import MessageBubble from "@/components/chat/MessageBubble";
import PlotlyFigure from "@/components/chat/PlotlyFigure";
import Welcome from "@/components/chat/Welcome";
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
  presence_penalty: number;
  frequency_penalty: number;
  seed: number;
  max_completion_tokens: number;
  parallel_tool_calls: boolean;
  max_tool_calls: number;
  max_context_turns: number;
  preview_rows: number;
  wikipedia_limit: number;
  max_tool_content: number;
}

export default function Chat() {
  const { user, hasRole } = useAuth();
  const isExpert = hasRole("dev");

  // Left column: collapsible, with two tabs (conversation history / info).
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);

  // Voice input. Preferred path: the browser's Web Speech API, which streams
  // the transcription word-by-word into the composer *while* you speak. Fallback
  // (Firefox etc.): record a clip and transcribe it via the backend on stop.
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null);
  // Composer text present before recording started — live transcript is appended
  // to this so we never clobber what the user had already typed.
  const baseInputRef = useRef("");

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
    presence_penalty: -0.2,
    frequency_penalty: 0.2,
    seed: 42,
    max_completion_tokens: 4096,
    parallel_tool_calls: false,
    max_tool_calls: 7,
    max_context_turns: 5,
    preview_rows: 50,
    wikipedia_limit: 4000,
    max_tool_content: 6000,
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

  async function send(override?: string) {
    const text = (override ?? input).trim();
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
            presence_penalty: expert.presence_penalty,
            frequency_penalty: expert.frequency_penalty,
            seed: expert.seed,
            max_completion_tokens: expert.max_completion_tokens,
            parallel_tool_calls: expert.parallel_tool_calls,
            max_tool_calls: expert.max_tool_calls,
            max_context_turns: expert.max_context_turns,
            preview_rows: expert.preview_rows,
            wikipedia_limit: expert.wikipedia_limit,
            max_tool_content: expert.max_tool_content,
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
            executed_codes: ev.executed_codes,
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

  function startRecording() {
    if (recording || streaming || transcribing) return;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const SpeechRecognitionCtor: any =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognitionCtor) {
      startLiveRecognition(SpeechRecognitionCtor);
    } else {
      startBatchRecording();
    }
  }

  // Live, word-by-word transcription in the browser (Chrome/Edge/Safari).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function startLiveRecognition(Ctor: any) {
    const recognition = new Ctor();
    recognition.lang = navigator.language || "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;

    // Keep what was already typed; append the transcript after it.
    baseInputRef.current = input ? input.replace(/\s+$/, "") + " " : "";

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onresult = (event: any) => {
      let finalText = "";
      let interim = "";
      for (let i = 0; i < event.results.length; i++) {
        const res = event.results[i];
        if (res.isFinal) finalText += res[0].transcript;
        else interim += res[0].transcript;
      }
      setInput(baseInputRef.current + (finalText + interim).replace(/^\s+/, ""));
    };
    recognition.onerror = () => {
      // no-speech / aborted / not-allowed — just end the session gracefully.
      setRecording(false);
      recognitionRef.current = null;
    };
    recognition.onend = () => {
      setRecording(false);
      recognitionRef.current = null;
      textareaRef.current?.focus();
    };

    try {
      recognition.start();
      recognitionRef.current = recognition;
      setRecording(true);
    } catch {
      // start() throws if already running or mic unavailable — fall back.
      startBatchRecording();
    }
  }

  // Fallback: record a clip, transcribe via the backend (Whisper) once stopped.
  async function startBatchRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        setTranscribing(true);
        try {
          const { text } = await api.postFile<{ text: string }>(
            "/api/transcribe",
            blob,
            "recording.webm",
          );
          // Drop the transcription into the composer (like Claude Code) so the
          // user can review/edit it before sending, instead of auto-sending.
          if (text) {
            setInput((prev) => (prev ? `${prev.replace(/\s+$/, "")} ${text}` : text));
            textareaRef.current?.focus();
          }
        } catch {
          // transcription failed server-side — let the user type instead
        } finally {
          setTranscribing(false);
        }
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch {
      // mic permission denied or unavailable — silently no-op, user can type
    }
  }

  function stopRecording() {
    if (recognitionRef.current) {
      // Live path: stop() fires onend, which flips `recording` off.
      recognitionRef.current.stop();
      return;
    }
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }

  return (
    <div className="flex h-full">
      {/* Left column — collapsible list of chats */}
      {sidebarOpen ? (
        <div className="flex w-64 shrink-0 flex-col border-r border-border bg-card">
          <div className="flex items-center gap-1 border-b border-border p-2">
            <div className="flex flex-1 items-center gap-1.5 px-2 py-1.5 text-sm font-medium text-muted-foreground">
              <MessageSquare className="h-4 w-4" /> Chats
            </div>
            <Button variant="ghost" size="icon" onClick={() => setSidebarOpen(false)} title="Collapse">
              <PanelLeftClose className="h-4 w-4" />
            </Button>
          </div>

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
      ) : (
        // Collapsed rail: just the expand toggle + a quick new-chat button.
        <div className="flex w-12 shrink-0 flex-col items-center gap-2 border-r border-border bg-card py-2">
          <Button variant="ghost" size="icon" onClick={() => setSidebarOpen(true)} title="Expand sidebar">
            <PanelLeftOpen className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" onClick={newConversation} title="New chat">
            <Plus className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Chat area */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div ref={scrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto p-6">
          {messages.length === 0 && !streaming && (
            <Welcome name={user?.first_name || ""} onExample={(q) => send(q)} />
          )}
          {messages.map((m, i) => (
            <MessageBubble
              key={i}
              msg={m}
              question={
                m.role === "assistant" && messages[i - 1]?.role === "user"
                  ? messages[i - 1].content
                  : m.role === "assistant"
                    ? ""
                    : undefined
              }
            />
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
                  max={2}
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
                  max={20}
                  step={1}
                  value={expert.max_tool_calls}
                  onChange={(e) => setExpert({ ...expert, max_tool_calls: +e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">
                  Presence penalty {expert.presence_penalty}
                </label>
                <Input
                  type="range"
                  min={-2}
                  max={2}
                  step={0.1}
                  value={expert.presence_penalty}
                  onChange={(e) => setExpert({ ...expert, presence_penalty: +e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">
                  Frequency penalty {expert.frequency_penalty}
                </label>
                <Input
                  type="range"
                  min={-2}
                  max={2}
                  step={0.1}
                  value={expert.frequency_penalty}
                  onChange={(e) => setExpert({ ...expert, frequency_penalty: +e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">Seed</label>
                <Input
                  type="number"
                  step={1}
                  value={expert.seed}
                  onChange={(e) => setExpert({ ...expert, seed: +e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">
                  Max completion tokens {expert.max_completion_tokens}
                </label>
                <Input
                  type="range"
                  min={512}
                  max={32768}
                  step={512}
                  value={expert.max_completion_tokens}
                  onChange={(e) =>
                    setExpert({ ...expert, max_completion_tokens: +e.target.value })
                  }
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">
                  Max context turns {expert.max_context_turns}
                </label>
                <Input
                  type="range"
                  min={1}
                  max={20}
                  step={1}
                  value={expert.max_context_turns}
                  onChange={(e) => setExpert({ ...expert, max_context_turns: +e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">
                  Preview rows {expert.preview_rows}
                </label>
                <Input
                  type="range"
                  min={5}
                  max={200}
                  step={5}
                  value={expert.preview_rows}
                  onChange={(e) => setExpert({ ...expert, preview_rows: +e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">
                  Wiki limit (chars) {expert.wikipedia_limit}
                </label>
                <Input
                  type="range"
                  min={500}
                  max={30000}
                  step={500}
                  value={expert.wikipedia_limit}
                  onChange={(e) => setExpert({ ...expert, wikipedia_limit: +e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground">
                  Max tool content (chars) {expert.max_tool_content}
                </label>
                <Input
                  type="range"
                  min={2000}
                  max={30000}
                  step={1000}
                  value={expert.max_tool_content}
                  onChange={(e) => setExpert({ ...expert, max_tool_content: +e.target.value })}
                />
              </div>
              <label className="flex items-center gap-2 text-xs text-muted-foreground">
                <input
                  type="checkbox"
                  checked={expert.parallel_tool_calls}
                  onChange={(e) =>
                    setExpert({ ...expert, parallel_tool_calls: e.target.checked })
                  }
                />
                Parallel tool calls
              </label>
            </div>
          </div>
        )}

        {/* Composer */}
        <div className="border-t border-border p-4">
          <p className="mb-2 text-center text-xs text-muted-foreground">
            ⚠️ AI is not magic — Results may contain errors and should be verified for scientific or medical use.
          </p>
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
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder={recording ? "Recording… speak your question" : "Ask about a virus, host, or dataset…"}
              rows={1}
              className="max-h-40 min-h-[2.5rem] flex-1 resize-none"
              disabled={streaming || recording || transcribing}
            />
            <Button
              variant={recording ? "primary" : "outline"}
              size="icon"
              onClick={() => (recording ? stopRecording() : startRecording())}
              disabled={streaming || transcribing}
              title={recording ? "Stop recording" : "Ask by voice"}
              className={recording ? "animate-pulse" : undefined}
            >
              {transcribing ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : recording ? (
                <Square className="h-4 w-4" />
              ) : (
                <Mic className="h-4 w-4" />
              )}
            </Button>
            <Button size="icon" onClick={() => send()} disabled={streaming || !input.trim()}>
              {streaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
