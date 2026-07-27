/**
 * sse.ts — POST a chat request and consume the Server-Sent Events stream.
 *
 * EventSource can't send an Authorization header or a POST body, so we use
 * fetch() + a ReadableStream reader and parse `data:` frames ourselves.
 */
import { authHeaders } from "./api";

export interface AgentEvent {
  type:
    | "conversation"
    | "status"
    | "tool_call"
    | "tool_result"
    | "figure"
    | "sources"
    | "final"
    | "error"
    | "saved";
  [key: string]: any;
}

export interface ChatRequest {
  message: string;
  conversation_id?: number | null;
  model?: string;
  temperature?: number;
  top_p?: number;
  max_tool_calls?: number;
}

export async function streamChat(
  req: ChatRequest,
  onEvent: (ev: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(req),
    signal,
  });

  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* ignore */
    }
    onEvent({ type: "error", message: detail });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice(6)) as AgentEvent);
      } catch {
        /* skip malformed frame */
      }
    }
  }
}
