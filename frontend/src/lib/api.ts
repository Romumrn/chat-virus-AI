/**
 * api.ts — thin fetch wrapper around the FastAPI backend.
 *
 * The JWT is kept in localStorage and attached as a Bearer header. A 401 clears
 * it and bounces to /login (handled by AuthContext via the onUnauthorized hook).
 */

const TOKEN_KEY = "viromechat_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token);
}
export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

let onUnauthorized: (() => void) | null = null;
export function setUnauthorizedHandler(fn: () => void) {
  onUnauthorized = fn;
}

export function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const t = getToken();
  return { ...extra, ...(t ? { Authorization: `Bearer ${t}` } : {}) };
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (res.status === 401) {
    clearToken();
    onUnauthorized?.();
    throw new ApiError(401, "Not authenticated");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) =>
    fetch(path, { headers: authHeaders() }).then((r) => handle<T>(r)),

  post: <T>(path: string, body?: unknown) =>
    fetch(path, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: body === undefined ? undefined : JSON.stringify(body),
    }).then((r) => handle<T>(r)),

  put: <T>(path: string, body?: unknown) =>
    fetch(path, {
      method: "PUT",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: body === undefined ? undefined : JSON.stringify(body),
    }).then((r) => handle<T>(r)),

  patch: <T>(path: string, body?: unknown) =>
    fetch(path, {
      method: "PATCH",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: body === undefined ? undefined : JSON.stringify(body),
    }).then((r) => handle<T>(r)),

  del: <T>(path: string) =>
    fetch(path, { method: "DELETE", headers: authHeaders() }).then((r) => handle<T>(r)),

  postFile: <T>(path: string, file: Blob, filename: string) => {
    const form = new FormData();
    form.append("file", file, filename);
    return fetch(path, { method: "POST", headers: authHeaders(), body: form }).then((r) =>
      handle<T>(r),
    );
  },
};

// ── Types mirrored from backend/schemas.py ───────────────────────────────────
export type Role = "user" | "dev" | "admin";

export interface UserInfo {
  email: string;
  first_name?: string;
  last_name?: string;
  role: Role;
  created_at?: string;
  last_login?: string;
  n_conversations?: number;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  role: Role;
  email: string;
  first_name: string;
  last_name: string;
}

export interface Conversation {
  id: number;
  title?: string;
  created_at?: string;
  updated_at?: string;
}

export interface ChatMessage {
  role: string;
  content: string;
  figures?: any[];
  wikipedia_urls?: string[];
  pubmed_urls?: string[];
  ncbi_urls?: string[];
  executed_codes?: string[];
}
