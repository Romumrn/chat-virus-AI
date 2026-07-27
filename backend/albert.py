"""
backend/albert.py — ALBERT + MCP helper functions, extracted verbatim (logic
unchanged) from the Streamlit app.py. None of these touch Streamlit; they are
the reusable core the FastAPI agent loop (backend.agent) builds on.

Contains:
  - HTTP plumbing for the ALBERT (Etalab) OpenAI-compatible API: headers, model
    listing, Whisper transcription, and the rate-limit-aware chat call.
  - MCP glue: convert MCP tool specs to OpenAI format, unwrap tool results,
    read dataset-description resources.
  - Output guards: strip hallucinated PMIDs and fake citation markers.
  - Tool-argument parsing + history cleaning.
"""

import json
import re
import time

import requests

from config import (
    ALBERT_BASE_URL, ALBERT_TIMEOUT, ALBERT_MODEL_DEFAULT, ALBERT_WHISPER_MODEL,
    ALBERT_MAX_RETRIES, ALBERT_RETRY_BACKOFF_CAP, LOG_DIR,
)
from logging_utils import setup_logger

logger = setup_logger(LOG_DIR)


# ==================== ALBERT API HELPERS ====================

class AlbertRateLimitError(RuntimeError):
    """Raised when ALBERT keeps returning HTTP 429 after all retries are
    exhausted — the (free) API is rate-limiting/saturating requests."""


def _albert_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def list_albert_models(api_key: str) -> list:
    """Available text-generation models, filtering out embedding/audio/rerank.
    Falls back to ALBERT_MODEL_DEFAULT if the API call fails."""
    try:
        r = requests.get(
            f"{ALBERT_BASE_URL}/models",
            headers=_albert_headers(api_key),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        excluded_keywords = ("embed", "whisper", "rerank")
        names = [
            m["id"] for m in data
            if m.get("object") == "model"
            and not any(kw in m["id"].lower() for kw in excluded_keywords)
        ]
        return names if names else [ALBERT_MODEL_DEFAULT]
    except Exception as e:
        logger.warning(f"MODEL_LIST_FAIL | {e} — using default model")
        return [ALBERT_MODEL_DEFAULT]


def transcribe_audio(audio_bytes: bytes, api_key: str) -> str:
    """Transcribe recorded audio to text via ALBERT's Whisper endpoint.
    Returns '' on failure so the caller can fall back to typed input."""
    try:
        r = requests.post(
            f"{ALBERT_BASE_URL}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("recording.wav", audio_bytes, "audio/wav")},
            data={"model": ALBERT_WHISPER_MODEL},
            timeout=ALBERT_TIMEOUT,
        )
        r.raise_for_status()
        text = (r.json().get("text") or "").strip()
        if not text:
            logger.warning("WHISPER_EMPTY | transcription returned empty text")
        return text
    except Exception as e:
        logger.error(f"WHISPER_FAIL | {e}")
        return ""


def albert_chat(
    messages: list,
    tools: list,
    model: str,
    api_key: str,
    temperature,
    top_p,
    presence_penalty=0,
    frequency_penalty=0,
    seed=42,
    max_completion_tokens=4096,
    parallel_tool_calls=False,
    retry: int = ALBERT_MAX_RETRIES,
) -> dict:
    """Chat completion against ALBERT with retry logic. Retries on HTTP 429
    honoring Retry-After when present, else exponential backoff capped at
    ALBERT_RETRY_BACKOFF_CAP. Raises AlbertRateLimitError if all attempts are
    rate-limited."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "seed": seed,
        "max_completion_tokens": max_completion_tokens,
        "parallel_tool_calls": parallel_tool_calls,
        "stream": False,
    }

    for attempt in range(1, retry + 1):
        try:
            r = requests.post(
                f"{ALBERT_BASE_URL}/chat/completions",
                headers=_albert_headers(api_key),
                json=payload,
                timeout=ALBERT_TIMEOUT,
            )
            if r.status_code == 429:
                retry_after = (r.headers.get("Retry-After") or "").strip()
                if retry_after.isdigit():
                    wait = min(int(retry_after), ALBERT_RETRY_BACKOFF_CAP)
                else:
                    wait = min(2 ** attempt, ALBERT_RETRY_BACKOFF_CAP)
                logger.warning(f"RATE_LIMIT | attempt {attempt}/{retry} — waiting {wait}s")
                if attempt < retry:
                    time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            logger.error(f"ALBERT_TIMEOUT | attempt {attempt}")
            if attempt == retry:
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"ALBERT_HTTP_ERROR | {e}")
            raise

    raise AlbertRateLimitError(
        f"Albert rate-limited the request after {retry} attempts (HTTP 429)."
    )


# ==================== MCP HELPERS ====================

def mcp_tools_to_openai_spec(tools) -> list[dict]:
    """Convert fastmcp Tool objects to the OpenAI `tools=[...]` format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def unwrap_mcp_result(result) -> dict:
    """Normalize a fastmcp CallToolResult into the plain dict our tools return."""
    if getattr(result, "data", None) is not None:
        return result.data
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content
    for block in getattr(result, "content", []) or []:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except json.JSONDecodeError:
                return {"success": False, "content": block.text, "artifacts": []}
    return {"success": False, "content": "Empty MCP tool response", "artifacts": []}


async def describe_available_datasets(mcp) -> str:
    """Render every MCP resource as a text block for the system prompt. The MCP
    server is the sole owner of dataset knowledge — no resource is assumed."""
    try:
        resources = await mcp.list_resources()
    except Exception as e:
        logger.warning(f"MCP_RESOURCES_LIST_FAIL | {e}")
        return ""

    blocks = []
    for r in resources:
        try:
            contents = await mcp.read_resource(r.uri)
            text = next((getattr(c, "text", None) for c in contents if getattr(c, "text", None)), None)
            if text:
                blocks.append(f"### {r.name or r.uri}\n{text}")
        except Exception as e:
            logger.warning(f"MCP_RESOURCE_READ_FAIL | {r.uri} | {e}")
    return "\n\n".join(blocks)


# ==================== TOOL ARGS & HISTORY ====================

def parse_tool_arguments(raw_args) -> dict:
    """Parse tool-call arguments from ALBERT, tolerating dict / JSON string /
    malformed partial JSON (a known gpt-oss-120b bug). Falls back to
    {"_raw": raw_args}."""
    if isinstance(raw_args, dict):
        return raw_args

    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            pass

        recovered = {}
        for m in re.finditer(
            r'"(\w+)"\s*:\s*("(?:[^"\\]|\\.)*"|\d+(?:\.\d+)?|true|false|null)',
            raw_args,
        ):
            try:
                recovered[m.group(1)] = json.loads(m.group(2))
            except Exception:
                recovered[m.group(1)] = m.group(2)

        if recovered:
            logger.warning(f"TOOL_ARG_PARTIAL_PARSE | recovered={recovered}")
            return recovered

        logger.error(f"TOOL_ARG_PARSE_FAIL | raw={raw_args[:300]}")
        return {"_raw": raw_args}

    return {}


def clean_history_messages(messages: list) -> list:
    """Strip tool-call bookkeeping (assistant tool_calls + tool results, and
    injected system reminders) — only user questions and final text answers
    carry over to the next turn."""
    return [
        m for m in messages
        if m["role"] == "user"
        or (m["role"] == "assistant" and not m.get("tool_calls"))
    ]


def build_context_window(messages: list, max_turns: int) -> list:
    """From a conversation's stored history ({role, content, ...}), build the
    bounded message list replayed to ALBERT: strip tool bookkeeping then keep
    only the last `max_turns` user/assistant exchanges."""
    cleaned = clean_history_messages(
        [{"role": m["role"], "content": m.get("content", "")} for m in messages]
    )
    if max_turns > 0:
        cleaned = cleaned[-(max_turns * 2):]
    return cleaned


# ==================== OUTPUT GUARDS ====================

def strip_hallucinated_pmids(text: str, real_pmids: set) -> tuple[str, list]:
    """Remove PMID references not returned by an actual pubmed_search call."""
    pattern = re.compile(r'\bPMID[:\s#]*([0-9]{5,9})\b', re.IGNORECASE)
    removed = []

    def _replace(match):
        pmid = match.group(1)
        if pmid in real_pmids:
            return match.group(0)
        removed.append(pmid)
        return ""

    cleaned = pattern.sub(_replace, text)
    cleaned = re.sub(r'\(e\.g\.\s*,?\s*on\s+[^)]{0,80}\)', '', cleaned)
    cleaned = re.sub(r'\(see\s*\)', '', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip(), removed


_FAKE_CITATION_PATTERN = re.compile(r'【[^【】]*】')


def strip_fake_citation_markers(text: str) -> tuple[str, int]:
    """Strip bracket-style citation markers like 【4†L13-L17】 that gpt-oss-120b
    sometimes emits — they never point to a real, resolvable source here."""
    cleaned, n = _FAKE_CITATION_PATTERN.subn('', text)
    if n:
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()
    return cleaned, n


def snippet(text: str, max_len: int = 120) -> str:
    """Collapse arbitrary tool output to a single-line preview (used in logs)."""
    s = " ".join(text.strip().split())
    return s[:max_len] + ("…" if len(s) > max_len else "")


def ui_search_keyword(call_args: dict) -> str:
    """Pick out the search keyword (search_term / query / name) for the UI
    status line, generically — no tool-name special-casing."""
    for key in ("search_term", "query", "name"):
        val = call_args.get(key)
        if isinstance(val, str) and val.strip():
            snip = " ".join(val.strip().split())
            return snip[:80] + ("…" if len(snip) > 80 else "")
    return ""


# UI labels for tool status display — kept in sync with server_mcp.py tools.
TOOL_LABELS = {
    "wikipedia_search":     ("📖", "Wikipedia search"),
    "pubmed_search":        ("🔬", "PubMed search"),
    "ncbi_taxonomy_search": ("🧬", "NCBI Taxonomy lookup"),
    "query_host_sql":       ("🪣", "Querying S3 host table"),
    "query_dataframe":      ("🔬", "Dataset query"),
    "create_visualization": ("📊", "Creating chart"),
    "create_map":           ("🗺️",  "Creating map"),
}
