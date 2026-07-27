"""
backend/agent.py — the agentic loop, extracted from app.py's albert_agent_loop
and refactored into an async generator of structured events.

Where the Streamlit version pushed UI updates through a `status_container` and
returned a tuple at the end, this version `yield`s events as they happen:

    {"type": "status",      "text": str}
    {"type": "tool_call",   "name": str, "keyword": str, "icon": str, "label": str}
    {"type": "tool_result", "name": str, "ok": bool}
    {"type": "figure",      "figure": <plotly json dict>}
    {"type": "sources",     "wikipedia": [...], "pubmed": [...], "ncbi": [...]}
    {"type": "final",       "content": str}
    {"type": "error",       "message": str}
    {"type": "done",        "history": [...], "figures": [...], "wikipedia": [...],
                            "pubmed": [...], "ncbi": [...], "executed_codes": [...]}

The terminal "done" event carries everything the caller needs to persist the
turn (via db.add_message) — figures are plotly JSON dicts there. The loop never
touches the database or Streamlit.
"""

import json

import plotly.io as pio
import requests
from fastmcp import Client

from config import (
    MCP_SERVER_URL,
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_PRESENCE_PENALTY,
    DEFAULT_FREQUENCY_PENALTY, DEFAULT_SEED, DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_PARALLEL_TOOL_CALLS, DEFAULT_MAX_TOOL_CALLS, DEFAULT_MAX_TOOL_CONTENT,
    DEFAULT_PREVIEW_ROWS, DEFAULT_WIKIPEDIA_LIMIT,
)
from prompt import build_system_prompt
from backend.albert import (
    albert_chat, AlbertRateLimitError,
    mcp_tools_to_openai_spec, unwrap_mcp_result, describe_available_datasets,
    parse_tool_arguments, clean_history_messages,
    strip_hallucinated_pmids, strip_fake_citation_markers,
    snippet, ui_search_keyword, TOOL_LABELS, logger,
)


async def check_mcp_connection() -> bool:
    """True if the MCP server answers a list_tools() call."""
    try:
        async with Client(MCP_SERVER_URL) as client:
            await client.list_tools()
            return True
    except Exception as e:
        logger.error(f"MCP_CONNECTION_FAIL | {e}")
        return False


async def run_agent(
    model: str,
    api_key: str,
    user_query: str,
    username: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
    wikipedia_limit: int = DEFAULT_WIKIPEDIA_LIMIT,
    max_tool_content: int = DEFAULT_MAX_TOOL_CONTENT,
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    seed: int = DEFAULT_SEED,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    parallel_tool_calls: bool = DEFAULT_PARALLEL_TOOL_CALLS,
    history_messages: list | None = None,
):
    """Async generator running the tool-calling loop against ALBERT + MCP,
    yielding events (see module docstring). Logic mirrors app.py's
    albert_agent_loop; only the I/O surface changed (events instead of a
    Streamlit container + a final tuple)."""

    used_wikipedia_urls: list = []
    used_pubmed_urls: list = []
    used_ncbi_urls: list = []
    executed_codes: list = []
    generated_figures: list = []   # Plotly Figure objects (for db.add_message)
    real_pmids: set = set()
    tool_call_count = 0

    def _done_event(history):
        # Figures serialized to plotly JSON dicts for the caller / SSE.
        return {
            "type": "done",
            "history": history,
            "figures": [json.loads(f.to_json()) for f in generated_figures],
            "wikipedia": used_wikipedia_urls,
            "pubmed": used_pubmed_urls,
            "ncbi": used_ncbi_urls,
            "executed_codes": executed_codes,
        }

    logger.info("=" * 50)
    logger.info(f"USER_QUERY | user={username or 'unknown'} | {user_query}")
    logger.info(
        f"CONFIG | model={model} temp={temperature} top_p={top_p} "
        f"max_calls={max_tool_calls} preview={preview_rows}rows "
        f"max_tool_content={max_tool_content}"
    )

    yield {"type": "status", "text": "Thinking"}

    async with Client(MCP_SERVER_URL) as mcp:
        tools = await mcp.list_tools()
        tools_spec = mcp_tools_to_openai_spec(tools)
        tool_schemas = {t.name: (t.inputSchema or {}) for t in tools}

        datasets_description = await describe_available_datasets(mcp)
        system_prompt = build_system_prompt(datasets_description)

        messages = [{"role": "system", "content": system_prompt}]
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": user_query})

        while True:
            logger.info(
                f"LLM_CALL | sending {len(messages)} messages | "
                f"roles={[m['role'] for m in messages]}"
            )

            try:
                resp = albert_chat(
                    messages=messages, tools=tools_spec, model=model, api_key=api_key,
                    temperature=temperature, top_p=top_p,
                    presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                    seed=seed, max_completion_tokens=max_completion_tokens,
                    parallel_tool_calls=parallel_tool_calls,
                )
            except requests.exceptions.Timeout:
                logger.error("ALBERT_TIMEOUT")
                yield {"type": "error",
                       "message": "The model took too long to respond (>120 s). Please try again."}
                yield _done_event(None)
                return
            except AlbertRateLimitError:
                logger.error("ALBERT_RATE_LIMITED")
                yield {"type": "error",
                       "message": ("⏳ Albert is rate-limiting requests right now — the free API is "
                                   "saturated, especially on the large models. Please wait a moment "
                                   "and try again, or pick a lighter model in Expert mode.")}
                yield _done_event(None)
                return
            except Exception as e:
                logger.error(f"ALBERT_ERROR | {e}")
                yield {"type": "error", "message": f"Could not reach Albert API: {e}"}
                yield _done_event(None)
                return

            choice = resp["choices"][0]
            msg = choice["message"]
            finish = choice.get("finish_reason", "")

            logger.info(
                f"LLM_RESPONSE | finish_reason={finish!r} | "
                f"has_tool_calls={bool(msg.get('tool_calls'))} | "
                f"content_len={len(msg.get('content') or '')}"
            )

            # ── CASE 1: Final answer (no tool calls) ─────────────────────────
            if finish != "tool_calls" or not msg.get("tool_calls"):
                final_text = (msg.get("content") or "").strip()

                if not final_text and tool_call_count > 0 and tool_call_count < min(3, max_tool_calls):
                    logger.warning(
                        f"EARLY_STOP | finish=stop, content empty, only {tool_call_count} tool calls — "
                        f"injecting continuation prompt"
                    )
                    messages.append(msg)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have only searched {tool_call_count} time(s) so far and haven't "
                            f"found enough information yet. Please continue your research: "
                            f"search Wikipedia and PubMed for viruses infecting the requested host, "
                            f"then provide a complete scientific answer."
                        ),
                    })
                    continue

                if not final_text and tool_call_count > 0:
                    logger.warning(
                        "EMPTY_FINAL_ANSWER | content blank after tool calls — "
                        "rebuilding clean context for synthesis (gpt-oss-120b workaround)"
                    )
                    yield {"type": "status", "text": "Synthesizing answer"}

                    tool_results_text = []
                    for m in messages:
                        if m.get("role") == "tool":
                            tool_name = m.get("name", "tool")
                            tool_results_text.append(
                                f"=== Result from {tool_name} ===\n{m.get('content', '')}"
                            )
                    context_block = "\n\n".join(tool_results_text)

                    error_count = sum(
                        1 for r in tool_results_text
                        if r.startswith("=== Result") and "Error:" in r
                    )
                    success_count = len(tool_results_text) - error_count

                    if success_count == 0 and error_count > 0:
                        logger.warning("SYNTHESIS_FALLBACK | all tool results are errors")
                        context_note = (
                            "Note: the dataset did not contain data for this query "
                            "(search returned no results). Answer from scientific knowledge only."
                        )
                    else:
                        context_note = f"Here is all the information gathered:\n\n{context_block}"

                    clean_messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Original question: {user_query}\n\n"
                                f"{context_note}\n\n"
                                f"Write a detailed, well-structured scientific answer. "
                                f"Requirements:\n"
                                f"- Cover each relevant virus with its taxonomy (family, genus, species), "
                                f"transmission route, pathogenesis, and key clinical signs\n"
                                f"- Use proper scientific nomenclature\n"
                                f"- Structure your answer with one section per virus\n"
                                f"- Be thorough and complete — do not summarize\n"
                                f"- Do not mention tools, datasets, or data retrieval\n"
                                f"- CRITICAL: do NOT invent or mention any PMID. "
                                f"Only reference PMIDs that appear verbatim in the context above. "
                                f"If no PMIDs are in the context, write none."
                            ),
                        },
                    ]

                    try:
                        retry_resp = albert_chat(
                            messages=clean_messages, tools=[], model=model, api_key=api_key,
                            temperature=temperature, top_p=top_p, max_completion_tokens=6144,
                        )
                        final_text = (
                            retry_resp["choices"][0]["message"].get("content") or ""
                        ).strip()

                        final_text, stripped = strip_hallucinated_pmids(final_text, real_pmids)
                        if stripped:
                            logger.warning(f"PMID_HALLUCINATION_SYNTHESIS | stripped {len(stripped)}: {stripped}")
                        final_text, n_fake = strip_fake_citation_markers(final_text)
                        if n_fake:
                            logger.warning(f"FAKE_CITATION_SYNTHESIS | stripped {n_fake} marker(s)")

                        logger.info(f"SYNTHESIS_OK | content_len={len(final_text)}")
                        if not final_text:
                            logger.error("SYNTHESIS_FAIL | still empty after clean context")
                            final_text = (
                                "⚠️ The model retrieved information but failed to generate "
                                "a final answer. Please try rephrasing your question."
                            )
                    except Exception as e:
                        logger.error(f"SYNTHESIS_RETRY_FAIL | {e}")
                        final_text = "The model did not produce a final answer. Please rephrase your question."

                final_text, stripped = strip_hallucinated_pmids(final_text, real_pmids)
                if stripped:
                    logger.warning(f"PMID_HALLUCINATION | stripped {len(stripped)}: {stripped}")
                final_text, n_fake = strip_fake_citation_markers(final_text)
                if n_fake:
                    logger.warning(f"FAKE_CITATION | stripped {n_fake} marker(s)")

                logger.info(
                    f"RESULT | len={len(final_text)} | "
                    f"{final_text[:500]}{'...' if len(final_text) > 500 else ''}"
                )
                messages.append(msg)

                if used_wikipedia_urls or used_pubmed_urls or used_ncbi_urls:
                    yield {"type": "sources",
                           "wikipedia": used_wikipedia_urls,
                           "pubmed": used_pubmed_urls,
                           "ncbi": used_ncbi_urls}
                yield {"type": "final", "content": final_text}
                yield _done_event(clean_history_messages(messages[1:]))
                return

            # ── CASE 2: Tool calls ───────────────────────────────────────────
            messages.append(msg)

            if tool_call_count >= max_tool_calls:
                logger.warning(f"MAX_TOOL_CALLS | Limit reached ({max_tool_calls})")
                messages.append({
                    "role": "system",
                    "content": (
                        f"Tool call limit reached ({max_tool_calls}). "
                        f"Synthesize a final answer to: '{user_query}'"
                    ),
                })
                continue

            for call in msg["tool_calls"]:
                tool_call_count += 1
                name = call["function"]["name"]
                raw_args = call["function"].get("arguments", {})
                args = parse_tool_arguments(raw_args)

                logger.info(
                    f"TOOL_CALL #{tool_call_count}/{max_tool_calls} | {name} | "
                    f"call_id={call.get('id','?')} | "
                    f"parsed_args={json.dumps(args, ensure_ascii=False)[:400]}"
                )

                icon, label = TOOL_LABELS.get(name, ("🔧", name))
                keyword = ui_search_keyword(args)
                yield {"type": "tool_call", "name": name,
                       "icon": icon, "label": label, "keyword": keyword}

                schema = tool_schemas.get(name) or {}
                props = schema.get("properties", {})
                required = schema.get("required", [])

                call_args = dict(args)
                if set(call_args.keys()) == {"_raw"} and len(required) == 1:
                    call_args[required[0]] = call_args.pop("_raw")

                if "preview_rows" in props:
                    call_args.setdefault("preview_rows", preview_rows)
                if "wikipedia_limit" in props:
                    call_args.setdefault("wikipedia_limit", wikipedia_limit)

                if props:
                    dropped = set(call_args) - set(props)
                    if dropped:
                        logger.warning(f"TOOL_ARG_DROPPED | {name} | unsupported keys: {sorted(dropped)}")
                        call_args = {k: v for k, v in call_args.items() if k in props}

                output = unwrap_mcp_result(await mcp.call_tool(name, call_args))
                content = output.get("content", "Unknown error")

                if output.get("success"):
                    for artifact in output.get("artifacts", []):
                        a_type = artifact.get("type")
                        if a_type == "url":
                            if artifact["url"] not in used_wikipedia_urls:
                                used_wikipedia_urls.append(artifact["url"])
                        elif a_type == "pubmed":
                            for pmid in artifact.get("pmids", []):
                                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                                if pubmed_url not in used_pubmed_urls:
                                    used_pubmed_urls.append(pubmed_url)
                                real_pmids.add(str(pmid))
                        elif a_type == "ncbi_taxonomy":
                            if artifact["url"] not in used_ncbi_urls:
                                used_ncbi_urls.append(artifact["url"])
                        elif a_type == "plotly":
                            fig = pio.from_json(json.dumps(artifact["figure"]))
                            generated_figures.append(fig)
                            yield {"type": "figure", "figure": artifact["figure"]}

                    if "code" in call_args:
                        executed_codes.append(call_args["code"])
                    elif "sql" in call_args:
                        executed_codes.append(f"-- SQL ({name})\n{call_args['sql']}")

                    logger.info(f"TOOL_OK | {name} | {snippet(content, 500)}")
                else:
                    logger.warning(f"TOOL_FAIL | {name} | {content}")

                yield {"type": "tool_result", "name": name, "ok": bool(output.get("success"))}

                if len(content) > max_tool_content:
                    original_len = len(content)
                    content = (
                        content[:max_tool_content]
                        + f"\n\n[...truncated — {original_len - max_tool_content} chars omitted]"
                    )
                    logger.warning(
                        f"TOOL_CONTENT_TRUNCATED | {name} | "
                        f"trimmed from {original_len} to {max_tool_content} chars"
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": content,
                })

            logger.info(
                f"LOOP_STATE | messages_in_history={len(messages)} | "
                f"tool_calls_used={tool_call_count}/{max_tool_calls} | "
                f"figures={len(generated_figures)} | codes={len(executed_codes)}"
            )
