#!/usr/bin/python3
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import re
import plotly.express as px
import plotly.graph_objects as go
import traceback


# ==================== CONSTANTS ==================== #
TAXO_DB_PATH = "data/viral_taxo.csv"
HOST_DB_PATH = "data/virushostdb.tsv"
OLLAMA_BASE_URL = "http://localhost:11434"
PAGE_TITLE = "Virus Dataset AI Agent 🦠"

# ==================== DEFAULT PARAMETERS ==================== #
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_REPEAT_PENALTY = 1.0
DEFAULT_SEED = 42
DEFAULT_MAX_TOOL_CALLS = 5
DEFAULT_PREVIEW_ROWS = 50
DEFAULT_WIKIPEDIA_LIMIT = 5000

# ==================== REAL TOOL IMPLEMENTATIONS ==================== #


def wikipedia_search(search_term: str, wikipedia_limit: int = DEFAULT_WIKIPEDIA_LIMIT) -> dict:
    LIMIT = wikipedia_limit
    term = re.sub(r"[^\w\s\-]", "", search_term.strip())
    title = requests.utils.quote(term)

    api_url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&format=json&prop=extracts&explaintext=1"
        f"&titles={title}"
    )

    r = requests.get(api_url, headers={"User-Agent": "VirusAgent/1.0"})
    if r.status_code != 200:
        return {"success": False, "message": f"No Wikipedia article found for {search_term}"}

    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))

    if "missing" in page:
        return {"success": False, "message": f"No Wikipedia article found for {search_term}"}

    page_title = page.get("title", term)
    extract = page.get("extract", "")
    page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"

    if len(extract) > LIMIT:
        extract = extract[:LIMIT] + "... [truncated]"

    return {
        "success": True,
        "title": page_title,
        "extract": extract,
        "url": page_url
    }

def query_dataframe(code: str, df_taxo: pd.DataFrame, df_host: pd.DataFrame, preview_rows: int = DEFAULT_PREVIEW_ROWS) -> dict:
    """
    Execute pandas code to query and extract data from dataframes.
    Code must assign the result to 'result' variable (pandas DataFrame).
    """
    try:
        env = {
            "df_taxo": df_taxo,
            "df_host": df_host,
            "pd": pd,
            "np": np
        }
        exec(code, {}, env)

        if "result" not in env:
            return {
                "success": False,
                "message": "Error: code must assign 'result' variable (pandas DataFrame)"
            }
        
        result = env["result"]
        if not isinstance(result, pd.DataFrame):
            return {
                "success": False,
                "message": f"Error: 'result' must be a pandas DataFrame, got {type(result)}"
            }

        preview = (
            result.to_string(index=False) if len(result) <= preview_rows
            else result.head(preview_rows).to_string(index=False) + f"\n... and {len(result)-preview_rows} more rows"
        )

        return {
            "success": True,
            "result": result,
            "shape": result.shape,
            "columns": list(result.columns),
            "preview": preview
        }

    except Exception:
        return {"success": False, "message": traceback.format_exc()}


def create_visualization(code: str, df_taxo: pd.DataFrame, df_host: pd.DataFrame) -> dict:
    """
    Execute pandas code to create a Plotly visualization.
    Code must assign the figure to 'fig' variable (Plotly figure).
    """
    try:
        env = {
            "df_taxo": df_taxo,
            "df_host": df_host,
            "pd": pd,
            "np": np,
            "px": px,
            "go": go
        }
        exec(code, {}, env)

        if "fig" not in env:
            return {
                "success": False,
                "message": "Error: code must assign 'fig' variable (Plotly figure)"
            }
        
        fig = env["fig"]
        if not isinstance(fig, (go.Figure, go.FigureWidget)):
            return {
                "success": False,
                "message": f"Error: 'fig' must be a Plotly figure, got {type(fig)}"
            }

        return {
            "success": True,
            "figure": fig
        }

    except Exception:
        return {"success": False, "message": traceback.format_exc()}

# ==================== OLLAMA TOOL DEFINITIONS ==================== #


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Get biological or scientific information from Wikipedia",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string"}
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_dataframe",
            "description": (
                "Execute pandas code to query and extract data from viral datasets. "
                "Use this tool when you need to retrieve, filter, aggregate, or analyze data.\n\n"

                "Available variables:\n"
                "- df_taxo: viral taxonomy DataFrame\n"
                "- df_host: virus-host relationships DataFrame\n"
                "- pd: pandas library\n"
                "- np: numpy library\n\n"

                "Output:\n"
                "- You MUST assign your query result to the variable 'result' (pandas DataFrame)\n\n"

                "Examples:\n"
                "# Count species per family:\n"
                "result = df_taxo.groupby('family').size().reset_index(name='count')\n\n"
                "# Filter by specific genus:\n"
                "result = df_taxo[df_taxo['genus'] == 'Orthopoxvirus']\n\n"
                "# Get top 10 families by species count:\n"
                "result = df_taxo.groupby('family').size().reset_index(name='count').sort_values('count', ascending=False).head(10)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Pandas code to execute. Must assign result to 'result' variable."}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_visualization",
            "description": (
                "Execute pandas code to create a Plotly visualization. "
                "Use this tool when you need to create charts, graphs, or plots.\n\n"

                "Available variables:\n"
                "- df_taxo: viral taxonomy DataFrame\n"
                "- df_host: virus-host relationships DataFrame\n"
                "- pd: pandas library\n"
                "- np: numpy library\n"
                "- px: plotly.express library\n"
                "- go: plotly.graph_objects library\n\n"

                "Output:\n"
                "- You MUST assign your Plotly figure to the variable 'fig'\n\n"

                "Examples:\n"
                "# Bar chart of species per family:\n"
                "data = df_taxo.groupby('family').size().reset_index(name='count')\n"
                "fig = px.bar(data, x='family', y='count', title='Species per Family')\n\n"
                "# Pie chart of genus distribution:\n"
                "data = df_taxo.groupby('genus').size().reset_index(name='count').head(10)\n"
                "fig = px.pie(data, values='count', names='genus', title='Top 10 Genera')\n\n"
                "# Horizontal bar chart:\n"
                "data = df_taxo.groupby('family').size().reset_index(name='count').sort_values('count', ascending=False).head(15)\n"
                "fig = px.bar(data, x='count', y='family', orientation='h', title='Top 15 Families')"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Pandas/Plotly code to execute. Must assign figure to 'fig' variable."}
                },
                "required": ["code"]
            }
        }
    }
]

# ====================      AGENT LOOP    ==================== #


def ollama_agent_loop(
    model: str,
    user_query: str,
    df_taxo: pd.DataFrame,
    df_host,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    seed: int = DEFAULT_SEED,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    preview_rows: int = DEFAULT_PREVIEW_ROWS,
    wikipedia_limit: int = DEFAULT_WIKIPEDIA_LIMIT,
):

    df_taxo_columns_str = ", ".join(df_taxo.columns)
    df_host_columns_str = ", ".join(df_host.columns)

    messages = [
        {
            "role": "system",
            "content": f"""You are a scientific bioinformatics assistant specialized in viruses.

You work with:

- A pandas DataFrame called `df_taxo` containing viral taxonomy data.
  Available columns:
  {df_taxo_columns_str}

- A pandas DataFrame called `df_host` containing virus-host relationships from VirusHostDB.
  Available columns:
  {df_host_columns_str}

- Access to Wikipedia ONLY via tools.

STRICT RULES:
- Do NOT show dataset structure or column names in the final response.
- NEVER invent column names.
- Use ONLY the columns explicitly listed above.
- Column names are case-sensitive.
- Do NOT invent facts, species, families, counts, or biological claims.
- If information is not present in the dataset or retrieved from a tool, explicitly say:
  "This information is not available in the current dataset or sources."
- Every biological or taxonomic statement must be grounded in:
  - a dataset query or
  - an external tool response (e.g. Wikipedia).
- Use tools when required.
- When using a dataset query, return only the minimal set of columns
  strictly necessary to answer the question.
- Report information EXACTLY as returned by tools or datasets, without interpretation.

TOOL USAGE:
- Use query_dataframe for data extraction, filtering, aggregation, and analysis
- Use create_visualization for creating charts, graphs, and plots
- Use wikipedia_search for external biological/scientific information
- For queries that need both data AND visualization, call query_dataframe first, then create_visualization

STYLE:
- Scientific, concise, neutral.
- No speculation.
- No storytelling.
- Start directly with the answer.

CRITICAL:
- Answer ONLY the specific question asked by the user.
- Do NOT assume intent or answer related questions.
- Do NOT retrieve or discuss drugs, cancer, treatments, or unrelated medical topics.
"""
        },
        {"role": "user", "content": user_query}
    ]

    used_sources = set()
    used_wikipedia_urls = []
    executed_codes = []
    tool_call_count = 0
    print("="*100)
    print("USER QUERY:", user_query)
    generated_figures = []

    while True:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "tools": TOOLS_SPEC,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty,
                    "seed": seed
                }
            },
            timeout=120
        )
        r.raise_for_status()
        response = r.json()
        msg = response["message"]

        print("\nRESPONSE:")
        if "thinking" in msg:
            print("thinking:", msg["thinking"])

        if "tool_calls" in msg:
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": msg["tool_calls"]
            })

            for call in msg["tool_calls"]:
                tool_call_count += 1
                print(call)
                name = call["function"]["name"]
                args = call["function"]["arguments"]

                if name == "wikipedia_search":
                    output = wikipedia_search(**args, wikipedia_limit=wikipedia_limit)
                    if output["success"]:
                        used_sources.add("Wikipedia")
                        used_wikipedia_urls.append(output["url"])
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": name,
                            "content": (
                                f"**{output['title']}**\n\n"
                                f"{output['extract']}\n\n"
                                f"🔗 **Source Wikipedia** : {output['url']}"
                            )
                        })
                        print("WIKI:", output)
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": name,
                            "content": output["message"]
                        })

                elif name == "query_dataframe":
                    output = query_dataframe(args["code"], df_taxo, df_host, preview_rows=preview_rows)

                    if output["success"]:
                        executed_codes.append(args["code"])
                        used_sources.add("Dataset query")

                        tool_msg = (
                            "Query executed successfully.\n"
                            f"- Shape: {output['shape']}\n"
                            f"- Columns: {', '.join(output['columns'])}\n"
                            f"- Preview:\n{output['preview']}"
                        )
                    else:
                        tool_msg = f"Error:\n{output['message']}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": tool_msg
                    })

                elif name == "create_visualization":
                    output = create_visualization(args["code"], df_taxo, df_host)

                    if output["success"]:
                        executed_codes.append(args["code"])
                        used_sources.add("Dataset visualization")
                        generated_figures.append(output["figure"])
                        tool_msg = "Visualization created successfully."
                    else:
                        tool_msg = f"Error:\n{output['message']}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": tool_msg
                    })

                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": f"Unknown tool: {name}"
                    })
            
            if tool_call_count >= max_tool_calls:
                messages.append({
                    "role": "system",
                    "content": f"You have reached the tool call limit ({max_tool_calls}). Now synthesize a final answer to the user's question: '{user_query}'"
                })

            continue

        else:
            messages.append(msg)
            print("Content:", msg["content"])
            return msg["content"], messages, generated_figures, used_sources, used_wikipedia_urls, executed_codes

# ==================== DATA LOADING ==================== #

@st.cache_data(show_spinner=False)
def load_dataframe(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_host_dataframe(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t", dtype=str)

# ==================== MODAL RENDERING ==================== #


@st.dialog("Result 🦠", width="large")
def show_response_modal(modal_idx):

    if modal_idx is None or modal_idx >= len(st.session_state.modal_data):
        st.error("No modal data to display")
        return

    modal_data = st.session_state.modal_data[modal_idx]

    # Header
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        ">
            <h3 style="margin: 0; color: white;">{modal_data["question"]}</h3>
        </div>
    """, unsafe_allow_html=True)

    # ================= LOADING STATE =================
    if modal_data["answer"] is None:

        with st.spinner("Thinking..."):

            answer, messages_history, new_figures, used_sources, used_wikipedia_urls, executed_codes = (
                ollama_agent_loop(
                    model=st.session_state.selected_model,
                    user_query=modal_data["question"],
                    df_taxo=st.session_state.df_taxo,
                    df_host=st.session_state.df_host,
                    temperature=st.session_state.agent_params["temperature"],
                    top_p=st.session_state.agent_params["top_p"],
                    repeat_penalty=st.session_state.agent_params["repeat_penalty"],
                    seed=st.session_state.agent_params["seed"],
                    max_tool_calls=st.session_state.agent_params["max_tool_calls"],
                    preview_rows=st.session_state.agent_params["preview_rows"],
                    wikipedia_limit=st.session_state.agent_params["wikipedia_limit"],
                )
            )

            # Update modal data
            st.session_state.modal_data[modal_idx] = {
                "question": modal_data["question"],
                "answer": answer,
                "figures": new_figures,
                "used_sources": used_sources,
                "wikipedia_urls": used_wikipedia_urls,
                "executed_codes": executed_codes
            }

        st.rerun()

    # ================= DISPLAY STATE =================
    else:
        st.markdown("### Answer:")
        st.markdown(modal_data["answer"])

        if modal_data.get("figures"):
            for idx, fig in enumerate(modal_data["figures"]):
                st.plotly_chart(
                    fig,
                    width='content',
                    key=f"modal_fig_{modal_idx}_{idx}"
                )

        if modal_data.get("used_sources"):
            st.markdown("---")
            st.markdown("### Used Sources:")

            if modal_data.get("wikipedia_urls"):
                wiki_links = ", ".join(
                    f"[{url.split('/')[-1].replace('_', ' ')}]({url})"
                    for url in modal_data["wikipedia_urls"]
                )
                st.markdown(f"**📘 Wikipedia**: {wiki_links}")

            if modal_data.get("executed_codes"):
                st.markdown("**📊 Dataset Query & Visualization**")
                with st.expander("See code"):
                    full_code = "\n\n---\n\n".join(
                        f"# Code {i}\n{code}"
                        for i, code in enumerate(
                            modal_data["executed_codes"], 1
                        )
                    )
                    st.code(full_code, language="python")

        if st.button("Close"):
            st.session_state.active_modal_idx = None
            st.rerun()

# ==================== STREAMLIT APP ==================== #


def main():
    st.set_page_config(page_icon="🦠", layout="wide")
    st.title("Welcome PRABI :) ")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            max-width: 1400px;
        }
        .stChatMessage {
            border-radius: 14px;
            padding: 0.75rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .stChatMessage:hover {
            background-color: rgba(100, 126, 234, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.caption("""
        This agent helps you explore viral taxonomy using structured datasets and public biological sources. You can ask about virus families, genera, species counts, and relationships — and even generate simple visualizations.

        Example questions:   
        
        - "Summarize Orthopoxvirus (family, genus, species count)"
        - "List virus families with more than 100 recorded species"
        - "How many genera are in Retroviridae?"
        - "Show a pie chart of genus distribution in Poxviridae"
        - "Compare species counts between Orthomyxoviridae and Coronaviridae"
        - "Give me hosts of Orthopoxvirus Abatino"
        
    """)


    with st.sidebar:
        st.header("Virus Dataset AI Agent 🦠 ")
        st.caption(
            """
            Explore viral taxonomy with AI-assisted analysis. 
            Here you have an access to a curated databases about virus : Viral taxonomie from sra and viral host by genome.jp. Soon more datatabe will be added and this agent will be amazing...)
            """
        )
        
        # ==================== AGENT PARAMETERS ==================== #
        st.markdown("---")
        
        st.subheader("Agent Parameters :")

        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                st.error("Ollama not running")
                st.stop()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Ollama. Make sure Ollama is running.")
            st.stop()

        df_taxo = load_dataframe(TAXO_DB_PATH)
        df_host = load_host_dataframe(HOST_DB_PATH)

        if df_taxo is None or df_host is None:
            st.error("Required dataset not found")
            st.stop()

        models_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags").json()
        model_names = [m["name"] for m in models_response.get("models", [])]


        
        if not model_names:
            st.error("No models available in Ollama")
            st.stop()

        default_index = next((i for i, model in enumerate(
            model_names) if model.startswith("gpt-oss")), 0)
        model = st.selectbox(
            "LLM Model",
            options=model_names,
            index=default_index,
            help="Model for agentic reasoning (Be sure to select a model able to use tools)"
        )

        with st.expander("🌡️ Sampling", expanded=False):
            temperature = st.slider(
                "Temperature",
                min_value=0.0, max_value=2.0,
                value=DEFAULT_TEMPERATURE, step=0.05,
                help="Controls randomness. 0 = deterministic, higher = more creative."
            )
            top_p = st.slider(
                "Top-p (nucleus sampling)",
                min_value=0.0, max_value=1.0,
                value=DEFAULT_TOP_P, step=0.05,
                help="Cumulative probability cutoff for token selection. 1.0 = disabled."
            )
            repeat_penalty = st.slider(
                "Repeat Penalty",
                min_value=0.5, max_value=2.0,
                value=DEFAULT_REPEAT_PENALTY, step=0.05,
                help="Penalizes repeated tokens. 1.0 = no penalty, higher = less repetition."
            )
            seed = st.number_input(
                "Seed",
                min_value=-1, max_value=99999,
                value=DEFAULT_SEED, step=1,
                help="Random seed for reproducibility. -1 = random."
            )

        with st.expander("🔧 Agent Behaviour", expanded=False):
            max_tool_calls = st.slider(
                "Max Tool Calls",
                min_value=1, max_value=20,
                value=DEFAULT_MAX_TOOL_CALLS, step=1,
                help="Maximum number of tool calls before forcing the model to produce a final answer."
            )
            preview_rows = st.slider(
                "DataFrame Preview Rows",
                min_value=5, max_value=200,
                value=DEFAULT_PREVIEW_ROWS, step=5,
                help="Number of rows shown to the LLM from query results. Higher = more context, more tokens."
            )
            wikipedia_limit = st.slider(
                "Wikipedia Extract Limit (chars)",
                min_value=500, max_value=20000,
                value=DEFAULT_WIKIPEDIA_LIMIT, step=500,
                help="Max number of characters fetched from Wikipedia. Higher = more context, more tokens."
            )

        # Summary badge
        st.markdown("---")
        st.markdown(
            f"**Active config** — temp `{temperature}` · top_p `{top_p}` · "
            f"penalty `{repeat_penalty}` · seed `{seed}` · "
            f"max calls `{max_tool_calls}` · preview `{preview_rows}` rows"
        )
        
        st.caption("🔗 GitHub: https://github.com/Romumrn/chat-virus-AI")

    # ==================== SESSION STATE ==================== #
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "modal_data" not in st.session_state:
        st.session_state.modal_data = []
    
    if "active_modal_idx" not in st.session_state:
        st.session_state.active_modal_idx = None
    
    st.session_state.selected_model = model

    # Store agent params in session state so modal can access them
    st.session_state.agent_params = {
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "seed": seed,
        "max_tool_calls": max_tool_calls,
        "preview_rows": preview_rows,
        "wikipedia_limit": wikipedia_limit,
    }

    if "df_taxo" not in st.session_state:
        st.session_state.df_taxo = df_taxo
    
    if "df_host" not in st.session_state:
        st.session_state.df_host = df_host

    # ==================== CHAT DISPLAY ==================== #
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:  # assistant
            with st.chat_message("assistant"):
                modal_idx = msg.get("modal_idx")
                if modal_idx is not None:
                    question = st.session_state.modal_data[modal_idx]["question"]
                    if st.button(
                        f"💬 {question}",
                        key=f"reopen_modal_{idx}",
                        use_container_width=True,
                        help="Click to open"
                    ):
                        st.session_state.active_modal_idx = modal_idx
                        st.rerun()
                else:
                    st.markdown(msg["content"])
    
    if st.session_state.active_modal_idx is not None:
        show_response_modal(st.session_state.active_modal_idx)

    query = st.chat_input("Ask about viruses...")

    if query:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        modal_idx = len(st.session_state.modal_data)

        st.session_state.modal_data.append({
            "question": query,
            "answer": None,
            "figures": None,
            "used_sources": None,
            "wikipedia_urls": None,
            "executed_codes": None
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": "",
            "modal_idx": modal_idx
        })

        st.session_state.active_modal_idx = modal_idx
        st.rerun()

    st.markdown("---")
    st.caption(
        "⚠️ **AI is not magic** —"
        "Results may contain errors and should be verified for scientific or medical use."
    )


if __name__ == "__main__":
    main()