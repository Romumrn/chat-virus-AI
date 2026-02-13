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

# ==================== REAL TOOL IMPLEMENTATIONS ==================== #

def wikipedia_search(search_term: str) -> dict:
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

    if len(extract) > 10000:  # Limit to ~10000 chars
        extract = extract[:10000] + "... [truncated]"

    return {
        "success": True,
        "title": page_title,
        "extract": extract,
        "url": page_url
    }


def query_and_plot(code: str, df_taxo: pd.DataFrame, df_host: pd.DataFrame) -> dict:
    """
    Execute pandas code that can query dataframes and/or create visualizations.
    Code can assign 'result' (DataFrame) and/or 'fig' (Plotly figure).
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

        response = {"success": True}
        
        # Check for result DataFrame
        if "result" in env:
            result = env["result"]
            if not isinstance(result, pd.DataFrame):
                return {
                    "success": False, 
                    "message": "Error: 'result' must be a pandas DataFrame"
                }
            
            preview = (
                result.to_string(index=False) if len(result) <= 50
                else result.head(50).to_string(index=False) + f"\n... and {len(result)-50} more rows"
            )
            
            response["has_result"] = True
            response["result"] = result
            response["shape"] = result.shape
            response["columns"] = list(result.columns)
            response["preview"] = preview
        else:
            response["has_result"] = False
        
        # Check for figure
        if "fig" in env:
            fig = env["fig"]
            if not isinstance(fig, (go.Figure, go.FigureWidget)):
                return {
                    "success": False,
                    "message": f"Error: 'fig' must be a Plotly figure, got {type(fig)}"
                }
            response["has_figure"] = True
            response["figure"] = fig
        else:
            response["has_figure"] = False
        
        # Must have at least one output
        if not response["has_result"] and not response["has_figure"]:
            return {
                "success": False,
                "message": "Error: code must assign 'result' (DataFrame) and/or 'fig' (Plotly figure)"
            }
        
        return response

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
            "name": "query_and_plot",
            "description": (
                "Execute pandas code on viral taxonomy (df_taxo) and host relationship (df_host) dataframes. "
                "You can query data and/or create visualizations in the same code block. "

                "Available variables:\n"
                "- df_taxo: viral taxonomy DataFrame\n"
                "- df_host: virus-host relationships DataFrame\n"
                "- pd, np: pandas and numpy\n"
                "- px, go: plotly.express and plotly.graph_objects\n"

                "Output variables:\n"
                "- Assign query results to 'result' (pandas DataFrame) - REQUIRED for data queries\n"
                "- Assign visualization to 'fig' (Plotly figure) - REQUIRED for plots\n"
                "- You can assign both in the same code block\n"

                "Examples:\n"
                "# Query only:\n"
                "result = df_taxo.groupby('family').size().reset_index(name='count')\n"
                "\n"
                "# Plot only:\n"
                "fig = px.bar(df_taxo.groupby('family').size().reset_index(name='count'), x='family', y='count')\n"
                "\n"
                "# Query + Plot:\n"
                "result = df_taxo.groupby('family').size().reset_index(name='count')\n"
                "fig = px.bar(result, x='family', y='count', title='Species per Family')"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"}
                },
                "required": ["code"]
            }
        }
    }
]

# ====================      AGENT LOOP    ==================== #

def ollama_agent_loop(model: str, user_query: str, df_taxo: pd.DataFrame, df_host):

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
- Use query_and_plot for ANY data query or visualization
- You can create both a query result AND a plot in the same code block
- For plots, always assign the figure to variable 'fig'
- For queries, always assign the result to variable 'result'

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
                "stream": False
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
                print(call)
                name = call["function"]["name"]
                args = call["function"]["arguments"]

                if name == "wikipedia_search":
                    output = wikipedia_search(**args)
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

                elif name == "query_and_plot":
                    output = query_and_plot(args["code"], df_taxo, df_host)

                    if output["success"]:
                        executed_codes.append(args["code"])
                        used_sources.add("Dataset query")

                        # Build response message
                        tool_msg_parts = []
                        
                        if output.get("has_result"):
                            tool_msg_parts.append(
                                "Query executed successfully.\n"
                                f"- Shape: {output['shape']}\n"
                                f"- Columns: {', '.join(output['columns'])}\n"
                                f"- Preview:\n{output['preview']}"
                            )
                        
                        if output.get("has_figure"):
                            generated_figures.append(output["figure"])
                            tool_msg_parts.append("Visualization created successfully.")
                        
                        tool_msg = "\n\n".join(tool_msg_parts)
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

# ==================== STREAMLIT APP ==================== #

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🦠", layout="wide")
    st.title(PAGE_TITLE)
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
        st.markdown("### 🧬 Virus AI Agent")
        st.caption("Explore viral taxonomy with AI-assisted analysis")
        st.header("Configuration")

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
            "🤖 LLM Model",
            options=model_names,
            index=default_index,
            help="Model for agentic reasoning (Be sure to select a model able to use tools)"
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "figures" not in st.session_state:
        st.session_state.figures = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # for fig in st.session_state.figures:
    #     st.plotly_chart(fig, width='stretch')

    query = st.chat_input("Ask about viruses...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, messages_history, new_figures, used_sources, used_wikipedia_urls, executed_codes = (
                    ollama_agent_loop(model, query, df_taxo, df_host)
                )
                st.markdown(answer)

                if used_sources:
                    st.markdown("##### 🔎 Sources used")

                    if used_wikipedia_urls:
                        wiki_str = ", ".join(
                            f"[{url.split('/')[-1].replace('_', ' ')}]({url})"
                            for url in used_wikipedia_urls)
                        st.markdown(f"**📘 Wikipedia**: {wiki_str}")

                    # Section Dataset Query
                    if executed_codes:
                        st.markdown("**📊 Dataset Query & Visualization**")
                        with st.expander("View all executed codes"):
                            full_code = "\n\n---\n\n".join(
                                f"# Code {i}\n{code}"
                                for i, code in enumerate(executed_codes, 1)
                            )
                            st.code(full_code, language="python")

            # Display figures outside the sources block
            for fig in new_figures:
                st.plotly_chart(fig, width='stretch')
                st.session_state.figures.append(fig)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    st.markdown("---")
    st.caption(
        "⚠️ **AI is not magic** —"
        "Results may contain errors and should be verified for scientific or medical use."
    )


if __name__ == "__main__":
    main()