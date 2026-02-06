import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import io
import re
import plotly.express as px
import plotly.graph_objects as go
import traceback
import warnings

warnings.filterwarnings("ignore")

# ==================== CONSTANTS ==================== #
DEFAULT_CSV_PATH = "viral_taxo.csv"
OLLAMA_BASE_URL = "http://localhost:11434"
WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
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

    return {
        "success": True,
        "title": page_title,
        "extract": extract,
        "url": page_url
    }

def query_dataframe(code: str, df: pd.DataFrame) -> dict:
    try:
        env = {"df": df, "pd": pd, "np": np}
        exec(code, {}, env)

        if "result" not in env:
            return {"success": False, "message": "Error: code must assign output to variable 'result'"}

        result = env["result"]

        if not isinstance(result, pd.DataFrame):
            return {"success": False, "message": "Error: query_dataframe must return a pandas DataFrame"}
        else:
            return {
                "success": True,
                "result": result,
                "code": code,
                "shape": result.shape,
                "columns": list(result.columns),
                "preview": result.head(5).to_string()
            }

    except Exception:
        return {"success": False, "message": traceback.format_exc()}

def plot_dataframe(code: str, df: pd.DataFrame, last_result=None):
    try:
        env = {
            "df": df,
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
            "last_result": last_result
        }
        exec(code, {}, env)

        if "fig" not in env:
            return None, "Error: code must assign Plotly figure to variable 'fig'"

        fig = env["fig"]

        if not isinstance(fig, (go.Figure, go.FigureWidget)):
            return None, f"Error: 'fig' must be a Plotly figure, got {type(fig)}"

        return fig, "Plot generated successfully"

    except Exception as e:
        return None, f"Error in plot generation: {traceback.format_exc()}"

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
                    "search_term": {
                        "type": "string",
                        "description": "Exact term to search on Wikipedia"
                    }
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_dataframe",
            "description": "Execute pandas code on df and RETURN a pandas DataFrame. The variable 'result' MUST be a pandas DataFrame. This DataFrame will automatically be stored as 'last_result' for subsequent tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code using df and assigning output to variable 'result'"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plot_dataframe",
            "description": "Create a Plotly chart ONLY from 'last_result'. DO NOT compute or modify data. DO NOT reference 'df'. Use 'last_result' exactly as provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code using df, px, go, and last_result (DataFrame from previous query). Assign figure to variable 'fig'."
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# ==================== REAL AGENT LOOP ==================== #

def ollama_agent_loop(model: str, user_query: str, df: pd.DataFrame):
    messages = [
        {
            "role": "system",
            "content": """You are a scientific bioinformatics assistant specialized in virus.

You work with:
• A pandas DataFrame called `df` containing viral taxonomy data collumns : ORGANISM_NAME,TAX_ID,PARENT_TAX_ID,RANK,SPECIES_NAME,GENUS_NAME,FAMILY_NAME,ORDER_NAME,CLASS_NAME,PHYLUM_NAME,KINGDOM_NAME,SUPERKINGDOM_NAME,HAS_GENOME_ASSEMBLY,NUM_GENOME_ASSEMBLIES,MAX_GENOME_SIZE,HAS_STAT_KMER,NUM_STAT_KMER,HAS_STAT,NUM_POSITIVE_STAT_SAMPLE,SUM_STAT_TOTAL_COUNT,HAS_STAT_ECO,NUM_POSITIVE_STAT_SAMPLE_ECO,SUM_STAT_TOTAL_COUNT_ECO,HAS_GBIF,NUM_GBIF_OBSERVATIONS,NUM_SRA_SAMPLES,NUM_GSA_SAMPLES,NUM_COLLECTION_DATE,FIRST_COLLECTION_DATE,LAST_COLLECTION_DATE,NUM_LATLON
• Trusted public biological sources (e.g. Wikipedia) accessed ONLY via tools

STRICT RULES:
- Do NOT invent facts, species, families, counts, or biological claims.
- If information is not present in the dataset or retrieved from a tool, explicitly say:
   "This information is not available in the current dataset or sources."
- Do not shown Dataset structure 
- Every biological or taxonomic statement must be grounded in:
   - the dataset (df / last_result), or
   - an external tool response (e.g. Wikipedia) 
- Use tools if needed

DATASET AWARENESS:
- At the beginning of the FIRST response in a conversation, briefly describe
  the dataset structure using df.columns.
- Column descriptions must be inferred conservatively from column names only.
- Do NOT guess biological meaning beyond column names.

STYLE:
- Scientific, concise, neutral
- No speculation
- No storytelling
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
    last_result = None

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
                        print( "WIKIIIII", output)
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": name,
                            "content": output["message"]
                        })

                elif name == "query_dataframe":
                    output = query_dataframe(args["code"], df)
                    used_sources.add("Dataset query")

                    if output["success"]:
                        last_result = output["result"]
                        executed_codes.append(output["code"])
                        tool_content = (
                            "DataFrame stored as last_result.\n"
                            f"- shape: {output['shape']}\n"
                            f"- columns: {output['columns']}\n"
                            f"- preview:\n{output['preview']}"
                        )
                    else:
                        last_result = None
                        tool_content = f"Error during dataframe query:\n{output['message']}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": tool_content
                    })

                elif name == "plot_dataframe":
                    fig, output = plot_dataframe(args["code"], df, last_result)
                    if fig is not None:
                        generated_figures.append(fig)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": output
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
            
        • "Summarize Orthopoxvirus (family, genus, species count)"
        • "List virus families with more than 100 recorded species"
        • "How many genera are in Retroviridae?"
        • "Show a pie chart of genus distribution in Poxviridae"
        • "Compare species counts between Orthomyxoviridae and Coronaviridae"
        
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

        df = load_dataframe(DEFAULT_CSV_PATH)
        if df is None:
            st.error("Dataset not found")
            st.stop()

        models_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags").json()
        model_names = [m["name"] for m in models_response.get("models", [])]

        if not model_names:
            st.error("No models available in Ollama")
            st.stop()

        default_index = next((i for i, model in enumerate(model_names) if model.startswith("gpt-oss")), 0)
        model = st.selectbox(
            "🤖 LLM Model",
            options=model_names,
            index=default_index,
            help="Model for agentic reasoning"
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "figures" not in st.session_state:
        st.session_state.figures = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    for fig in st.session_state.figures:
        st.plotly_chart(fig, width='stretch')

    query = st.chat_input("Ask about viruses...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, messages_history, new_figures, used_sources, used_wikipedia_urls, executed_codes = ollama_agent_loop(model, query, df)

                st.markdown(answer)

                if used_sources:
                    st.markdown("##### 🔎 Sources used")

                    if used_wikipedia_urls:
                        wiki_str = ", ".join(
                            f"[{url.split('/')[-1].replace('_', ' ')}]({url})"
                            for url in used_wikipedia_urls  )
                        st.markdown(f"**📘 Wikipedia**: {wiki_str}")

                    # Section Dataset Query
                    if executed_codes:
                        st.markdown("**📊 Dataset Query**")
                        with st.expander("View all executed codes"):
                            full_code = "\n\n---\n\n".join(
                                f"# Code {i}\n{code}"
                                for i, code in enumerate(executed_codes, 1)
                            )
                            st.code(full_code, language="python")

                    for fig in new_figures:
                        st.plotly_chart(fig, width='stretch')
                        st.session_state.figures.append(fig)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    if st.button("🗑 Clear history"):
        st.session_state.messages = []
        st.session_state.figures = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "⚠️ **AI is not magic** — This assistant generates answers using statistical models, "
        "public sources (e.g. Wikipedia), and your dataset. "
        "Results may contain errors and should be verified for scientific or medical use."
    )

if __name__ == "__main__":
    main()
